"""
Mesh to flexible dual grid on the CPU, in NumPy.

A port of o_voxel's `mesh_to_flexible_dual_grid_cpu` (C++ and Eigen), which
is what the shape encoder of the texturing pipeline takes as input. For each
voxel the surface passes through it returns one dual vertex, placed by a
quadric error function, and three flags saying which of the voxel's edges
the surface crosses.

The steps and their conventions follow the C++ source:
  1. Intersect every triangle with the grid lines along x, y and z. Each
     hit adds the triangle's plane quadric to the four voxels around that
     grid line, and flags the edge on the first of them.
  2. Add the plane quadric of every triangle to each voxel it overlaps
     (`face_weight` only switches this on, as upstream).
  3. Add a line quadric for every open edge to the voxels it runs through.
  4. Minimise each voxel's quadric inside the voxel, with a pull towards
     the mean of its intersection points.

Voxels come out sorted by coordinate, not in upstream's insertion order.
"""

import numpy as np

# Rows of the work arrays built at once; bounds memory on large meshes.
CHUNK = 4_000_000


def _unit_normals(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0)


def _plane_quadrics(tri, normals):
    plane = np.concatenate([normals, -np.einsum("ij,ij->i", normals, tri[:, 0])[:, None]], axis=1)
    return plane[:, :, None] * plane[:, None, :]


def _lerp(a, b, t, val_a, val_b):
    """val_a at a, val_b at b; val_a where a == b, as upstream."""
    span = b - a
    alpha = np.divide(t - a, span, out=np.zeros_like(span), where=span != 0)
    return (1 - alpha)[:, None] * val_a + alpha[:, None] * val_b


def _repeat_ranges(start, stop):
    """Owner index and value for every integer in [start[i], stop[i])."""
    count = np.maximum(stop - start, 0)
    owner = np.repeat(np.arange(len(start)), count)
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    return owner, start[owner] + offset


def _chunks(cost, limit=CHUNK):
    """Slices of consecutive items whose summed cost stays near `limit`."""
    total = np.cumsum(cost)
    start = 0
    while start < len(cost):
        done = total[start - 1] if start else 0
        stop = max(int(np.searchsorted(total, done + limit, side="right")), start + 1)
        yield slice(start, stop)
        start = stop


def _grid_line_hits(tri, h, grid, ax2):
    """Hits of grid lines along `ax2` with the triangles, by scan line.

    Returns the triangle index, the two line indices (the line sits at
    index + 1 on each of the other axes), the voxel index along `ax2` and
    the hit points.
    """
    ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
    t = tri[:, :, [ax0, ax1, ax2]]
    t = np.take_along_axis(t, np.argsort(t[:, :, 1], axis=1, kind="stable")[:, :, None], axis=1)

    def clamp_index(value, axis):
        return np.clip((value / h[axis]).astype(np.int64), 0, grid[axis] - 1)

    start, mid, end = (clamp_index(t[:, i, 1], ax1) for i in range(3))
    out = []
    for sl in _chunks(np.maximum(end - start, 0) + 1, CHUNK // 8):
        rows_tri, rows_y = _repeat_ranges(start[sl], end[sl])
        if not len(rows_tri):
            continue
        rows_tri += sl.start
        y = (rows_y + 1) * h[ax1]
        # Above `mid` the scan runs between the two edges leaving the top
        # vertex, below it between the two leaving the bottom one.
        upper = rows_y < mid[rows_tri]
        apex = np.where(upper[:, None], t[rows_tri, 0], t[rows_tri, 2])
        far = np.where(upper[:, None], t[rows_tri, 2], t[rows_tri, 0])
        middle = t[rows_tri, 1]
        t3 = _lerp(apex[:, 1], middle[:, 1], y, apex[:, [0, 2]], middle[:, [0, 2]])
        t4 = _lerp(apex[:, 1], far[:, 1], y, apex[:, [0, 2]], far[:, [0, 2]])
        swap = t3[:, 0] > t4[:, 0]
        t3[swap], t4[swap] = t4[swap], t3[swap]
        line_start, line_end = clamp_index(t3[:, 0], ax0), clamp_index(t4[:, 0], ax0)
        for rs in _chunks(np.maximum(line_end - line_start, 0) + 1):
            hit_row, hit_x = _repeat_ranges(line_start[rs], line_end[rs])
            if not len(hit_row):
                continue
            hit_row += rs.start
            x = (hit_x + 1) * h[ax0]
            z = _lerp(t3[hit_row, 0], t4[hit_row, 0], x, t3[hit_row, 1:], t4[hit_row, 1:])[:, 0]
            z_idx = (z / h[ax2]).astype(np.int64)
            ok = (z_idx >= 0) & (z_idx < grid[ax2])
            point = np.empty((int(ok.sum()), 3))
            point[:, ax0], point[:, ax1], point[:, ax2] = x[ok], y[hit_row[ok]], z[ok]
            out.append((rows_tri[hit_row[ok]], hit_x[ok], rows_y[hit_row[ok]], z_idx[ok], point))
    return out


def _overlapped_voxels(tri, normals, h, grid):
    """(triangle, voxel key) for every voxel a triangle's box test passes.

    Upstream tests every voxel of the triangle's bounding box. Here the
    candidates are the columns of the box along the normal's dominant axis,
    and in each column only the voxels the plane can reach, which is the
    same set after the same test but far fewer candidates.
    """
    tri_ids = np.flatnonzero(np.abs(normals).max(axis=1) > 0)
    dominant = np.argmax(np.abs(normals[tri_ids]), axis=1)
    lo = np.maximum((tri[tri_ids].min(axis=1) / h).astype(np.int64), 0)
    hi = np.minimum((tri[tri_ids].max(axis=1) / h + 1).astype(np.int64), grid)
    for d in range(3):
        a, b = (d + 1) % 3, (d + 2) % 3
        sel = np.flatnonzero(dominant == d)
        if not len(sel):
            continue
        width_a = np.maximum(hi[sel, a] - lo[sel, a], 0)
        width_b = np.maximum(hi[sel, b] - lo[sel, b], 0)
        for sl in _chunks(width_a * width_b * 4 + 1):
            ids = sel[sl]
            t = tri_ids[ids]
            owner, cell = _repeat_ranges(np.zeros(len(ids), dtype=np.int64), width_a[sl] * width_b[sl])
            ca = lo[ids[owner], a] + cell // width_b[sl][owner]
            cb = lo[ids[owner], b] + cell % width_b[sl][owner]
            n, v0 = normals[t[owner]], tri[t[owner], 0]
            # Height of the plane over the column's centre, and how far it
            # tilts across the column.
            centre = v0[:, d] - (n[:, a] * ((ca + 0.5) * h[a] - v0[:, a]) + n[:, b] * ((cb + 0.5) * h[b] - v0[:, b])) / n[:, d]
            tilt = (np.abs(n[:, a]) * h[a] + np.abs(n[:, b]) * h[b]) / (2 * np.abs(n[:, d]))
            first = np.maximum(np.floor((centre - tilt) / h[d] - 1e-6).astype(np.int64), lo[ids[owner], d])
            last = np.minimum(np.floor((centre + tilt) / h[d] + 1e-6).astype(np.int64) + 1, hi[ids[owner], d])
            col, cd = _repeat_ranges(first, last)
            voxel = np.empty((len(col), 3), dtype=np.int64)
            voxel[:, a], voxel[:, b], voxel[:, d] = ca[col], cb[col], cd
            face = t[owner[col]]
            keep = _triangle_overlaps_voxel(tri[face], normals[face], voxel * h, h)
            yield face[keep], voxel[keep]


def _triangle_overlaps_voxel(tri, n, p, h):
    """Schwarz and Seidel's triangle-box test, for boxes [p, p + h]."""
    c = np.where(n > 0, h, 0.0)
    d1 = np.einsum("ij,ij->i", n, c - tri[:, 0])
    d2 = np.einsum("ij,ij->i", n, h - c - tri[:, 0])
    n_dot_p = np.einsum("ij,ij->i", n, p)
    keep = (n_dot_p + d1) * (n_dot_p + d2) <= 0
    edges = [tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]]
    for a, b, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):   # xy, yz and zx projections
        sign = np.where(n[:, k] < 0, -1.0, 1.0)
        for i, e in enumerate(edges):
            na, nb = -sign * e[:, b], sign * e[:, a]
            dist = -(na * tri[:, i, a] + nb * tri[:, i, b]) + np.maximum(na, 0) * h[a] + np.maximum(nb, 0) * h[b]
            keep &= na * p[:, a] + nb * p[:, b] + dist >= 0
    return keep


def _boundary_segments(vertices, faces):
    """End points of the edges that belong to exactly one face."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys, count = np.unique(edges.min(axis=1) * len(vertices) + edges.max(axis=1), return_counts=True)
    keys = keys[count == 1]
    return vertices[keys // len(vertices)], vertices[keys % len(vertices)]


def _segment_voxels(v0, v1, h):
    """(segment, voxel) for every voxel a segment passes through."""
    delta = v1 - v0
    length = np.linalg.norm(delta, axis=1)
    keep = np.flatnonzero(length >= 1e-6)
    v0, delta, length = v0[keep], delta[keep], length[keep]
    direction = delta / length[:, None]
    seg, t = [np.arange(len(v0))], [np.zeros(len(v0))]
    for axis in range(3):
        lo = np.minimum(v0[:, axis], v0[:, axis] + delta[:, axis]) / h[axis]
        hi = np.maximum(v0[:, axis], v0[:, axis] + delta[:, axis]) / h[axis]
        owner, plane = _repeat_ranges(np.floor(lo).astype(np.int64) + 1, np.floor(hi).astype(np.int64) + 1)
        crossing = (plane * h[axis] - v0[owner, axis]) / direction[owner, axis]
        ok = (crossing > 0) & (crossing <= length[owner])
        seg.append(owner[ok])
        t.append(crossing[ok])
    seg, t = np.concatenate(seg), np.concatenate(t)
    order = np.lexsort((t, seg))
    seg, t = seg[order], t[order]
    # Each piece between two crossings lies in one voxel: that of its middle.
    last = np.r_[seg[1:] != seg[:-1], True]
    middle = (t + np.where(last, length[seg], np.r_[t[1:], 0.0])) / 2
    point = v0[seg] + middle[:, None] * direction[seg]
    return keep[seg], np.floor(point / h).astype(np.int64)


def _solve_in_box(Q, lo, hi):
    """Minimise [p, 1] Q [p, 1]^T over the box [lo, hi], one box per row.

    As upstream: the free minimiser if it lies in the box, otherwise the
    best of the minimisers on the faces and edges that land inside them and
    of the eight corners.
    """
    def solve(A, b):
        try:
            return np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return np.einsum("nij,nj->ni", np.linalg.pinv(A), b)

    def error(p):
        ph = np.concatenate([p, np.ones((len(p), 1))], axis=1)
        return np.einsum("ni,nij,nj->n", ph, Q, ph)

    out = solve(Q[:, :3, :3], -Q[:, :3, 3])
    outside = np.flatnonzero(~((out >= lo) & (out <= hi)).all(axis=1) | ~np.isfinite(out).all(axis=1))
    if not len(outside):
        return out
    Q, lo, hi = Q[outside], lo[outside], hi[outside]
    best = np.full(len(Q), np.inf)
    best_p = np.zeros((len(Q), 3))

    def offer(p, feasible):
        err = np.where(feasible, error(p), np.inf)
        better = err < best
        best[better], best_p[better] = err[better], p[better]

    bounds = (lo, hi)
    for fixed in range(3):                      # on a face
        a1, a2 = (fixed + 1) % 3, (fixed + 2) % 3
        A = Q[:, [a1, a2]][:, :, [a1, a2]]
        for bound in bounds:
            rhs = -(Q[:, [a1, a2], fixed] * bound[:, [fixed]] + Q[:, [a1, a2], 3])
            x = solve(A, rhs)
            p = np.empty((len(Q), 3))
            p[:, fixed], p[:, a1], p[:, a2] = bound[:, fixed], x[:, 0], x[:, 1]
            offer(p, (x[:, 0] >= lo[:, a1]) & (x[:, 0] <= hi[:, a1]) & (x[:, 1] >= lo[:, a2]) & (x[:, 1] <= hi[:, a2]))
    for free in range(3):                       # on an edge
        a1, a2 = (free + 1) % 3, (free + 2) % 3
        for b1 in bounds:
            for b2 in bounds:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = -(Q[:, free, a1] * b1[:, a1] + Q[:, free, a2] * b2[:, a2] + Q[:, free, 3]) / Q[:, free, free]
                p = np.empty((len(Q), 3))
                p[:, free], p[:, a1], p[:, a2] = x, b1[:, a1], b2[:, a2]
                offer(np.nan_to_num(p), (x >= lo[:, free]) & (x <= hi[:, free]))
    for corner in range(8):
        p = np.stack([bounds[(corner >> axis) & 1][:, axis] for axis in range(3)], axis=1)
        offer(p, np.ones(len(Q), dtype=bool))
    out[outside] = best_p
    return out


def mesh_to_flexible_dual_grid(vertices, faces, grid_size, aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
                               face_weight=1.0, boundary_weight=1.0, regularization_weight=0.1):
    """Voxelise a triangle mesh into a flexible dual grid.

    Args:
        vertices: [V, 3] positions inside `aabb`.
        faces: [F, 3] vertex indices.
        grid_size: voxels per axis, one int or three.
        aabb: [2, 3] corners of the grid.
        face_weight: above zero, triangles add their plane to every voxel
            they overlap, not only at grid-line hits.
        boundary_weight: weight of the open-edge term.
        regularization_weight: pull towards the mean intersection point.

    Returns:
        coords [N, 3] int32, dual vertices [N, 3] float32 in the units of
        `vertices` measured from `aabb[0]`, and edge flags [N, 3] bool.
    """
    aabb = np.asarray(aabb, dtype=np.float64)
    grid = np.broadcast_to(np.asarray(grid_size, dtype=np.int64), (3,)).copy()
    h = (aabb[1] - aabb[0]) / grid
    vertices = np.asarray(vertices, dtype=np.float32).astype(np.float64) - aabb[0]
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    normals = _unit_normals(tri)
    scale = np.array([grid[1] * grid[2], grid[2], 1], dtype=np.int64)

    # 1. Grid-line hits. Each one counts for the four voxels around its
    # grid line: the first holds the crossed edge, the others lie one step
    # along each of the two other axes.
    base, step0, step1, hit_axis, hit_tri, points = [], [], [], [], [], []
    for ax2 in range(3):
        ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
        for t, i0, i1, i2, point in _grid_line_hits(tri, h, grid, ax2):
            voxel = np.empty((len(t), 3), dtype=np.int64)
            voxel[:, ax0], voxel[:, ax1], voxel[:, ax2] = i0, i1, i2
            base.append(voxel @ scale)
            step0.append(np.full(len(t), scale[ax0]))
            step1.append(np.full(len(t), scale[ax1]))
            hit_axis.append(np.full(len(t), ax2))
            hit_tri.append(t)
            points.append(point)
    if not base or not sum(len(b) for b in base):
        return np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32), np.zeros((0, 3), bool)
    base, step0, step1 = np.concatenate(base), np.concatenate(step0), np.concatenate(step1)
    hit_axis, hit_tri, points = np.concatenate(hit_axis), np.concatenate(hit_tri), np.concatenate(points)
    voxel_keys, inverse = np.unique(np.concatenate([base, base + step0, base + step1, base + step0 + step1]),
                                    return_inverse=True)
    inverse = inverse.reshape(4, len(base))
    del base, step0, step1
    N = len(voxel_keys)
    count = sum(np.bincount(rows, minlength=N) for rows in inverse).astype(np.float64)
    mean = np.stack([sum(np.bincount(rows, weights=points[:, axis], minlength=N) for rows in inverse)
                     for axis in range(3)], axis=1) / count[:, None]
    del points

    def add_quadrics(Q, rows, quadrics):
        for i in range(4):
            for j in range(i, 4):
                total = np.bincount(rows, weights=quadrics[:, i, j], minlength=N)
                Q[:, i, j] += total
                if i != j:
                    Q[:, j, i] += total

    Q = np.zeros((N, 4, 4))
    for start in range(0, len(hit_tri), CHUNK):
        sl = slice(start, start + CHUNK)
        quadrics = _plane_quadrics(tri[hit_tri[sl]], normals[hit_tri[sl]])
        for rows in inverse:
            add_quadrics(Q, rows[sl], quadrics)
    intersected = np.zeros((N, 3), dtype=bool)
    intersected[inverse[0], hit_axis] = True
    del hit_tri, hit_axis, inverse

    def rows_of(voxel):
        """Rows of the voxels that exist, and which of the inputs those are."""
        inside = ((voxel >= 0) & (voxel < grid)).all(axis=1)
        key = voxel @ scale
        pos = np.searchsorted(voxel_keys, key).clip(max=N - 1)
        found = inside & (voxel_keys[pos] == key)
        return pos[found], found

    # 2. Planes of the triangles that overlap each voxel.
    if face_weight > 0:
        for face, voxel in _overlapped_voxels(tri, normals, h, grid):
            rows, found = rows_of(voxel)
            add_quadrics(Q, rows, _plane_quadrics(tri[face[found]], normals[face[found]]))

    # 3. Lines of the open edges.
    if boundary_weight > 0:
        v0, v1 = _boundary_segments(vertices, faces)
        if len(v0):
            seg, voxel = _segment_voxels(v0, v1, h)
            rows, found = rows_of(voxel)
            seg = seg[found]
            direction = v1[seg] - v0[seg]
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            A = np.eye(3)[None] - direction[:, :, None] * direction[:, None, :]
            b = -np.einsum("nij,nj->ni", A, v0[seg])
            line = np.zeros((len(seg), 4, 4))
            line[:, :3, :3], line[:, :3, 3], line[:, 3, :3] = A, b, b
            line[:, 3, 3] = -np.einsum("ni,ni->n", b, v0[seg])
            add_quadrics(Q, rows, boundary_weight * line)

    # 4. Dual vertices.
    if regularization_weight > 0:
        pull = np.zeros((N, 4, 4))
        pull[:, :3, :3] = np.eye(3)
        pull[:, :3, 3] = pull[:, 3, :3] = -mean
        pull[:, 3, 3] = np.einsum("ni,ni->n", mean, mean)
        Q += (regularization_weight * count)[:, None, None] * pull
    coords = np.stack([voxel_keys // scale[0], voxel_keys // scale[1] % grid[1], voxel_keys % grid[2]], axis=1)
    dual = _solve_in_box(Q, coords * h, (coords + 1) * h)
    return coords.astype(np.int32), dual.astype(np.float32), intersected
