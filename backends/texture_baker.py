"""
UV unwrap, texture baking and GLB export for TRELLIS.2 meshes.

Matches the behaviour of upstream's o_voxel.postprocess.to_glb without its
dependencies. Do not import that module: it imports nvdiffrast, whose licence
forbids commercial use.
  - xatlas for UV unwrapping
  - NumPy UV-space rasteriser
  - Sparse trilinear sampling, same weights as FlexGEMM's grid_sample_3d
  - cv2.inpaint for gaps between UV charts
"""

import time

import cv2
import numpy as np


def orient_faces(vertices, faces):
    """Make face winding consistent, facing outwards, as far as topology allows.

    The dual-grid extractor winds quads by axis, not by surface side, and both
    the simplifier and xatlas fail on that. Propagates winding over a BFS tree
    of the manifold-edge face graph. These meshes contain odd cycles, so a
    perfect assignment does not exist; the conflicts stay on tree-cut edges.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    # int64 throughout: the edge key below multiplies two vertex ids, which
    # overflows int32 (the dtype SciPy's component labels and the simplifier
    # return) on any mesh with more than 46,341 vertices. Colliding keys then
    # pair unrelated edges, and the vertex split downstream merges corners of
    # different vertices into triangles that span the model.
    faces = np.asarray(faces, dtype=np.int64)
    F = len(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    face_ids = np.tile(np.arange(F), 3)
    forward = edges[:, 0] < edges[:, 1]
    keys = edges.min(axis=1) * (int(faces.max()) + 1) + edges.max(axis=1)
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    run_start = np.r_[True, k[1:] != k[:-1]]
    run_id = np.cumsum(run_start) - 1
    first = np.flatnonzero(run_start & (np.bincount(run_id)[run_id] == 2))
    a, b = face_ids[order[first]], face_ids[order[first + 1]]
    # Same traversal direction across a shared edge means opposite winding.
    differ = forward[order[first]] == forward[order[first + 1]]

    # A virtual root (node F) joins one face per surface, so BFS covers them all.
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(F, F))
    _, surface = connected_components(graph, directed=False)
    _, seeds = np.unique(surface, return_index=True)
    rows = np.r_[a, np.full(len(seeds), F)]
    cols = np.r_[b, seeds]
    tree = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(F + 1, F + 1))
    _, parent = breadth_first_order(tree.tocsr(), F, directed=False)
    parent = parent.astype(np.int64)
    parent[F] = F
    parent[seeds] = F

    # Flip relative to the parent, then accumulate to the root by pointer jumping.
    lookup = np.r_[np.minimum(a, b) * (F + 1) + np.maximum(a, b)]
    idx = np.argsort(lookup)
    lookup, rel = lookup[idx], differ[idx]
    child = np.setdiff1d(np.arange(F), seeds)
    pair = np.minimum(child, parent[child]) * (F + 1) + np.maximum(child, parent[child])
    flip = np.zeros(F + 1, dtype=bool)
    flip[child] = rel[np.searchsorted(lookup, pair)]
    while (parent != F).any():
        flip = flip ^ flip[parent]
        parent = parent[parent]
    flip = flip[:F]
    out = faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]

    # Point each surface outwards by the sign of its volume.
    tri = np.asarray(vertices, dtype=np.float64)[out]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    inward = np.bincount(surface, weights=volume)[surface] < 0
    out[inward] = out[inward][:, [0, 2, 1]]
    return out


def split_non_manifold_vertices(vertices, faces, pair_sheets=True):
    """Split vertices so that edges shared by three or more faces come apart.

    The extractor and vertex clustering both produce such edges, which stall
    the simplifier and make xatlas take hours. The faces around each vertex
    are grouped into fans that connect through manifold edges, and each fan
    gets its own copy of the vertex.

    With `pair_sheets`, an edge shared by three or four faces keeps joined
    the two faces that continue through it most directly: the pair whose far
    vertices lie in the most opposite directions across the edge. So a
    T-junction cuts off only the sheet that ends there, and two walls that
    merged into one edge separate into two intact sheets. The test is purely
    geometric, so it does not depend on face winding, which is unreliable on
    these meshes. Without pairing every such edge becomes a cut, which
    fragments the mesh into many small pieces, and xatlas into many small
    charts. Edges shared by five or more faces, and anything pairing leaves
    non-manifold, are always cut.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64)  # see orient_faces on int32 overflow
    F = len(faces)
    # Edge slot c * F + i runs from corner c to corner (c + 1) % 3 of face i.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = edges.min(axis=1) * (int(faces.max()) + 1) + edges.max(axis=1)
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    run_start = np.r_[True, k[1:] != k[:-1]]
    run_first = np.flatnonzero(run_start)
    counts = np.diff(np.r_[run_first, len(k)])
    two = run_first[counts == 2]
    ea, eb = order[two], order[two + 1]
    if pair_sheets:
        v = np.asarray(vertices, dtype=np.float64)
        # For each edge slot: unit direction from the edge to the face's far
        # vertex, perpendicular to the edge.
        slot = np.arange(3 * F)
        far = faces[slot % F, (slot // F + 2) % 3]
        axis = v[edges[:, 1]] - v[edges[:, 0]]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-30
        side = v[far] - v[edges[:, 0]]
        side -= (side * axis).sum(axis=1, keepdims=True) * axis
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-30

        def agreement(slots, i, j):
            return -(side[slots[:, i]] * side[slots[:, j]]).sum(axis=1)

        three = run_first[counts == 3]
        if len(three):
            slots = np.stack([order[three + i] for i in range(3)], axis=1)
            best = np.argmax(np.stack([agreement(slots, 0, 1), agreement(slots, 0, 2), agreement(slots, 1, 2)], axis=1), axis=1)
            i, j = np.array([0, 0, 1])[best], np.array([1, 2, 2])[best]
            rows = np.arange(len(slots))
            ea, eb = np.r_[ea, slots[rows, i]], np.r_[eb, slots[rows, j]]
        four = run_first[counts == 4]
        if len(four):
            slots = np.stack([order[four + i] for i in range(4)], axis=1)
            pairings = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])
            scores = np.stack([agreement(slots, a[0], a[1]) + agreement(slots, b[0], b[1]) for a, b in pairings], axis=1)
            chosen = pairings[np.argmax(scores, axis=1)]            # [M, 2, 2]
            rows = np.arange(len(slots))[:, None]
            ea, eb = np.r_[ea, slots[rows, chosen[:, :, 0]].ravel()], np.r_[eb, slots[rows, chosen[:, :, 1]].ravel()]
    fa, ca = ea % F, ea // F
    fb, cb = eb % F, eb // F
    # Corner node 3 * i + c. Across a manifold edge, join the corners that
    # hold the same vertex; the two faces may traverse the edge either way.
    same = edges[ea, 0] == edges[eb, 0]
    a0, a1 = 3 * fa + ca, 3 * fa + (ca + 1) % 3
    b0, b1 = 3 * fb + cb, 3 * fb + (cb + 1) % 3
    rows = np.r_[a0, a1]
    cols = np.r_[np.where(same, b0, b1), np.where(same, b1, b0)]
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(3 * F, 3 * F))
    _, fan = connected_components(graph, directed=False)
    fan = fan.astype(np.int64)
    _, fan_first = np.unique(fan, return_index=True)
    return np.asarray(vertices)[faces.reshape(-1)[fan_first]], fan.reshape(F, 3)


def make_edge_manifold(vertices, faces, pair_sheets=False):
    """Make every edge manifold and the winding consistent.

    Cutting alone is the default. Pairing first gives a better-connected
    mesh with about a third fewer UV charts, but xatlas then takes 6 to 11
    minutes per asset instead of under a minute on the meshes measured.
    """
    v, f = vertices, faces
    if pair_sheets:
        v, f = split_non_manifold_vertices(v, f, pair_sheets=True)
    v, f = split_non_manifold_vertices(v, f, pair_sheets=False)
    return v, orient_faces(v, f)


def cluster_vertices(vertices, faces, cell):
    """Merge vertices that fall in the same cube of side `cell`.

    A fast, memory-light first pass for meshes too large for quadric
    simplification. Faces that collapse to a line or point are dropped, as are
    duplicate faces.
    """
    v = np.asarray(vertices, dtype=np.float64)
    cells = np.floor((v - v.min(axis=0)) / cell).astype(np.int64)
    extent = cells.max(axis=0) + 1
    keys = (cells[:, 0] * extent[1] + cells[:, 1]) * extent[2] + cells[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    merged = np.stack([np.bincount(inverse, weights=v[:, axis]) for axis in range(3)], axis=1) / counts[:, None]
    f = inverse[np.asarray(faces, dtype=np.int64)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = f[np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]]
    return merged, f


def _mean_edge_length(vertices, faces):
    return float(np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1).mean())


def clean_faces(vertices, faces):
    """Drop faces with a repeated vertex, zero area, or a duplicate triangle.

    The simplifier leaves a few thousand of these on large meshes, and xatlas
    does not cope well with them.
    """
    f = np.asarray(faces, dtype=np.int64)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    tri = np.asarray(vertices)[f]
    keep &= np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 1e-14
    f = f[keep]
    return f[np.sort(np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1])]


def _decimate(v, f, target):
    """Quadric-decimate to about `target` faces.

    One call can stop early, so repeat until it reaches the target or stalls.
    """
    import fast_simplification

    for _ in range(6):
        if len(f) <= target * 1.05:
            break
        before = len(f)
        v, f = fast_simplification.simplify(v, f, 1.0 - target / len(f))
        f = np.asarray(f, dtype=np.int64)
        if len(f) > before * 0.98:
            break
    return v, f


def off_surface_fraction(vertices, faces, coords, origin, voxel_size):
    """Share of face area whose centroid has no occupied voxel within one voxel.

    A proxy for the texels the baker will miss: faces that bridge a gap or
    span the interior of a solid instead of following the voxel shell.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    c = np.asarray(coords, dtype=np.int64)
    span = int(c.max()) + 2

    def key(cells):
        return (cells[:, 0] * span + cells[:, 1]) * span + cells[:, 2]

    occupied = np.unique(key(c))
    cell = np.floor((v[f].mean(axis=1) - np.asarray(origin)) / voxel_size).astype(np.int64)
    hit = np.zeros(len(f), dtype=bool)
    for d in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        n = cell + d
        inside = ((n >= 0) & (n < span)).all(axis=1)
        k = key(n[inside])
        i = np.minimum(np.searchsorted(occupied, k), len(occupied) - 1)
        hit[np.flatnonzero(inside)[occupied[i] == k]] = True
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(area[~hit].sum() / max(area.sum(), 1e-30))


def _repair_and_decimate(v, f, target, pair):
    v, f = make_edge_manifold(v, orient_faces(v, f), pair)
    v, f = _decimate(v, f, target)
    if len(f) > target * 1.3:
        # The simplifier keeps boundary edges, so it can stall on meshes with
        # many of them. Cluster the rest of the way.
        v, f = cluster_vertices(v, f, _mean_edge_length(v, f) * np.sqrt(len(f) / target))
        v, f = _decimate(v, f, target)
    # Simplification creates new non-manifold edges, and xatlas can take tens
    # of minutes on them. Clean, orient, then repair again before unwrapping.
    f = clean_faces(v, f)
    return make_edge_manifold(v, orient_faces(v, f), pair)


def simplify_mesh(vertices, faces, target, voxel_size=None, max_raw_faces=8_000_000, pair_sheets=False, repair="pair",
                  coords=None, origin=None, off_surface_limit=0.03, log=print):
    """Repair, then decimate to about `target` faces.

    `repair` says what to do with edges shared by three or more faces, which
    the extractor and vertex clustering both produce:
      - "pair" (default): keep the two faces that continue straight through
        each such edge together and cut the rest into fans, before and after
        decimation. Half the cracks of "cut", but xatlas takes minutes to an
        hour. On some meshes the decimator stalls above the target on the
        paired mesh and flattens it into large facets off the surface; when
        `coords` and `origin` are given, the cut repair is run as well and
        chosen if it leaves `off_surface_limit` less of the area off the
        voxel shell, or whenever pairing stalled.
      - "cut": split every such edge into fans; unwraps in seconds.
    `pair_sheets=True` is the old spelling of `repair="pair"`.

    Either way the decimator moves the two sides of a cut independently, so
    cuts open into cracks: on a 512-level mesh 3% of edges are open in the
    raw mesh and 30% after decimation. Decimating before cutting leaves 9%
    but makes xatlas run for over half an hour, and welding the sides back
    together afterwards, by distance or by the decimator's collapse record,
    removes a quarter to a half of the triangles. See docs/xpu-measurements.md.

    Meshes above `max_raw_faces` get a vertex-clustering pass first: the repair
    steps and the quadric simplifier would otherwise need tens of GiB of RAM
    and many minutes. Triangle count on a surface scales with 1 / cell^2.
    """
    if pair_sheets:
        repair = "pair"
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > max_raw_faces:
        # Cells of two voxels merge the two sides of one-voxel walls into one
        # sheet, which the repair below handles well. Three-voxel cells merge
        # far more and left a 57M-triangle mesh fragmented, so stop at two.
        base = voxel_size if voxel_size else _mean_edge_length(v, f)
        v, f = cluster_vertices(v, f, base * min(2.0, np.ceil(np.sqrt(len(f) / max_raw_faces))))
    if repair != "pair":
        return _repair_and_decimate(v, f, target, False)
    vp, fp = _repair_and_decimate(v, f, target, True)
    stalled = len(fp) > target * 1.1
    if coords is None or voxel_size is None:
        return vp, fp
    off_pair = off_surface_fraction(vp, fp, coords, origin, voxel_size)
    if not stalled and off_pair <= off_surface_limit:
        log(f"Repair: pair, {off_pair:.1%} of the area off the surface")
        return vp, fp
    vc, fc = _repair_and_decimate(v, f, target, False)
    off_cut = off_surface_fraction(vc, fc, coords, origin, voxel_size)
    why = "stalled at {:,} faces".format(len(fp)) if stalled else f"{off_pair:.1%} of the area off the surface"
    if stalled or off_cut + off_surface_limit < off_pair:
        log(f"Repair: cut, because pairing {why} (cut: {off_cut:.1%} off the surface)")
        return vc, fc
    log(f"Repair: pair, {off_pair:.1%} of the area off the surface (cut: {off_cut:.1%})")
    return vp, fp


def uv_unwrap(vertices, faces):
    """
    Compute UV coordinates for a mesh using xatlas.

    Returns:
        new_vertices: Remapped vertices
        new_faces: Triangle indices into new_vertices
        uvs: UV coordinates per new vertex, in [0, 1]
        vmapping: Maps new vertex index -> original vertex index
    """
    import xatlas

    v = np.ascontiguousarray(vertices.astype(np.float32))
    f = np.ascontiguousarray(faces.astype(np.uint32))

    vmapping, indices, uvs = xatlas.parametrize(v, f)

    new_vertices = v[vmapping]
    new_faces = indices.reshape(-1, 3)

    return new_vertices, new_faces, uvs, vmapping


def _rasterize_uv_triangles(vertices, faces, uvs, texture_size):
    """
    Rasterize all triangles in UV space. For each texel, determine which
    triangle covers it and compute 3D position via barycentric interpolation.

    Args:
        vertices: [N, 3] mesh vertices
        faces: [F, 3] triangle indices
        uvs: [N, 2] UV coordinates in [0, 1]
        texture_size: output texture resolution

    Returns:
        positions: [H, W, 3] 3D position at each texel
        mask: [H, W] bool mask of filled texels
    """
    H = W = texture_size
    positions = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    n_faces = len(faces)
    uv_scale = np.array([W - 1, H - 1], dtype=np.float32)

    for fi in range(n_faces):
        if fi > 0 and fi % 100000 == 0:
            print(f"    Rasterizing: {fi:,}/{n_faces:,}")

        i0, i1, i2 = faces[fi]
        uv0 = uvs[i0] * uv_scale
        uv1 = uvs[i1] * uv_scale
        uv2 = uvs[i2] * uv_scale
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), W - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), H - 1)

        if max_x < min_x or max_y < min_y:
            continue

        d00 = uv1[0] - uv0[0]
        d01 = uv2[0] - uv0[0]
        d10 = uv1[1] - uv0[1]
        d11 = uv2[1] - uv0[1]
        denom = d00 * d11 - d01 * d10
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        px_range = np.arange(min_x, max_x + 1, dtype=np.float32)
        py_range = np.arange(min_y, max_y + 1, dtype=np.float32)
        if len(px_range) == 0 or len(py_range) == 0:
            continue

        px_grid, py_grid = np.meshgrid(px_range, py_range)
        dx = px_grid - uv0[0]
        dy = py_grid - uv0[1]

        u = (dx * d11 - d01 * dy) * inv_denom
        v = (d00 * dy - dx * d10) * inv_denom
        w = 1.0 - u - v

        inside = (u >= -0.001) & (v >= -0.001) & (w >= -0.001)
        if not inside.any():
            continue

        pos_3d = w[..., None] * p0 + u[..., None] * p1 + v[..., None] * p2

        iy, ix = np.where(inside)
        positions[py_range.astype(int)[iy], px_range.astype(int)[ix]] = pos_3d[iy, ix]
        mask[py_range.astype(int)[iy], px_range.astype(int)[ix]] = True

    return positions, mask


def sample_sparse_trilinear(coords, attrs, query):
    """Sample sparse voxels at continuous voxel-space points.

    Returns [M, C] values and [M] bool, False where no neighbor voxel is active.
    """
    coords = np.asarray(coords, dtype=np.int64)
    extent = coords.max(axis=0) + 3
    scale = np.array([extent[1] * extent[2], extent[2], 1], dtype=np.int64)
    keys = coords @ scale
    order = np.argsort(keys)
    sorted_keys = keys[order]

    base = np.floor(query - 0.5).astype(np.int64)
    out = np.zeros((len(query), attrs.shape[1]), dtype=np.float32)
    weight_sum = np.zeros(len(query), dtype=np.float32)
    for corner in range(8):
        voxel = base + np.array([corner & 1, (corner >> 1) & 1, (corner >> 2) & 1])
        inside = ((voxel >= 0) & (voxel < extent)).all(axis=1)
        q = voxel @ scale
        pos = np.searchsorted(sorted_keys, q).clip(max=len(keys) - 1)
        hit = inside & (sorted_keys[pos] == q)
        w = np.prod(1 - np.abs(query[hit] - voxel[hit] - 0.5), axis=1)
        out[hit] += w[:, None] * attrs[order[pos[hit]]]
        weight_sum[hit] += w
    valid = weight_sum > 0
    out[valid] /= weight_sum[valid, None]
    return out, valid


def sample_nearest_voxel(coords, attrs, query, max_distance):
    """Attributes of the nearest active voxel within `max_distance` voxels.

    Returns [M, C] values and [M] bool, False where no voxel is that close.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(coords, dtype=np.float64) + 0.5)
    distance, index = tree.query(query, distance_upper_bound=max_distance)
    found = np.isfinite(distance)
    out = np.zeros((len(query), attrs.shape[1]), dtype=np.float32)
    out[found] = attrs[index[found]]
    return out, found


# Simplification moves the surface off the voxel shell, by more at high
# resolutions. Texels that miss it take the nearest voxel within this radius.
NEAREST_VOXEL_RADIUS = 3.0


def bake_texture(vertices, faces, uvs, voxel_coords, voxel_attrs, origin, voxel_size, texture_size=2048):
    """Bake voxel attributes into textures.

    Attribute layout: base colour 0:3, metallic 3, roughness 4, alpha 5.
    Returns the RGB base colour image, the glTF metallic-roughness image, and
    the share of covered texels that had a voxel to sample trilinearly.
    """
    t0 = time.time()
    positions, mask = _rasterize_uv_triangles(vertices, faces, uvs, texture_size)
    query = (positions[mask] - np.asarray(origin)) / voxel_size
    voxel_attrs = np.asarray(voxel_attrs)
    sampled, valid = sample_sparse_trilinear(voxel_coords, voxel_attrs, query)
    hit_rate = float(valid.mean()) if len(valid) else 0.0
    missed = np.flatnonzero(~valid)
    if len(missed):
        near, found = sample_nearest_voxel(voxel_coords, voxel_attrs, query[missed], NEAREST_VOXEL_RADIUS)
        sampled[missed[found]] = near[found]
        valid = valid.copy()
        valid[missed[found]] = True
    print(f"  Texels: {len(valid):,}, trilinear hits {hit_rate:.2%}, "
          f"with nearest voxel {float(valid.mean()) if len(valid) else 0.0:.2%}, {time.time() - t0:.1f}s")

    filled = np.zeros_like(mask)
    filled[mask] = valid
    image = np.zeros((texture_size, texture_size, sampled.shape[1]), dtype=np.uint8)
    image[mask] = (np.clip(sampled, 0, 1) * 255).astype(np.uint8)

    # Fill everything outside the sampled texels, so UV seams do not show black.
    gaps = (~filled).astype(np.uint8)
    base_color = cv2.inpaint(image[..., :3], gaps, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(image[..., 3], gaps, 1, cv2.INPAINT_TELEA)
    roughness = cv2.inpaint(image[..., 4], gaps, 1, cv2.INPAINT_TELEA)

    # glTF packs roughness in G and metallic in B.
    mr = np.stack([np.zeros_like(metallic), roughness, metallic], axis=-1)
    return base_color, mr, hit_rate


def export_glb_with_texture(vertices, faces, uvs, base_color_img, mr_img, output_path, normals=None):
    """Export a textured GLB, converting to glTF's Y-up axes."""
    import trimesh
    from PIL import Image

    v = np.array(vertices, dtype=np.float32)
    v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1]
    if normals is not None:
        normals = np.array(normals, dtype=np.float32)
        normals[:, 1], normals[:, 2] = normals[:, 2].copy(), -normals[:, 1]

    # trimesh flips V again on export, so texture row r ends up at glTF v = r / H.
    uv = np.array(uvs, dtype=np.float32)
    uv[:, 1] = 1 - uv[:, 1]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base_color_img),
        metallicRoughnessTexture=Image.fromarray(mr_img),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=v, faces=faces, vertex_normals=normals, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )
    mesh.export(output_path)
    return output_path
