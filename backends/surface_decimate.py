"""Quadric edge-collapse decimation that keeps faces on the voxel shell.

Quadric error measures how far a vertex moves from the planes around it, and
nothing about the triangle between the vertices. On a facade with recessed
windows, sliding a vertex along the wall costs nothing, so the plain
decimator fans large triangles over every opening while it spends the budget
on foliage and balustrades, where every collapse has a high error. On the
town at 1024, 11% of the area ended more than a voxel off the surface that
way. This decimator processes collapses in error order and refuses any whose
new triangles would have a centroid farther than one voxel from an occupied
voxel, the same test as `texture_baker.off_surface_fraction`; such collapses
are kept with a penalty and only taken once no clean collapse is left.

Pure Python over a heap, about 200,000 collapses a minute, so
`texture_baker.simplify_mesh` lets the C++ decimator take the mesh most of
the way down first.
"""

import heapq
import time

import numpy as np

PENALTY = 1e6
# Weight of the plane that holds each open edge in place, relative to the
# edge's squared length. Without it every collapse along a flat sheet's
# border is free, the borders erode and the cracks between cut sheets open.
BORDER_WEIGHT = 10.0
# Reject a collapse that turns a face by more than about 78 degrees, which
# includes flips and slivers.
MIN_NORMAL_DOT = 0.2
_OFFSETS = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T  # [27, 3]


class VoxelShell:
    """Occupied voxels as sorted keys; answers whether points lie within one voxel."""

    def __init__(self, coords, origin, voxel_size):
        c = np.asarray(coords, dtype=np.int64)
        self.span = int(c.max()) + 2
        self.keys = np.unique(self._key(c))
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel_size = float(voxel_size)

    def _key(self, cells):
        return (cells[:, 0] * self.span + cells[:, 1]) * self.span + cells[:, 2]

    def near(self, points):
        """[M] bool: an occupied voxel lies in the 3x3x3 cells around each point's cell."""
        cell = np.floor((np.atleast_2d(points) - self.origin) / self.voxel_size).astype(np.int64)
        n = (cell[:, None, :] + _OFFSETS[None]).reshape(-1, 3)
        inside = ((n >= 0) & (n < self.span)).all(axis=1)
        k = self._key(np.where(inside[:, None], n, 0))
        i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return ((self.keys[i] == k) & inside).reshape(len(cell), 27).any(axis=1)


def _face_quadrics(v, f):
    """Area-weighted plane quadrics, as the upper triangle of a 4x4 matrix, [F, 10]."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(n, axis=1)
    n /= area2[:, None] + 1e-300
    plane = np.concatenate([n, -(n * tri[:, 0]).sum(axis=1)[:, None]], axis=1)
    iu = np.triu_indices(4)
    return (0.5 * area2)[:, None] * (plane[:, :, None] * plane[:, None, :])[:, iu[0], iu[1]]


def _quadric_error(Q, p):
    """p^T Q p for packed quadrics Q [k, 10] and points p [k, 3]."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    a, b, c, d, e, f, g, h, i, j = (Q[:, k] for k in range(10))
    return a * x * x + e * y * y + h * z * z + j + 2 * (b * x * y + c * x * z + d * x + f * y * z + g * y + i * z)


def _quadric_minimum(Q, mid, edge_length):
    """Minimiser of each quadric nearest the edge midpoint; NaN where it is far away.

    The quadric of a flat wall is singular along the wall, and of a crease
    along the crease line. The pseudo-inverse keeps the free directions at
    the midpoint, so the vertex stays where it was instead of sliding off to
    a spike or a displaced hub, which the plain decimator's exact solve allows.
    """
    A = np.stack([Q[:, [0, 1, 2]], Q[:, [1, 4, 5]], Q[:, [2, 5, 7]]], axis=1)
    residual = -Q[:, [3, 6, 8]] - np.einsum("kij,kj->ki", A, mid)
    scale = np.abs(A).max(axis=(1, 2)) + 1e-300
    step = np.einsum("kij,kj->ki", np.linalg.pinv(A / scale[:, None, None], rcond=1e-4, hermitian=True), residual / scale[:, None])
    out = mid + step
    out[np.linalg.norm(step, axis=1) > 2.0 * edge_length] = np.nan
    return out


def _border_quadrics(v, f, border_weight):
    """Constraint quadrics for open edges: the plane through the edge perpendicular
    to its face, weighted by `border_weight` times the squared edge length.
    Returns the edge array [B, 2] and the quadrics [B, 10]."""
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    face = np.tile(np.arange(len(f)), 3)
    key = np.sort(edges, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    single = first[counts == 1]
    e, fi = edges[single], face[single]
    if not len(e):
        return e, np.zeros((0, 10))
    tri = v[f[fi]]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    d = v[e[:, 1]] - v[e[:, 0]]
    n = np.cross(d, fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    plane = np.concatenate([n, -(n * v[e[:, 0]]).sum(axis=1)[:, None]], axis=1)
    iu = np.triu_indices(4)
    w = border_weight * (d * d).sum(axis=1)
    return e, w[:, None] * (plane[:, :, None] * plane[:, None, :])[:, iu[0], iu[1]]


class SurfaceDecimator:
    def __init__(self, vertices, faces, shell, border_weight=BORDER_WEIGHT):
        self.V = np.array(vertices, dtype=np.float64)
        self.F = np.array(faces, dtype=np.int64)
        self.alive = np.ones(len(self.F), dtype=bool)
        self.shell = shell
        N = len(self.V)
        fq = _face_quadrics(self.V, self.F)
        self.Q = np.zeros((N, 10))
        for c in range(3):
            np.add.at(self.Q, self.F[:, c], fq)
        if border_weight:
            be, bq = _border_quadrics(self.V, self.F, border_weight)
            for c in range(2):
                np.add.at(self.Q, be[:, c], bq)
        self.vf = [set() for _ in range(N)]
        for i, (a, b, c) in enumerate(self.F.tolist()):
            self.vf[a].add(i)
            self.vf[b].add(i)
            self.vf[c].add(i)
        edges = np.sort(np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]]), axis=1)
        self.edges, counts = np.unique(edges, axis=0, return_counts=True)
        self.border = np.zeros(N, dtype=bool)
        self.border[self.edges[counts == 1].ravel()] = True
        self.version = np.zeros(N, dtype=np.int64)
        self.heap = []
        self.removed = 0

    def _candidates(self, a, b):
        """Positions [k, 4, 3] and errors [k, 4] for collapsing edges (a, b[i])."""
        Va, Vb = np.broadcast_to(self.V[a], (len(b), 3)), self.V[b]
        mid = 0.5 * (Va + Vb)
        Q = self.Q[a][None] + self.Q[b]
        cands = np.stack([_quadric_minimum(Q, mid, np.linalg.norm(Va - Vb, axis=1)), mid, Va, Vb], axis=1)
        # An interior vertex merges into a border vertex, which stays put; two
        # border vertices keep one of their positions, so open edges hold their shape.
        ba, bb = self.border[a], self.border[b]
        cands[bb & ~ba, :3] = Vb[bb & ~ba][:, None]
        cands[~bb & ba, :2] = Va[~bb & ba][:, None]
        cands[~bb & ba, 3] = Va[~bb & ba]
        cands[bb & ba, :2] = Va[bb & ba][:, None]
        err = np.stack([_quadric_error(Q, cands[:, i]) for i in range(4)], axis=1)
        return cands, np.where(np.isnan(err), np.inf, err)

    def _push_edges(self, a):
        nbrs = self._neighbours(a)
        if not nbrs:
            return
        b = np.fromiter(nbrs, dtype=np.int64, count=len(nbrs))
        cost = self._candidates(a, b)[1].min(axis=1)
        va = int(self.version[a])
        for c, bi, vb in zip(cost.tolist(), b.tolist(), self.version[b].tolist()):
            heapq.heappush(self.heap, (c, a, bi, va, vb, False))

    def _neighbours(self, a):
        s = set()
        for fi in self.vf[a]:
            s.update(self.F[fi].tolist())
        s.discard(a)
        return s

    def _try_collapse(self, a, b, penalized):
        """True if collapsed, "penalize" if only the shell test failed, else False."""
        fa, fb = self.vf[a], self.vf[b]
        shared = fa & fb
        if not shared:
            return False
        # Link condition: a common neighbour outside the shared faces would
        # make a non-manifold edge.
        third = set()
        for fi in shared:
            third.update(self.F[fi].tolist())
        if (self._neighbours(a) & self._neighbours(b)) - third:
            return False
        moving = list((fa | fb) - shared)
        if not moving:
            return False
        tri = self.F[moving]
        P = self.V[tri]
        n_old = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        n_old /= np.linalg.norm(n_old, axis=1, keepdims=True) + 1e-300
        corner = ((tri == a) | (tri == b))[..., None]
        cands, err = self._candidates(a, np.array([b]))
        flips_ok = False
        for ci in np.argsort(err[0]):
            if not np.isfinite(err[0, ci]):
                break
            p = cands[0, ci]
            P2 = np.where(corner, p, P)
            n_new = np.cross(P2[:, 1] - P2[:, 0], P2[:, 2] - P2[:, 0])
            area2 = np.linalg.norm(n_new, axis=1)
            if (area2 < 1e-300).any() or ((n_new * n_old).sum(axis=1) < MIN_NORMAL_DOT * area2).any():
                continue
            flips_ok = True
            if not penalized and not self.shell.near(P2.mean(axis=1)).all():
                continue
            self._apply(a, b, p, shared)
            return True
        return "penalize" if flips_ok and not penalized else False

    def _apply(self, a, b, p, shared):
        for fi in shared:
            self.alive[fi] = False
            for vtx in self.F[fi].tolist():
                self.vf[vtx].discard(fi)
        for fi in self.vf[b]:
            row = self.F[fi]
            row[row == b] = a
            self.vf[a].add(fi)
        self.vf[b] = set()
        self.V[a] = p
        self.Q[a] += self.Q[b]
        self.border[a] |= self.border[b]
        self.version[a] += 1
        self.version[b] = -1
        self.removed += len(shared)
        self._push_edges(a)

    def run(self, target, log=print):
        t0 = time.time()
        E = self.edges
        for s in range(0, len(E), 500_000):
            e = E[s:s + 500_000]
            Q = self.Q[e[:, 0]] + self.Q[e[:, 1]]
            Va, Vb = self.V[e[:, 0]], self.V[e[:, 1]]
            mid = 0.5 * (Va + Vb)
            opt = _quadric_minimum(Q, mid, np.linalg.norm(Va - Vb, axis=1))
            cost = np.nanmin(np.stack([_quadric_error(Q, c) for c in (opt, mid, Va, Vb)], 1), axis=1)
            cost = np.where(np.isnan(cost), np.inf, cost)
            self.heap.extend(zip(cost.tolist(), e[:, 0].tolist(), e[:, 1].tolist(),
                                 [0] * len(e), [0] * len(e), [False] * len(e)))
        heapq.heapify(self.heap)
        n_alive = len(self.F)
        penalized = taken = 0
        while n_alive > target and self.heap:
            cost, a, b, va, vb, pen = heapq.heappop(self.heap)
            if self.version[a] != va or self.version[b] != vb:
                continue
            result = self._try_collapse(a, b, pen)
            if result is True:
                taken += pen
                n_alive = len(self.F) - self.removed
            elif result == "penalize":
                penalized += 1
                heapq.heappush(self.heap, (cost + PENALTY, a, b, va, vb, True))
        log(f"  Surface-aware decimation: {len(self.F):,} -> {n_alive:,} faces in {time.time() - t0:.0f}s; "
            f"{penalized:,} collapses would leave the shell, {taken:,} of them taken")
        f = self.F[self.alive]
        used, inverse = np.unique(f, return_inverse=True)
        return self.V[used], inverse.reshape(-1, 3)


def decimate_on_shell(vertices, faces, target, shell, border_weight=BORDER_WEIGHT, log=print):
    """Decimate to about `target` faces, keeping new triangles on the `VoxelShell`."""
    return SurfaceDecimator(vertices, faces, shell, border_weight).run(target, log)
