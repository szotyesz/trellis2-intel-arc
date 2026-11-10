"""The NumPy voxeliser against a loop-for-loop transcription of o_voxel's C++."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from backends.dual_grid import mesh_to_flexible_dual_grid  # noqa: E402


def reference(vertices, faces, grid, face_weight, boundary_weight, regularization_weight):
    """flexible_dual_grid.cpp, statement by statement, in float64."""
    h = 1.0 / grid
    tri = (np.asarray(vertices, np.float32).astype(np.float64) + 0.5)[faces]
    voxels = {}     # coord -> [sum of points, count, flags, Q]

    def plane_q(v0, v1, v2):
        n = np.cross(v1 - v0, v2 - v1)
        length = np.linalg.norm(n)
        n = n / length if length > 0 else n
        plane = np.r_[n, -n @ v0]
        return n, np.outer(plane, plane)

    def lerp(a, b, t, va, vb):
        if a == b:
            return va
        alpha = (t - a) / (b - a)
        return (1 - alpha) * va + alpha * vb

    def clamp(value):
        return min(max(int(value / h), 0), grid - 1)

    for v0, v1, v2 in tri:
        _, Q = plane_q(v0, v1, v2)
        for ax2 in range(3):
            ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
            t = sorted((np.array([v[ax0], v[ax1], v[ax2]]) for v in (v0, v1, v2)), key=lambda p: p[1])
            start, mid, end = (clamp(p[1]) for p in t)
            for lo, hi, (a, b, c) in ((start, mid, (t[0], t[1], t[2])), (mid, end, (t[2], t[1], t[0]))):
                for y_idx in range(lo, hi):
                    y = (y_idx + 1) * h
                    t3 = lerp(a[1], b[1], y, a[[0, 2]], b[[0, 2]])
                    t4 = lerp(a[1], c[1], y, a[[0, 2]], c[[0, 2]])
                    if t3[0] > t4[0]:
                        t3, t4 = t4, t3
                    for x_idx in range(clamp(t3[0]), clamp(t4[0])):
                        x = (x_idx + 1) * h
                        z = lerp(t3[0], t4[0], x, t3[1], t4[1])
                        z_idx = int(z / h)
                        if not 0 <= z_idx < grid:
                            continue
                        point = np.zeros(3)
                        point[ax0], point[ax1], point[ax2] = x, y, z
                        for dx in range(2):
                            for dy in range(2):
                                coord = [0, 0, 0]
                                coord[ax0], coord[ax1], coord[ax2] = x_idx + dx, y_idx + dy, z_idx
                                entry = voxels.setdefault(tuple(coord), [np.zeros(3), 0, [False] * 3, np.zeros((4, 4))])
                                entry[0] += point
                                entry[1] += 1
                                entry[3] += Q
                                if dx == 0 and dy == 0:
                                    entry[2][ax2] = True

    if face_weight > 0:
        for v0, v1, v2 in tri:
            n, Q = plane_q(v0, v1, v2)
            e = [v1 - v0, v2 - v1, v0 - v2]
            lo = np.maximum((np.minimum(np.minimum(v0, v1), v2) / h).astype(int), 0)
            hi = np.minimum((np.maximum(np.maximum(v0, v1), v2) / h + 1).astype(int), grid)
            c = np.where(n > 0, h, 0.0)
            d1, d2 = n @ (c - v0), n @ (h - c - v0)
            for x in range(lo[0], hi[0]):
                for y in range(lo[1], hi[1]):
                    for z in range(lo[2], hi[2]):
                        if (x, y, z) not in voxels:
                            continue
                        p = np.array([x, y, z]) * h
                        ok = (n @ p + d1) * (n @ p + d2) <= 0
                        for a, b, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                            mul = -1 if n[k] < 0 else 1
                            for edge, corner in zip(e, (v0, v1, v2)):
                                ne = np.array([-mul * edge[b], mul * edge[a]])
                                d = -ne @ corner[[a, b]] + np.maximum(ne, 0).sum() * h
                                ok &= ne @ p[[a, b]] + d >= 0
                        if ok:
                            voxels[(x, y, z)][3] += Q

    if boundary_weight > 0:
        count = {}
        for f in faces:
            for i in range(3):
                key = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
                count[key] = count.get(key, 0) + 1
        shifted = np.asarray(vertices, np.float32).astype(np.float64) + 0.5
        for (i0, i1), n_faces in sorted(count.items()):
            if n_faces != 1:
                continue
            v0, v1 = shifted[i0], shifted[i1]
            direction = v1 - v0
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction /= length
            A = np.eye(3) - np.outer(direction, direction)
            b = -A @ v0
            Q = np.zeros((4, 4))
            Q[:3, :3], Q[:3, 3], Q[3, :3], Q[3, 3] = A, b, b, v0 @ A @ v0
            current = np.floor(v0 / h).astype(int)
            step = np.where(direction > 0, 1, -1)
            t_max, t_delta = np.full(3, np.inf), np.full(3, np.inf)
            for axis in range(3):
                if direction[axis] != 0:
                    border = h * (current[axis] + (1 if step[axis] > 0 else 0))
                    t_max[axis] = (border - v0[axis]) / direction[axis]
                    t_delta[axis] = h / abs(direction[axis])
            passed = [tuple(current)]
            while True:
                axis = int(np.argmin(t_max))
                if t_max[axis] > length:
                    break
                current[axis] += step[axis]
                t_max[axis] += t_delta[axis]
                passed.append(tuple(current))
            for coord in passed:
                if coord in voxels:
                    voxels[coord][3] += boundary_weight * Q

    out = {}
    for coord, (total, n, flags, Q) in voxels.items():
        lo, hi = np.array(coord) * h, (np.array(coord) + 1) * h
        mean = total / n
        if regularization_weight > 0:
            R = np.zeros((4, 4))
            R[:3, :3], R[:3, 3], R[3, :3], R[3, 3] = np.eye(3), -mean, -mean, mean @ mean
            Q = Q + regularization_weight * n * R
        p = np.linalg.solve(Q[:3, :3], -Q[:3, 3])
        if not ((p >= lo) & (p <= hi)).all():
            best = [np.inf, p]

            def offer(q):
                err = np.r_[q, 1] @ Q @ np.r_[q, 1]
                if err < best[0]:
                    best[:] = [err, q.copy()]

            bounds = (lo, hi)
            for fixed in range(3):
                a1, a2 = (fixed + 1) % 3, (fixed + 2) % 3
                for bound in bounds:
                    x = np.linalg.solve(Q[np.ix_([a1, a2], [a1, a2])], -(Q[[a1, a2], fixed] * bound[fixed] + Q[[a1, a2], 3]))
                    if lo[a1] <= x[0] <= hi[a1] and lo[a2] <= x[1] <= hi[a2]:
                        q = np.zeros(3)
                        q[fixed], q[a1], q[a2] = bound[fixed], x[0], x[1]
                        offer(q)
            for free in range(3):
                a1, a2 = (free + 1) % 3, (free + 2) % 3
                for b1 in bounds:
                    for b2 in bounds:
                        x = -(Q[free, a1] * b1[a1] + Q[free, a2] * b2[a2] + Q[free, 3]) / Q[free, free]
                        if lo[free] <= x <= hi[free]:
                            q = np.zeros(3)
                            q[free], q[a1], q[a2] = x, b1[a1], b2[a2]
                            offer(q)
            for corner in range(8):
                offer(np.array([bounds[(corner >> axis) & 1][axis] for axis in range(3)]))
            p = best[1]
        out[coord] = (p, tuple(flags))
    return out


def open_cap():
    """An icosphere with its top cut off, so it has open edges."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.37)
    faces = sphere.faces[sphere.triangles_center[:, 2] < 0.2]
    return np.asarray(sphere.vertices) + [0.013, -0.021, 0.007], faces


@pytest.mark.parametrize("weights", [(1.0, 0.2, 1e-2), (0.0, 0.0, 0.1), (1.0, 1.0, 0.05)])
def test_matches_the_cpp_algorithm(weights):
    vertices, faces = open_cap()
    grid = 24
    expected = reference(vertices, faces, grid, *weights)
    coords, dual, flags = mesh_to_flexible_dual_grid(
        vertices, faces, grid, face_weight=weights[0], boundary_weight=weights[1], regularization_weight=weights[2])
    assert {tuple(c) for c in coords.tolist()} == set(expected)
    want = np.array([expected[tuple(c)][0] for c in coords.tolist()])
    assert np.array_equal(flags, np.array([expected[tuple(c)][1] for c in coords.tolist()]))
    # A handful of voxels have two candidates of equal error; allow those.
    close = np.abs(dual - want).max(axis=1) < 1e-5
    assert close.mean() > 0.995, np.abs(dual - want).max(axis=1)[~close]


def test_large_and_sliver_triangles():
    """Two triangles spanning the grid and a degenerate one."""
    vertices = np.array([[-0.45, -0.4, -0.3], [0.45, -0.35, 0.1], [0.4, 0.45, 0.35], [-0.4, 0.4, -0.1], [0.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 4, 4]])
    expected = reference(vertices, faces, 32, 1.0, 0.2, 1e-2)
    coords, dual, flags = mesh_to_flexible_dual_grid(vertices, faces, 32, boundary_weight=0.2, regularization_weight=1e-2)
    assert {tuple(c) for c in coords.tolist()} == set(expected)
    want = np.array([expected[tuple(c)][0] for c in coords.tolist()])
    assert (np.abs(dual - want).max(axis=1) < 1e-5).mean() > 0.995


def test_round_trip_through_the_mesh_extractor():
    """Voxelise a sphere, extract it again: the surface comes back closed."""
    import torch
    from backends.mesh_extract import flexible_dual_grid_to_mesh

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    grid = 48
    coords, dual, flags = mesh_to_flexible_dual_grid(sphere.vertices, sphere.faces, grid,
                                                     boundary_weight=0.2, regularization_weight=1e-2)
    offsets = dual * grid - coords
    assert offsets.min() >= -1e-4 and offsets.max() <= 1 + 1e-4
    v, f = flexible_dual_grid_to_mesh(torch.from_numpy(coords), torch.from_numpy(offsets), torch.from_numpy(flags),
                                      None, [[-0.5] * 3, [0.5] * 3], grid_size=grid)
    out = trimesh.Trimesh(v.numpy(), f.numpy(), process=False)
    assert np.abs(np.linalg.norm(out.vertices, axis=1) - 0.4).max() < 0.5 / grid
    assert out.is_watertight
    assert abs(out.area - sphere.area) / sphere.area < 0.02
