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


def unweld_non_manifold(vertices, faces):
    """Give faces on edges shared by 3+ faces private vertices. Removes no geometry.

    The extractor produces many such edges. They stall the simplifier and make
    xatlas take hours.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = edges.min(axis=1) * (int(faces.max()) + 1) + edges.max(axis=1)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    bad = (counts[inverse] > 2).reshape(3, -1).any(axis=0)
    out = faces.copy()
    out[bad] = len(vertices) + np.arange(3 * int(bad.sum())).reshape(-1, 3)
    return np.vstack([vertices, vertices[faces[bad]].reshape(-1, 3)]), out


def simplify_mesh(vertices, faces, target):
    """Repair, then decimate to about `target` faces."""
    import fast_simplification

    v = np.asarray(vertices, dtype=np.float64)
    v, f = unweld_non_manifold(v, orient_faces(v, faces))
    # One call can stop early, so repeat until it reaches the target or stalls.
    for _ in range(6):
        if len(f) <= target * 1.05:
            break
        before = len(f)
        v, f = fast_simplification.simplify(v, f, 1.0 - target / len(f))
        if len(f) > before * 0.98:
            break
    return v, orient_faces(v, f)


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


def bake_texture(vertices, faces, uvs, voxel_coords, voxel_attrs, origin, voxel_size, texture_size=2048):
    """Bake voxel attributes into textures.

    Attribute layout: base colour 0:3, metallic 3, roughness 4, alpha 5.
    Returns the RGB base colour image, the glTF metallic-roughness image, and
    the share of covered texels that had a voxel to sample.
    """
    t0 = time.time()
    positions, mask = _rasterize_uv_triangles(vertices, faces, uvs, texture_size)
    query = (positions[mask] - np.asarray(origin)) / voxel_size
    sampled, valid = sample_sparse_trilinear(voxel_coords, np.asarray(voxel_attrs), query)
    hit_rate = float(valid.mean()) if len(valid) else 0.0
    print(f"  Texels: {len(valid):,}, with voxel data: {hit_rate:.2%}, {time.time() - t0:.1f}s")

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
