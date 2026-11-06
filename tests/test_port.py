import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

os.environ.update(ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa", SPARSE_CONV_BACKEND="none")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "TRELLIS.2")]
BAKER = os.environ.get("BAKER_PATH", str(ROOT / "backends/texture_baker.py"))


def load_baker():
    spec = importlib.util.spec_from_file_location("baker_under_test", BAKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def color_field(p):
    """Smooth, asymmetric colours in [0, 1] for positions in [-0.5, 0.5]."""
    return np.stack([p[:, 0] + 0.5, p[:, 1] + 0.5, 0.25 + 0.5 * (p[:, 2] + 0.5)], axis=1)


def test_glb_round_trip(tmp_path):
    """Catches V flip, axis and gamma errors together."""
    baker = load_baker()
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    res = 128
    voxel = 1 / res
    g = np.arange(res)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    centers = (grid + 0.5) * voxel - 0.5
    r = np.linalg.norm(centers, axis=1)
    coords = grid[np.abs(r - 0.4) < 3 * voxel]
    p = (coords + 0.5) * voxel - 0.5
    attrs = np.concatenate([color_field(p), np.full((len(p), 1), 0.2), np.full((len(p), 1), 0.7), np.ones((len(p), 1))], axis=1).astype(np.float32)

    v, f, uv, _ = baker.uv_unwrap(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    out = baker.bake_texture(v, f, uv, coords, attrs, np.array([-0.5] * 3), voxel, texture_size=512)
    path = tmp_path / "t.glb"
    baker.export_glb_with_texture(v, f, uv, out[0], out[1], str(path))

    loaded = trimesh.load(path, force="mesh", process=False)
    tex = np.asarray(loaded.visual.material.baseColorTexture.convert("RGB")) / 255.0
    lv = np.asarray(loaded.vertices)
    original = np.stack([lv[:, 0], -lv[:, 2], lv[:, 1]], axis=1)  # undo Y-up
    luv = np.asarray(loaded.visual.uv)
    rows = np.clip(((1 - luv[:, 1]) * 511).round().astype(int), 0, 511)
    cols = np.clip((luv[:, 0] * 511).round().astype(int), 0, 511)
    assert np.mean(np.abs(tex[rows, cols] - color_field(original))) < 0.03
    # Y-up: the sphere's own axes cannot show this, so check the transform directly.
    assert np.allclose(np.sort(lv[:, 1])[-1], np.asarray(v)[:, 2].max(), atol=1e-5)


def brute_force_conv(feats, coords, weight, bias, dilation):
    Co, Kd, Kh, Kw, Ci = weight.shape
    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    out = torch.zeros(len(coords), Co)
    for i, (b, x, y, z) in enumerate(coords.tolist()):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    j = index.get((b, x + (kd - Kd // 2) * dilation[0], y + (kh - Kh // 2) * dilation[1], z + (kw - Kw // 2) * dilation[2]))
                    if j is not None:
                        out[i] += weight[:, kd, kh, kw, :] @ feats[j]
    return out + bias


def test_conv_matches_brute_force():
    from trellis2.modules import sparse as sp

    torch.manual_seed(0)
    for dilation in (1, 2):
        coords = torch.randint(0, 12, (400, 3)).unique(dim=0)
        coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], dim=1).int()
        conv = sp.SparseConv3d(5, 7, 3, dilation=dilation)
        feats = torch.randn(len(coords), 5)
        with torch.no_grad():
            out = conv(sp.SparseTensor(feats=feats, coords=coords)).feats
            ref = brute_force_conv(feats, coords, conv.weight, conv.bias, (dilation,) * 3)
        assert torch.allclose(out, ref, atol=1e-4)


def test_mesh_extract_matches_dict():
    from backends.mesh_extract import flexible_dual_grid_to_mesh

    torch.manual_seed(0)
    coords = torch.randint(0, 16, (1500, 3)).unique(dim=0).int()
    flags = torch.rand(len(coords), 3) > 0.3
    dual = torch.rand(len(coords), 3)
    _, faces = flexible_dual_grid_to_mesh(coords, dual, flags, None, [[-0.5] * 3, [0.5] * 3], grid_size=16)

    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    offsets = [[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]]
    quads = 0
    for i, c in enumerate(coords.tolist()):
        for axis in range(3):
            if flags[i, axis] and all(tuple(a + b for a, b in zip(c, o)) in index for o in offsets[axis]):
                quads += 1
    assert len(faces) == 2 * quads and quads > 0


def test_rmbg_never_loads():
    src = (ROOT / "TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py").read_text()
    assert "pipeline.rembg_model = None" in src
    assert "self.rembg_model(input)" not in src
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline as P

    p = P.__new__(P)
    p.low_vram = True
    try:
        p.preprocess_image(Image.new("RGB", (8, 8)))
    except ValueError as e:
        assert "transparency" in str(e)
    else:
        raise AssertionError("opaque input must be rejected")


def test_token_not_in_image():
    assert not (ROOT / ".secrets").exists()


def test_orient_faces_repairs_flipped_cube():
    baker = load_baker()
    box = trimesh.creation.box()
    faces = np.array(box.faces)
    flipped = faces.copy()
    flipped[::3] = flipped[::3][:, [0, 2, 1]]
    fixed = baker.orient_faces(np.array(box.vertices), flipped)
    assert trimesh.Trimesh(box.vertices, fixed, process=False).is_winding_consistent
    assert trimesh.Trimesh(box.vertices, fixed, process=False).volume > 0


def _edge_counts(f):
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_split_separates_merged_sheets_but_keeps_them_connected():
    baker = load_baker()
    # Two sheets, each a closed ring of four triangles around vertex 1, merged
    # along the edge 1-2, which four faces share. Sheet 1 lies in the plane
    # z = 0 and sheet 2 in y = 0, so each sheet continues straight through
    # the merged edge and pairing keeps its two faces there together.
    v = np.array([[0, -1, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [2, 0, 1], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    f = np.array([[1, 0, 2], [1, 2, 3], [1, 3, 4], [1, 4, 0],      # sheet 1
                  [1, 2, 5], [1, 5, 6], [1, 6, 7], [1, 7, 2]])     # sheet 2
    nv, nf = baker.split_non_manifold_vertices(v, f, pair_sheets=True)
    np.testing.assert_array_equal(nv[nf], v[f])      # geometry unchanged
    assert _edge_counts(nf).max() == 2
    assert len(set(nf[:4, 0])) == 1 and len(set(nf[4:, 0])) == 1
    assert len(nv) == 10                             # vertices 1 and 2 each split in two
    assert not set(nf[:4].ravel()) & set(nf[4:].ravel())
    assert len(set(nf[0]) & set(nf[1])) == 2         # sheet 1 still shares the edge
    assert len(set(nf[4]) & set(nf[7])) == 2         # and so does sheet 2

    # Cutting only: vertex 2 reaches the other faces only through the merged
    # edge, so it splits four ways.
    nv, nf = baker.split_non_manifold_vertices(v, f, pair_sheets=False)
    assert len(nv) == 12
    assert _edge_counts(nf).max() == 2


def test_split_keeps_a_manifold_mesh_unchanged():
    baker = load_baker()
    sphere = trimesh.creation.icosphere(subdivisions=2)
    nv, nf = baker.split_non_manifold_vertices(np.array(sphere.vertices), np.array(sphere.faces))
    assert len(nv) == len(sphere.vertices)
    assert trimesh.Trimesh(nv, nf, process=False).is_watertight


def test_cluster_vertices_merges_cells_and_drops_degenerate_faces():
    baker = load_baker()
    sphere = trimesh.creation.icosphere(subdivisions=4)   # 5,120 faces
    v, f = baker.cluster_vertices(np.array(sphere.vertices), np.array(sphere.faces), cell=0.2)
    assert 0 < len(f) < len(sphere.faces) / 4
    assert (f[:, 0] != f[:, 1]).all() and (f[:, 1] != f[:, 2]).all() and (f[:, 0] != f[:, 2]).all()
    assert len(np.unique(np.sort(f, axis=1), axis=0)) == len(f)
    assert np.allclose(v.min(axis=0), sphere.vertices.min(axis=0), atol=0.2)
    assert np.allclose(v.max(axis=0), sphere.vertices.max(axis=0), atol=0.2)


def test_simplify_mesh_reaches_target_through_clustering_pass():
    baker = load_baker()
    sphere = trimesh.creation.icosphere(subdivisions=5)   # 20,480 faces
    v, f = baker.simplify_mesh(np.array(sphere.vertices), np.array(sphere.faces), 500, max_raw_faces=5000)
    assert len(f) <= 650
    out = trimesh.Trimesh(v, f, process=False)
    assert abs(out.volume - sphere.volume) < 0.25 * sphere.volume
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    assert np.unique(edges, axis=0, return_counts=True)[1].max() <= 2


def test_nearest_voxel_fills_texels_off_the_shell():
    baker = load_baker()
    coords = np.array([[5, 5, 5], [9, 5, 5]])
    attrs = np.array([[1.0, 0, 0, 0, 0, 1], [0, 0, 1.0, 0, 0, 1]], dtype=np.float32)
    query = np.array([[5.5, 5.5, 5.5], [7.2, 5.5, 5.5], [20.0, 20.0, 20.0]])
    sampled, valid = baker.sample_sparse_trilinear(coords, attrs, query)
    assert valid.tolist() == [True, False, False]
    near, found = baker.sample_nearest_voxel(coords, attrs, query, max_distance=3.0)
    assert found.tolist() == [True, True, False]
    assert near[1, 0] == 1.0          # closer to the red voxel at 5 than the blue one at 9


def test_sparse_conv_matches_flexgemm_layout():
    """Slot order and direction follow FlexGEMM's _compute_neighbor_cache_torch."""
    from trellis2.modules.sparse import SparseTensor
    from trellis2.modules.sparse.conv.conv_none import sparse_conv3d_forward

    torch.manual_seed(0)
    Co, K, Ci = 3, 3, 2
    coords = torch.unique(torch.randint(0, 6, (120, 3)), dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=coords.dtype), coords], dim=1).int()
    feats = torch.randn(len(coords), Ci)

    class Conv:
        weight = torch.randn(Co, K, K, K, Ci)
        bias = torch.randn(Co)
        dilation = (1, 2, 1)

    index = {tuple(c): i for i, c in enumerate(coords[:, 1:].tolist())}
    expected = torch.zeros(len(coords), Co)
    for i, (a, b, c) in enumerate(coords[:, 1:].tolist()):
        for ka in range(K):
            for kb in range(K):
                for kc in range(K):
                    n = index.get((a + (ka - 1) * 1, b + (kb - 1) * 2, c + (kc - 1) * 1))
                    if n is not None:
                        expected[i] += Conv.weight[:, ka, kb, kc] @ feats[n]
    expected += Conv.bias

    x = SparseTensor(feats=feats, coords=coords)
    out = sparse_conv3d_forward(Conv, x).feats
    assert torch.allclose(out, expected, atol=1e-5)
