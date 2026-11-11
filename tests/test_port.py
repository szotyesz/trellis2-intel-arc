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


def test_repair_keeps_geometry_with_int32_faces_on_large_meshes():
    """The edge keys in orient_faces and split_non_manifold_vertices multiply two
    vertex ids. With int32 faces, which SciPy and the simplifier hand back, that
    overflowed on meshes above 46,341 vertices and stitched unrelated vertices
    into triangles spanning the model."""
    baker = load_baker()
    sphere = trimesh.creation.icosphere(subdivisions=7)   # 163,842 vertices
    v = np.asarray(sphere.vertices)
    f = np.asarray(sphere.faces, dtype=np.int32)
    for pair in (False, True):
        v2, f2 = baker.make_edge_manifold(v, f, pair)
        assert len(v2) == len(v)                      # a closed sphere needs no splits
        lengths = np.linalg.norm(v2[f2[:, 0]] - v2[f2[:, 1]], axis=1)
        assert lengths.max() < 3 * np.median(lengths)
        out = trimesh.Trimesh(v2, f2, process=False)
        assert abs(out.volume - sphere.volume) < 1e-6 * sphere.volume
        assert f2.dtype == np.int64


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


def _voxel_shell_of(mesh, pitch, samples=300_000):
    """Occupied voxels of a mesh surface, in TRELLIS's coords/origin convention."""
    points = trimesh.sample.sample_surface(mesh, samples)[0]
    origin = mesh.bounds[0] - pitch
    coords = np.unique(np.floor((points - origin) / pitch).astype(np.int64), axis=0)
    return coords, origin


def _heightfield(n, height):
    """A connected grid mesh over [0, 1]^2 with z = height(x, y)."""
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    v = np.stack([x.ravel(), y.ravel(), height(x, y).ravel()], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel(), i[:-1, 1:].ravel()
    return v, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def test_shell_decimation_keeps_faces_on_the_shell():
    """Towers with alleys between them next to a bumpy field. Plain quadric
    decimation spends the budget on the bumps and fans faces over the
    alleys; the shell-aware stage keeps the faces on the surface."""
    baker = load_baker()
    rng = np.random.default_rng(0)
    bumps = rng.uniform(0, 0.03, (40, 40))

    def height(x, y):
        tower = ((x * 8) % 2 < 1.2) & ((y * 8) % 2 < 1.2) & (x < 0.5)
        field = bumps[np.minimum((x * 40).astype(int), 39), np.minimum((y * 40).astype(int), 39)]
        return np.where(tower, 0.3, 0.0) + np.where(x >= 0.5, field, 0.0)

    v, f = _heightfield(160, height)
    pitch = 1 / 128
    coords, origin = _voxel_shell_of(trimesh.Trimesh(v, f, process=False), pitch)
    kw = dict(voxel_size=pitch, coords=coords, origin=origin, repair="cut", log=lambda *a: None)
    vs, fs = baker.simplify_mesh(v, f, 1500, **kw)
    vp, fp = baker.simplify_mesh(v, f, 1500, shell_decimation=False, **kw)
    off_shell = baker.off_surface_fraction(vs, fs, coords, origin, pitch)
    off_plain = baker.off_surface_fraction(vp, fp, coords, origin, pitch)
    assert len(fs) <= 1600 and len(fp) <= 1600
    assert off_plain > 0.05, off_plain
    assert off_shell < 0.01, off_shell


def test_shell_decimation_keeps_open_borders():
    """An open sheet keeps its outline: the border quadrics stop the free
    in-plane collapses along its edge from eroding it."""
    from backends.surface_decimate import VoxelShell, decimate_on_shell

    v, f = _heightfield(60, lambda x, y: np.zeros_like(x))
    coords, origin = _voxel_shell_of(trimesh.Trimesh(v, f, process=False), 1 / 32)
    nv, nf = decimate_on_shell(v, f, 200, VoxelShell(coords, origin, 1 / 32), log=lambda *a: None)
    assert len(nf) <= 210
    assert abs(trimesh.Trimesh(nv, nf, process=False).area - 1.0) < 0.02


def test_voxel_shell_near_uses_the_neighbouring_cells():
    from backends.surface_decimate import VoxelShell

    shell = VoxelShell(np.array([[2, 2, 2]]), origin=np.zeros(3), voxel_size=1.0)
    near = shell.near(np.array([[2.5, 2.5, 2.5], [1.2, 3.8, 2.5], [0.5, 2.5, 2.5], [2.5, 2.5, 4.5]]))
    np.testing.assert_array_equal(near, [True, True, False, False])


def test_sparse_attention_handles_several_samples():
    """Each sample attends only within itself, whatever the lengths."""
    import torch.nn.functional as F
    from trellis2.modules.sparse import VarLenTensor
    from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention as attend

    torch.manual_seed(0)
    lengths, heads, dim = [5, 9, 2], 2, 8
    layout = VarLenTensor.layout_from_seqlen(lengths)
    qkv = VarLenTensor(torch.randn(sum(lengths), 3, heads, dim), layout)

    def dense(q, k, v):
        return F.scaled_dot_product_attention(*(t.unsqueeze(0).transpose(1, 2) for t in (q, k, v))).transpose(1, 2)[0]

    out = attend(qkv).feats
    for s in layout:
        assert torch.allclose(out[s], dense(*qkv.feats[s].unbind(dim=1)), atol=1e-6)

    # Cross-attention to a dense context, as the flow models condition on the image.
    q = VarLenTensor(torch.randn(sum(lengths), heads, dim), layout)
    context = torch.randn(len(lengths), 7, 2, heads, dim)
    out = attend(q, context).feats
    for i, s in enumerate(layout):
        assert torch.allclose(out[s], dense(q.feats[s], *context[i].unbind(dim=1)), atol=1e-6)


def glass_and_solid(tmp_path, alpha_mode):
    """A sphere whose upper half has an alpha of 0.3, baked and exported."""
    baker = load_baker()
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    res = 64
    g = np.arange(res)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    centers = (grid + 0.5) / res - 0.5
    shell = np.abs(np.linalg.norm(centers, axis=1) - 0.4) < 3 / res
    coords, p = grid[shell], centers[shell]
    alpha = np.where(p[:, 2] > 0, 0.3, 1.0)[:, None]
    attrs = np.concatenate([color_field(p), np.full((len(p), 2), 0.5), alpha], axis=1).astype(np.float32)
    v, f, uv, _ = baker.uv_unwrap(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    bake = baker.bake_texture(v, f, uv, coords, attrs, np.array([-0.5] * 3), 1 / res, texture_size=256)
    path = tmp_path / f"{alpha_mode}.glb"
    used = baker.export_glb_with_texture(v, f, uv, bake.base_color, bake.metallic_roughness, str(path),
                                         alpha_img=bake.alpha, face_alpha=bake.face_alpha, alpha_mode=alpha_mode)
    return used, path, v, f, bake


def gltf_header(path):
    import json
    import struct

    data = Path(path).read_bytes()
    return json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])


def test_transparent_faces_get_a_blended_material(tmp_path):
    used, path, v, f, bake = glass_and_solid(tmp_path, "auto")
    assert used == "blend"
    upper = v[f].mean(axis=1)[:, 2] > 0.05
    assert (bake.face_alpha[upper] < 0.5).all() and (bake.face_alpha[v[f].mean(axis=1)[:, 2] < -0.05] > 0.95).all()

    header = gltf_header(path)
    assert sorted(m["alphaMode"] for m in header["materials"]) == ["BLEND", "OPAQUE"]
    # Both materials use the same two images, stored once.
    assert len({i["bufferView"] for i in header["images"]}) == 2
    scene = trimesh.load(path, process=False)
    parts = {name: g for name, g in scene.geometry.items()}
    assert sum(len(g.faces) for g in parts.values()) == len(f)
    blended = next(g for g in parts.values() if g.visual.material.alphaMode == "BLEND")
    assert abs(len(blended.faces) / len(f) - 0.5) < 0.1
    # Y-up after export: the transparent half is the top.
    assert blended.vertices[:, 1].min() > -0.1
    texture = np.asarray(blended.visual.material.baseColorTexture)
    assert texture.shape[2] == 4 and texture[..., 3].min() < 100 and texture[..., 3].max() == 255


def test_alpha_modes_follow_the_request(tmp_path):
    for mode, expected in (("opaque", ["OPAQUE"]), ("mask", ["MASK"])):
        used, path, *_ = glass_and_solid(tmp_path, mode)
        header = gltf_header(path)
        assert used == mode and [m["alphaMode"] for m in header["materials"]] == expected
    assert header["materials"][0]["alphaCutoff"] == 0.5


def test_opaque_surface_stays_opaque_in_auto(tmp_path):
    baker = load_baker()
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    assert baker.choose_alpha_mode("auto", v, f, np.array([1.0, 0.99, 0.95, 0.91])) == "opaque"
    assert baker.choose_alpha_mode("auto", v, f, np.array([1.0, 0.99, 0.95, 0.2])) == "blend"
    assert baker.choose_alpha_mode("auto", v, f, None) == "opaque"


def test_batch_jobs_get_distinct_outputs():
    import generate

    jobs = generate.plan_jobs(["/input/a.png", "/input/b.webp"], None, "/out", 7, 2)
    assert [(j.image, j.seed, j.output) for j in jobs] == [
        ("/input/a.png", 7, "/out/a_s7"), ("/input/a.png", 8, "/out/a_s8"),
        ("/input/b.webp", 7, "/out/b_s7"), ("/input/b.webp", 8, "/out/b_s8"),
    ]
    assert [j.output for j in generate.plan_jobs(["/input/a.png"], "/out/x", "/out", 42, 1)] == ["/out/x"]
    for bad in (dict(images=["/a/x.png", "/b/x.png"], output=None), dict(images=["/a/x.png", "/a/y.png"], output="/out/z")):
        try:
            generate.plan_jobs(bad["images"], bad["output"], "/out", 42, 1)
        except SystemExit:
            continue
        raise AssertionError("clashing outputs must be rejected")


def test_raw_only_writes_the_bake_input_and_nothing_else(tmp_path):
    """`--raw-only` leaves just the .raw.npz that scripts/bake_raw.py reads."""
    import generate

    raw = dict(vertices=np.random.rand(5, 3), faces=np.array([[0, 1, 2], [2, 3, 4]]),
               coords=np.zeros((2, 3), np.int32), attrs=np.ones((2, 6), np.float32),
               origin=np.zeros(3, np.float32), voxel_size=1 / 1024)
    generate.save_raw(raw, str(tmp_path / "a"), obj=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.raw.npz"]
    d = np.load(tmp_path / "a.raw.npz")
    assert d["vertices"].dtype == np.float32 and d["faces"].dtype == np.int32
    assert np.allclose(d["vertices"], raw["vertices"]) and (d["faces"] == raw["faces"]).all()
    assert float(d["voxel_size"]) == np.float32(1 / 1024)
    generate.save_raw(raw, str(tmp_path / "b"), obj=True)
    assert len((tmp_path / "b.obj").read_text().splitlines()) == 5 + 2
    assert (tmp_path / "b.raw.npz").exists() and not list(tmp_path.glob("*.tmp.npz"))


def test_jobs_keep_their_own_noise_across_stages():
    """Interleaved jobs draw the numbers they would draw alone."""
    import generate

    def alone(seed):
        torch.manual_seed(seed)
        return [torch.randn(3) for _ in range(2)]

    jobs = [generate.Job("a.png", 1, "a"), generate.Job("b.png", 2, "b")]
    drawn = {1: [], 2: []}
    for stage in ("first", "second"):
        generate.for_each(jobs, stage, lambda job: drawn[job.seed].append(torch.randn(3)))
    for seed in (1, 2):
        assert all(torch.equal(a, b) for a, b in zip(drawn[seed], alone(seed)))

    def boom(job):
        if job.seed == 1:
            raise ValueError("bad input")
        drawn[job.seed].append(None)

    generate.for_each(jobs, "third", boom)
    assert jobs[0].failed == "third: ValueError" and not jobs[0].gpu_error and jobs[1].failed is None
    generate.for_each(jobs, "fourth", lambda job: drawn[job.seed].append(None))
    assert len(drawn[1]) == 2 and len(drawn[2]) == 4


def test_texturing_keeps_usable_uvs_and_normalises_the_mesh():
    import texture

    v = np.array([[2.0, 10.0, -1.0], [4.0, 10.0, -1.0], [2.0, 14.0, 0.0]])
    unit = texture.normalise(v, z_up=False)
    assert np.abs(unit).max() <= 0.5 and np.isclose(np.ptp(unit, axis=0).max(), 0.99999)
    # glTF's up (Y) becomes Z, and its Z becomes -Y, as upstream's preprocess_mesh.
    assert unit[2, 2] > unit[0, 2] and np.isclose(unit[2, 1], -unit[2, 1].__abs__())
    assert np.allclose(texture.normalise(v, z_up=True), (v - [3, 12, -0.5]) * 0.99999 / 4)

    mesh = trimesh.Trimesh(v, [[0, 1, 2]], process=False)
    assert texture.usable_uv(mesh) is None
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [0, 0.25]])
    assert np.allclose(texture.usable_uv(mesh), [[0, 1], [1, 1], [0, 0.75]])
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [3, 0], [0, 1]])
    assert texture.usable_uv(mesh) is None
