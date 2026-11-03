"""Render front, side and three-quarter views of a GLB to a PNG, without a GPU."""
import sys

import numpy as np
import trimesh
from PIL import Image

mesh = trimesh.load(sys.argv[1], force="mesh", process=False)
tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
H, W = tex.shape[:2]
uv = np.asarray(mesh.visual.uv)
S = 512
panels = []
for yaw in (0, 90, 35):
    a = np.radians(yaw)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = np.asarray(mesh.vertices) @ R.T
    n = np.asarray(mesh.vertex_normals) @ R.T
    x = ((p[:, 0] + 0.55) / 1.1 * S).astype(int).clip(0, S - 1)
    y = ((0.55 - p[:, 1]) / 1.1 * S).astype(int).clip(0, S - 1)
    # Splat dense surface samples, nearest to the camera last.
    bary = np.random.default_rng(0).dirichlet((1, 1, 1), (len(mesh.faces), 8))
    f = np.asarray(mesh.faces)
    pts = (p[f][:, None] * bary[..., None]).sum(2).reshape(-1, 3)
    uvs = (uv[f][:, None] * bary[..., None]).sum(2).reshape(-1, 2)
    shade = (n[f][:, None] * bary[..., None]).sum(2).reshape(-1, 3)[:, 2]
    order = np.argsort(pts[:, 2])
    pts, uvs, shade = pts[order], uvs[order], shade[order]
    img = np.full((S, S, 3), 255, np.uint8)
    px = ((pts[:, 0] + 0.55) / 1.1 * S).astype(int).clip(0, S - 1)
    py = ((0.55 - pts[:, 1]) / 1.1 * S).astype(int).clip(0, S - 1)
    c = tex[((1 - uvs[:, 1]) * (H - 1)).astype(int).clip(0, H - 1), (uvs[:, 0] * (W - 1)).astype(int).clip(0, W - 1)]
    img[py, px] = (c * (0.55 + 0.45 * np.abs(shade[:, None]))).astype(np.uint8)
    panels.append(img)
Image.fromarray(np.hstack(panels)).save(sys.argv[2])
