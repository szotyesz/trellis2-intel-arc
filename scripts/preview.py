"""Render front, side and three-quarter views of a GLB to a PNG, without a GPU.

Blended materials are drawn over the opaque ones, so glass shows what is
inside it.
"""
import sys

import numpy as np
import trimesh
from PIL import Image

S = 512
LAYERS = 16     # depth slices the blended samples are composited in
SAMPLES = 3_000_000

loaded = trimesh.load(sys.argv[1], process=False)
parts = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
# Generated meshes sit in a unit cube and keep their size; anything larger is fitted to the frame.
lo = np.min([m.bounds[0] for m in parts], axis=0)
hi = np.max([m.bounds[1] for m in parts], axis=0)
fits = np.abs([lo, hi]).max() <= 0.55
CENTRE, SCALE = (0.0, 1.0) if fits else ((lo + hi) / 2, 1.0 / (hi - lo).max())


TOTAL_AREA = sum(m.area for m in parts)


def samples(mesh, R):
    """Surface samples, spread by area: position, RGBA colour and shading."""
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGBA"))
    H, W = tex.shape[:2]
    p = (np.asarray(mesh.vertices) - CENTRE) * SCALE @ R.T
    n = np.asarray(mesh.vertex_normals) @ R.T
    f = np.asarray(mesh.faces)
    f = np.repeat(f, np.maximum(np.rint(mesh.area_faces / TOTAL_AREA * SAMPLES).astype(int), 1), axis=0)
    bary = np.random.default_rng(0).dirichlet((1, 1, 1), len(f))[..., None]
    pts = (p[f] * bary).sum(1)
    uvs = (np.asarray(mesh.visual.uv)[f] * bary).sum(1)
    shade = (n[f] * bary).sum(1)[:, 2]
    c = tex[((1 - uvs[:, 1]) * (H - 1)).astype(int).clip(0, H - 1), (uvs[:, 0] * (W - 1)).astype(int).clip(0, W - 1)]
    rgb = c[:, :3] * (0.55 + 0.45 * np.abs(shade[:, None]))
    blended = mesh.visual.material.alphaMode == "BLEND"
    return pts, rgb, c[:, 3] / 255.0 if blended else np.ones(len(pts))


panels = []
for yaw in (0, 90, 35):
    a = np.radians(yaw)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pts, rgb, alpha = (np.concatenate(x) for x in zip(*(samples(mesh, R) for mesh in parts)))
    # Splat nearest to the camera last.
    order = np.argsort(pts[:, 2])
    pts, rgb, alpha = pts[order], rgb[order], alpha[order]
    px = ((pts[:, 0] + 0.55) / 1.1 * S).astype(int).clip(0, S - 1)
    py = ((0.55 - pts[:, 1]) / 1.1 * S).astype(int).clip(0, S - 1)
    img = np.full((S, S, 3), 255.0)
    depth = np.full((S, S), -np.inf)
    solid = alpha >= 1
    img[py[solid], px[solid]] = rgb[solid]
    depth[py[solid], px[solid]] = pts[solid, 2]
    # Blended samples in front of the opaque surface, far slices first.
    glass = np.flatnonzero(~solid & (pts[:, 2] > depth[py, px]))
    for chunk in np.array_split(glass, LAYERS):
        y, x, t = py[chunk], px[chunk], alpha[chunk, None]
        img[y, x] = rgb[chunk] * t + img[y, x] * (1 - t)
    panels.append(img.astype(np.uint8))
Image.fromarray(np.hstack(panels)).save(sys.argv[2])
