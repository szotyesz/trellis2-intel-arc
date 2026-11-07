#!/usr/bin/env python3
"""Bake a raw mesh saved by `generate.py --save-raw` into a textured GLB.

Needs no GPU, so baking changes can be tried on a saved mesh without
regenerating it. Runs in the cpu-tests container:

    docker compose run --rm --no-deps -T -v "$PWD:/repo" --workdir /repo \\
      --entrypoint /opt/venv/bin/python cpu-tests scripts/bake_raw.py \\
      out/x.raw.npz out/x.glb --repair cut
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", help=".raw.npz written by generate.py --save-raw")
    parser.add_argument("output", help="Output GLB path")
    parser.add_argument("--texture-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--target-faces", type=int, default=200000)
    parser.add_argument("--repair", default="pair", choices=["cut", "pair"])
    args = parser.parse_args()

    import trimesh
    from backends.texture_baker import bake_texture, export_glb_with_texture, simplify_mesh, uv_unwrap

    d = np.load(args.raw)
    verts, faces = d["vertices"].astype(np.float64), d["faces"].astype(np.int64)
    voxel_size = float(d["voxel_size"])
    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} triangles")
    t0 = time.time()
    bake_verts, bake_faces = simplify_mesh(verts, faces, args.target_faces, voxel_size=voxel_size,
                                           repair=args.repair)
    del verts, faces
    print(f"Simplified to {len(bake_faces):,} triangles in {time.time() - t0:.0f}s")
    normals = trimesh.Trimesh(bake_verts, bake_faces, process=False).vertex_normals
    t0 = time.time()
    new_verts, new_faces, uvs, vmapping = uv_unwrap(bake_verts, bake_faces)
    print(f"Unwrapped in {time.time() - t0:.0f}s")
    t0 = time.time()
    base_color, mr, _ = bake_texture(new_verts, new_faces, uvs, d["coords"], d["attrs"], d["origin"], voxel_size,
                                     texture_size=args.texture_size)
    export_glb_with_texture(new_verts, new_faces, uvs, base_color, mr, args.output, normals=normals[vmapping])
    print(f"Baked in {time.time() - t0:.0f}s: {args.output}")


if __name__ == "__main__":
    main()
