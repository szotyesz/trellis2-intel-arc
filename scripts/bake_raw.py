#!/usr/bin/env python3
"""Bake a raw mesh saved by `generate.py --save-raw` or `--raw-only` into a textured GLB.

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

from backends.texture_baker import ALPHA_MODES, bake_to_glb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", help=".raw.npz written by generate.py --save-raw or --raw-only")
    parser.add_argument("output", help="Output GLB path")
    parser.add_argument("--texture-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--target-faces", type=int, default=200000)
    parser.add_argument("--repair", default="pair", choices=["cut", "pair"])
    parser.add_argument("--alpha-mode", default="auto", choices=ALPHA_MODES)
    args = parser.parse_args()

    d = np.load(args.raw)
    print(f"Mesh: {len(d['vertices']):,} vertices, {len(d['faces']):,} triangles")
    t0 = time.time()
    mode = bake_to_glb(d["vertices"].astype(np.float64), d["faces"].astype(np.int64), d["coords"], d["attrs"],
                       d["origin"], float(d["voxel_size"]), args.output, target_faces=args.target_faces,
                       texture_size=args.texture_size, repair=args.repair, alpha_mode=args.alpha_mode)
    print(f"Baked in {time.time() - t0:.0f}s: {args.output}, alpha {mode}")


if __name__ == "__main__":
    main()
