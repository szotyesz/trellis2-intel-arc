"""
Generate a textured 3D mesh from an RGBA image using TRELLIS.2 on an Intel GPU.
"""

import argparse
import gc
import os
import sys
import time

# PYTORCH_ENABLE_XPU_FALLBACK stays unset, so a missing op raises an error
# instead of silently running on the CPU.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "TRELLIS.2"))
sys.path.append(os.path.join(ROOT, "stubs"))

import numpy as np
import torch
from PIL import Image

FAILURE_HELP = """
The GPU run failed. Common causes:
  - Out of device memory: try a lower --pipeline-type, and stop anything
    else that is using the GPU.
  - Device lost: check `journalctl -k | grep xe` on the host.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Input image with a transparent background")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/out/output_3d", help="Output path without extension")
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--texture-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--no-texture", action="store_true", help="Export the raw mesh as OBJ only")
    parser.add_argument("--save-raw", action="store_true", help="Also write the raw mesh as OBJ (hundreds of MB)")
    parser.add_argument("--target-faces", type=int, default=200000, help="Triangles in the final GLB")
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps for all phases")
    args = parser.parse_args()

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    if args.pipeline_type == "512":
        # The 1024 flow models would only take up RAM.
        Trellis2ImageTo3DPipeline.model_names_to_load = [
            n for n in Trellis2ImageTo3DPipeline.model_names_to_load if not n.endswith("_1024")
        ]

    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to(torch.device("xpu"))
    print(f"Loaded in {time.time() - t0:.0f}s on {torch.xpu.get_device_name(0)}")

    overrides = {"steps": args.steps} if args.steps else {}
    t0 = time.time()
    try:
        mesh = pipeline.run(
            Image.open(args.image),
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            max_num_tokens=args.max_tokens,
            sparse_structure_sampler_params=overrides,
            shape_slat_sampler_params=overrides,
            tex_slat_sampler_params=overrides,
        )[0]
    except RuntimeError:
        print(FAILURE_HELP, file=sys.stderr)
        raise
    print(f"Generated in {time.time() - t0:.1f}s, peak VRAM {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")

    verts = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    if args.no_texture or args.save_raw:
        with open(f"{args.output}.obj", "w") as f:
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, faces + 1, fmt="f %d %d %d")

    if args.no_texture:
        return

    import trimesh

    sys.path.insert(0, ROOT)
    from backends.texture_baker import bake_texture, export_glb_with_texture, simplify_mesh, uv_unwrap

    # Baking runs on the CPU: release the models and the raw GPU mesh first.
    coords = mesh.coords.cpu().numpy()
    attrs = mesh.attrs.cpu().float().numpy()
    origin = mesh.origin.cpu().numpy()
    voxel_size = mesh.voxel_size
    del mesh, pipeline
    gc.collect()
    torch.xpu.empty_cache()

    t0 = time.time()
    bake_verts, bake_faces = simplify_mesh(verts, faces, args.target_faces)
    del verts, faces
    gc.collect()
    print(f"Simplified to {len(bake_faces):,} triangles in {time.time() - t0:.0f}s")
    normals = trimesh.Trimesh(bake_verts, bake_faces, process=False).vertex_normals
    t0 = time.time()
    new_verts, new_faces, uvs, vmapping = uv_unwrap(bake_verts, bake_faces)
    print(f"Unwrapped in {time.time() - t0:.0f}s")
    t0 = time.time()
    base_color, mr, _ = bake_texture(
        new_verts, new_faces, uvs, coords, attrs, origin, voxel_size,
        texture_size=args.texture_size,
    )
    export_glb_with_texture(new_verts, new_faces, uvs, base_color, mr, f"{args.output}.glb", normals=normals[vmapping])
    print(f"Baked in {time.time() - t0:.0f}s: {args.output}.glb")


if __name__ == "__main__":
    main()
