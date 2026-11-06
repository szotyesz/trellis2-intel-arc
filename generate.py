"""
Generate a textured 3D mesh from an RGBA image using TRELLIS.2 on an Intel GPU.
"""

import argparse
import ctypes
import gc
import os
import sys
import threading
import time
import traceback

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


def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def free_ram():
    """Run the GC and hand freed heap pages back to the OS."""
    gc.collect()
    torch.xpu.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


# Models no later stage needs. System RAM, not VRAM, is what runs out here.
RELEASE_AFTER = {
    "sample_sparse_structure": ("sparse_structure_flow_model", "sparse_structure_decoder"),
    "sample_shape_slat": ("shape_slat_flow_model_512", "shape_slat_flow_model_1024"),
    "sample_shape_slat_cascade": ("shape_slat_flow_model_512", "shape_slat_flow_model_1024"),
    "sample_tex_slat": ("tex_slat_flow_model_512", "tex_slat_flow_model_1024"),
    "decode_shape_slat": ("shape_slat_decoder",),
    "decode_tex_slat": ("tex_slat_decoder",),
}


def instrument_stages(pipeline):
    """Print peak RAM and VRAM per pipeline stage, and release finished models."""
    peak = [0.0]

    def sample():
        while True:
            peak[0] = max(peak[0], rss_gib())
            time.sleep(0.2)

    threading.Thread(target=sample, daemon=True).start()

    def wrap(name):
        fn = getattr(pipeline, name)

        def wrapped(*a, **k):
            peak[0] = rss_gib()
            torch.xpu.reset_peak_memory_stats()
            t0 = time.time()
            result = fn(*a, **k)
            if name == "sample_sparse_structure":
                pipeline.image_cond_model = None
            for model in RELEASE_AFTER.get(name, ()):
                pipeline.models.pop(model, None)
            free_ram()
            print(
                f"[mem] {name}: {time.time() - t0:.0f}s, RAM peak {peak[0]:.1f} now {rss_gib():.1f} GiB, "
                f"VRAM peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )
            return result

        setattr(pipeline, name, wrapped)

    for name in ("get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade",
                 "sample_tex_slat", "decode_shape_slat", "decode_tex_slat"):
        wrap(name)


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
    parser.add_argument("--repair", default="cut", choices=["cut", "pair"],
                        help="Non-manifold edge repair: 'pair' keeps sheets connected and makes fewer UV charts, "
                             "but UV unwrapping then takes 6-11 minutes instead of under one")
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps for all phases")
    args = parser.parse_args()

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    # Skip flow models this level never uses: each one costs system RAM.
    unused = {
        "512": ("_1024",),
        "1024": ("_512",),
        "1024_cascade": ("tex_slat_flow_model_512",),
        "1536_cascade": ("tex_slat_flow_model_512",),
    }[args.pipeline_type]
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        n for n in Trellis2ImageTo3DPipeline.model_names_to_load if not n.endswith(unused)
    ]

    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to(torch.device("xpu"))
    print(f"Loaded in {time.time() - t0:.0f}s on {torch.xpu.get_device_name(0)}, RAM {rss_gib():.1f} GiB")
    instrument_stages(pipeline)

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
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # A normal exit can hang in the Level Zero runtime after a device error.
        os._exit(1)
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
    free_ram()
    print(f"RAM before baking: {rss_gib():.1f} GiB")

    t0 = time.time()
    bake_verts, bake_faces = simplify_mesh(verts, faces, args.target_faces, voxel_size=voxel_size,
                                           pair_sheets=args.repair == "pair")
    del verts, faces
    free_ram()
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
    print(f"Baked in {time.time() - t0:.0f}s: {args.output}.glb, RAM {rss_gib():.1f} GiB")


if __name__ == "__main__":
    main()
