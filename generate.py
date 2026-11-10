"""
Generate textured 3D meshes from RGBA images using TRELLIS.2 on an Intel GPU.

Several images, or several samples of one, run in one process: the models
load once and each stage runs for every job before the next stage starts.
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
# The decoders allocate a few multi-GiB tensors on top of many small ones.
# Growing segments in place instead of reserving new ones stops the allocator
# from failing with gigabytes reserved but unusable; the two inputs that ran
# out of memory at 1024 and 1536 complete with this, at 19.6 and 20.2 GiB.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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


def log_stages(pipeline, names):
    """Print time, peak RAM and peak VRAM for every call of the named stages."""
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
            free_ram()
            print(
                f"[mem] {name}: {time.time() - t0:.0f}s, RAM peak {peak[0]:.1f} now {rss_gib():.1f} GiB, "
                f"VRAM peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )
            return result

        setattr(pipeline, name, wrapped)

    for name in names:
        wrap(name)


def release(pipeline, *models):
    """Drop models no later stage needs. System RAM, not VRAM, is what runs out here."""
    for model in models:
        pipeline.models.pop(model, None)
    free_ram()


def exit_after_gpu_error():
    print(FAILURE_HELP, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    # A normal exit can hang in the Level Zero runtime after a device error.
    os._exit(1)


class Job:
    """One mesh to generate: an image, a seed and where the result goes."""

    def __init__(self, image, seed, output):
        self.image, self.seed, self.output = image, seed, output
        self.failed = None      # the stage and error that ended this job
        self.gpu_error = False
        self.rng = None         # CPU generator state, carried from stage to stage

    def __str__(self):
        return f"{os.path.basename(self.image)}, seed {self.seed}"


def plan_jobs(images, output, output_dir, seed, num_samples):
    """One job per image and sample. Sample i of an image uses seed + i."""
    if output and len(images) > 1:
        raise SystemExit("--output names one result; use --output-dir with several images")
    jobs = []
    for image in images:
        stem = output or os.path.join(output_dir, os.path.splitext(os.path.basename(image))[0])
        for i in range(num_samples):
            jobs.append(Job(image, seed + i, stem if num_samples == 1 else f"{stem}_s{seed + i}"))
    taken = {}
    for job in jobs:
        if taken.setdefault(job.output, job.image) != job.image:
            raise SystemExit(f"{taken[job.output]} and {job.image} would both write {job.output}.glb")
    return jobs


def for_each(jobs, stage, fn):
    """Run one stage for every job still alive, with each job's own noise.

    A job that fails is dropped and the others carry on, so one image that
    runs out of memory does not cost the rest of the batch.
    """
    for job in jobs:
        if job.failed:
            continue
        if len(jobs) > 1:
            print(f"--- {stage}: {job}", flush=True)
        if job.rng is None:
            torch.manual_seed(job.seed)
        else:
            torch.set_rng_state(job.rng)
        try:
            fn(job)
        except Exception as e:
            traceback.print_exc()
            job.failed = f"{stage}: {type(e).__name__}"
            job.gpu_error = isinstance(e, RuntimeError)
            free_ram()
        job.rng = torch.get_rng_state()


def sample_latents(pipeline, jobs, pipeline_type, max_tokens, overrides):
    """The sampling stages of upstream's run(), one stage at a time across jobs.

    Each job draws its noise in the order run() does, from a generator
    seeded with the job's seed, so a job gives the same latents here as
    alone. Sets job.shape_slat, job.tex_slat and job.res.
    """
    models = pipeline.models
    conds = {}

    def cond(job):
        if job.image not in conds:
            image = pipeline.preprocess_image(Image.open(job.image))
            conds[job.image] = (pipeline.get_cond([image], 512),
                                pipeline.get_cond([image], 1024) if pipeline_type != "512" else None)
        job.cond_512, job.cond_1024 = conds[job.image]

    for_each(jobs, "conditioning", cond)
    pipeline.image_cond_model = None
    conds.clear()
    free_ram()

    ss_res = 64 if pipeline_type == "1024" else 32

    def structure(job):
        job.coords = pipeline.sample_sparse_structure(job.cond_512, ss_res, 1, overrides)

    for_each(jobs, "sparse structure", structure)
    release(pipeline, "sparse_structure_flow_model", "sparse_structure_decoder")

    def shape(job):
        if pipeline_type == "512":
            job.shape_slat = pipeline.sample_shape_slat(job.cond_512, models["shape_slat_flow_model_512"], job.coords, overrides)
            job.res = 512
        elif pipeline_type == "1024":
            job.shape_slat = pipeline.sample_shape_slat(job.cond_1024, models["shape_slat_flow_model_1024"], job.coords, overrides)
            job.res = 1024
        else:
            job.shape_slat, job.res = pipeline.sample_shape_slat_cascade(
                job.cond_512, job.cond_1024, models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
                512, 1024 if pipeline_type == "1024_cascade" else 1536, job.coords, overrides, max_tokens)
        del job.coords

    for_each(jobs, "shape", shape)
    release(pipeline, "shape_slat_flow_model_512", "shape_slat_flow_model_1024")

    def texture(job):
        size = "512" if pipeline_type == "512" else "1024"
        job.tex_slat = pipeline.sample_tex_slat(job.cond_512 if size == "512" else job.cond_1024,
                                                models[f"tex_slat_flow_model_{size}"], job.shape_slat, overrides)
        del job.cond_512, job.cond_1024

    for_each(jobs, "texture", texture)
    release(pipeline, "tex_slat_flow_model_512", "tex_slat_flow_model_1024")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", metavar="image", help="Input image with a transparent background; several run as a batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Meshes per image. Sample i uses seed + i and is written with a _s<seed> suffix")
    parser.add_argument("--output", default=None, help="Output path without extension, for one image (default: /out/output_3d)")
    parser.add_argument("--output-dir", default="/out", help="With several images: each is written here under its own name")
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--texture-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--no-texture", action="store_true", help="Export the raw mesh as OBJ only")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also write the raw mesh as OBJ (hundreds of MB) and a .raw.npz for scripts/bake_raw.py")
    parser.add_argument("--target-faces", type=int, default=200000, help="Triangles in the final GLB")
    parser.add_argument("--repair", default="pair", choices=["cut", "pair"],
                        help="Non-manifold edge repair: 'pair' keeps sheets connected, which halves the cracks "
                             "simplification opens but makes the UV unwrap take 2-20 minutes; "
                             "'cut' splits every such edge and unwraps in seconds")
    parser.add_argument("--alpha-mode", default="auto", choices=["auto", "opaque", "blend", "mask"],
                        help="How the GLB uses the alpha the model predicts: 'blend' draws transparent faces "
                             "blended, 'mask' cuts them out, 'opaque' stores alpha but ignores it as upstream "
                             "does, 'auto' blends only when part of the surface is transparent")
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps for all phases")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if len(args.images) == 1 and not args.output:
        args.output = "/out/output_3d"
    jobs = plan_jobs(args.images, args.output, args.output_dir, args.seed, args.num_samples)
    for image in args.images:
        if not os.path.isfile(image):
            parser.error(f"no such image: {image}")

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
    log_stages(pipeline, ("get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade",
                          "sample_tex_slat", "decode_shape_slat", "decode_tex_slat"))

    sys.path.insert(0, ROOT)
    from backends.texture_baker import bake_to_glb

    t0 = time.time()
    with torch.no_grad():
        sample_latents(pipeline, jobs, args.pipeline_type, args.max_tokens, {"steps": args.steps} if args.steps else {})
    sampled = time.time() - t0
    torch.xpu.empty_cache()

    def decode(job):
        """Decode on the GPU and move what baking needs to the CPU."""
        t0 = time.time()
        with torch.no_grad():
            mesh = pipeline.decode_latent(job.shape_slat, job.tex_slat, job.res)[0]
        del job.shape_slat, job.tex_slat
        raw = dict(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy(),
                   coords=mesh.coords.cpu().numpy(), attrs=mesh.attrs.cpu().float().numpy(),
                   origin=mesh.origin.cpu().numpy(), voxel_size=float(mesh.voxel_size))
        # With one job this is the time of the whole GPU phase, as before.
        print(f"Generated in {time.time() - t0 + sampled / len(jobs):.1f}s, "
              f"peak VRAM {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
        print(f"Mesh: {len(raw['vertices']):,} vertices, {len(raw['faces']):,} triangles")
        return raw

    def finish(job):
        raw = decode(job)
        if job is alive[-1]:
            # Baking runs on the CPU: release the decoders before the last one.
            release(pipeline, "shape_slat_decoder", "tex_slat_decoder")
        else:
            free_ram()
        os.makedirs(os.path.dirname(os.path.abspath(job.output)), exist_ok=True)
        if args.no_texture or args.save_raw:
            with open(f"{job.output}.obj", "w") as f:
                np.savetxt(f, raw["vertices"], fmt="v %.6f %.6f %.6f")
                np.savetxt(f, raw["faces"] + 1, fmt="f %d %d %d")
            # Everything scripts/bake_raw.py needs to redo the baking without a GPU.
            np.savez(f"{job.output}.raw.npz", **{**raw, "vertices": raw["vertices"].astype(np.float32),
                                                 "faces": raw["faces"].astype(np.int32),
                                                 "voxel_size": np.float32(raw["voxel_size"])})
        if args.no_texture:
            return
        print(f"RAM before baking: {rss_gib():.1f} GiB")
        t0 = time.time()
        mode = bake_to_glb(raw.pop("vertices"), raw.pop("faces"), **raw, output_path=f"{job.output}.glb",
                           target_faces=args.target_faces, texture_size=args.texture_size, repair=args.repair,
                           alpha_mode=args.alpha_mode)
        print(f"Baked in {time.time() - t0:.0f}s: {job.output}.glb, alpha {mode}, RAM {rss_gib():.1f} GiB")

    alive = [job for job in jobs if not job.failed]
    for_each(jobs, "decode and bake", finish)

    failed = [job for job in jobs if job.failed]
    if len(jobs) > 1:
        print(f"Done: {len(jobs) - len(failed)} of {len(jobs)} succeeded")
    for job in failed:
        print(f"FAILED {job}: {job.failed}", file=sys.stderr)
    if any(job.gpu_error for job in failed):
        exit_after_gpu_error()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
