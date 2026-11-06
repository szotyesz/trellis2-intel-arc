# XPU measurements

Run 2026-11-02 with `scripts/xpu_probe.py` in the `trellis` container: torch 2.15.0+xpu, compute-runtime 26.35, Arc Pro B60.

## Runtime

- `torch.xpu` initialises, with one device. `libsycl` loads from `/opt/venv`.
- `mem_get_info`: 23.91 GiB total (`clinfo` reports 22.71 GiB).
- `_fused_sdp_choice` returns 4 (not MATH) for bf16, fp16 and fp32.
- All smoke-tested ops ran with the CPU fallback disabled. A 6 GiB allocation works without relaxed limits.

## SDPA memory, `[1, 12, L, 128]`, torch peak in GiB

| L | bf16 | fp16 | fp32 |
|---|---|---|---|
| 4096 | 0.01 | 0.01 | 0.77 |
| 8192 | 0.02 | 0.02 | 3.05 |
| 16384 | 0.05 | 0.05 | 12.09 |
| 32768 | 0.09 | 0.09 | OOM, tried 48 GiB |
| 49152 | 0.14 | 0.14 | not run |

bf16 and fp16 scale with L, so no chunked attention is needed. **fp32 builds the L×L tensor, so do not fall back to fp32 flow models.**

Driver-level peaks are not useful here: the sampler stayed near 6.47 GiB after the 6 GiB allocation test.

Timings are not recorded. They imply an implausible throughput (0.26 s at L=49152), so `torch.xpu.synchronize()` may not have waited for the work. Measure real speed from pipeline runs.

## Accuracy against CPU

- SDPA at L=1024, maximum absolute error: bf16 0.029, fp16 0.0034, fp32 0.00001. No NaNs.
- `torch.mm`, 2^18 to 2^22 rows: fp32 error 0.0. fp16 error stays about 0.027 at every size, so no corruption appears at large row counts.

## Pipeline runs

Measured 2026-11-03 and 04 on one machine with 12 inputs, seed 42. Total is wall time for one image in a fresh container, with a 200,000-triangle target and a 1024 px texture. Figures cover successful runs from the matrix below, so higher levels look cheaper than they are.

| Type | Completed | Total, median (range), min | Generate, median, s | Peak VRAM, median, GiB | Raw triangles, median, M |
|---|---|---|---|---|---|
| `512` | 12/12 | 1.9 (1.3–2.9) | 29 | 2.8 | 3.3 |
| `1024` | 10/12 | 5.2 (1.7–14.1) | 92 | 8.7 | 12.1 |
| `1024_cascade` | 7/12 | 5.0 (2.1–7.9) | 81 | 7.9 | 10.1 |
| `1536_cascade` | 4/12 | 6.4 (3.3–10.6) | 176 | 12.0 | 16.1 |

This matrix ran **before** the GPU memory fixes below, so its completion counts are out of date. It needs repeating. Peak system RAM was about 17 GiB at every level.

Same-seed runs are not identical: one input gave 3.01M and 3.08M raw triangles in two runs.

## System RAM

RAM, not VRAM, first limited `1024_cascade`. All models stayed in RAM, reaching 23.8 GiB before baking added about 6 GiB, over the container's 26 GiB cap. Skipping unused models, dropping each after its last stage and calling `malloc_trim` cut the peak to about 16 GiB.

## Baking

The extracted mesh has many edges shared by three or more faces (221K of 4.3M on one input), and about 9% of manifold edges have inconsistent winding. These stalled the simplifier near 248K triangles and made xatlas take about 3 hours. Fixing winding and unwelding those faces before simplifying brings a 3M-triangle mesh to under a minute of unwrapping. A 12.7M-triangle mesh took 37 s to simplify and 247 s to unwrap.

xatlas still makes many small charts (about 32K on a 100K-face test), so UV seams are plentiful.

## GPU memory failures

All 15 GPU failures in the matrix happened while decoding. Of the 10 that still failed after the first fix, all were in the `conv_none` matmul. Successive fixes, each retested on inputs that still failed:

| Fix in `backends/conv_none.py` | Result |
|---|---|
| Sync the queue after each kernel slot | Recovered 5 of 15 |
| Chunk the matmul at 2M rows | Recovered 1 of the remaining 10; failures moved to ordinary tensor allocations |
| `empty_cache` before convolutions over 4M rows | Recovered 2 of 5 tested |

Not every run was retested with the final code, so only the repeat matrix will give the true completion counts. Known failures on the final code: `farmer` at `1536_cascade`, `fishbowl` at `1024_cascade`, `interior` at `1024`.

The remaining failures still show about 10–12 GiB reserved by the allocator but unused, with 8–10 GiB in use, so fragmentation inside a decoder pass looks like the main cause. The control run (`machine`, `1024_cascade`) kept its speed: 377 s against 386 s.

After a device error, Python can hang on exit, so `generate.py` now exits immediately on a GPU error.

## Colour

Output is often darker or less saturated than the input, and differs between levels. Findings so far:

- **Baking is not the cause by inspection.** The channel layout, value scaling and inpainting match upstream's `to_glb`, with no gamma step in either.
- **XPU sampling is not the cause.** From identical inputs and noise at 512, XPU and CPU texture latents differ by a mean of 0.0064 against a latent standard deviation of 2.66 (0.24%).
- **Not yet tested:** decoder output on XPU against CPU (the CPU decode ran out of RAM in the test container), `conv_none` against upstream's FlexGEMM, and DINOv3 features on XPU against CPU.
- **Possibly expected:** the model predicts PBR base colour, which removes lighting, so some darkening may be normal. A reference run of upstream on CUDA would settle this.
