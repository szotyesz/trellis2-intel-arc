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
