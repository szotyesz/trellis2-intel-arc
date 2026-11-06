# Port TRELLIS.2 to the Intel Arc Pro B60 (XPU)

**Status:** plan only, nothing implemented. Researched 2026-11-02. Licensing notes are engineering research, not legal advice.

**Sources read:** TRELLIS.2 `75fbf01`, FlexGEMM `6dd94a8`, torch `v2.15.0`, torch-xpu-ops `1dda266` (the commit torch 2.15.0 pins), oneDNN `main` (torch pins `v3.13.3`).

## Outcome

Inside a Docker container on the B60, `generate.py` turns an image with a transparent background into a textured GLB whose colours and orientation match upstream's `o_voxel.postprocess.to_glb`. The order is the 512 pipeline, then `1024_cascade`, then `1536_cascade` if the measurements allow it.

**Constraints from the project owner:**

- No component may prohibit commercial use of the generated output.
- This project itself can use any open-source licence. It keeps the MIT licence it inherited.
- Everything builds and runs in Docker, so the wheel's Intel runtime cannot conflict with the host.

## Recommendation

First results should need almost no kernel work. The XPU build already provides a fused attention kernel, and everything else can run through vectorised PyTorch or CPU libraries. SYCL rewrites are not justified by anything found so far.

| Upstream | Replacement | SYCL needed? |
|---|---|---|
| `flash_attn` | `sdpa`, which XPU routes to oneDNN's fused kernel | No, it already exists |
| `flex_gemm` sparse conv | `conv_none.py`, vectorised in Phase 4; Triton only if it passes the Phase 6 gate | No |
| `o_voxel` CUDA hashmap | `mesh_extract.py`, vectorised in Phase 4 | No |
| `flex_gemm` `grid_sample_3d` | Small sparse trilinear function (Phase 3) | No |
| `nvdiffrast` | Existing NumPy UV rasteriser in `texture_baker.py` | No |
| `cumesh` | `fast_simplification` and `xatlas`; hole filling is deferred to Phase 6 | No |
| RMBG-2.0 background removal | **Patched out.** Input is RGBA by default; optional BiRefNet in Phase 6E | No |

## Licensing

**The test:** does any component prohibit commercial use of the generated GLBs? Being non-MIT is not a failure. Copyleft and proprietary terms pass unless they limit commercial use or the output.

| Component | Licence | Blocks commercial use of output? | In the plan |
|---|---|---|---|
| TRELLIS.2 code and weights, `TRELLIS-image-large` weights | MIT | No | Used |
| DINOv3 weights | Custom DINOv3 License | No, but it has end-use limits (below) | Used |
| Python packages (torch, transformers, trimesh, xatlas, OpenCV, `easydict`, ...) | MIT, BSD, Apache-2.0, MPL-2.0, LGPL | No | Used |
| GPU stack in the image: compute-runtime, IGC, gmmlib, Level Zero | MIT | No | Used |
| 21 Intel runtime packages (`intel-sycl-rt`, `mkl`, ...) | Intel EULA and Simplified Software License | No, but it limits hosted services (below) | Used |
| FlexGEMM | MIT | No | Phase 6A only |
| BiRefNet | MIT weights | Not by its licence, but see its training data (below) | **Off by default,** Phase 6E |
| **RMBG-2.0** | CC BY-NC 4.0 | **Yes** | **Out** |
| **`nvdiffrast`, `nvdiffrec`** | NVIDIA source licence | **Yes** | **Out** |

### The two that fail

**RMBG-2.0.** Its card says: *"The model is released under a Creative Commons license for non-commercial use. Commercial use is subject to a commercial agreement with BRIA."* Upstream's `pipeline.json` selects it and `from_pretrained` loads it on every run, even for RGBA input, so it must be patched out (Phase 2).

**`nvdiffrast` and `nvdiffrec`.** Section 3.3 of `nvdiffrast`'s licence says the work *"only may be used or intended for use non-commercially"*, meaning *"for research or evaluation purposes only and not for any direct or indirect monetary gain"*. `nvdiffrec` has the same clause.

This matters more than it first looks. Upstream's own `o_voxel.postprocess.to_glb` imports `nvdiffrast` at the top of the file, so **upstream's standard CUDA export path would fail this test.** This port's own baker is what keeps the output clean. Do not port `nvdiffrast` to XPU later as an improvement, and do not import `trellis2/renderers`, `trellis2_texturing.py` or upstream's `postprocess.py`.

### Passes, with limits worth knowing

**DINOv3.** It has no non-commercial clause, and it treats output separately from the licensed materials (*"THE DINO MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM"*), so its redistribution and deletion terms apply to the materials and not to your GLBs. Its limits:

- No use for activities under ITAR, or end uses prohibited by trade controls, including military, nuclear, espionage, or the development or use of guns or illegal weapons. **Read this clause if your project touches those fields.**
- No reverse engineering.
- You indemnify Meta against third-party claims, and the licence ends if you sue Meta over it.
- Meta can change the terms, effective on continued use.

The weights stay in a Docker volume and are never committed or baked into an image, so anyone else running this project accepts the licence themselves.

**Intel runtime.** The EULA allows *"personal or business use"* and nothing in it claims or restricts what you produce. This was read through a summarising tool, so read the text before relying on it. One limit matters: section 3.1(xi) prohibits *"SaaS services or service bureau purposes"*. Generating assets for your own project is unaffected, but **re-read this before offering generation as a hosted service to customers.** Publishing the image has its own redistribution terms, and nothing in this plan needs that.

**Training data.** MIT on the weights is Microsoft's grant, and no licence you accept restricts the output. Some training data does carry non-commercial terms, though. Upstream's `data_toolkit/README.md` labels these as training sets:

| Dataset | Licence |
|---|---|
| HSSD | CC BY-NC 4.0 |
| TexVerse | ODC-By overall; per object, 131,823 of 858,669 (15.4%) are CC BY-NC variants |
| Objaverse-XL | ODC-By overall; individual objects vary |
| ABO | CC BY 4.0 |

I found no statement on whether non-commercial objects were excluded. Those terms bind whoever used the datasets, not you, and whether they reach a model's output is unsettled law that affects generative models generally. No engineering step changes this. If the commercial stakes are high, have counsel look at it.

**BiRefNet.** The weights are MIT, so it passes the licence test. Its card says it was trained on DIS-TR, part of DIS5K, whose terms state: *"Without permission from the original authors, commercial use of this dataset is prohibited even after copying, editing, processing or any operations of this database."* This is the same category as the note above, but the wording explicitly claims to reach processed forms, and the component is optional, so the plan leaves it off by default. Whether to enable it is the owner's call.

### Rules for the implementer

- Do not run RMBG-2.0 or install `nvdiffrast` or `nvdiffrec`, even for a quality comparison.
- Before adding any model or package, read its terms for non-commercial, research-only or evaluation-only limits, and for limits on generated output. Copyleft alone is not a problem for this test.
- `scripts/check_licenses.py` and `scripts/fetch_weights.py` enforce this at build and download time (Phase 1).

## What the research changed

These points differ from the starting assumptions.

1. **The out-of-memory risk at 1024 is probably in attention, not in the convolutions, and it may not occur at all.**
   - Sparse conv features are small because channel count falls as resolution rises. At 1024 the estimate is 1 to 3 GiB including 0.9 GiB of weights.
   - Attention is the only term that can explode. A math-path SDPA builds an `[H, L, L]` tensor:

     | Tokens `L` | 10,000 | 20,000 | 30,000 | 49,152 (cascade cap) |
     |---|---|---|---|---|
     | One bf16 tensor, 12 heads | 2.2 GiB | 8.9 GiB | 20.1 GiB | 54 GiB |

   - However, torch 2.15 tries `overrideable` (oneDNN Graph) first on XPU and explicitly accepts `kBFloat16`, the flow models' dtype. The conditions are 4-D inputs, no dropout, no grad and head dim up to 576, and this model's head dim is 128.
   - oneDNN's `sdpa/micro.cl` is tile-based like flash attention, so it should not build the L×L tensor.
   - This has only been read, not run, so Phase 1 measures it before anything depends on it.
2. **Triton on XPU is not the blocker.** `torch==2.15.0+xpu` already depends on `triton-xpu~=3.9.0`. What blocks FlexGEMM is its compiled CUDA extension and its `torch.cuda.*` calls (listed in Phase 6). Its kernels are also a poor fit for Intel's fast path, so the gain is uncertain.
3. **A SYCL attention already exists.** torch-xpu-ops contains a SYCL flash attention (`transformers/xpu/flash_attn/sycltla`), ranked after oneDNN in the backend order. Writing another would duplicate it.
4. **The fallback baker becomes the main path and has defects that the Metal path hid.** See Phase 3. On a Mac almost nobody reaches this code, but here every run does.
5. **Usable VRAM is 22.71 GiB, not 24.** This is what `clinfo` reports for the device.

## Test hardware

Everything was developed and measured on one host.

- **GPU:** Arc Pro B60 on the `xe` driver. `clinfo` reports fp16, fp64 and bf16 conversions, and a 22.71 GiB maximum single allocation through OpenCL.
- **Host:** 30 GiB RAM, kernel 7.2.9, Docker 29.8.2, Compose 5.6.1.
- **Host GPU stack:** compute-runtime 26.35.39758.10, IGC 2.41.5, Level Zero loader 1.32. The image installs the same versions.
- **`xpu-smi` did not work:** it reported "No device discovered". `nvtop` on the host, the `torch.xpu` counters and `/proc/self/fdinfo` inside the container did.
- **Why Docker helps:** the torch wheel pins `intel-sycl-rt==2026.1.2`, and Intel's Triton README calls its backend incompatible with the oneAPI Base Toolkit. A host with oneAPI in its shell environment would conflict with both; a container without oneAPI does not.

### Verified

- **Wheels exist for cp311:** `torch 2.15.0+xpu`, `torchvision 0.29.1+xpu`, `triton_xpu 3.9.0`.
- **The dependency set resolves:** the `requirements-xpu.in` below locks to 61 packages with `uv pip compile`, without installing anything.
- **GPU packages fit the base image:** `libze-intel-gpu1` 26.35 requires `libc6 >= 2.38` and `libstdc++6 >= 13.1`, which Ubuntu 24.04 meets, and Level Zero publishes `libze1_1.32.0+u24.04`. Intel validated this compute-runtime release on Ubuntu 26.04, not 24.04.
- **All patches apply:** a dry run of `patches/mps_compat.py` on a copy of TRELLIS.2 `75fbf01` landed all 8 source patches.
- **Skipping Metal works at import level:** a static import trace shows the image-to-3D path never imports `cumesh`, `nvdiffrast` or `flex_gemm` unguarded, and does not reach `cv2` or `utils3d`. `install_stubs()` is never called, and it is not needed. `easydict` is imported, so it must be installed.
- **Test input has real transparency:** upstream's `T.png` has a minimum alpha of 0, with 2.3% of pixels opaque.
- **Op coverage:** v2.15 `native_functions.yaml` has an XPU or composite kernel for every op audited (`polar`, `searchsorted`, `unique_dim`, `index_add`, `grid_sampler_3d`, and others). Composites were not traced further, and dtype-specific gaps would not show here, so the Phase 1 smoke test still matters.
- **Neighbour lookup cost**, measured on CPU with NumPy on a synthetic shell, for one coordinate set with a 3×3×3 kernel. The dict figures are a lower bound, because the real code also calls `tensor.tolist()` per element.

  | Voxels | Dict (current approach) | Sort + searchsorted |
  |---|---|---|
  | 395K (near the 512 top level, based on the M4 Pro's ~400K vertices) | 3.17 s, 536 MiB | 0.30 s, 43 MiB |
  | 1.58M (estimated 1024 top level, 4× the 512 figure) | 15.6 s, 2174 MiB | 1.33 s, 187 MiB |
  | 6.32M | not run | 6.51 s, 665 MiB |

### Not verified

Nothing has been built or run yet, so every runtime statement above comes from reading source. Still open:

- whether `torch.xpu` initialises with this image's GPU stack on the host kernel
- whether oneDNN selects its micro kernel for bf16
- how `xe` treats long dispatches
- whether `transformers` needs `accelerate` for DINOv3 (it was left out)
- all timings

## Prerequisites

Request access to `facebook/dinov3-vitl16-pretrain-lvd1689m` and put a Hugging Face read token in `.secrets/hf_token` (gitignored, mode 600). Accepting the gate is agreeing to its licence, so read the end-use clause above first. The gate is reviewed manually and blocks every real run. RMBG-2.0 access is not needed and should not be requested.

**Defaults assumed:** this fork becomes XPU-only and Docker-only, with the Metal code removed. Input is RGBA unless BiRefNet is explicitly enabled.

## Sharing the GPU

A run can use nearly all of the card's memory, so stop any other GPU workload before a GPU job and group GPU runs together. Building the image, fetching weights and running `cpu-tests` do not need the GPU. Do not run experimental kernels alongside another GPU service: a GPU hang would take it down.

## Phase 1: Container image and runtime gate

Delete `setup.sh`. Add `Dockerfile`, `compose.yaml`, `.dockerignore`, `requirements-xpu.in`, `requirements-xpu.lock`, `scripts/check_licenses.py` and `scripts/fetch_weights.py`. Add `input/` and `out/` to `.gitignore`.

**Lock the dependencies.** `requirements-xpu.in`:

```text
torch==2.15.0+xpu
torchvision==0.29.1+xpu
transformers==4.57.6
safetensors
pillow
numpy
trimesh
scipy
tqdm
easydict
opencv-python-headless<5
xatlas
fast-simplification
pytest
```

```bash
uv pip compile requirements-xpu.in \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 --only-binary :all: \
  --index-strategy unsafe-best-match \
  --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/xpu \
  --generate-hashes -o requirements-xpu.lock
```

- This exact command resolved successfully without `--generate-hashes`.
- The hashes matter, because `unsafe-best-match` would otherwise accept a package from either index.
- Install from the lock with `--require-hashes`.
- `transformers` is pinned for reproducibility. The DINOv3 patch handles both the 4.57 and 5.x layouts.
- OpenCV stays below 5 to avoid a new major version for one `cv2.inpaint` call. It resolves to 4.14.0.94.
- `torchaudio`, `utils3d`, `accelerate` and `imageio` are left out because nothing on this path was found to use them. Add one back only if a run proves it necessary.

**`Dockerfile`, in this layer order so that code edits rebuild in seconds:**

1. **Base:** `ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852`. Do not base it on `intel/oneapi-basekit`, because the toolkit is exactly what this image must not contain.
2. **GPU user space only,** matching the host's versions and checked against the published checksums (`ww35.sum` for compute-runtime, the IGC release page for IGC):
   - compute-runtime 26.35.39758.10: `intel-opencl-icd`, `libze-intel-gpu1`, `intel-ocloc`, `libigdgmm12` 22.10.0
   - IGC 2.41.5: `intel-igc-core-2`, `intel-igc-opencl-2`
   - Level Zero: `libze1_1.32.0+u24.04`
3. **Python:** `uv`, Python 3.11, and a venv at `/opt/venv` from the hash lock.
4. **Upstream:** clone TRELLIS.2 to `/opt/app/TRELLIS.2` and check out exactly `75fbf01`. It is currently an unpinned `--depth 1` clone, and the patches are exact string matches.
5. **Our code:** copy it to `/opt/app`, apply `patches/xpu_compat.py`, and run `scripts/check_licenses.py`.
6. **Runtime user:** create `/cache` owned by UID 1000, so the named volume inherits that, and set `USER 1000:1000` and `WORKDIR /opt/app`.

The image contains **no weights, no Hugging Face token and no oneAPI.**

**`compose.yaml`.** Drop any device setting that the probe shows is unnecessary.

```yaml
name: trellis-xpu
services:
  trellis:                  # GPU runs: no network, weights come from the volume
    build: .
    image: trellis-xpu:dev
    network_mode: none
    devices: ["/dev/dri:/dev/dri"]
    group_add: ["983", "987"]   # host video and render
    ipc: host
    cap_add: [IPC_LOCK]
    ulimits: { memlock: { soft: -1, hard: -1 } }
    environment:
      HF_HOME: /cache/hf
      HF_HUB_OFFLINE: "1"
      ONEAPI_DEVICE_SELECTOR: level_zero:gpu
      SYCL_CACHE_PERSISTENT: "1"
      SYCL_CACHE_DIR: /cache/sycl
    volumes: ["trellis-cache:/cache", "./input:/input:ro", "./out:/out"]
  fetch-weights:            # network and token, no GPU
    image: trellis-xpu:dev
    environment: { HF_HOME: /cache/hf, HF_TOKEN_PATH: /run/secrets/hf_token }
    secrets: [hf_token]
    volumes: ["trellis-cache:/cache"]
    command: ["/opt/venv/bin/python", "scripts/fetch_weights.py"]
  cpu-tests:                # no GPU
    image: trellis-xpu:dev
    network_mode: none
    command: ["/opt/venv/bin/python", "-m", "pytest", "tests"]
secrets:
  hf_token: { file: ./.secrets/hf_token }
volumes:
  trellis-cache: {}
```

`network_mode: none` means a GPU run cannot silently fetch a different model. Never set a token through `ENV` or a build `ARG`, and never print it. **`.dockerignore` must list `.secrets/`,** so that copying the repo into the image cannot include the token. After the first build, confirm with `docker compose run --rm cpu-tests test ! -e /opt/app/.secrets`.

**`scripts/check_licenses.py`** runs at build time and guards against the two real failures:

- Fail if `nvdiffrast` or `nvdiffrec` is installed.
- Fail if any installed package's licence metadata mentions non-commercial, research-only or evaluation-only use. Today nothing in the lock does.
- Write an inventory of every package and licence to `docs/third-party-licenses.txt`, so a change shows up in review.

**`scripts/fetch_weights.py`** downloads the three repos below at their default revision, confirms each commit matches, and fails if the cache contains `models--briaai*`:

| Repo | Commit |
|---|---|
| `microsoft/TRELLIS.2-4B` | `af44b45f2e35a493886929c6d786e563ec68364d` |
| `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |

**Add `scripts/xpu_probe.py`,** run it with `docker compose run --rm trellis /opt/venv/bin/python scripts/xpu_probe.py`, and record results in `docs/xpu-measurements.md`:

1. Device name, `get_device_properties(0)` and `mem_get_info(0)`. Confirm exactly one device, and confirm through `/proc/self/maps` that `libsycl` loads from `/opt/venv`.
2. `torch._fused_sdp_choice` for `[1, 12, L, 128]` in bf16, fp16 and fp32.
3. An SDPA sweep with L in 4096, 8192, 16384, 32768 and 49152, recording `max_memory_allocated`, `drm-total-vram0` sampled from `/proc/self/fdinfo` (confirm the key name first) and wall time (call `torch.xpu.synchronize()` before stopping the clock). Run once with `ONEDNN_VERBOSE=1` to see the implementation chosen. `_fused_sdp_choice` cannot reveal an internal oneDNN fallback, so **peak memory is the real test**.
4. Numerical agreement with CPU math-path SDPA at L=1024, and a NaN check in bf16.
5. `torch.mm` agreement with CPU at 2^18 to 2^22 rows in fp16 and fp32. A ROCm port of this project hit silent GEMM corruption above 2^19 rows on other hardware, and this primitive carries the whole port.
6. A 6 GiB tensor write and read-back. If it fails, retest with `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. Setting an environment variable does not prove a driver mode changed, so confirm by behaviour.
7. An op smoke test with `PYTORCH_DEBUG_XPU_FALLBACK=1` and `PYTORCH_ENABLE_XPU_FALLBACK` **unset**, so any missing op raises an error instead of silently using the CPU.

**Gate:**

| Result | Action |
|---|---|
| `torch.xpu` does not initialise | Try `ubuntu:26.04`, which Intel validated for this release, with its archive `libze1` 1.28.2. Then try the older Kobuk PPA stack (compute-runtime 26.05) |
| SDPA peak memory is flat as L grows | Skip the chunked attention in Phase 5 |
| SDPA peak memory grows as L² | Chunked attention becomes required before any complex 512 asset |
| bf16 gives NaNs or wrong results | Convert flow models with `convert_to`. Try fp16. Do not use fp32: the probe showed fused SDPA builds an L×L tensor and runs out of memory at L=32768 |

## Phase 2: First geometry at 512

**`patches/mps_compat.py` becomes `patches/xpu_compat.py`:**

- Make every replacement fail hard if its target is not found. `str.replace` currently does nothing silently, which would produce a half-patched tree, and for RMBG that would be a licensing failure.
- **Remove RMBG-2.0.** In `trellis2_image_to_3d.py`, replace `pipeline.rembg_model = getattr(rembg, ...)(...)` with `pipeline.rembg_model = None`. `from_pretrained` currently loads it even for RGBA input. Make `preprocess_image` raise a clear error when the image has no real transparency. Delete `patch_birefnet`.
- `patch_pipeline`: use `torch.xpu.empty_cache()`, called between stages.
- `patch_pipeline_base`: use `torch.device("xpu")`.
- `patch_mesh_base`: keep the no-ops and correct the comments, because `cumesh` is absent rather than crashing.
- `patch_sparse_attention`:
  - Add a direct path for one sample with no padded copies.
  - Reject batches above 1, because the padded path passes no mask and shorter samples would attend to zero padding.
  - Keep the last dimension contiguous, because the fused backend requires a stride of 1.
- Delete the unused `install_stubs()` and `backends/stubs.py`.

**`generate.py`:**

- Call `pipeline.to(torch.device("xpu"))`.
- Remove the Metal bake branch, the MPS fallback variable and the macOS watchdog text.
- Print per-stage timing and peak VRAM.
- Replace the watchdog help with out-of-memory and device-lost guidance that points to `journalctl -k` on the host.

**Tests (`cpu-tests`):** `tests/test_no_rmbg.py` confirms the patched pipeline source never instantiates a remover and that an opaque RGB input raises the clear error.

**Verify** (use existing `conv_none.py` and `mesh_extract.py` unchanged here, to get a baseline):

```bash
docker compose build
docker compose run --rm fetch-weights
docker compose run --rm trellis /opt/venv/bin/python generate.py \
  TRELLIS.2/assets/example_image/T.png --pipeline-type 512 --no-texture --output /out/t512
```

Pass criteria:

- The run exits 0 with no NaNs and no fallback warnings, with no network, and with no remover in the Hugging Face cache.
- The vertex count is in the hundreds of thousands (the M4 Pro's shoe reference had about 400K).
- The OBJ looks correct.
- Stage times are recorded against the README's M4 Pro table.

`assets/shoe_input.png` is RGB, so it needs a manual cut-out or Phase 6E. Upstream also ships 72 `.webp` examples, so use ones with real alpha for later phases.

## Phase 3: Correct GLB output

`backends/texture_baker.py` differs from upstream's `to_glb` in ways that change the result. Fix these:

| Issue | Current code | Upstream `to_glb` |
|---|---|---|
| Gamma | Applies `power(1/2.2)`, which brightens colours | Writes `attrs * 255` directly |
| V coordinate | No flip, yet trimesh's exporter does `uv[:, 1] = 1.0 - uv[:, 1]`, so lookups appear mirrored | Flips V first to compensate |
| Axes | Exports raw Z-up | `y, z = z, -y` for vertices and normals |
| Material factors | `metallicFactor=0.0` multiplies the metallic texture to zero | `1.0` and `1.0`, plus `doubleSided=True` |
| Sampling | KDTree with inverse-distance weighting | Sparse trilinear |
| Gap filling | Box filter on base colour only | `cv2.inpaint` (Telea) on every channel |

Match upstream's behaviour in each row, but write the code here. Do not import upstream's `postprocess.py`, because it imports `nvdiffrast`.

**Trilinear details**, from `grid_sample.cu`:

- Query point: `q = (pos - aabb_min) / voxel_size`
- Base voxel: `floor(q - 0.5)`
- Weight per corner: `prod(1 - |q - x - 0.5|)`
- Missing voxels get zero weight, and weights are normalised by their sum. Share the lookup helper with Phase 4.

**Surface projection:** upstream projects texels onto the original high-resolution mesh with a BVH before sampling. Start without it and log the share of texels with non-zero weight. If that is below 99%, add an approximation using a `cKDTree` over the original vertices.

**Test first with `tests/test_baker.py` (`cpu-tests`).** Use a mesh with a known colour field, bake, export to GLB, reload with trimesh, sample the texture at the vertex UVs, and compare with the expected colours. This single round trip catches the V, axis and gamma errors. It must **fail on the current code**, which also confirms the V-flip reading, and pass after the fix.

**Verify:** `T.png` and two `.webp` examples with real alpha open in a glTF viewer upright, with sensible colours. No NVIDIA baseline exists here, so parity is structural and visual.

## Phase 4: Vectorise the Python hot spots

**`conv_none.py`:**

- Build an `[N, V]` int32 neighbour map with sort and `searchsorted` on the device. Upstream's `_compute_neighbor_cache_torch` is a reference.
- Loop over kernel slots using `index_select`, matmul and `index_add_`. This also removes today's scan of all edges for each of the 27 slots.
- Accumulate in fp32, as FlexGEMM's kernels do (`tl.zeros(..., dtype=tl.float32)`). The current code accumulates in fp16.
- Keep the per-slot loop rather than upstream's explicit GEMM, because its im2col buffer costs `N * 27 * Ci` of memory.

**`mesh_extract.py`:** use the same lookup in place of the Python dict.

**Tests (`cpu-tests`):**

- New conv equals old `conv_none` on random coordinates and dilations, within fp32 tolerance.
- New `mesh_extract` produces identical faces to the old version.

**Verify:** rerun the Phase 3 assets. Decode time drops, RAM spikes shrink, and the output is visually unchanged.

## Phase 5: Higher resolutions

- Add `1536_cascade` to `--pipeline-type` (it is currently missing) and add `--max-tokens`, which maps to `max_num_tokens`.
- For `--pipeline-type 512`, skip loading the two 1024 flow models. That saves about 5 GB of the 30 GiB RAM.
- **Only if Phase 1 showed L² growth:** add chunked attention by slicing Q into blocks in the SDPA path. Each query's softmax is independent, so the result is mathematically identical. Test equality against the unchunked result at small L. Chunking also shortens individual GPU dispatches.
- Run three RGBA assets at each level and record tokens, peak VRAM, RAM and time. Runtime scales with L², so expect a large jump at 49K tokens.

**Pass criteria for each level:** no out-of-memory error or device loss, and a correct GLB. If a level fails, record why and stop there. A documented limit is an acceptable outcome.

## Phase 6: Optional work, only if measurements or needs justify it

Use the Phase 5 stage profile. Do not start any of these from intuition.

**A. FlexGEMM Triton kernels (MIT).** Consider this only if decoders exceed 25% of end-to-end time at the target resolution. Timebox it to one day. Required changes, from reading the source:

- `kernels/__init__.py` imports the compiled `cuda` module unconditionally. Add a backend switch, as mtlgemm did with `FLEX_GEMM_BACKEND`.
- `setup.py` uses `CUDAExtension`. It needs a pure-Python install path.
- `ops/spconv/submanifold_conv3d.py` raises `NotImplementedError` when `coords.is_cuda` is false. Three native calls need PyTorch equivalents, all simple bit and sort operations:
  - `hashmap_build_submanifold_conv_neighbour_map`
  - `neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd`
  - `neighbor_map_post_process_for_masked_implicit_gemm_2`
- `kernels/triton/utils.py` and `utils/autotuner.py` call `torch.cuda.*`. Use the `torch.xpu` equivalents.
- `kernels/triton/spconv/config.py` has no `xpu` platform, so `get_autotune_config` raises `ValueError` at import. Add a config.
- Start with `implicit_gemm`, which needs only the neighbour map, then try the masked variant.

**Why the gain is uncertain:** these kernels gather inputs through per-row neighbour pointers. Intel's backend README recommends 2-D tensor descriptors on `tl.dot` and claims more than 2× from them, and the neighbour-indexed gather does not fit that pattern. A oneDNN matmul per slot may already be competitive.

**Gate:** at least 2× faster than Phase 4 on real decoder time, with matching output. Otherwise drop it.

**B. FlexAttention.** Use it only if oneDNN's SDPA is unavailable or fails Phase 1. Intel's B580 data shows FlexAttention on par with oneDNN for standard multi-head attention, so otherwise no gain is expected.

**C. SYCL kernels.** This is the last resort. Consider it only if a single kernel exceeds 30% of runtime and both oneDNN/PyTorch and Triton have failed the gates. The costs are real: the compiler must match the wheel's runtime (2026.1.2, so build it in a dedicated image stage), and the extension must be rebuilt at every torch upgrade. The likely verdict is that it stays unnecessary.

**D. Hole filling.** `fill_holes` is a no-op on this path, so meshes may have small holes. If they are visible, add a CPU step limited to upstream's `max_hole_perimeter=3e-2`. Check the library's terms against the licensing test first. Do not port CuMesh.

**E. Opt-in background removal.** Do this when RGB input is wanted and the owner has decided to accept BiRefNet's training-data question. It does not depend on the stage profile.

- Add `--remove-background`. Without it, RGB input keeps raising the clear error from Phase 2.
- Load `ZhengPeng7/BiRefNet` at commit `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`, fetched only by `fetch-weights --with-remover`. Its custom code runs through `trust_remote_code`, so the pinned commit matters.
- Run it on the **CPU.** It calls torchvision's `deform_conv2d`, whose kernel I found only for CPU, and it runs once per image, so speed barely matters.
- Add `timm`, `kornia` and `einops` to the lock.

A paid BRIA licence for RMBG-2.0 is the alternative. Any other remover must pass the licensing test, including a look at its training data.

## Phase 7: Documentation

- Rewrite the README for Docker and XPU: requirements, RGBA input, the GPU-sharing note, measured results and limitations. Remove every RMBG-2.0 instruction.
- Replace the README's licence section with the test and verdicts from this plan, including the DINOv3 and Intel limits, and state that DINOv3 is downloaded separately under its own licence.
- Keep the existing MIT notice and the credits. `LICENSE` reads `Copyright (c) 2026` with no holder, so add a holder line for your changes as small cleanup.
- Rename the project in `pyproject.toml` and update its dependencies.
- Keep `docs/xpu-measurements.md` as the evidence behind any claim.

## Risks

| Risk | Detection | Response |
|---|---|---|
| Image's GPU stack does not work with the host kernel | Phase 1 probe | Fallback stacks in the Phase 1 gate |
| RMBG-2.0 or `nvdiffrast` comes back through an upgrade or a quality fix | `check_licenses.py` fails the build, `fetch_weights.py` checks the cache, GPU runs have no network | Remove it |
| Project later becomes a hosted service for customers | Owner decision | Re-read Intel EULA section 3.1(xi) first |
| Upstream weights change under the same name | Commit check in `fetch_weights.py` | Review the change and update the pin deliberately |
| Weights or token leak into an image | Inspect with `docker history`; only volumes and secrets hold them | Rebuild; rotate the token |
| oneDNN falls back to an L×L path | SDPA sweep | Chunked attention |
| Silent wrong results from large GEMMs | Phase 1 CPU comparison | Chunk the matmul |
| `xe` kills a long dispatch (`ccs` shows `job_timeout_ms=5000`, `preempt_timeout_us=640000`; whether they apply to compute jobs is unconfirmed) | Device-lost error, `journalctl -k` on the host | Shorter dispatches through chunking |
| System RAM pressure (all 8 models load to CPU first) | RSS logging | Selective model loading (Phase 5), vectorised lookups (Phase 4) |

## Sources

**Licensing:**

- [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [RMBG-2.0 model card](https://huggingface.co/briaai/RMBG-2.0)
- [BiRefNet model card](https://huggingface.co/ZhengPeng7/BiRefNet) and [DIS5K Terms of Use](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf)
- [TRELLIS.2-4B model card](https://huggingface.co/microsoft/TRELLIS.2-4B) and [`data_toolkit` README](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)
- [HSSD](https://huggingface.co/datasets/hssd/hssd-models), [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse), [Objaverse-XL](https://huggingface.co/datasets/allenai/objaverse-xl), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [nvdiffrast licence](https://github.com/NVlabs/nvdiffrast/blob/v0.4.0/LICENSE.txt) and [nvdiffrec licence](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt)
- [Intel Developer Tools EULA](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html)

**Runtime and image:**

- [PyTorch XPU getting started](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) and [XPU wheel index](https://download.pytorch.org/whl/xpu/torch/)
- [compute-runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10) and [Level Zero v1.32.0](https://github.com/oneapi-src/level-zero/releases/tag/v1.32.0)
- [XPU SDPA backend selection, torch v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mkldnn/xpu/Attention.cpp)
- [oneDNN SDPA kernel](https://github.com/uxlfoundation/oneDNN/blob/main/src/gpu/intel/sdpa/micro.cl)
- [torch-xpu-ops](https://github.com/intel/torch-xpu-ops) and [Intel XPU backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)
- [FlexAttention on Intel GPUs](https://pytorch.org/blog/pytorch-2-9-flexattention-optimization-practice-on-intel-gpus/)
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), [mtlgemm](https://github.com/pedronaugusto/mtlgemm)
- [TRELLIS.2 ROCm port notes](https://github.com/toastmanAu/trellis-2-rocm-comfyui)
