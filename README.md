# TRELLIS.2 on Intel Arc Pro B60: image-to-3D with PyTorch XPU, no CUDA

Run Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D model (`TRELLIS.2-4B`) on an **Intel Arc Pro B60** (24 GB) under Linux, using PyTorch's XPU backend in Docker. No NVIDIA GPU, no CUDA and no oneAPI install on the host. One cut-out image goes in, a textured PBR mesh (GLB) comes out. It can also texture a mesh you already have, and run several images in one go.

![Input image and three views of the mesh TRELLIS.2 generated from it on an Intel Arc Pro B60](assets/machine_1024_cascade.jpg)

*Input (left, one of upstream's example images) and the GLB generated at `1024_cascade` on the B60, rendered from three sides.*

Upstream TRELLIS.2 runs only on NVIDIA cards: it calls `.cuda()` directly and depends on the CUDA extensions `flex_gemm`, `o_voxel`, `cumesh`, `nvdiffrast` and `flash_attn`. This port replaces each of them, so the image-to-3D pipeline and the texturing pipeline both run on an Intel GPU through `torch.xpu`.

| | |
|---|---|
| Tested GPU | Intel Arc Pro B60, 24 GB, `xe` driver. Other Arc cards (B580, B570, A770) are untested; see the [FAQ](#faq) |
| Stack | `torch 2.15.0+xpu`, `triton-xpu`, compute-runtime 26.35, Level Zero 1.32, in an Ubuntu 24.04 container |
| Pipelines | `512`, `1024`, `1024_cascade`, `1536_cascade`: all 12 benchmark inputs complete at every level |
| Time on the GPU | Median 30 s at `512`, 100 s at `1024` and `1024_cascade`, 300 s at `1536_cascade`, plus CPU post-processing |
| Peak VRAM | 2.6–4.6 GiB at `512`, up to about 20 GiB at the higher levels |
| Output | GLB with 200,000 triangles, base colour with alpha and metallic-roughness textures; glass is exported as a blended material |
| Texturing | `texture.py` paints an existing mesh from an image, as upstream's `Trellis2TexturingPipeline` does |
| Batches | Several images, or several seeds of one, in a single run that loads the models once |
| Not ported | Upstream's Gradio apps, video rendering and training |

This started as a fork of [trellis-mac](https://github.com/shivampkumar/trellis-mac), the Apple Silicon port. The Mac and Metal paths have been removed.

## Requirements

- Linux host with an Intel Arc GPU and the `xe` kernel driver. Only the Arc Pro B60 has been tested. The host needs no oneAPI, Level Zero or PyTorch install: the image carries the whole user-space GPU stack.
- Docker with Compose.
- 30 GB of system RAM has been enough; a run peaks at about 17 GiB, and the container is capped at 26 GiB.
- Disk space for the model weights.
- A Hugging Face read token, with access approved for the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). For the optional SAM 3 background removal, also for the gated [SAM 3 weights](https://huggingface.co/facebook/sam3).

## Setup

```sh
mkdir -p .secrets
printf '%s' 'YOUR_HF_READ_TOKEN' > .secrets/hf_token   # gitignored
chmod 600 .secrets/hf_token

docker compose build trellis
docker compose run --rm fetch-weights                  # the only step with network and token
docker compose run --rm --no-deps cpu-tests            # no GPU needed
```

For SAM 3 background removal, fetch its weights too, after accepting Meta's licence on the Hugging Face page:

```sh
docker compose run --rm fetch-weights /opt/venv/bin/python scripts/fetch_weights.py --with sam3
```

The container needs the host's `video` and `render` group IDs to open the GPU. `compose.yaml` defaults to `983` and `987`; check yours with `getent group video render` and, if they differ, set `VIDEO_GID` and `RENDER_GID` in the environment or in a `.env` file next to `compose.yaml`.

## Generate

Put an image with a **transparent background** in `input/`. An opaque image stops with an error; cut the object out first with `scripts/cutout.py` (below).

```sh
docker compose run --rm trellis /opt/venv/bin/python generate.py \
  /input/object.png --pipeline-type 1024_cascade --output /out/object
```

The result is `out/object.glb`. To preview it without a GPU, render it to a PNG with `scripts/preview.py`.

A run can use nearly all of the card's memory, so stop anything else that is using the GPU first.

| Option | Default | Meaning |
|---|---|---|
| `--pipeline-type` | `512` | `512`, `1024`, `1024_cascade` or `1536_cascade` |
| `--seed` | `42` | Random seed. Results still vary slightly between runs |
| `--num-samples` | `1` | Meshes per image. Sample *i* uses seed + *i* and is written as `<name>_s<seed>.glb` |
| `--output` | `/out/output_3d` | Output path without extension, for one image |
| `--output-dir` | `/out` | Where several images are written, each under its own file name |
| `--target-faces` | `200000` | Triangles in the GLB |
| `--texture-size` | `1024` | `512`, `1024`, `2048` or `4096` |
| `--repair` | `pair` | How non-manifold edges are repaired before baking. `pair` halves the cracks that simplification opens and removes most speckle on cloth and paper, but UV unwrapping takes minutes to an hour instead of seconds. If the simplifier stalls on a paired mesh and flattens it, `pair` falls back to `cut` for that mesh, printing why. `cut` always cuts, for quick previews |
| `--alpha-mode` | `auto` | What the GLB does with the alpha the model predicts. `auto` blends when part of the surface is transparent and is opaque otherwise; `blend`, `mask` and `opaque` force one. Alpha is stored in the base colour texture in every mode. See [Transparency](#transparency) |
| `--max-tokens` | `49152` | Upper limit on sparse tokens in cascade pipelines |
| `--steps` | model default | Sampler steps for all phases |
| `--no-texture` | off | Write only the raw mesh as OBJ |
| `--save-raw` | off | Also write the raw OBJ, which can reach hundreds of MB, and a `.raw.npz` that `scripts/bake_raw.py` can bake again on the CPU with other options |
| `--raw-only` | off | Write only the `.raw.npz` and exit, without the OBJ or baking, so the GPU is free before the CPU work. Bake it with `scripts/bake_raw.py` |

### Several images or seeds in one run

Give `generate.py` more than one image, or `--num-samples`, and it loads the models once and runs each stage for every job before moving to the next stage, so every model is still dropped from RAM as soon as its stage is over:

```sh
docker compose run --rm trellis /opt/venv/bin/python generate.py \
  /input/chair.png /input/lamp.png --pipeline-type 1024_cascade --output-dir /out
docker compose run --rm trellis /opt/venv/bin/python generate.py \
  /input/chair.png --num-samples 4 --output /out/chair       # chair_s42.glb … chair_s45.glb
```

Each job draws its noise from its own seed, in the order a run on its own would, so a batch does not change what a seed produces. A job that fails, for example an opaque image or one that runs out of GPU memory, is reported at the end and the others still finish; the exit status is then 1.

The jobs run one after another on the GPU, not side by side: one mesh can already take most of 24 GB at the higher levels. The sparse attention itself accepts several samples in one pass, so upstream's `pipeline.run(image, num_samples=N)` works from Python where the memory allows; at `512` two samples in one pass took 36 s against 26.5 s for one.

### Transparency

TRELLIS.2 predicts an alpha value for every voxel: about 1.0 on solid surfaces, 0.5 or less on glass. Upstream stores it in the base colour texture but marks the material opaque, so viewers ignore it. Here `--alpha-mode auto` (the default) checks how much of the surface is transparent. If it is under 0.5% of the area the GLB is opaque, as before. Otherwise the faces with transparent texels go into a second primitive with a blended material while the rest stay opaque, so the solid parts still write depth and do not sort against the glass. `--alpha-mode opaque` gives upstream's behaviour, `blend` and `mask` force those modes.

### Background removal

`scripts/cutout.py` writes the RGBA image from a photo, using Meta's Segment Anything models. Name the object and each thing it carries for SAM 3, or give nothing and SAM 2 finds a free-standing object on its own:

```sh
docker compose run --rm trellis /opt/venv/bin/python scripts/cutout.py \
  /input/photo.jpg /out/photo.png --prompt farmer basket vegetables
docker compose run --rm trellis /opt/venv/bin/python scripts/cutout.py \
  /input/photo.jpg /out/photo.png
```

Measured on the 12 example images composited over white, grey, gradient and noise backgrounds, against their true alpha:

| | Needs | Mean IoU | Where it fails |
|---|---|---|---|
| SAM 3, `--prompt` | Noun phrases, gated weights | 0.85 | Keeps only what the phrases name: the farmer alone scores 0.72, with "basket" and "vegetables" added 0.98. A generic word like "object" finds nothing |
| SAM 2, no prompt | Nothing | 0.84 | Object and background alike, such as a grey object on grey: 0.2–0.5. Scenes: 0.5–0.7 |
| SAM 2, `--box` | A pixel box | 0.78 | Scenes |

Both take about a second on the GPU. SAM 3 was 848M parameters and about 2 s per image; a plain-PyTorch group norm replaces the XPU kernel that crashed on its mask decoder.

## Texture an existing mesh

`texture.py` is the port of upstream's texturing pipeline. It takes a mesh and an image of the object, and writes the same mesh with a PBR texture painted to match the image:

```sh
docker compose run --rm trellis /opt/venv/bin/python texture.py \
  /input/knight.glb /input/knight.png --output /out/knight
```

The mesh is voxelised on the CPU, encoded to a shape latent on the GPU, and the texture model then generates colour, metallic, roughness and alpha for exactly those voxels, which are baked into a texture. The geometry is not changed: the output has the input's vertices, faces, size and position.

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | `1024` | Voxel grid the mesh is encoded at, `512` or `1024` |
| `--texture-size` | `2048` | `512`, `1024`, `2048` or `4096` |
| `--seed` | `42` | Random seed |
| `--steps` | model default | Sampler steps |
| `--z-up` | off | The mesh's up axis is Z. The default is Y, as in glTF |
| `--unwrap` | off | Make new UVs with xatlas even if the mesh has its own. Meshes without UVs, or with UVs outside the unit square, are always unwrapped |
| `--alpha-mode` | `auto` | As for `generate.py` |

The image needs a transparent background, like the input of `generate.py`, and should show the object from the front, upright. The mesh should be upright too; if it comes from a Z-up tool as OBJ or PLY, pass `--z-up`.

## Choosing a pipeline type

`1024_cascade` is the best balance for solid objects. `512` is the fallback for anything that fails. `1536_cascade` shows little extra detail once baked to 200,000 triangles.

Generation on the GPU takes a median of about 30 s at `512`, 100 s at `1024` and `1024_cascade`, and 300 s at `1536_cascade`. The CPU steps after it usually take longer: with the default `--repair pair` a whole run took a median of 11–12 minutes at every level on the 12 benchmark inputs, from 2 minutes to over an hour, nearly all of it UV unwrapping. `--repair cut` brings a run back to a few minutes. Measurements are in [docs/xpu-measurements.md](docs/xpu-measurements.md).

## Limitations

- **GPU memory:** every benchmark input completes at every level once the allocator is allowed to grow its segments (`PYTORCH_ALLOC_CONF=expandable_segments:True`, set by `generate.py`), with peak VRAM up to 20.2 GiB on the town at `1536_cascade`. Without it, two scene-like inputs failed with gigabytes reserved but unusable. A run that does run out exits with an error rather than hanging.
- **First run:** Triton compiles the convolution kernel on first use, about 10 s once per cache volume.
- **Fine parts:** thin structures can break up or become floating specks.
- **Cracks:** the repair of non-manifold edges cuts the mesh, and simplification then opens the cuts into hairline cracks, which show as speckles on cloth and paper and as small holes at the cascade levels. The default `--repair pair` removes most of them at the price of a slow unwrap; `--repair cut` brings them back together with a fast unwrap. The measurements document records the cause, the fixes that did not work, and the meshes where pairing has to fall back to cutting.
- **Colour:** output is often darker or less saturated than the input. The cause has not been fully traced; see the measurements.
- **Transparency:** glass is exported with a blended material (see [Transparency](#transparency)), which is plain alpha blending: no refraction, and viewers that do not sort blended triangles can draw the far side of a bowl over the near side.
- **Scenes:** whole rooms and towns only look right from the input's side. The triangle budget is tight for them: at 200,000 triangles the town's facades are a few voxels off the surface over a tenth of their area, which the final decimation stage keeps from turning into triangles that span windows and alleys.

## FAQ

**Does it run on an Arc B580, B570 or A770?** Untested. Nothing in the code is specific to the B60: it needs a GPU that PyTorch exposes as `xpu`. Going by the measured peaks, `512` (2.6–4.6 GiB) should fit a 12 GB card, while the higher levels reached 16–20 GiB on heavy inputs. Please open an issue with your result either way.

**Do I need CUDA, an NVIDIA card or oneAPI?** No. The container installs Intel's compute runtime, Level Zero and the XPU build of PyTorch. The host only provides the kernel driver and `/dev/dri`.

**Upstream fails with `AssertionError: Torch not compiled with CUDA enabled`, or cannot build `flex_gemm`, `cumesh`, `nvdiffrast` or `flash_attn`.** That is what this port fixes: those packages are CUDA-only, and the table below lists what stands in for each.

**`torch.xpu.is_available()` is false in the container.** Check that `/dev/dri` exists on the host and that `VIDEO_GID` and `RENDER_GID` match `getent group video render`. Then run `docker compose run --rm trellis /opt/venv/bin/python scripts/xpu_probe.py`, which reports the device, its memory and any missing operator.

**`torch.OutOfMemoryError: XPU out of memory` or `UR_RESULT_ERROR_OUT_OF_RESOURCES`.** Stop anything else that is using the GPU and try a lower `--pipeline-type`. `generate.py` already sets `PYTORCH_ALLOC_CONF=expandable_segments:True`, which cured the fragmentation failures seen on 24 GB.

**Why does a run take ten minutes when generation takes one or two?** UV unwrapping on the CPU, with the default `--repair pair`. Use `--repair cut` for quick previews.

**Windows, WSL2 or ComfyUI?** Untested and not provided. This is a command-line tool in a Linux container.

**The first TRELLIS, or text-to-3D?** No, only TRELLIS.2 image-to-3D.

**Apple Silicon or AMD?** See [trellis-mac](https://github.com/shivampkumar/trellis-mac) for Macs. This repository has no ROCm path.

## How the port works

| Upstream needs | Here |
|---|---|
| `flash_attn` or `xformers` attention | PyTorch's `scaled_dot_product_attention`, which XPU runs with oneDNN's fused kernel, one call per sample of a batch; memory grows linearly with tokens in bf16 and fp16 |
| CUDA sparse convolution (FlexGEMM) | `backends/conv_none.py`: a Triton gather-matmul kernel on XPU, a plain PyTorch loop on CPU and as fallback |
| CUDA mesh extraction (`o_voxel`) | `backends/mesh_extract.py`, plain PyTorch |
| Mesh voxelisation for texturing (`o_voxel`, C++) | `backends/dual_grid.py`, NumPy on the CPU, tested against a transcription of the C++ |
| `nvdiffrast` texture baking | `backends/texture_baker.py`, NumPy and xatlas on the CPU |
| CuMesh repair and simplification | `backends/texture_baker.py`: vertex clustering for huge meshes, winding repair, non-manifold vertex splitting, `fast_simplification` to four times the target, then `backends/surface_decimate.py`, a quadric decimator that refuses collapses whose triangles leave the voxel shell |
| RMBG-2.0 background removal | Removed, because of its non-commercial licence. `scripts/cutout.py` offers SAM 3 and SAM 2 instead |

`patches/xpu_compat.py` patches upstream at image build time and fails the build if an expected line is missing. CPU fallback stays disabled, so a missing XPU operator raises an error rather than silently running slowly. GPU containers run with no network.

The design and its reasoning are in [docs/XPU_PORT_PLAN.md](docs/XPU_PORT_PLAN.md).

## Licences and commercial use

The port's own code is MIT. The goal is that nothing in the default pipeline prohibits commercial use of the generated output. `scripts/check_licenses.py` checks the installed packages at build time and the image keeps its inventory in `/opt/app/third-party-licenses.txt`. This is an engineering summary, not legal advice.

- **TRELLIS.2 code and weights:** [MIT](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE). A residual risk remains around the data the model was trained on.
- **DINOv3 weights:** [Meta's custom licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md), gated and with end-use restrictions. Review it before commercial use.
- **Intel runtime components:** Intel's EULA forbids offering them as a hosted service or service bureau. Check this if you plan to run generation as a service for others.
- **SAM 2.1 weights:** [Apache 2.0](https://github.com/facebookresearch/sam2), trained on Meta's own data.
- **SAM 3 weights:** [Meta's SAM License](https://huggingface.co/facebook/sam3), gated and optional. Royalty-free and commercial use is allowed, but it is not an open-source licence: it excludes military and sanctioned uses, must be passed on with any redistribution of the model, and Meta can change it. The SAM 2 path stays as the fallback.
- **Excluded because they forbid commercial use:** RMBG-2.0, `nvdiffrast` and `nvdiffrec`. Do not add them back, and do not import upstream's `o_voxel.postprocess`, which imports `nvdiffrast`. Also excluded: BiRefNet, BEN2, InSPyReNet and u2net, whose MIT or Apache weights were trained on DIS5K or DUTS, datasets whose terms forbid commercial use even of derived work.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research: the model and codebase
- [DINOv3](https://github.com/facebookresearch/dinov3) by Meta: image features
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by shivampkumar: the port this fork started from
