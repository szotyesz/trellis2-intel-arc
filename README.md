# TRELLIS.2 on Intel Arc

Runs [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D generation on an Intel Arc Pro B60 (24 GB) with PyTorch XPU, in Docker. It turns one cut-out image into a textured GLB.

This is a fork of [trellis-mac](https://github.com/shivampkumar/trellis-mac). The Mac and Metal paths have been removed.

## Requirements

- Linux host with an Intel Arc GPU and the `xe` driver. Only the B60 has been tested.
- Docker with Compose.
- 30 GB of system RAM has been enough; a run peaks at about 17 GiB, and the container is capped at 26 GiB.
- Disk space for the model weights.
- A Hugging Face read token, with access approved for the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

## Setup

```sh
mkdir -p .secrets
printf '%s' 'YOUR_HF_READ_TOKEN' > .secrets/hf_token   # gitignored
chmod 600 .secrets/hf_token

docker compose build trellis
docker compose run --rm fetch-weights                  # the only step with network and token
docker compose run --rm --no-deps cpu-tests            # no GPU needed
```

`compose.yaml` sets the host's `video` and `render` group IDs (`983`, `987`). Change them if yours differ: `getent group video render`.

## Generate

Put an image with a **transparent background** in `input/`. Background removal is not included, and an opaque image stops with an error.

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
| `--target-faces` | `200000` | Triangles in the GLB |
| `--texture-size` | `1024` | `512`, `1024`, `2048` or `4096` |
| `--repair` | `cut` | How non-manifold edges are repaired before baking. `pair` keeps surfaces better connected and makes about a third fewer UV charts, but UV unwrapping then takes 6–11 minutes instead of under one |
| `--max-tokens` | `49152` | Upper limit on sparse tokens in cascade pipelines |
| `--steps` | model default | Sampler steps for all phases |
| `--no-texture` | off | Write only the raw mesh as OBJ |
| `--save-raw` | off | Also write the raw OBJ, which can reach hundreds of MB |

## Choosing a pipeline type

`1024_cascade` is the best balance for solid objects, at about 5 minutes per image. `512` takes about 2 minutes and is the fallback for anything that fails. `1536_cascade` takes 2–3 times longer than `1024_cascade` and shows little extra detail once baked to 200,000 triangles. Measurements are in [docs/xpu-measurements.md](docs/xpu-measurements.md).

## Limitations

- **GPU memory:** every input retested after the convolution fixes completes at every level, with peak VRAM up to about 19 GiB at `1536_cascade`. A run that does run out exits with an error rather than hanging.
- **First run:** Triton compiles the convolution kernel on first use, about 10 s once per cache volume.
- **Fine parts:** thin structures can break up or become floating specks.
- **Colour:** output is often darker or less saturated than the input. The cause has not been fully traced; see the measurements.
- **Transparency:** glass comes out as an opaque shell.
- **Scenes:** whole rooms and towns only look right from the input's side.

## How the port works

| Upstream needs | Here |
|---|---|
| CUDA sparse convolution (FlexGEMM) | `backends/conv_none.py`: a Triton gather-matmul kernel on XPU, a plain PyTorch loop on CPU and as fallback |
| CUDA mesh extraction (`o_voxel`) | `backends/mesh_extract.py`, plain PyTorch |
| `nvdiffrast` texture baking | `backends/texture_baker.py`, NumPy and xatlas on the CPU |
| CuMesh repair and simplification | `backends/texture_baker.py`: vertex clustering for huge meshes, winding repair, non-manifold vertex splitting, `fast_simplification` |
| RMBG-2.0 background removal | Removed, because of its non-commercial licence |

`patches/xpu_compat.py` patches upstream at image build time and fails the build if an expected line is missing. CPU fallback stays disabled, so a missing XPU operator raises an error rather than silently running slowly. GPU containers run with no network.

The design and its reasoning are in [docs/XPU_PORT_PLAN.md](docs/XPU_PORT_PLAN.md).

## Licences and commercial use

The port's own code is MIT. The goal is that nothing in the default pipeline prohibits commercial use of the generated output. `scripts/check_licenses.py` checks the installed packages. This is an engineering summary, not legal advice.

- **TRELLIS.2 code and weights:** [MIT](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE). A residual risk remains around the data the model was trained on.
- **DINOv3 weights:** [Meta's custom licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md), gated and with end-use restrictions. Review it before commercial use.
- **Intel runtime components:** Intel's EULA forbids offering them as a hosted service or service bureau. Check this if you plan to run generation as a service for others.
- **Excluded because they forbid commercial use:** RMBG-2.0, `nvdiffrast` and `nvdiffrec`. Do not add them back, and do not import upstream's `o_voxel.postprocess`, which imports `nvdiffrast`.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research: the model and codebase
- [DINOv3](https://github.com/facebookresearch/dinov3) by Meta: image features
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by shivampkumar: the port this fork started from
