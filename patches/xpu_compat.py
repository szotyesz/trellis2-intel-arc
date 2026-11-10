"""
Apply all XPU patches to a fresh TRELLIS.2 clone.

Modifies source files in-place to replace CUDA-only code paths with
device-aware code for Intel GPUs (XPU).

Run once after cloning TRELLIS.2:
    python patches/xpu_compat.py
"""

import os
import shutil

TRELLIS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TRELLIS.2")
BACKENDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backends")


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def must_replace(src, old, new, count=-1):
    """str.replace that fails instead of silently leaving a half-patched tree."""
    if old not in src:
        raise SystemExit(f"Patch target not found, upstream changed?\n{old}")
    return src.replace(old, new, count)


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched: {os.path.relpath(path, TRELLIS_ROOT)}")


def patch_sparse_config():
    """Add 'sdpa' and 'naive' to the allowed attention backends."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/config.py")
    src = read_file(path)

    if "'sdpa'" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']",
    )
    write_file(path, src)


def patch_sparse_attention():
    """Add SDPA backend to the sparse attention dispatch."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/attention/full_attn.py")
    src = read_file(path)

    if "'sdpa'" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # One call per sample. Samples differ in length, and padding them into
    # one batch would need a mask to keep short samples from attending to the
    # padding. The last dim must keep stride 1, which the fused oneDNN
    # backend requires; slicing rows keeps it.
    sdpa_block = """\
    elif config.ATTN in ('sdpa', 'naive'):
        from torch.nn.functional import scaled_dot_product_attention as sdpa_fn
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        out, q_start, kv_start = [], 0, 0
        for q_len, kv_len in zip(q_seqlen, kv_seqlen):
            out.append(sdpa_fn(
                q[q_start:q_start + q_len].unsqueeze(0).transpose(1, 2),
                k[kv_start:kv_start + kv_len].unsqueeze(0).transpose(1, 2),
                v[kv_start:kv_start + kv_len].unsqueeze(0).transpose(1, 2),
            ).transpose(1, 2).squeeze(0))
            q_start += q_len
            kv_start += kv_len
        out = out[0] if len(out) == 1 else torch.cat(out)
"""

    src = must_replace(src,
        "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")",
        sdpa_block + "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")",
    )
    write_file(path, src)


def patch_image_feature_extractor():
    """Add device property and replace .cuda() calls with device-aware code."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/image_feature_extractor.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # DinoV2FeatureExtractor: add device property, fix cuda()
    src = must_replace(src,
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "    \n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",

        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",
        1,  # only first occurrence
    )

    # Fix hardcoded .cuda() in both extractors
    src = must_replace(src,
        "            image = torch.stack(image).cuda()",
        "            image = torch.stack(image).to(self.device)",
    )
    src = must_replace(src,
        "        image = self.transform(image).cuda()",
        "        image = self.transform(image).to(self.device)",
    )

    # DinoV3FeatureExtractor: add device property, fix cuda()
    # The second class has the same to/cuda/cpu pattern
    src = must_replace(src,
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",

        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
    )

    # Fix DINOv3 model.layer -> model.model.layer (HuggingFace structure)
    src = must_replace(src,
        "        for i, layer_module in enumerate(self.model.layer):",
        "        layers = self.model.model.layer if hasattr(self.model, 'model') and hasattr(self.model.model, 'layer') else self.model.layer\n"
        "        for i, layer_module in enumerate(layers):",
    )

    write_file(path, src)


def patch_no_rembg():
    """RMBG-2.0 is CC BY-NC, so it must never load. Input must be RGBA."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "XPU_NO_REMBG" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        "        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n",
        "        pipeline.rembg_model = None  # XPU_NO_REMBG\n",
    )
    src = must_replace(src,
        "            input = input.convert('RGB')\n"
        "            if self.low_vram:\n"
        "                self.rembg_model.to(self.device)\n"
        "            output = self.rembg_model(input)\n"
        "            if self.low_vram:\n"
        "                self.rembg_model.cpu()\n",
        "            raise ValueError(\n"
        "                'The input image has no transparency. Cut out the object first '\n"
        "                'and pass an RGBA image.'\n"
        "            )\n",
    )
    write_file(path, src)


def patch_mesh_base():
    """Guard cumesh/flex_gemm imports and unconditionally skip in-place mesh
    ops. cumesh is CUDA-only and not installed. Mesh simplification happens
    later via fast_simplification before texture bake.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/representations/mesh/base.py")
    src = read_file(path)

    if "except (ImportError, RuntimeError)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # Guard imports — cumesh/flex_gemm may or may not be present
    src = must_replace(src,
        "import cumesh\n"
        "from flex_gemm.ops.grid_sample import grid_sample_3d",
        "try:\n"
        "    import cumesh\n"
        "except (ImportError, RuntimeError):\n"
        "    cumesh = None\n"
        "try:\n"
        "    from flex_gemm.ops.grid_sample import grid_sample_3d\n"
        "except (ImportError, RuntimeError):\n"
        '    def grid_sample_3d(*args, **kwargs):\n'
        '        raise RuntimeError("flex_gemm requires CUDA")',
    )

    # cumesh is absent, so these become no-ops
    src = must_replace(src,
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        return  # cumesh is not installed on this path\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    # Unconditionally return from remove_faces
    src = must_replace(src,
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    # Unconditionally return from simplify
    src = must_replace(src,
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    write_file(path, src)


def patch_fdg_vae():
    """Force our pure-Python flexible_dual_grid_to_mesh over any installed
    o_voxel. The Metal-port o_voxel.convert segfaults on decoder output even
    when it imports cleanly, so we always prefer our stub implementation.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/models/sc_vaes/fdg_vae.py")
    src = read_file(path)

    if "o_voxel_override_convert" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        "from o_voxel.convert import flexible_dual_grid_to_mesh\n",
        "# Use our mesh extraction: upstream o_voxel.convert needs CUDA.\n"
        "# stubs/ is appended (not prepended) so a pip-installed o_voxel still wins\n"
        "# for other submodules like o_voxel.postprocess.\n"
        "import sys, os\n"
        "_stubs = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'stubs')\n"
        "if _stubs not in sys.path:\n"
        "    sys.path.append(_stubs)\n"
        "try:\n"
        "    from o_voxel_override_convert import flexible_dual_grid_to_mesh\n"
        "except ImportError:\n"
        "    try:\n"
        "        from o_voxel.convert import flexible_dual_grid_to_mesh\n"
        "    except (ImportError, RuntimeError):\n"
        "        def flexible_dual_grid_to_mesh(*args, **kwargs):\n"
        '            raise RuntimeError("flexible_dual_grid_to_mesh unavailable")\n',
    )
    write_file(path, src)


def patch_pipeline():
    """Use the XPU cache call."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "torch.xpu.empty_cache()" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        "        torch.cuda.empty_cache()\n",
        "        torch.xpu.empty_cache()\n",
    )
    write_file(path, src)


def patch_num_samples():
    """Give every sample its own copy of the image conditioning.

    Upstream's run() draws `num_samples` noise tensors but conditions on a
    batch of one image, which its sparse cross-attention rejects.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "[image] * num_samples" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        "        cond_512 = self.get_cond([image], 512)\n"
        "        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None\n",
        "        cond_512 = self.get_cond([image] * num_samples, 512)\n"
        "        cond_1024 = self.get_cond([image] * num_samples, 1024) if pipeline_type != '512' else None\n",
    )
    write_file(path, src)


def patch_pipeline_base():
    """Fix hardcoded cuda device in Pipeline.cuda()."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/base.py")
    src = read_file(path)

    if 'torch.device("xpu")' in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = must_replace(src,
        '        self.to(torch.device("cuda"))',
        '        self.to(torch.device("xpu"))',
    )
    write_file(path, src)


def install_conv_backend():
    """Copy the pure-PyTorch sparse convolution backend into place."""
    src = os.path.join(BACKENDS_DIR, "conv_none.py")
    dst = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/conv/conv_none.py")

    if os.path.exists(dst):
        print(f"  Already installed: trellis2/modules/sparse/conv/conv_none.py")
        return

    shutil.copy2(src, dst)
    print(f"  Installed: trellis2/modules/sparse/conv/conv_none.py")


def install_mesh_extract():
    """Copy the pure-Python mesh extraction into the o_voxel stub and also as
    a flat override module. The flat module takes precedence over any
    Metal/CUDA o_voxel package that might be installed alongside us.
    """
    stubs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stubs")
    ovoxel_dir = os.path.join(stubs_dir, "o_voxel")
    os.makedirs(ovoxel_dir, exist_ok=True)

    src = os.path.join(BACKENDS_DIR, "mesh_extract.py")

    # Flat override module — loaded before real o_voxel by fdg_vae patch
    flat_dst = os.path.join(stubs_dir, "o_voxel_override_convert.py")
    shutil.copy2(src, flat_dst)
    print(f"  Installed: stubs/o_voxel_override_convert.py")

    # Also the stub package for environments without any o_voxel install
    dst = os.path.join(ovoxel_dir, "convert.py")
    shutil.copy2(src, dst)
    print(f"  Installed: stubs/o_voxel/convert.py")

    # __init__.py
    with open(os.path.join(ovoxel_dir, "__init__.py"), "w") as f:
        f.write("pass\n")

    # io.py stub
    with open(os.path.join(ovoxel_dir, "io.py"), "w") as f:
        f.write('def read(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n')
        f.write('def write(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n')
        f.write('def read_vxz(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n')

    # rasterize.py stub
    with open(os.path.join(ovoxel_dir, "rasterize.py"), "w") as f:
        f.write('class VoxelRenderer:\n    def __init__(self, *args, **kwargs):\n')
        f.write('        raise RuntimeError("o_voxel.rasterize requires CUDA")\n')


def main():
    print("Applying XPU patches to TRELLIS.2...")
    print(f"  TRELLIS root: {TRELLIS_ROOT}")
    print()

    if not os.path.isdir(TRELLIS_ROOT):
        print(f"Error: TRELLIS.2 not found at {TRELLIS_ROOT}")
        print("The Dockerfile clones it at a pinned commit.")
        return False

    patch_sparse_config()
    patch_sparse_attention()
    patch_image_feature_extractor()
    patch_no_rembg()
    patch_mesh_base()
    patch_fdg_vae()
    patch_pipeline()
    patch_num_samples()
    patch_pipeline_base()
    install_conv_backend()
    install_mesh_extract()

    print()
    print("All patches applied.")
    return True


if __name__ == "__main__":
    main()
