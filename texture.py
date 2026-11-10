"""
Texture an existing mesh from an RGBA image using TRELLIS.2 on an Intel GPU.

The port of upstream's Trellis2TexturingPipeline: the mesh is voxelised and
encoded to a shape latent, the texture model paints that latent to match the
image, and the result is baked into a texture for the mesh. The mesh keeps
its vertices, faces, size and position.
"""

import argparse
import os
import sys
import time
import traceback

from generate import ROOT, exit_after_gpu_error, free_ram, log_stages, rss_gib  # sets up the environment first

import numpy as np
import torch
from PIL import Image

from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import samplers
from trellis2.pipelines.base import Pipeline
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

sys.path.insert(0, ROOT)
from backends.dual_grid import mesh_to_flexible_dual_grid  # noqa: E402
from backends.texture_baker import ALPHA_MODES, bake_texture, export_glb_with_texture, uv_unwrap  # noqa: E402


class TexturingPipeline(Trellis2ImageTo3DPipeline):
    """Upstream's texturing pipeline without its CUDA extensions.

    Image preprocessing, conditioning and texture sampling are the
    image-to-3D pipeline's own methods. The voxeliser is
    backends/dual_grid.py and the baker is backends/texture_baker.py.
    """

    @classmethod
    def from_pretrained(cls, path, resolution):
        cls.model_names_to_load = ["shape_slat_encoder", "tex_slat_decoder", f"tex_slat_flow_model_{resolution}"]
        pipeline = Pipeline.from_pretrained.__func__(cls, path, "texturing_pipeline.json")
        args = pipeline._pretrained_args
        pipeline.__init__(
            pipeline.models,
            tex_slat_sampler=getattr(samplers, args["tex_slat_sampler"]["name"])(**args["tex_slat_sampler"]["args"]),
            tex_slat_sampler_params=args["tex_slat_sampler"]["params"],
            shape_slat_normalization=args["shape_slat_normalization"],
            tex_slat_normalization=args["tex_slat_normalization"],
            low_vram=args.get("low_vram", True),
        )
        from trellis2.modules import image_feature_extractor
        cond_model = args["image_cond_model"]
        pipeline.image_cond_model = getattr(image_feature_extractor, cond_model["name"])(**cond_model["args"])
        return pipeline

    def encode_shape_slat(self, vertices, faces, resolution):
        """Voxelise a mesh in [-0.5, 0.5] on the CPU and encode it on the GPU."""
        t0 = time.time()
        coords, dual, intersected = mesh_to_flexible_dual_grid(
            vertices, faces, grid_size=resolution, face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
        print(f"Voxelised to {len(coords):,} voxels at {resolution} in {time.time() - t0:.0f}s")
        coords, dual = torch.from_numpy(coords), torch.from_numpy(dual)
        offsets = SparseTensor(
            feats=dual * resolution - coords,
            coords=torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1),
        ).to(self.device)
        flags = offsets.replace(torch.from_numpy(intersected).to(self.device))
        encoder = self.models["shape_slat_encoder"]
        encoder.to(self.device)
        slat = encoder(offsets, flags)
        encoder.cpu()
        return slat

    def decode_tex_slat(self, slat):
        """Decode onto the voxels of the mesh, which the encoder remembered."""
        decoder = self.models["tex_slat_decoder"]
        decoder.to(self.device)
        voxels = decoder(slat) * 0.5 + 0.5
        decoder.cpu()
        return voxels


def normalise(vertices, z_up):
    """Centre the mesh, scale it into the unit cube and turn it to Z up."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    v = (vertices - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    if not z_up:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return v


def usable_uv(mesh):
    """The mesh's own UVs in texture-row space, if they fit one texture."""
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return None
    uv = np.asarray(uv, dtype=np.float32)
    if not np.isfinite(uv).all() or uv.min() < -1e-4 or uv.max() > 1 + 1e-4:
        return None     # tiled or mirrored: texels would be shared
    return np.stack([uv[:, 0], 1 - uv[:, 1]], axis=1).clip(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", help="Mesh to texture: GLB, OBJ, PLY or anything else trimesh reads")
    parser.add_argument("image", help="Image of the object with a transparent background")
    parser.add_argument("--output", default="/out/textured", help="Output path without extension")
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 1024],
                        help="Voxel grid the mesh is encoded at")
    parser.add_argument("--texture-size", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps")
    parser.add_argument("--z-up", action="store_true",
                        help="The mesh's up axis is Z. The default is Y, as in glTF and as upstream assumes")
    parser.add_argument("--unwrap", action="store_true",
                        help="Make new UVs with xatlas even if the mesh has its own")
    parser.add_argument("--alpha-mode", default="auto", choices=ALPHA_MODES,
                        help="As in generate.py: how the GLB uses the predicted alpha")
    args = parser.parse_args()

    import trimesh

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces):
        parser.error(f"{args.mesh} has no triangles")
    print(f"Mesh: {len(vertices):,} vertices, {len(faces):,} triangles")
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    unit = normalise(vertices, args.z_up)

    t0 = time.time()
    pipeline = TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", args.resolution)
    pipeline.to(torch.device("xpu"))
    print(f"Loaded in {time.time() - t0:.0f}s on {torch.xpu.get_device_name(0)}, RAM {rss_gib():.1f} GiB")
    log_stages(pipeline, ("get_cond", "encode_shape_slat", "sample_tex_slat", "decode_tex_slat"))

    t0 = time.time()
    try:
        with torch.no_grad():
            image = pipeline.preprocess_image(Image.open(args.image))
            torch.manual_seed(args.seed)
            cond = pipeline.get_cond([image], args.resolution)
            pipeline.image_cond_model = None
            shape_slat = pipeline.encode_shape_slat(unit, faces, args.resolution)
            tex_slat = pipeline.sample_tex_slat(cond, pipeline.models[f"tex_slat_flow_model_{args.resolution}"],
                                                shape_slat, {"steps": args.steps} if args.steps else {})
            pipeline.models.pop(f"tex_slat_flow_model_{args.resolution}")
            voxels = pipeline.decode_tex_slat(tex_slat)
            coords, attrs = voxels.coords[:, 1:].cpu().numpy(), voxels.feats.float().cpu().numpy()
    except RuntimeError:
        traceback.print_exc()
        exit_after_gpu_error()
    print(f"Generated in {time.time() - t0:.1f}s, {len(coords):,} textured voxels")
    del pipeline, shape_slat, tex_slat, voxels, cond
    free_ram()

    t0 = time.time()
    uv = None if args.unwrap else usable_uv(mesh)
    if uv is None:
        # xatlas splits vertices along seams; `vmapping` leads back to the originals.
        _, faces, uv, vmapping = uv_unwrap(unit, faces)
        vertices, unit, normals = vertices[vmapping], unit[vmapping], normals[vmapping]
        print(f"Unwrapped in {time.time() - t0:.0f}s")
    else:
        print("Keeping the mesh's own UVs")
    t0 = time.time()
    bake = bake_texture(unit.astype(np.float32), faces, uv, coords, attrs, np.full(3, -0.5), 1 / args.resolution,
                        texture_size=args.texture_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # The original vertices go out, so the mesh keeps its size, place and axes.
    mode = export_glb_with_texture(vertices, faces, uv, bake.base_color, bake.metallic_roughness, f"{args.output}.glb",
                                   normals=normals, alpha_img=bake.alpha, face_alpha=bake.face_alpha,
                                   alpha_mode=args.alpha_mode, z_up=False)
    print(f"Baked in {time.time() - t0:.0f}s: {args.output}.glb, alpha {mode}, RAM {rss_gib():.1f} GiB")


if __name__ == "__main__":
    main()
