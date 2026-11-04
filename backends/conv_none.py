"""
Pure-PyTorch sparse 3D convolution backend.

Implements submanifold sparse convolution with one gather, matmul and
index_add per kernel slot. Needs no CUDA extensions: works on XPU and CPU.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor


MAX_ROWS_PER_MATMUL = 1 << 21
RELEASE_CACHE_ABOVE_ROWS = 1 << 22


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), \
        "Naive implementation only supports submanifold sparse convolution (stride=1, padding=None)"

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Match flex_gemm weight layout: (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def _neighbor_map(coords, kernel_size, dilation):
    """[N, V] int32 map: index of the active voxel at each kernel slot, or -1."""
    Kd, Kh, Kw = kernel_size
    reach = torch.tensor(
        [0, (Kd // 2) * dilation[0], (Kh // 2) * dilation[1], (Kw // 2) * dilation[2]],
        device=coords.device,
    )
    # Shift so neighbors stay non-negative; the keys are then unique per position.
    c = coords.long() + reach
    extent = c.amax(dim=0) + reach + 1
    scale = torch.stack([extent[1] * extent[2] * extent[3], extent[2] * extent[3], extent[3], torch.ones_like(extent[3])])
    keys = (c * scale).sum(dim=1)
    sorted_keys, order = torch.sort(keys)

    N = coords.shape[0]
    neighbor = torch.full((N, Kd * Kh * Kw), -1, dtype=torch.int32, device=coords.device)
    for kd in range(Kd):
        for kh in range(Kh):
            for kw in range(Kw):
                offset = (
                    (kd - Kd // 2) * dilation[0] * scale[1]
                    + (kh - Kh // 2) * dilation[1] * scale[2]
                    + (kw - Kw // 2) * dilation[2] * scale[3]
                )
                query = keys + offset
                pos = torch.searchsorted(sorted_keys, query).clamp_max(N - 1)
                hit = sorted_keys[pos] == query
                neighbor[hit, kd * Kh * Kw + kh * Kw + kw] = order[pos[hit]].int()
    return neighbor


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    feats = x.feats
    N = feats.shape[0]

    cache_key = f'SubMConv3d_naive_neighbor_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor = x.get_spatial_cache(cache_key)
    if neighbor is None:
        neighbor = _neighbor_map(x.coords, (Kd, Kh, Kw), self.dilation)
        x.register_spatial_cache(cache_key, neighbor)

    V = Kd * Kh * Kw
    w = self.weight.reshape(Co, V, Ci).permute(1, 2, 0)  # (V, Ci, Co)

    # Freed blocks stay reserved by the allocator and fragment, so a later large
    # tensor can fail with most of the card unused. Give them back first.
    if feats.device.type == "xpu" and N > RELEASE_CACHE_ABOVE_ROWS:
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    # Accumulate in fp32 like FlexGEMM's kernels, even for fp16 inputs.
    out = torch.zeros(N, Co, device=feats.device, dtype=torch.float32)
    on_xpu = feats.device.type == "xpu"
    for v in range(V):
        slot = neighbor[:, v]
        target = (slot >= 0).nonzero().squeeze(1)
        if target.numel() == 0:
            continue
        # Fixed-size chunks keep every allocation small. The matmul on all rows at
        # once needs a single block of several GiB on large meshes.
        for chunk in target.split(MAX_ROWS_PER_MATMUL):
            source = feats.index_select(0, slot[chunk].long())
            out.index_add_(0, chunk, (source @ w[v]).float())
            del source
            # The XPU queue is asynchronous: temporaries stay allocated until it drains.
            if on_xpu:
                torch.xpu.synchronize()

    if self.bias is not None:
        out = out + self.bias.float()

    return x.replace(out.to(feats.dtype))


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")
