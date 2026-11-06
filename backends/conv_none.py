"""
Sparse 3D convolution backend for XPU and CPU. No CUDA extensions.

Submanifold sparse convolution, as FlexGEMM computes it: the output row of
each active voxel is the sum over kernel slots of (feature row of the active
neighbour at that slot) @ (weight of that slot).

Two implementations:
  - A Triton kernel that gathers neighbour rows and multiplies them in one
    pass, used on XPU for fp16 and bf16. It needs Triton's JIT to work, so the
    image carries g++ and the Level Zero headers.
  - A plain PyTorch loop over kernel slots with an index_add per slot, used on
    CPU, for fp32, or when Triton is unavailable (TRELLIS_SPARSE_CONV_TRITON=0
    forces it).
"""

import math
import os
import time

import torch
import torch.nn as nn
from .. import SparseTensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments
    triton = None


MAX_CHUNK_BYTES = 1 << 28
RELEASE_CACHE_ABOVE_ROWS = 1 << 22
TRITON_ROWS = 64        # output rows per program
TRITON_K = 32           # input channels per inner step
# Below this many rows the GPU cannot be filled and oneDNN's GEMMs in the loop
# win: 2,048 rows at 1024 channels took 0.5 s per call in Triton, 0.04 s looped.
TRITON_MIN_ROWS = 16384
_triton_state = {"enabled": None}

# TRELLIS_CONV_PROFILE=1 prints time per (rows, Ci, Co, dtype, path) at exit.
_profile = {} if os.environ.get("TRELLIS_CONV_PROFILE") == "1" else None
if _profile is not None:
    import atexit

    def _print_profile():
        print("[SPARSE] convolution profile: rows, Ci, Co, dtype, path: calls, total s")
        for key, (calls, secs) in sorted(_profile.items(), key=lambda kv: -kv[1][1]):
            print(f"  {key[0]:>10,} {key[1]:>5} {key[2]:>5} {str(key[3])[6:]:>9} {key[4]:>6}: {calls:>4} {secs:8.2f}")

    atexit.register(_print_profile)


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


if triton is not None:

    @triton.jit
    def _gather_conv_kernel(feats, neighbor, weight, bias, out, N, Ci, Co,
                            V: tl.constexpr, BCO: tl.constexpr, HAS_BIAS: tl.constexpr,
                            B: tl.constexpr, BK: tl.constexpr):
        """out[rows] = sum_v feats[neighbor[rows, v]] @ weight[v], fp32 accumulation.

        One program handles B output rows and BCO output channels. Slots with
        no neighbour (-1) load zeros. Row offsets are 64-bit: N * Ci exceeds
        2^31 on large meshes.

        Ci and Co are runtime values on purpose: the decoders use about twenty
        channel configurations, and compiling and tuning a kernel per
        configuration cost more than the loop it replaced.
        """
        rows = (tl.program_id(0) * B + tl.arange(0, B)).to(tl.int64)
        rmask = rows < N
        co = tl.program_id(1) * BCO + tl.arange(0, BCO)
        cmask = co < Co
        acc = tl.zeros((B, BCO), dtype=tl.float32)
        for v in range(V):
            src = tl.load(neighbor + rows * V + v, mask=rmask, other=-1).to(tl.int64)
            smask = rmask & (src >= 0)
            for k0 in range(0, Ci, BK):
                ci = k0 + tl.arange(0, BK)
                kmask = ci < Ci
                x = tl.load(feats + src[:, None] * Ci + ci[None, :], mask=smask[:, None] & kmask[None, :], other=0.0)
                w = tl.load(weight + (v * Ci + ci[:, None]) * Co + co[None, :], mask=kmask[:, None] & cmask[None, :], other=0.0)
                acc = tl.dot(x, w, acc)
        if HAS_BIAS:
            acc += tl.load(bias + co, mask=cmask, other=0.0)[None, :]
        tl.store(out + rows[:, None] * Co + co[None, :], acc.to(out.dtype.element_ty), mask=rmask[:, None] & cmask[None, :])


def triton_available():
    """Triton on XPU, unless disabled. A failed first launch disables it too."""
    if _triton_state["enabled"] is None:
        _triton_state["enabled"] = (
            triton is not None
            and os.environ.get("TRELLIS_SPARSE_CONV_TRITON", "1") != "0"
            and torch.xpu.is_available()
        )
    return _triton_state["enabled"]


def _triton_forward(self, feats, neighbor):
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw
    N = feats.shape[0]
    feats = feats.contiguous()
    # Weight as (V, Ci, Co) in the activation dtype, prepared once per layer.
    key = (self.weight.data_ptr(), self.weight._version, feats.dtype, feats.device)
    cache = getattr(self, "_xpu_conv_cache", None)
    if cache is None or cache[0] != key:
        w = self.weight.detach().reshape(Co, V, Ci).permute(1, 2, 0).contiguous().to(feats.dtype)
        b = self.bias.detach().float().contiguous() if self.bias is not None else w
        self._xpu_conv_cache = cache = (key, w, b)
    _, w, b = cache
    out = torch.empty(N, Co, device=feats.device, dtype=feats.dtype)
    BCO = max(16, min(128, triton.next_power_of_2(Co)))
    grid = (triton.cdiv(N, TRITON_ROWS), triton.cdiv(Co, BCO))
    _gather_conv_kernel[grid](feats, neighbor, w, b, out, N, Ci, Co, V=V, BCO=BCO, HAS_BIAS=self.bias is not None,
                              B=TRITON_ROWS, BK=TRITON_K, num_warps=4)
    return out


def _loop_forward(self, feats, neighbor):
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw
    N = feats.shape[0]
    w = self.weight.reshape(Co, V, Ci).permute(1, 2, 0)  # (V, Ci, Co)

    # Accumulate each chunk in fp32 like FlexGEMM's kernels, even for fp16 inputs.
    # Store at the input precision: an fp32 output on an fp16 network costs twice
    # the memory for the whole tensor.
    out = torch.empty(N, Co, device=feats.device, dtype=feats.dtype)
    on_xpu = feats.device.type == "xpu"
    bias = self.bias.float() if self.bias is not None else None
    # Small chunks keep every allocation small. index_add_ needs a temporary the
    # size of its target on XPU, so it must never run on the full output.
    chunk_rows = max(1, MAX_CHUNK_BYTES // (4 * max(Co, Ci)))
    for start in range(0, N, chunk_rows):
        rows = neighbor[start:start + chunk_rows]
        acc = torch.zeros(rows.shape[0], Co, device=feats.device, dtype=torch.float32)
        for v in range(V):
            slot = rows[:, v]
            target = (slot >= 0).nonzero().squeeze(1)
            if target.numel() > 0:
                source = feats.index_select(0, slot[target].long())
                acc.index_add_(0, target, (source @ w[v]).float())
        if bias is not None:
            acc += bias
        out[start:start + rows.shape[0]] = acc.to(feats.dtype)
        del acc
        # The XPU queue is asynchronous: temporaries stay allocated until it drains.
        if on_xpu:
            torch.xpu.synchronize()
    return out


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    feats = x.feats
    N = feats.shape[0]

    cache_key = f'SubMConv3d_naive_neighbor_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor = x.get_spatial_cache(cache_key)
    if neighbor is None:
        neighbor = _neighbor_map(x.coords, (Kd, Kh, Kw), self.dilation)
        x.register_spatial_cache(cache_key, neighbor)

    # Freed blocks stay reserved by the allocator and fragment, so a later large
    # tensor can fail with most of the card unused. Give them back first.
    if feats.device.type == "xpu" and N > RELEASE_CACHE_ABOVE_ROWS:
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    path = "loop"
    if _profile is not None:
        torch.xpu.synchronize() if feats.device.type == "xpu" else None
        t0 = time.time()
    out = None
    if (feats.device.type == "xpu" and feats.dtype in (torch.float16, torch.bfloat16)
            and N >= TRITON_MIN_ROWS and triton_available()):
        try:
            out = _triton_forward(self, feats, neighbor)
            path = "triton"
        except Exception as e:  # JIT or launch failure: keep going on the slow path
            _triton_state["enabled"] = False
            print(f"[SPARSE] Triton convolution failed, using the torch loop from now on: {str(e)[:200]}")
    if out is None:
        out = _loop_forward(self, feats, neighbor)
    if _profile is not None:
        torch.xpu.synchronize() if feats.device.type == "xpu" else None
        key = (1 << max(0, N.bit_length() - 1), Ci, Co, feats.dtype, path)   # rows bucketed by power of two
        calls, secs = _profile.get(key, (0, 0.0))
        _profile[key] = (calls + 1, secs + time.time() - t0)
    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")
