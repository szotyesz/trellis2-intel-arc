"""GPU-only checks. Skipped where no XPU is visible, such as the cpu-tests service."""
import os
import sys
import time
from pathlib import Path

import pytest
import torch

os.environ.update(ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa", SPARSE_CONV_BACKEND="none")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "TRELLIS.2")]

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="needs an XPU")


def _random_sparse(n_points, channels, dtype, seed=0):
    from trellis2.modules.sparse import SparseTensor

    g = torch.Generator().manual_seed(seed)
    coords = torch.unique(torch.randint(0, 48, (n_points, 3), generator=g), dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=coords.dtype), coords], dim=1).int()
    feats = torch.randn(len(coords), channels, generator=g).to(dtype)
    return SparseTensor(feats=feats.to("xpu"), coords=coords.to("xpu"))


class _Conv:
    def __init__(self, ci, co, dtype, bias=True):
        g = torch.Generator().manual_seed(1)
        self.weight = (torch.randn(co, 3, 3, 3, ci, generator=g) / ci**0.5).to(dtype).to("xpu")
        self.bias = torch.randn(co, generator=g).to(dtype).to("xpu") if bias else None
        self.dilation = (1, 1, 1)


@pytest.mark.parametrize("ci,co,dtype", [(64, 64, torch.float16), (128, 32, torch.float16), (32, 96, torch.bfloat16)])
def test_triton_conv_matches_torch_loop(ci, co, dtype):
    from trellis2.modules.sparse.conv import conv_none

    assert conv_none.triton_available(), "Triton should work on this GPU"
    x = _random_sparse(40_000, ci, dtype)
    conv = _Conv(ci, co, dtype, bias=True)
    fast = conv_none.sparse_conv3d_forward(conv, x).feats.float()
    slow = conv_none._loop_forward(conv, x.feats, conv_none._neighbor_map(x.coords, (3, 3, 3), conv.dilation)).float()
    torch.xpu.synchronize()
    # Both paths accumulate in fp32 and round once to the activation dtype, in a
    # different summation order, so they may land one ulp apart.
    eps = torch.finfo(dtype).eps
    diff = (fast - slow).abs()
    assert diff.max().item() <= 2 * eps * slow.abs().max().item() + 1e-3
    assert diff.mean().item() <= eps * slow.abs().mean().item()


def test_triton_conv_is_faster_than_loop():
    from trellis2.modules.sparse.conv import conv_none

    x = _random_sparse(400_000, 64, torch.float16)
    conv = _Conv(64, 64, torch.float16)
    neighbor = conv_none._neighbor_map(x.coords, (3, 3, 3), conv.dilation)
    for fn in (lambda: conv_none.sparse_conv3d_forward(conv, x), lambda: conv_none._loop_forward(conv, x.feats, neighbor)):
        fn()                     # warm up, compile
    torch.xpu.synchronize()
    times = []
    for fn in (lambda: conv_none.sparse_conv3d_forward(conv, x), lambda: conv_none._loop_forward(conv, x.feats, neighbor)):
        t = time.time()
        for _ in range(3):
            fn()
        torch.xpu.synchronize()
        times.append((time.time() - t) / 3)
    print(f"triton {times[0] * 1000:.1f} ms, loop {times[1] * 1000:.1f} ms")
    assert times[0] < times[1]
