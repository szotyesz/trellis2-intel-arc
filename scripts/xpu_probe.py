"""Runtime gate for the B60. Prints JSON lines; see Phase 1 of docs/XPU_PORT_PLAN.md."""
import glob
import json
import re
import threading
import time

import torch
import torch.nn.functional as F

D = "xpu"
GIB = 2**30


def emit(**kw):
    print(json.dumps(kw), flush=True)


class VramPeak:
    """Driver-level VRAM for this process, which torch counters can miss."""

    def __enter__(self):
        self.peak, self.stop = 0, False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()
        return self

    def run(self):
        while not self.stop:
            total = 0
            for fd in glob.glob("/proc/self/fdinfo/*"):
                try:
                    m = re.search(r"drm-total-vram0:\s+(\d+)", open(fd).read())
                    total = max(total, int(m.group(1)) * 1024) if m else total
                except OSError:
                    pass
            self.peak = max(self.peak, total)
            time.sleep(0.05)

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def sdpa_sweep():
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        for L in (4096, 8192, 16384, 32768, 49152):
            try:
                q = torch.randn(1, 12, L, 128, dtype=torch.float32, device=D).to(dt)
                torch.xpu.empty_cache()
                torch.xpu.reset_peak_memory_stats()
                base = torch.xpu.memory_allocated()
                with VramPeak() as v, torch.no_grad():
                    t = time.time()
                    out = F.scaled_dot_product_attention(q, q, q)
                    torch.xpu.synchronize()
                    dt_s = time.time() - t
                emit(test="sdpa", dtype=str(dt), L=L, seconds=round(dt_s, 2),
                     torch_peak_gib=round((torch.xpu.max_memory_allocated() - base) / GIB, 2),
                     driver_peak_gib=round(v.peak / GIB, 2), nan=bool(out.float().isnan().any()))
                del q, out
            except RuntimeError as e:
                emit(test="sdpa", dtype=str(dt), L=L, error=str(e)[:200])
                torch.xpu.empty_cache()
                break


def sdpa_agreement():
    q = torch.randn(1, 12, 1024, 128)
    ref = torch._scaled_dot_product_attention_math(q, q, q)[0]
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        with torch.no_grad():
            out = F.scaled_dot_product_attention(*[q.to(D).to(dt)] * 3).float().cpu()
        emit(test="sdpa_vs_cpu", dtype=str(dt), max_abs_err=float((out - ref).abs().max()),
             nan=bool(out.isnan().any()))


def gemm():
    for dt in (torch.float16, torch.float32):
        for logn in (18, 19, 20, 21, 22):
            a = torch.randn(2**logn, 64)
            b = torch.randn(64, 64)
            ref = (a @ b).float()
            out = (a.to(D).to(dt) @ b.to(D).to(dt)).float().cpu()
            emit(test="gemm", dtype=str(dt), rows=2**logn, max_abs_err=float((out - ref).abs().max()))
            del out


def big_alloc():
    try:
        x = torch.ones(6 * GIB // 4, dtype=torch.float32, device=D)
        x[-1] = 7
        emit(test="alloc_6gib", ok=float(x[-1]) == 7 and float(x[0]) == 1)
        del x
    except RuntimeError as e:
        emit(test="alloc_6gib", ok=False, error=str(e)[:200])


def op(name, fn):
    try:
        fn()
        emit(test="op", name=name, ok=True)
    except Exception as e:  # any failure here is the finding
        emit(test="op", name=name, ok=False, error=str(e)[:200])


def ops():
    c = torch.randint(0, 64, (1000, 4), device=D, dtype=torch.int32)
    k = (c[:, 1] * 4096 + c[:, 2] * 64 + c[:, 3]).long()
    op("polar_complex", lambda: torch.view_as_real(torch.view_as_complex(torch.ones(4, 2, device=D)) * torch.polar(torch.ones(4, device=D), torch.ones(4, device=D))))
    op("sort_searchsorted", lambda: torch.searchsorted(torch.sort(k)[0], k))
    op("unique_dim", lambda: c.unique(dim=0))
    op("argwhere", lambda: torch.argwhere(c > 3))
    op("max_pool3d", lambda: F.max_pool3d(torch.ones(1, 1, 8, 8, 8, device=D), 2, 2, 0))
    for dt in (torch.float16, torch.bfloat16):
        x = torch.ones(1000, 8, device=D, dtype=dt)
        op(f"index_add_{dt}", lambda: torch.zeros(1000, 8, device=D, dtype=dt).index_add_(0, k % 1000, x))
        op(f"matmul_{dt}", lambda: x @ torch.ones(8, 8, device=D, dtype=dt))
        op(f"silu_{dt}", lambda: F.silu(x))
        op(f"conv3d_{dt}", lambda: F.conv3d(torch.ones(1, 8, 8, 8, 8, device=D, dtype=dt), torch.ones(8, 8, 3, 3, 3, device=D, dtype=dt)))


emit(test="device", available=torch.xpu.is_available(), count=torch.xpu.device_count(),
     name=torch.xpu.get_device_name(0), torch=torch.__version__)
p = torch.xpu.get_device_properties(0)
emit(test="properties", **{a: str(getattr(p, a)) for a in dir(p) if not a.startswith("_") and not callable(getattr(p, a))})
emit(test="mem_get_info_gib", free_total=[round(x / GIB, 2) for x in torch.xpu.mem_get_info(0)])
_ = torch.zeros(1, device=D)
emit(test="libsycl", paths=sorted({l.split()[-1] for l in open("/proc/self/maps") if "libsycl" in l}))
q = torch.zeros(1, 12, 8, 128, device=D)
emit(test="sdp_choice", values={str(t): torch._fused_sdp_choice(*[q.to(t)] * 3) for t in (torch.bfloat16, torch.float16, torch.float32)},
     backends={k: int(v) for k, v in torch.backends.cuda.SDPBackend.__members__.items()})
ops()
big_alloc()
sdpa_agreement()
sdpa_sweep()
gemm()
