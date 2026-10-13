#!/usr/bin/env python3
"""枚举可用的 hc_pre 实现并对照计时（decode 档 T=48，K=40 批量）。"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
HM, H = 4, 5120
MIX_HC = (HM + 2) * HM
K = 40
T = 48


def bench_batch(call, k=K, warmup=2, iters=15):
    for _ in range(warmup):
        for _ in range(k):
            call()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(k):
            call()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6 / k)
    return statistics.median(ts), min(ts)


x = torch.randn((T, HM, H), dtype=torch.bfloat16, device=DEV) * 0.1
fn_w = torch.randn((MIX_HC, HM * H), dtype=torch.float32, device=DEV) * 0.02
scale = torch.randn((3,), dtype=torch.float32, device=DEV)
base = torch.randn((MIX_HC,), dtype=torch.float32, device=DEV) * 0.1
pre_mix = torch.zeros((T, HM), dtype=torch.float32, device=DEV)

KW = dict(hc_mult=HM, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)

cands = [
    ("_C_ascend.npu_hc_pre_v2", lambda: torch.ops._C_ascend.npu_hc_pre_v2(x, fn_w, scale, base, pre_mix, **KW)),
    ("_C_ascend.npu_hc_pre", lambda: torch.ops._C_ascend.npu_hc_pre(x, fn_w, scale, base, pre_mix, **KW)),
    ("custom.npu_hc_pre", lambda: torch.ops.custom.npu_hc_pre(x, fn_w, scale, base, pre_mix, **KW)),
    ("cann_ops_transformer.mhc_pre_sinkhorn", lambda: torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(
        x, phi=fn_w, scale=scale, base=base, hc_mult=HM, numIters=20, norm_eps=1e-6, hc_eps=1e-6)),
]

print("T=%d, K=%d calls per measurement" % (T, K))
for name, fn in cands:
    try:
        fn()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print("%-42s UNAVAILABLE: %s" % (name, repr(exc)[:110]))
        continue
    med, mn = bench_batch(fn)
    print("%-42s med %7.1f us  min %7.1f us   (x40 = %.2f ms/step)" % (name, med, mn, med * 40 / 1000))
