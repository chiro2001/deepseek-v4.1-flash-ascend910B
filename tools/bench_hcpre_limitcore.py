#!/usr/bin/env python3
"""HcPre 核数预算微基准：不同 limit_core_num 下的单算子设备时长。

背景（实测 2026-10-07 纯 conc=1）：HcPre 单次 33.60 µs，
真正数学仅 6.2%（aic_mac 1.28 + aiv_vec 0.81），≈46% 是 AIV 未归因（同步/等待），
标量 31.5%，搬运 24.5%。hc_fn=[24,20480]、block=24（用满全部 cube 核）。
⇒ 测：少用核是否更快。
"""
import statistics as st
import time

import torch
import torch_npu  # noqa

DEV = "npu:0"
torch.npu.set_device(DEV)

M, HC, H = 6, 4, 5120
NFN, WIDTH = 24, HC * H
dev = torch.device(DEV)

x = torch.randn(M, HC, H, dtype=torch.bfloat16, device=dev) * 0.1
hc_fn = torch.randn(NFN, WIDTH, dtype=torch.float32, device=dev) * 0.01
hc_scale = torch.randn(NFN, dtype=torch.float32, device=dev) * 0.01
hc_base = torch.randn(NFN, dtype=torch.float32, device=dev) * 0.01
pre_mix = torch.rand(M, HC, dtype=torch.float32, device=dev)

OP = torch.ops._C_ascend.npu_hc_pre_v2
KW = dict(hc_mult=HC, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6)


def run_once(aic, aiv):
    if aic <= 0 and aiv <= 0:
        return OP(x, hc_fn, hc_scale, hc_base, pre_mix, **KW)
    from torch.npu.npugraph_ex.scope import limit_core_num
    with limit_core_num(aic or None, aiv or None):
        return OP(x, hc_fn, hc_scale, hc_base, pre_mix, **KW)


def bench(aic, aiv=0, warm=60, reps=200):
    for _ in range(warm):
        run_once(aic, aiv)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_once(aic, aiv)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return st.median(ts), min(ts)


print(f"shapes: x={tuple(x.shape)} hc_fn={tuple(hc_fn.shape)} M={M} (真实 decode 口径)")
print(("{:>14}{:>12}{:>12}{:>10}").format("AIC", "中位us", "最小us", "vs 不限"))
base = None
for aic in [0, 24, 20, 16, 12, 8, 4, 2]:
    med, mn = bench(aic, 0)
    if base is None:
        base = med
    print(("{:>14}{:>12.2f}{:>12.2f}{:>9.2f}x").format(
        "不限" if aic == 0 else aic, med, mn, med / base))
