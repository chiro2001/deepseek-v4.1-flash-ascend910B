#!/usr/bin/env python3
"""pingpong 验证 v2：用**延迟受限的小算子**（贴近真实 profile）。

真实 profile 的关键特征：除 MoE GEMM（683 GB/s，58% 可达）外，
其余 AIC 工作只有 137 GB/s（12% 可达）⇒ 大量时间在等依赖/启动，不是等带宽。

Arm A：1 条流，2L 次 [AIC ; AIV]
Arm B：2 条流，各 L 次 [AIC ; AIV]，交错提交（两个 micro-batch 的数据独立）
Arm C：4 条流，各 L/2 次
总工作量相同。若"两条独立链能让硬件把 AIV 塞进 AIC 空档"，B 应显著快于 A。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
L = int(os.environ.get("L", "80"))
M = int(os.environ.get("M", "6"))
K, N = 5120, 2048                    # 权重 21 MB

W = torch.randn(K, N, dtype=torch.bfloat16, device="npu:0")
xs = [torch.randn(M, K, dtype=torch.bfloat16, device="npu:0") for _ in range(4)]
vs = [(torch.randn(M * K, dtype=torch.float32, device="npu:0"),
       torch.randn(M * K, dtype=torch.float32, device="npu:0")) for _ in range(4)]


def aic(i):  return xs[i] @ W
def aiv(i):  return torch.add(vs[i][0], vs[i][1], out=vs[i][0])


def bench(fn, n=5, warm=2):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


tg = bench(lambda: aic(0))
tv = bench(lambda: aiv(0))
print("单算子：AIC %.1f µs | AIV %.1f µs  ⇒ AIC/AIV = %.2f" % (tg*1e6, tv*1e6, tg/tv))


def arm_a():
    for _ in range(2 * L):
        aic(0); aiv(0)


def arm_multi(ns):
    streams = [torch.npu.Stream() for _ in range(ns)]
    main = torch.npu.current_stream()
    ev = torch.npu.Event()
    ev.record(main)
    for s in streams: s.wait_event(ev)
    per = (2 * L) // ns
    # 交错提交：每轮给每条流提交一个 [AIC;AIV]
    for _ in range(per):
        for i in range(ns):
            with torch.npu.stream(streams[i]):
                aic(i); aiv(i)
        aic(0); aiv(0)          # 主流也占一份
    for s in streams:
        main.wait_stream(s)


def arm_b(): arm_multi(1)
def arm_c(): arm_multi(3)


tA = bench(arm_a)
tB = bench(arm_b)
tC = bench(arm_c)
print("\n=== 结果（2L=%d 层，M=%d）===" % (2 * L, M))
print("  A 单流（现状）        : %8.3f ms" % (tA*1e3))
print("  B 双流 pingpong       : %8.3f ms   (%.2f× vs A)" % (tB*1e3, tA/tB))
print("  C 四流               : %8.3f ms   (%.2f× vs A)" % (tC*1e3, tA/tC))
print("\n  若相位完全重叠，下界 ≈ max(2L·AIC, 2L·AIV) = %.3f ms（%.2f× 上限）"
      % (max(2*L*tg, 2*L*tv)*1e3, tA/max(2*L*tg, 2*L*tv)))
