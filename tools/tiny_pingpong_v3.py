#!/usr/bin/env python3
"""pingpong 验证 v3：用 **NPUGraph 捕获**消除 host 提交开销（真实 decode 也是图捕获）。

Arm A：一张图，含 2L 次 [AIC ; AIV]，单流重放
Arm B：两张图，各含 L 次 [AIC ; AIV]，**两条流并发重放**（= 两个 micro-batch）
Arm C：四张图，各含 L/2 次，四条流并发
总工作量相同，且都没有 host 提交开销。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
L = int(os.environ.get("L", "80"))
M = int(os.environ.get("M", "6"))
K, N = 5120, 2048
REPLAY = int(os.environ.get("REPLAY", "30"))

W = torch.randn(K, N, dtype=torch.bfloat16, device="npu:0")
xs = [torch.randn(M, K, dtype=torch.bfloat16, device="npu:0") for _ in range(4)]
vs = [(torch.randn(M * K, dtype=torch.float32, device="npu:0"),
       torch.randn(M * K, dtype=torch.float32, device="npu:0")) for _ in range(4)]


def aic(i): xs[i] @ W
def aiv(i): torch.add(vs[i][0], vs[i][1], out=vs[i][0])


def capture(n_iter, i, stream):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        for _ in range(n_iter):
            aic(i); aiv(i)
    return g


def timed(fn, n=10, warm=2):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


# 单算子设备侧耗时（用图内 500 次摊销）
def dev_time(op, n=500):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(n):
            op()
    return timed(lambda: g.replay(), 10) / n


tg = dev_time(lambda: aic(0))
tv = dev_time(lambda: aiv(0))
print("设备侧单算子：AIC %.2f µs | AIV %.2f µs | 合计 %.2f µs" % (tg*1e6, tv*1e6, (tg+tv)*1e6))

s1 = torch.npu.Stream(); s2 = torch.npu.Stream()
s3 = torch.npu.Stream(); s4 = torch.npu.Stream()
gA = capture(2 * L, 0, torch.npu.Stream())
gB1 = capture(L, 1, s1); gB2 = capture(L, 2, s2)
gC1 = capture(L // 2, 1, s1); gC2 = capture(L // 2, 2, s2)
gC3 = capture(L // 2, 3, s3); gC4 = capture(L // 2 % 4, 0, s4)

main = torch.npu.current_stream()


def arm_a():
    gA.replay()


def arm_b():
    with torch.npu.stream(s1): gB1.replay()
    with torch.npu.stream(s2): gB2.replay()
    main.wait_stream(s1); main.wait_stream(s2)


def arm_c():
    for s, g in ((s1, gC1), (s2, gC2), (s3, gC3), (s4, gC4)):
        with torch.npu.stream(s): g.replay()
    for s in (s1, s2, s3, s4): main.wait_stream(s)


tA = timed(arm_a); tB = timed(arm_b); tC = timed(arm_c)
print("\n=== 结果（总计 %d 层，M=%d，无 host 开销）===" % (2 * L, M))
print("  A 单流（现状）       : %8.3f ms" % (tA*1e3))
print("  B 双流 pingpong      : %8.3f ms  (%.2f× vs A)" % (tB*1e3, tA/tB))
print("  C 四流              : %8.3f ms  (%.2f× vs A)" % (tC*1e3, tA/tC))
print("\n  串行下界 = 2L·(AIC+AIV) = %.3f ms" % (2*L*(tg+tv)*1e3))
print("  完美重叠 = 2L·max(AIC,AIV) = %.3f ms ⇒ 上限 %.2f×" % (
    max(2*L*tg, 2*L*tv)*1e3, (tg+tv)/max(tg,tv)))
