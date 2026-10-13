#!/usr/bin/env python3
"""pingpong 流数扫描：把 AIC:AIV 比例调成我们真实的 1.49（实测 22.05:14.75）。

Arm k：k 条独立链（= k 个 micro-batch），每条含 (2L/k) 次 [AIC;AIV]，图捕获后并发重放。
总 token-层数相同、无 host 开销。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
TOTAL = int(os.environ.get("TOTAL", "160"))   # 总"层"数
M = int(os.environ.get("M", "6"))
K, N = 5120, 2048
AIV_SCALE = float(os.environ.get("AIV_SCALE", "4"))   # 放大 AIV 工作量以贴近真实比例
NS = [int(x) for x in os.environ.get("NS", "1,2,3,4,6,8").split(",")]

W = torch.randn(K, N, dtype=torch.bfloat16, device="npu:0")
VW = int(M * K)
NAIV = int(os.environ.get("NAIV", "3"))   # 每个 AIC 后跟几个小 AIV（真实是 2082 AIV : 609 AIC）
xs = [torch.randn(M, K, dtype=torch.bfloat16, device="npu:0") for _ in range(16)]
vs = [(torch.randn(VW, dtype=torch.float32, device="npu:0"),
       torch.randn(VW, dtype=torch.float32, device="npu:0")) for _ in range(16)]
streams = [torch.npu.Stream() for _ in range(17)]


def aic(i): xs[i] @ W
def aiv(i):
    for _ in range(NAIV):
        torch.add(vs[i][0], vs[i][1], out=vs[i][0])


def dev_time(op, n=400):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(n):
            op()
    for _ in range(2): g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / 10 / n


tg = dev_time(lambda: aic(0)); tv = dev_time(lambda: aiv(0))
print("设备侧：AIC %.2f µs | AIV %.2f µs | 比例 %.2f（真实 1.49）" % (tg*1e6, tv*1e6, tg/tv))
print("串行下界 %.3f ms | 完美重叠 %.3f ms ⇒ 上限 %.2f×\n"
      % (TOTAL*(tg+tv)*1e3, TOTAL*max(tg, tv)*1e3, (tg+tv)/max(tg, tv)))


def timed(fn, n=15, warm=3):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


main = torch.npu.current_stream()
print("%6s %12s %10s %s" % ("流数", "时间ms", "vs 1流", "每条流层数"))
base = None
for k in NS:
    per = TOTAL // k
    graphs = []
    for i in range(k):
        g = torch.npu.NPUGraph()
        st = streams[i]
        with torch.npu.graph(g, stream=st):
            for _ in range(per):
                aic(i); aiv(i)
        graphs.append((g, st))

    def run():
        for g, st in graphs:
            with torch.npu.stream(st):
                g.replay()
        for g, st in graphs:
            main.wait_stream(st)

    t = timed(run)
    if base is None: base = t
    print("%6d %12.3f %9.2f× %d" % (k, t*1e3, base/t, per))
