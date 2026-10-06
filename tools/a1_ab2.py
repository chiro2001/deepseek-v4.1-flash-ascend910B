#!/usr/bin/env python3
"""A1 忠实 A/B（用两张图并发重放，复现已验证可行的模式）。

  A 现状：g_kv（kv_matmul 满核 → kv尾AIV）串行于 g_qb（q_b_matmul 满核）
  B 控核：g_kv（kv_matmul x 核 → kv尾AIV）‖ g_qb（q_b_matmul 24-x 核）
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
main = torch.npu.current_stream()
aux = torch.npu.Stream()
qst = torch.npu.Stream()   # q_b 也用非默认流（捕获要求）
lc = torch.npu.npugraph_ex.scope.limit_core_num

M = int(os.environ.get("M", "6"))
QL, QB_OUT = 1280, 4096
KV_IN, KV_OUT = 5120, 512
INNER = int(os.environ.get("INNER", "40"))     # 图内重复 40 次 ≈ 40 层

Wqb = torch.randn(QL, QB_OUT, dtype=torch.bfloat16, device="npu:0")
Wkv = torch.randn(KV_IN, KV_OUT, dtype=torch.bfloat16, device="npu:0")
xq = torch.randn(M, QL, dtype=torch.bfloat16, device="npu:0")
xk = torch.randn(M, KV_IN, dtype=torch.bfloat16, device="npu:0")
ta = torch.randn(M, KV_OUT, dtype=torch.float32, device="npu:0")
tb = torch.randn(M, KV_OUT, dtype=torch.float32, device="npu:0")


def cap_kv(n, x):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=aux):
        for _ in range(n):
            with lc(x, 48, aux):
                kv = xk @ Wkv
            torch.add(ta, tb, out=ta)          # kv_norm + rope + scatter 的 AIV 替代
    return g


def cap_qb(n, x):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=qst):
        for _ in range(n):
            with lc(x, 48, qst):
                q = xq @ Wqb
    return g


def timed(fn, n=15, warm=3):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3      # ms


# A 现状：串行（kv 全跑完，再跑 q_b）
gkv = cap_kv(INNER, 24)
gqb = cap_qb(INNER, 24)


def arm_serial():
    with torch.npu.stream(aux):
        gkv.replay()
    main.wait_stream(aux)
    with torch.npu.stream(qst):
        gqb.replay()
    main.wait_stream(qst)


tA = timed(arm_serial)
print("INNER=%d（≈%d 层）" % (INNER, INNER))
print("  A 现状（串行）            : %7.3f ms" % tA)

print("\n  逐档控核并发：")
best = None
for x in (20, 18, 16, 14, 12, 10, 8, 6, 4):
    gk = cap_kv(INNER, x)
    gq = cap_qb(INNER, 24 - x)

    def arm_par(gk=gk, gq=gq):
        with torch.npu.stream(aux):
            gk.replay()
        with torch.npu.stream(qst):
            gq.replay()
        main.wait_stream(aux); main.wait_stream(qst)

    t = timed(arm_par)
    mark = ""
    if best is None or t < best[1]:
        best = (x, t); mark = "  ← 最优"
    print("    kv=%2d, qb=%2d           : %7.3f ms  (%+.1f%%)%s" % (x, 24 - x, t, 100 * (tA / t - 1), mark))

print("\n  ⇒ 最优 kv=%d 核：省 %.3f ms / %d 层 = %.2f µs/层" % (best[0], tA - best[1], INNER, (tA - best[1]) * 1000 / INNER))
print("     折算到每步（40 层 MLA prolog）: %.3f ms ⇒ **%+.1f%%**（步长 24.59 ms）"
      % ((tA - best[1]) / INNER * 40, 100 * (24.59 / (24.59 - (tA - best[1]) / INNER * 40) - 1)))
