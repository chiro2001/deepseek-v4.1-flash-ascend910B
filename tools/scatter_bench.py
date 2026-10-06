#!/usr/bin/env python3
"""ScatterNdUpdateSk 微基准 —— 判断 0.72 ms/步的真实暴露是**启动开销**还是**带宽**。

profile 事实（armF_r6_base，tp8-class）：
  · `aclnnScatterNdUpdateSk` 1.180 profile ms = 0.721 真实 ms/步，**58 个/步**，中位 20.4 µs；
  · 每次搬 8 token × 512 dim × 2 B ≈ 8 KB ⇒ 58×8 KB = 464 KB/步，
    而 0.73 ms 内 HBM 可搬 ~860 MB ⇒ **远不是带宽瓶颈**。
⇒ 若微基准显示"小 shape 单次 ~20 µs 且与数据量无关"，则是**启动开销主导**，
   优化方向是"减少调用次数"或"换更轻的算子"，而不是带宽优化。

用法: python3 scatter_bench.py
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
D = 512          # head_dim（含 rope）
PAGE = 128       # block_size
NB = 4096        # cache 页数


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts)


print("=== ScatterNdUpdateSk 单次耗时 vs token 数 ===")
print("var=[%d*%d, %d] bf16; indices=[T,2] int32; updates=[T,%d] bf16" % (NB, PAGE, D, D))
print("%6s %10s %10s %12s %10s" % ("T", "med_us", "min_us", "KB", "GB/s"))
for T in (1, 8, 16, 64, 512):
    var = torch.zeros((NB * PAGE, D), dtype=torch.bfloat16, device=DEV)
    idx = torch.zeros((T, 2), dtype=torch.int32, device=DEV)
    idx[:, 0] = torch.arange(T, dtype=torch.int32, device=DEV)
    idx[:, 1] = 0
    upd = torch.ones((T, D), dtype=torch.bfloat16, device=DEV)

    def call(var=var, idx=idx, upd=upd):
        torch.ops._C_ascend.npu_scatter_nd_update_sk(var, idx, upd)

    med, mn = bench(call)
    kb = T * D * 2 / 1024
    gbs = (kb / 1024 / 1024) / (med / 1e6) if med > 0 else 0
    print("%6d %10.1f %10.1f %12.1f %10.2f" % (T, med, mn, kb, gbs))

print()
print("=== control: torch_npu.npu_scatter_nd_update_ (generic) ===")
print("%6s %10s" % ("T", "med_us"))
for T in (8, 64):
    var = torch.zeros((NB * PAGE, D), dtype=torch.bfloat16, device=DEV)
    rows = torch.arange(T, dtype=torch.int64, device=DEV)
    upd = torch.ones((T, D), dtype=torch.bfloat16, device=DEV)

    def call2(var=var, rows=rows, upd=upd):
        torch_npu.npu_scatter_nd_update_(var, rows.view(-1, 1), upd)

    try:
        med, mn = bench(call2)
        print("%6d %10.1f  (min %.1f)" % (T, med, mn))
    except Exception as exc:  # noqa: BLE001
        print("%6d  FAILED: %r" % (T, exc))

print()
print("=== control: plain indexing (PyTorch semantics, not AscendC) ===")
for T in (8, 64):
    var = torch.zeros((NB * PAGE, D), dtype=torch.bfloat16, device=DEV)
    rows = torch.arange(T, dtype=torch.int64, device=DEV)
    upd = torch.ones((T, D), dtype=torch.bfloat16, device=DEV)

    def call3(var=var, rows=rows, upd=upd):
        var[rows] = upd

    med, mn = bench(call3)
    print("%6d %10.1f  (min %.1f)" % (T, med, mn))

print()
print("=== 58 calls (simulate one step) ===")
T = 8
var = torch.zeros((NB * PAGE, D), dtype=torch.bfloat16, device=DEV)
idx = torch.zeros((T, 2), dtype=torch.int32, device=DEV)
idx[:, 0] = torch.arange(T, dtype=torch.int32, device=DEV)
upd = torch.ones((T, D), dtype=torch.bfloat16, device=DEV)


def loop58():
    for _ in range(58):
        torch.ops._C_ascend.npu_scatter_nd_update_sk(var, idx, upd)


med, mn = bench(loop58, warmup=2, iters=20)
print("58 calls: med %.1f us = %.3f ms/step (profile exposed 0.721 ms)" % (med, med / 1000))
