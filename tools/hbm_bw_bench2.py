#!/usr/bin/env python3
"""更严格的 HBM 读带宽上限（第一版可能被归约逻辑限制）。

第一版用 `x.sum()` 测纯读，得 1299 GB/s。但随后出现矛盾：
MoE 每步 ≈9.6 GB / 4.5 ms = 2133 GB/s ⇒ 若峰值真是 1299，物理上不可能。
⇒ 说明 `sum()` 不是好的带宽探针（归约有额外开销/不是纯流式）。

这里换三种更"纯搬运"的口径：
  1) 大矩阵 × 小向量（GEMV）：读 A[8192×8192] = 134 MB，只写 16 KB ⇒ 近乎纯读
  2) 广播加（y = x + 标量，写回小张量）：避开归约树
  3) `x.max()`（另一种归约，做交叉验证）
并做尺寸扫描，确认是否饱和。

用法: hbm_bw2.py
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


print("=== 1) GEMV：A[N,N] @ v[N]  ⇒ 读 N*N*2 B，写 N*2 B ===")
print("%8s %14s %12s %12s %14s" % ("N", "读入GB", "中位ms", "最小ms", "GB/s(中位)"))
for N in (4096, 8192, 12288, 16384):
    a = torch.randn((N, N), dtype=torch.bfloat16, device=DEV) * 0.01
    v = torch.randn((N, 1), dtype=torch.bfloat16, device=DEV) * 0.01
    nb = a.numel() * a.element_size()

    def call(a=a, v=v):
        return torch.mm(a, v)

    med, mn = bench(call)
    print("%8d %14.3f %12.3f %12.3f %14.1f"
          % (N, nb / 1e9, med * 1e3, mn * 1e3, nb / med / 1e9))
    del a, v
    torch.npu.empty_cache()

print()
print("=== 2) 广播加：y[i] = x[i] + 1.0 —— 读+写各一份 ===")
print("%8s %14s %12s %14s %14s" % ("元素", "张量GB", "中位ms", "有效GB/s", "合计GB/s"))
for ne in (1 << 26, 1 << 27, 1 << 28):
    x = torch.ones(ne, dtype=torch.bfloat16, device=DEV)
    y = torch.empty_like(x)
    nb = ne * 2

    def call(x=x, y=y):
        torch.add(x, 1.0, out=y)

    med, mn = bench(call)
    print("%8d %14.3f %12.3f %14.1f %14.1f"
          % (ne, nb / 1e9, med * 1e3, nb / med / 1e9, 2 * nb / med / 1e9))
    del x, y
    torch.npu.empty_cache()

print()
print("=== 3) max 归约（与 sum 交叉验证）===")
for ne in (1 << 27, 1 << 28):
    x = torch.randn(ne, dtype=torch.bfloat16, device=DEV)
    nb = ne * 2
    med, mn = bench(lambda x=x: x.max())
    print("  %10d  %8.3f GB  中位 %7.3f ms  ⇒ %8.1f GB/s" % (ne, nb / 1e9, med * 1e3, nb / med / 1e9))
    del x
    torch.npu.empty_cache()

print()
print("=== 4) 同规模 GEMV vs 分组 GMM 口径对照 ===")
N, K, M = 8192, 5120, 4608
w = (torch.randn((N, K), dtype=torch.bfloat16, device=DEV) * 0.01)
x1 = torch.randn((1, K), dtype=torch.bfloat16, device=DEV) * 0.1
nb = w.numel() * w.element_size()


def call_mm():
    return torch.mm(x1, w.t())


med, mn = bench(call_mm)
print("  dense mm [1x%d]@[%d,%d]: 读 %.3f GB  中位 %.3f ms ⇒ %.1f GB/s"
      % (K, N, K, nb / 1e9, med * 1e3, nb / med / 1e9))
