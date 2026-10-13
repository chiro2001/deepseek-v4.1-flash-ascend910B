#!/usr/bin/env python3
"""A3 (910_93) HBM 带宽实测：读 / 读+写(copy) 两种口径。

动机：目标把 HBM 上限定为 1182 GB/s，并用它判断"是否打爆 HBM"。
但 decode 被证实是**带宽受限**（官方技术报告原话 + 我们每 rank 每步要搬 34 GB），
所以这个上限是所有结论的分母，必须先实测而不是沿用假设值。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"


def bench(fn, warmup=3, iters=12):
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


print("=== 1) 纯读（大张量求和）===")
for gb in (0.5, 1.0, 2.0):
    n = int(gb * 1024**3 // 2)
    x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    nbytes = x.numel() * x.element_size()
    med, mn = bench(lambda x=x: x.sum())
    print("  %.1f GiB  med %7.2f ms ⇒ %8.1f GB/s   (best %8.1f)" %
          (nbytes / 1024**3, med * 1e3, nbytes / med / 1e9, nbytes / mn / 1e9))
    del x
    torch.npu.empty_cache()

print()
print("=== 2) 读+写（copy_）===")
for gb in (0.25, 0.5, 1.0):
    n = int(gb * 1024**3 // 2)
    x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    med, mn = bench(lambda x=x, y=y: y.copy_(x))
    # copy 计：读 nbytes + 写 nbytes
    print("  %.2f GiB  med %7.2f ms ⇒ 有效 %8.1f GB/s（读写合计 %8.1f）" %
          (nbytes / 1024**3, med * 1e3, nbytes / med / 1e9, 2 * nbytes / med / 1e9))
    del x, y
    torch.npu.empty_cache()

print()
print("=== 3) 读+写（mul_ 原地，只读写一遍）===")
n = int(1.0 * 1024**3 // 2)
x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
nb = x.numel() * x.element_size()
med, mn = bench(lambda x=x: x.mul_(1.000001))
print("  1.0 GiB  med %7.2f ms ⇒ 读写合计 %8.1f GB/s" % (med * 1e3, 2 * nb / med / 1e9))

print()
print("=== 4) 设备信息 ===")
try:
    import torch_npu
    print("  soc:", torch_npu.npu.get_device_name(0))
except Exception as e:
    print("  n/a", e)
