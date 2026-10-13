#!/usr/bin/env python3
"""细扫：定位"片上缓存 → HBM"的拐点（= 有效缓存容量）。

改进点：
  * 用**两尺寸差分**去掉固定下发开销：BW_marginal = 3*(n2-n1)/(t2-t1)
  * 尺寸从 4 MB 扫到 1 GB 足迹，密集覆盖拐点区
  * 每次重建张量（避免 allocator 复用带来的假命中）
"""
from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=80)
ap.add_argument("--device", default="npu:0")
a = ap.parse_args()
dev = a.device
torch.npu.set_device(dev)


def timed(x, y, z, iters, warm=8):
    for _ in range(warm):
        torch.add(x, y, out=z)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.add(x, y, out=z)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


# 足迹 = 3 × n × 4B；n 从 0.33M（4MB 足迹）到 89.5M（1.07GB 足迹）
SIZES = [1 << 20, 1 << 21, 3 << 20, 1 << 22, 3 << 21, 1 << 23, 3 << 22, 1 << 24,
         3 << 23, 1 << 25, 3 << 24, 1 << 26, 3 << 25]
print("%14s %14s %12s %12s" % ("footprint_MB", "per_iter_us", "GB/s", "marginal_GB/s"))
prev = None
for n in SIZES:
    x = torch.randn(n, dtype=torch.float32, device=dev)
    y = torch.randn(n, dtype=torch.float32, device=dev)
    z = torch.empty(n, dtype=torch.float32, device=dev)
    foot = 3 * n * 4
    t = timed(x, y, z, a.iters)
    bw = foot / t / 1e9
    marg = float("nan")
    if prev is not None:
        dn = foot - prev[0]
        dt = t - prev[1]
        marg = dn / dt / 1e9 if dt > 0 else float("inf")
    print("%14.2f %14.2f %12.1f %12.1f" % (foot / 1e6, t * 1e6, bw, marg))
    prev = (foot, t)
    del x, y, z
