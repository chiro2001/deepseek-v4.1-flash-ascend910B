#!/usr/bin/env python3
"""精确扫 L2 拐点：足迹 100→260 MB，步长 10 MB，同时报 L1/L2 级带宽。"""
import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
IT = 200


def t3(n):
    x = torch.randn(n, dtype=torch.float32, device="npu:0")
    y = torch.randn(n, dtype=torch.float32, device="npu:0")
    z = torch.empty(n, dtype=torch.float32, device="npu:0")
    for _ in range(20):
        torch.add(x, y, out=z)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(IT):
        torch.add(x, y, out=z)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / IT
    del x, y, z
    return dt


print("%14s %12s %12s" % ("footprint_MB", "per_us", "GB/s"))
prev = None
for foot_mb in range(60, 300, 10):
    n = int(foot_mb * 1e6 / 12)          # 3 × n × 4B = foot
    dt = t3(n)
    bw = 3 * n * 4 / dt / 1e9
    marg = ""
    if prev:
        dn = 3 * n * 4 - prev[0]
        dtt = dt - prev[1]
        if dtt > 1e-6:
            marg = "%8.0f" % (dn / dtt / 1e9)
    print("%14.0f %12.2f %12.0f %s" % (foot_mb, dt * 1e6, bw, marg))
    prev = (3 * n * 4, dt)
