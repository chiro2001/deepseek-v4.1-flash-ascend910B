#!/usr/bin/env python3
"""L2 预取（torch_npu.npu_prefetch）对"跨层权重流"是否有效 —— A/B。

场景：模拟 decode 的权重流 —— 41 个互不相同的权重矩阵（共 ~967 MB，远超 192 MiB L2），
每个用一次就换下一个。这正是"权重 > L2、无复用"的情形。

A 臂：顺序执行 x @ W_i.T
B 臂：算 W_i 之前，把 W_{i+1} 预取进 L2（侧流）
比较总时间与有效带宽（每次搬运 = W 的字节数）。

用法：python3 prefetch_ab.py [Layers] [M]
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch_npu  # noqa: F401

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 41
M = int(sys.argv[2]) if len(sys.argv) > 2 else 6
K, N = 5120, 2304          # 与真实 expert 的 w1 同形
REP = int(os.environ.get("REP", "5"))

torch.npu.set_device("npu:0")
print(f"[cfg] layers={NL} M={M} K={K} N={N} rep={REP} 权重总量={NL*K*N*2/2**20:.0f} MiB")

# 每层一个独立权重（避免共享导致缓存命中）
W = [torch.randn(N, K, dtype=torch.bfloat16, device="npu:0") for _ in range(NL)]
x = torch.randn(M, K, dtype=torch.bfloat16, device="npu:0")
wbytes = K * N * 2


def run(prefetch: bool):
    for _ in range(2):                       # warmup
        for i in range(NL):
            _ = x @ W[i].T
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(REP):
        for i in range(NL):
            if prefetch and i + 1 < NL:
                torch_npu.npu_prefetch(W[i + 1], None, wbytes)
            _ = x @ W[i].T
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / REP
    return dt


t_base = run(False)
print("A 顺序      : %.3f ms/次(41层)  有效带宽 %.0f GB/s" % (t_base * 1e3, NL * wbytes / t_base / 1e9))
t_pf = run(True)
print("B 预取下一层: %.3f ms/次(41层)  有效带宽 %.0f GB/s" % (t_pf * 1e3, NL * wbytes / t_pf / 1e9))
print("⇒ 预取%s（%+.1f%%）" % ("有效" if t_pf < t_base * 0.98 else "无显著收益", 100 * (t_pf / t_base - 1)))
