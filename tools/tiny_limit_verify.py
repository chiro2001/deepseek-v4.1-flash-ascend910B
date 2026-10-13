#!/usr/bin/env python3
"""先验证 limit_core_num 是否真的生效：单个 GEMM 在不同核预算下的耗时。"""
import os, time
import torch, torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
lc = torch.npu.npugraph_ex.scope.limit_core_num
REP = int(os.environ.get("REP", "50"))
M, N, K = 2048, 4096, 4096
a = torch.randn(M, K, dtype=torch.bfloat16, device="npu:0")
b = torch.randn(K, N, dtype=torch.bfloat16, device="npu:0")
s = torch.npu.current_stream()

def bench(aic, aiv):
    def one():
        if aic is None:
            return a @ b
        with lc(aic, aiv, s):
            return a @ b
    for _ in range(5):
        one()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(REP):
        one()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / REP * 1e3

base = bench(None, None)
print("无限制          : %8.3f ms" % base)
for aic in (24, 16, 12, 8, 4):
    t = bench(aic, aic * 2)
    print("limit AIC=%2d AIV=%2d: %8.3f ms  (%.2f× vs 无限制)" % (aic, aic * 2, t, t / base))
