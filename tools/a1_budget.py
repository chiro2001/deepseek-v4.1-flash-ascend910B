#!/usr/bin/env python3
"""A1：为 kv_matmul ‖ q_b_matmul 找最优核预算分配。

真实 shape（v41-flat-verify3，TP8，decode M=6）：
  q_b_matmul: [M, 1280] @ [1280, 4096]   权重 10.0 MB
  kv_matmul : [M, 5120] @ [5120, 512]    权重  5.0 MB

现状（dsa_v1.py）：main_stream.wait_event(e_kv_matmul_done) ⇒ 两个 cube op **串行**。
目标：给各自分配 cube 预算，让它们**同时跑**，总时间 = max(两者)。

本脚本先测【单独执行】下核预算→时间，再测【并发】下的真实总时间（含 kv_norm/rope/scatter 的 AIV 尾）。
"""
from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
lc = torch.npu.npugraph_ex.scope.limit_core_num
main = torch.npu.current_stream()
aux = torch.npu.Stream()
e1 = torch.npu.Event(); e2 = torch.npu.Event()

M = int(os.environ.get("M", "6"))
QL, NH, HD = 1280, 64, 512          # tiny/真实 都是 q_lora=1280（tiny dummy 是 512，这里用真实）
QB_OUT = (NH // 8) * HD              # TP8 ⇒ 8 heads × 512 = 4096
KV_IN, KV_OUT = 5120, 512
REP = int(os.environ.get("REP", "200"))

Wqb = torch.randn(QL, QB_OUT, dtype=torch.bfloat16, device="npu:0")
xq = torch.randn(M, QL, dtype=torch.bfloat16, device="npu:0")
Wkv = torch.randn(KV_IN, KV_OUT, dtype=torch.bfloat16, device="npu:0")
xk = torch.randn(M, KV_IN, dtype=torch.bfloat16, device="npu:0")
# kv 尾部的 AIV 工作（norm+rope+scatter 的粗略替代：同尺寸的 elementwise）
kv_tail_a = torch.randn(M, KV_OUT, dtype=torch.float32, device="npu:0")
kv_tail_b = torch.randn(M, KV_OUT, dtype=torch.float32, device="npu:0")

print("q_b_matmul [%d,%d]@[%d,%d] 权重 %.1f MB" % (M, QL, QL, QB_OUT, QL*QB_OUT*2/2**20))
print("kv_matmul  [%d,%d]@[%d,%d] 权重 %.1f MB\n" % (M, KV_IN, KV_IN, KV_OUT, KV_IN*KV_OUT*2/2**20))


def dev(op, n=REP):
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=aux):
        for _ in range(n):
            op()
    for _ in range(3): g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / 10 / n * 1e6      # µs


def op_qb(aic):
    return lambda: (lc(aic, 48, main).__enter__(), xq @ Wqb, lc(aic, 48, main).__exit__())[1] \
        if False else _qb(aic)


def _qb(aic):
    with lc(aic, aic * 2, main):
        return xq @ Wqb


def _kv(aic):
    with lc(aic, 48, aux):
        return xk @ Wkv


print("=== 单独执行（图捕获摊销）===")
t = {}
for a in (24, 20, 16, 12, 10, 8, 6, 4):
    tq = dev(lambda a=a: _qb(a)); tk = dev(lambda a=a: _kv(a))
    t_a = dev(lambda: torch.add(kv_tail_a, kv_tail_b, out=kv_tail_a))
    t[a] = (tq, tk)
    print("  AIC=%2d: q_b %.2f µs | kv %.2f µs | kv尾AIV %.2f µs" % (a, tq, tk, t_a))
print("  AIC=24（现状满核）: q_b %.2f µs | kv %.2f µs" % (t[24][0], t[24][1]))
