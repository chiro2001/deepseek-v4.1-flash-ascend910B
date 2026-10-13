#!/usr/bin/env python3
"""多流图捕获 v2：fork event 必须**在捕获流上 record**。

规则（从 v1 的错误推出）：捕获区内，任何 `stream.wait_event(e)` 都要求
`e.record(...)` **也在捕获区内**。所以：
  根 = 捕获流 s1；在 s1 上 record fork event；s2 wait 它；最后 s1 wait s2 的 join event。
"""
from __future__ import annotations
import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
root = torch.npu.Stream()      # 捕获根流（非默认）
s2 = torch.npu.Stream()
e_fork = torch.npu.Event()
e_join = torch.npu.Event()

M, K, N = 6, 5120, 2048
W = torch.randn(K, N, dtype=torch.bfloat16, device="npu:0")
x1 = torch.randn(M, K, dtype=torch.bfloat16, device="npu:0")
x2 = torch.randn(M, K, dtype=torch.bfloat16, device="npu:0")
v1 = torch.randn(M * K, dtype=torch.float32, device="npu:0")
v2 = torch.randn(M * K, dtype=torch.float32, device="npu:0")
v1b = torch.randn(M * K, dtype=torch.float32, device="npu:0")
v2b = torch.randn(M * K, dtype=torch.float32, device="npu:0")
NITER = 40

def body():
    e_fork.record(root)                 # ← 在捕获根流上 record
    with torch.npu.stream(s2):
        s2.wait_event(e_fork)
        for _ in range(NITER):
            _ = x2 @ W
            torch.add(v2, v2b, out=v2)
        e_join.record(s2)
    for _ in range(NITER):               # 根流上跑另一条链
        _ = x1 @ W
        torch.add(v1, v1b, out=v1)
    root.wait_event(e_join)              # 汇合

print("=== 步 1：eager（在 root 上）===")
try:
    with torch.npu.stream(root):
        body()
    torch.npu.synchronize()
    print("  ✓ eager OK")
except Exception as e:
    print("  ✗ %s" % str(e)[:160]); raise SystemExit(0)

print("=== 步 2：多流图捕获（根流 = root）===")
g = torch.npu.NPUGraph()
try:
    with torch.npu.graph(g, stream=root):
        body()
    print("  ✓ 多流捕获成功")
except Exception as e:
    print("  ✗ 捕获失败: %s" % str(e)[:260]); raise SystemExit(0)

print("=== 步 3：重放测时 ===")
for _ in range(3): g.replay()
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(20): g.replay()
torch.npu.synchronize()
tg = (time.perf_counter() - t0) / 20 * 1e3
t0 = time.perf_counter()
with torch.npu.stream(root):
    for _ in range(20): body()
torch.npu.synchronize()
te = (time.perf_counter() - t0) / 20 * 1e3
print("  图重放 %.3f ms | eager %.3f ms ⇒ %.2f×" % (tg, te, te / tg))

# 串行对照（同工作量，单流）
print("=== 步 4：串行对照（同工作量，单流）===")
def serial():
    for _ in range(NITER * 2):
        _ = x1 @ W
        torch.add(v1, v1b, out=v1)
t0 = time.perf_counter()
with torch.npu.stream(root):
    for _ in range(20): serial()
torch.npu.synchronize()
ts = (time.perf_counter() - t0) / 20 * 1e3
print("  串行 %.3f ms ⇒ 多流图 %.2f× vs 串行" % (ts, ts / tg))
