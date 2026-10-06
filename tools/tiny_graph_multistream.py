#!/usr/bin/env python3
"""前置验证：**一张 NPUGraph 能否捕获"两条独立流的并行分支"**。

上游 GPU 版 DBO 就是把所有 ubatch 线程的 join 包进一次 torch.cuda.graph。
若 NPU 的 torch.npu.graph 支持多流捕获（capture_begin 时指定一个流，
其它流通过 event 依赖接入），则 B2 可行。
"""
from __future__ import annotations
import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
main = torch.npu.current_stream()
s1 = torch.npu.Stream()
s2 = torch.npu.Stream()
e_fork = torch.npu.Event()
e_j1 = torch.npu.Event(); e_j2 = torch.npu.Event()

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
    e_fork.record(main)
    with torch.npu.stream(s1):
        s1.wait_event(e_fork)
        for _ in range(NITER):
            _ = x1 @ W
            torch.add(v1, v1b, out=v1)
        e_j1.record(s1)
    with torch.npu.stream(s2):
        s2.wait_event(e_fork)
        for _ in range(NITER):
            _ = x2 @ W
            torch.add(v2, v2b, out=v2)
        e_j2.record(s2)
    main.wait_event(e_j1); main.wait_event(e_j2)

print("=== 步 1：eager 下跑通（含跨流 event）===")
try:
    body(); torch.npu.synchronize(); print("  ✓ eager 双流 OK")
except Exception as e:
    print("  ✗ eager 失败: %s" % str(e)[:150]); raise SystemExit(0)

print("=== 步 2：把整段（含双流 + join）捕获进一张图 ===")
g = torch.npu.NPUGraph()
try:
    with torch.npu.graph(g, stream=s1):
        body()
    print("  ✓ 多流捕获成功")
except Exception as e:
    print("  ✗ 多流捕获失败: %s" % str(e)[:220])
    raise SystemExit(0)

print("=== 步 3：重放并测时 ===")
try:
    for _ in range(3): g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): g.replay()
    torch.npu.synchronize()
    t_graph = (time.perf_counter() - t0) / 20 * 1e3
    print("  图重放 %.3f ms" % t_graph)
    t0 = time.perf_counter()
    for _ in range(20): body()
    torch.npu.synchronize()
    t_eager = (time.perf_counter() - t0) / 20 * 1e3
    print("  eager  %.3f ms" % t_eager)
    print("  ⇒ 图捕获%s（%.2f× vs eager）" % ("更快" if t_graph < t_eager else "更慢", t_eager / t_graph))
except Exception as e:
    print("  ✗ 重放失败: %s" % str(e)[:200])
