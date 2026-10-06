#!/usr/bin/env python3
"""并发分支 + 控核：找到"侧分支能被真隐藏"的分配方式。

关键推理：
  设主分支需 T_m（给 m 核）、侧分支需 T_s（给 s 核），核数→时间倍率 f(24)=1, f(16)=1.42, f(8)=2.81。
  · 串行            = T_m + T_s
  · 并发(主满核+侧小核) = max(T_m, T_s·f(s))
  ⇒ 只要 T_s·f(s) ≤ T_m，侧分支**完全免费**，总时间 = T_m。
  而"给主分支也限核"只会拖慢主分支（1.42×），通常不划算 —— 除非 T_s 与 T_m 同量级。
本脚本把这两种策略都测出来。
"""
import os, time, statistics as st
import torch, torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
lc = torch.npu.npugraph_ex.scope.limit_core_num
main = torch.npu.current_stream()
side = torch.npu.Stream()
ef = torch.npu.Event(); ej = torch.npu.Event()
REP = int(os.environ.get("REP", "40"))

# 主分支：大 GEMM
Am = torch.randn(2048, 4096, dtype=torch.bfloat16, device="npu:0")
Bm = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0")
# 侧分支：可调大小
def side_tensors(scale):
    n = max(64, int(2048 * scale))
    return (torch.randn(n, 4096, dtype=torch.bfloat16, device="npu:0"),
            torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0"))


def serial(As, Bs):
    for _ in range(3): _ = Am @ Bm; _ = As @ Bs
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(REP):
        _ = Am @ Bm; _ = As @ Bs
    torch.npu.synchronize(); return (time.perf_counter() - t0) / REP * 1e3


def concur(As, Bs, lim_main, lim_side):
    def one():
        ef.record(main); side.wait_event(ef)
        with torch.npu.stream(side):
            if lim_side:
                with lc(lim_side, lim_side * 2, side): _ = As @ Bs
            else:
                _ = As @ Bs
            ej.record(side)
        if lim_main:
            with lc(lim_main, lim_main * 2, main): _ = Am @ Bm
        else:
            _ = Am @ Bm
        main.wait_event(ej)
    for _ in range(3): one()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(REP): one()
    torch.npu.synchronize(); return (time.perf_counter() - t0) / REP * 1e3


print("%-8s %9s %9s %11s %11s %11s %11s" % (
    "侧/主比例", "串行ms", "主单独ms", "并发无限制", "主满+侧8", "主16+侧8", "主20+侧4"))
for scale in (0.0, 0.1, 0.25, 0.5, 1.0):
    As, Bs = side_tensors(scale) if scale > 0 else (Am[:64], Bm)
    tS = serial(As, Bs)
    # 主分支单独
    for _ in range(3): _ = Am @ Bm
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(REP): _ = Am @ Bm
    torch.npu.synchronize(); tM = (time.perf_counter() - t0) / REP * 1e3

    tA = concur(As, Bs, None, None)
    tC = concur(As, Bs, None, 8)
    tD = concur(As, Bs, 16, 8)
    tE = concur(As, Bs, 20, 4)
    print("%-8.2f %9.3f %9.3f %11.3f %11.3f %11.3f %11.3f" % (
        scale, tS, tM, tA, tC, tD, tE))

# fork/join 开销
for _ in range(5):
    ef.record(main); side.wait_event(ef); ej.record(side); main.wait_event(ej)
torch.npu.synchronize(); t0 = time.perf_counter()
for _ in range(REP * 4):
    ef.record(main); side.wait_event(ef); ej.record(side); main.wait_event(ej)
torch.npu.synchronize()
print("\n空 fork/join 开销: %.3f ms/对" % ((time.perf_counter() - t0) / (REP * 4) * 1e3))
