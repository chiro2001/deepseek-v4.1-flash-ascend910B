#!/usr/bin/env python3
"""真实场景：AIC 密集分支（GEMM）与 AIV 密集分支（elementwise）并发。

这才是我们模型的情形：AIC 22.05 ms/步、AIV 14.75 ms/步，但两者只重叠 3.50 ms。
关键问：AIV 分支能否"免费"藏在 AIC 分支后面？AIC 预算该怎么分？
"""
import os, time
import torch, torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
lc = torch.npu.npugraph_ex.scope.limit_core_num
main = torch.npu.current_stream(); side = torch.npu.Stream()
ef = torch.npu.Event(); ej = torch.npu.Event()
REP = int(os.environ.get("REP", "40"))

# AIC 密集：GEMM
Ag = torch.randn(2048, 4096, dtype=torch.bfloat16, device="npu:0")
Bg = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0")
# AIV 密集：大 elementwise（fp32，纯 vector）
N = 32 * 1024 * 1024
Xv = torch.randn(N, dtype=torch.float32, device="npu:0")
Yv = torch.randn(N, dtype=torch.float32, device="npu:0")


def tm(fn, n=REP):
    for _ in range(3): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize(); return (time.perf_counter() - t0) / n * 1e3


# 各算子在"满核"下的单独耗时
T_gemm = tm(lambda: Ag @ Bg)
T_vec = tm(lambda: torch.add(Xv, Yv, out=Xv))
print("满核单独耗时：GEMM %.3f ms | elementwise %.3f ms | 串行合计 %.3f ms" % (
    T_gemm, T_vec, T_gemm + T_vec))

# 限核后的单独耗时
print("\n=== 单独执行时的核预算敏感度 ===")
print("%-20s %10s %10s" % ("限制", "GEMM ms", "elemwise ms"))
for aic, aiv in ((None, None), (20, 40), (16, 32), (12, 24), (8, 16), (4, 8), (24, 24), (24, 12)):
    if aic is None:
        tg, tv = T_gemm, T_vec
    else:
        def _g():
            with lc(aic, aiv, main):
                _ = Ag @ Bg
        def _v():
            with lc(aic, aiv, main):
                torch.add(Xv, Yv, out=Xv)
        tg = tm(_g, 20); tv = tm(_v, 20)
    print("%-20s %10.3f %10.3f" % (f"AIC={aic} AIV={aiv}", tg, tv))

print("\n=== 并发：GEMM(主) ∥ elementwise(侧) ===")
print("%-26s %10s %10s" % ("分配（主/侧）", "总ms", "vs 串行"))

def concur(lmg, lvg, lmw, lvw):
    def one():
        ef.record(main); side.wait_event(ef)
        with torch.npu.stream(side):
            if lmw is not None:
                with lc(lmw, lvw, side): torch.add(Xv, Yv, out=Xv)
            else:
                torch.add(Xv, Yv, out=Xv)
            ej.record(side)
        if lmg is not None:
            with lc(lmg, lvg, main): _ = Ag @ Bg
        else:
            _ = Ag @ Bg
        main.wait_event(ej)
    return tm(one)

serial = T_gemm + T_vec
cands = [
    ("不限 / 不限", None, None, None, None),
    ("GEMM满 / vec AIV24", None, None, 24, 24),
    ("GEMM满 / vec AIV16", None, None, 24, 16),
    ("GEMM AIC16 / vec AIV16", 16, 32, 24, 16),
    ("GEMM AIC16 / vec AIV12", 16, 32, 24, 12),
    ("GEMM AIC12 / vec AIV24", 12, 24, 24, 24),
    ("GEMM AIC20 / vec AIV16", 20, 40, 24, 16),
]
for tag, a, b, c, d in cands:
    t = concur(a, b, c, d)
    print("%-26s %10.3f %9.1f%%" % (tag, t, 100 * (serial / t - 1)))
