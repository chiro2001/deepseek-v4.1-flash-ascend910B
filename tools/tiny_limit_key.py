#!/usr/bin/env python3
"""决定性测试：把 AIV 预算从 cube 密集流"让"给纯向量流。

假设：GEMM 是 cube-only，给它 48 个 vector 核没用 ⇒ 把主流的 AIV 压到 8，
侧流（纯 elementwise）拿 40 个 vector 核，两者应能真并行。
"""
import os, time
import torch, torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
main = torch.npu.current_stream(); side = torch.npu.Stream()
ef = torch.npu.Event(); ej = torch.npu.Event()
REP = int(os.environ.get("REP", "40"))

Ag = torch.randn(2048, 4096, dtype=torch.bfloat16, device="npu:0")
Bg = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0")
N = 16 * 1024 * 1024
X = torch.randn(N, dtype=torch.float32, device="npu:0")
Y = torch.randn(N, dtype=torch.float32, device="npu:0")


def tm(fn, n=REP, warm=3):
    for _ in range(warm): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize(); return (time.perf_counter() - t0) / n * 1e3


def with_limits(ma, mv, sa, sv, fn):
    mo = torch.npu.get_stream_limit(main); so = torch.npu.get_stream_limit(side)
    torch.npu.set_stream_limit(main, ma, mv); torch.npu.set_stream_limit(side, sa, sv)
    try:
        return fn()
    finally:
        torch.npu.set_stream_limit(main, mo["cube_core_num"], mo["vector_core_num"])
        torch.npu.set_stream_limit(side, so["cube_core_num"], so["vector_core_num"])


# ① GEMM 给不同 AIV 预算 —— 验证"cube-only 不看 AIV"
print("=== ① GEMM 在 AIC=24 下、不同 AIV 预算 ===")
for aiv in (48, 32, 16, 8, 4, 2):
    t = with_limits(24, aiv, 24, 48, lambda: tm(lambda: Ag @ Bg, 30))
    print("   AIV=%2d → %.3f ms" % (aiv, t))

# ② elementwise 给不同 AIV
print("\n=== ② elementwise 在 AIV 预算下的耗时 ===")
for aiv in (48, 40, 32, 24, 16, 8):
    t = with_limits(24, 48, 1, aiv, lambda: tm(lambda: torch.add(X, Y, out=X), 30))
    print("   AIV=%2d → %.3f ms" % (aiv, t))

# ③ 并发：主 GEMM(24,8) ∥ 侧 elem(1,40)
Tg = tm(lambda: Ag @ Bg); Tv = tm(lambda: torch.add(X, Y, out=X)); ser = Tg + Tv
print("\n=== ③ 并发：串行基准 %.3f ms（GEMM %.3f + vec %.3f）===" % (ser, Tg, Tv))
print("  %-26s %9s %9s" % ("(主AIC,主AIV)/(侧AIC,侧AIV)", "总ms", "vs串行"))
for c in [(24,48,24,48), (24,8,1,40), (24,8,4,40), (24,12,4,36), (24,16,1,32), (23,8,1,40), (24,8,1,24)]:
    ma, mv, sa, sv = c
    if ma + sa > 24 or mv + sv > 48:
        print("  %-26s  (超出预算，跳过)" % f"({ma},{mv})/({sa},{sv})"); continue
    def body():
        def one():
            ef.record(main); side.wait_event(ef)
            with torch.npu.stream(side):
                torch.add(X, Y, out=X)
                ej.record(side)
            _ = Ag @ Bg
            main.wait_event(ej)
        return tm(one)
    t = with_limits(ma, mv, sa, sv, body)
    print("  %-26s %9.3f %8.1f%%" % (f"({ma},{mv})/({sa},{sv})", t, 100 * (ser / t - 1)))
