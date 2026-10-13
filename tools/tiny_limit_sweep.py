#!/usr/bin/env python3
"""2D 扫描：给定侧/主规模比，找最优 (主核, 侧核) 分配。

约束：主核 + 侧核 ≤ 24（A3 每 die 的 cube 数）。
输出：每个规模比下的最优分配、相对串行的增益，以及"是否值得并发"的判据。
"""
import os, time
import torch, torch_npu  # noqa: F401

torch.npu.set_device("npu:0")
lc = torch.npu.npugraph_ex.scope.limit_core_num
main = torch.npu.current_stream(); side = torch.npu.Stream()
ef = torch.npu.Event(); ej = torch.npu.Event()
REP = int(os.environ.get("REP", "30"))

Bm = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0")
As = {s: torch.randn(max(64, int(2048 * s)), 4096, dtype=torch.bfloat16, device="npu:0")
      for s in (0.25, 0.5, 0.75, 1.0)}
Bs = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu:0")


def t_serial(Am, As_):
    for _ in range(3): _ = Am @ Bm; _ = As_ @ Bs
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(REP): _ = Am @ Bm; _ = As_ @ Bs
    torch.npu.synchronize(); return (time.perf_counter() - t0) / REP * 1e3


def t_concur(Am, As_, lm, ls):
    def one():
        ef.record(main); side.wait_event(ef)
        with torch.npu.stream(side):
            if ls: 
                with lc(ls, ls * 2, side): _ = As_ @ Bs
            else:
                _ = As_ @ Bs
            ej.record(side)
        if lm:
            with lc(lm, lm * 2, main): _ = Am @ Bm
        else:
            _ = Am @ Bm
        main.wait_event(ej)
    for _ in range(3): one()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(REP): one()
    torch.npu.synchronize(); return (time.perf_counter() - t0) / REP * 1e3


Am = torch.randn(2048, 4096, dtype=torch.bfloat16, device="npu:0")
print("主分支固定 2048x4096 GEMM；A3 每 die 24 cube\n")
print("%-8s %8s | %-24s %8s %8s" % ("侧/主", "串行ms", "最优(主核,侧核)", "最优ms", "增益"))
for s in (0.25, 0.5, 0.75, 1.0):
    As_ = As[s]
    tS = t_serial(Am, As_)
    best = (None, 1e9)
    rows = []
    for lm in (None, 20, 16, 12):
        for ls in (4, 6, 8, 10, 12):
            used = (lm or 24) + ls
            if used > 24:
                continue
            t = t_concur(Am, As_, lm, ls)
            rows.append((t, lm, ls))
            if t < best[1]:
                best = ((lm, ls), t)
    gain = 100 * (tS / best[1] - 1)
    print("%-8.2f %8.3f | %-24s %8.3f %+7.1f%%" % (
        s, tS, "(%s, %d)" % (best[0][0] or "满", best[0][1]), best[1], gain))
    top = sorted(rows)[:4]
    for t, lm, ls in top:
        print("        └ (主%s, 侧%d): %.3f ms" % (lm or "满", ls, t))
