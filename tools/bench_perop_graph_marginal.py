#!/usr/bin/env python3
"""图模式 vs eager 的"单算子边际成本"（含 footprint 扫描）。"""
from __future__ import annotations
import glob, os, csv
import torch
import torch_npu  # noqa

dev = torch.device("npu:0")
torch.npu.set_device(dev)


def prof(tag, fn):
    for _ in range(3): fn()
    torch.npu.synchronize()
    d = f"/tmp/gp2_{tag}"
    os.makedirs(d, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False, data_simplification=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=exp,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(d)):
        fn(); torch.npu.synchronize()
    fs = glob.glob(d + "/**/kernel_details.csv", recursive=True)
    if not fs: return None, 0
    tot = 0.0; n = 0
    with open(fs[0], newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                tot += float(r["Duration(us)"]); n += 1
            except Exception: pass
    return tot, n


out = []
out.append(f"{'M':>5}{'模式':<8}{'N':>5}{'总us':>10}{'算子数':>8}{'边际us/算子':>13}")
for M in (6, 48):
    H = 5120
    buf = torch.empty(M, H, dtype=torch.bfloat16, device=dev).uniform_(-1, 1)
    w = torch.empty(M, H, dtype=torch.bfloat16, device=dev).uniform_(0.5, 1.5)
    MB = 2 * M * H * 2 / 1e6

    def chain(N):
        for _ in range(N):
            buf.mul_(w)

    for mode in ("eager", "graph"):
        prev = None
        for N in (4, 16, 64):
            if mode == "eager":
                t, k = prof(f"e{M}_{N}", lambda N=N: chain(N))
            else:
                g = torch.npu.NPUGraph()
                torch.npu.synchronize()
                with torch.npu.graph(g):
                    chain(N)
                t, k = prof(f"g{M}_{N}", lambda: g.replay())
            slope = "" if prev is None or t is None else f"{(t-prev)/(N-4 if prev is None else 12 if prev==16 else 48):.2f}"
            if prev is not None and t is not None:
                # 用相邻点算斜率
                pass
            out.append(f"{M:>5}{mode:<8}{N:>5}{(t if t else -1):>10.2f}{k:>8}")
            prev = t
            if hasattr(prev, "__len__"): prev = prev[0]
            # 记录 (N, t) 以便后面算斜率
            out[-1] += ""
        out.append("")
    # 明确算斜率
    for mode in ("eager", "graph"):
        pts = []
        for N in (4, 16, 64):
            tag = f"e{M}_{N}" if mode == "eager" else f"g{M}_{N}"
            d = f"/tmp/gp2_{tag}"
            fs = glob.glob(d + "/**/kernel_details.csv", recursive=True)
            if not fs: continue
            tot = 0.0
            with open(fs[0], newline="") as fh:
                for r in csv.DictReader(fh):
                    try: tot += float(r["Duration(us)"])
                    except Exception: pass
            pts.append((N, tot))
        if len(pts) >= 2:
            s = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
            out.append(f"  ⇒ M={M} {mode} 边际 = {s:.2f} us/算子 "
                       f"（footprint {MB:.2f} MB，按 845GB/s 需 {MB*1000/845:.2f} us）")
    out.append("")

print("\n".join(out))
