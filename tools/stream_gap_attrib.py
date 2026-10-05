#!/usr/bin/env python3
"""**某个流内部**的等待归因：该流的任务之间在等什么。

与 `idle_who.py` 的区别：那个看"全系统真空闲"，本工具看"**这条流自己被卡住**"。
对主计算流（decode 的 stream 146/109）尤其重要 —— 它的任务是**严格串行**的依赖链，
所以它自己的空隙 = 它真的在等别人（通信/其它流），这就是可优化面。

用法: stream_gap_attrib.py <profdir> <stream> [n_steps] [min_gap_ms] [anchor_stream]
输出：按 (前一个算子 → 后一个算子) 聚合的空隙总量；以及 Top-N 单个空隙现场。
"""

from __future__ import annotations

import argparse
import glob

import pandas as pd


def load(prof_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(prof_dir + "/op_summary*.csv")) or [
        prof_dir + "/kernel_details.csv"
    ]
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df = df.rename(columns={
            "Name": "Op Name", "Type": "OP Type",
            "Start Time(us)": "Task Start Time(us)", "Duration(us)": "Task Duration(us)",
            "Accelerator Core": "Task Type",
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = out["Task Start Time(us)"].min()
    out["s"] = (out["Task Start Time(us)"] - origin) / 1000.0
    out["e"] = out["s"] + out["Task Duration(us)"] / 1000.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir")
    ap.add_argument("stream", type=float)
    ap.add_argument("n_steps", type=int, nargs="?", default=16)
    ap.add_argument("min_gap_ms", type=float, nargs="?", default=0.005)
    ap.add_argument("anchor_stream", type=float, nargs="?", default=146.0)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    df = load(a.prof_dir)
    anchors = df[(df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == a.anchor_stream)].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    if len(starts) < 3:
        print("锚点不足")
        return 1
    mid = len(starts) // 3
    lo, hi = starts[mid], starts[mid + a.n_steps]
    print(f"窗口 {lo:.1f}–{hi:.1f} ms（{hi-lo:.1f} ms，{a.n_steps} 步）；流 {a.stream:.0f}；阈值 {a.min_gap_ms*1000:.0f} µs")

    ms = df[(df["Stream ID"] == a.stream) & (df["s"] >= lo) & (df["s"] < hi)].sort_values("s")
    has_name = "OP Name" in ms.columns
    gaps = []
    prev = None
    for _, r in ms.iterrows():
        if prev is not None:
            g = r["s"] - prev["e"]
            if g >= a.min_gap_ms:
                gaps.append((prev["OP Type"], r["OP Type"], g, prev["e"] - lo,
                             r["OP Name"] if has_name else ""))
        prev = r

    total = sum(x[2] for x in gaps)
    n_steps = a.n_steps
    print(f"该流内空隙（≥阈值）合计 **{total:.2f} ms / {n_steps} 步 = {total/n_steps:.3f} ms/步**，{len(gaps)} 个")
    print()
    agg: dict[tuple[str, str], list[float]] = {}
    for p, q, g, _, _ in gaps:
        agg.setdefault((p, q), []).append(g)
    rows = sorted(((sum(v), len(v), k) for k, v in agg.items()), reverse=True)
    print(f"{'ms_total':>10}{'n':>6}{'ms/步':>9}  prev → next")
    for s, n, (p, q) in rows[: a.top]:
        print(f"{s:>10.2f}{n:>6}{s/n_steps:>9.3f}  {p[:26]:<26} → {q[:26]}")
    print()
    print("Top-N 单个空隙（看现场）:")
    for p, q, g, t, name in sorted(gaps, key=lambda x: -x[2])[: a.top]:
        print(f"  {g*1000:8.1f} µs  @t={t:8.3f} ms  {p[:24]:<24} → {q[:24]:<24} {name[:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
