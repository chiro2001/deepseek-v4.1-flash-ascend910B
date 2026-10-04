#!/usr/bin/env python3
"""打印某个 stream 在一步内的算子时间线（按时间顺序，只列 ≥N 个）。

比"按类型聚合"多出来的信息：**顺序** —— 能看出这是"一串小算子背靠背"
还是"散落在别处"，也便于把链子对应回代码里的某个函数。

用法: stream_timeline.py <mindstudio_profiler_output> <stream> [n_step] [min_us] [max_rows] [t0_ms] [t1_ms]
      stream 也可写成 "47,35,41"（多流合并按时间排序）；t0/t1 是**相对步首**的毫秒。
"""

from __future__ import annotations

import glob
import sys

import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    stream_arg = sys.argv[2]
    n_step = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    min_us = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    max_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    t0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    t1 = float(sys.argv[7]) if len(sys.argv) > 7 else None

    cols = ["OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)", "Input Shapes"]
    frames = [
        pd.read_csv(f, usecols=cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv"))
    ]
    df = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = df["Task Start Time(us)"].min()
    df["s"] = (df["Task Start Time(us)"] - origin) / 1000.0
    df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0

    anchors = df[
        (df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == 158.0)
    ].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    mid = len(starts) // 3
    lo, hi = starts[mid + n_step], starts[mid + n_step + 1]

    streams = [float(x) for x in str(stream_arg).split(",")]
    sub = df[(df["s"] < hi) & (df["e"] > lo) & (df["Stream ID"].isin(streams))].copy()
    sub = sub[(sub["Task Duration(us)"] >= min_us)].sort_values("s")
    sub = sub[(sub["s"] >= lo + t0) & ((sub["e"] <= lo + t1) if t1 is not None else True)]
    print(f"step {mid+n_step}  {hi-lo:.2f} ms  streams={stream_arg}  rows={len(sub)}  t={t0}..{t1}")
    prev_end = None
    for i, (_, r) in enumerate(sub.iterrows()):
        if i >= max_rows:
            print(f"  … 其余 {len(sub)-max_rows} 行省略")
            break
        gap = "" if prev_end is None else f"+{r['s']-prev_end:7.3f}"
        print(
            f"  {r['s']-lo:8.3f} {gap:>9} s{int(r['Stream ID']):<4}{r['OP Type'][:28]:<28}"
            f"{r['Task Duration(us)']:8.1f} µs  {str(r['Input Shapes'])[:40]}"
        )
        prev_end = r["e"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
