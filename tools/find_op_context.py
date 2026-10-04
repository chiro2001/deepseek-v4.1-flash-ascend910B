#!/usr/bin/env python3
"""定位某个算子的**调用上下文**：它在哪、前后是谁、同一 stream 上重复几次。

用法: find_op_context.py <profdir> <op_type> <stream_id> <anchor_stream> [n_step] [pad_us]
例:  find_op_context.py ~/tmp/n1 ViewCopy 47 146 0 8000
"""

from __future__ import annotations

import glob
import sys

import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    op_type = sys.argv[2]
    stream = float(sys.argv[3])
    anchor = float(sys.argv[4])
    n_step = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    pad_us = float(sys.argv[6]) if len(sys.argv) > 6 else 8000.0

    frames = [
        pd.read_csv(f, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv")
                        or glob.glob(prof_dir + "/kernel_details.csv"))
    ]
    df = pd.concat(frames, ignore_index=True)
    anchors = df[(df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == anchor)].sort_values(
        "Task Start Time(us)"
    )
    starts = anchors["Task Start Time(us)"].to_numpy()[::40]
    lo, hi = starts[len(starts) // 2 + n_step], starts[len(starts) // 2 + n_step + 1]

    step = df[(df["Task Start Time(us)"] >= lo) & (df["Task Start Time(us)"] < hi)]
    hits = step[(step["OP Type"] == op_type) & (step["Stream ID"] == stream)].sort_values("Task Start Time(us)")
    print(f"步长 {(hi-lo)/1000:.2f} ms；{op_type} @ s{stream:.0f} 命中 {len(hits)} 次")
    if hits.empty:
        return 0
    t0 = hits["Task Start Time(us)"].min() - pad_us
    t1 = hits["Task Start Time(us)"].max() + pad_us
    window = step[(step["Task Start Time(us)"] >= t0) & (step["Task Start Time(us)"] <= t1)].sort_values(
        "Task Start Time(us)"
    )
    prev_end = None
    for _, r in window.iterrows():
        gap = "" if prev_end is None else f"+{(r['Task Start Time(us)']-prev_end)/1000:6.3f}"
        sid = -1 if pd.isna(r["Stream ID"]) else int(r["Stream ID"])
        mark = " <== 目标" if r["OP Type"] == op_type and r["Stream ID"] == stream else ""
        print(
            f"  {(r['Task Start Time(us)']-lo)/1000:8.3f} {gap:>8} s{sid:<4}"
            f"{r['OP Type'][:30]:<30}{r['Task Duration(us)']:8.1f}us "
            f"{str(r['Input Shapes'])[:30]}{mark}"
        )
        prev_end = r["Task Start Time(us)"] + r["Task Duration(us)"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
