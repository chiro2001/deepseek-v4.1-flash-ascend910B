#!/usr/bin/env python3
"""空闲归因：把 decode 步里的"系统真空闲"逐段列出，并给出每段的**并行覆盖者**。

定义：某段 [a,b] 内**没有任何设备任务**（AI core / vector / AICPU 都不算）时为"真空闲"，
长度 >= 阈值才计入。对每段给出：
  * 上一段忙的最后一个算子（谁刚结束）
  * 下一段忙的第一个算子（谁在等）
  * 与该段重叠（含跨界覆盖）的算子清单 —— 用来判断"是不是被某个跨步的长任务遮住了"

用法: idle_who.py <mindstudio_profiler_output> [anchor_stream] [n_steps] [min_gap_ms]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    anchor_stream = float(sys.argv[2]) if len(sys.argv) > 2 else 158.0
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    min_gap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.020

    cols = ["OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)", "Task Type"]
    frames = [
        pd.read_csv(f, usecols=cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv")
        or glob.glob(prof_dir + "/kernel_details.csv"))
    ]
    df = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = df["Task Start Time(us)"].min()
    df["s"] = (df["Task Start Time(us)"] - origin) / 1000.0
    df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0

    anchors = df[
        (df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == anchor_stream)
    ].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    mid = len(starts) // 3
    bounds = starts[mid : mid + n_steps + 1]
    print(f"取 {len(bounds)-1} 个 decode 步（锚点切步，阈值 {min_gap*1000:.0f} µs）")

    buckets: dict[tuple[str, str], dict] = {}
    total_idle = 0.0
    total_span = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        total_span += hi - lo
        sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
        sub["s"] = sub["s"].clip(lo, hi)
        sub["e"] = sub["e"].clip(lo, hi)
        order = sub.sort_values(["s", "e"]).reset_index(drop=True)
        # 逐段扫描并集
        seg_start = None
        seg_end = None
        prev_op = None
        segs: list[tuple[float, float, str]] = []
        for _, row in order.iterrows():
            if seg_start is None:
                seg_start, seg_end = row["s"], row["e"]
                continue
            if row["s"] > seg_end:
                segs.append((seg_start, seg_end, prev_op))
                seg_start, seg_end = row["s"], row["e"]
            else:
                seg_end = max(seg_end, row["e"])
            prev_op = row["OP Type"]
        if seg_start is not None:
            segs.append((seg_start, seg_end, prev_op))

        for i in range(len(segs) - 1):
            gap = segs[i + 1][0] - segs[i][1]
            if gap < min_gap:
                continue
            total_idle += gap
            a, b = segs[i][1], segs[i + 1][0]
            nxt = order[order["s"] >= b - 1e-9].sort_values("s").head(1)
            next_op = nxt["OP Type"].iloc[0] if len(nxt) else "?"
            key = (str(segs[i][2]), str(next_op))
            rec = buckets.setdefault(
                key, {"n": 0, "ms": 0.0, "sids": {}, "dur": []}
            )
            rec["n"] += 1
            rec["ms"] += gap
            rec["dur"].append(gap)
            covers = order[(order["s"] <= a) & (order["e"] >= b)]
            for _, c in covers.iterrows():
                sid = int(c["Stream ID"]) if not pd.isna(c["Stream ID"]) else -1
                rec["sids"][sid] = rec["sids"].get(sid, 0) + 1

    print(
        f"真空闲合计 {total_idle:.3f} ms / {total_span:.1f} ms "
        f"=> 每步 {total_idle/(len(bounds)-1):.3f} ms（{total_idle/total_span*100:.2f}%）"
    )
    print()
    print(f"{'gap_ms':>9}{'n':>5}  {'prev_op':<34} -> {'next_op':<34}  被谁覆盖(stream:次数)")
    for (prev_op, next_op), rec in sorted(buckets.items(), key=lambda kv: -kv[1]["ms"])[:20]:
        env = " ".join(f"s{s}:{c}" for s, c in sorted(rec["sids"].items(), key=lambda kv: -kv[1]))
        print(f"{rec['ms']:>9.3f}{rec['n']:>5}  {prev_op[:34]:<34} -> {next_op[:34]:<34}  {env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
