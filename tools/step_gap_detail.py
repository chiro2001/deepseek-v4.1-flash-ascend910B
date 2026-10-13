#!/usr/bin/env python3
"""打印一个 decode 步里每个"真空闲"缺口的前后文。

对比 `idle_who.py`（跨多步按 (prev_op → next_op) 聚合），本工具看**单步现场**：
每个 ≥MINGAP 的缺口，列出缺口前 150 µs 与后 150 µs 内所有设备任务。

用法: step_gap_detail.py <mindstudio_profiler_output> <n_step> [min_gap_ms]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    n_step = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    min_gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    cols = ["OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)", "Task Type", "Block Num"]
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
        (df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == 158.0)
    ].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    mid = len(starts) // 3
    lo, hi = starts[mid + n_step], starts[mid + n_step + 1]
    sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
    sub["s"] = sub["s"].clip(lo, hi)
    sub["e"] = sub["e"].clip(lo, hi)

    ss = np.sort(sub[["s", "e"]].to_numpy(), axis=0)
    gaps = []
    _, ce = ss[0]
    for s, e in ss[1:]:
        if s > ce + 1e-9:
            if s - ce >= min_gap:
                gaps.append((ce, s, s - ce))
            ce = e
        else:
            ce = max(ce, e)

    total = sum(g[2] for g in gaps)
    print(f"step {mid + n_step} len={hi - lo:.2f} ms  gaps>= {min_gap} ms: {len(gaps)}  total {total:.3f} ms")

    for a0, b0, gap in gaps:
        print(f"\n--- gap {gap:.3f} ms   t = {a0 - lo:.3f} .. {b0 - lo:.3f} (step-relative)")
        window = sub[(sub["e"] > a0 - 0.15) & (sub["s"] < b0 + 0.15)].sort_values("s")
        for _, row in window.iterrows():
            sid = -1 if pd.isna(row["Stream ID"]) else int(row["Stream ID"])
            mark = "pre " if row["e"] <= a0 + 1e-9 else ("post" if row["s"] >= b0 - 1e-9 else "GAP!")
            print(
                f"   {row['s'] - lo:8.3f} {sid:>4} {row['OP Type'][:34]:<34}"
                f"{row['Task Duration(us)'] / 1000:9.4f} ms  {mark}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
