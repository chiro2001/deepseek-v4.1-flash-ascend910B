#!/usr/bin/env python3
"""把一个 stream 上的算子按 (OP Type, 输入形状) 聚类，给出**每步次数**。

用途：把"420 个 eager 小算子"还原成"少数几条调用链 × 每步几次"，
从而知道该去改哪段代码（形状是关键线索：SWA 链的宽 = window，压缩链的宽 = index_topk）。

用法: op_chain_cluster.py <mindstudio_profiler_output> <stream|all> [anchor_stream] [n_steps]
"""

from __future__ import annotations

import glob
import sys

import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "all"
    anchor_stream = float(sys.argv[3]) if len(sys.argv) > 3 else 158.0
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 16

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
        (df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == anchor_stream)
    ].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    mid = len(starts) // 3
    lo, hi = starts[mid], starts[mid + n_steps]

    sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
    sub["ms"] = sub["Task Duration(us)"] / 1000.0
    if target != "all":
        sub = sub[sub["Stream ID"] == float(target)]

    grp = (
        sub.groupby(["OP Type", "Input Shapes"])
        .agg(n=("ms", "size"), ms=("ms", "sum"))
        .reset_index()
    )
    grp["per_step"] = grp["n"] / n_steps
    grp["ms_per_step"] = grp["ms"] / n_steps
    grp = grp[grp["per_step"] >= 0.5].sort_values("ms_per_step", ascending=False)
    print(f"窗口 {hi-lo:.2f} ms = {n_steps} 步；stream={target}；只列 ≥0.5 次/步的项")
    print(f"{'per_step':>9}{'ms/step':>10}  {'OP Type':<34} 输入形状")
    for _, r in grp.head(45).iterrows():
        shape = str(r["Input Shapes"])[:58]
        print(f"{r['per_step']:>9.2f}{r['ms_per_step']:>10.4f}  {r['OP Type'][:34]:<34} {shape}")
        _ = r
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
