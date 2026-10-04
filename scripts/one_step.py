#!/usr/bin/env python3
"""打印一个 decode 步的跨 stream 时间线（只列 >=min_us 的算子），定位阶段边界。
用法: one_step.py <profdir> <min_us> [step_index]
"""
import glob, sys
import pandas as pd
import numpy as np
M = sys.argv[1]; MIN_US = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
# 用主导 stream 的大间隙作为步边界
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
g = df[df["Stream ID"] == 146].sort_values("s").reset_index(drop=True)
gaps = (g["s"].shift(-1) - g["e"])
starts = g.loc[gaps[gaps >= 2.0].index + 1, "s"].to_numpy()
print(f"步边界数 = {len(starts)}")
if len(starts) < K + 2:
    K = max(1, len(starts) - 2)
lo, hi = starts[K], starts[K+1]
print(f"=== 第 {K} 步：{lo:.3f} – {hi:.3f} ms（{hi-lo:.2f} ms）===")
w = df[(df["s"] >= lo) & (df["s"] < hi) & (df["Task Duration(us)"] >= MIN_US)].sort_values("s")
print(f"{'t-rel(ms)':>10} {'dur(us)':>9} {'stream':>7} {'type':>16}  op")
for _, r in w.iterrows():
    print(f"{r['s']-lo:10.3f} {r['Task Duration(us)']:9.1f} {str(r['Stream ID']):>7} "
          f"{str(r['Task Type'])[:16]:>16}  {str(r['OP Type'])[:42]}")
