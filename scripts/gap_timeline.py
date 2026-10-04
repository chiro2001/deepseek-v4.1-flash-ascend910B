#!/usr/bin/env python3
"""dump 主导 stream 某个间隙内的全部任务（按时间排序），看串行链。
用法: gap_timeline.py <profdir> [gap_index] [min_us]
"""
import glob, sys
import pandas as pd
import numpy as np
M = sys.argv[1]; GI = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MIN_US = float(sys.argv[3]) if len(sys.argv) > 3 else 20
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
g = w[w["Stream ID"] == 146].sort_values("s").reset_index(drop=True)
gaps = (g["s"].shift(-1) - g["e"])
big = gaps[gaps >= 2.0]
idx = list(big.index)
i = idx[min(GI, len(idx)-1)]
lo, hi = g.loc[i, "e"], g.loc[i+1, "s"]
print(f"间隙 #{i}: {lo:.3f} – {hi:.3f} ms（长 {hi-lo:.2f} ms）")
sub = w[(w["s"] >= lo-0.05) & (w["s"] < hi+0.05) & (w["Task Duration(us)"] >= MIN_US)].sort_values("s")
print(f"{'t-rel':>9} {'dur(us)':>9} {'stream':>7} {'type':>16}  op")
for _, r in sub.iterrows():
    print(f"{r['s']-lo:9.3f} {r['Task Duration(us)']:9.1f} {str(r['Stream ID']):>7} "
          f"{str(r['Task Type'])[:16]:>16}  {str(r['OP Type'])[:40]}")
# 各 stream 在间隙内的占用
print("\n=== 各 stream 在间隙内的 busy（ms）===")
for sd, gg in sub.groupby("Stream ID"):
    busy = (gg["e"].clip(lo, hi) - gg["s"].clip(lo, hi)).sum()
    if busy > 0.05:
        print(f"  stream {sd:>7}: {busy:6.2f} ms  ({busy/(hi-lo)*100:5.1f}%)  n={len(gg)}")
