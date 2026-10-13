#!/usr/bin/env python3
"""间隙内按 0.2ms 桶统计：任务数 / busy / idle / 涉及哪些 stream。
用法: gap_buckets.py <profdir> <lo_ms> <hi_ms>
"""
import glob, sys
import pandas as pd
import numpy as np
M, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
sub = df[(df["s"] >= lo - 0.02) & (df["s"] < hi + 0.02)].sort_values("s").reset_index(drop=True)
print("任务总数:", len(sub))
bins = np.arange(lo, hi + 0.2, 0.2)
print("%7s %5s %9s %9s  %s" % ("t-rel", "n", "busy_ms", "idle_ms", "streams"))
tot_idle = 0.0
for a, b in zip(bins[:-1], bins[1:]):
    sel = sub[(sub["s"] < b) & (sub["e"] > a)]
    busy = (sel["e"].clip(a, b) - sel["s"].clip(a, b)).sum() if len(sel) else 0.0
    st = ",".join(sorted(set(str(x) for x in sel["Stream ID"].unique())))[:22]
    print("%7.2f %5d %9.3f %9.3f  %s" % (a - lo, len(sel), busy, (b - a) - busy, st))
    tot_idle += (b - a) - busy
print(f"\n桶内合计 idle（无人跑任何任务）= {tot_idle:.3f} ms / {hi-lo:.2f} ms = {tot_idle/(hi-lo)*100:.1f}%")
