#!/usr/bin/env python3
"""主导 stream 的每个大间隙里，其它 stream 在跑什么（按 Op Type 聚合）。
用法: gap_other_streams.py <profdir> <main_stream> [gap_ms]
"""
import glob, sys, collections
import pandas as pd
import numpy as np
M = sys.argv[1]; SID = float(sys.argv[2]); GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
g = w[w["Stream ID"] == SID].sort_values("s").reset_index(drop=True)
gaps = (g["s"].shift(-1) - g["e"])
big = gaps[gaps >= GAP]
print(f"主导 stream {SID}: {len(big)} 个大间隙(>= {GAP}ms), 合计 {big.sum():.1f} ms")
a = w[w["Stream ID"] != SID]
asi = a.sort_values("s")[["s","e","OP Type","Stream ID"]].to_numpy()
byst = collections.defaultdict(float); byop = collections.defaultdict(float)
tot = 0.0
for i, gp in big.items():
    lo, hi = g.loc[i,"e"], g.loc[i+1,"s"]
    tot += (hi-lo)
    lo_i = np.searchsorted(asi[:,0].astype(float), hi, "right")
    for k in range(max(0,lo_i-4000), lo_i):
        s,e,op,sd = asi[k]
        s=float(s); e=float(e)
        if e <= lo or s >= hi: continue
        ov = min(e,hi)-max(s,lo)
        byst[str(sd)] += ov; byop[str(op)] += ov
print(f"\n=== 间隙内其它 stream 的占用（ms, 占间隙总长 {tot:.1f}ms）===")
for k,v in sorted(byst.items(), key=lambda x:-x[1])[:8]:
    print(f"  stream {k:>7s}: {v:8.1f} ms  ({v/tot*100:5.1f}%)")
print(f"\n=== 间隙内跑得最多的算子（top 15）===")
for k,v in sorted(byop.items(), key=lambda x:-x[1])[:15]:
    print(f"  {v:8.1f} ms  {k[:52]}")
