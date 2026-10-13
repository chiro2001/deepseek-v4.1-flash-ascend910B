#!/usr/bin/env python3
"""指定 stream 的算子构成 + 每步量。用法: stream_breakdown.py <profdir> <stream_id>
"""
import glob, sys
import pandas as pd
import numpy as np
M = sys.argv[1]; SID = float(sys.argv[2])
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID","Input Shapes"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
win = w["s"].max() - w["s"].min()
g = w[w["Stream ID"] == SID].sort_values("s")
busy = (g["e"] - g["s"]).sum()
# 用 gmm1 簇数估计步数
gmm = w[w["OP Type"] == "GroupedMatmulSwigluQuantV2"]
steps = max(1, len(gmm) // 43)
print(f"窗口 {win:.1f} ms  步数≈{steps}  stream {SID}: {len(g)} 算子, busy {busy:.1f} ms "
      f"({busy/win*100:.1f}% 窗口, {busy/steps:.2f} ms/步)")
agg = g.assign(dur=(g["e"]-g["s"])*1000).groupby("OP Type")["dur"].agg(["count","sum"])
agg["sum"] = agg["sum"]/1000
agg["cnt/步"] = (agg["count"]/steps).round(1)
agg["us/步"] = (agg["sum"]*1000/steps).round(1)
agg["mean_us"] = (agg["sum"]*1000/agg["count"]).round(2)
print(f"\n=== stream {SID} 算子构成（按总时长降序 top 20）===")
print(agg.sort_values("sum", ascending=False).head(20)[["count","cnt/步","us/步","mean_us"]].to_string())
print(f"\n占该 stream 时长比例（top 8）:")
tot = agg["sum"].sum()
for k, v in agg.sort_values("sum", ascending=False).head(8)["sum"].items():
    print(f"   {v/tot*100:5.1f}%  {v:8.1f} ms  {k}")
