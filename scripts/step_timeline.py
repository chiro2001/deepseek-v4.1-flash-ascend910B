#!/usr/bin/env python3
"""一个 decode 步内、主导 stream 的时间线（含大间隙），定位"在等谁"。
用法: step_timeline.py <profdir> <stream_id> [gap_ms]
"""
import glob, sys
import pandas as pd
import numpy as np
M = sys.argv[1]; SID = float(sys.argv[2]); GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
g = w[w["Stream ID"] == SID].sort_values("s").reset_index(drop=True)
print(f"stream {SID}: {len(g)} 算子")
gaps = g["s"].shift(-1) - g["e"]
big = gaps[gaps > GAP]
print(f">= {GAP*1000:.0f}us 的间隙: {len(big)} 个, 合计 {big.sum():.1f} ms "
      f"({big.sum()/(w['s'].max()-w['s'].min())*100:.1f}% 窗口)")
print(f"平均每步大间隙数 ≈ {len(big)/251:.1f}, 每步耗时 ≈ {big.sum()/251:.2f} ms")
print("\n=== 最大 12 个间隙及其前后 ===")
idx = big.sort_values(ascending=False).head(12).index
for i in sorted(idx):
    prev = g.loc[i]; nxt = g.loc[i+1]
    print(f"  gap={gaps[i]*1000:8.1f}us  t={prev['e']/1000:7.3f}s  "
          f"{str(prev['OP Type'])[:26]:26s} -> {str(nxt['OP Type'])[:26]}")
print("\n=== 间隙大小分布 ===")
for lo, hi in [(0.1,0.2),(0.2,0.5),(0.5,1.0),(1.0,2.0),(2.0,10.0)]:
    sel = big[(big>=lo)&(big<hi)]
    print(f"  [{lo},{hi}) ms: {len(sel):5d} 个, 合计 {sel.sum():7.1f} ms")
