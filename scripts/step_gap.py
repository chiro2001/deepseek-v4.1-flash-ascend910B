#!/usr/bin/env python3
"""用 gmm1(43次/步) 切步，输出：簇内设备跨度 vs 簇间隔（gap）。
判据：若簇内跨度 ≈ 步周期、gap≈0 ⇒ 设备忙满（device-bound）；
      若簇内跨度 << 步周期、gap 大 ⇒ host 开销主导。
用法: step_gap.py <mindstudio_profiler_output> [gmm_per_step]
"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
GPS = int(sys.argv[2]) if len(sys.argv) > 2 else 43
files = sorted(glob.glob(M + "/op_summary_slice_*.csv"))
cols = ["OP Type", "Task Start Time(us)", "Task Duration(us)"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files], ignore_index=True)
df = df.sort_values("Task Start Time(us)").reset_index(drop=True)
g = df[df["OP Type"] == "GroupedMatmulSwigluQuantV2"]
starts = g["Task Start Time(us)"].to_numpy()
print(f"gmm1={len(starts)}  steps={len(starts)//GPS}  span={(starts[-1]-starts[0])/1e6:.2f}s")

# 已按时间排序的全表，用于取每簇的设备跨度
allst = df["Task Start Time(us)"].to_numpy()
alldur = df["Task Duration(us)"].to_numpy()
allend = allst + alldur

starts_sel = starts[: (len(starts)//GPS)*GPS].reshape(-1, GPS)
rows = []
for i in range(len(starts_sel)-1):
    b0, b1 = starts_sel[i][0], starts_sel[i+1][0]
    lo = np.searchsorted(allst, b0, "left")
    hi = np.searchsorted(allst, b1, "left")
    if hi <= lo:
        continue
    dev_span = allend[lo:hi].max() - allst[lo:hi].min()
    rows.append((b0, b1-b0, dev_span, (b1-b0)-dev_span))
r = pd.DataFrame(rows, columns=["t0", "period", "dev_span", "gap"])
# 稳态：取后 60%
r = r.iloc[int(len(r)*0.4):]
print(f"steady steps={len(r)}")
for c in ["period", "dev_span", "gap"]:
    print(f"  {c:9s} p50={r[c].median()/1000:7.2f}ms  p10={r[c].quantile(.1)/1000:7.2f}  p90={r[c].quantile(.9)/1000:7.2f}")
print(f"  设备占步周期比例 p50 = {(r['dev_span']/r['period']).median()*100:.1f}%")
