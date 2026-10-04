#!/usr/bin/env python3
"""空闲间隔的粒度：是"每步一个 ~4ms 大洞"还是"很多小洞"。
用法: idle_gaps.py <mindstudio_profiler_output>
"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary_slice_*.csv"))
cols = ["Task Start Time(us)", "Task Duration(us)", "Task Type"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut]
core = w[w["Task Type"] != "AI_CPU"][["s", "e"]].to_numpy()
core = core[np.argsort(core[:, 0])]
# 合并 AI core 区间
merged = []
for s, e in core:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
merged = np.array(merged)
gaps = merged[1:, 0] - merged[:-1, 1]
gaps = gaps[gaps > 0.005]            # >5us
total = gaps.sum()
print("AI core 忙区间数 = %d  空闲洞数 = %d" % (len(merged), len(gaps)))
print("空闲合计 = %.1f ms  占窗口 %.2f%%" % (total, total / (merged[-1,1]-merged[0,0]) * 100))
for q in (50, 75, 90, 95, 99):
    print("  p%-3d 洞大小 = %8.3f ms" % (q, np.percentile(gaps, q)))
print("  max         = %8.3f ms" % gaps.max())
for th in (0.05, 0.2, 0.5, 1.0, 2.0):
    sel = gaps[gaps >= th]
    print("  洞 >= %4.2f ms : %5d 个, 合计 %8.1f ms (占空闲 %.1f%%)"
          % (th, len(sel), sel.sum(), sel.sum() / total * 100))
