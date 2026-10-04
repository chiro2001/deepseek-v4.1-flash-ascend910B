#!/usr/bin/env python3
"""把窗口切成固定桶，看 AI core 空闲的分布（均匀散布 = 可填的同步气泡；集中 = 结构性）。"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
BUCKET_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Task Start Time(us)", "Task Duration(us)", "Task Type"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut]
core = w[w["Task Type"] != "AI_CPU"][["s", "e"]].to_numpy()

t_start, t_end = w["s"].min(), w["s"].max()
edges = np.arange(t_start, t_end + BUCKET_MS, BUCKET_MS)
idle = []
for a, b in zip(edges[:-1], edges[1:]):
    seg = core[(core[:, 1] > a) & (core[:, 0] < b)].copy()
    if len(seg) == 0:
        cov = 0.0
    else:
        seg[:, 0] = np.clip(seg[:, 0], a, b)
        seg[:, 1] = np.clip(seg[:, 1], a, b)
        seg = seg[np.argsort(seg[:, 0])]
        tot = 0.0
        cs, ce = seg[0]
        for s, e in seg[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                tot += ce - cs
                cs, ce = s, e
        cov = tot + ce - cs
    idle.append(1.0 - cov / (b - a))
idle = np.array(idle)
print("桶大小=%gms 桶数=%d 平均空闲=%.1f%%" % (BUCKET_MS, len(idle), idle.mean() * 100))
for q in (10, 25, 50, 75, 90):
    print("  p%-3d 空闲率 = %6.2f%%" % (q, np.percentile(idle, q) * 100))
print("  完全无空闲(<1%%)的桶占比 = %5.1f%%" % ((idle < 0.01).mean() * 100))
print("  空闲 >50%% 的桶占比      = %5.1f%%" % ((idle > 0.5).mean() * 100))
