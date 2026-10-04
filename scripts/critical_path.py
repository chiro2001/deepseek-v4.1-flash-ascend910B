#!/usr/bin/env python3
"""正向找关键链：按 stream 找"零间隙"连续算子链（前驱结束即后继开始），
最长的那条链 + 其所在 stream 就是关键链的候选。
用法: critical_path.py <mindstudio_profiler_output> [gap_thresh_ms]
"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02   # 20us 内视为"紧接"
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type", "Task Start Time(us)", "Task Duration(us)", "Task Type", "Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
win = w["s"].max() - w["s"].min()
print(f"窗口 {win:.1f} ms  行数 {len(w)}  stream 数 {w['Stream ID'].nunique()}")

# 按 stream 统计（只统计 AI core / vector 类，排除 AICPU 与 host）
core = w[w["Task Type"].isin(["AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"]) |
         w["Task Type"].astype(str).str.contains("CORE|MIX", na=False)].copy()
print(f"AI core 类任务 {len(core)}")
rows = []
for sd, g in core.groupby("Stream ID"):
    g = g.sort_values("s")
    busy = (g["e"] - g["s"]).sum()
    gaps = (g["s"].shift(-1) - g["e"]).dropna()
    tight = gaps[gaps <= TH]
    rows.append((str(sd), len(g), busy, busy / win * 100,
                 len(gaps), len(tight), (tight.sum() if len(tight) else 0.0)))
r = pd.DataFrame(rows, columns=["stream", "n", "busy_ms", "busy%", "gaps", "tight", "tight_ms"])
r = r.sort_values("busy_ms", ascending=False)
print("\n=== 按 stream（busy 降序）top 12 ===")
print(r.head(12).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
print(f"\n所有 stream busy 之和 = {r['busy_ms'].sum():.1f} ms "
      f"（{r['busy_ms'].sum()/win*100:.1f}% of 窗口；>100% 说明跨 stream 并行）")

# 全局最长的零间隙链
print(f"\n=== 最长零间隙链（<= {TH*1000:.0f} us） ===")
best = None
for sd, g in core.groupby("Stream ID"):
    g = g.sort_values("s").reset_index(drop=True)
    i = 0
    while i < len(g):
        j = i
        while j + 1 < len(g) and (g.loc[j+1, "s"] - g.loc[j, "e"]) <= TH:
            j += 1
        if j > i:
            span = g.loc[j, "e"] - g.loc[i, "s"]
            if best is None or span > best[0]:
                best = (span, str(sd), i, j, g)
        i = j + 1
if best:
    span, sd, i, j, g = best
    print(f"最长链：stream {sd}，{j-i+1} 个算子，跨度 {span:.2f} ms")
    ops = g.loc[i:j+1, "OP Type"].value_counts().head(12)
    print("链内算子构成（top 12）:")
    for k, v in ops.items():
        print(f"   {v:5d}  {k}")
    print(f"链跨度 / 窗口 = {span/win*100:.1f}%")
