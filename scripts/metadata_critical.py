#!/usr/bin/env python3
"""判定 metadata 类 AI_CPU 任务是否在关键路径：算"落在设备忙区间之外"的比例。
用法: metadata_critical.py <mindstudio_profiler_output>
"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary_slice_*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = df["Task Start Time(us)"] - t0
df["e"] = df["s"] + df["Task Duration(us)"]

META = {"SparseFlashMlaMetadata","SparseAttnSharedkvMetadata","QuantLightningIndexerV2Metadata"}
# 稳态窗口：后 70%
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut]
dev = w[w["Task Type"] != "AI_CPU"][["s","e"]].to_numpy()
met = w[w["OP Type"].isin(META)][["s","e"]].to_numpy()

def union_ms(arr):
    if len(arr) == 0: return 0.0
    a = arr[np.argsort(arr[:,0])]
    tot=0.0; cs,ce=a[0]
    for s,e in a[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    tot+=ce-cs
    return tot/1000.0

U = union_ms(dev); Um = union_ms(met)
# metadata 落在设备忙区间之外的部分
dev_sorted = dev[np.argsort(dev[:,0])]
merged=[]
for s,e in dev_sorted:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
    else: merged.append([s,e])
outside=0.0
for s,e in met:
    cover=0.0
    for a,b in merged:
        if b<=s or a>=e: continue
        cover += min(b,e)-max(a,s)
    outside += (e-s)-cover
print(f"稳态窗口 {w['s'].min()/1e6:.2f}–{w['s'].max()/1e6:.2f}s")
print(f"设备并集     = {U:9.1f} ms")
print(f"metadata 并集= {Um:9.1f} ms  （{len(met)} 次调用）")
print(f"metadata 落在设备忙区间**之外** = {outside/1000:9.1f} ms  "
      f"= 设备并集的 {outside/1000/U*100:5.2f}%")
per_step = U  # 用设备并集近似总步时
print(f"→ 若把这部分消掉，墙钟最多省 {outside/1000/U*100:.2f}%")
