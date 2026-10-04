#!/usr/bin/env python3
"""同时移除三类 metadata 的独占收益（上限）。用法: meta_excl.py <mindstudio_profiler_output>"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = df["Task Start Time(us)"] - t0
df["e"] = df["s"] + df["Task Duration(us)"]
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut]
META = ["SparseFlashMlaMetadata","SparseAttnSharedkvMetadata","QuantLightningIndexerV2Metadata"]

def union_ms(a):
    if len(a) == 0: return 0.0
    a = a[np.argsort(a[:,0])]
    tot=0.0; cs,ce=a[0]
    for s,e in a[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    return (tot+ce-cs)/1000.0

allv = w[["s","e"]].to_numpy()
keep = w[~w["OP Type"].isin(META)][["s","e"]].to_numpy()
U, Uk = union_ms(allv), union_ms(keep)
win = (w["s"].max()-w["s"].min())/1000.0
print(f"稳态窗口          = {win:9.1f} ms")
print(f"union(全部)       = {U:9.1f} ms  (占窗口 {U/win*100:.1f}%)")
print(f"union(去掉metadata)= {Uk:9.1f} ms")
print(f"→ 三类 metadata 的**合计独占** = {U-Uk:8.1f} ms = 墙钟的 {(U-Uk)/win*100:5.2f}%")
for m in META:
    sub = w[w["OP Type"]==m][["s","e"]].to_numpy()
    print(f"   {m:34s} 并集 {union_ms(sub):7.1f} ms  {len(sub):5d} 次")
