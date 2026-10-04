#!/usr/bin/env python3
"""服务 profile 里三类 metadata 的 Task Duration 分布。"""
import glob, sys
import pandas as pd
M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Wait Time(us)","Input Shapes"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True)
df = df.sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = df["Task Start Time(us)"] - t0
w = df[df["s"] >= df["s"].max() * 0.3]
for ot in ["SparseFlashMlaMetadata","SparseAttnSharedkvMetadata","QuantLightningIndexerV2Metadata"]:
    sub = w[w["OP Type"] == ot]["Task Duration(us)"]
    if len(sub) == 0: continue
    q = sub.quantile([.05,.25,.5,.75,.95]).round(1).to_dict()
    print(f"{ot:34s} n={len(sub):5d}  p5={q[0.05]:7.1f} p25={q[0.25]:7.1f} "
          f"p50={q[0.5]:7.1f} p75={q[0.75]:7.1f} p95={q[0.95]:7.1f}  mean={sub.mean():7.1f}")
    # 前 3 个形状
    print(f"     shapes: {w[w['OP Type']==ot]['Input Shapes'].value_counts().head(2).to_dict()}")
