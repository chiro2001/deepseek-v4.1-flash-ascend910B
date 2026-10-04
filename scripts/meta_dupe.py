#!/usr/bin/env python3
"""检查 metadata 行是否双记账（同一 (start,dur) 出现多次）。"""
import glob, sys
import pandas as pd
M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True)
for otype in ["SparseFlashMlaMetadata","SparseAttnSharedkvMetadata","QuantLightningIndexerV2Metadata"]:
    sub = df[df["OP Type"] == otype]
    if len(sub) == 0:
        continue
    sig = sub.assign(k=sub["Task Start Time(us)"].round(3).astype(str) + "|" +
                       sub["Task Duration(us)"].round(3).astype(str))
    n_all = len(sig); n_uni = sig["k"].nunique()
    print(f"{otype:34s} rows={n_all:5d} unique(start,dur)={n_uni:5d} "
          f"倍数={n_all/max(n_uni,1):.2f}  TaskType={sub['Task Type'].unique()[:3]} "
          f"Stream={sub['Stream ID'].unique()[:4]}")
    if n_all > n_uni:
        d = sig.groupby("k").size()
        print(f"    重复分布: {d.value_counts().to_dict()}")
        print(f"    Op Name 种类: {sub['Op Name'].nunique()}  {sub['Op Name'].unique()[:3]}")
