#!/usr/bin/env python3
"""对比 HcPre 的设备时长与 AIC/AIV 构成。用法: hcpre_cmp.py <profdir> [tag]"""
import glob, sys
import pandas as pd
M = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type",
        "aicore_time(us)","aic_scalar_time(us)","aiv_time(us)","aiv_scalar_time(us)","Input Shapes"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True)
sub = df[df["OP Type"] == "HcPre"].sort_values("Task Start Time(us)")
print(f"--- {TAG}  n={len(sub)}")
print(sub[["Task Duration(us)","aicore_time(us)","aic_scalar_time(us)","aiv_time(us)","aiv_scalar_time(us)"]]
      .median().round(2).to_string())
print("shapes:", sub["Input Shapes"].value_counts().head(3).to_dict())
