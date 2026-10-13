#!/usr/bin/env python3
"""HcPre 精确形状匹配（修子串污染）+ 取 Track B 要的列。
用法: hcpre_mask.py <profdir> [shape_substr] [tag]
"""
import glob, sys
import pandas as pd
M = sys.argv[1]
SHAPE = sys.argv[2] if len(sys.argv) > 2 else "6,4,5120"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Input Shapes",
        "aic_mac_time(us)","aic_total_cycles","aiv_total_cycles","Block Num","Mix Block Num",
        "Task Wait Time(us)","aicore_time(us)","aic_scalar_time(us)","aiv_time(us)"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True)
sub = df[df["OP Type"] == "HcPre"].copy()
# ★ 精确匹配：Input Shapes 的第 1 段以 SHAPE 开头（避免 "6," 命中 "16,"）
def first_seg(s):
    s = str(s).strip('"')
    return s.split(";")[0]
sub["seg0"] = sub["Input Shapes"].map(first_seg)
exact = sub[sub["seg0"] == SHAPE]
substr = sub[sub["seg0"].str.startswith(SHAPE) & (sub["seg0"] != SHAPE)]
print(f"--- {TAG}  总 HcPre={len(sub)}  精确 '{SHAPE}'={len(exact)}  被旧子串误纳={len(substr)}")
if len(substr):
    print(f"    误纳的 seg0 分布: {substr['seg0'].value_counts().head(5).to_dict()}")
if len(exact) == 0:
    print("    无精确匹配"); sys.exit()
k = [c for c in ["Task Duration(us)","aicore_time(us)","aic_scalar_time(us)","aiv_time(us)",
                 "aic_mac_time(us)","aic_total_cycles","aiv_total_cycles",
                 "Block Num","Mix Block Num","Task Wait Time(us)"] if c in exact.columns]
print(exact[k].median().to_string())
