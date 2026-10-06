#!/usr/bin/env python3
"""逐算子暴露度（归并式，快）。"""
import sys
import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
LO, HI = marks[2], marks[-3]
w = df[(df["st"] >= LO) & (df["st"] < HI)].copy()
nst = len([m for m in marks if LO <= m < HI])
STEP_P = (HI - LO) / 1000 / nst
K = STEP_P / 24.59


def merged(sub):
    iv = sorted(zip(sub["st"].values, sub["en"].values))
    if not iv: return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return out


def overlap(a, b):
    tot = 0.0; i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: tot += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot / 1000


CANDS = ["aclnnScatterNdUpdateSk", "RmsNorm", "DynamicQuant", "DequantSwigluQuant",
         "HcPre", "HcPost", "InplacePartialRotaryMul", "SparseFlashMla",
         "aclnnMatmul_MatMulV2", "aclnnQuantMatmulWeightNz",
         "GroupedMatmulSwigluQuant", "GroupedMatmulWeightNz_GroupedMatmul",
         "aclnnMatmul_MatMulV3", "aclnnAdd_AddAiCore", "AivKernel", "aclnnSparseFlashMlaMetadata"]
print("步长 profile %.2f ms，K=%.3f" % (STEP_P, K))
print("%-40s %8s %9s %9s %12s" % ("算子", "自身ms", "并集ms", "暴露ms", "真实暴露ms"))
tot = 0.0
names = w["name"].astype(str)
for c in CANDS:
    mask = names.str.contains(c, regex=False, na=False)
    g = w[mask]
    if g.empty: continue
    own = g["dur"].sum() / 1000 / nst
    iv = merged(g)
    u = sum(e - s for s, e in iv) / 1000 / nst
    oth = merged(w[~mask])
    exp = u - overlap(iv, oth) / nst
    real = exp / K
    tot += real
    print("%-40s %8.3f %9.3f %9.3f %9.3f (%.1f%%)" % (c[:40], own, u, exp, real, 100 * real / 24.59))
print("\n合计真实暴露 %.2f ms/步" % tot)
