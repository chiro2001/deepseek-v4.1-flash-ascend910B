#!/usr/bin/env python3
"""对比两次 profile 的 AIC/AIV 占用与重叠（用图捕获边界切步）。"""
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
TAG = sys.argv[2]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df[df["dur"] > 0].sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
# 用 allgatherAicpuKernel 作步标志（每步一次）
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
if len(marks) < 3:
    # 退化：用大间隔切
    u = sorted(df["st"].unique()); marks = [u[0]]
    for i in range(1, len(u)):
        if u[i] - u[i-1] > 3000: marks.append(u[i])
    marks.append(df["en"].max())
print("[%s] 算子 %d，步 %d" % (TAG, len(df), len(marks) - 1))

CAT = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV",
       "MIX_AIV": "AIV", "COMMUNICATION": "COMM", "AI_CPU": "CPU"}
df["cat"] = df["core"].map(CAT).fillna("?")


def union(sub):
    iv = sorted(zip(sub["st"].values, sub["en"].values))
    if not iv: return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return out


def inter(a, b):
    t = 0.0; i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: t += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return t / 1000


# 稳态：跳过首尾各 1 步
LO, HI = marks[1], marks[-2]
w = df[(df["st"] >= LO) & (df["st"] < HI)]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst
print("  稳态 %d 步，步长 %.2f ms，算子 %.0f/步" % (nst, STEP, len(w) / nst))
res = {}
for c in ("AIC", "AIV", "COMM", "CPU"):
    s = w[w["cat"] == c]
    iv = union(s)
    res[c] = sum(e - x for x, e in iv) / 1000 / nst
    print("  %-5s busy %7.3f ms/步  (%.0f%%)  算子 %6.0f/步" % (c, res[c], 100 * res[c] / STEP, len(s) / nst))
for a, b in (("AIC", "AIV"), ("AIC", "COMM"), ("AIV", "COMM"), ("AIC", "CPU")):
    ov = inter(union(w[w["cat"] == a]), union(w[w["cat"] == b])) / nst
    print("  %s ∩ %s = %6.3f ms  (%.1f%% of %s busy)" % (a, b, ov, 100 * ov / max(res[a], 1e-9), a))
print("  ⇒ AIC 利用率 %.1f%%；AIC 空转 %.3f ms/步" % (100 * res["AIC"] / STEP, STEP - res["AIC"]))
