#!/usr/bin/env python3
"""AI_CPU 的完整构成 + 暴露程度（A3 的目标）。"""
import sys
import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
LO, HI = marks[2], marks[-3]
w = df[(df["st"] >= LO) & (df["st"] < HI)].copy()
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst
K = STEP / (24.59 / 1000 * 1000) if False else STEP / 24.59   # profile→真实

def union(sub):
    iv = sorted(zip(sub["st"], sub["en"]))
    if not iv: return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return out

def cover(iv, other):
    tot = 0.0
    for s, e in iv:
        for s2, e2 in other:
            if s2 >= e: break
            tot += max(0.0, min(e, e2) - max(s, s2))
    return tot / 1000

cpu = w[w["core"] == "AI_CPU"]
oth = union(w[w["core"] != "AI_CPU"])
civ = union(cpu)
csum = cpu["dur"].sum() / 1000 / nst
cu = sum(e - s for s, e in civ) / 1000 / nst
cov = cover(civ, oth) / nst
print("AI_CPU: %.1f 个/步 | 原始合计 %.3f ms | union %.3f ms | 被别的资源盖住 %.3f" % (
    len(cpu) / nst, csum, cu, cov))
print("  ⇒ **暴露 %.3f profile ms = %.3f 真实 ms/步（%.1f%% of step）**" % (cu - cov, (cu - cov) / K, 100 * (cu - cov) / STEP))
print()
print("%-52s %8s %9s %9s" % ("算子", "个数/步", "ms/步", "中位µs"))
g = cpu.groupby("name").agg(n=("dur", lambda s: s.size / nst), ms=("dur", lambda s: s.sum() / 1000 / nst),
                            med=("dur", "median"))
for n, r in g.sort_values("ms", ascending=False).head(14).iterrows():
    print("%-52s %8.1f %9.3f %9.1f" % (str(n)[:52], r["n"], r["ms"], r["med"]))
print("\n合计 Top14 = %.3f ms" % g.sort_values("ms", ascending=False).head(14)["ms"].sum())
