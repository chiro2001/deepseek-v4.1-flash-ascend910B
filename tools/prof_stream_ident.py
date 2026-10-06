#!/usr/bin/env python3
"""给每条流"命名"：它到底在算什么；并定位关键算子在步内的位置。"""
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur", "Stream ID": "sid"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
LO, HI = marks[2], marks[-3]
w = df[(df["st"] >= LO) & (df["st"] < HI)].copy()
nst = len([m for m in marks if LO <= m < HI])
T = marks[3] - marks[2]

print("=== 每条流的完整算子清单（Top 12 by 个数）===")
rows = []
for sid, g in w.groupby("sid"):
    tot = g["dur"].sum() / 1000 / nst
    if tot < 0.5: continue
    rows.append((tot, sid, g))
for tot, sid, g in sorted(rows, reverse=True)[:10]:
    print("\n-- stream %s（%.2f ms/步，%.0f 算子/步）" % (sid, tot, len(g) / nst))
    c = Counter(str(k)[:44] for k in g["name"])
    for name, n in c.most_common(10):
        sub = g[g["name"].astype(str).str.startswith(name[:40])]
        print("     %-46s %7.1f/步  中位 %6.1f µs" % (name, n / nst, sub["dur"].median()))

print("\n\n=== 关键算子在步内的位置（相对 allgather 标志）===")
KEY = ["SparseFlashMla", "GroupedMatmulSwigluQuantWeightNz", "GroupedMatmulWeightNz_GroupedMatmul",
       "MoeGatingTopKHash", "MoeInitRoutingV3", "SparseFlashMlaMetadata", "SparseAttnSharedkvMetadata",
       "Markov", "sampling", "ArgMax", "TopK", "RmsNorm", "HcPre", "HcPost"]
print("%-46s %8s %8s  位置分布(0..100pct)" % ("算子", "个数/步", "ms/步"))
for k in KEY:
    g = w[w["name"].astype(str).str.contains(k, case=False, na=False)]
    if g.empty: continue
    hs = []
    for m in marks[2:-2]:
        seg = g[(g["st"] >= m) & (g["st"] < m + T)]
        if seg.empty: continue
        hs.extend(((seg["st"] - m) / T).tolist())
    if not hs: continue
    import bisect
    hh = [0] * 10
    for f in hs: hh[min(9, int(f * 10))] += 1
    tot_all = sum(hh) or 1
    print("%-46s %8.1f %8.3f  %s" % (k[:46], len(g) / nst, g["dur"].sum() / 1000 / nst,
                                     " ".join("%4.0f" % (100 * x / tot_all) for x in hh)))
