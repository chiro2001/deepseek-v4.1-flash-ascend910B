#!/usr/bin/env python3
"""纯 AIV 算子的完整构成（按个数），以及 AIV 块数分布。"""
import sys
import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur"})
df = df.sort_values("st")
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
lo, hi = marks[2], marks[-3]
w = df[(df["st"] >= lo) & (df["st"] < hi)]
nst = len([m for m in marks if lo <= m < hi])
av = w[w["Accelerator Core"] == "AI_VECTOR_CORE"]
print("纯 AIV：%.0f 个/步，%.3f ms/步，%d 种算子\n" % (len(av)/nst, av["dur"].sum()/1000/nst, av["name"].nunique()))
g = av.groupby("name").agg(n=("dur", lambda s: s.size/nst), ms=("dur", lambda s: s.sum()/1000/nst),
                           med=("dur", "median"),
                           blk=("Block Num", lambda s: pd.to_numeric(s, errors="coerce").median()))
print("%-56s %9s %9s %8s %6s" % ("算子", "个数/步", "ms/步", "中位µs", "块数"))
for name, r in g.sort_values("n", ascending=False).head(22).iterrows():
    print("%-56s %9.1f %9.3f %8.1f %6.0f" % (str(name)[:56], r["n"], r["ms"], r["med"], r["blk"]))
print("\n合计覆盖 %.0f 个/步（%.0f%%）" % (g["n"].head(22).sum(), 100*g["n"].head(22).sum()/(len(av)/nst)))
print("\n=== AIV 块数分布 ===")
blk = pd.to_numeric(av["Block Num"], errors="coerce").dropna()
for k in (1, 2, 4, 6, 8, 12, 16, 24, 48):
    n = (blk == k).sum()
    if n: print("  %2d 块: %7d 个 (%.1f%%)  时长合计 %7.1f ms" % (k, n, 100*n/len(blk),
                                                                 av.loc[blk[blk==k].index, "dur"].sum()/1000/nst))
print("  中位 %.0f 块，均值 %.1f 块" % (blk.median(), blk.mean()))
