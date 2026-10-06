#!/usr/bin/env python3
"""定位 draft（DSpark proposal）的算子在时间线上的位置。

线索：HcPre/HcPost 每层各 2 次 ⇒ 86/步 = 43 层 = 40 主层 + 3 proposal 层。
若 draft 与主层同流，二者无法区分；若在侧流则可分离。
另外查 draft 专有算子名（markov / proposal / eagle / mtp / confidence）。
"""
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

print("=== 找 draft 专有算子名 ===")
for kw in ("markov", "propos", "eagle", "mtp", "confidence", "draft", "spec", "reject", "accept"):
    hits = w[w["name"].astype(str).str.contains(kw, case=False, na=False)]
    if not hits.empty:
        print("  '%s': %d 个不同算子, %.1f 个/步, %.3f ms/步" % (
            kw, hits["name"].nunique(), len(hits) / nst, hits["dur"].sum() / 1000 / nst))
        for n, c in Counter(hits["name"].astype(str)).most_common(3):
            print("      %-52s %6.1f/步" % (n[:52], c / nst))

print("\n=== 层计数核对（每层应有 2 次 HcPre / 2 次 HcPost）===")
for nm in ("HcPre", "HcPost", "MoeGatingTopKHash", "SparseFlashMla", "RmsNorm"):
    g = w[w["name"].astype(str).str.fullmatch(nm, na=False)]
    if g.empty:
        g = w[w["name"].astype(str).str.startswith(nm, na=False)]
    print("  %-22s %8.1f 个/步" % (nm, len(g) / nst))

print("\n=== 主流内 AIC 与 AIV 的块长（连续同类的算子数）===")
CAT = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV", "MIX_AIV": "AIV"}
w2 = w.copy(); w2["cat"] = w2["Accelerator Core"].map(CAT)
main_sid = w2.groupby("sid")["dur"].sum().idxmax()
m = w2[w2["sid"] == main_sid].sort_values("st")
seq = [c for c in m["cat"] if c]
runs = []
cur, cnt, dur = seq[0], 0, 0.0
durs = list(m[m["cat"].notna()]["dur"])
k = 0
for c in seq:
    d = durs[k]; k += 1
    if c == cur: cnt += 1; dur += d
    else: runs.append((cur, cnt, dur / 1000)); cur, cnt, dur = c, 1, d
runs.append((cur, cnt, dur / 1000))
for typ in ("AIC", "AIV"):
    v = [(n, d) for c, n, d in runs if c == typ]
    if not v: continue
    ns = sorted(n for n, _ in v); ds = [d for _, d in v]
    print("  %s: %d 段/窗口（%.1f 段/步）；块内算子数 中位 %.0f 均值 %.1f；单块时长 中位 %.1f µs 均值 %.1f µs"
          % (typ, len(v), len(v) / nst, ns[len(ns)//2], sum(ns)/len(ns),
             sorted(ds)[len(ds)//2]*1000, sum(ds)/len(ds)*1000))
