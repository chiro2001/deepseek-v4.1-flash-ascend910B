#!/usr/bin/env python3
"""AIC/AIV 资源占用 + 重叠矩阵 + 流水可行性（正确分类版）。"""
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
lo, hi = marks[2], marks[-3]
w = df[(df["st"] >= lo) & (df["st"] < hi)].copy()
nst = len([m for m in marks if lo <= m < hi])
step_ms = (hi - lo) / 1000 / nst

CATS = {
    "AI_CORE": "AIC纯净", "MIX_AIC": "AIC(混合)",
    "AI_VECTOR_CORE": "AIV纯净", "MIX_AIV": "AIV(混合)",
    "COMMUNICATION": "通信", "AI_CPU": "AICPU",
}
w["cat"] = w["core"].map(CATS).fillna(w["core"].astype(str))


def union_ms(sub):
    iv = sorted(zip(sub["st"], sub["en"]))
    if not iv: return 0.0
    tot = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return (tot + ce - cs) / 1000


def overlap_ms(a, b):
    ea = sorted(zip(a["st"], a["en"])); eb = sorted(zip(b["st"], b["en"]))
    i = j = 0; tot = 0.0
    while i < len(ea) and j < len(eb):
        s = max(ea[i][0], eb[j][0]); e = min(ea[i][1], eb[j][1])
        if e > s: tot += e - s
        if ea[i][1] < eb[j][1]: i += 1
        else: j += 1
    return tot / 1000


print("稳态 %d 步，步长(profile) %.2f ms，算子 %.0f/步\n" % (nst, step_ms, len(w) / nst))
print("%-14s %10s %12s %12s %8s %10s" % ("类别", "算子/步", "时长合计/步", "并集busy/步", "中位µs", "占步长%"))
cats = {}
for c, g in sorted(w.groupby("cat"), key=lambda kv: -kv[1]["dur"].sum()):
    u = union_ms(g) / nst
    cats[c] = (g, u)
    print("%-14s %10.1f %12.3f %12.3f %8.1f %9.1f%%" % (
        c, len(g) / nst, g["dur"].sum() / 1000 / nst, u, g["dur"].median(), 100 * u / step_ms))

def busy_of(*names):
    sub = w[w["cat"].isin(names)]
    return union_ms(sub) / nst if len(sub) else 0.0

aic = busy_of("AIC纯净", "AIC(混合)")
aiv = busy_of("AIV纯净", "AIV(混合)")
comm = busy_of("通信")
cpu = busy_of("AICPU")
print("\n★ 资源并集：AIC %.2f ms/步 (%.0f%%) | AIV %.2f ms/步 (%.0f%%) | 通信 %.2f (%.0f%%) | AICPU %.2f (%.0f%%)"
      % (aic, 100*aic/step_ms, aiv, 100*aiv/step_ms, comm, 100*comm/step_ms, cpu, 100*cpu/step_ms))

# 两两重叠
pairs = [("AIC纯", ["AIC纯净"]), ("AIC混合", ["AIC(混合)"]), ("AIV纯", ["AIV纯净"]), ("AIV混合", ["AIV(混合)"]),
         ("通信", ["通信"]), ("AICPU", ["AICPU"])]
print("\n=== 两两重叠（ms/步，对角=自身 busy）===")
print("%-10s" % "" + "".join("%11s" % p[0] for p in pairs))
for an, ac in pairs:
    row = []
    for bn, bc in pairs:
        a = w[w["cat"].isin(ac)]; b = w[w["cat"].isin(bc)]
        row.append(overlap_ms(a, b) / nst)
    print("%-10s" % an + "".join("%11.3f" % v for v in row))

print("\n=== 流水上界估算 ===")
print("  当前步长            %7.2f ms" % step_ms)
print("  AIC 并集            %7.2f ms" % aic)
print("  AIV 并集            %7.2f ms" % aiv)
print("  AIC+AIV 完全流水后  %7.2f ms  （下界 = max(AIC, AIV)）" % max(aic, aiv))
print("  再加通信也重叠      %7.2f ms" % max(aic, aiv, comm))
print("  ⇒ 理论加速上限 %.2f×（AIC/AIV 完美流水）" % (step_ms / max(aic, aiv)))

# AIV 大户
print("\n=== AIV（纯净+混合）耗时 Top12 ===")
av = w[w["cat"].isin(["AIV纯净", "AIV(混合)"])]
g = av.groupby("name").agg(n=("dur", lambda s: s.size / nst), ms=("dur", lambda s: s.sum() / 1000 / nst),
                           med=("dur", "median"))
for name, r in g.sort_values("ms", ascending=False).head(12).iterrows():
    print("  %-52s %8.1f 个/步 %8.3f ms/步 中位 %6.1f µs" % (str(name)[:52], r["n"], r["ms"], r["med"]))

print("\n=== AIC（纯净+混合）耗时 Top12 ===")
ac = w[w["cat"].isin(["AIC纯净", "AIC(混合)"])]
g = ac.groupby("name").agg(n=("dur", lambda s: s.size / nst), ms=("dur", lambda s: s.sum() / 1000 / nst),
                           med=("dur", "median"))
for name, r in g.sort_values("ms", ascending=False).head(12).iterrows():
    print("  %-52s %8.1f 个/步 %8.3f ms/步 中位 %6.1f µs" % (str(name)[:52], r["n"], r["ms"], r["med"]))

print("\n=== 通信（HCCL）Top8 ===")
cm = w[w["cat"] == "通信"]
g = cm.groupby("name").agg(n=("dur", lambda s: s.size / nst), ms=("dur", lambda s: s.sum() / 1000 / nst),
                           med=("dur", "median"))
for name, r in g.sort_values("ms", ascending=False).head(8).iterrows():
    print("  %-52s %8.1f 个/步 %8.3f ms/步 中位 %6.1f µs" % (str(name)[:52], r["n"], r["ms"], r["med"]))
