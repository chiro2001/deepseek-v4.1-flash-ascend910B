#!/usr/bin/env python3
"""算子实际申请的 AIC/AIV 块数 ⇒ 资源占用率与并行度。
A3 每 die 24 cube / 48 vector。
"""
import sys

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur"})
df = df.sort_values("st")
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
lo, hi = marks[2], marks[-3]
w = df[(df["st"] >= lo) & (df["st"] < hi)].copy()
nst = len([m for m in marks if lo <= m < hi])
print("列:", [c for c in df.columns if "lock" in c or "Num" in c])
print("稳态 %d 步\n" % nst)

CATS = {"AI_CORE": "AIC纯净", "MIX_AIC": "AIC(混合)", "AI_VECTOR_CORE": "AIV纯净",
        "MIX_AIV": "AIV(混合)", "COMMUNICATION": "通信", "AI_CPU": "AICPU"}
w["cat"] = w["Accelerator Core"].map(CATS).fillna(w["Accelerator Core"].astype(str))

print("=== 各类别的块数占用（A3: 24 cube / 48 vector per die）===")
print("%-12s %9s %14s %14s %14s" % ("类别", "算子/步", "Block Num中位", "最大", "累计块·µs/步"))
for c, g in sorted(w.groupby("cat"), key=lambda kv: -kv[1]["dur"].sum()):
    bn = pd.to_numeric(g.get("Block Num"), errors="coerce")
    work = (g["dur"] * bn).sum() / 1000 / nst      # 块·毫秒/步
    print("%-12s %9.1f %14s %14s %14.1f" % (
        c, len(g) / nst, f"{bn.median():.0f}" if bn.notna().any() else "NA",
        f"{bn.max():.0f}" if bn.notna().any() else "NA", work))

print("\n=== 资源占用（把块数折算成核·时间）===")
# AIC = AI_CORE + MIX_AIC；AIV = AI_VECTOR_CORE + MIX_AIV
aic_ops = w[w["cat"].isin(["AIC纯净", "AIC(混合)"])]
aiv_ops = w[w["cat"].isin(["AIV纯净", "AIV(混合)"])]
for nm, sub, cap in (("AIC", aic_ops, 24), ("AIV", aiv_ops, 48)):
    bn = pd.to_numeric(sub["Block Num"], errors="coerce").fillna(1)
    core_ms = (sub["dur"] * bn).sum() / 1000 / nst
    print("  %s: 算子 %6.0f/步；核·时间合计 %8.1f ms·核/步；理论可用 %d 核 × 40.02 ms = %.0f ms·核"
          % (nm, len(sub) / nst, core_ms, cap, cap * 40.02))
    print("      ⇒ 核利用率 %.1f%%（按 step 窗口）" % (100 * core_ms / (cap * 40.02)))

print("\n=== Top 算子的块数（判断是否为小算子）===")
g = w.groupby("name").agg(n=("dur", lambda s: s.size / nst), ms=("dur", lambda s: s.sum() / 1000 / nst),
                          med=("dur", "median"),
                          block=("Block Num", lambda s: pd.to_numeric(s, errors="coerce").median()))
for name, r in g.sort_values("ms", ascending=False).head(16).iterrows():
    print("  %-50s %7.1f/步 %7.3fms 中位%6.1fµs 块数%6.0f" % (str(name)[:50], r["n"], r["ms"], r["med"], r["block"]))
