#!/usr/bin/env python3
"""**主路径 busy 的算子构成**（只统计主流上的任务，按算子聚合）。

这是最关键的一张表：主流 busy ≈ 步长的 62%，而它就是"串行链"本身。
把它按算子拆开，才知道该攻击谁。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_mainpath_ops.py <profdir> [topN]
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

D = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

MATH_F = ("aic_mac_time(us)", "aiv_vec_time(us)")
SCAL_F = ("aic_scalar_time(us)", "aiv_scalar_time(us)")
MOVE_F = ("aic_mte1_time(us)", "aic_mte2_time(us)", "aic_fixpipe_time(us)",
          "aiv_mte2_time(us)", "aiv_mte3_time(us)")

marks, rows = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        def g(f):
            try: return float(r.get(f) or 0)
            except (TypeError, ValueError): return 0.0
        rows.append((a, du, nm, str(r.get("Stream ID") or ""),
                     sum(g(f) for f in MATH_F), sum(g(f) for f in SCAL_F),
                     sum(g(f) for f in MOVE_F), r.get("Accelerator Core") or ""))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

busy = defaultdict(float)
for a, du, nm, s, *_ in sel: busy[s] += du
MAIN = max(busy.items(), key=lambda kv: kv[1])[0]

agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])   # op -> [dur, math, scal, move, n]
tot = [0.0, 0.0, 0.0, 0.0, 0]
for a, du, nm, s, ma, sc, mv, core in sel:
    if s != MAIN: continue
    key = (nm[:46], core[:14])
    e = agg[key]; e[0] += du; e[1] += ma; e[2] += sc; e[3] += mv; e[4] += 1
    tot[0] += du; tot[1] += ma; tot[2] += sc; tot[3] += mv; tot[4] += 1

def ms(x): return x / 1000 / nst
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜主流(自动) s{MAIN}")
print(f"主流 busy = {ms(tot[0]):.3f} ms/步（{100*ms(tot[0])/STEP:.1f}% 步长），任务 {tot[4]/nst:.0f}/步")
print(f"  └ 数学 {ms(tot[1]):.3f} ({100*tot[1]/tot[0]:.0f}%)"
      f"｜标量 {ms(tot[2]):.3f} ({100*tot[2]/tot[0]:.0f}%)"
      f"｜搬运 {ms(tot[3]):.3f} ({100*tot[3]/tot[0]:.0f}%)"
      f"｜未归因 {ms(tot[0]-tot[1]-tot[2]-tot[3]):.3f} ({100*(tot[0]-tot[1]-tot[2]-tot[3])/tot[0]:.0f}%)")
print()
print(("{:<46}{:<15}{:>8}{:>9}{:>9}{:>9}{:>9}").format("算子","核","次/步","时长ms","数学ms","标量ms","搬运ms"))
cum = 0.0
for (nm, core), (du, ma, sc, mv, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:TOPN]:
    cum += du
    print(("{:<46}{:<15}{:>8.1f}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}").format(
        nm, core, c/nst, ms(du), ms(ma), ms(sc), ms(mv)))
print("-"*106)
print(f"Top{TOPN} 累计 {ms(cum):.3f} ms = 主流 busy 的 {100*cum/tot[0]:.1f}%"
      f"，步长的 {100*ms(cum)/STEP:.1f}%")
