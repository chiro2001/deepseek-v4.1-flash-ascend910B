#!/usr/bin/env python3
"""按设备自身时长排 top 算子（锚点可配，纯标准库）。K 已隐含（步长直接取 profile）。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_top_ops_anchored.py <profdir> [topN]
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

D = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        rows.append((st, du, nm, r.get("Accelerator Core") or "", r.get("Input Shapes") or "",
                     r.get("Block Num") or ""))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

agg = defaultdict(lambda: [0, 0.0, set(), ""])
for st, du, nm, core, shp, blk in sel:
    key = (nm[:56], core[:14])
    a = agg[key]; a[0] += 1; a[1] += du; a[2].add(shp[:22]); a[3] = blk

print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长(profile) {STEP:.3f} ms｜算子 {len(sel)/nst:.0f}/步")
print(f"{'#':>3} {'算子':<56}{'核':<15}{'次/步':>8}{'单次us':>9}{'ms/步':>9}{'占步长':>8}{'块':>5}")
ranked = sorted(agg.items(), key=lambda kv: -kv[1][1])[:TOPN]
tot = 0.0
for i, ((nm, core), (c, d, shapes, blk)) in enumerate(ranked, 1):
    per = c / nst; ms = d / 1000 / nst
    tot += ms
    print(f"{i:>3} {nm:<56}{core:<15}{per:>8.1f}{d/c:>9.2f}{ms:>9.3f}{100*ms/STEP:>7.1f}%{str(blk)[:5]:>5}")
print(f"\nTop{TOPN} 合计 {tot:.3f} ms = {100*tot/STEP:.1f}% 步长")
