#!/usr/bin/env python3
"""某个算子族的明细（按 形状×dtype×块数×核 聚合），用于判断"能否优化"。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_family_ops_detail.py <profdir> <关键词,逗号分隔> [topN]
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

D = sys.argv[1]
KEYS = tuple(sys.argv[2].split(","))
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        if not any(k in nm for k in KEYS): continue
        try:
            rows.append((float(r["Start Time(us)"]), float(r["Duration(us)"]), nm,
                         r.get("Input Shapes") or "", r.get("Input Data Types") or "",
                         str(r.get("Block Num") or ""), r.get("Accelerator Core") or ""))
        except Exception: continue
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

agg = defaultdict(lambda: [0, 0.0])
for st, du, nm, sh, dt, blk, core in sel:
    agg[(nm[:44], sh[:20], dt[:16], blk, core[:12])][0] += 1
    agg[(nm[:44], sh[:20], dt[:16], blk, core[:12])][1] += du

print(f"步数 {nst} | 步长 {STEP:.3f} ms | 命中 {len(sel)/nst:.1f} 个/步")
hdr = ("{:<44}{:<22}{:<18}{:>5}{:<13}{:>7}{:>8}{:>8}").format(
    "算子", "形状", "类型", "块", "核", "次/步", "单次us", "ms/步")
print(hdr)
tot = 0.0
for k, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:TOPN]:
    nm, sh, dt, blk, core = k
    ms = d / 1000 / nst
    tot += ms
    print(("{:<44}{:<22}{:<18}{:>5}{:<13}{:>7.1f}{:>8.2f}{:>8.3f}").format(
        nm, sh, dt, blk, core, c / nst, d / c, ms))
print(f"\nTop{TOPN} 合计 {tot:.3f} ms = {100*tot/STEP:.1f}% 步长")
