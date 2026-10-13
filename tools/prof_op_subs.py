#!/usr/bin/env python3
"""拆某个算子的子计数（aic_mac / aic_scalar / aic_mte1 / aic_mte2 / aiv_*），看时间花在哪。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_op_subs.py <profdir> <算子名子串>
"""
from __future__ import annotations
import csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]; KEY = sys.argv[2]
ANCHOR = os.environ.get("ANCHOR", "HcPre"); PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))
FIELDS = ["aicore_time(us)", "aic_mac_time(us)", "aic_scalar_time(us)", "aic_mte1_time(us)",
          "aic_mte2_time(us)", "aic_fixpipe_time(us)", "aiv_time(us)", "aiv_vec_time(us)",
          "aiv_scalar_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)"]

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    have = [f for f in FIELDS if f in rd.fieldnames]
    for r in rd:
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        if KEY not in nm: continue
        try:
            du = float(r["Duration(us)"]); s = float(r["Start Time(us)"])
        except Exception: continue
        def _f(x):
            try: return float(x)
            except (TypeError, ValueError): return 0.0
        rows.append((s, du, {f: _f(r.get(f)) for f in have}, r.get("Block Num") or "",
                     r.get("Input Shapes") or ""))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst
if not sel:
    raise SystemExit("无命中")
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜命中 {len(sel)/nst:.1f} 个/步")
print(f"单次 Duration 中位 = {st.median([d for _, d, _, _, _ in sel]):.2f} µs")
print(f"\n{'子计数':<24}{'单次中位us':>12}{'占Duration':>12}{'每步ms':>10}")
dmed = st.median([d for _, d, _, _, _ in sel])
for f in FIELDS:
    vals = [m[f] for _, _, m, _, _ in sel if f in m and m[f] > 0]
    if not vals: continue
    med = st.median(vals)
    print(f"{f:<24}{med:>12.2f}{100*med/dmed:>11.1f}%{med*len(sel)/nst/1000:>10.3f}")
print(f"\n块数分布：")
c = defaultdict(int)
for _, _, _, b, _ in sel: c[str(b)] += 1
for b, n in sorted(c.items(), key=lambda kv: -kv[1])[:6]:
    print(f"   block={b:<6} {n/len(sel)*100:5.1f}%")
print(f"\n形状分布（top3）：")
c2 = defaultdict(int)
for _, _, _, _, sh in sel: c2[sh[:44]] += 1
for sh, n in sorted(c2.items(), key=lambda kv: -kv[1])[:3]:
    print(f"   {n/len(sel)*100:5.1f}%  {sh}")
