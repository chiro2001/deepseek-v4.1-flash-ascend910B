#!/usr/bin/env python3
"""纯标准库的 decode 资源账 + 重叠矩阵（锚点可配）。"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "HcPre"); PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))
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
        rows.append((st, st + du, du, r.get("Accelerator Core") or "", nm))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长(profile) {STEP:.3f} ms｜算子 {len(sel)/nst:.0f}/步\n")

def merged(iv):
    if not iv: return []
    iv = sorted(iv); out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce)); return out
def tot(iv): return sum(e - s for s, e in iv)
def ov(a, b):
    t = 0.0; i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: t += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return t

CAT = {
    "AIC":  ("AI_CORE", "MIX_AIC"),
    "AIV":  ("AI_VECTOR_CORE", "MIX_AIV"),
    "COMM": ("COMMUNICATION",),
    "CPU":  ("AI_CPU",),
}
groups = defaultdict(list)
for st, en, du, core, nm in sel:
    hit = None
    for k, keys in CAT.items():
        if any(x == core or x in core for x in keys): hit = k; break
    groups[hit or "OTHER"].append((st, en, du))

print(f"{'资源':<8}{'busy ms/步':>12}{'占步长':>9}")
bys = {}
for k in list(CAT) + ["OTHER"]:
    g = groups.get(k, [])
    if not g: continue
    iv = merged([(s, e) for s, e, d in g]); u = tot(iv) / 1000 / nst
    bys[k] = u
    print(f"{k:<8}{u:>12.3f}{100*u/STEP:>8.1f}%")
alliv = merged([(s, e) for st, en, du, c, n in sel for s, e in [(st, en)]])
print(f"{'全并集':<8}{tot(alliv)/1000/nst:>12.3f}{100*tot(alliv)/1000/nst/STEP:>8.1f}%")
print(f"{'纯空闲':<8}{STEP - tot(alliv)/1000/nst:>12.3f}{100*(STEP-tot(alliv)/1000/nst)/STEP:>8.1f}%")

print("\n重叠矩阵（ms/步）：")
ks = [k for k in ["AIC", "AIV", "COMM", "CPU"] if k in bys]
print("        " + "".join(f"{k:>9}" for k in ks))
for a in ks:
    ia = merged([(s, e) for s, e, d in groups[a]])
    line = f"{a:<8}"
    for b in ks:
        ib = merged([(s, e) for s, e, d in groups[b]])
        line += f"{ov(ia, ib)/1000/nst:>9.3f}"
    print(line)
if "AIC" in bys and "AIV" in bys:
    print(f"\nAIC busy {bys['AIC']:.3f} | AIV busy {bys['AIV']:.3f} | max={max(bys['AIC'],bys['AIV']):.3f}")
    print(f"若 AIC/AIV 完美重叠：步长 {STEP:.3f} → {max(bys['AIC'],bys['AIV']):.3f} = {STEP/max(bys['AIC'],bys['AIV']):.2f}×")
    nA = merged([(s,e) for k in ("AIC","AIV","COMM","CPU") if k in bys for s,e,d in groups[k]])
    print(f"若全资源完美重叠：{tot(nA)/1000/nst:.3f} = {STEP/(tot(nA)/1000/nst):.2f}×")
