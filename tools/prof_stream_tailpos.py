#!/usr/bin/env python3
"""每条流的"最后任务落在步内百分之几" —— 判断尾链是否在关键路径上。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_stream_tailpos.py <profdir>
"""
from __future__ import annotations
import csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]
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
            rows.append((float(r["Start Time(us)"]), float(r["Duration(us)"]), nm,
                         str(r.get("Stream ID") or "")))
        except Exception: continue
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
starts = [m for m in marks if LO <= m < HI]
nst = len(starts)

byst = defaultdict(list)
for s, d, nm, sid in rows:
    if LO <= s < HI:
        byst[sid].append((s, s + d))

print(f"步数 {nst}｜步长 {(HI-LO)/1000/nst:.3f} ms")
print(("{:>5}{:>10}{:>18}{:>16}{:>14}").format("流", "任务/步", "最后结束(步内%)", "busy(ms/步)", "任务时长(ms)"))
res = []
for sid, v in byst.items():
    rel, busy, dur = [], 0.0, 0.0
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        seg = [(s, e) for s, e in v if a <= s < b]
        if not seg: continue
        rel.append((max(e for s, e in seg) - a) / (b - a))
        busy += sum(e - s for s, e in seg) / 1000
        dur += sum(e - s for s, e in seg) / 1000
    if len(rel) < 50: continue
    res.append((sid, len(v) / nst, st.median(rel), busy / nst, dur / nst))
for sid, n, med, busy, dur in sorted(res, key=lambda x: -x[3])[:9]:
    print(("{:>5}{:>10.0f}{:>17.1f}%{:>16.3f}{:>14.3f}").format(sid, n, med * 100, busy, dur))

main = [x for x in res if x[0] == "109"]
if main:
    sid, n, med, busy, dur = main[0]
    print(f"\n★ 主流 s109：最后任务中位落在步内 {med*100:.1f}% ⇒ 步尾还有 {(1-med)*100:.1f}% 是它在空等")
    tail = [x for x in res if x[0] != "109" and x[2] > med]
    if tail:
        print("   在主流结束之后仍在跑的流：")
        for sid, n, m2, b2, d2 in sorted(tail, key=lambda x: -x[4])[:5]:
            print(f"     s{sid}: 最后 {m2*100:.1f}%，任务时长 {d2:.3f} ms/步")
