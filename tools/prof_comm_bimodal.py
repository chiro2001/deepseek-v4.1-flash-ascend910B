#!/usr/bin/env python3
"""看通信算子（AivKernel / hcom_*）的时长是否双峰，并按步内位置分组。

背景：旧 profile 里 allreduce 有两个族——MoE 侧 73 µs 与 attention 侧 32 µs，
payload 相同却差 2.3×，至今未解释。本工具用新 profile 复核并定位。
"""
from __future__ import annotations
import bisect, csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]
KEYS = tuple((sys.argv[2] if len(sys.argv) > 2 else "AivKernel").split(","))
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
            rows.append((float(r["Start Time(us)"]), float(r["Duration(us)"]),
                         str(r.get("Stream ID") or ""), nm[:24]))
        except Exception: continue
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
starts = [m for m in marks if LO <= m < HI]
nst = len(starts) - 1
sel = [x for x in rows if LO <= x[0] < HI]
if not sel: raise SystemExit("无命中")
durs = sorted(d for _, d, _, _ in sel)
print(f"命中 {len(sel)/nst:.1f} 个/步 | 时长分位："
      f"p10={durs[len(durs)//10]:.1f} p25={durs[len(durs)//4]:.1f} "
      f"p50={st.median(durs):.1f} p75={durs[3*len(durs)//4]:.1f} p90={durs[9*len(durs)//10]:.1f} max={durs[-1]:.1f}")

# 直方图
bins = [(0,10),(10,20),(20,30),(30,45),(45,60),(60,80),(80,120),(120,1e9)]
hist = defaultdict(int)
for d in durs:
    for lo,hi in bins:
        if lo <= d < hi: hist[(lo,hi)] += 1; break
print("\n时长直方图：")
for lo,hi in bins:
    c = hist[(lo,hi)]
    if c: print(f"  [{lo:>4},{hi if hi<1e8 else '∞':>4}) us : {c:>6} ({100*c/len(durs):5.1f}%) {'#'*int(60*c/len(durs))}")

# 按步内位置分组
print("\n按步内位置分三段（前/中/后 1/3）：")
seg = defaultdict(list)
for a, d, sid, nm in sel:
    i = bisect.bisect_right(starts, a) - 1
    if i < 0 or i >= nst: continue
    rel = (a - starts[i]) / (starts[i+1] - starts[i])
    seg["前1/3" if rel < 1/3 else ("中1/3" if rel < 2/3 else "后1/3")].append(d)
for k in ("前1/3","中1/3","后1/3"):
    v = seg.get(k, [])
    if v: print(f"  {k}: n={len(v):>5} 中位={st.median(v):6.2f} us  均值={sum(v)/len(v):6.2f}")

# 按 stream
print("\n按 stream：")
bys = defaultdict(list)
for a, d, sid, nm in sel: bys[sid].append(d)
for sid, v in sorted(bys.items(), key=lambda kv: -len(kv[1]))[:6]:
    print(f"  s{sid:<5} n={len(v):>5} 中位={st.median(v):6.2f} us 总={sum(v)/1000/nst:6.3f} ms/步")
