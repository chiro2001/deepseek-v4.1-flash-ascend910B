#!/usr/bin/env python3
"""主流跨度里的 gap 在等谁（按流合并区间 + 双指针，O(流数×(G+I))）。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_main_gap_who.py <profdir> [gap_us] [topN]
"""
from __future__ import annotations
import bisect, csv, os, sys
from collections import defaultdict

D = sys.argv[1]
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 14
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))


def merge(iv):
    if not iv: return []
    iv.sort(); out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out


def overlap_len(A, B):
    """两个已排序不重叠区间表的交叠总长（双指针）。"""
    i = j = 0; tot = 0.0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0]); e = min(A[i][1], B[j][1])
        if e > s: tot += e - s
        if A[i][1] < B[j][1]: i += 1
        else: j += 1
    return tot


marks, tasks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        tasks.append((a, a + du, str(r.get("Stream ID") or ""), nm[:40]))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
starts = [m for m in marks if LO <= m < HI]
nst = len(starts) - 1
STEP = (HI - LO) / 1000 / (nst + 1)

busy = defaultdict(float)
for a, e, s, nm in tasks: busy[s] += e - a
MAIN = max(busy.items(), key=lambda kv: kv[1])[0]
mt = sorted([t for t in tasks if t[2] == MAIN], key=lambda t: t[0])
mt_ts = [t[0] for t in mt]

# 主流空隙
gaps = []
for i in range(nst):
    a, b = starts[i], starts[i + 1]
    lo = bisect.bisect_left(mt_ts, a); hi = bisect.bisect_left(mt_ts, b)
    seg = sorted(mt[lo:hi], key=lambda t: t[0])
    prev = a
    for t in seg:
        if t[0] - prev >= GAP: gaps.append((prev, t[0]))
        prev = max(prev, t[1])
    if b - prev >= GAP: gaps.append((prev, b))
gaps = merge(gaps)
G = sum(e - s for s, e in gaps)

# 按流合并（同时也按 流×算子 合并）
by_stream, by_sop = defaultdict(list), defaultdict(list)
for a, e, s, nm in tasks:
    if s == MAIN: continue
    by_stream[s].append((a, e)); by_sop[(s, nm)].append((a, e))
by_stream = {k: merge(v) for k, v in by_stream.items()}
all_non = merge([iv for v in by_stream.values() for iv in v])
cov = {s: overlap_len(iv, gaps) for s, iv in by_stream.items()}
cov_all = overlap_len(all_non, gaps)
# 只对"覆盖量大的流"做算子级拆解（否则 键数×gap数 会爆）
top_streams = [s for s, v in sorted(cov.items(), key=lambda kv: -kv[1])[:6] if v > 0]
by_sop = {k: merge(v) for k, v in by_sop.items() if k[0] in top_streams}
sop = sorted(((k, overlap_len(v, gaps)) for k, v in by_sop.items()),
             key=lambda kv: -kv[1])[:TOPN]

print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜主流(自动) s{MAIN}")
print(f"主流 busy = {busy[MAIN]/1000/nst:.3f} ms/步 ⇒ 跨度内等待 = {STEP - busy[MAIN]/1000/nst:.3f} ms")
print(f"\n主流空隙（≥{GAP:.0f}µs）：{len(gaps)/nst:.1f} 个/步，合计 {G/1000/nst:.3f} ms/步")
print(f"  └ 被非主流覆盖 {cov_all/1000/nst:.3f} ms（{100*cov_all/max(1e-9,G):.0f}%）")
print(f"  └ 真正空白     {(G-cov_all)/1000/nst:.3f} ms")
print(f"\n覆盖者（按流）：")
for s, v in sorted(cov.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  s{s:<6} {v/1000/nst:7.3f} ms/步")
print(f"\n覆盖者（流×算子，Top{TOPN}）：")
for (s, nm), v in sop:
    if v <= 0: continue
    print(f"  s{s:<6} {v/1000/nst:7.3f} ms/步  {nm}")
