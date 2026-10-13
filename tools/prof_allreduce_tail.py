#!/usr/bin/env python3
"""allreduce（AivKernel/hcom_*）的**长尾归因**：慢的那些是不是在跟 AIC/AIV 抢核。

背景（新 profile）：94 次/步的 allreduce，中位 17 µs，但 p90 = 65 µs；
直方图显示 21% 超过 30 µs。若把长尾压到中位水平，可省 ~0.8 ms（3% 步长）。
旧文档的假说：「AIV 核在 MIX 算子占用时，allreduce 自旋等待被计入时长」——**从未验证**。

本工具做判别：对每个 allreduce，统计其时间窗内**并发**的 AIC/AIV/mix 算子数，
看慢的那些是否并发更多。
"""
from __future__ import annotations
import bisect, csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))
COMM_KEY = os.environ.get("COMM_KEY", "AivKernel")

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
        tasks.append((a, a + du, nm, str(r.get("Stream ID") or ""),
                      r.get("Accelerator Core") or ""))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in tasks if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

comm = [t for t in sel if COMM_KEY in t[2]]
others = [t for t in sel if COMM_KEY not in t[2]]
others_ts = [t[0] for t in others]
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms")
print(f"通信算子（{COMM_KEY}）：{len(comm)/nst:.1f} 个/步，合计 {sum(t[1]-t[0] for t in comm)/1000/nst:.3f} ms/步")
if not comm: raise SystemExit("无命中")

# 对每个通信算子，统计时间窗内并发算子（按核类型）
FAST, SLOW = 25.0, 45.0
rows = []
for a, e, nm, s, core in comm:
    lo = bisect.bisect_left(others_ts, a - 0.0)
    # 向前回退，覆盖开始早于 a 但未结束的
    j = lo
    while j > 0 and others[j-1][1] > a: j -= 1
    conc = defaultdict(int)
    conc_dur = defaultdict(float)
    k = j
    while k < len(others) and others[k][0] < e:
        a2, e2, nm2, s2, c2 = others[k]
        if e2 > a and a2 < e:
            conc[c2] += 1
            conc_dur[c2] += min(e2, e) - max(a2, a)
        k += 1
    rows.append((e - a, s, conc, conc_dur, nm))

def summarize(label, subset):
    if not subset: return
    d = sorted(x[0] for x in subset)
    aic = st.median([x[2].get("MIX_AIC", 0) + x[2].get("AI_CORE", 0) for x in subset])
    aiv = st.median([x[2].get("AI_VECTOR_CORE", 0) + x[2].get("MIX_AIV", 0) for x in subset])
    aicd = st.median([x[3].get("MIX_AIC", 0) + x[3].get("AI_CORE", 0) for x in subset])
    aivd = st.median([x[3].get("AI_VECTOR_CORE", 0) + x[3].get("MIX_AIV", 0) for x in subset])
    print(f"  {label:<14} n={len(subset):>5}  中位时长={st.median(d):6.1f}us"
          f"  并发AIC中位={aic:4.0f}（{aicd:6.1f}us）  并发AIV中位={aiv:4.0f}（{aivd:6.1f}us）")

print(f"\n全部：")
summarize("all", rows)
summarize(f"<{FAST:.0f}us（快）", [x for x in rows if x[0] < FAST])
summarize(f"{FAST:.0f}-{SLOW:.0f}us", [x for x in rows if FAST <= x[0] < SLOW])
summarize(f">{SLOW:.0f}us（慢）", [x for x in rows if x[0] >= SLOW])

print(f"\n按核类型统计慢速占比：")
bycore = defaultdict(lambda: [0, 0])
for d, s, conc, cd, nm in rows:
    for c, n in conc.items():
        bycore[c][0] += 1
        if d >= SLOW: bycore[c][1] += 1
for c, (tot, slow) in sorted(bycore.items(), key=lambda kv: -kv[1][0])[:6]:
    print(f"  {c:<18} 出现在 {tot:>6} 个通信窗口，其中慢速 {slow:>5}（{100*slow/tot:4.1f}%）")
print(f"\n慢速通信（≥{SLOW:.0f}us）的并发构成样例（前 6 个）：")
for d, s, conc, cd, nm in sorted([x for x in rows if x[0] >= SLOW], key=lambda x: -x[0])[:6]:
    top = sorted(cd.items(), key=lambda kv: -kv[1])[:4]
    print(f"  时长{d:7.1f}us  并发: " + "  ".join(f"{c}={n}({v:.0f}us)" for c, (n, v) in
          [(c, (conc[c], cd[c])) for c, _ in top]) if False else
          f"  时长{d:7.1f}us  并发: " + "  ".join(f"{c}:{v:.0f}us" for c, v in top))
