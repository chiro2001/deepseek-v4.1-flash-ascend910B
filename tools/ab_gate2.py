#!/usr/bin/env python3
"""ab_gate 的稳健版：按 n 给出样本数、p50、p10-p90 离散，并**只对低离散档位**做配对。
用法: ab_gate2.py <runA> <runB>
"""
import re, statistics, sys, os
BASE = "/home/l00886679/cedpd-repo/results"

def load(run, nmax=60):
    p = f"{BASE}/{run}/serve.log"
    if not os.path.exists(p): return None
    agg = {}
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "[bneck]" not in line: continue
            mn = re.search(r"\bn=(\d+)", line); mh = re.search(r"\bhp=([\d.]+)", line)
            if not (mn and mh): continue
            n = int(mn.group(1))
            if n > nmax: continue
            agg.setdefault(n, []).append(float(mh.group(1)))
    return agg

A, B = load(sys.argv[1]), load(sys.argv[2])
print(f"{'n':>4} {'A_n':>6} {'A_p50':>8} {'A_p10-90':>10} {'B_n':>6} {'B_p50':>8} {'B_p10-90':>10} {'Δp50':>8}  {'相对离散':>8}")
deltas = []
for n in sorted(set(A) & set(B)):
    a, b = sorted(A[n]), sorted(B[n])
    if len(a) < 8 or len(b) < 8: continue
    pa, pb = statistics.median(a), statistics.median(b)
    sa = (a[9*len(a)//10] - a[len(a)//10]) / pa
    sb = (b[9*len(b)//10] - b[len(b)//10]) / pb
    rel = max(sa, sb)
    ok = "✓" if rel < 0.05 else "✗"
    deltas.append((n, pb - pa, rel, ok))
    print(f"{n:>4} {len(a):>6} {pa:>8.2f} {sa*100:>9.1f}% {len(b):>6} {pb:>8.2f} {sb*100:>9.1f}% "
          f"{pb-pa:>+8.2f}  {rel*100:>6.1f}%{ok}")
good = [d for d in deltas if d[3] == "✓"]
print(f"\n全部档位配对差中位 = {statistics.median([d[1] for d in deltas]):+.3f} ms（{len(deltas)} 档）")
if good:
    print(f"**低离散档位（p10-p90 < 5%）配对差中位 = {statistics.median([d[1] for d in good]):+.3f} ms**"
          f"（{len(good)} 档: {[d[0] for d in good]}）")
