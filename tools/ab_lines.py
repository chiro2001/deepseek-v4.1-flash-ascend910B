#!/usr/bin/env python3
"""按行号区间取 [bneck] 样本做配对比较（排除污染分段）。
用法: ab_lines.py <runA> [startA] -- <runB> [startB] [--nmax 60]
"""
import re, statistics, sys, os
BASE = "/home/l00886679/cedpd-repo/results"

def load(run, start, nmax):
    p = f"{BASE}/{run}/serve.log"
    agg = {}
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i < start or "[bneck]" not in line: continue
            mn = re.search(r"\bn=(\d+)", line); mh = re.search(r"\bhp=([\d.]+)", line)
            if not (mn and mh): continue
            n = int(mn.group(1))
            if n > nmax: continue
            agg.setdefault(n, []).append(float(mh.group(1)))
    return agg

a = sys.argv[1:]
i = a.index("--")
runA, startA = a[0], (int(a[1]) if i > 1 else 0)
rest = a[i+1:]
nmax = 60
if "--nmax" in rest:
    j = rest.index("--nmax"); nmax = int(rest[j+1]); rest = rest[:j] + rest[j+2:]
runB, startB = rest[0], (int(rest[1]) if len(rest) > 1 else 0)
A, B = load(runA, startA, nmax), load(runB, startB, nmax)
print(f"A={runA} (from line {startA})   B={runB} (from line {startB})")
print(f"{'n':>4} {'A_n':>6} {'A_p50':>8} {'A_离散':>7} {'B_n':>6} {'B_p50':>8} {'B_离散':>7} {'Δp50':>8}")
ds = []
for n in sorted(set(A) & set(B)):
    a1, b1 = sorted(A[n]), sorted(B[n])
    if len(a1) < 8 or len(b1) < 8: continue
    pa, pb = statistics.median(a1), statistics.median(b1)
    sa = (a1[9*len(a1)//10] - a1[len(a1)//10]) / pa * 100
    sb = (b1[9*len(b1)//10] - b1[len(b1)//10]) / pb * 100
    ds.append((n, pb-pa, max(sa,sb)))
    print(f"{n:>4} {len(a1):>6} {pa:>8.2f} {sa:>6.1f}% {len(b1):>6} {pb:>8.2f} {sb:>6.1f}% {pb-pa:>+8.2f}")
if ds:
    print(f"\n配对差中位 = {statistics.median([d[1] for d in ds]):+.3f} ms（{len(ds)} 档）")
    low = [d for d in ds if d[2] < 12]
    if low:
        print(f"低离散档(<12%)中位 = {statistics.median([d[1] for d in low]):+.3f} ms"
              f"（n={[d[0] for d in low]}）")
