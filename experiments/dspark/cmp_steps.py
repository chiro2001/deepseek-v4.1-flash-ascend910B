#!/usr/bin/env python3
"""按**步索引**窗口比较两个 profiler 的 step 周期（避免时间窗对齐问题）。"""
import csv, statistics, sys

def series(path, per_step):
    ts = []
    for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
        if (r.get("Name") or "").split("_")[0] != "HcPre":
            continue
        v = (r.get("Start Time(us)") or "").strip()
        if v:
            try: ts.append(float(v))
            except ValueError: pass
    ts.sort()
    b = ts[::per_step]
    return [(b[i]-b[i-1])/1000.0 for i in range(1, len(b))]

def stat(d, a, b, label):
    p = d[a:b]
    if not p: return
    s = sorted(p)
    print(f"   {label:<26} n={len(p):>4}  median={statistics.median(s):>6.2f}  "
          f"p10={s[len(s)//10]:>6.2f}  p90={s[-max(1,len(s)//10)]:>6.2f}")

pa, ppa, na = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pb, ppb, nb = sys.argv[4], int(sys.argv[5]), sys.argv[6]
A, B = series(pa, ppa), series(pb, ppb)
print(f"== {na}  vs  {nb}")
for lo, hi in ((0, 28), (28, 56), (56, 84), (84, 9999)):
    print(f"  -- 步 {lo}..{min(hi,len(A)) if hi<9999 else len(A)} --")
    stat(A, lo, hi, na)
    stat(B, lo, hi, nb)
    if hi >= 9999: break
