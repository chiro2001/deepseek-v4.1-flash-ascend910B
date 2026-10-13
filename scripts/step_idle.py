#!/usr/bin/env python3
"""单步内的真空闲区间（所有任务都没跑）——定位同步停顿。
用法: step_idle.py <profdir> <lo_ms> <hi_ms> [min_us]
"""
import glob, sys
import pandas as pd
import numpy as np
M, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
MIN_US = float(sys.argv[4]) if len(sys.argv) > 4 else 20
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
sub["s"] = sub["s"].clip(lo, hi); sub["e"] = sub["e"].clip(lo, hi)
iv = sub[["s","e"]].to_numpy(); iv = iv[np.argsort(iv[:,0])]
mv = []
for s, e in iv:
    if mv and s <= mv[-1][1]: mv[-1][1] = max(mv[-1][1], e)
    else: mv.append([s, e])
mv = np.array(mv)
idles = []
prev = lo
for s, e in mv:
    if s - prev > MIN_US/1000.0: idles.append((prev, s))
    prev = max(prev, e)
if hi - prev > MIN_US/1000.0: idles.append((prev, hi))
print(f"步 {lo:.2f}–{hi:.2f}（{hi-lo:.2f} ms）  忙 {len(mv)} 段  真空闲 {len(idles)} 段")
if not idles:
    print("  无 >= %g us 的空闲" % MIN_US); sys.exit()
tot = sum(b-a for a,b in idles)
print(f"  空闲合计 {tot:.3f} ms ({tot/(hi-lo)*100:.1f}%)")
print(f"\n{'t-rel(ms)':>10} {'len(us)':>9}   前一个任务 -> 后一个任务")
for a, b in idles:
    before = sub[sub["e"] <= a + 1e-6]
    after = sub[sub["s"] >= b - 1e-6]
    pb = before.iloc[-1] if len(before) else None
    pa = after.iloc[0] if len(after) else None
    s1 = f"{str(pb['OP Type'])[:24]}(s{pb['Stream ID']:.0f})" if pb is not None else "?"
    s2 = f"{str(pa['OP Type'])[:24]}(s{pa['Stream ID']:.0f})" if pa is not None else "?"
    print(f"{a-lo:10.3f} {(b-a)*1000:9.1f}   {s1} -> {s2}")
