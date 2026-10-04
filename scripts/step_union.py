#!/usr/bin/env python3
"""单步的真实占用：全部任务的并集 / AI core 并集 / 各 stream 并集。
用法: step_union.py <profdir> <lo_ms> <hi_ms>
"""
import glob, sys
import pandas as pd
import numpy as np
M, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type","Stream ID"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
sub["s"] = sub["s"].clip(lo, hi); sub["e"] = sub["e"].clip(lo, hi)
dur = hi - lo

def union(a):
    if len(a) == 0: return 0.0
    a = a[np.argsort(a[:,0])]
    tot = 0.0; cs, ce = a[0]
    for s, e in a[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + ce - cs

print(f"步长 {dur:.2f} ms  任务数 {len(sub)}")
allu = union(sub[["s","e"]].to_numpy())
core = sub[sub["Task Type"] != "AI_CPU"]
coreu = union(core[["s","e"]].to_numpy())
acpu = sub[sub["Task Type"] == "AI_CPU"]
acpuu = union(acpu[["s","e"]].to_numpy())
print(f"  全部任务并集     = {allu:7.3f} ms ({allu/dur*100:5.1f}%)   真空闲 = {dur-allu:6.3f} ms ({(dur-allu)/dur*100:4.1f}%)")
print(f"  AI core 并集     = {coreu:7.3f} ms ({coreu/dur*100:5.1f}%)   AI core 空闲 = {dur-coreu:6.3f} ms ({(dur-coreu)/dur*100:4.1f}%)")
print(f"  AICPU 并集       = {acpuu:7.3f} ms ({acpuu/dur*100:5.1f}%)")
print(f"  busy 之和        = {(sub['e']-sub['s']).sum():7.3f} ms（>并集说明跨 stream 并行）")
print("\n各 stream 占用（降序）:")
for sd, gg in sub.groupby("Stream ID"):
    u = union(gg[["s","e"]].to_numpy())
    if u > 0.05:
        print(f"  stream {str(sd):>7}: {u:7.3f} ms ({u/dur*100:5.1f}%)  n={len(gg):5d}")
