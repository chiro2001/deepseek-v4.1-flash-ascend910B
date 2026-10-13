#!/usr/bin/env python3
"""修正：区分 AI core / AICPU 的占空比。用法: duty_fix.py <mindstudio_profiler_output>"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Task Type"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = df["Task Start Time(us)"] - t0
df["e"] = df["s"] + df["Task Duration(us)"]
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut]
win = (w["s"].max() - w["s"].min()) / 1000.0

def union_ms(a):
    if len(a) == 0: return 0.0
    a = a[np.argsort(a[:,0])]
    tot = 0.0; cs, ce = a[0]
    for s, e in a[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return (tot + ce - cs) / 1000.0

core = w[w["Task Type"] != "AI_CPU"][["s","e"]].to_numpy()
acpu = w[w["Task Type"] == "AI_CPU"][["s","e"]].to_numpy()
allv = w[["s","e"]].to_numpy()
Uc, Ua, Uall = union_ms(core), union_ms(acpu), union_ms(allv)
print(f"窗口                = {win:9.1f} ms")
print(f"**AI core 并集**    = {Uc:9.1f} ms  ⇒ 占空比 {Uc/win*100:5.1f}%")
print(f"AICPU 并集          = {Ua:9.1f} ms  ⇒ 占空比 {Ua/win*100:5.1f}%")
print(f"并集(全部, 含重叠)  = {Uall:9.1f} ms  ⇒ {Uall/win*100:5.1f}%")
print(f"⇒ **AI core 空闲**  = {(win-Uc):9.1f} ms = {(win-Uc)/win*100:5.1f}%")
print(f"   其中被 AICPU 占用 = {Ua:9.1f} ms = {Ua/win*100:5.1f}%")
print(f"   纯空闲（谁都没在跑） = {(win-Uall):9.1f} ms = {(win-Uall)/win*100:5.1f}%")
