#!/usr/bin/env python3
"""服务 profile 里 AICPU 的占空比与构成。用法: aicpu_duty.py <mindstudio_profiler_output>
判据：若 AICPU 并集 / 窗口 ≈ 100%，则 metadata 变慢来自 AICPU 饱和（排队）。
"""
import glob, sys
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Task Start Time(us)","Task Duration(us)","Task Type"]
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

ai = w[w["Task Type"] == "AI_CPU"][["s","e"]]
print(f"稳态窗口 = {win:9.1f} ms")
print(f"AICPU 任务数 = {len(ai):6d}  （每步 ≈ {len(ai)/541:.2f}）")
Ua = union_ms(ai.to_numpy())
print(f"AICPU 时间并集 = {Ua:9.1f} ms  ⇒ **占空比 = {Ua/win*100:5.1f}%**")
print(f"AICPU 时长之和 = {ai['e'].sub(ai['s']).sum()/1000:9.1f} ms  "
      f"（sum/union = {ai['e'].sub(ai['s']).sum()/1000/max(Ua,1e-9):.2f}）")
# AICPU 按算子名前 12
g = ai.assign(dur=(ai["e"]-ai["s"])/1000).join(w[["OP Type"]]).groupby("OP Type")["dur"].agg(["count","sum"])
g["sum"] = g["sum"].round(1)
print("\nAICPU 任务构成（按总时长）:")
print(g.sort_values("sum", ascending=False).head(12).to_string())
