#!/usr/bin/env python3
"""按 GroupedMatmulSwigluQuantV2 (40次/步) 切 decode 步, 输出每步 device 时间分解.
用法: python3 analyze_steps.py <mindstudio_profiler_output 目录> [tail_frac]
"""
import glob, sys
import pandas as pd

M = sys.argv[1]
TAIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Stream ID", "Op Name", "OP Type", "Task Start Time(us)", "Task Duration(us)", "Task Wait Time(us)"]
dfs = [pd.read_csv(f, usecols=cols, low_memory=False) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"rows={len(df)}")

df = df.sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["t_rel"] = df["Task Start Time(us)"] - t0

gmm1 = df[df["OP Type"] == "GroupedMatmulSwigluQuantV2"]
print(f"gmm1 rows={len(gmm1)}")
blocks = gmm1["t_rel"].to_numpy()
# 用第 0,40,80.. 次 gmm1 的起点作步边界
step_starts = blocks[::40]
print(f"steps={len(step_starts)} span={step_starts[-1]/1e6:.2f}s step={ (step_starts[-1]-step_starts[0])/1e6/max(1,len(step_starts)-1)*1000:.2f}ms")

# 只取尾部 TAIL 比例的步做稳态
n0 = int(len(step_starts) * (1 - TAIL))
bounds = step_starts[n0:]
b0, b1 = bounds[0], step_starts[-1] if len(step_starts) * 0 + 1 else bounds[-1]
# 稳态窗口 = bounds[0] .. 最后一步的结束(用最后一条记录的 t_rel)
win = df[(df["t_rel"] >= b0)]
nsteps = len(bounds)

print(f"steady window: {nsteps} steps from {b0/1e6:.2f}s")
bytype = win.groupby("OP Type")["Task Duration(us)"].agg(["count", "sum"])
bytype["per_step_us"] = bytype["sum"] / nsteps
bytype["cnt_per_step"] = bytype["count"] / nsteps
bytype = bytype.sort_values("per_step_us", ascending=False)
tot = bytype["per_step_us"].sum()
print(f"device_sum_per_step={tot:.0f}us  top25:")
print(bytype.head(25).to_string(float_format=lambda x: f"{x:,.1f}"))
print(f"\nwin_rows={len(win)}")

# stream 分布
bystream = win.groupby("Stream ID")["Task Duration(us)"].agg(["count","sum"]).sort_values("sum", ascending=False)
print("\nstream distribution:")
print(bystream.head(12).to_string())
