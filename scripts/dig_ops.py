#!/usr/bin/env python3
"""深挖: 通信 Op Name 细分 + MatMulV2/HcPre/QBMV3 形状细分 (稳态窗口)"""
import glob, sys, re
import pandas as pd

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["Stream ID", "Op Name", "OP Type", "Task Start Time(us)", "Task Duration(us)", "Input Shapes"]
dfs = [pd.read_csv(f, usecols=cols, low_memory=False) for f in files]
df = pd.concat(dfs, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["t_rel"] = df["Task Start Time(us)"] - t0
gmm1 = df[df["OP Type"] == "GroupedMatmulSwigluQuantV2"]["t_rel"].to_numpy()
starts = gmm1[::40]
n0 = int(len(starts) * 0.2)
win = df[df["t_rel"] >= starts[n0]].copy()
nsteps = len(starts) - n0
print(f"steps={nsteps}")

print("\n===== hcom_allReduce_ 明细 (Op Name top15) =====")
ar = win[win["OP Type"] == "hcom_allReduce_"]
g = ar.groupby("Op Name")["Task Duration(us)"].agg(["count", "sum", "mean"])
g["per_step_cnt"] = g["count"] / nsteps
g["per_step_us"] = g["sum"] / nsteps
print(g.sort_values("sum", ascending=False).head(15).to_string(float_format=lambda x: f"{x:,.1f}"))

print("\n===== MatMulV2 形状 top15 =====")
mm = win[win["OP Type"] == "MatMulV2"]
g2 = mm.groupby("Input Shapes")["Task Duration(us)"].agg(["count", "sum", "mean"])
g2["per_step_cnt"] = g2["count"] / nsteps
g2["per_step_us"] = g2["sum"] / nsteps
print(g2.sort_values("sum", ascending=False).head(15).to_string(float_format=lambda x: f"{x:,.1f}"))

print("\n===== MatMulV2 Op Name top10 =====")
g2b = mm.groupby("Op Name")["Task Duration(us)"].agg(["count", "sum", "mean"])
g2b["per_step_cnt"] = g2b["count"] / nsteps
g2b["per_step_us"] = g2b["sum"] / nsteps
print(g2b.sort_values("sum", ascending=False).head(10).to_string(float_format=lambda x: f"{x:,.1f}"))

print("\n===== HcPre stream 分布 =====")
hp = win[win["OP Type"] == "HcPre"]
print(hp.groupby("Stream ID")["Task Duration(us)"].agg(["count","sum","mean"]).sort_values("sum",ascending=False).head(8).to_string())

print("\n===== allreduce by stream =====")
print(ar.groupby("Stream ID")["Task Duration(us)"].agg(["count","sum","mean"]).sort_values("count",ascending=False).head(8).to_string())
