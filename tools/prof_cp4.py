#!/usr/bin/env python3
"""逐算子：实际耗时 vs 该算子大小在满带宽下应耗时 ⇒ 找出"离带宽饱和差多少"。

字节估算（按算子名匹配）：
  * HcPre/HcPost: M=6 × 5120 × 2B（读+写各一份）
  * RmsNorm/DynamicQuant/RoPE 等 elementwise: M×5120×2B × 2
  * MoE grouped GEMM: 每专家 16.87 MiB（W4）
  * attention 投影: QKV/O 权重按 M=6 的 GEMM
  * SparseFlashMla: 读 KV（上下文 × 层）
关键是**量级**，不是精确到 B。
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 1180e9     # 可达带宽 B/s
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur"})
df = df.sort_values("st")
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
lo, hi = marks[2], marks[-3]
w = df[(df["st"] >= lo) & (df["st"] < hi)]
nst = len([m for m in marks if lo <= m < hi])
tot_ms = w["dur"].sum() / 1000 / nst
print("稳态 %d 步；算子里程合计 %.1f ms/步" % (nst, tot_ms))

# 粗估每个算子的搬运字节（按名字）
M = 6
H = 5120
def est_bytes(n: str) -> float:
    n = str(n)
    if "HcPre" in n or "HcPost" in n:
        return 2 * M * H * 2            # 读+写
    if n in ("RmsNorm", "DynamicQuant", "InplacePartialRotaryMul", "AivKernel"):
        return 2 * M * H * 2
    if "GroupedMatmulSwigluQuant" in n or "GroupedMatmulWeightNz" in n:
        return 4.5 * 16.87 * 2**20      # 每步每 die 约 4.5 个专家 × 16.87 MiB
    if "SparseFlashMla" in n and "Metadata" not in n:
        return 60 * 1024 * 0.5 * M      # 粗估：topk 512 × 576B × M / 分片
    if "QuantBatchMatmulV3" in n or "MatMulV2" in n or "MatMulV3" in n:
        return 4 * M * H * 2            # 投影类（量级估计）
    if "allreduce" in n.lower() or "allReduce" in n:
        return 2 * M * H * 2
    return 0.0

g = w.groupby("name").agg(tot_ms=("dur", lambda s: s.sum() / 1000 / nst),
                          n=("dur", lambda s: s.size / nst),
                          med_us=("dur", "median"))
g["bytes_per_step"] = [est_bytes(i) * g.loc[i, "n"] for i in g.index]
g["bw_time_ms"] = g["bytes_per_step"] / BW * 1000
g["excess_ms"] = g["tot_ms"] - g["bw_time_ms"]
g = g.sort_values("tot_ms", ascending=False)
print("\n%-52s %9s %8s %9s %10s %10s %8s" % ("算子", "ms/步", "次数/步", "中位µs", "字节/步MB", "满带宽ms", "超额ms"))
for name, r in g.head(18).iterrows():
    print("%-52s %9.3f %8.1f %9.1f %10.2f %10.3f %8.3f" % (
        str(name)[:52], r["tot_ms"], r["n"], r["med_us"], r["bytes_per_step"] / 2**20,
        r["bw_time_ms"], r["excess_ms"]))

tot_ex = g["excess_ms"].sum()
tot_bwt = g["bw_time_ms"].sum()
print("\n全部算子：合计 %.1f ms/步；估算搬运 %.2f GiB/步 ⇒ 满带宽需 %.2f ms" % (
    g["tot_ms"].sum(), g["bytes_per_step"].sum() / 2**30, tot_bwt))
print("⇒ **超额（非带宽）时间 %.1f ms/步 = %.0f%%**" % (tot_ex, 100 * tot_ex / g["tot_ms"].sum()))
print("   （口径：按可达带宽 %.0f GB/s 折算；超额≈下发/同步/依赖等待/算力）" % (BW / 1e9))
