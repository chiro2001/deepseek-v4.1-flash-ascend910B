#!/usr/bin/env python3
"""从 profile 的 MoE 算子耗时**反推活跃专家数**（不需要 Input Shapes）。

原理：
  · w1/w3 每专家权重 = 2·I·H·0.5 B（int4）；w2 = I·H·0.5 B
  · 该算子已被证明是纯权重带宽受限（token 64× 只 +4%，见 tools/moe_gmm_bench.py）
  · 若它以接近实测峰值带宽运行，则  字节 = 耗时 × BW ⇒  专家数 = 字节 / 每专家字节

用法: moe_expert_reverse.py <ASCEND_PROFILER_OUTPUT> [BW_GBps]
"""
import sys

import pandas as pd

D = sys.argv[1]
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 1300.0   # 实测纯读上限

H, I = 5120, 2304
B_W13 = 2 * I * H * 0.5
B_W2 = I * H * 0.5

df = pd.read_csv(D + "/operator_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Device Self Duration(us)": "dur_us"})

steps = 72.0   # armF_r6_base 窗口步数（与其它分析脚本一致）

print("每专家权重：w1/w3 = %.1f MB，w2 = %.1f MB   （假设带宽 %.0f GB/s）"
      % (B_W13 / 1e6, B_W2 / 1e6, BW))
print()
print("%-46s %7s %12s %12s %12s" % ("算子", "次数", "自耗时ms/步", "µs/层", "⇒活跃专家"))

for pat, bytes_per, label in (
    ("GroupedMatmulSwigluQuant", B_W13, "w1/w3"),
    ("GroupedMatmul", B_W2, "w2"),
):
    sub = df[df["name"].astype(str).str.contains(pat, regex=False, na=False)]
    # "GroupedMatmul" 会同时匹配 SwigluQuant 那个，去掉
    if pat == "GroupedMatmul":
        sub = sub[~sub["name"].astype(str).str.contains("Swiglu", regex=False, na=False)]
    if sub.empty:
        print("%-46s  (无)" % label)
        continue
    n = len(sub)
    tot_ms = sub["dur_us"].sum() / 1000.0 / steps
    per_layer_us = sub["dur_us"].median()
    # 每层活跃专家 = (中位耗时 × BW) / 每专家字节
    experts = (per_layer_us * 1e-6 * BW * 1e9) / bytes_per
    print("%-46s %7d %12.2f %12.1f %12.1f"
          % (label + "  (" + pat + ")", n, tot_ms, per_layer_us, experts))

print()
print("=== 参考：若读到全部 48 个本地专家，单层耗时应为 ===")
print("   w1/w3: %.1f µs     w2: %.1f µs"
      % (48 * B_W13 / (BW * 1e9) * 1e6, 48 * B_W2 / (BW * 1e9) * 1e6))
print()
print("=== 若按 EP=8 均匀分散推算的活跃数 ===")
print("   该 profile 是 conc=6（n=6）⇒ 36 token/步 × top6 = 216 routes")
print("   均匀分散到 8 rank ⇒ 27 routes/rank ⇒ 若完全打散最多 27 个专家")
