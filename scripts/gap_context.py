#!/usr/bin/env python3
"""空闲洞的上下文：每个洞的前驱/后继是哪个算子类型（定位最贵的依赖边）。
用法: gap_context.py <mindstudio_profiler_output>
"""
import glob, sys, collections
import pandas as pd
import numpy as np

M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type", "Task Start Time(us)", "Task Duration(us)", "Task Type"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files],
               ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
cut = df["s"].max() * 0.3
w = df[df["s"] >= cut].reset_index(drop=True)
core = w[w["Task Type"] != "AI_CPU"].reset_index(drop=True)

# 合并 AI core 忙区间，同时记录该区间的"最后结束的算子"与"最开始的下一个算子"
core = core.sort_values("s").reset_index(drop=True)
ivs = []   # [start, end, owner_op]
for s, e, ot in zip(core["s"], core["e"], core["OP Type"]):
    if ivs and s <= ivs[-1][1] + 1e-9:
        ivs[-1][1] = max(ivs[-1][1], e)
        ivs[-1][2].add(ot)
    else:
        ivs.append([s, e, {ot}])

# 把每个原始算子映射到它所属（合并后）区间的下标，用于找"洞前最后一个算子"
pred = collections.defaultdict(float)
pairs = []
for k in range(len(ivs) - 1):
    gap = ivs[k+1][0] - ivs[k][1]
    if gap <= 0.005:
        continue
    # 洞前：区间 k 里结束最晚的算子
    endk = ivs[k][1]
    cand = core[(core["e"] >= endk - 0.05) & (core["s"] <= endk)]
    prev = cand.iloc[-1]["OP Type"] if len(cand) else "?"
    startn = ivs[k+1][0]
    cand2 = core[(core["s"] >= startn - 0.05) & (core["e"] >= startn)]
    nxt = cand2.iloc[0]["OP Type"] if len(cand2) else "?"
    pairs.append((gap, prev, nxt))
pairs.sort(reverse=True)
tot = sum(p[0] for p in pairs)
print(f"洞数={len(pairs)}  空闲合计={tot:.1f}ms")
print("\n=== 最大 15 个洞 ===")
for gap, p, n in pairs[:15]:
    print(f"  {gap*1000:8.1f} us   {str(p)[:38]:38s} -> {str(n)[:38]}")
print("\n=== 按洞大小加权的 (前驱->后继) top 15 ===")
agg = collections.defaultdict(float); cnt = collections.Counter()
for gap, p, n in pairs:
    agg[(p, n)] += gap; cnt[(p, n)] += 1
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v*1000:9.1f} us ({cnt[k]:5d} 个, 均值 {v/cnt[k]*1000:7.1f} us)  {str(k[0])[:34]} -> {str(k[1])[:34]}")
print("\n=== 前驱侧 top 10（不管后继）===")
aggp = collections.defaultdict(float)
for gap, p, n in pairs: aggp[p] += gap
for k, v in sorted(aggp.items(), key=lambda x: -x[1])[:10]:
    print(f"  {v*1000:9.1f} us   {str(k)[:50]}")
