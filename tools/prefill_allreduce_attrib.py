#!/usr/bin/env python3
"""prefill allreduce 归因：算出每个 forward 里 allreduce 的**次数、间隔、时长分布**，
并据此推断调用点（每层几次/每次多大）。
用法: ar_attrib.py <kernel_details.csv 目录>
"""
import statistics
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]

ar = df[df["name"].str.startswith("hcom_allReduce", na=False)].reset_index(drop=True)
print("hcom_allReduce 总数 = %d" % len(ar))
print("时长 µs: min=%.1f p25=%.1f 中位=%.1f p75=%.1f max=%.1f  合计=%.1f ms"
      % (ar["dur"].min(), ar["dur"].quantile(.25), ar["dur"].median(),
         ar["dur"].quantile(.75), ar["dur"].max(), ar["dur"].sum() / 1000))

# 用 HcPre 的出现次数作为 "层" 的标尺（每层 2 次）
hc = df[df["name"] == "HcPre"]
print("\nHcPre 总数 = %d ⇒ 约 %d 个 (层,forward) 组合（每层 2 次）" % (len(hc), len(hc) // 2))

# 相邻 allreduce 的间隔
gaps = [(ar["st"][i + 1] - ar["en"][i]) for i in range(len(ar) - 1)]
pos = [g for g in gaps if g >= 0]
print("\n相邻 allReduce 间隔（µs）: 中位=%.0f  均值=%.0f  p90=%.0f  max=%.0f"
      % (statistics.median(pos), statistics.mean(pos),
         sorted(pos)[int(.9 * len(pos))], max(pos)))
print("间隔 <50µs 的占比 = %.1f%%（同时发起/背靠背）"
      % (100.0 * sum(1 for g in pos if g < 50) / max(1, len(pos))))

# 每个 forward 内 allreduce 数：用大间隔（>2ms）切分
big = [i for i, g in enumerate(gaps) if g > 2000]
groups = []
start = 0
for i in big:
    groups.append((start, i))
    start = i + 1
groups.append((start, len(ar) - 1))
sizes = [b - a + 1 for a, b in groups if b - a + 1 >= 20]
print("\n按 >2ms 间隔切分 ⇒ %d 段；段内 allreduce 数：%s"
      % (len(groups), sorted(sizes)[:20]))
if sizes:
    print("  中位段大小 = %d  ⇒ 每段（≈一个 chunk 的 forward）约这么多次 allreduce"
          % statistics.median(sizes))

# 时长直方（区分不同消息大小的调用点）
print("\n时长分桶（µs）:")
buckets = Counter()
for d in ar["dur"]:
    for lo, hi in ((0, 200), (200, 1000), (1000, 3000), (3000, 8000), (8000, 20000), (20000, 1e9)):
        if lo <= d < hi:
            buckets["%d-%d" % (lo, hi if hi < 1e9 else 99999)] += 1
            break
for k in ("0-200", "200-1000", "1000-3000", "3000-8000", "8000-20000", "20000-99999"):
    if buckets[k]:
        print("  %-14s %d 次" % (k, buckets[k]))

# 按 handle（名字里的 __503_X_Y）分组
ar["handle"] = ar["name"].str.extract(r"__503_(\d+_\d+)")
top = ar.groupby("handle").agg(n=("dur", "size"), med=("dur", "median"), tot=("dur", "sum"))
top["pct"] = 100 * top["tot"] / ar["dur"].sum()
print("\n按 handle 分组（次数 ≥10 的）:")
for h, r in top[top["n"] >= 10].sort_values("tot", ascending=False).head(12).iterrows():
    print("  %-10s n=%-5d 中位=%8.1f µs  合计=%9.1f ms  %5.1f%%"
          % (h, r["n"], r["med"], r["tot"] / 1000, r["pct"]))
