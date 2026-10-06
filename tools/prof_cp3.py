#!/usr/bin/env python3
"""为什么 96% 占用却只有 11% 带宽：算子粒度与并发度分析。"""
import statistics as st
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)

# 稳态窗口（跳过前 2 步与后 2 步）
lo, hi = marks[2], marks[-3]
w = df[(df["st"] >= lo) & (df["st"] < hi)].copy()
nst = len([m for m in marks if lo <= m < hi])
span = (hi - lo) / 1000.0
print("稳态窗口 %.1f ms / %d 步 ⇒ 步长 %.2f ms，算子 %d（每步 %.0f）" % (
    span, nst, span / nst, len(w), len(w) / nst))

d = w["dur"].values
print("\n=== 算子时长分布（全 %d 个）===" % len(d))
import numpy as np
for q in (10, 25, 50, 75, 90, 95, 99):
    print("  p%-3d %10.1f µs" % (q, np.percentile(d, q)))
print("  合计 %.1f ms ⇒ 每步 %.1f ms（占步长 %.0f%%）" % (
    d.sum() / 1000, d.sum() / 1000 / nst, 100 * (d.sum() / 1000 / nst) / (span / nst)))
for th in (1, 2, 5, 10, 20, 50, 100):
    n = (d <= th).sum()
    print("  ≤%4d µs: %6d 个 (%.0f%%)  合计 %8.1f ms (%.0f%% of 步长×步数)" % (
        th, n, 100 * n / len(d), d[d <= th].sum() / 1000,
        100 * (d[d <= th].sum() / 1000) / span))

# 并发度：任一时刻同时在跑的算子数
ev = []
for s, e in zip(w["st"], w["en"]):
    ev.append((s, 1)); ev.append((e, -1))
ev.sort()
cur = 0; mx = 0; acc = 0.0; prev = None; hist = Counter()
for t, delta in ev:
    if prev is not None and cur > 0:
        acc += (t - prev) * cur
        hist[cur] += t - prev
    prev = t; cur += delta; mx = max(mx, cur)
print("\n=== 并发度（同时运行的算子数）===")
print("  峰值 %d，时间加权均值 %.2f" % (mx, acc / (ev[-1][0] - ev[0][0])))
for k in sorted(hist)[:10]:
    print("  同时 %2d 个算子: %8.1f ms (%.0f%%)" % (k, hist[k] / 1000, 100 * hist[k] / (ev[-1][0] - ev[0][0])))

# 主计算流
print("\n=== 各 stream 的 busytime 与算子上限（每步 ms）===")
per = {}
for sid, g in w.groupby("sid"):
    iv = sorted(zip(g["st"], g["en"])); tot = 0.0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    tot += ce - cs
    per[sid] = dict(busy=tot / 1000 / nst, n=len(g) / nst, sumdur=g["dur"].sum() / 1000 / nst)
for sid, v in sorted(per.items(), key=lambda kv: -kv[1]["busy"])[:8]:
    print("  stream %-7s busy %7.2f ms/步   算子 %6.0f 个/步   时长合计 %7.2f ms/步  均值 %5.1f µs" % (
        sid, v["busy"], v["n"], v["sumdur"], 1000 * v["sumdur"] / max(v["n"], 1)))
