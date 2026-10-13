#!/usr/bin/env python3
"""用"每步只出现一次"的标志算子找真实步长，再切步。"""
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur", "Stream ID": "sid"})
df = df.sort_values("st").reset_index(drop=True)
span_s = (df["st"].max() - df["st"].min()) / 1e6
print("profile 跨度 %.2f s，算子总数 %d" % (span_s, len(df)))

c = Counter(df["name"])
print("\n出现次数最少的 20 个算子（候选'每步一次'标志）：")
for n, k in c.most_common()[-20:]:
    print("  %-62s %6d" % (str(n)[:62], k))

# 找出现次数在 50~200 之间、且时间间隔稳定的算子
print("\n候选标志算子（50<=count<=250）：")
for n, k in c.items():
    if 50 <= k <= 250:
        ts = sorted(df[df["name"] == n]["st"].values)
        d = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
        if not d:
            continue
        import statistics as st
        med = st.median(d)
        if med > 1000 and st.pstdev(d) / med < 0.35:
            print("  %-56s n=%4d 间隔中位 %8.1f µs  变异 %.2f" % (str(n)[:56], k, med, st.pstdev(d)/med))
