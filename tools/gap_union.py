#!/usr/bin/env python3
"""步长缺口分析：找出「设备上什么都没有在跑」的时间段及其邻居。

动机：conc=1 的步长是 24.58 ms，而 profile 显示 AIC 只忙 13.55 ms。
如果 AIC∪AIV∪COMM∪AICPU 的并集也明显小于步长，差额就是**纯空闲**——
它既不是计算也不是通信，而是 host/调度/同步的产物。

用法: gap_union.py <ASCEND_PROFILER_OUTPUT> [min_gap_us]
"""
import sys
from collections import Counter

import pandas as pd

D = sys.argv[1]
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0

df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]

marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
LO, HI = marks[2], marks[-3]
w = df[(df["st"] >= LO) & (df["st"] < HI)].copy()
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst
print("窗口 %.1f ms / %d 步 ⇒ STEP = %.3f ms" % ((HI - LO) / 1000, nst, STEP))

# 每步的窗口边界
marks_in = [m for m in marks if LO <= m < HI]


def union_with_names(sub):
    """返回 (并集区间, 该区间内的算子名集合)。"""
    iv = sorted(zip(sub["st"], sub["en"], sub["name"]))
    out = []
    cs, ce, names = iv[0][0], iv[0][1], [iv[0][2]]
    for s, e, n in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
            names.append(n)
        else:
            out.append((cs, ce, names))
            cs, ce, names = s, e, [n]
    out.append((cs, ce, names))
    return out


allu = union_with_names(w)
busy = sum(e - s for s, e, _ in allu) / 1000 / nst

cat = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV", "MIX_AIV": "AIV",
       "AI_CPU": "CPU"}
w["cat"] = w["core"].map(cat).fillna("COMM")
print("全设备并集 %.3f ms/步（%.1f%% of STEP）⇒ **纯空闲 %.3f ms/步（%.1f%%）**"
      % (busy, 100 * busy / STEP, STEP - busy, 100 * (STEP - busy) / STEP))

for lbl, sel in (("AIC", w[w["cat"] == "AIC"]), ("AIV", w[w["cat"] == "AIV"]),
                 ("COMM", w[w["cat"] == "COMM"]), ("CPU", w[w["cat"] == "CPU"])):
    if len(sel):
        u = union_with_names(sel)
        print("  %-5s 并集 %.3f ms/步" % (lbl, sum(e - s for s, e, _ in u) / 1000 / nst))

# ---- 空闲缺口（窗口内，全设备并集的补集）----
gaps = []
for i in range(len(allu) - 1):
    g0, g1 = allu[i][1], allu[i + 1][0]
    if g1 - g0 >= MIN_GAP:
        gaps.append((g0, g1, g1 - g0, allu[i][2], allu[i + 1][2]))
print("\n=== ≥%.0f µs 的空闲缺口：共 %d 个，合计 %.3f ms/步 ==="
      % (MIN_GAP, len(gaps), sum(g[2] for g in gaps) / 1000 / nst))

tot = Counter()
for g0, g1, d, before, after in gaps:
    tot["%.0f-%.0fµs" % (0, 0)]  # 占位
buckets = Counter()
for g0, g1, d, _b, _a in gaps:
    if d < 100:
        buckets["50-100us"] += 1
    elif d < 200:
        buckets["100-200us"] += 1
    elif d < 500:
        buckets["200-500us"] += 1
    elif d < 1000:
        buckets["0.5-1ms"] += 1
    elif d < 3000:
        buckets["1-3ms"] += 1
    else:
        buckets["≥3ms"] += 1
print("  按长度分布:", dict(buckets))

gaps.sort(key=lambda x: -x[2])
print("\n=== 最大的 12 个缺口 ===")
print("%10s %10s %12s   %-40s → %-40s" % ("起点µs", "长度µs", "折算ms/步", "前驱算子", "后继算子"))
for g0, g1, d, before, after in gaps[:12]:
    bn = Counter(before).most_common(1)[0][0]
    an = Counter(after).most_common(1)[0][0]
    print("%10.0f %10.0f %12.4f   %-40s → %-40s"
          % (g0, d, d / 1000 / nst, str(bn)[:40], str(an)[:40]))
