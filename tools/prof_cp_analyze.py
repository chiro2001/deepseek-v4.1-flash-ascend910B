#!/usr/bin/env python3
"""decode 关键路径解析：为什么带宽上不去。

口径：
  * 步边界 = 相邻任务起始时间间隔 > 阈值处切分（decode 步之间有明显空隙）
  * 每步：各 stream 的 busy 时间、跨 stream 的 union（= 真实"有人在算"的时间）
  * 主计算流 = union 贡献最大的 stream；它的 **串行链长度** 与 **步长** 之比是核心指标
  * 关键路径下界：Σ(每个算子的固定开销) —— 用各算子的**中位时长**估
"""
import sys
from collections import defaultdict

import pandas as pd

D = sys.argv[1]
GAP_US = float(sys.argv[2]) if len(sys.argv) > 2 else 3000.0
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]

# --- 步边界 ---
uniq = sorted(df["st"].unique())
bd = [uniq[0]]
for i in range(1, len(uniq)):
    if uniq[i] - uniq[i - 1] > GAP_US:
        bd.append(uniq[i])
bd.append(df["en"].max())
print("步边界 %d 个 ⇒ %d 步（间隔阈值 %.0f µs）" % (len(bd), len(bd) - 1, GAP_US))
if len(bd) < 3:
    print("步太少，改用全局窗口")
    bd = [df["st"].min(), df["en"].max()]

# --- 每步统计 ---
rows = []
for k in range(len(bd) - 1):
    lo, hi = bd[k], bd[k + 1]
    w = df[(df["st"] >= lo) & (df["st"] < hi)]
    if w.empty:
        continue
    span = (w["en"].max() - w["st"].min()) / 1000.0
    # 每 stream busy（并集）
    per = {}
    for sid, g in w.groupby("sid"):
        iv = sorted(zip(g["st"], g["en"]))
        tot, cs, ce = 0.0, None, None
        for s, e in iv:
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                tot += ce - cs
                cs, ce = s, e
        tot += ce - cs
        per[sid] = tot / 1000.0
    union = 0.0
    alliv = sorted(zip(w["st"], w["en"]))
    cs = ce = None
    for s, e in alliv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            union += ce - cs
            cs, ce = s, e
    union += ce - cs
    rows.append(dict(step=k, span=span, union=union / 1000.0, nops=len(w),
                     sum_dur=w["dur"].sum() / 1000.0, per=per))
R = pd.DataFrame(rows)
print("\n=== 每步统计（ms）===")
print("%5s %9s %9s %9s %9s %9s %8s" % ("step", "span", "union", "sum_dur", "sum/span", "union/span", "nops"))
for _, r in R.head(14).iterrows():
    print("%5d %9.3f %9.3f %9.3f %9.3f %9.2f %8d" % (
        r["step"], r["span"], r["union"], r["sum_dur"], r["sum_dur"] / r["span"],
        r["union"] / r["span"], r["nops"]))
med = R.iloc[1:-1] if len(R) > 4 else R
print("\n中位步：span=%.3f ms  union=%.3f ms (%.0f%%)  算子数=%.0f" % (
    med["span"].median(), med["union"].median(),
    100 * (med["union"] / med["span"]).median(), med["nops"].median()))

# --- 各 stream 在典型步上的占比 ---
mid = R.iloc[len(R) // 2]
per = mid["per"]
print("\n=== 典型步（step %d，span %.3f ms）各 stream busy（ms）===" % (mid["step"], mid["span"]))
for sid, v in sorted(per.items(), key=lambda kv: -kv[1])[:12]:
    print("  stream %-6s %8.3f ms  (%.0f%% of span)" % (sid, v, 100 * v / mid["span"]))
