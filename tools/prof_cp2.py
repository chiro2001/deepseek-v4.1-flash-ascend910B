#!/usr/bin/env python3
"""按 engram all_gather 标志切步，做关键路径归因。

回答：为什么图下发 + free 不高，带宽却只有 11%？
"""
import statistics as st
import sys

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]

marks = sorted(df[df["name"] == "allgatherAicpuKernel"]["st"].values)
print("标志 %d 个 ⇒ 步数 %d" % (len(marks), len(marks) - 1))
if len(marks) < 5:
    raise SystemExit("标志太少")

def busy(g):
    iv = sorted(zip(g["st"], g["en"]))
    tot, cs, ce = 0.0, None, None
    segs = []
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs; segs.append((cs, ce)); cs, ce = s, e
    tot += ce - cs; segs.append((cs, ce))
    return tot / 1000.0, segs

rows = []
for k in range(len(marks) - 1):
    lo, hi = marks[k], marks[k + 1]
    w = df[(df["st"] >= lo) & (df["st"] < hi)]
    if w.empty: continue
    span = (hi - lo) / 1000.0
    ubusy, segs = busy(w)
    rows.append(dict(step=k, span=span, union=ubusy, nops=len(w),
                     sumdur=w["dur"].sum() / 1000.0,
                     maxend=(w["en"].max() - lo) / 1000.0,
                     minst=(w["st"].min() - lo) / 1000.0))
R = pd.DataFrame(rows)
w = R.iloc[2:-2]
print("\n=== 每步（ms）：span=步长 union=至少一个核在忙 头尾=首个/最后一个算子相对步起点 ===")
print("%5s %8s %8s %8s %8s %8s %8s %8s" % ("step", "span", "union", "sumdur", "首算子", "末算子", "union%", "nops"))
for _, r in R.head(10).iterrows():
    print("%5d %8.2f %8.2f %8.2f %8.2f %8.2f %7.0f%% %8d" % (
        r["step"], r["span"], r["union"], r["sumdur"], r["minst"], r["maxend"],
        100 * r["union"] / r["span"], r["nops"]))
print("\n中位：span=%.2f  union=%.2f (%.0f%%)  sumdur=%.2f (%.2f×span)  nops=%.0f" % (
    w["span"].median(), w["union"].median(), 100 * (w["union"] / w["span"]).median(),
    w["sumdur"].median(), (w["sumdur"] / w["span"]).median(), w["nops"].median()))
print("尾部：末算子比步末早 %.2f ms ⇒ 步尾**没有任何任务**的时间" % (
    w["span"].median() - w["maxend"].median()))
print("头部：首算子比步起点晚 %.2f ms" % w["minst"].median())

# 典型步的 stream 占用
mid = R.iloc[len(R) // 2]
lo, hi = marks[mid["step"]], marks[mid["step"] + 1]
w2 = df[(df["st"] >= lo) & (df["st"] < hi)]
print("\n=== 典型步（step %d，span %.2f ms）各 stream（ms / %） ===" % (mid["step"], mid["span"]))
per = []
for sid, g in w2.groupby("sid"):
    b, _ = busy(g)
    per.append((sid, b, len(g)))
for sid, b, n in sorted(per, key=lambda x: -x[1])[:10]:
    print("  stream %-7s %8.2f ms %5.0f%%   n=%d" % (sid, b, 100 * b / mid["span"], n))
