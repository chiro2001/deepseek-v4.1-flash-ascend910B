#!/usr/bin/env python3
"""在步内的哪个位置、什么在跑；主流 AIC/AIV 是"块状交替"还是"细碎交替"。"""
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
LO, HI = marks[2], marks[-3]
w = df[(df["st"] >= LO) & (df["st"] < HI)].copy()
nst = len([m for m in marks if LO <= m < HI])
CAT = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV",
       "MIX_AIV": "AIV", "COMMUNICATION": "COMM", "AI_CPU": "CPU"}
w["cat"] = w["core"].map(CAT).fillna("?")
main_sid = w.groupby("sid")["dur"].sum().idxmax()

# ---- 步内位置分布（10 等分）----
print("=== 步内位置分布：每条流把多少 busy 时间花在步的第几成 ===")
print("（用 allgatherAicpuKernel 标志定义步起点；10 格 = 0-10%..90-100%）")
print("%-9s %8s %s" % ("stream", "busy/步", " ".join("%5d%%" % (i * 10) for i in range(10))))
rows = []
for sid, g in w.groupby("sid"):
    tot = g["dur"].sum() / 1000 / nst
    if tot < 0.3: continue
    h = [0.0] * 10
    for m in marks[2:-2]:
        seg = g[(g["st"] >= m) & (g["st"] < m + (marks[3] - marks[2]))]
        if seg.empty: continue
        frac = (seg["st"] - m) / (marks[3] - marks[2])
        for f, d in zip(frac, seg["dur"]):
            i = min(9, max(0, int(f * 10)))
            h[i] += d / 1000
    nseg = max(1, len(marks) - 4)
    rows.append((tot, sid, [x / nseg for x in h]))
for tot, sid, h in sorted(rows, reverse=True)[:9]:
    tag = "  ← 主" if sid == main_sid else ""
    print("%-9s %8.3f %s%s" % (sid, tot, " ".join("%5.2f" % x for x in h), tag))

# ---- 主流 AIC/AIV 交替结构 ----
print("\n=== 主流 %s 的 AIC/AIV 交替结构 ===" % main_sid)
m = w[w["sid"] == main_sid].sort_values("st")
aic_t = m[m["cat"] == "AIC"]["dur"].sum() / 1000 / nst
aiv_t = m[m["cat"] == "AIV"]["dur"].sum() / 1000 / nst
oth_t = m[~m["cat"].isin(["AIC", "AIV"])]["dur"].sum() / 1000 / nst
print("  主流 busy 合计 %.2f ms/步  = AIC %.2f + AIV %.2f + 其它 %.2f" % (
    aic_t + aiv_t + oth_t, aic_t, aiv_t, oth_t))
seq = list(m["cat"])
sw = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
print("  类别切换 %d 次/窗口 = %.1f 次/步 ⇒ 平均每 %.1f µs 切一次" % (
    sw, sw / nst, (m["dur"].sum() / nst / max(sw / nst, 1))))
# 连续同类的"块长"分布
runs, cur, cnt = [], seq[0], 0
for c in seq:
    if c == cur: cnt += 1
    else: runs.append((cur, cnt)); cur, cnt = c, 1
runs.append((cur, cnt))
from collections import defaultdict as dd
by = dd(list)
for c, n in runs: by[c].append(n)
for c in ("AIC", "AIV"):
    v = by.get(c, [])
    if v:
        v.sort()
        print("  %s 连续块：%d 段/窗口，块内算子数 中位 %.0f p90 %.0f 最大 %d" % (
            c, len(v), v[len(v)//2], v[int(len(v)*0.9)], v[-1]))

# ---- 步尾（最后 20%）是谁在跑 ----
print("\n=== 步尾（最后 20%）的活动归属 ===")
tail_attr = Counter(); tail_tot = 0.0
for m0 in marks[2:-2]:
    seg = m0 + (marks[3] - marks[2])
    t0, t1 = seg - 0.2 * (marks[3] - marks[2]), seg
    for sid, g in w.groupby("sid"):
        ov = 0.0
        for s, e in zip(g["st"], g["en"]):
            ov += max(0.0, min(t1, e) - max(t0, s))
        if ov > 0:
            tail_attr[sid] += ov / 1000
            tail_tot += ov / 1000
nseg = max(1, len(marks) - 4)
for sid, v in tail_attr.most_common(7):
    g = w[w["sid"] == sid]
    cc = Counter(g["cat"])
    print("  stream %-7s %6.3f ms/步 (%.0f%%)  AIC=%d AIV=%d  主力: %s" % (
        sid, v / nseg, 100 * v / (tail_tot or 1), cc.get("AIC", 0), cc.get("AIV", 0),
        "; ".join("%s×%d" % (str(k)[:22], c) for k, c in Counter(g["name"]).most_common(2))))
