#!/usr/bin/env python3
"""到底什么在并行、掩盖了什么延迟。

三问：
  Q1 每条流上是什么算子（AIC 还是 AIV？）
  Q2 主流的空闲窗口被谁填了（逐窗口归因）
  Q3 主流为什么不能和侧流重叠更多（同流内是否 AIC/AIV 混合 ⇒ 自串行）
"""
import sys
from collections import Counter, defaultdict

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
STEP = (HI - LO) / 1000 / nst
print("稳态 %d 步，步长 %.2f ms，算子 %.0f/步\n" % (nst, STEP, len(w) / nst))

CAT = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV",
       "MIX_AIV": "AIV", "COMMUNICATION": "COMM", "AI_CPU": "CPU"}
w["cat"] = w["core"].map(CAT).fillna("?")


def union(sub):
    iv = sorted(zip(sub["st"], sub["en"]))
    if not iv: return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return out


# ---- Q1 每条流是什么 ----
main_sid = w.groupby("sid")["dur"].sum().idxmax()
print("=== Q1 各流构成（主 = stream %s）===" % main_sid)
print("%-9s %7s %8s %8s %7s %7s %7s  %s" % ("stream", "算子/步", "busy/步", "占步长", "AIC", "AIV", "COMM", "主要算子"))
rows = []
for sid, g in w.groupby("sid"):
    b = sum(e - s for s, e in union(g)) / 1000 / nst
    cc = Counter(g["cat"])
    top = Counter(g["name"]).most_common(3)
    rows.append((b, sid, len(g) / nst, cc, top, g))
for b, sid, n, cc, top, g in sorted(rows, reverse=True)[:10]:
    print("%-9s %7.1f %8.3f %7.0f%% %7d %7d %7d  %s" % (
        sid, n, b, 100 * b / STEP, cc.get("AIC", 0), cc.get("AIV", 0), cc.get("COMM", 0),
        "; ".join("%s×%d" % (str(k)[:26], v) for k, v in top)))

# ---- Q2 主流空闲窗口被谁填 ----
main = w[w["sid"] == main_sid]
miv = union(main)
gaps = []
for i in range(len(miv) - 1):
    gs, ge = miv[i][1], miv[i + 1][0]
    if ge - gs > 50:                      # >50 µs
        gaps.append((gs, ge))
tot_gap = sum(e - s for s, e in gaps) / 1000 / nst
print("\n=== Q2 主流空闲窗口（>50µs）：%.0f 个/步，合计 %.2f ms/步（占步长 %.0f%%）===" % (
    len(gaps) / nst, tot_gap, 100 * tot_gap / STEP))
attrib = defaultdict(float)
for gs, ge in gaps[:4000]:
    for sid, g in w.groupby("sid"):
        if sid == main_sid: continue
        ov = 0.0
        for s, e in zip(g["st"], g["en"]):
            ov += max(0.0, min(ge, e) - max(gs, s))
        if ov > 0:
            attrib[sid] += ov
tot_att = sum(attrib.values()) or 1
print("  窗口内谁在跑（按占用 µs/步 排序）：")
for sid, v in sorted(attrib.items(), key=lambda kv: -kv[1])[:8]:
    g = w[w["sid"] == sid]
    cc = Counter(g["cat"])
    print("    stream %-7s %8.2f µs/步  (%.0f%%)  AIC=%d AIV=%d  主力: %s" % (
        sid, v / 1000 / nst, 100 * v / tot_att, cc.get("AIC", 0), cc.get("AIV", 0),
        "; ".join("%s×%d" % (str(k)[:24], c) for k, c in Counter(g["name"]).most_common(2))))

# ---- Q3 同流内 AIC/AIV 是否混合 ----
print("\n=== Q3 同一流内 AIC/AIV 混合度（混合 ⇒ 该流内自串行）===")
for b, sid, n, cc, top, g in sorted(rows, reverse=True)[:8]:
    aic, aiv = cc.get("AIC", 0), cc.get("AIV", 0)
    if aic + aiv == 0: continue
    # 交替次数：相邻算子的类别变化
    gg = g.sort_values("st")
    seq = list(gg["cat"])
    alt = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    print("  stream %-7s AIC=%5d AIV=%5d  类别切换 %5d 次/窗口（%.1f/步）  %s" % (
        sid, aic, aiv, alt, alt / nst,
        "**混合(自串行)**" if aic and aiv and alt > n * 0.2 else "较单一"))
