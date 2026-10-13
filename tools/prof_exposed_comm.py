#!/usr/bin/env python3
"""通信有多少是"暴露"的 —— 即那段时间里**没有别的资源在跑**。

如果通信与任何 AIC/AIV 都不重叠，它就在关键路径上；
但"能不能被隐藏"还取决于**依赖**：如果下一层的第一个算子依赖它，就藏不了。
本脚本先量前者（暴露时长），并列出通信窗口前后紧邻的算子（判断依赖）。
"""
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
STEP = (HI - LO) / 1000 / nst

def union(sub):
    iv = sorted(zip(sub["st"], sub["en"]))
    if not iv: return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return out

def inter(a_iv, b_iv):
    tot = 0.0; i = j = 0
    while i < len(a_iv) and j < len(b_iv):
        s = max(a_iv[i][0], b_iv[j][0]); e = min(a_iv[i][1], b_iv[j][1])
        if e > s: tot += e - s
        if a_iv[i][1] < b_iv[j][1]: i += 1
        else: j += 1
    return tot / 1000

cat = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV",
       "MIX_AIV": "AIV", "AI_CPU": "CPU"}
w["cat"] = w["core"].map(cat).fillna("COMM")

comm_iv = union(w[w["cat"] == "COMM"])
aic_iv  = union(w[w["cat"] == "AIC"])
aiv_iv  = union(w[w["cat"] == "AIV"])
cpu_iv  = union(w[w["cat"] == "CPU"])
other   = union(w[w["cat"].isin(["AIC", "AIV", "CPU"])])

print("每步：COMM union %.3f ms（真实 %.3f）" % (sum(e-s for s,e in comm_iv)/1000/nst,
                                             sum(e-s for s,e in comm_iv)/1000/nst/ (STEP/ (HI-LO)/1000*nst) * (STEP/((HI-LO)/1000)) ))
c = sum(e - s for s, e in comm_iv) / 1000 / nst
a = sum(e - s for s, e in aic_iv) / 1000 / nst
v = sum(e - s for s, e in aiv_iv) / 1000 / nst
u = sum(e - s for s, e in cpu_iv) / 1000 / nst
ov = inter(comm_iv, other) / nst
print("  COMM %.3f | AIC %.3f | AIV %.3f | CPU %.3f" % (c, a, v, u))
print("  COMM ∩ (AIC∪AIV∪CPU) = %.3f  ⇒ **暴露 %.3f ms/步（%.1f%% of step）**" % (ov, c - ov, 100*(c-ov)/STEP))

# 通信窗口的"前后紧邻"算子（判断依赖）
print("\n=== 通信窗口的紧邻算子（前后各 3 个，按时间）===")
main_sid = w.groupby("sid")["dur"].sum().idxmax()
alliv = sorted(zip(w["st"], w["en"], w["name"], w["sid"], w["cat"]))
before = Counter(); after = Counter()
for s, e in comm_iv[:60]:
    # 找通信窗口开始前最后一个结束的算子
    b = None
    for s2, e2, n2, sid2, c2 in alliv:
        if e2 <= s and (b is None or e2 > b[1]): b = (s2, e2, n2, sid2, c2)
    a2_ = None
    for s2, e2, n2, sid2, c2 in alliv:
        if s2 >= e and (a2_ is None or s2 < a2_[0]): a2_ = (s2, e2, n2, sid2, c2)
    if b: before[str(b[2])[:40]] += 1
    if a2_: after[str(a2_[2])[:40]] += 1
print("  通信开始前最后完成的算子 Top6：")
for n, k in before.most_common(6): print("    %-44s %d" % (n, k))
print("  通信结束后第一个开始的算子 Top6：")
for n, k in after.most_common(6): print("    %-44s %d" % (n, k))
