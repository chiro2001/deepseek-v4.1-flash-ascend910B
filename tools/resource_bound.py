#!/usr/bin/env python3
"""真正的「完美重叠上界」是多少？

目标文档写的上界是 max(AIC, AIV, comm, cpu) = AIC = 13.55 real ms ⇒ 1.81×。
但那是**把每个资源各自的并集**拿来取 max —— 只有当"其它所有资源的总工作量"
也能塞进 AIC 的忙时窗口时，这个上界才成立。

正确做法：
  a = union(AIC)
  o = union(AIV ∪ COMM ∪ CPU)        ← 非 AIC 的**总占用时间**（不是求和）
  ⇒ 即使完美并行，步长也不可能小于 max(a, o)
本脚本把 a、o、以及各项 union / 两两重叠一起算出来，给出**真实可及上界**。

用法: resource_bound.py <ASCEND_PROFILER_OUTPUT>
"""
import sys

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
STEP_P = (HI - LO) / 1000 / nst          # profile ms/步
K = 40.020 / 24.59                        # profile→real 换算
STEP_R = STEP_P / K

cat = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV", "MIX_AIV": "AIV",
       "AI_CPU": "CPU"}
w["cat"] = w["core"].map(cat).fillna("COMM")


def union(sub):
    iv = sorted(zip(sub["st"], sub["en"]))
    if not iv:
        return []
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def dur(iv):
    return sum(e - s for s, e in iv) / 1000 / nst      # profile ms/步


def inter(a_iv, b_iv):
    tot = 0.0
    i = j = 0
    while i < len(a_iv) and j < len(b_iv):
        s = max(a_iv[i][0], b_iv[j][0])
        e = min(a_iv[i][1], b_iv[j][1])
        if e > s:
            tot += e - s
        if a_iv[i][1] < b_iv[j][1]:
            i += 1
        else:
            j += 1
    return tot / 1000 / nst


U = {c: union(w[w["cat"] == c]) for c in ("AIC", "AIV", "COMM", "CPU")}
non_aic = union(w[w["cat"].isin(("AIV", "COMM", "CPU"))])

print("STEP(profile) = %.3f ms/步   换算 real = %.3f ms/步" % (STEP_P, STEP_R))
print()
print("%-24s %12s %12s %10s" % ("资源", "union(prof)", "union(real)", "占步长"))
for c in ("AIC", "AIV", "COMM", "CPU"):
    d = dur(U[c])
    print("%-24s %12.3f %12.3f %9.1f%%" % (c, d, d / K, 100 * d / STEP_P))
d_na = dur(non_aic)
print("%-24s %12.3f %12.3f %9.1f%%" % ("非AIC(AIV∪COMM∪CPU)", d_na, d_na / K, 100 * d_na / STEP_P))

allu = union(w)
d_all = dur(allu)
print()
print("全设备并集   = %.3f prof ms (%.1f%%)  ⇒ 纯空闲 %.3f prof ms = %.3f real ms"
      % (d_all, 100 * d_all / STEP_P, STEP_P - d_all, (STEP_P - d_all) / K))

a = dur(U["AIC"])
print()
print("=== 完美重叠上界 ===")
print("  目标文档写法 : max(AIC, AIV, COMM, CPU) = %.3f prof = %.3f real"
      % (a, a / K))
print("  正确写法     : max(AIC, 非AIC总占用)    = max(%.3f, %.3f) = %.3f prof = %.3f real"
      % (a, d_na, max(a, d_na), max(a, d_na) / K))
print("  ⇒ 真实可及上界 = %.2f×（相对当前 %.3f real ms）"
      % (STEP_R / (max(a, d_na) / K), STEP_R))
print()
print("=== 各资源与 AIC 的重叠（决定还能再挤多少）===")
print("%-10s %12s %12s" % ("资源", "与AIC重叠", "自身未被AIC覆盖"))
for c in ("AIV", "COMM", "CPU"):
    ov = inter(U[c], U["AIC"])
    print("%-10s %12.3f %12.3f" % (c, ov, dur(U[c]) - ov))
print()
print("AIC 忙时被覆盖        = %.3f（AIC 与至少一个其它资源重叠）" % inter(U["AIC"], non_aic))
print("AIC 独忙（无人陪跑）  = %.3f prof ms" % (a - inter(U["AIC"], non_aic)))
