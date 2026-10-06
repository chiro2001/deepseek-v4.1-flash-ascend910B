#!/usr/bin/env python3
"""把上一版的"COMM"拆成三份，避免把 AIV 上跑的 AivKernel 误算成"通信暴露"：
  (a) AivKernel（engram all_gather，vector 核）
  (b) hcom_*（真正的 HCCL 集合通信）
  (c) 其它
并对 (b) 单独算 union / 与 AIC∪AIV 的重叠 / 暴露。
用法: python3 prof_comm_split.py <ASCEND_PROFILER_OUTPUT>
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
print("每步 STEP = %.3f ms（%d 步）" % (STEP, nst))

cat = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV",
       "MIX_AIV": "AIV", "AI_CPU": "CPU"}
w["cat"] = w["core"].map(cat).fillna("COMM")

print("\n=== AivKernel 的 Accelerator Core 取值 ===")
ak = w[w["name"] == "AivKernel"]
print(ak["core"].value_counts().to_dict())
print("   AivKernel 合计 %.3f ms/步（%.1f 个/步）" %
      (ak["dur"].sum() / 1000 / nst, len(ak) / nst))

hcom = w[w["name"].str.startswith("hcom_")]
print("\n=== hcom_* 合计 ===")
print("   合计 %.3f ms/步（%.1f 个/步）" %
      (hcom["dur"].sum() / 1000 / nst, len(hcom) / nst))
byp = hcom.assign(prefix=hcom["name"].str.replace(r"_\d+$", "", regex=True)) \
          .groupby("prefix")["dur"].agg(["size", "sum"])
byp["ms/step"] = byp["sum"] / 1000 / nst
byp["n/step"] = byp["size"] / nst
print(byp.sort_values("sum", ascending=False).head(12)[["n/step", "ms/step"]])


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
    return tot / 1000


compute = w[w["cat"].isin(["AIC", "AIV", "CPU"])]
c_iv = union(compute)
for label, sub in (("AivKernel", ak), ("hcom_*", hcom)):
    iv = union(sub)
    u = sum(e - s for s, e in iv) / 1000 / nst
    ov = inter(iv, c_iv) / nst
    print("\n[%s] union %.3f ms/步；与 AIC∪AIV∪CPU 重叠 %.3f；**暴露 %.3f ms/步（%.1f%% of step）**"
          % (label, u, ov, u - ov, 100 * (u - ov) / STEP))

print("\n=== hcom_* 的流分布 ===")
print((hcom.groupby("sid")["dur"].sum() / 1000 / nst).sort_values(ascending=False).head(6))

print("\n=== hcom_* 时间分布（10 等分）===")
bins = Counter()
for _, r in hcom.iterrows():
    bins[min(9, max(0, int((r["st"] - LO) / (HI - LO) * 10)))] += r["dur"]
for i in range(10):
    print("   %2d0%%-%2d0%%  %.3f ms/步" % (i, i + 1, bins[i] / 1000 / nst))

print("\n=== hcom_* 窗口的紧邻算子（判断能否重叠）===")
alliv = sorted(zip(w["st"], w["en"], w["name"], w["cat"]))
before, after = Counter(), Counter()
for s, e in union(hcom)[:100]:
    b = a2_ = None
    for s2, e2, n2, c2 in alliv:
        if e2 <= s and (b is None or e2 > b[1]):
            b = (s2, e2, n2, c2)
        if s2 >= e and (a2_ is None or s2 < a2_[0]):
            a2_ = (s2, e2, n2, c2)
    if b:
        before["%s [%s]" % (str(b[2])[:46], b[3])] += 1
    if a2_:
        after["%s [%s]" % (str(a2_[2])[:46], a2_[3])] += 1
print("  前：")
for n, k in before.most_common(6):
    print("    %-58s %d" % (n, k))
print("  后：")
for n, k in after.most_common(6):
    print("    %-58s %d" % (n, k))
