#!/usr/bin/env python3
"""判定 AivKernel(core=COMMUNICATION) 与 hcom_* 是否同一批事件，并区分：
  · 谁在 comm 窗口里占着 AIC / AIV（AIV 是否只是驱动 kernel 在空转）
  · 排除"驱动 kernel 自身"后，通信的真实暴露是多少
用法: python3 prof_comm_identity.py <ASCEND_PROFILER_OUTPUT>
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
STEP = (HI - LO) / 1000 / nst
print("STEP = %.3f ms，%d 步" % (STEP, nst))


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
        if a_iv[i][1] < a_iv[j][1] if False else a_iv[i][1] < b_iv[j][1]:
            i += 1
        else:
            j += 1
    return tot / 1000


ak = w[w["name"] == "AivKernel"]
hc = w[w["name"].str.startswith("hcom_")]
ak_iv, hc_iv = union(ak), union(hc)
q = inter(ak_iv, hc_iv) / nst
a = sum(e - s for s, e in ak_iv) / 1000 / nst
c = sum(e - s for s, e in hc_iv) / 1000 / nst
print("AivKernel union %.3f | hcom union %.3f | 二者重叠 %.3f  ⇒ 同一批事件? %s"
      % (a, c, q, "YES（重叠≈min）" if q > 0.9 * min(a, c) else "NO"))

cat = {"AI_CORE": "AIC", "MIX_AIC": "AIC", "AI_VECTOR_CORE": "AIV", "MIX_AIV": "AIV"}
compute = w[w["core"].isin(cat)].copy()
compute["cat"] = compute["core"].map(cat)
aic_iv = union(compute[compute["cat"] == "AIC"])
aiv_iv = union(compute[compute["cat"] == "AIV"])
oth_iv = union(w[~w["core"].isin(cat) & (w["name"] != "AivKernel")])

print("\n=== 通信窗口里到底谁在忙（以 hcom 窗口为样本）===")
print("  AIC 在 comm 窗口内的时长        : %.3f ms/步" % (inter(hc_iv, aic_iv) / nst))
print("  AIV(真实计算核) 在窗口内        : %.3f ms/步" % (inter(hc_iv, aiv_iv) / nst))
print("  其它非计算核 kernel 在窗口内    : %.3f ms/步" % (inter(hc_iv, oth_iv) / nst))
print("  comm union                      : %.3f ms/步" % c)
print("  ⇒ AIC 在通信期间基本空闲        : %s" % ("YES" if inter(hc_iv, aic_iv) / nst < 0.1 * c else "NO"))

print("\n=== 通信窗口内的 Top kernel（按核心类型）===")
inside = w[(w["st"] >= LO) & (w["st"] < HI)]
rows = []
for s, e in hc_iv:
    m = (inside["st"] < e) & (inside["en"] > s)
    sub = inside[m]
    if len(sub):
        rows.append(sub.groupby("name")["dur"].sum())
if rows:
    agg = pd.concat(rows, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
    for n, v in agg.head(12).items():
        print("   %-56s %.3f ms/步" % (str(n)[:56], v / 1000 / nst))

print("\n=== allreduceAicpuKernel / allgatherAicpuKernel（真正的集合通信入口）===")
for nm in ("allreduceAicpuKernel", "allgatherAicpuKernel", "alltoallAicpuKernel"):
    sub = w[w["name"] == nm]
    if len(sub):
        print("   %-24s n=%d  合计 %.3f ms/步  中位 %.1f us"
              % (nm, len(sub), sub["dur"].sum() / 1000 / nst, sub["dur"].median()))

print("\n=== hcom_* 与 allreduceAicpuKernel 的时间关系（前 3 个样本）===")
for s, e in hc_iv[:3]:
    near = w[(w["st"] > s - 200) & (w["st"] < e + 200)][["name", "st", "dur", "core"]]
    print("  --- 窗口 %.0f-%.0f us ---" % (s, e))
    for _, r in near.iterrows():
        print("     %-40s st=%.0f dur=%.1f core=%s" % (str(r["name"])[:40], r["st"], r["dur"], r["core"]))
