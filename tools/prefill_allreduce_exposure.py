#!/usr/bin/env python3
"""prefill 的 allreduce 到底是"真在关键路径"还是"重叠累加的假象"？

疑点：`allreduceAicpuKernel` 均值 9.5 ms，而实际搬运 `hcom_allReduce` 中位只有 0.81 ms（差 12×）。
⇒ 需要按**暴露度**（union 与其它 kernel 的重叠）来判断，而不是简单求和。

输出：
  · 窗口总长、各资源的 union
  · allreduceAicpuKernel 的 union 与"被其它 kernel 覆盖"的时长 ⇒ 暴露
  · hcom_allReduce 的同样指标

用法: ar_exposure.py <目录>
"""
import sys

import pandas as pd

D = sys.argv[1]
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Stream ID": "sid", "Accelerator Core": "core"})
df = df.sort_values("st").reset_index(drop=True)
df["en"] = df["st"] + df["dur"]
LO, HI = df["st"].min(), df["en"].max()
print("窗口 = %.1f ms（%d 个 kernel）" % ((HI - LO) / 1000, len(df)))


def union(iv):
    if not iv:
        return []
    iv = sorted(iv)
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
    return sum(e - s for s, e in iv) / 1000.0


def inter(a, b):
    tot = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot / 1000.0


ar_cpu = df[df["name"] == "allreduceAicpuKernel"]
ar_com = df[df["name"].str.startswith("hcom_allReduce", na=False)]
others = df[(df["name"] != "allreduceAicpuKernel")
            & (~df["name"].str.startswith("hcom_allReduce", na=False))]

u_cpu, u_com, u_oth = union(list(zip(ar_cpu["st"], ar_cpu["en"]))), \
                      union(list(zip(ar_com["st"], ar_com["en"]))), \
                      union(list(zip(others["st"], others["en"])))

print()
print("%-28s %10s %12s %12s" % ("项", "次数", "求和ms", "union ms"))
print("%-28s %10d %12.1f %12.1f" % ("allreduceAicpuKernel", len(ar_cpu),
                                     ar_cpu["dur"].sum() / 1000, dur(u_cpu)))
print("%-28s %10d %12.1f %12.1f" % ("hcom_allReduce", len(ar_com),
                                     ar_com["dur"].sum() / 1000, dur(u_com)))
print("%-28s %10s %12s %12.1f" % ("其它 kernel", "-", "-", dur(u_oth)))

print()
print("=== 暴露度 ===")
ov_cpu = inter(u_cpu, u_oth)
ov_com = inter(u_com, u_oth)
print("allreduceAicpuKernel: union=%.1f  被其它覆盖=%.1f  **暴露=%.1f ms（占窗口 %.1f%%）**"
      % (dur(u_cpu), ov_cpu, dur(u_cpu) - ov_cpu,
         100 * (dur(u_cpu) - ov_cpu) / ((HI - LO) / 1000)))
print("hcom_allReduce:       union=%.1f  被其它覆盖=%.1f  **暴露=%.1f ms（占窗口 %.1f%%）**"
      % (dur(u_com), ov_com, dur(u_com) - ov_com,
         100 * (dur(u_com) - ov_com) / ((HI - LO) / 1000)))

print()
print("=== 对照：AICPU 与 hcom 的关系 ===")
print("  allreduceAicpuKernel ∩ hcom_allReduce = %.1f ms" % inter(u_cpu, u_com))
print("  二者 union = %.1f ms" % dur(union(list(zip(ar_cpu["st"], ar_cpu["en"]))
                                          + list(zip(ar_com["st"], ar_com["en"])))))

print()
print("=== 窗口里 only-allreduce 的时长（union 减去与其它 kernel 的重叠）===")
u_ar = union(list(zip(ar_cpu["st"], ar_cpu["en"])) + list(zip(ar_com["st"], ar_com["en"])))
ov = inter(u_ar, u_oth)
print("  allreduce 族 union=%.1f ms  与其它覆盖=%.1f  **独占窗口=%.1f ms（%.1f%%）**"
      % (dur(u_ar), ov, dur(u_ar) - ov, 100 * (dur(u_ar) - ov) / ((HI - LO) / 1000)))


# ---------------- 空闲段分析（39.5% 空闲在哪） ----------------
print()
print("=== 空闲段分析（>=0.5 ms）===")
alliv = []
alliv = union(list(zip(df['st'], df['en'])))
gaps = []
for i in range(len(alliv) - 1):
    g0, g1 = alliv[i][1], alliv[i + 1][0]
    if (g1 - g0) / 1000.0 >= 0.5:
        gaps.append((g0, g1, (g1 - g0) / 1000.0))
print(">=0.5ms 空闲段: %d 个，合计 %.1f ms（占窗口 %.1f%%）"
      % (len(gaps), sum(g[2] for g in gaps), 100 * sum(g[2] for g in gaps) / ((HI - LO) / 1000)))
from collections import Counter as _C
buck = _C()
for _, _, d in gaps:
    for lo, hi, lbl in ((0.5, 1, "0.5-1ms"), (1, 2, "1-2ms"), (2, 5, "2-5ms"),
                        (5, 10, "5-10ms"), (10, 50, "10-50ms"), (50, 1e9, ">=50ms")):
        if lo <= d < hi:
            buck[lbl] += 1
            break
print("  长度分布:", {k: buck[k] for k in ("0.5-1ms", "1-2ms", "2-5ms", "5-10ms", "10-50ms", ">=50ms") if buck[k]})
print()
print("最长的 10 个空闲段:")
for g0, g1, d in sorted(gaps, key=lambda x: -x[2])[:10]:
    print("   %10.1f ms 起（相对 %.1f ms）  长度 %8.1f ms" % ((g0 - LO) / 1000, LO, d))
