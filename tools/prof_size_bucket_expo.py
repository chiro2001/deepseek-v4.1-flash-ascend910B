#!/usr/bin/env python3
"""按**算子规模分桶**统计聚合暴露度（纯标准库，无需 pandas）。

为什么需要它：`prof_expo2.py` 是**逐个算子**排名，把 ~1000 个 1–10 µs 的小算子
每个都排到榜外；但它们在设备上是**背靠背串行**的，聚合起来才是真实可回收时间。
本工具按时长分桶，给出每桶的「次数/自身合计/并集/聚合暴露」。

用法: prof_size_bucket_expo.py <ASCEND_PROFILER_OUTPUT 目录> [step_real_ms]
"""

from __future__ import annotations

import csv
import sys

STEP_REAL_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 24.59
D = sys.argv[1]

rows: list[tuple[float, float, float]] = []  # (start, end, dur)
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        try:
            st = float(r["Start Time(us)"])
            du = float(r["Duration(us)"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((st, st + du, du))

rows.sort()
marks = sorted(st for st, en, du in rows if "allgatherAicpu" in str(du) or False)
# 步边界：allgatherAicpuKernel 每步恰好一次 —— 需要按名字筛，重新扫一遍名字列
marks = []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        if "allgatherAicpu" in (r.get("Name") or "") or "allgatherAicpu" in (
            r.get("OP Type") or ""
        ):
            try:
                marks.append(float(r["Start Time(us)"]))
            except (KeyError, TypeError, ValueError):
                pass
marks.sort()
if len(marks) < 6:
    raise SystemExit("找不到足够的 allgatherAicpu 锚点，无法切步")

LO, HI = marks[2], marks[-3]
sel = [(st, en, du) for st, en, du in rows if LO <= st < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP_P = (HI - LO) / 1000 / nst
K = STEP_P / STEP_REAL_MS


def merged(starts_ends):
    if not starts_ends:
        return []
    iv = sorted(starts_ends)
    out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def total(intervals):
    return sum(e - s for s, e in intervals)


def overlap(a, b):
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
    return tot


BUCKETS = [
    (0, 2, "<2us"),
    (2, 5, "2-5us"),
    (5, 10, "5-10us"),
    (10, 20, "10-20us"),
    (20, 50, "20-50us"),
    (50, 100, "50-100us"),
    (100, 500, "0.1-0.5ms"),
    (500, 2000, "0.5-2ms"),
    (2000, float("inf"), ">2ms"),
]

print(
    "步长 profile %.2f ms | K=%.3f | 步数 %d | 算子 %.0f/步"
    % (STEP_P, K, nst, len(sel) / nst)
)
print(
    "%-11s %9s %10s %10s %11s %10s %9s"
    % ("桶", "次数/步", "自身ms", "并集ms", "暴露ms", "真实ms", "占步长")
)

durs = [du for st, en, du in sel]
tot_real = 0.0
for lo, hi, label in BUCKETS:
    grp = [(st, en) for st, en, du in sel if lo <= du < hi]
    if not grp:
        continue
    cnt = len(grp) / nst
    own = sum(en - st for st, en in grp) / 1000 / nst
    iv = merged(grp)
    uni = total(iv) / 1000 / nst
    oth = merged([(st, en) for st, en, du in sel if not (lo <= du < hi)])
    exp = (total(iv) - overlap(iv, oth)) / 1000 / nst
    real = exp / K
    tot_real += real
    print(
        "%-11s %9.1f %10.3f %10.3f %11.3f %10.3f %8.1f%%"
        % (label, cnt, own, uni, exp / K * K, real, 100 * real / STEP_REAL_MS)
    )

print(
    "\n桶真实暴露合计 %.2f ms/步（%.1f%%）"
    % (tot_real, 100 * tot_real / STEP_REAL_MS)
)
print("全设备并集 %.2f ms/步（profile）" % (total(merged([(st, en) for st, en, du in sel])) / 1000 / nst))
