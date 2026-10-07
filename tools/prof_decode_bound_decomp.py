#!/usr/bin/env python3
"""decode bound 拆解 v2：在 MATH-IS-7PCT 的四桶基础上，再往下拆三层。

四桶（math/scalar/MTE/unattributed）已由 prof_math_vs_overhead.py 给出。
本工具补充三个此前未拆的维度：

  ① **unattributed 归因**：哪些算子贡献了那 37.5%？（它是 duration − (mac+vec+scalar+mte)）
  ② **MTE 有效性**：MTE 活跃窗口内的有效带宽 vs 峰值 ⇒ 是"搬得多"还是"搬得慢"
  ③ **结构性地板**：按算子时长分桶，量化"纯下发开销"占多少

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_decode_bound_decomp.py <profdir>
"""
from __future__ import annotations
import csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))
MATH = ("aic_mac_time(us)", "aiv_vec_time(us)")
SCAL = ("aic_scalar_time(us)", "aiv_scalar_time(us)")
MOVE = ("aic_mte1_time(us)", "aic_mte2_time(us)", "aic_fixpipe_time(us)",
        "aiv_mte2_time(us)", "aiv_mte3_time(us)")

def f(x):
    try: return float(x)
    except: return 0.0

rows, marks = [], []
with open(os.path.join(D, "kernel_details.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        d = f(r.get("Duration(us)"))
        if d <= 0: continue
        m = sum(f(r.get(c)) for c in MATH)
        s = sum(f(r.get(c)) for c in SCAL)
        v = sum(f(r.get(c)) for c in MOVE)
        rows.append(dict(t=f(r.get("Start Time(us)")), d=d, core=(r.get("Accelerator Core") or "").strip(),
                         nm=(r.get("Name") or ""), math=m, scal=s, move=v, unattr=max(0.0, d-m-s-v)))
        if (r.get("Name") or "").startswith(ANCHOR): marks.append(rows[-1]["t"])
rows.sort(key=lambda x: x["t"]); marks.sort()
if len(marks) > 2 * PER:
    lo, hi = marks[PER], marks[-PER]
    steps = max(1, (len(marks) - 2 * PER) // PER)
else:
    lo, hi = rows[0]["t"], rows[-1]["t"]; steps = 1
sel = [x for x in rows if lo <= x["t"] < hi]
STEP = (hi - lo) / steps / 1000.0
n = len(sel) / steps

T = lambda k: sum(x[k] for x in sel) / steps / 1000.0
sd, sm, ss, sv, su = T("d"), T("math"), T("scal"), T("move"), T("unattr")
print("步长 %.3f ms | 算子 %.0f 个/步 | 窗口 %d 步" % (STEP, n, steps))
print("Σ算子时长 %.3f ms  (并发度 %.3f)" % (sd, sd / STEP))
print()
print("=== ① 四桶（占步长）===")
for lab, val in (("真数学  mac+vec", sm), ("标量    scalar", ss),
                 ("搬运    MTE", sv), ("未归因  duration−三项", su)):
    print("  %-22s %7.3f ms  %6.1f%%" % (lab, val, val / STEP * 100))

print("\n=== ② unattributed Top12（谁在'等'）===")
agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
for x in sel:
    a = agg[x["nm"][:44]]
    a[0] += x["d"]; a[1] += x["unattr"]; a[2] += x["math"]; a[3] += x["move"]; a[4] += 1
print("  %-44s %8s %8s %7s" % ("算子", "时长ms", "未归因ms", "占未归因"))
for k, a in sorted(agg.items(), key=lambda y: -y[1][1])[:12]:
    print("  %-44s %8.3f %8.3f %6.1f%%" % (k, a[0] / steps / 1000, a[1] / steps / 1000,
                                            a[1] / sum(v[1] for v in agg.values()) * 100))

print("\n=== ③ 算子时长分布（量化'纯下发'）===")
buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
print("  %-12s %8s %8s %9s %9s" % ("时长区间µs", "个数/步", "占比", "合计ms", "数学ms"))
for lo_, hi_ in buckets:
    g = [x for x in sel if lo_ <= x["d"] < hi_]
    if not g: continue
    lab = "%.0f-%.0f" % (lo_, hi_) if hi_ < 1e8 else ">50"
    print("  %-12s %8.0f %7.1f%% %9.3f %9.3f" % (lab, len(g) / steps, len(g) / len(sel) * 100,
          sum(x["d"] for x in g) / steps / 1000, sum(x["math"] for x in g) / steps / 1000))

print("\n=== ④ MTE 有效性 ===")
BYTES_GB = 4.38   # 每步搬运量（A3-DECODE-BANDWIDTH 解析式，已与 npu-smi 178 GB/s 交叉验证）
PEAK = 1619.0     # GB/s（npu-smi 标定：1181.7 → 73%）
print("  每步搬运 %.2f GB | 峰值 %.0f GB/s" % (BYTES_GB, PEAK))
print("  整步平均带宽      %6.1f GB/s  (%4.1f%% of peak)" % (BYTES_GB / (STEP / 1000), BYTES_GB / (STEP / 1000) / PEAK * 100))
print("  Σ MTE 活跃 = %.3f ms (占 Σ算子时长 %.1f%%)" % (sv, sv / sd * 100))
print("  ⇒ MTE 窗口内有效带宽 ≤ %.1f GB/s (%4.1f%% of peak)  [上界；实际更低]" %
      (BYTES_GB / sv * 1000, BYTES_GB / sv * 1000 / PEAK * 100))

print("\n=== ⑤ 结构性地板 ===")
print("  Σ 真数学 = %.3f ms  ⇒ 若步长 = Σ数学，加速比 %.1fx" % (sm, STEP / sm))
print("  Σ 标量   = %.3f ms  ⇒ 若标量完全消除，省 %.1f%%" % (ss, ss / STEP * 100))
print("  Σ 未归因 = %.3f ms  ⇒ 若等待完全消除，省 %.1f%%" % (su, su / STEP * 100))
print("  算子数 %.0f 个/步；按合成链实测边际成本 2.0 µs/算子 ⇒ 下发地板 %.2f ms (%.1f%%)" %
      (n, n * 0.002, n * 0.002 / STEP * 100))
