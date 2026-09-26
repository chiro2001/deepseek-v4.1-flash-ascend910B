#!/usr/bin/env python3
"""用 HcPre 次数精确定位步边界，给出**每步周期**（不受 prefill/尾部污染）。

每步 target forward 每层调 2 次 `HcPre`（attn 前 + ffn 前）：
  SPEC=0 → 40 层 × 2 = 80 次/步
  SPEC=7 → (40 + 3 草稿) × 2 = 86 次/步
把 HcPre 的 start time 排序后每隔 per_step 取一个作为步边界，
相邻边界之差即一个 step 周期。比"窗口 ÷ 步数"准，也不受批大小变化影响。

用法：
  python3 step_period.py kernel_details.csv 80 "A SPEC=0 conc4"
"""
import csv
import statistics
import sys

path, per_step, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
every = int(sys.argv[4]) if len(sys.argv) > 4 else 20
ts = []
for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
    if (r.get("Name") or "").split("_")[0] != "HcPre":
        continue
    v = (r.get("Start Time(us)") or "").strip()
    if v:
        try:
            ts.append(float(v))
        except ValueError:
            pass
ts.sort()
b = ts[::per_step]
o = b[0]
d = [(b[i] - b[i - 1]) / 1000.0 for i in range(1, len(b))]
s = sorted(d)
print(f"== {label}")
print(f"   HcPre={len(ts)}  per_step={per_step}  →  步数={len(b)}")
print(f"   step 周期：median={statistics.median(s):.2f}ms  "
      f"p10={s[len(s)//10]:.2f}  p90={s[-max(1,len(s)//10)]:.2f}  "
      f"min={s[0]:.2f}  max={s[-1]:.2f}")
print(f"\n{'step':>6}{'t(s)':>9}{'period_ms':>11}   |  {every}步窗口")
for i in range(0, len(d), every):
    p = d[i:i + every]
    t = (b[i + 1] - o) / 1e6
    print(f"{i:>6}{t:>9.3f}{statistics.median(p):>11.2f}   |  "
          f"max={max(p):>7.1f} min={min(p):>6.1f}")
