#!/usr/bin/env python3
"""小算子海（<N µs）按**算子名**聚合排名（纯标准库）。

用途：`prof_size_bucket_expo.py` 证明 <20µs 的 2383 个算子合计暴露 8.24 ms/步
（33.5%），但那是**分桶**；本工具回答"具体是哪几种算子、各占多少"，
直接给出融合靶点清单。

用法: prof_small_op_ranking.py <ASCEND_PROFILER_OUTPUT> [阈值us] [step_real_ms]
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict

THR = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
STEP_REAL_MS = float(sys.argv[3]) if len(sys.argv) > 3 else 24.59
D = sys.argv[1]

rows = []
marks = []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = (r.get("Name") or "")
        ty = (r.get("OP Type") or "")
        if "allgatherAicpu" in nm or "allgatherAicpu" in ty:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception:
            continue
        core = r.get("Accelerator Core") or r.get("Task Type") or ""
        rows.append((st, st + du, du, nm, core))
marks.sort()
if len(marks) < 6: raise SystemExit("锚点不足")
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP_P = (HI - LO) / 1000 / nst
K = STEP_P / STEP_REAL_MS

agg = defaultdict(lambda: [0, 0.0])   # name -> [count, dur_sum]
core_of = {}
for st, en, du, nm, core in sel:
    if du < THR:
        a = agg[nm]; a[0] += 1; a[1] += du
        core_of[nm] = core

tot_cnt = sum(v[0] for v in agg.values())
tot_ms = sum(v[1] for v in agg.values()) / 1000 / nst
tot_all = len(sel) / nst
print(f"步长 profile {STEP_P:.2f} ms | K={K:.3f} | 全步算子 {tot_all:.0f}/步")
print(f"<{THR:.0f}us 算子: {tot_cnt/nst:.1f}/步 ({100*tot_cnt/len(sel):.1f}% of count) | 自身 {tot_ms:.3f} ms/步 ({100*tot_ms/STEP_REAL_MS:.1f}% real, 未扣重叠)")
print()
print(f"{'算子名':<52}{'次数/步':>9}{'us/步':>10}{'ms/步':>9}{'占步长':>8}  core")
for nm, (c, s) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]:
    per = c / nst
    ms = s / 1000 / nst
    print(f"{nm[:52]:<52}{per:>9.1f}{s/nst:>10.1f}{ms:>9.3f}{100*ms/STEP_REAL_MS:>7.1f}%  {core_of.get(nm,'')[:14]}")
