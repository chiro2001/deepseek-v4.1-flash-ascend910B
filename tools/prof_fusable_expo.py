#!/usr/bin/env python3
"""把「小算子海」按**是否可融合**分类，算每类的聚合暴露度。

可融合类 = 非 matmul/非集合通信 的 elementwise / 转换 / 搬运 / 索引算子。
这些是"融合靶点"；matmul 类即使小也不能靠融合消除。

用法: prof_fusable_expo.py <ASCEND_PROFILER_OUTPUT> [阈值us] [step_real_ms]
"""
from __future__ import annotations
import csv, sys

THR = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
STEP_REAL_MS = float(sys.argv[3]) if len(sys.argv) > 3 else 24.59
D = sys.argv[1]

MATMUL_KEYS = ("Matmul", "MatMul", "GroupedMatmul", "BatchMatmul")
COMM_KEYS = ("allreduce", "allgather", "AllReduce", "AllGather", "Hccl", "AivKernel",
             "hcom", "AicpuKernel", "Metadata")

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""; ty = r.get("OP Type") or ""
        if "allgatherAicpu" in nm or "allgatherAicpu" in ty:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        rows.append((st, st + du, du, nm))
marks.sort()
if len(marks) < 6: raise SystemExit("锚点不足")
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP_P = (HI - LO) / 1000 / nst; K = STEP_P / STEP_REAL_MS

def merged(iv):
    if not iv: return []
    iv = sorted(iv); out, cs, ce = [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce)); return out
def total(iv): return sum(e - s for s, e in iv)
def overlap(a, b):
    t = 0.0; i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: t += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return t

def cls(nm):
    if any(k in nm for k in MATMUL_KEYS): return "matmul"
    if any(k in nm for k in COMM_KEYS): return "comm/metadata"
    return "fusable"

groups = {"fusable": [], "matmul": [], "comm/metadata": []}
small = {"fusable": [], "matmul": [], "comm/metadata": []}
for st, en, du, nm in sel:
    c = cls(nm); groups[c].append((st, en))
    if du < THR: small[c].append((st, en, du))

print(f"步长 profile {STEP_P:.2f} ms | K={K:.3f} | 算子 {len(sel)/nst:.0f}/步 | 阈值 {THR:.0f}us")
print(f"{'类别':<16}{'次数/步':>9}{'自身ms':>9}{'并集ms':>9}{'暴露ms':>9}{'真实ms':>9}{'占步长':>8}")
for c, lst in groups.items():
    if not lst: continue
    iv = merged(lst); uni = total(iv) / 1000 / nst
    oth = merged([x for k, v in groups.items() if k != c for x in v])
    exp = (total(iv) - overlap(iv, oth)) / 1000 / nst
    print(f"{c:<16}{len(lst)/nst:>9.1f}{sum(e-s for s,e in lst)/1000/nst:>9.3f}{uni:>9.3f}{exp:>9.3f}{exp/K:>9.3f}{100*exp/K/STEP_REAL_MS:>7.1f}%")

print(f"\n只看 <{THR:.0f}us：")
print(f"{'类别':<16}{'次数/步':>9}{'自身ms':>9}{'并集ms':>9}{'暴露ms':>9}{'真实ms':>9}{'占步长':>8}")
for c, lst in small.items():
    if not lst: continue
    iv = merged([(s, e) for s, e, d in lst])
    uni = total(iv) / 1000 / nst
    oth = merged([(st, en) for st, en, du, nm in sel if cls(nm) != c])
    exp = (total(iv) - overlap(iv, oth)) / 1000 / nst
    print(f"{c:<16}{len(lst)/nst:>9.1f}{sum(d for s,e,d in lst)/1000/nst:>9.3f}{uni:>9.3f}{exp:>9.3f}{exp/K:>9.3f}{100*exp/K/STEP_REAL_MS:>7.1f}%")
