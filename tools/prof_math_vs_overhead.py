#!/usr/bin/env python3
"""用 profile 的 per-op 子计数，量"真数学" vs "非数学开销"（每步）。

为什么不用形状估字节：稀疏注意力/分页 KV/MoE 分片算子的 `Input Shapes` 是**整块缓存或整份权重**，
不是实际流量 ⇒ 形状法会把 1.7 GB 的 indexer cache 整块算进去（实测高估 200×）。

改用实测计数器：
  math = aic_mac_time + aiv_vec_time      （乘加 + 向量算术）
  rest = duration − math                  （标量寻址 + 搬运 + 同步 + 等待）
⇒ `rest` 就是"把算子融成极少数 kernel"能触及的上界（现实要打折）。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_math_vs_overhead.py <profdir> [topN]
"""
from __future__ import annotations
import csv, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

MATH_F = ("aic_mac_time(us)", "aiv_vec_time(us)")
SCAL_F = ("aic_scalar_time(us)", "aiv_scalar_time(us)")
MOVE_F = ("aic_mte1_time(us)", "aic_mte2_time(us)", "aic_fixpipe_time(us)",
          "aiv_mte2_time(us)", "aiv_mte3_time(us)")

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    have = set(rd.fieldnames or ())
    need = [f for f in MATH_F + SCAL_F + MOVE_F if f in have]
    if not need:
        raise SystemExit("该 profile 没有 per-op 子计数列（需要 Level1 导出）")
    for r in rd:
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            st_ = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        def g(f):
            try: return float(r.get(f) or 0)
            except (TypeError, ValueError): return 0.0
        rows.append((st_, du, nm,
                     sum(g(f) for f in MATH_F),
                     sum(g(f) for f in SCAL_F),
                     sum(g(f) for f in MOVE_F),
                     r.get("Accelerator Core") or ""))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

by_core = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])   # dur, math, scal, move, n
tot = [0.0, 0.0, 0.0, 0.0, 0]
for st_, du, nm, math, scal, move, core in sel:
    c = by_core[core]
    c[0] += du; c[1] += math; c[2] += scal; c[3] += move; c[4] += 1
    tot[0] += du; tot[1] += math; tot[2] += scal; tot[3] += move; tot[4] += 1

def ms(x): return x / 1000 / nst
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜算子 {len(sel)/nst:.0f}/步")
print()
print(f"  Σ 算子时长                        = {ms(tot[0]):7.3f} ms  ({100*ms(tot[0])/STEP:5.1f}% 步长)")
print(f"  └ Σ 真数学 (aic_mac + aiv_vec)    = {ms(tot[1]):7.3f} ms  ({100*ms(tot[1])/STEP:5.1f}%)")
print(f"  └ Σ 标量   (aic/aiv_scalar)        = {ms(tot[2]):7.3f} ms  ({100*ms(tot[2])/STEP:5.1f}%)")
print(f"  └ Σ 搬运   (mte1/2/3 + fixpipe)    = {ms(tot[3]):7.3f} ms  ({100*ms(tot[3])/STEP:5.1f}%)")
rest = tot[0] - tot[1]
print(f"  ★ 非数学（时长 − 数学）            = {ms(rest):7.3f} ms  ({100*ms(rest)/STEP:5.1f}%)")
print()
print("按核类型（每步 ms）：")
print(("{:<18}{:>10}{:>10}{:>10}{:>10}{:>10}").format("核", "时长", "数学", "标量", "搬运", "算子/步"))
for core, c in sorted(by_core.items(), key=lambda kv: -kv[1][0])[:8]:
    print(("{:<18}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.0f}").format(
        core[:18], ms(c[0]), ms(c[1]), ms(c[2]), ms(c[3]), c[4] / nst))
print()
# 按算子聚合 top
agg = defaultdict(lambda: [0.0, 0.0, 0])
for st_, du, nm, math, scal, move, core in sel:
    a = agg[nm[:46]]; a[0] += du; a[1] += math; a[2] += 1
print(f"{'算子':<48}{'次/步':>8}{'时长ms':>9}{'数学ms':>9}{'非数学ms':>10}{'数学%':>7}")
for nm, (du, ma, c) in sorted(agg.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:TOPN]:
    print(("{:<48}{:>8.1f}{:>9.3f}{:>9.3f}{:>10.3f}{:>6.0f}%").format(
        nm, c / nst, ms(du), ms(ma), ms(du - ma), 100 * ma / du if du else 0))
