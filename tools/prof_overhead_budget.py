#!/usr/bin/env python3
"""量化"纯 per-operator 开销"：每步 Σ(duration − 数据搬运时间)。

用途：回答"如果把所有小算子的启动/同步开销都消掉，能省多少" —— 这是**深度融合（多合 1）
的理论上界**，也是判断"是否值得立项做 AscendC 算子"的关键数字。

方法：从 profile 的 Input/Output Shapes + Dtypes 估字节数，按给定带宽算净搬运时间，
overhead = duration − data_time（负值截断为 0）。

用法: ANCHOR=HcPre ANCHOR_PER_STEP=86 prof_overhead_budget.py <profdir> [BW_GBps]
"""
from __future__ import annotations
import csv, os, sys, re
from collections import defaultdict

D = sys.argv[1]
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 845.0   # GB/s（MoE 实测达峰档位）
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

DT_BYTES = {
    "DT_BF16": 2, "DT_FLOAT16": 2, "FLOAT16": 2, "DT_FLOAT": 4, "FLOAT": 4,
    "INT8": 1, "DT_INT8": 1, "INT32": 4, "DT_INT32": 4, "INT64": 8, "DT_INT64": 8,
    "BOOL": 1, "DT_BOOL": 1, "DT_UINT8": 1, "UINT8": 1, "DT_FLOAT8_E4M3": 1,
}


def parse_shapes(field: str):
    """把 '6,4,5120;24,20480;3;24;6,4' 解析成 [[6,4,5120],[24,20480],[3],[24],[6,4]]。
    注意：CSV 里字段被双引号包住，逗号在内部分是分隔符。"""
    if not field:
        return []
    s = field.strip().strip('"')
    return [[int(t) for t in p.split(",") if t.strip()] for p in s.split(";") if p.strip()]


def numel(sh):
    n = 1
    for x in sh:
        n *= x
    return n


rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        try:
            ins = parse_shapes(r.get("Input Shapes") or "")
            idt = [t.strip().strip('"') for t in (r.get("Input Data Types") or "").split(";")]
            outs = parse_shapes(r.get("Output Shapes") or "")
            odt = [t.strip().strip('"') for t in (r.get("Output Data Types") or "").split(";")]
        except Exception:
            ins = outs = []; idt = odt = []
        rows.append((st, du, nm, ins, idt, outs, odt))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

tot_dur = tot_data = tot_ovh = 0.0
by_core = defaultdict(lambda: [0.0, 0.0])   # core -> [dur, overhead]
n_unknown = 0
for st, du, nm, ins, idt, outs, odt in sel:
    b = 0
    ok = False
    for sh, dt in list(zip(ins, idt)) + list(zip(outs, odt)):
        nb = DT_BYTES.get(dt)
        if nb is None or not sh:
            continue
        b += numel(sh) * nb
        ok = True
    if not ok:
        n_unknown += 1
        continue
    # b bytes / (BW GB/s * 1e9 B/GB) = seconds；×1e6 => µs  =>  b / (BW*1e3)
    data_us = b / (BW * 1e3)
    ovh = max(0.0, du - data_us)
    tot_dur += du / 1000 / nst
    tot_data += data_us / 1000 / nst
    tot_ovh += ovh / 1000 / nst

print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜带宽假设 {BW:.0f} GB/s")
print(f"可解析形状的算子 {len(sel)-n_unknown}/{len(sel)}（{n_unknown} 个缺 dtype/shape 被跳过）")
print()
print(f"  Σ 算子时长（profile，每步）      = {tot_dur:7.3f} ms   ({100*tot_dur/STEP:.1f}% 步长)")
print(f"  Σ 净数据搬运时间（@{BW:.0f}GB/s）  = {tot_data:7.3f} ms   ({100*tot_data/STEP:.1f}%)")
print(f"  ★ Σ 纯开销（时长 − 搬运）        = {tot_ovh:7.3f} ms   ({100*tot_ovh/STEP:.1f}%)")
print()
print("说明：'纯开销'= 启动 + 标量寻址 + 跨核同步 + 等待。")
print("      它是'把所有算子融成极少数 kernel'的理论上界（现实要打折，因为融合后的 kernel 自身也有开销）。")
print()
for core in ["MIX_AIC", "AI_CORE", "AI_VECTOR_CORE", "MIX_AIV", "COMMUNICATION", "AI_CPU"]:
    if core not in by_core: continue
