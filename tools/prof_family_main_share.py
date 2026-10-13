#!/usr/bin/env python3
"""各融合族里有多少"在主流上"（= 在关键路径上，融合才有效）。

判据：主流 = busy 最大的流。族里落在主流上的算子数与时长，才是"融合能省"的分母。
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

FAMILIES = [
    ("F1 位置/槽位链", ["FloorMod","FloorDiv","Remainder","GreaterEqual","GeTensor","GeScalar",
                        "LtTensor","Less","SelectV2","SWhere","ClipByValue","Abs","Neg",
                        "LogicalOr","BitwiseOr","Arange","BroadcastTo","Range","Where","Equal","NotEqual"]),
    ("F2 dtype/搬运链", ["InplaceCopy_Cast","InplaceCopy_ViewCopy","InplaceCopy_TensorMove",
                         "InplaceCopy_Slice","InplaceFillScalar","CastAiCore","ViewCopy",
                         "TensorMove","SliceAiCore","Contiguous","ZerosLike","FillAiCore"]),
    ("F3 norm+quant", ["RmsNorm","DynamicQuant","DequantSwigluQuant","SwigluQuant","Quantize","RmsNormCast"]),
    ("F4 MoE 路由链", ["MoeGatingTopK","MoeInitRouting","MoeTokenUnpermute","Topk","MoeMask","ExpertMask"]),
    ("F6 indexer 小 kernel", ["_prepare_indexer_indices","_quantize_indexer_query","LightningIndexer","Indexer"]),
    ("F7 RoPE/取表链", ["InplacePartialRotaryMul","GatherV3","GatherV2","IndexSelect","Embedding","RotaryMul","Cos","Sin"]),
    ("F8 归约/逐元素杂项", ["AddAiCore","MulAiCore","SubAiCore","DivAiCore","Exp","Sigmoid","Softmax",
                            "Log","Sqrt","Pow","Clamp","Minimum","Maximum","Sum","Mean","InplaceAdd","InplaceMul"]),
]
NOT_FUSABLE = ("Matmul","MatMul","GroupedMatmul","allreduce","allgather","AllReduce","AllGather","Hccl",
               "hcom","AivKernel","Metadata","AicpuKernel","SparseFlashMla","SparseAttn","HcPre","HcPost",
               "ScatterNdUpdate","ScatterElements")

def fam_of(n):
    for k in NOT_FUSABLE:
        if k in n: return None
    for lbl, keys in FAMILIES:
        for k in keys:
            if k in n: return lbl
    return None

marks, rows = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        rows.append((a, du, nm, str(r.get("Stream ID") or "")))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

busy = defaultdict(float)
for a, du, nm, s in sel: busy[s] += du
MAIN = max(busy.items(), key=lambda kv: kv[1])[0]

agg = defaultdict(lambda: [0, 0.0, 0, 0.0])   # fam -> [n_all, t_all, n_main, t_main]
for a, du, nm, s in sel:
    f = fam_of(nm)
    if not f: continue
    e = agg[f]
    e[0] += 1; e[1] += du
    if s == MAIN:
        e[2] += 1; e[3] += du

def ms(x): return x / 1000 / nst
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms｜主流(自动) s{MAIN}")
print()
print(("{:<20}{:>9}{:>9}{:>9}{:>9}{:>10}").format("融合族","总次数","主流次数","总ms","主流ms","主流占比"))
T = [0]*4
for lbl, _ in FAMILIES:
    if lbl not in agg: continue
    n_all, t_all, n_m, t_m = agg[lbl]
    for i, v in enumerate((n_all, t_all, n_m, t_m)): T[i] += v
    print(("{:<20}{:>9.1f}{:>9.1f}{:>9.3f}{:>9.3f}{:>9.0f}%").format(
        lbl, n_all/nst, n_m/nst, ms(t_all), ms(t_m), 100*n_m/max(1,n_all)))
print("-"*66)
print(("{:<20}{:>9.1f}{:>9.1f}{:>9.3f}{:>9.3f}{:>9.0f}%").format(
    "合计", T[0]/nst, T[2]/nst, ms(T[1]), ms(T[3]), 100*T[2]/max(1,T[0])))

# 预测：4:1 / 8:1 融合能省多少
print()
print("基于实测每算子边际开销 2.0~3.1 us，且只算主流上的算子：")
for ovh in (2.0, 3.0):
    for k in (4, 8):
        saved = (T[2]/nst - T[2]/nst/k) * ovh / 1000
        print(f"  开销 {ovh:.1f} us/算子, {k}:1 融合 -> 省 {saved:.3f} ms = {100*saved/STEP:.1f}% 步长")
