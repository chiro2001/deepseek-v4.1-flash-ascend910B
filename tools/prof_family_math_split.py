#!/usr/bin/env python3
"""把融合族按"数学 / 标量 / 搬运 / 未归因"拆开（用 per-op 子计数）。

判据：**若某族的 MTE（搬运）远小于 标量+未归因，则多合 1 融合能真正省时间**；
若 MTE 主导，则融合只能省 launch，收益有限。
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

MATH_F = ("aic_mac_time(us)", "aiv_vec_time(us)")
SCAL_F = ("aic_scalar_time(us)", "aiv_scalar_time(us)")
MOVE_F = ("aic_mte1_time(us)", "aic_mte2_time(us)", "aic_fixpipe_time(us)",
          "aiv_mte2_time(us)", "aiv_mte3_time(us)")

rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        def g(f):
            try: return float(r.get(f) or 0)
            except (TypeError, ValueError): return 0.0
        rows.append((a, du, nm, sum(g(f) for f in MATH_F), sum(g(f) for f in SCAL_F),
                     sum(g(f) for f in MOVE_F)))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP = (HI - LO) / 1000 / nst

agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
for a, du, nm, ma, sc, mv in sel:
    f = fam_of(nm)
    if not f: continue
    e = agg[f]; e[0] += du; e[1] += ma; e[2] += sc; e[3] += mv; e[4] += 1

def ms(x): return x / 1000 / nst
print(f"锚={ANCHOR}/{PER}｜步数 {nst}｜步长 {STEP:.3f} ms")
print()
print(("{:<20}{:>8}{:>9}{:>8}{:>8}{:>8}{:>9}{:>10}").format(
    "融合族","次/步","时长ms","数学ms","标量ms","搬运ms","未归因ms","搬运占比"))
T = [0.0]*5
for lbl, _ in FAMILIES:
    if lbl not in agg: continue
    du, ma, sc, mv, c = agg[lbl]
    un = max(0.0, du - ma - sc - mv)
    for i, v in enumerate((du, ma, sc, mv, c)): T[i] += v
    print(("{:<20}{:>8.1f}{:>9.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>9.3f}{:>9.0f}%").format(
        lbl, c/nst, ms(du), ms(ma), ms(sc), ms(mv), ms(un), 100*mv/du if du else 0))
du, ma, sc, mv, c = T
un = max(0.0, du - ma - sc - mv)
print("-"*72)
print(("{:<20}{:>8.1f}{:>9.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>9.3f}{:>9.0f}%").format(
    "合计", c/nst, ms(du), ms(ma), ms(sc), ms(mv), ms(un), 100*mv/du if du else 0))
print()
print(f"*** 若这些族被多合1融合，可省的上界 ~= 标量 + 未归因 = {ms(sc)+ms(un):.3f} ms "
      f"({100*(ms(sc)+ms(un))/STEP:.1f}% 步长)**（搬运与数学仍要花）")
