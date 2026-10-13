#!/usr/bin/env python3
"""融合族分析（支持自定义步锚点）。默认锚 allgatherAicpuKernel；可用 ANCHOR=名字 ANCHOR_PER_STEP=N。

为什么需要：纯 conc=1 的新 profile 里 allgatherAicpuKernel 只有 5 次（<6），
而 HcPre 是 86 次/步（159874 / 86 = 1859 步，整数）⇒ 换锚点。
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

STEP_REAL_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 24.59
D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "allgatherAicpuKernel")
PER_STEP = int(os.environ.get("ANCHOR_PER_STEP", "1"))

FAMILIES = [
    ("F1 位置/槽位链", ["FloorMod","FloorDiv","Remainder","GreaterEqual","GeTensor","GeScalar",
                        "LtTensor","Less","SelectV2","SWhere","ClipByValue","Abs","Neg",
                        "LogicalOr","BitwiseOr","Arange","BroadcastTo","Range","Where","Equal","NotEqual"]),
    ("F2 dtype/搬运链", ["InplaceCopy_Cast","InplaceCopy_ViewCopy","InplaceCopy_TensorMove",
                         "InplaceCopy_Slice","InplaceFillScalar","CastAiCore","ViewCopy",
                         "TensorMove","SliceAiCore","Contiguous","ZerosLike","FillAiCore"]),
    ("F3 norm+quant 融合", ["RmsNorm","DynamicQuant","DequantSwigluQuant","SwigluQuant","Quantize","RmsNormCast"]),
    ("F4 MoE 路由链", ["MoeGatingTopK","MoeInitRouting","MoeTokenUnpermute","Topk","MoeDistribute",
                       "MoeMask","ExpertMask","MoeIndex"]),
    ("F5 spec-decode 后处理", ["IndexCheck","ArgMax","MaskedFill","ReduceSum","GatherElements",
                               "IndexFill","RejectionSample","rejection_greedy","TopKTopP","Sampling","Multinomial"]),
    ("F6 indexer 小 kernel", ["_prepare_indexer_indices","_quantize_indexer_query","LightningIndexer","Indexer"]),
    ("F7 RoPE/取表链", ["InplacePartialRotaryMul","GatherV3","GatherV2","IndexSelect","Embedding",
                        "RotaryMul","RoPE","Cos","Sin"]),
    ("F8 归约/逐元素杂项", ["AddAiCore","MulAiCore","SubAiCore","DivAiCore","Exp","Sigmoid","Softmax",
                            "Log","Sqrt","Pow","Clamp","Minimum","Maximum","Sum","Mean","InplaceAdd","InplaceMul"]),
]
NOT_FUSABLE = ("Matmul","MatMul","GroupedMatmul","allreduce","allgather","AllReduce","AllGather","Hccl",
               "hcom","AivKernel","Metadata","AicpuKernel","SparseFlashMla","SparseAttn","HcPre","HcPost",
               "ScatterNdUpdate","ScatterElements","MoeDistributeDispatch","MoeDistributeCombine")

def fam_of(name):
    for k in NOT_FUSABLE:
        if k in name: return None
    for label, keys in FAMILIES:
        for k in keys:
            if k in name: return label
    return None

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
        rows.append((st, st + du, du, nm))
marks.sort()
if PER_STEP > 1:
    marks = marks[::PER_STEP]
if len(marks) < 6:
    raise SystemExit(f"锚点不足：{ANCHOR} 只有 {len(marks)*PER_STEP} 次（切分后 {len(marks)}）")
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

groups = defaultdict(list); other = []
for st, en, du, nm in sel:
    f = fam_of(nm)
    (groups[f] if f else other).append((st, en, du))
oth_iv = merged([(s, e) for s, e, d in other])
print(f"锚={ANCHOR}/{PER_STEP}｜步长 profile {STEP_P:.2f} ms｜K={K:.3f}｜步数 {nst}｜算子 {len(sel)/nst:.0f}/步")
print(f"{'融合族':<22}{'次数/步':>9}{'自身ms':>9}{'并集ms':>9}{'暴露ms':>9}{'真实ms':>9}{'占步长':>8}")
tot = 0.0
for label, _ in FAMILIES:
    g = groups.get(label, [])
    if not g: continue
    iv = merged([(s, e) for s, e, d in g])
    uni = total(iv) / 1000 / nst
    exp = (total(iv) - overlap(iv, oth_iv)) / 1000 / nst
    real = exp / K; tot += real
    own = sum(d for s, e, d in g) / 1000 / nst
    print(f"{label:<22}{len(g)/nst:>9.1f}{own:>9.3f}{uni:>9.3f}{exp:>9.3f}{real:>9.3f}{100*real/STEP_REAL_MS:>7.1f}%")
print(f"\n可融合族合计：真实暴露 {tot:.3f} ms/步 = {100*tot/STEP_REAL_MS:.1f}% 步长")
print(f"（不可融合类 {len(other)/nst:.0f} 个/步，自身 {sum(d for s,e,d in other)/1000/nst:.3f} ms）")
