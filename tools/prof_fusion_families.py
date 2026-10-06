#!/usr/bin/env python3
"""按**融合族**归类小算子，给出每族的规模与可回收量（纯标准库）。

分类依据：同一族 = 同一条数据流上连续/可合并的 elementwise 运算，
通常可融为 1 个 kernel。判据用**聚合暴露度**（union − 与其它算子的重叠），
不是自身时长之和（后者会把被覆盖的部分重复计算）。

用法: prof_fusion_families.py <ASCEND_PROFILER_OUTPUT> [step_real_ms]
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict

STEP_REAL_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 24.59
D = sys.argv[1]

# ---- 族定义：名字里含任一关键词即归入该族（先匹配先生效）----
FAMILIES = [
    ("F1 位置/槽位链", ["FloorMod", "FloorDiv", "Remainder", "GreaterEqual", "GeTensor",
                        "GeScalar", "LtTensor", "Less", "SelectV2", "SWhere", "ClipByValue",
                        "Abs", "Neg", "LogicalOr", "BitwiseOr", "Arange", "BroadcastTo",
                        "Range", "Where", "Equal", "NotEqual"]),
    ("F2 dtype/搬运链", ["InplaceCopy_Cast", "InplaceCopy_ViewCopy", "InplaceCopy_TensorMove",
                         "InplaceCopy_Slice", "InplaceFillScalar", "CastAiCore", "ViewCopy",
                         "TensorMove", "SliceAiCore", "Contiguous", "ZerosLike", "FillAiCore"]),
    ("F3 norm+quant 融合", ["RmsNorm", "DynamicQuant", "DequantSwigluQuant", "SwigluQuant",
                            "Quantize", "RmsNormCast"]),
    ("F4 MoE 路由链", ["MoeGatingTopK", "MoeInitRouting", "MoeTokenUnpermute", "Topk",
                       "MoeDistribute", "MoeMask", "ExpertMask", "MoeIndex"]),
    ("F5 spec-decode 后处理", ["IndexCheck", "ArgMax", "MaskedFill", "ReduceSum",
                               "GatherElements", "IndexFill", "RejectionSample",
                               "rejection_greedy", "TopKTopP", "Sampling", "Multinomial"]),
    ("F6 indexer 小 kernel", ["_prepare_indexer_indices", "_quantize_indexer_query",
                              "LightningIndexer", "Indexer"]),
    ("F7 RoPE/取表链", ["InplacePartialRotaryMul", "GatherV3", "GatherV2", "IndexSelect",
                        "Embedding", "RotaryMul", "RoPE", "Cos", "Sin"]),
    ("F8 归约/逐元素杂项", ["AddAiCore", "MulAiCore", "SubAiCore", "DivAiCore", "Exp",
                            "Sigmoid", "Softmax", "Log", "Sqrt", "Pow", "Clamp", "Minimum",
                            "Maximum", "Sum", "Mean", "InplaceAdd", "InplaceMul"]),
]
NOT_FUSABLE = ("Matmul", "MatMul", "GroupedMatmul", "allreduce", "allgather", "AllReduce",
               "AllGather", "Hccl", "hcom", "AivKernel", "Metadata", "AicpuKernel",
               "SparseFlashMla", "SparseAttn", "HcPre", "HcPost", "ScatterNdUpdate",
               "ScatterElements", "MoeDistributeDispatch", "MoeDistributeCombine")


def fam_of(name: str):
    for key in NOT_FUSABLE:
        if key in name:
            return None
    for label, keys in FAMILIES:
        for k in keys:
            if k in name:
                return label
    return None


rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
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

groups = defaultdict(list)
other = []
for st, en, du, nm in sel:
    f = fam_of(nm)
    (groups[f] if f else other).append((st, en, du))

oth_iv = merged([(s, e) for s, e, d in other])
print(f"步长 profile {STEP_P:.2f} ms | K={K:.3f} | 算子 {len(sel)/nst:.0f}/步 | 真实口径 {STEP_REAL_MS} ms")
print()
print(f"{'融合族':<22}{'次数/步':>9}{'自身ms':>9}{'并集ms':>9}{'暴露ms':>9}{'真实ms':>9}{'占步长':>8}{'可融度':>7}"
      f"{'4:1省ms':>9}{'真实ms':>9}")
tot_real = 0.0
for label, _ in FAMILIES:
    g = groups.get(label, [])
    if not g: continue
    iv = merged([(s, e) for s, e, d in g])
    uni = total(iv) / 1000 / nst
    # 暴露 = union − 与"非本族"的重叠
    exp = (total(iv) - overlap(iv, oth_iv)) / 1000 / nst
    real = exp / K
    tot_real += real
    own = sum(d for s, e, d in g) / 1000 / nst
    # 可融度 = 并集/自身（越接近1说明内部越连续，融合越划算）
    ratio = total(iv) / max(1e-9, sum(d for s, e, d in g))
    # ★ 可回收量：融合把 n 次调用降到 n/target 次，省下 (n − n/target) × 每次固定开销
    #   固定开销实测 ≈ 3.5 µs/算子（TAIL-OP-COUNT 的 ~4µs 地板，取保守值）
    n_per = len(g) / nst
    for tgt in (4.0,):
        saved_us = (n_per - n_per / tgt) * 3.5
    print(f"{label:<22}{n_per:>9.1f}{own:>9.3f}{uni:>9.3f}{exp:>9.3f}{real:>9.3f}{100*real/STEP_REAL_MS:>7.1f}%"
          f"{ratio:>7.2f}{saved_us/1000:>9.3f}{saved_us/1000/K:>9.3f}")
print()
print(f"可融合族合计：真实暴露 {tot_real:.3f} ms/步 = {100*tot_real/STEP_REAL_MS:.1f}% 步长")
print(f"（对比：不可融合类 {len(other)/nst:.0f} 个/步，自身 {sum(d for s,e,d in other)/1000/nst:.3f} ms）")
