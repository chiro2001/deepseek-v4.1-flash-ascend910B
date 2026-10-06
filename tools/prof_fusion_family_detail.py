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



import csv, sys, statistics as st
from collections import defaultdict
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
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""; ty = r.get("OP Type") or ""
        if "allgatherAicpu" in nm or "allgatherAicpu" in ty:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            stt = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        rows.append((stt, du, nm))
marks.sort()
LO, HI = marks[2], marks[-3]
sel = [x for x in rows if LO <= x[0] < HI]
nst = len([m for m in marks if LO <= m < HI])
STEP_P = (HI - LO) / 1000 / nst; K = STEP_P / STEP_REAL_MS

fam = defaultdict(lambda: defaultdict(list))
cnt = defaultdict(int)
for stt, du, nm in sel:
    f = fam_of(nm)
    if not f: continue
    fam[f][nm].append(du); cnt[f] += 1

print(f"步长 profile {STEP_P:.2f} ms | K={K:.3f} | 每步算子 {len(sel)/nst:.0f}")
tot = 0.0
for label, _ in FAMILIES:
    if label not in fam: continue
    all_d = [d for v in fam[label].values() for d in v]
    med = st.median(all_d)
    n_per = cnt[label] / nst
    saved_p = (n_per - n_per / 4.0) * med / 1000
    tot += saved_p / K
    print(f"\n=== {label}：{n_per:.1f} 个/步，中位 {med:.2f} us/算子，自身 {sum(all_d)/1000/nst:.3f} ms")
    print(f"    4:1 融合 ⇒ 省 {saved_p:.3f} ms(profile) = {saved_p/K:.3f} ms(真实)")
    for nm, ds in sorted(fam[label].items(), key=lambda kv: -sum(kv[1]))[:6]:
        print(f"      {nm[:54]:<54} {len(ds)/nst:>7.1f}/步  中位{st.median(ds):>7.2f}us  合计{sum(ds)/1000/nst:>7.3f}ms")
print(f"\n>>> 全部族 4:1 融合合计（真实口径）：{tot:.3f} ms/步 = {100*tot/STEP_REAL_MS:.1f}% 步长")
