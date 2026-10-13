#!/usr/bin/env python3
"""核对官方 A3 支持算子在我们 vllm-ascend 里的使用情况（遍历文件内容）。"""
import pathlib
import re

ROOT = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")
OPS = {
    "npu_hc_pre_v2": "HcPre 融合",
    "npu_hc_post": "HcPost 融合",
    "mhc_post": "HcPost(库版)",
    "mhc_pre_sinkhorn": "HcPre(库版)",
    "npu_compressor": "Compressor",
    "inplace_partial_rotary_mul": "RoPE",
    "npu_rms_norm_dynamic_quant": "RmsNorm+DynamicQuant 融合",
    "npu_swiglu_clip_quant": "SwiGLU 融合",
    "npu_moe_gating_top_k": "MoE gating 融合",
    "quant_lightning_indexer": "Indexer 融合",
    "sparse_flash_mla": "SparseFlashMla",
    "npu_sparse_attn_sharedkv": "SparseAttnSharedkv",
    "npu_scatter_nd_update_asc": "scatter KV 写",
    "npu_gather_selection_kv_cache": "KV gather",
    "mega_moe": "MegaMoE",
    "lightning_indexer": "lightning indexer",
    "flash_attn": "flash attention",
    "mixed_quant_sparse_flash_mla": "混合量化 SFA",
}

texts = []
for p in ROOT.rglob("*.py"):
    if ".pyc" in str(p):
        continue
    try:
        texts.append((str(p), p.read_text(errors="ignore")))
    except Exception:
        pass
print("扫了 %d 个 py 文件\n" % len(texts))

print("%-32s %-30s %s" % ("算子", "用途", "使用点"))
for op, desc in OPS.items():
    hits = [(f, i + 1) for f, t in texts for i, ln in enumerate(t.splitlines()) if op in ln]
    if hits:
        locs = ", ".join("%s:%d" % (f.split("vllm_ascend/")[-1], n) for f, n in hits[:2])
        print("%-32s %-30s 已用 (%d 处) %s" % (op, desc, len(hits), locs))
    else:
        print("%-32s %-30s -" % (op, desc))
