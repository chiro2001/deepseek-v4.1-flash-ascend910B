#!/usr/bin/env python3
"""让前向路径也能取到本 ubatch 的 metadata。

本仓的图上 ubatching 是**无线程**实现（顺序提交 + 换流），所以
`dbo_current_ubatch_id()` 恒为 0，不能直接用它索引。
改为：wrapper 在每个 per-ubatch forward context 上写 `ubatch_id`，
`_get_layer_metadata` 遇到 list 时按它索引。
"""
import ast, sys
from pathlib import Path

# ① wrapper：在 per-ubatch forward context 上写 ubatch_id
wp = Path("/home/l00886679/dcpw/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = wp.read_text()
a = """            forward_contexts.append(fc)"""
b = """            setattr(fc, "ubatch_id", i)   # [DBO] 无线程实现：显式标注本 ubatch
            forward_contexts.append(fc)"""
if "_get_layer_metadata" not in s and "ubatch_id" not in s:
    if a in s:
        s = s.replace(a, b, 1); ast.parse(s); wp.write_text(s)
        print("① wrapper: 已写 forward_context.ubatch_id")
    else:
        print("① 锚点未命中")

# ② dsa_v41._get_layer_metadata：list 时按 ubatch_id 索引
dp = Path("/home/l00886679/dcpw/vllm_ascend/attention/dsa_v41.py")
t = dp.read_text()
a2 = """    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:"""
b2 = """    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        # [DBO] ubatching 下 attn_metadata 是 per-ubatch 的 list，
        # 按 forward context 上的 ubatch_id 取本 ubatch 的那份。
        if isinstance(metadata, list):
            from vllm.forward_context import get_forward_context
            _ub = getattr(get_forward_context(), "ubatch_id", 0) or 0
            metadata = metadata[min(int(_ub), len(metadata) - 1)]
        try:"""
if a2 in t and "[DBO] ubatching 下 attn_metadata" not in t:
    t = t.replace(a2, b2, 1); ast.parse(t); dp.write_text(t)
    print("② dsa_v41._get_layer_metadata: 已加 list 分支")
else:
    print("② 锚点未命中或已改")
