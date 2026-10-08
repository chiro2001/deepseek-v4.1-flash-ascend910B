#!/usr/bin/env python3
"""融合点 B 集成补丁：input_layernorm + wq_a.quantize -> 一次 RmsNormDynamicQuantBf16 调用。

默认**关闭**（环境变量 V41_LNORM_FUSE=1 才启用），因此部署补丁本身不会改变行为。
启用后：
  * decoder layer 用 npu_rms_norm_dynamic_quant_bf16 同时产出
      - bf16 归一化结果（供 wkv / compressor / indexer 继续使用）
      - int8 + scale（供 wq_a.matmul 直接使用）
  * attention 的 multistream_preprocess 消费这对 int8/scale，跳过 wq_a.quantize

安全回退：若 wq_a 有 TP 通信或不是 W8A8-dynamic，则仍走原来的 quantize 分支
（此时 hidden_states 已是归一化 bf16，语义不变）。

用法（在容器内）：python3 apply_lnorm_fuse.py [--dry-run]
"""
import pathlib
import shutil
import sys
import time

MODEL = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4/model.py")
ATTN = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py")
MARK_MODEL = "[LNORM-FUSE]"
MARK_ATTN = "[LNORM-FUSE]"


def backup(p: pathlib.Path):
    dst = p.with_suffix(p.suffix + ".bak-lnorm-" + time.strftime("%H%M%S"))
    shutil.copy2(p, dst)
    return dst


# ---------------------------------------------------------------- model.py
MODEL_ANCHOR = """        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)
"""

MODEL_NEW = """        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        # [LNORM-FUSE] 可选融合：一次算子同时给出 bf16 归一化结果与 wq_a 需要的 int8/scale
        # D>=513：实测算子正确区间（D<=512 是上游 rms_norm_dynamic_quant 已有缺陷，见 docs §9.3）
        if _lnorm_fuse_on() and hidden_states.shape[-1] >= 513:
            _w = getattr(self, "_lnorm_w_bf16", None)
            if _w is None:
                _w = self.input_layernorm.weight.data.to(torch.bfloat16)
                self._lnorm_w_bf16 = _w
            _normed, _q_i8, _q_sc = torch.ops._C_ascend.npu_rms_norm_dynamic_quant_bf16(
                hidden_states.contiguous(), _w, epsilon=self.norm_eps
            )
            # 读侧是 forward_context.no_compile_layers[prefix] = AscendDeepseekSparseAttention
            # 实例，也就是 layer.self_attn.dsa_attn；写到 self.self_attn 上读不到（会静默失效）。
            _fuse_tgt = getattr(self.self_attn, "dsa_attn", self.self_attn)
            _fuse_tgt._lnorm_fused_quant = (_q_i8, _q_sc)
            hidden_states = _normed
        else:
            hidden_states = self.input_layernorm(hidden_states)
"""

MODEL_HELPER = '''
_LNORM_FUSE_CACHE = None


def _lnorm_fuse_on():
    # [LNORM-FUSE] 默认关闭；置 V41_LNORM_FUSE=1 启用
    global _LNORM_FUSE_CACHE
    if _LNORM_FUSE_CACHE is None:
        import os as _os_ln

        _LNORM_FUSE_CACHE = _os_ln.environ.get("V41_LNORM_FUSE", "0") == "1"
    return _LNORM_FUSE_CACHE

'''

# ---------------------------------------------------------------- dsa_v41.py
ATTN_ANCHOR = """        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
"""

ATTN_NEW = """        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        _fused_quant = getattr(attn, "_lnorm_fused_quant", None)   # [LNORM-FUSE]
        if _fused_quant is not None and not wq_a._has_communication and getattr(wq_a, "_is_w8a8_dynamic", False):
            q_quant, q_scale = _fused_quant
        else:
            q_quant, q_scale = wq_a.quantize(hidden_states)
        if _fused_quant is not None:
            attn._lnorm_fused_quant = None   # [LNORM-FUSE] 每步消费一次，避免残留
"""


def patch_model(dry: bool):
    s = MODEL.read_text()
    if MARK_MODEL in s:
        print("  model.py: already patched")
        return
    if MODEL_ANCHOR not in s:
        print("  model.py: ANCHOR NOT FOUND"); sys.exit(2)
    # helper 插到 "class DeepseekV2DecoderLayer" 之前
    cls_anchor = "class DeepseekV2DecoderLayer(nn.Module):"
    if cls_anchor not in s:
        print("  model.py: class anchor missing"); sys.exit(2)
    s = s.replace(cls_anchor, MODEL_HELPER + "\n" + cls_anchor, 1)
    s = s.replace(MODEL_ANCHOR, MODEL_NEW, 1)
    if dry:
        print("  model.py: would patch (dry-run)")
        return
    b = backup(MODEL)
    MODEL.write_text(s)
    print("  model.py: patched (backup %s)" % b.name)


def patch_attn(dry: bool):
    s = ATTN.read_text()
    if MARK_ATTN in s:
        print("  dsa_v41.py: already patched")
        return
    if ATTN_ANCHOR not in s:
        print("  dsa_v41.py: ANCHOR NOT FOUND"); sys.exit(2)
    s = s.replace(ATTN_ANCHOR, ATTN_NEW, 1)
    if dry:
        print("  dsa_v41.py: would patch (dry-run)")
        return
    b = backup(ATTN)
    ATTN.write_text(s)
    print("  dsa_v41.py: patched (backup %s)" % b.name)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    print("== 融合点 B 集成补丁 (dry=%s) ==" % dry)
    patch_model(dry)
    patch_attn(dry)
    print("完成。启用方式：V41_LNORM_FUSE=1 启动服务")
