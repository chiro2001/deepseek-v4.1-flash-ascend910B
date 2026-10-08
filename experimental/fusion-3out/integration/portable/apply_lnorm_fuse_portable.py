#!/usr/bin/env python3
"""LNORM-FUSE 补丁（锚点驱动，兼容 A3 TP8 / A3 CED-PD）。

★★ 2026-10-08 重要更正 ★★
第一版把补丁打在 `models/deepseek_v4/model.py` 的 `DeepseekV2DecoderLayer.forward`。
**那是错的** —— 实测 profiler 里 `aclnnRmsNormDynamicQuantBf16` 出现 0 次，融合从未生效：

    models/deepseek_v41/model.py:890  class DeepseekV41Model(DeepseekV4Model):
    models/deepseek_v41/model.py:893      decoder_layer_cls = DeepseekV41DecoderLayer
    models/deepseek_v41/model.py:768  class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):
    models/deepseek_v41/model.py:872      x = self.input_layernorm(x)   ← 真正跑的是这里

`DeepseekV41DecoderLayer` **override 了 forward**（签名也不同：多一个 `pre_mix`），
所以 v4 那份的 forward 根本不会被调用。本版改打 v41 的那份。

两处改动：

  models/deepseek_v41/model.py
    ① `class DeepseekV41DecoderLayer` 前插 `_lnorm_fuse_on()` 助手
    ② 把 **forward 里**（紧跟 `x = self.self_attn(positions, x, llama_4_scaling)` 的那处）
       `x = self.input_layernorm(x)` 换成融合分支
       ⚠️ 同文件第 851 行（`write_global_source_from_encoder`，CED 层20 源投影）
          也被 `write_global_source_only` 消费、不走 multistream_preprocess ⇒ **不能动**

  attention/dsa_v41.py
    ③ 把 `q_quant, q_scale = wq_a.quantize(hidden_states)` 换成消费融合结果

用法（容器内）：
    python3 apply_lnorm_fuse_portable.py            # 应用（幂等）
    python3 apply_lnorm_fuse_portable.py --check    # 只查锚点
    python3 apply_lnorm_fuse_portable.py --revert   # 回滚
"""
import pathlib
import shutil
import sys
import time

ASC = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")
MODEL = ASC / "models/deepseek_v41/model.py"
DSA = ASC / "attention/dsa_v41.py"

MODEL_HELPER = '''

_LNORM_FUSE_CACHE = None


def _lnorm_fuse_on():
    # [LNORM-FUSE] 默认关闭；置 V41_LNORM_FUSE=1 启用
    global _LNORM_FUSE_CACHE
    if _LNORM_FUSE_CACHE is None:
        import os as _os_ln

        _LNORM_FUSE_CACHE = _os_ln.environ.get("V41_LNORM_FUSE", "0") == "1"
    return _LNORM_FUSE_CACHE


def _lnorm_fuse_register_fake():
    """给自研三输出算子注册 fake/meta 实现 —— torch.compile tracing 需要。

    不注册会直接炸：`RuntimeError: Operator does not support running with fake tensors`。
    这就是 A3 第一次打对类之后 worker 起不来的根因（v4 那份因为打错类、从未被 trace 到，
    所以一直没暴露）。先例见 quant_lightning_indexer_v2 的 register_fake。
    """
    try:
        import torch.library

        @torch.library.register_fake("_C_ascend::npu_rms_norm_dynamic_quant_bf16")
        def _fake(x, gamma, smooth_scale=None, beta=None, epsilon=1e-6):
            return (
                x.new_empty(x.shape, dtype=torch.bfloat16, device="meta"),
                x.new_empty(x.shape, dtype=torch.int8, device="meta"),
                x.new_empty(x.shape[:-1], dtype=torch.float32, device="meta"),
            )

        return True
    except Exception as _e:  # 已注册/不支持都不该阻断 import
        import sys as _sys

        print("[LNORM-FUSE] register_fake skipped: %r" % (_e,), file=_sys.stderr)
        return False


_LNORM_FAKE_OK = _lnorm_fuse_register_fake()


'''

# ★ 用两行做锚点消歧：851 行后面跟的是 write_global_source_only
MODEL_OLD = """        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
"""
MODEL_NEW = """        # [LNORM-FUSE] 一次算子同时给出 bf16 归一化结果与 wq_a 需要的 int8/scale
        # D>=513：实测算子正确区间（D<=512 是上游 rms_norm_dynamic_quant 已有缺陷）
        if _lnorm_fuse_on() and x.shape[-1] >= 513:
            # ★ 直接传 nn.Parameter，**不要**用 .data / .to() / 缓存到普通属性：
            #   dynamo 会把 Parameter 当模块状态并 fake 化；而普通属性里的张量被当"常量"、
            #   **不会**被 fake 化 ⇒ 与 fake 张量混用 ⇒
            #   `fake_tensor.py merge_devices → RuntimeError: Dynamo failed to run FX node
            #    with fake tensors`（实测就是这个把 npu_hc_post 报出来的）。
            _normed, _q_i8, _q_sc = torch.ops._C_ascend.npu_rms_norm_dynamic_quant_bf16(
                x.contiguous(), self.input_layernorm.weight, epsilon=self.norm_eps
            )
            # 读侧 = forward_context.no_compile_layers[prefix]，绑定的是
            # AscendDeepseekSparseAttention 实例（= layer.self_attn.dsa_attn）
            _fuse_tgt = getattr(self.self_attn, "dsa_attn", self.self_attn)
            _fuse_tgt._lnorm_fused_quant = (_q_i8, _q_sc)
            x = _normed
        else:
            x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
"""

DSA_OLD = "        q_quant, q_scale = wq_a.quantize(hidden_states)\n"
DSA_NEW = """        _fused_quant = getattr(attn, "_lnorm_fused_quant", None)   # [LNORM-FUSE]
        if _fused_quant is not None and not wq_a._has_communication and getattr(wq_a, "_is_w8a8_dynamic", False):
            q_quant, q_scale = _fused_quant
        else:
            q_quant, q_scale = wq_a.quantize(hidden_states)
        if _fused_quant is not None:
            attn._lnorm_fused_quant = None   # [LNORM-FUSE] 每步消费一次，避免残留
"""


def backup(p):
    dst = pathlib.Path(str(p) + ".bak-portable-" + time.strftime("%H%M%S"))
    shutil.copy2(p, dst)
    return dst


def revert():
    n = 0
    for p in (MODEL, DSA):
        cands = sorted(p.parent.glob(p.name + ".bak-portable-*"))
        if not cands:
            print("  [skip] %s 无备份" % p.name)
            continue
        shutil.copy2(cands[-1], p)
        print("  [ok] %s <- %s" % (p.name, cands[-1].name))
        n += 1
    return 0 if n else 1


def main():
    check = "--check" in sys.argv
    if "--revert" in sys.argv:
        print("== 回滚 ==")
        return revert()

    rc = 0
    for p, kind in ((MODEL, "model"), (DSA, "dsa")):
        print("-- %s: %s" % (kind, p))
        if not p.is_file():
            print("   [FAIL] 文件不存在")
            rc = 1
            continue
        s = p.read_text()

        if kind == "model":
            if "_lnorm_fuse_on" in s:
                print("   [skip] 已打过")
                continue
            cls_anchor = "class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):"
            if cls_anchor not in s:
                print("   [FAIL] 找不到 class 锚点")
                rc = 1
                continue
            if MODEL_OLD not in s:
                print("   [FAIL] 找不到 forward 里的 input_layernorm 锚点")
                rc = 1
                continue
            if check:
                print("   [ok] 两个锚点都在")
                continue
            b = backup(p)
            s = s.replace(cls_anchor, MODEL_HELPER.lstrip("\n") + "\n" + cls_anchor, 1)
            s = s.replace(MODEL_OLD, MODEL_NEW, 1)
            p.write_text(s)
            print("   [ok] 已打（备份 %s）" % b.name)
        else:
            if "_lnorm_fused_quant" in s:
                print("   [skip] 已打过")
                continue
            if DSA_OLD not in s:
                print("   [FAIL] 找不到 wq_a.quantize 锚点")
                rc = 1
                continue
            if check:
                print("   [ok] 锚点在")
                continue
            b = backup(p)
            p.write_text(s.replace(DSA_OLD, DSA_NEW, 1))
            print("   [ok] 已打（备份 %s）" % b.name)

    if check and rc == 0:
        print("\n锚点检查通过")
    elif rc == 0:
        print("\n完成。启用：V41_LNORM_FUSE=1 + ASCEND_CUSTOM_OPP_PATH")
    return rc


if __name__ == "__main__":
    sys.exit(main())
