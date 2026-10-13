#!/usr/bin/env python3
"""把 LNORM-FUSE 打成**任意 DeepSeek-V4.1 形态**的补丁（锚点驱动，非整文件覆盖）。

为什么需要它：`apply_lnorm_fuse.py` 是整文件覆盖式，只适用于「A3 TP8 单实例」那一份
`dsa_v41.py`。而 **CED-PD 形态的 `dsa_v41.py` 是定制版**（含 `[CED-SWA-CLIP]` 等 168 行
CED 逻辑）——整文件覆盖会把 CED 的修复抹掉。

本脚本只做两处**锚点替换**，与变体无关：

  model.py   （`models/deepseek_v4/model.py`，被 `models/deepseek_v41/model.py` import 共用）
    ① 在 `class DeepseekV2DecoderLayer` 前插 `_lnorm_fuse_on()` 助手
    ② 把 `hidden_states = self.input_layernorm(hidden_states)` 换成融合分支

  dsa_v41.py （`attention/dsa_v41.py`）
    ③ 把 `q_quant, q_scale = wq_a.quantize(hidden_states)` 换成消费融合结果的分支

两处锚点在 **A3 TP8 / A3 CED-PD / v4 / v41 四种变体里都存在**（已核对）。

用法（容器内）：
    python3 apply_lnorm_fuse_portable.py            # 应用（幂等）
    python3 apply_lnorm_fuse_portable.py --check    # 只检查锚点，不改
    python3 apply_lnorm_fuse_portable.py --revert   # 用最近的 .bak 回滚
"""
import pathlib
import shutil
import sys
import time

ASC = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")
TARGETS = {
    # 逻辑名: (文件, 锚点集合)
    "model": (ASC / "models/deepseek_v4/model.py", "model"),
    "dsa_v41": (ASC / "attention/dsa_v41.py", "dsa"),
}

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

MODEL_OLD = "        hidden_states = self.input_layernorm(hidden_states)\n"
MODEL_NEW = """        # [LNORM-FUSE] 可选融合：一次算子同时给出 bf16 归一化结果与 wq_a 需要的 int8/scale
        # D>=513：实测算子正确区间（D<=512 是上游 rms_norm_dynamic_quant 已有缺陷，见 docs 9.3）
        if _lnorm_fuse_on() and hidden_states.shape[-1] >= 513:
            _w = getattr(self, "_lnorm_w_bf16", None)
            if _w is None:
                _w = self.input_layernorm.weight.data.to(torch.bfloat16)
                self._lnorm_w_bf16 = _w
            _normed, _q_i8, _q_sc = torch.ops._C_ascend.npu_rms_norm_dynamic_quant_bf16(
                hidden_states.contiguous(), _w, epsilon=self.norm_eps
            )
            # 读侧 = forward_context.no_compile_layers[prefix]（AscendDeepseekSparseAttention），
            # 即 layer.self_attn.dsa_attn；写到 self.self_attn 上读不到
            _fuse_tgt = getattr(self.self_attn, "dsa_attn", self.self_attn)
            _fuse_tgt._lnorm_fused_quant = (_q_i8, _q_sc)
            hidden_states = _normed
        else:
            hidden_states = self.input_layernorm(hidden_states)
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


def backup(p: pathlib.Path) -> pathlib.Path:
    dst = pathlib.Path(str(p) + ".bak-portable-" + time.strftime("%H%M%S"))
    shutil.copy2(p, dst)
    return dst


def revert() -> int:
    n = 0
    for _, (p, _) in TARGETS.items():
        cands = sorted(p.parent.glob(p.name + ".bak-portable-*"))
        if not cands:
            print("  [skip] %s 没有 .bak-portable-*" % p.name)
            continue
        shutil.copy2(cands[-1], p)
        print("  [ok] %s <- %s" % (p.name, cands[-1].name))
        n += 1
    return 0 if n else 1


def main() -> int:
    check = "--check" in sys.argv
    if "--revert" in sys.argv:
        print("== 回滚 ==")
        return revert()

    rc = 0
    for label, (p, kind) in TARGETS.items():
        print("-- %s: %s" % (label, p))
        if not p.is_file():
            print("   [FAIL] 文件不存在")
            rc = 1
            continue
        s = p.read_text()

        if kind == "model":
            if "_lnorm_fuse_on" in s:
                print("   [skip] 已打过（含 _lnorm_fuse_on）")
                continue
            cls_anchor = "class DeepseekV2DecoderLayer(nn.Module):"
            if cls_anchor not in s:
                print("   [FAIL] 找不到 class 锚点（变体不兼容？）")
                rc = 1
                continue
            if MODEL_OLD not in s:
                print("   [FAIL] 找不到 input_layernorm 锚点")
                rc = 1
                continue
            if check:
                print("   [ok] 两个锚点都在（check 模式不改）")
                continue
            b = backup(p)
            s = s.replace(cls_anchor, MODEL_HELPER.lstrip("\n") + "\n" + cls_anchor, 1)
            s = s.replace(MODEL_OLD, MODEL_NEW, 1)
            p.write_text(s)
            print("   [ok] 已打（备份 %s）" % b.name)

        else:
            if "_lnorm_fused_quant" in s:
                print("   [skip] 已打过（含 _lnorm_fused_quant）")
                continue
            if DSA_OLD not in s:
                print("   [FAIL] 找不到 wq_a.quantize 锚点")
                rc = 1
                continue
            # 锚点唯一性：命中多行会替换错地方
            if s.count(DSA_OLD) != 1:
                print("   [warn] 锚点出现 %d 次（期望 1），将只替换第一处" % s.count(DSA_OLD))
            if check:
                print("   [ok] 锚点在（check 模式不改）")
                continue
            b = backup(p)
            s = s.replace(DSA_OLD, DSA_NEW, 1)
            p.write_text(s)
            print("   [ok] 已打（备份 %s）" % b.name)

    if check and rc == 0:
        print("\n锚点检查通过：可安全应用")
    elif rc == 0:
        print("\n完成。启用：V41_LNORM_FUSE=1 + ASCEND_CUSTOM_OPP_PATH（见 SWITCHES.md）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
