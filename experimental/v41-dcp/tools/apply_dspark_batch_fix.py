#!/usr/bin/env python3
"""DSpark × DCP 并发崩溃的正式修复（两个独立缺陷，均在 dsa_v41.py）。

背景与证据：docs/V41-DSPARK-BATCH1-ROOTCAUSE-20261001.md

## 缺陷 1（主因）：第二次纯 ori 的 SMLA 调用 `seqused_cmp_kv` 非零

`if dcp_active:` 段会做第二次 SMLA 调用，只为拿 `(A, A·O_ori)` 供跨 rank 合并。
它把 `cmp_sparse_indices` 传全 -1，但 `seqused_cmp_kv` 传**本 rank 的压缩长度**。
注释声称"全 -1 ⇒ actCmpS2Size=0 ⇒ 不读 cmp"，但实测 batch>=2 时该前提不成立
⇒ device 侧 `SparseFlashMla_..._mix_aic` 报
  `The scalar instruction accesses an invalid GM address`。

代码里本就有正确语义的开关 `ori_zero_cmp`，但它是**文件驱动**的，而 decode 走
整图捕获 ⇒ 生产路径上永远来不及生效（`dsa_v41.py:652-653` 自己承认过）。
⇒ 把"确定性置零"变成默认；旧行为用 `V41_DCP_ORI_RAW_CMP=1` 退回。

## 缺陷 2（顺带）：metadata 的 num_heads_q 在 DCP=1 下被错误放大

`dsa_v41.py:4240` 的条件是 `_v41_dcp_on()`（只读 env），而
`_v41_dcp_gather_heads` 的实际行为是 `if dcp_size <= 1: return q`（no-op）。
⇒ DCP=1 且 env 开关为真时，q 仍是 TP 分片（32 head），metadata 却按 64 head 建
⇒ 2× 错配。收紧条件为 `world_size > 1`。

单独**不修复崩溃**（实测），但它是真缺陷，且对 DCP8 无影响。

用法（a3-21）：python3 apply_dspark_batch_fix.py
幂等：已打过会报"已应用"。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "dcpw/vllm_ascend/attention/dsa_v41.py"

FIX1_OLD = """                _ori_cmp_lens = cmp_seq_lens
                if _perf_flags().get("ori_zero_cmp") == "1" and cmp_seq_lens is not None:
                    _ori_cmp_lens = torch.zeros_like(cmp_seq_lens)
"""

FIX1_NEW = """                _ori_cmp_lens = cmp_seq_lens
                # [V41-DCP-ORI-ZEROCMP 2026-10-01] ★★ 默认把 cmp 长度**确定性置零**。
                # 第二次调用只要纯 ori；旧写法依赖"全 -1 索引 ⇒ actCmpS2Size=0"
                # 这个**隐含前提**，而实测 batch>=2 时该前提不成立
                # （`SparseFlashMla` 报 invalid GM address）。
                # 置零后算子没有任何机会去读 cmp 键 —— 语义更严格、行为更确定。
                # 退回旧行为：V41_DCP_ORI_RAW_CMP=1
                _raw_cmp = __import__("os").environ.get("V41_DCP_ORI_RAW_CMP", "0") == "1"
                if (
                    not _raw_cmp
                    and cmp_seq_lens is not None
                ) or (
                    _perf_flags().get("ori_zero_cmp") == "1" and cmp_seq_lens is not None
                ):
                    _ori_cmp_lens = torch.zeros_like(cmp_seq_lens)
"""

FIX2_OLD = """        if cache_kind == "long_kv" and _v41_dcp_on() and has_compressed:
            n_local_heads = int(_config_value(text_config, "num_attention_heads"))
"""

FIX2_NEW = """        # [V41-DCP-HEADS-FIX 2026-10-01] ★★ 必须与 `_v41_dcp_gather_heads`
        # 的**实际行为**对齐：那个函数是 `if dcp_size <= 1: return q`（no-op）。
        # 原条件只看 `_v41_dcp_on()`（env 开关），而 DCP=1 时它仍可能为 True
        # （`V41_DCP_ALLOW_CAPACITY_PROBE=1` 也会让它为真）
        # ⇒ q 仍是 TP 分片（32 head）而 metadata 按全量（64 head）建 ⇒ 2× 错配。
        # 对 DCP>1 无影响（world_size>1 时条件本就成立）。
        if (
            cache_kind == "long_kv"
            and _v41_dcp_on()
            and has_compressed
            and _v41_dcp_group().world_size > 1
        ):
            n_local_heads = int(_config_value(text_config, "num_attention_heads"))
"""


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def apply_once(s, old, new, tag, marker=None):
    """marker 用于幂等判定（默认取 new 里的特征串）。"""
    if marker is None:
        marker = new.strip().splitlines()[0].strip()
    if marker in s:
        print("[ok] %s 已应用" % tag)
        return s
    n = s.count(old)
    if n != 1:
        raise SystemExit("[FAIL] %s: 锚点 %d 次（应为 1）" % (tag, n))
    print("[apply] %s" % tag)
    return s.replace(old, new, 1)


def main():
    print("[before] dsa_v41.py md5=%s" % md5(P))
    s = P.read_text()
    s = apply_once(s, FIX1_OLD, FIX1_NEW, "缺陷1（ori_zero_cmp 默认开）",
                   marker="[V41-DCP-ORI-ZEROCMP 2026-10-01]")
    s = apply_once(s, FIX2_OLD, FIX2_NEW, "缺陷2（num_heads_q 只看 world_size>1）",
                   marker="[V41-DCP-HEADS-FIX 2026-10-01]")
    P.write_text(s)
    print("[after ] dsa_v41.py md5=%s" % md5(P))
    import ast
    ast.parse(P.read_text())
    print("[ok] 语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
