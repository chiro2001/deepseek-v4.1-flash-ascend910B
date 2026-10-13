#!/usr/bin/env python3
"""DSpark x DCP 修复（第 4 处）：不要为 parallel drafting 计算 MTP 步进元数据。

`_propose` 里 `dcp_manager.prepare_spec_decode_mtp_drafting_inputs(...)` 的结果
只喂给 `should_update_next_steps` 的步进循环：

    should_update_next_steps = not self.parallel_drafting and (...)
    if should_update_next_steps:
        for draft_index in range(1, self.num_speculative_tokens): ...

DSpark/DFlash 的 parallel_drafting=True ⇒ 该循环**永不执行** ⇒ dcp_mtp_inputs
与 draft_cp_kwargs 都是死值。而这次调用在 DSpark 上必崩：
DSpark 的 attn_metadata 既无 seq_lens 也无 seq_lens_cpu
（不走 Ascend 那条 builder），dcp_utils.py:236 直接 assert。

⇒ 加一个 `not self.parallel_drafting` 的门。非 parallel 路径行为**完全不变**。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "cedpd-repo/patches/files/draft/llm_base_proposer.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit("[FAIL] %s: 锚点出现 %d 次（应为 1）" % (tag, n))
    return text.replace(old, new, 1)


OLD = """        if dcp_manager is not None:
            dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs(
"""

NEW = """        # [V41-DCP-DSPARK] parallel drafting（DSpark/DFlash）**不需要** MTP 步进元数据：
        # 下面 `should_update_next_steps = not self.parallel_drafting and (...)` 恒为
        # False ⇒ `for draft_index in range(1, num_speculative_tokens)` 循环从不执行
        # ⇒ dcp_mtp_inputs / draft_cp_kwargs 全是死值。不加这个门会崩：DSpark 的
        # attn_metadata 既没有 seq_lens 也没有 seq_lens_cpu（不走 Ascend 那条
        # builder），dcp_utils.prepare_spec_decode_mtp_drafting_inputs:236 直接 assert。
        if dcp_manager is not None and not self.parallel_drafting:
            dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs(
"""


def main():
    print("[before] llm_base_proposer.py md5=%s" % md5(P))
    s = P.read_text()
    s = sub_once(s, OLD, NEW, "_propose::MTP 步进元数据加门")
    P.write_text(s)
    print("[after ] llm_base_proposer.py md5=%s" % md5(P))
    return 0


if __name__ == "__main__":
    sys.exit(main())
