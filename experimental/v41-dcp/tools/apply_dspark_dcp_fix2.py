#!/usr/bin/env python3
"""DSpark x DCP blocker 修复（第 2 处）：AscendDSparkProposer 自己的
set_inputs_first_pass 覆盖（dspark_proposer.py:406），末尾硬编码 `None`。

第 1 处补的是基类 `AscendSpecDecodeBaseProposer.set_inputs_first_pass`，
但 DSpark 走的是**子类 override**（不调用 super），所以必须在这里同样补一次。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "cedpd-repo/patches/files/draft/dspark_proposer.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit("[FAIL] %s: 锚点出现 %d 次（应为 1）" % (tag, n))
    return text.replace(old, new, 1)


HEAD_OLD = """    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # [DYNAMIC-SPEC] replay 路径：先按本步 K 归一 num_query_per_req，
"""

HEAD_NEW = """    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # [V41-DCP-DSPARK] 抓取**本 override 会覆盖掉**的入参（cad 布局的
        # 逐请求 sample 索引），末段要用它给 DCP 产 long_seq_args。
        # 语义：prepare_inputs_padded 给出的是
        #   query_start_loc[1:] - 1 - num_rejected
        # 而 DCP 消费端算 num_reject = cu_num_tokens - idx - 1 ⇒ 正好还原。
        # 必须在下面 `cad.query_start_loc = ...` 就地改写之前算（否则坐标系就变了）。
        ori_token_indices_to_sample = token_indices_to_sample
        if ori_token_indices_to_sample is None:
            ori_token_indices_to_sample = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                ori_token_indices_to_sample = ori_token_indices_to_sample - num_rejected_tokens_gpu
        # [DYNAMIC-SPEC] replay 路径：先按本步 K 归一 num_query_per_req，
"""

TAIL_OLD = """        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None
"""

TAIL_NEW = """        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        # [V41-DCP-DSPARK] DSpark 是 parallel-drafting，上游只按 draft 架构名
        # `K3DSparkModel` 拒绝 DSpark x DCP，漏掉了 `DSparkDeepseekV41ForCausalLM`
        # ⇒ 我们绕过保护、直接撞上 `_propose` 的
        # `assert long_seq_args is not None`。
        # 复用与 EAGLE 分支同一个 `prepare_spec_decode_first_pass_inputs`
        # （它只做两件事：给 cad 挂 context_parallel_metadata、产出 long_seq_args），
        # 但**丢弃**它对 token 布局的覆盖 —— DSpark 的 token 布局是展开后的
        # (num_query_total = batch * num_query_per_req)，不能被它改写。
        long_seq_args = None
        assert self.runner is not None
        dcp_manager = getattr(self.runner, "dcp_manager", None)
        if dcp_manager is not None:
            _first_pass = dcp_manager.prepare_spec_decode_first_pass_inputs(
                input_ids=self.input_ids[:num_query_total],
                target_positions=self.positions[:num_query_total],
                target_hidden_states=self._dflash_hidden_states[:num_query_total],
                token_indices_to_sample=ori_token_indices_to_sample,
                common_attn_metadata=cad,
                long_seq_metadata=long_seq_metadata,
                req_scheduled_tokens=req_scheduled_tokens,
                req_ids=self.runner.input_batch.req_ids,
                logits_indices=self.runner.logits_indices,
                num_tokens=num_query_total,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
                uses_mrope=self.uses_mrope,
            )
            long_seq_args = _first_pass.long_seq_args

        return num_query_total, token_indices_to_sample, cad, long_seq_args
"""


def main():
    print("[before] dspark_proposer.py md5=%s" % md5(P))
    s = P.read_text()
    s = sub_once(s, HEAD_OLD, HEAD_NEW, "dspark::捕获入参")
    s = sub_once(s, TAIL_OLD, TAIL_NEW, "dspark::return")
    P.write_text(s)
    print("[after ] dspark_proposer.py md5=%s" % md5(P))
    return 0


if __name__ == "__main__":
    sys.exit(main())
