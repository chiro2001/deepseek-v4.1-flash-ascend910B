#!/usr/bin/env python3
"""DSpark x DCP blocker 的最小修复脚本（在 a3-21 上运行，幂等性：锚点唯一）。

两个独立 bug（证据见 docs/V41-DSPARK-X-DCP-BLOCKER-20261001.md）：

  #1 spec_decode/llm_base_proposer.py::set_inputs_first_pass 的
     needs_extra_input_slots 分支（= parallel drafting，DSpark/DFlash）
     完全没有 DCP 处理，末尾硬编码 return ..., None
     => 调用方 _propose 的 assert long_seq_args is not None 崩。
     修法：用与 EAGLE 分支相同的 prepare_spec_decode_first_pass_inputs
     产出 long_seq_args（只取元组，丢弃它对 token 布局的覆盖），
     索引取展开前的 token_indices_to_sample。

  #2 worker/block_table.py::BlockTable.compute_slot_mapping_draft
     在 effective_dcp_world_size == 1（复制态组：SWA / draft）分支里
     对 device 张量直接 raise ValueError，而 DCP 的推测解码 device 侧重建
     (dcp_utils.rebuild_async_spec_decode_inputs) 会主动传 device 张量。
     修法：补一条与 numpy 路径逐字等价的纯 device 实现。
"""

import hashlib
import pathlib
import sys

BLOCK_TABLE = pathlib.Path.home() / "dcpw/vllm_ascend/worker/block_table.py"
PROPOSER = pathlib.Path.home() / "cedpd-repo/patches/files/draft/llm_base_proposer.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit("[FAIL] %s: 锚点出现 %d 次（应为 1）" % (tag, n))
    return text.replace(old, new, 1)


BT_OLD = '''        else:
            if isinstance(req_indices, torch.Tensor):
                if req_indices.device.type != "cpu":
                    raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
                req_indices = req_indices.numpy()
'''

BT_NEW = '''        else:
            # [V41-DCP-DSPARK] 复制态组（effective_dcp_world_size == 1）的
            # device 侧输入：原先直接 raise。触发场景是 DCP 的推测解码 device 侧
            # 重建（dcp_utils.rebuild_async_spec_decode_inputs）——它为 MTP slot
            # 主动传 device 张量，而 SWA / draft 组正是复制态。
            _device_input = (
                isinstance(req_indices, torch.Tensor) and req_indices.device.type != "cpu"
            ) or (isinstance(positions, torch.Tensor) and positions.device.type != "cpu")
            if _device_input:
                self._compute_replicated_slot_mapping(req_indices, positions)
                return
            if isinstance(req_indices, torch.Tensor):
                req_indices = req_indices.numpy()
'''

BT_ANCHOR = '''    def _compute_dcp_slot_mapping(
        self,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
'''

BT_HELPER = '''    def _compute_replicated_slot_mapping(
        self,
        req_indices: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
    ) -> None:
        """[V41-DCP-DSPARK] 复制态组的纯 device slot-mapping。

        与 compute_slot_mapping_draft 的 numpy 路径逐字等价：
            logical_block_idx = positions // block_size
            block_table_idx   = req_idx * (max_num_blocks_per_req
                                           * blocks_per_phys_block)
                                + logical_block_idx
            slot              = block_number * block_size
                                + positions % block_size
        复制态组不做 DCP 交织切分（每 rank 存整份）⇒ 不需要 mask，
        与 _compute_dcp_slot_mapping 的唯一区别是没有 interleave 折算。
        """
        if not isinstance(req_indices, torch.Tensor):
            req_indices = torch.from_numpy(req_indices)
        if not isinstance(positions, torch.Tensor):
            positions = torch.from_numpy(positions)
        if positions.device != req_indices.device:
            positions = positions.to(req_indices.device)
        assert self.kernel_sizes is not None
        assert self.block_size == self.kernel_sizes[0]
        logical_block_idx = (positions // self.block_size).to(torch.int64)
        block_table_indices = (
            req_indices.to(torch.int64) * self.max_num_blocks_per_req * self.blocks_per_phys_block
            + logical_block_idx
        )
        block_offsets = (positions % self.block_size).to(torch.int64)
        block_numbers = self.block_table.gpu.flatten()[block_table_indices].to(torch.int64)
        num_tokens = req_indices.shape[0]
        slots = block_numbers * self.block_size + block_offsets
        self.slot_mapping.gpu[:num_tokens] = slots.to(self.slot_mapping.gpu.dtype)

'''

PROP_OLD_HEAD = '''        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
'''

PROP_NEW_HEAD = '''        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # [V41-DCP-DSPARK] 抓取**展开前**的 sample 索引。
            # DCP 的 prepare_spec_decode_mtp_drafting_inputs 需要与
            # query_start_loc_full（scheduler 布局）同坐标系的逐请求索引；
            # 而下面 npu_copy_and_expand_eagle_inputs 返回的是**展开后**布局的
            # 索引（batch_size * num_spec 个），两者不可混用 ⇒ 必须在这里先记住入参。
            # 语义核对（prepare_inputs_padded）：入参 =
            #   query_start_loc[1:] - 1 - num_rejected
            # 消费端算：
            #   num_reject = cu_num_tokens - idx - 1  ⇒ 正好还原 num_rejected。
            ori_token_indices_to_sample = token_indices_to_sample
            if ori_token_indices_to_sample is None:
                ori_token_indices_to_sample = cad.query_start_loc[1:] - 1
            # 1.
'''

PROP_OLD_TAIL = '''            return total_num_output_tokens, token_indices_to_sample, new_cad, None
'''

PROP_NEW_TAIL = '''            # [V41-DCP-DSPARK] parallel-drafting（DSpark/DFlash）分支也必须提供
            # DCP 的 first-pass 元数据：上游只按 draft 架构名 K3DSparkModel
            # 拒绝 DSpark x DCP，漏掉了我们的 DSparkDeepseekV41ForCausalLM，
            # 于是绕过那条保护、直接撞上 _propose 的
            # assert long_seq_args is not None。
            # 复用与 EAGLE 分支**完全相同**的
            # prepare_spec_decode_first_pass_inputs（它只做两件事：给 cad 挂上
            # context_parallel_metadata、产出 long_seq_args），但**丢弃**它对
            # num_tokens / 输入张量的覆盖 —— parallel drafting 是展开后的布局，
            # 与 EAGLE 分支不同，不能被它改写。
            long_seq_args = None
            assert self.runner is not None
            dcp_manager = getattr(self.runner, "dcp_manager", None)
            if dcp_manager is not None:
                _first_pass = dcp_manager.prepare_spec_decode_first_pass_inputs(
                    input_ids=self.input_ids[:total_num_output_tokens],
                    target_positions=self.positions[:total_num_output_tokens],
                    target_hidden_states=self.hidden_states[:total_num_output_tokens],
                    token_indices_to_sample=ori_token_indices_to_sample,
                    common_attn_metadata=new_cad,
                    long_seq_metadata=long_seq_metadata,
                    req_scheduled_tokens=req_scheduled_tokens,
                    req_ids=self.runner.input_batch.req_ids,
                    logits_indices=self.runner.logits_indices,
                    num_tokens=total_num_output_tokens,
                    num_prefill_reqs=num_prefill_reqs,
                    num_decode_reqs=num_decode_reqs,
                    uses_mrope=self.uses_mrope,
                )
                long_seq_args = _first_pass.long_seq_args

            return total_num_output_tokens, token_indices_to_sample, new_cad, long_seq_args
'''


def main():
    print("[before] block_table.py      md5=%s" % md5(BLOCK_TABLE))
    print("[before] llm_base_proposer.py md5=%s" % md5(PROPOSER))

    bt = BLOCK_TABLE.read_text()
    bt = sub_once(bt, BT_OLD, BT_NEW, "block_table::else 分支")
    bt = sub_once(bt, BT_ANCHOR, BT_HELPER + BT_ANCHOR, "block_table::新 helper")
    BLOCK_TABLE.write_text(bt)

    prop = PROPOSER.read_text()
    prop = sub_once(prop, PROP_OLD_HEAD, PROP_NEW_HEAD, "proposer::else 头部")
    prop = sub_once(prop, PROP_OLD_TAIL, PROP_NEW_TAIL, "proposer::return")
    PROPOSER.write_text(prop)

    print("[after ] block_table.py      md5=%s" % md5(BLOCK_TABLE))
    print("[after ] llm_base_proposer.py md5=%s" % md5(PROPOSER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
