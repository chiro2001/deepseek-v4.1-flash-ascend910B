# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_pcp
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.llm_base_proposer import (
    _DSPARK_PTR_PROBE,
    _dspark_ptr_line,
    _dspark_row_dump,
    _dspark_ptr_snapshot,
)
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler


# [DSV41 fix] DSpark draft ACL-graph capture must bake in *real* draft attention
# metadata. Without it, AscendDSAImpl.forward() takes its "no metadata" fallback
# branch while the graph is captured, so the replay contains no attention at all.
#
# DSPARK_GRAPH_CAPTURE_METADATA=1 enables the fixed capture path.
# DSPARK_GRAPH_AB_LEGACY_CAPTURES=N keeps the first N capture buckets on the old
# (metadata-less) behaviour so a single server run can A/B both variants.
_DSPARK_CAPTURE_METADATA = os.environ.get("DSPARK_GRAPH_CAPTURE_METADATA", "0") == "1"
_DSPARK_AB_LEGACY_CAPTURES = int(os.environ.get("DSPARK_GRAPH_AB_LEGACY_CAPTURES", "0"))
_DSPARK_CAPTURE_INDEX = 0
_DSPARK_GRAPH_DEBUG = os.environ.get("DSPARK_GRAPH_DEBUG", "0") == "1"
_DSPARK_ROW_DUMP = os.environ.get("DSPARK_ROW_DUMP", "0") == "1"
# [DSV41 fix-candidate] capture 期把 draft 的 slot_mapping 置 -1（不写 KV）。
# 见 `_build_capture_draft_attn_metadata` 里的长注释与 `[dspark-capture-pad]` 日志。
_DSPARK_CAPTURE_PAD_SLOTS = os.environ.get("DSPARK_CAPTURE_PAD_SLOTS", "0") == "1"

# [DSV41 CAPTURE-VALUE-FIX] ★★★ 根因修复（2026-09-20，单 chip 双向对照实测）
#
# 单 chip 上的决定性证据（同输入、同 metadata、同 kernel 直方图，50 种/2088 次逐项相同）：
#
#   | # | context KV 写入 | **捕获期 `runner.seq_lens`** | replay==eager |
#   |---|---|---|---|
#   | ① | 图内写 | 5（dummy） | ❌ |
#   | ②③ | 图外写(+sync) | 0（dummy） | ❌ |
#   | ④ | 图外写 | **1032（真实）** | **✅** |
#   | ⑤ | 图内写 | **1032（真实）** | **✅** |
#   | ⑥⑦ | 任意 | capture 0/1032，**replay 期怎么改都无效** | ❌ |
#
# ⇒ **"值固化"**：被烘进图的是**捕获期的值域**，replay 期改 buffer 内容救不回来
#   （⑥⑦ 是双向对照，排除了"地址漂移"与"图读同一 buffer"两种解释）。
# ⇒ 修法**两部分缺一不可**：
#   (a) 捕获期的 draft metadata 必须用**代表性值域**（`runner.seq_lens` /
#       `optimistic_seq_lens_cpu` 不能是 0/5 这种 dummy 值）；
#   (b) **恢复 context KV 写入**：捕获期把 `_context_slot_mapping_buffers` 填成
#       真实的 per-group 缓冲 list（这样"写 KV"的算子被捕获进图；地址常驻，
#       replay 期由 `set_inputs_first_pass` 原地刷新内容 ⇒ 写到正确位置）。
#
# 用法：`DSPARK_CAPTURE_VALUE_FIX=1` 开启；`DSPARK_CAPTURE_SEQ_LEN=<n>` 指定代表值
#       （默认取 `max_num_batched_tokens`，因为 `num_context` 的上界就是它）。
_DSPARK_CAPTURE_VALUE_FIX = os.environ.get("DSPARK_CAPTURE_VALUE_FIX", "0") == "1"
# [DSV41 CAPTURE-DISPATCH] 见 dummy_run 里的长注释：让 capture 的 bucket 与 replay 一致。
_DSPARK_CAPTURE_DISPATCH = os.environ.get("DSPARK_CAPTURE_DISPATCH", "0") == "1"


def _safe_capturing_flag():
    """[fix] dummy_run reads `capturing` before set_ascend_forward_context, where the
    forward context may not exist yet; a bare get_forward_context() asserts and kills
    profile_run. Swallow the error and return None."""
    try:
        return getattr(get_forward_context(), "capturing", None)
    except Exception:
        return None
# [DSV41 fix step 2] The draft graph also needs the device-metadata *wait* inside
# the graph, otherwise the metadata stream (sas_metadata / dspark_swa_indices) and
# the graph replay can race. Mirrors the target path:
#   worker/model_runner_v1.py:3577-3581 (submit with batch_descriptor)
#   worker/model_runner_v1.py:3013-3021 (_prepare_device_metadata_for_forward)
_DSPARK_DEVICE_METADATA = os.environ.get("DSPARK_GRAPH_DEVICE_METADATA", "0") == "1"
_DSPARK_DEVICE_METADATA_FROM = int(os.environ.get("DSPARK_GRAPH_DEVICE_METADATA_FROM", "0"))
# [DSV41 fix 0005 / executor contract] Mirror of the flag that
# ``llm_base_proposer`` uses to pick the draft metadata execution contract.
# The capture path must run the deferred draft metadata *inline* instead of
# arming an ExternalEvent: with the old submit(capture_tasks, batch_descriptor)
# the captured graph records a wait on an event that the per-step path never
# re-arms, so replay blocks forever (observed: capture_waits=6, replay_submits=0,
# generation throughput 0.06 tok/s).
#
# Read from the environment here instead of importing it, because the two whole
# files are bind-mounted independently (``serve_draftgraph.sh`` mounts each one
# on its own); the default - sync - is deliberately the same expression in both.
_DSPARK_DRAFT_METADATA_MODE = os.environ.get("DSPARK_DRAFT_METADATA_MODE", "sync").strip().lower()
_DSPARK_DRAFT_METADATA_SYNC = _DSPARK_DRAFT_METADATA_MODE != "async"
DSPARK_DRAFT_METADATA_ASYNC_UNIMPLEMENTED = (
    "DSPARK_DRAFT_METADATA_MODE=async is not implemented: the draft graph capture may not arm an "
    "ExternalEvent (the per-step path never re-arms it, so the replayed wait hangs) and may not "
    "submit into the main model's DeviceMetadataExecutor (it raises 'The previous device metadata "
    "submission has not been released' / 'Device metadata frontiers changed for an existing "
    "full-graph batch descriptor'). Run with DSPARK_DRAFT_METADATA_MODE=sync (default)."
)


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        hf_config = self.draft_model_config.hf_config
        hf_config = getattr(hf_config, "text_config", hf_config)
        self.sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        blk = 1 + self.num_speculative_tokens
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # Replace the target-sized DFlash buffers with the draft model's hidden
        # size. Assignment releases the old tensors without an explicit del.
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        dynamic_spec_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec = None

        if dynamic_spec_config.method == "dspark":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dspark",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )
        # [DSV41 patch] 原实现硬编码 eager；这里改为受 spec config 的 enforce_eager 控制，
        # 以便试验把 draft 前向也放进 ACL Graph（默认行为不变：enforce_eager=True → 仍 eager）。
        #
        # [DRAFT-DELTA-BISECT] DSPARK_DRAFT_USE_CUDAGRAPH=0 强制 draft 保持 eager，
        # **但主模型仍然走图** —— 这是把"draft 版文件的非图改动"与"入图"分开的唯一办法：
        # 走 `enforce_eager=1` 会把主模型也变成 eager，那就不是单变量了。
        # 三臂：① stock 文件（基线 A≈2.85）② 本文件 + USE_CUDAGRAPH=0 ③ 本文件 + 入图。
        _draft_use_cudagraph = os.environ.get("DSPARK_DRAFT_USE_CUDAGRAPH", "1") != "0"
        _runner_use_aclgraph = getattr(runner, "_use_aclgraph", None)
        self.use_cuda_graph = bool(
            callable(_runner_use_aclgraph)
            and _runner_use_aclgraph()
            and not vllm_config.speculative_config.enforce_eager
            and _draft_use_cudagraph
        )
        # Max query tokens depend on whether sampling from anchor or not.
        #
        # [DRAFT-GRAPH-NUMINPUT-FIX] This attribute is a *capacity* for the
        # per-query buffers (positions / query slot mappings), not the shape the
        # draft graph computes: DSpark's real query block is
        # `max_batch_size * num_query_per_req` (5 per request here, one row per
        # speculative token — `copy_and_expand_dflash_and_dspark_inputs_kernel`
        # loops `num_query_total = batch_size * num_query_per_req` and its
        # SAMPLE_FROM_ANCHOR branch maps all of a request's rows onto that
        # request's `num_speculative_tokens` samples), and `dummy_run` clamps the
        # *capture* count to this capacity. Under ACLGraph, `_propose` however
        # pads the draft to the runner's capture bucket
        # (`cudagraph_dispatcher.dispatch(...).num_tokens`, always a multiple of
        # `1 + num_speculative_tokens` — see
        # `adjust_cudagraph_sizes_for_spec_decode`) and then indexes the per-query
        # buffers with that padded count (`_pad_draft_buffers`,
        # `build_draft_attn_metadata` -> `dsa_v1.build_for_drafting` ->
        # `spec_slot_mapping[draft_index - 1][:num_input_tokens]`).
        # A capacity below the bucket makes `_pad_draft_buffers` a silent no-op
        # (its `buf[num_tokens:num_input_tokens].fill_(-1)` then covers nothing)
        # while the padded slice still stays shorter than the assignment target —
        # that is the `[6, 2] = [5, 2]` crash. Size it like DFlash, i.e. for the
        # largest bucket the runner can dispatch a draft step to; the extra rows
        # stay inert (slot -1 / position 0 / parallel-drafting token id) and are
        # never read by the captured graph, which is captured with the *unpadded*
        # count, so FIA never computes a junk row.
        self.max_query_tokens = max(
            self.max_batch_size * self.num_query_per_req,
            self.max_batch_size * (1 + self.num_speculative_tokens),
        )
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # The v1 runner owns block tables and slot mappings. Keep per-group
        # references here because K3 draft layers can span multiple cache
        # groups with different logical block sizes.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Per-gid logical block size used to expand slot mappings. The KV
        # manager's physical page can be larger when hybrid cache groups share
        # one allocation, so kv_cache_spec.block_size is not interchangeable
        # with the attention kernel's block size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    def _compute_confidence(
        self,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.num_speculative_tokens
        flat_hidden = last_hidden_states.reshape(num_tokens, last_hidden_states.shape[-1])
        # Markov embeddings of the draft input tokens (cheap lookup, so they
        # are recomputed here instead of being captured in the drafting loop).
        markov_embs = self.model.markov_embed(draft_token_ids[:, : self.num_speculative_tokens])
        # The confidence head concatenates both inputs, so their dtypes must
        # match; it upcasts to float32 internally.
        flat_markov = markov_embs.reshape(num_tokens, markov_embs.shape[-1]).to(flat_hidden.dtype)
        conf_raw = self.model.compute_confidence(flat_hidden, flat_markov)
        confidence = self._dspark_confidence_logits_buffer[:num_reqs]
        confidence.copy_(conf_raw.reshape(num_reqs, self.num_speculative_tokens))
        return confidence

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}
        self.draft_attn_groups: list[AttentionGroup] = []

        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)

                if key not in attention_groups:
                    kernel_block_size = int(
                        kernel_block_sizes[kv_cache_gid]
                        if kernel_block_sizes is not None and kv_cache_gid < len(kernel_block_sizes)
                        else layer_kv_cache_spec.block_size
                    )
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            self.draft_attn_groups.extend(attention_groups.values())

        if (
            getattr(self.runner, "device_metadata_executor", None) is not None
            and self.dcp_size == 1
            and not enable_pcp()
        ):
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, AscendDSAMetadataBuilder):
                    builder.enable_dspark_device_metadata(self.max_query_tokens)

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers[gid]
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel[
                (_compute_num_programs(self._dflash_num_context, num_query_total),)
            ](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        # The model runner has already corrected this canonical host mirror
        # with the accepted-token count. Extend it on CPU alongside the device
        # lengths, without another reject D2H copy or attention-side wait.
        if cad._seq_lens_cpu is not None:
            draft_seq_lens_cpu = cad._seq_lens_cpu.clone()
            draft_seq_lens_cpu[:batch_size].add_(self.num_query_per_req)
            cad._seq_lens_cpu = draft_seq_lens_cpu
            if getattr(cad, "seq_lens_cpu", None) is not None:
                cad.seq_lens_cpu = draft_seq_lens_cpu
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    def _build_capture_draft_attn_metadata(
        self,
        num_reqs: int,
        num_input_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Build the draft attention metadata that FULL-graph capture bakes in.

        Only used from ``dummy_run`` while capturing. Every tensor referenced by
        the returned metadata must be one of the persistent buffers that the
        replay path (``set_inputs_first_pass`` + ``build_draft_attn_metadata``)
        refreshes in place on every step: ``spec_slot_mapping``,
        ``spec_sas_metadata``, ``dspark_swa_indices_buffer``, ``seq_lens_group``,
        ``query_start_loc_group``, the per-group block tables / query slot
        mappings and ``positions``. The construction therefore mirrors
        ``build_draft_attn_metadata`` so capture and replay resolve to the exact
        same device addresses.
        """
        batch_size = num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_actual_tokens = min(num_query_total, num_input_tokens)
        primary_gid = getattr(self, "kv_cache_gid", 0)

        if not self._per_group_block_table_buffers:
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                block_table = self._per_group_block_tables.get(gid)
                if block_table is not None:
                    self._per_group_block_table_buffers[gid] = block_table
        if primary_gid not in self._per_group_block_table_buffers:
            for gid, block_table in self._per_group_block_table_buffers.items():
                primary_gid = gid
                break
        if primary_gid not in self._per_group_query_slot_mapping_buffers:
            query_slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        else:
            query_slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_input_tokens]

        query_start_loc = self.query_start_loc_group[0]
        query_start_loc[: batch_size + 1].copy_(
            self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        )
        query_start_loc[batch_size + 1 :].fill_(0)
        seq_lens = self.seq_lens_group[0]
        seq_lens[:batch_size].copy_(self.runner.seq_lens[:batch_size] + self.num_query_per_req)
        seq_lens[batch_size:].fill_(0)

        # [DSV41 fix-candidate KV-POLLUTION] capture 期的 slot_mapping 是 **dummy 值**
        # （实测 `[[0,0]]×5` ⇒ 物理 block0 / offset0..4）。若 attention 真的按它去写 KV，
        # 就等于往**真实 KV 缓存**的早期槽位写 5 行垃圾，且 capture 发生在引擎初始化阶段、
        # 之后所有请求都会读到被污染的区域。
        # 处置：capture 期把 slot_mapping 置 **-1（pad，不写）**。
        # 依据：`dsa_attn_kv_plan.dsa_kv_compress_scatter` 的注释明确说
        # "padded [-1, -1] rows 会直接传给 SparseFlashMla 的 scatter 路径"（即由 kernel 忽略）。
        # 开关 `DSPARK_CAPTURE_PAD_SLOTS=1` 才生效，便于同一套代码 A/B。
        if _DSPARK_CAPTURE_PAD_SLOTS:
            _slot = self._per_group_query_slot_mapping_buffers.get(primary_gid)
            if _slot is None:
                _slot = self._slot_mapping_buffer
            _slot[:num_input_tokens].fill_(-1)
            logger.warning(
                "[dspark-capture-pad] capture 期 slot_mapping 已置 -1（num_input_tokens=%d）"
                " —— 避免向真实 KV 缓存的 block0/offset0..4 写 dummy 数据",
                num_input_tokens,
            )

        cad = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc[: batch_size + 1],
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
            ).to(torch.int32),
            seq_lens=seq_lens[:batch_size],
            seq_lens_cpu=self.runner.optimistic_seq_lens_cpu[:batch_size],
            num_reqs=batch_size,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            max_query_len=self.num_query_per_req,
            max_seq_len=0,
            decode_token_per_req=self.num_query_per_req,
            slot_mapping=query_slot_mapping,
            block_table_tensor=self._per_group_block_table_buffers[primary_gid][:batch_size],
            positions=self.positions,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            is_prefilling=torch.zeros(batch_size, dtype=torch.bool),
        )

        shared_cache: dict[str, Any] = dict(common_ratio_to_sas_metadata=dict()) if self.use_compress else {}
        per_layer_attn_metadata: dict[str, Any] = {}
        captured_tasks: list[Any] = []
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            builder = attn_group.get_metadata_builder()
            group_cad = copy.copy(cad)
            block_table = self._per_group_block_table_buffers.get(gid)
            if block_table is not None:
                group_cad.block_table_tensor = block_table[:batch_size]
            slot_mapping = self._per_group_query_slot_mapping_buffers.get(gid)
            if slot_mapping is not None:
                group_cad.slot_mapping = slot_mapping[:num_input_tokens]
            if self.sliding_window is not None:
                self.sliding_window.apply(group_cad)
            attn_metadata = builder.build_for_drafting(group_cad, draft_index=1, **dict(shared_cache))
            take_tasks = getattr(builder, "take_device_metadata_tasks", None)
            if callable(take_tasks):
                captured_tasks.extend(take_tasks())
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        # [DSV41 ptr-probe] capture 侧：记下契约张量的地址，供 replay 比对。
        if _DSPARK_ROW_DUMP:
            _dspark_row_dump("CAPTURE", self, num_rows=6)
        if _DSPARK_PTR_PROBE and per_layer_attn_metadata:
            _first = next(iter(per_layer_attn_metadata.values()))
            _desc_key = f"{num_input_tokens}x{num_reqs}"
            logger.warning(
                "%s",
                _dspark_ptr_line("CAPTURE", _desc_key, _dspark_ptr_snapshot(_first)),
            )
            # [DSV41 capture-audit] capture 期的 metadata **是不是真的被模型消费了**
            # —— 光"我们构造了 metadata"不等于"注意力用了它"。
            # 这里把要交给 forward_context 的那个 dict 的键打出来（模型按 key 查），
            # 之后和 dsa-probe 的 `hit=` 字段对照即可。
            logger.warning(
                "[dspark-capture-audit] 交给模型的 per_layer keys=%s (共 %d 层) "
                "slot_mapping_shape=%s num_query_total=%d num_input_tokens=%d num_reqs=%d",
                list(per_layer_attn_metadata.keys())[:4],
                len(per_layer_attn_metadata),
                tuple(_first.req_metadata.slot_mapping.shape)
                if getattr(_first, "req_metadata", None) is not None
                and getattr(_first.req_metadata, "slot_mapping", None) is not None
                else None,
                num_query_total,
                num_input_tokens,
                num_reqs,
            )
        return [per_layer_attn_metadata], captured_tasks

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_total = num_reqs * self.num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)

        # [DSV41 CAPTURE-DISPATCH] ★ 2026-09-20 根因修复候选：**capture 与 replay 的
        # bucket 必须一致**。
        #
        # 事实（ptr 探针实测）：
        #   capture key = `5x1`（num_input_tokens=5 = num_reqs*num_query_per_req）
        #   replay  key = `6x1`（num_tokens=6 = cudagraph_dispatcher 的 bucket）
        # ⇒ 图是按 **5 行**捕获的，却按 **6 行**重放。
        # 而 eager 臂两侧都是 5（`_propose` 里 use_cuda_graph=False 时不 dispatch）⇒ 正常。
        #
        # 原因是两条路径的 bucket 选择逻辑不同：
        #   `_propose`（replay）: dispatch(raw) → sync_metadata_across_dp → dispatch(再次) → 用 bucketed 值
        #   `dummy_run`（capture）: 只做 sync_metadata_across_dp，**从不 dispatch**
        # 本开关让 capture 走和 replay **完全相同**的两次 dispatch。
        _use_dispatch = (
            _DSPARK_CAPTURE_DISPATCH
            and self.use_cuda_graph
            and aclgraph_runtime_mode == CUDAGraphMode.FULL
            and num_reqs > 0
        )
        if _use_dispatch:
            try:
                _, _bd_cap1 = self.runner.cudagraph_dispatcher.dispatch(
                    num_tokens=num_query_tokens, uniform_decode=True, has_lora=False
                )
                num_query_tokens = _bd_cap1.num_tokens
            except Exception as _exc:  # pragma: no cover - probe only
                logger.warning("[dspark-capture-dispatch] 第一次 dispatch 失败：%r", _exc)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if _use_dispatch:
            try:
                _, _bd_cap2 = self.runner.cudagraph_dispatcher.dispatch(
                    num_tokens=num_input_tokens, uniform_decode=True, has_lora=False
                )
                if _bd_cap2 is not None and _bd_cap2.num_tokens is not None:
                    num_input_tokens = _bd_cap2.num_tokens
            except Exception as _exc:  # pragma: no cover - probe only
                logger.warning("[dspark-capture-dispatch] 第二次 dispatch 失败：%r", _exc)
        if _DSPARK_CAPTURE_DISPATCH:
            logger.warning(
                "[dspark-capture-dispatch] capture: num_reqs=%s num_query_total=%s "
                "num_query_tokens(after disp)=%s num_input_tokens(final)=%s（replay 侧用 "
                "dispatcher 的 bucket，两者必须相等）",
                num_reqs,
                num_query_total,
                num_query_tokens,
                num_input_tokens,
            )

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        if _DSPARK_GRAPH_DEBUG:
            logger.warning(
                "[dspark-graph-probe] dummy_run num_tokens=%s num_reqs=%s num_query_per_req=%s "
                "num_query_total=%s num_input_tokens=%s runtime_mode=%s is_profile=%s "
                "use_cuda_graph=%s capturing=%s draft_attn_groups=%s draft_attn_layers=%s",
                num_tokens,
                num_reqs,
                self.num_query_per_req,
                num_query_total,
                num_input_tokens,
                aclgraph_runtime_mode,
                is_profile,
                self.use_cuda_graph,
                _safe_capturing_flag(),
                len(getattr(self, "draft_attn_groups", [])),
                len(getattr(self, "attn_layer_names", [])),
            )

        multi_steps_attn_metadata: list[dict[str, Any]] = []
        capture_device_metadata_executor = None
        if (
            _DSPARK_CAPTURE_METADATA
            and aclgraph_runtime_mode == CUDAGraphMode.FULL
            and not is_profile
            and num_reqs > 0
        ):
            # [DSV41 CAPTURE-VALUE-FIX] ★ 必须在 `_build_capture_draft_attn_metadata` **之前**
            # 执行 —— 该函数会用 `runner.seq_lens` / `optimistic_seq_lens_cpu` 构建捕获期
            # metadata。2026-09-20 实测：把这两段放在它**之后**（原实现的位置）会让修复
            # **完全不生效**（同一组参数，只改顺序：early ✅ / late ❌）。
            if _DSPARK_CAPTURE_VALUE_FIX and num_reqs > 0:
                # (b) 恢复 context KV 写入：捕获期填成真实 per-group 缓冲 list。
                # 这些缓冲在 `initialize_attn_backend`（init 期）就建好、地址常驻；
                # replay 期 `set_inputs_first_pass` 会原地刷新内容 ⇒ 写到正确位置。
                self._context_slot_mapping_buffers = [
                    self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
                ]
                # (a) 代表性值域：让捕获期 metadata 用代表值构建（不能是 0/5 这类 dummy 值）。
                #     实测：`V = max_num_tokens`（生产 8192）配合"索引缓冲常驻"后，
                #     replay 的 R 从 6 到 262144 **全部 ✅**；V 太小（如 6）则失败。
                _cap_len = int(os.environ.get("DSPARK_CAPTURE_SEQ_LEN", "0") or 0)
                if _cap_len <= 0:
                    _cap_len = int(getattr(self, "max_num_tokens", 0) or 0) or 8192
                try:
                    self.runner.seq_lens[:num_reqs].fill_(_cap_len)
                    self.runner.optimistic_seq_lens_cpu[:num_reqs].fill_(_cap_len)
                except Exception as _ce:  # pragma: no cover
                    logger.warning("[dspark-capture-value-fix] 设置代表值失败：%r", _ce)
                logger.warning(
                    "[dspark-capture-value-fix] 捕获期代表值 seq_len=%d（batch=%d）+ context KV 写入已恢复（%d 组）",
                    _cap_len, num_reqs, len(self._context_slot_mapping_buffers),
                )
            global _DSPARK_CAPTURE_INDEX
            capture_index = _DSPARK_CAPTURE_INDEX
            _DSPARK_CAPTURE_INDEX += 1
            if capture_index >= _DSPARK_AB_LEGACY_CAPTURES:
                multi_steps_attn_metadata, capture_tasks = self._build_capture_draft_attn_metadata(
                    num_reqs, num_input_tokens
                )
                if _DSPARK_DEVICE_METADATA and capture_index >= _DSPARK_DEVICE_METADATA_FROM:
                    if not _DSPARK_DRAFT_METADATA_SYNC:
                        raise RuntimeError(DSPARK_DRAFT_METADATA_ASYNC_UNIMPLEMENTED)
                    # [fix 0005] Sync contract: the deferred metadata is produced by
                    # ``_build_capture_draft_attn_metadata`` (the DSA builder hands
                    # it over as tasks because the draft executor gate is set), so
                    # run it inline on the current stream - exactly like the per-step
                    # path in ``llm_base_proposer.build_draft_attn_metadata``.
                    # Nothing is submitted and no executor is handed to
                    # ``set_ascend_forward_context`` below, so
                    # ``wait_for_device_metadata()`` records no wait into the graph.
                    for _task in capture_tasks:
                        _task.run()
                logger.warning(
                    "[dspark-graph-capture] capture #%d descriptor=%s: built draft attention metadata "
                    "(groups=%d layers=%d num_query_total=%d num_input_tokens=%d tasks=%d "
                    "device_metadata_wait=%s)",
                    capture_index,
                    batch_descriptor,
                    len(self.draft_attn_groups),
                    len(self.attn_layer_names),
                    num_query_total,
                    num_input_tokens,
                    len(capture_tasks),
                    capture_device_metadata_executor is not None,
                )
                # [DSV41 capture-audit] 静默降级探针：capture 期我们把 metadata 交给了
                # `set_ascend_forward_context(draft_attn_metadatas=...)`，但**模型是否真的用它**
                # 取决于 forward 里按 key 的查找。这里记下"我们给了什么"，
                # 与 dsa-probe 打的 `hit=` 对照：hit=None 就说明注意力走了无 metadata 的分支，
                # 图里将**完全没有 attention**（这正是 pos0≈0.07 的头号解释）。
                _md0 = multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None
                logger.warning(
                    "[dspark-capture-audit] capture #%d 提供的 metadata: dict=%s n_layers=%s "
                    "keys=%s",
                    capture_index,
                    "yes" if _md0 else "no",
                    (len(_md0) if _md0 else 0),
                    (sorted(_md0.keys())[:3] if _md0 else None),
                )
            else:
                logger.warning(
                    "[dspark-graph-capture] capture #%d descriptor=%s: LEGACY metadata-less capture "
                    "(A/B control arm)",
                    capture_index,
                    batch_descriptor,
                )

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            device_metadata_executor=capture_device_metadata_executor,
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    # [TARGETPOS-FIX-v3 已撤销] 曾改为共享常驻缓冲
                    # （`_ensure_draft_target_positions`），2026-09-20 臂 F 实测**无效果**
                    # （A=1.070，与未改前逐位相同）⇒ 回退，保持与上游同形。
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )
            forward_context = get_forward_context()
            if (
                multi_steps_attn_metadata
                and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not getattr(forward_context, "capturing", False)
            ):
                # Mirror DFlash: refresh per-step graph params after a real replay
                # (a no-op for DSA, kept for parity with the other draft backends).
                self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)

        if capture_device_metadata_executor is not None:
            capture_device_metadata_executor.release()
