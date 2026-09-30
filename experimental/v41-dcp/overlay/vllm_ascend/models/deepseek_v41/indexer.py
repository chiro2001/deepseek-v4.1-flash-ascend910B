# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    _is_capturing,
    scatter_cache_sk,
)
from vllm_ascend.core.deepseek_v41 import DeepseekV41IndexerSpec
from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active as _v41_dcp_active
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .compressor import DeepseekV41RMSNorm, _read

_PERF_FLAG_PATH = "/tmp/v41_perf_flags"
_PERF_FLAG_CACHE = {"t": None, "v": {}}


def _perf_flags_indexer() -> dict:
    """文件驱动开关（与 `dsa_v41.py` 的同名机制一致）；只供诊断探针使用。"""
    import os as _o

    try:
        st = _o.stat(_PERF_FLAG_PATH)
    except OSError:
        return {}
    if st.st_mtime != _PERF_FLAG_CACHE["t"]:
        out = {}
        try:
            with open(_PERF_FLAG_PATH) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
        except OSError:
            pass
        _PERF_FLAG_CACHE["t"] = st.st_mtime
        _PERF_FLAG_CACHE["v"] = out
    return _PERF_FLAG_CACHE["v"]


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.vllm_config = vllm_config
        self.compress_ratio = compress_ratio
        self.n_heads = int(_read(config, "index_n_heads"))
        self.width = int(_read(config, "index_head_dim"))
        self.rope_width = int(_read(config, "qk_rope_head_dim"))
        self.index_topk = int(_read(config, "index_topk"))
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        self.wq_b = ReplicatedLinear(
            _read(config, "q_lora_rank"),
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            _read(config, "hidden_size"),
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                _read(config, "head_dim"),
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41IndexerSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width,
                    dtype=torch.int8,
                    compress_ratio=compress_ratio,
                    scale_dim=1,
                    scale_dtype=torch.float16,
                    # [V41-DCP 2026-09-29] 复制态：每个 rank 留一份全量 indexer K，
                    # 这样 8 个 rank 各自独立算出**同一份全局 top-k**（零通信）。
                    # 见 DeepseekV41IndexerSpec.dcp_world_size 的注释。
                    dcp_world_size=(
                        int(getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
                        if _v41_dcp_active()
                        else 1
                    ),
                ),
            )

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        key = key.squeeze(1)
        quantized, scale = torch_npu.npu_dynamic_quant(
            key, dst_type=torch.int8
        )
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        scatter_cache_sk(k_cache, slots, quantized)
        scatter_cache_sk(
            scale_cache,
            slots,
            scale.unsqueeze(-1).to(torch.float16),
        )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * self.weights_scale

        return self.select_projected(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
        )

    def select_projected(
        self,
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
    ):
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.

        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. Query quantization and position
        ordering stay outside the native QLI/candidate operator.
        """
        if is_candidate_source and uses_candidate_filter:
            raise ValueError("A candidate source must use the unfiltered position TopK")
        if uses_candidate_filter and candidates is None:
            raise RuntimeError("V4.1 candidate-filtering indexer ran before its source")
        if self.width != 128 or self.n_heads not in (32, 64):
            raise ValueError("A3 QLI requires index_head_dim=128 and 32 or 64 index heads")
        if not 1 <= self.index_topk <= 2048:
            raise ValueError("A3 QLI requires index_topk in [1, 2048]")
        if self.compress_ratio not in (1, 2):
            raise ValueError("Aurora QLI supports compression ratios 1 and 2")
        if is_candidate_source or uses_candidate_filter:
            if not 0 < candidate_topk_blocks <= 2048 or candidate_topk_blocks % 64:
                raise ValueError("candidate_topk_blocks must be a multiple of 64 in [64, 2048]")
            if candidate_block_size != 8:
                raise ValueError("The current A3 candidate kernel requires candidate_block_size=8")
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidates.shape != candidate_shape or candidates.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
        topk = self.index_topk
        if query.shape[0] == 0:
            selected = torch.full(
                (0, topk), -1, dtype=torch.int32, device=query.device
            )
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates

        quantized_query, query_scale = quantize_indexer_query(query)
        weights = weights.to(torch.float16)
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.
        cu_seqlens_q = source_metadata.query_start_loc
        seqused_k = source_metadata.cache_seq_lens
        residual = source_metadata.cmp_residual
        common = dict(
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            max_seqlen_q=source_metadata.max_query_len,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
        )
        op_metadata = source_metadata.qli_metadata
        if op_metadata is None:
            raise RuntimeError("V4.1 QLI metadata was not built")
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        # =====================================================================
        # ★★★★★★ [V41-QLISYNC 2026-09-30 17:05] **metadata 异步竞态的直接判据**。
        #
        # 现场（§12）：同一条 prompt，第 1 次请求正确、第 2 次起错；而
        # index K 本页指纹 / long_kv 内容 / positions / complete_groups / 块表结构
        # **全部逐位相同**，只有 `cmp_indices` 差 6 个元素、`lse` 出现 NaN。
        # ⇒ 剩下三选一：算子非确定 / metadata 竞态 / 越界列读。
        # 本开关在 QLI **之前**插一次设备同步：若第 2 次请求变正确 ⇒ metadata 竞态。
        # 只在非 capture 下用（capture 区 host 同步会崩）。
        # =====================================================================
        if not _is_capturing() and _perf_flags_indexer().get("qlisync") == "1":
            try:
                torch.npu.synchronize()
            except Exception:  # noqa: BLE001
                pass
        mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
        selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            quantized_query,
            key,
            weights,
            query_scale,
            key_scale,
            topk,
            2,
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            candidate_topk_index=candidates if uses_candidate_filter else None,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common,
        )
        # =====================================================================
        # ★★★★★★ [V41-IDXDET 2026-09-30 17:10] **QLI 算子自身的确定性判据**。
        #
        # 本轮最关键的一刀：**用逐位相同的输入、在同一个 forward 里再调一次 QLI**，
        # 然后逐位比对 `selected`。
        #   · 两次不同 ⇒ **算子自身非确定**（未初始化 workspace / 并列分数的
        #     tie-break 不稳定）⇒ 与本 DCP 的槽位、块表、合并全部无关；
        #   · 两次相同 ⇒ 单次 forward 内确定 ⇒ 差异来自**跨请求**状态。
        # 只读调用（QLI 不写 KV），且只在非 capture 下跑。
        # =====================================================================
        if (
            not _is_capturing()
            and _perf_flags_indexer().get("idxdet") == "1"
            and int(selected.shape[0]) > 300
        ):
            try:
                _sel2, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
                    quantized_query,
                    key,
                    weights,
                    query_scale,
                    key_scale,
                    topk,
                    2,
                    block_table=source_metadata.block_table,
                    metadata=op_metadata,
                    candidate_topk_index=candidates if uses_candidate_filter else None,
                    candidate_mode=mode,
                    candidate_topk_blocks=candidate_topk_blocks,
                    candidate_block_size=candidate_block_size,
                    **common,
                )
                _a = selected.detach().to(torch.int64).cpu()
                _b = _sel2.detach().to(torch.int64).cpu()
                _diff = int((_a != _b).sum())
                print(
                    "[V41-IDXDET] ratio=%d topk=%d rows=%d cols=%d | 两次调用逐位相同=%s 差异元素=%d "
                    "n_valid=%d/%d sum=%d/%d max=%d/%d"
                    % (int(self.compress_ratio), int(topk),
                       int(_a.shape[0]), int(_a.shape[1]), str(_diff == 0), _diff,
                       int((_a >= 0).sum()), int((_b >= 0).sum()),
                       int(_a[_a >= 0].sum()) if int((_a >= 0).sum()) else -1,
                       int(_b[_b >= 0].sum()) if int((_b >= 0).sum()) else -1,
                       int(_a.max()), int(_b.max())),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-IDXDET] 探针失败：%r" % (_e,), flush=True)
        # ★★ [V41-PERF 2026-09-29] 只跑**一次** `prepare_indexer_indices`。
        #
        # 旧写法先把**全局**位置喂进去过滤一次，再用 `_fix_visibility_for_sharded_k`
        # 把位置换成局部等价编码、**再过滤一次**。而
        # `prepare_indexer_indices` 不是纯比较——它在 triton 核里对每行做一次
        # `tl.extra.cann.extension.sort`（`ops/triton/prepare_indexer_indices.py`），
        # 是这条路径上最贵的算子；跑两遍等于白付一次排序。
        #
        # 两次与一次**结果逐位相同**：过滤条件是单调的（全局界 `(p+1)//ratio`
        # 恒 ≥ 局部可见数 `vlc`），所以「先全局过滤再局部过滤」等价于
        # 「直接按局部过滤」；而局部过滤正是把 `(p'+1)//ratio == vlc` 代回核内，
        # 与旧实现的第二次调用逐字等价（同一函数、同一 `ratio`）。
        #
        # DCP 关闭或 indexer 复制态时 `_dcp_visibility_positions` **原样返回**
        # `positions` ⇒ 单次调用的语义与旧实现的第一步完全一致（旧实现的第二步
        # 本来就会 early-return）。
        _vis = self._dcp_visibility_positions(positions)
        # =====================================================================
        # ★★★★★★ [V41-IDXDET 2026-09-30 17:08] **QLI 算子自身的确定性判据**。
        #
        # 这是本轮最重要的一次判别：**用完全相同的输入、在同一个 forward 里
        # 连续调用两次 QLI**，逐位比对 `selected`。
        #   · 两次不同 ⇒ **算子自身非确定**（读未初始化 workspace / tie-break 不稳定）
        #     ⇒ 与 DCP 的合并、槽位、块表全部无关，必须改算子或 tie-break；
        #   · 两次相同 ⇒ 算子在单次 forward 内确定 ⇒ 差异来自**跨请求**的
        #     某些我们还没测到的状态。
        #
        # 背景（§12）：输入指纹逐位相同、输出差 6 个元素，所以这一刀必须切。
        # 只读调用（QLI 不写 KV），且只在非 capture 下跑。
        # =====================================================================
        # =====================================================================
        # ★★★★★★ [V41-IDXVIS 2026-09-30 13:35] **可见性过滤的现场对拍**。
        #
        # 动机（实测）：层2/T=407 上 DCP1 的有效索引总数 = 41412（= Σ_t (t+1)//2，
        # 即全部可见压缩键），而 DCP8 八个 rank 合计只有 **5796 = 14%**。
        # ⇒ 分片态下 86% 的可见键被丢掉，与"top-k 集合语义"无关，是**真实缺陷**。
        #
        # 本探针打印前几行的 `positions`（全局位置）、`_vis`（重映射后的等价位置，
        # `(vis+1)//ratio == vlc`）、以及过滤后每行的有效索引数。
        #   · 若 `nvalid[t] == vlc[t]` ⇒ 过滤没问题，丢键发生在算子侧（QLI 的 mask）；
        #   · 若 `nvalid[t] << vlc[t]` ⇒ QLI 返回的有效索引本身就少。
        # 文件开关 `idxvis=1`（prefill 是 eager ⇒ 免重启）。
        # =====================================================================
        if (
            __import__("os").environ.get("V41_IDXVIS") == "1"
            or _perf_flags_indexer().get("idxvis") == "1"
        ) and int(selected.shape[0]) > 300:
            try:
                _p_cpu = positions.detach().to(torch.int64).cpu()
                _v_cpu = _vis.detach().to(torch.int64).cpu()
                _pre = selected.squeeze(1).detach().to(torch.int64).cpu()
                _post = prepare_indexer_indices(
                    selected.squeeze(1), _vis, self.compress_ratio
                ).detach().to(torch.int64).cpu()
                _n_pre = (_pre >= 0).sum(dim=1)
                _n_post = (_post >= 0).sum(dim=1)
                _r = min(6, int(_p_cpu.numel()))
                print(
                    "[V41-IDXVIS] ratio=%d rows=%d cols=%d "
                    "pos=%s vis=%s vlc=%s npre=%s npost=%s | tot_pre=%d tot_post=%d"
                    % (
                        self.compress_ratio, int(_p_cpu.numel()), int(_pre.shape[1]),
                        _p_cpu[:_r].tolist(), _v_cpu[:_r].tolist(),
                        [int((v + 1) // self.compress_ratio) for v in _v_cpu[:_r].tolist()],
                        _n_pre[:_r].tolist(), _n_post[:_r].tolist(),
                        int(_n_pre.sum()), int(_n_post.sum()),
                    ),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-IDXVIS] 探针失败：%r" % (_e,), flush=True)
        selected = prepare_indexer_indices(selected.squeeze(1), _vis, self.compress_ratio)
        return selected, candidate_out if is_candidate_source else candidates

    def _dcp_visibility_positions(self, positions):
        """★★ [V41-DCP 2026-09-29] 把 top-k 的**因果可见性过滤**改到局部坐标系。

        `prepare_indexer_indices` 内部用 `visible = (positions+1)//ratio` 过滤，
        其中 `positions` 是**全局**位置。当 indexer K 缓存是**分片态**时
        `selected` 是**本 rank 局部**压缩索引，两者坐标系不一致 ⇒ rank>0 会把
        大量**未来键**判为可见（离线对拍：1803 次错判；rank=5、p=40 时让 query
        看到位置 160+ 的键）。这正是"L≤59 正常、L≥128 开始乱"的根因。

        这里用 `local_visible_positions` 把全局位置换成等价编码，**供唯一那次**
        `prepare_indexer_indices` 使用（见 `select_projected` 里的说明）。
        """
        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer, v41_dcp_active

        if not v41_dcp_active() or replicate_indexer():
            return positions
        parallel = self.vllm_config.parallel_config
        dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
        if dcp_size <= 1:
            return positions
        from vllm.distributed import get_dcp_group

        from vllm_ascend.attention.context_parallel.v41_dcp import (
            local_visible_positions,
        )

        return local_visible_positions(
            positions,
            interleave=int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1),
            ratio=self.compress_ratio,
            dcp_size=dcp_size,
            dcp_rank=int(get_dcp_group().rank_in_group),
        )
