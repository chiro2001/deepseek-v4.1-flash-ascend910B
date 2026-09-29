# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.circular_buffer import AscendCircularBufferManager, AscendCircularBufferSpec


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        storage_block_sizes = {
            getattr(spec, "storage_block_size", spec.block_size) for spec in kv_cache_spec.kv_cache_specs.values()
        }
        assert len(storage_block_sizes) == 1, "All specs in one KV cache group must use the same storage block size."
        return storage_block_sizes.pop()
    return getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    # Sparse C8 changes the main cache into one packed byte tensor. Keep that
    # main-cache property here; indexer-specific C8 properties belong to the
    # indexer spec.
    cache_sparse_sfa_c8: bool = False
    store_on_host: bool = False

    @property
    def storage_block_size(self) -> int:
        """Return the physical block size consumed by Ascend kernels."""
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.storage_block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        ascend_layouts = {
            (
                spec.scale_dim,
                spec.scale_dtype,
                spec.cache_sparse_sfa_c8,
                spec.store_on_host,
                spec.alignment,
            )
            for spec in specs
        }
        assert len(ascend_layouts) == 1, (
            "All attention layers in the same KV cache group must use the same Ascend KV cache layout."
        )
        non_causal_multi_token_decode_set = set(spec.non_causal_multi_token_decode for spec in specs)
        assert len(non_causal_multi_token_decode_set) == 1, (
            "Causal target layers and non-causal multi-token draft layers must use separate KV cache groups."
        )
        first_spec = specs[0]
        merged = super().merge(specs)
        return replace(
            merged,
            scale_dim=first_spec.scale_dim,
            scale_dtype=first_spec.scale_dtype,
            alignment=first_spec.alignment,
            cache_sparse_sfa_c8=first_spec.cache_sparse_sfa_c8,
            store_on_host=first_spec.store_on_host,
            indexes_kv_by_block_stride=first_spec.indexes_kv_by_block_stride,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The scheduler should treat this as a full-attention-compatible cache so it
    can share block ids with the MLA cache in the same UniformType group. The
    model runner still allocates it as an independent physical cache tensor.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_li_c8: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return (
            self.sfa_dcp_replicated_indexer_size
            * num_heads_per_page
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSFAIndexerCacheSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        dtype_set = set(spec.dtype for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        cache_sparse_li_c8_set = set(spec.cache_sparse_li_c8 for spec in specs)
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(dtype_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(cache_sparse_li_c8_set) == 1
            and len(sfa_dcp_replicated_indexer_size_set) == 1
        ), (
            "All SFA indexer cache layers in the same KV cache group must use "
            "the same dtype, scale layout, quantization method, sparse LI C8 "
            "setting and DCP replication size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=dtype_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            cache_sparse_li_c8=cache_sparse_li_c8_set.pop(),
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    # ==================================================================
    # [V41-DCP 2026-09-29] 去掉上游的 `assert decode_context_parallel_size == 1`。
    #
    # 实测依据：run `dcpcap_0929_121645` 在起服时直接抛
    #     AssertionError: DCP not support sliding window.
    # 这条 assert 是「上游没有滑窗 DCP 实现」的保守拒绝，而 DCP 下**滑窗语义
    # 本身是自洽的**（用户判定 + 逐行读 kernel 确认）：
    #   · 物理 block 仍装 `block_size` 个 token ⇒ 单块内存不涨；
    #   · 每个 rank 只写自己那一份（kernel 的 rank 过滤），块表行索引 =
    #     `pos // (block_size * dcp)` ⇒ 一个 block 覆盖 `block_size*dcp` 个全局 token；
    #   · 查询被复制到每个 rank，各 rank 各自算「窗口 ∩ 本 rank 分片」的 partial，
    #     段与段互不重叠，LSE 合并后等于全局窗口注意力 ⇒ **不是重复计算**。
    # 所以这里只去掉拒绝，公式与上游逐字相同。
    # ==================================================================
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """与上游逐字相同的公式，只去掉 `assert dcp == 1`。

        ★★ 修正（2026-09-29，被 `dcp_equiv_harness` 的实机探针推翻了一个假设）：
        滑窗平面在 A3 上**不可分片**，必须**复制**。三条硬证据：

        1. `ori_sparse_indices`（显式给出可见键集合）是 **A5-only**：
           `sparse_flash_mla_tiling.cpp:1246` → `ori_sparse_indices is only supported on A5`。
        2. `ori_mask_mode` 必须为 4、`ori_win_left` 必须为 127（metadata 报 EZ0024/EZ0027）
           ⇒ 滑窗只能表达成「以本地 KV 末端为右沿、宽 128 的连续带」，
           没有任何通道表达「全局窗口 ∩ 本 rank 分片」。
        3. 因此分片只有在「窗口完全落在拥有该 query 的 rank 内」时才精确，
           即 `p ≡ 127 (mod 128)`。实测反例：T=129/130/201、dcp=2 时相对误差
           8.4% / 12.4% / 62%。

        ⇒ 滑窗组按 **DCP=1 语义**处理（`patch_v41_dcp.py` 把 manager 的
        `dcp_world_size` 归 1、`block_table.py` 关掉 rank 过滤），
        本函数因此保持上游公式不动。

        附带一个重要结论：**复制滑窗并不贵** —— 它是「最近 128 token + 在飞 token」
        的滚动窗口，**与序列长度无关**（DCP1 与 DCP8 都是 130 块/组）。
        而按序列分片反而是 `L/dcp/128 = 1024` 块（1M 上下文），
        比滚动窗口贵 8 倍。所以复制不只是「唯一可行」，也是**长期最优**。
        """
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    def __post_init__(self):
        pass

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
                    model_version=model_version_set.pop(),
                )


def register_ascend_kv_cache_specs() -> None:
    from vllm_ascend.core.deepseek_v41 import (
        DeepseekV41CompressorStateSpec,
        DeepseekV41DraftSWASpec,
        DeepseekV41FullSpec,
        DeepseekV41IndexerSpec,
        DeepseekV41SWASpec,
    )

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendCircularBufferSpec,
        manager_class=AscendCircularBufferManager,
        uniform_type_base_spec=AscendCircularBufferSpec,
    )
    for spec, manager in (
        (DeepseekV41FullSpec, FullAttentionManager),
        (DeepseekV41IndexerSpec, FullAttentionManager),
        (DeepseekV41SWASpec, SlidingWindowManager),
        (DeepseekV41DraftSWASpec, SlidingWindowManager),
        (DeepseekV41CompressorStateSpec, AscendCircularBufferManager),
    ):
        KVCacheSpecRegistry.register(kvcache_spec_cls=spec, manager_class=manager, uniform_type_base_spec=spec)
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendMLAAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFAIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSlidingWindowMLASpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    # Imported lazily so this module stays independent of any single model.
    from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager, KpoolTailSpec

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=KpoolTailSpec,
        manager_class=KpoolTailManager,
        uniform_type_base_spec=KpoolTailSpec,
    )
