# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

# [SWA_pergroup] per-group `blocks_per_chunk` 的共享解析（与 spec/manager 同一份实现）。
try:  # [L3_8card] 8 卡挂载链：pgp_manager 作为 vllm 子模块挂进去
    from pgp_manager import BPC_BY_GROUP_KEY, bpc_map_from_extra  # noqa: E402
except ImportError:  # pragma: no cover - 单卡 c2 的 PYTHONPATH 路径走上面那条
    from vllm.v1.kv_offload.cpu.pgp_manager import (  # noqa: E402
        BPC_BY_GROUP_KEY,
        bpc_map_from_extra,
    )

logger = init_logger(__name__)

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    # Number of this group's offloaded chunks per full-attention alignment
    # segment. Used to skip storing SWA chunks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_chunk_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    # [D2_offload] 该 group 是否参与卸载（False = 调度侧既不存也不查）。
    offload_participating: bool = True
    # [SWA_pergroup] 该组一个 chunk 由几个 GPU block 组成（per-group bpc）。
    # 单卡 tiny/DSV4.1：full=8、SWA=1；标量配置下全部 = spec.blocks_per_chunk。
    blocks_per_chunk: int = 1


class _BPCFixProbe:
    """[M_bpcfix] spec loop 的"四件套"探针（生产代码自己的真值，不重推）。

    背景：`_build_store_jobs` 里 `blocks_per_chunk` 曾经是**跨 loop 泄漏**的局部
    变量（收集 loop 逐组赋值、spec loop 直接用），于是 spec loop 用的是最后一个
    参与组的值（本配置 = SWA 的 1）⇒ full 组（bpc=8）每个 chunk 只搬 1 个 GPU
    block。修法见 spec loop 内的 `bpc_g = group_config.blocks_per_chunk`。

    本探针把 loop 内**真正用到的**四个量打出来（full 组与一个 SWA 组各一行）：
        true_bpc / gpu_block_idx / len(src.block_ids) / Σgroup_sizes
    `BPC_FIX_PROBE=1` 打开（默认关 ⇒ 零开销、零日志体积变化）；
    `BPC_FIX_PROBE_JOBS`（默认 3）= 打几个 store job；`BPC_FIX_PROBE_GROUPS`
    （默认 2）= 每个 job 打几行（首个 full 组 + 首个 SWA 组）。
    判定：**bpc=8 的组必须打 8、bpc=1 的组必须打 1**，且 `Σgroup_sizes` 要等于
    `len(src.block_ids)`（修前 full 组是 1/chunk ⇒ Σ 偏小）。
    """

    def __init__(self) -> None:
        raw = os.environ.get("BPC_FIX_PROBE", "0").strip().lower()
        self.enabled = raw not in ("", "0", "false", "no", "off")
        self.jobs = _env_int("BPC_FIX_PROBE_JOBS", 3, 0)
        self.groups = _env_int("BPC_FIX_PROBE_GROUPS", 2, 0)
        self.printed_jobs = 0
        self.printed = 0

    def record(
        self,
        is_sliding_window: bool,
        row: dict[str, int],
        rows: list[tuple[bool, dict[str, int]]],
    ) -> None:
        """在 spec loop 内按组记录（每个 job 取首个 full 组 + 首个 SWA 组）。"""
        if not self.enabled or len(rows) >= self.groups:
            return
        if any(sliding == is_sliding_window for sliding, _ in rows):
            return
        rows.append((is_sliding_window, row))

    def log_job(
        self,
        req_id: str,
        rows: list[tuple[bool, dict[str, int]]],
        len_src_block_ids: int,
        group_sizes: list[int],
    ) -> None:
        """spec loop 结束后打行（此时 `Σgroup_sizes` 才是整个 job 的真值）。"""
        if not self.enabled or not rows or self.printed_jobs >= self.jobs:
            return
        self.printed_jobs += 1
        for is_sliding_window, row in rows:
            self.printed += 1
            logger.info(
                "[M_bpcfix] #%d req=%s group=%d kind=%s true_bpc=%d "
                "gpu_block_idx=%d chunk=%d len(src.block_ids)=%d "
                "Σgroup_sizes=%d group_sizes=%s",
                self.printed,
                req_id,
                row["group"],
                "swa" if is_sliding_window else "full",
                row["true_bpc"],
                row["gpu_block_idx"],
                row["chunk"],
                len_src_block_ids,
                sum(group_sizes),
                group_sizes,
            )


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


# [M_bpcfix] 四件套探针（默认关闭）。模块级单例：每个进程一份打印预算。
_BPCFIX_PROBE = _BPCFixProbe()


def _prefix_cacheable_offload_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """[D2_offload] 递归判断一个 KV cache group 是否参与前缀缓存。

    DSV4.1 的 13 个 group 全部是 `UniformTypeKVCacheSpecs` 包装，必须递归；
    成员里只要有一个 opt-out（`AscendCircularBufferSpec.prefix_cacheable=False`），
    整个 group 就不参与前缀缓存（与 vllm_ascend 的 `prefix_cacheable()` 同口径）。
    """
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        members = list(kv_cache_spec.kv_cache_specs.values())
        return bool(members) and all(
            _prefix_cacheable_offload_spec(member) for member in members
        )
    return bool(getattr(kv_cache_spec, "prefix_cacheable", True)) and bool(
        getattr(kv_cache_spec, "participates_in_prefix_caching", True)
    )


def _effective_offload_spec(kv_cache_spec: KVCacheSpec) -> KVCacheSpec:
    """[D2_offload] 取出一个 group 的"代表 spec"（unwrap 包装类型）。

    对"是不是滑窗 / 是不是 full attention"这两问，同族成员答案相同。
    """
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        members = list(kv_cache_spec.kv_cache_specs.values())
        if members:
            return members[0]
    return kv_cache_spec


def _offload_participates(kv_cache_spec: KVCacheSpec) -> bool:
    """[D2_offload] 根因修复点：该 group 是否参与 KV 卸载（存 + 查）。

    判据 = 是否参与前缀缓存。为什么必须排除"不参与前缀缓存"的组：

      * store 侧：DSV4.1 的 `state` 组每请求只有 1 页环形 scratch
        （`AscendCircularBufferManager.get_num_blocks_to_allocate` 恒返回 1），
        `storable_chunks() = len(block_ids)//blocks_per_chunk = 1//8 = 0`
        => 它的 key 一个都不会被存；
      * load 侧：`_lookup()` 仍然拿它的 key 去 `_maximal_prefix_lookup`，
        首个 key 必然 MISS => `num_hit_chunks == 0` => `return 0`
        => 整轮请求（含 full / 40 个 SWA 资源）的命中全部被判死。

    GPU 侧前缀缓存路径正是靠"把 `prefix_cacheable=False` 的组排除在命中计算之外"
    来支持这类混合模型（vllm_ascend/patch/platform/patch_kv_cache_coordinator.py
    `verify_and_split_kv_cache_groups()`），这里对齐同一语义：
    state 组在 HBM 侧本来也是每次新分配一页、内容由模型自己重算，
    不参与卸载不改变数值语义。
    """
    return _prefix_cacheable_offload_spec(kv_cache_spec)


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    # [D2_offload] 统一在入口 unwrap：本函数有两处调用（alignment 探测 +
    # 逐 group 建配置），放在函数入口比改两处调用点更小、更不易漏。
    kv_cache_spec = _effective_offload_spec(kv_cache_spec)
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    # [D2_offload] 原为 assert isinstance(kv_cache_spec, FullAttentionSpec)。
    # DSV4.1 的 compressor state 组是 AttentionSpec 子类（AscendCircularBufferSpec），
    # 由 FullAttentionManager 系列管理，语义上等价于无滑窗的 full attention。
    assert isinstance(kv_cache_spec, AttentionSpec), (
        f"[D2_offload] 未知的 offload group spec：{type(kv_cache_spec).__name__}"
    )
    return None


def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
) -> bool:
    """Return whether an SWA chunk can participate in an external-cache hit."""
    if alignment_chunk_count is None:
        return True
    assert sliding_window_chunks is not None
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(
        alignment_chunk_count, storable_chunk_count - segment_start
    )
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    return position_in_segment >= actual_segment_length - reachable_tail


# --------------------------------------------------------------------------- #
# [SWA_trim] 实验开关（只影响"存什么"，不改命中判定）
#
#   SWA_TRIM=off      D2 行为（= 上游 `is_store_reachable_swa_chunk` 原样）
#   SWA_TRIM=window   每个 SWA 组每个请求**只存窗口 chunk**
#                     （= 本批 store 的最后一个 chunk，覆盖请求尾部那 128 token 的窗口）
#
# 为什么第二种是"实验"而不是"修复"：见 logs/017 —— window chunk 只是"命中长度恰好
# 等于本请求长度"时被 `_sliding_window_lookup` 查询的那一块；若另一个请求命中更短的
# 前缀（前进方向上的对齐点），它会去查更早的那一块，被裁掉就会让它整轮 0 命中。
# --------------------------------------------------------------------------- #
_SWA_TRIM = os.environ.get("SWA_TRIM", "off")


def _swa_trim_keep_chunk(
    group_config: GroupOffloadConfig,
    abs_chunk_idx: int,
    storable_chunk_count: int,
    start_chunk_idx: int,
) -> bool:
    """[SWA_trim] store 侧裁剪的统一入口（默认逐字等价于上游/D2 行为）。"""
    if (
        _SWA_TRIM == "window"
        and group_config.offload_participating
        and group_config.sliding_window_size_in_chunks is not None
        and not group_config.is_eagle_group
    ):
        # 只留"本批的最后一个 chunk"= 覆盖请求当前尾部窗口的那一块。
        del start_chunk_idx  # 仅用于下一条分支的可读性
        return abs_chunk_idx == storable_chunk_count - 1
    return is_store_reachable_swa_chunk(
        abs_chunk_idx,
        storable_chunk_count,
        group_config.alignment_chunk_count,
        group_config.sliding_window_size_in_chunks,
        group_config.is_eagle_group,
    )


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded chunk size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    offload_prompt_only: bool
    # [SWA_pergroup] True ⇒ `blocks_per_chunk` 是 per-group 的（unit 池模式）。
    per_group_bpc: bool = False

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # [SWA_pergroup] per-group `blocks_per_chunk`。
        #   bpc_map 为 None 时逐字回退到镜像内行为（全局标量 `spec.blocks_per_chunk`）。
        bpc_map = bpc_map_from_extra(spec.extra_config)

        def bpc_of(idx: int, kv_spec: KVCacheSpec) -> int:
            if bpc_map is None:
                return int(spec.blocks_per_chunk)
            return int(bpc_map.get(idx, 1))

        def bpc_for(idx: int) -> int:
            return bpc_of(idx, kv_cache_config.kv_cache_groups[idx].kv_cache_spec)

        logger.info(
            "[SWA_pergroup] per-group bpc=%s（spec.blocks_per_chunk=%s，"
            "unit 模式=%s）",
            bpc_map if bpc_map is None else {k: v for k, v in sorted(bpc_map.items())},
            spec.blocks_per_chunk,
            bpc_map is not None,
        )
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        # [D2_offload] group 清单（第一手证据，进 serve.log；纯日志，无行为改变）
        logger.info(
            "[D2_offload] KV 卸载 group 清单 n=%d: %s",
            len(kv_cache_config.kv_cache_groups),
            [
                (
                    idx,
                    type(g.kv_cache_spec).__name__,
                    getattr(g.kv_cache_spec, "block_size", None),
                    len(g.layer_names),
                    type(_effective_offload_spec(g.kv_cache_spec)).__name__,
                    _offload_participates(g.kv_cache_spec),
                )
                for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            ],
        )
        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            if not _offload_participates(kv_spec):
                # [D2_offload] 不参与卸载的组不进 alignment 统计：
                # 否则 {1024, 256} 会让 alignment_tokens=None（策略退化）。
                continue
            _bpc = bpc_of(idx, kv_spec)
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * _bpc
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * _bpc)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        _fa_snapshot = sorted(full_attn_tokens_per_chunk)  # [SWA_trim] 日志用
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()
        # [SWA_trim] 判据 A 的第一手证据：alignment 到底算出来了没有。
        logger.info(
            "[SWA_trim] alignment: blocks_per_chunk=%d "
            "full_attn_tokens_per_chunk=%s alignment_tokens=%s",
            spec.blocks_per_chunk,
            _fa_snapshot,
            alignment_tokens,
        )

        def _alignment_chunk_count(
            tokens_per_chunk: int,
            sliding_window_size_in_chunks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_chunks is None:
                return None
            if alignment_tokens <= tokens_per_chunk:
                return None
            per_segment = alignment_tokens // tokens_per_chunk
            if sliding_window_size_in_chunks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing chunk of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        kv_group_configs = tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * bpc_for(idx),
                    hashes_per_chunk=(
                        (tokens_per_block * bpc_for(idx))
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=(
                        sw := get_sliding_window_size_in_chunks(
                            kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            tokens_per_block * bpc_for(idx),
                        )
                    ),
                    alignment_chunk_count=_alignment_chunk_count(
                        tokens_per_block * bpc_for(idx),
                        sw,
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(
                        kv_cache_config.kv_cache_groups[idx]
                    ),
                    is_eagle_group=idx in eagle_groups,
                    offload_participating=_offload_participates(
                        kv_cache_config.kv_cache_groups[idx].kv_cache_spec
                    ),
                    blocks_per_chunk=bpc_for(idx),
                )
                for idx, tokens_per_block in enumerate(spec.tokens_per_block)
            )
        # [SWA_trim] 判据 A 的第二半：每个组的 tpc / 窗口 / 裁剪力度。
        #   alignment_chunk_count = 该组每"full-attention 对齐段"里**可达的尾部 chunk 数上限**；
        #   None ⇒ 该组不做裁剪（全存）。
        logger.info(
            "[SWA_trim] SWA_TRIM=%s group 表 "
            "(idx, tokens_per_block, tokens_per_chunk, bpc, sw_chunks, "
            "alignment_chunk_count, is_eagle, participates): %s",
            _SWA_TRIM,
            [
                (
                    c.group_idx,
                    c.tokens_per_block,
                    c.tokens_per_chunk,
                    c.blocks_per_chunk,
                    c.sliding_window_size_in_chunks,
                    c.alignment_chunk_count,
                    c.is_eagle_group,
                    c.offload_participating,
                )
                for c in kv_group_configs
            ],
        )
        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=max(
                (c.blocks_per_chunk for c in kv_group_configs), default=1
            ),
            offload_prompt_only=spec.offload_prompt_only,
            per_group_bpc=bpc_map is not None,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if not group_config.offload_participating:
                # [D2_offload] 非参与组不生成 offload key（发了也永远查不中）
                continue
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated leading offloaded chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        The exclusion must be applied consistently everywhere
        ``next_stored_chunk_idx`` is derived: otherwise the trailing chunk of
        each step is skipped on collection but jumped over by
        ``next_stored_chunk_idx``, so it is never re-considered and a
        permanent hole breaks prefix-reuse lookup.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        if group_config.is_eagle_group and is_decoding:
            num_chunks = max(0, num_chunks - 1)
        # [SWA_pergroup] 每组的 chunk = blocks_per_chunk 个 GPU block。
        num_allocated_chunks = (
            len(group_state.block_ids) // group_config.blocks_per_chunk
        )
        return min(num_chunks, num_allocated_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        load_tier_filter=load_filter,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if not group_config.offload_participating:
                # [D2_offload] 核心修复：不参与卸载的组既不查也不进 lookup 组列表；
                # 否则它的 key 永远 MISS => _lookup() 直接 return 0 => 全轮零命中。
                continue
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # [D2_offload] 卸载参与组清单（第一手证据）
        logger.info(
            "[D2_offload] 参与卸载的组：full_attention=%s sliding_window=%s；"
            "被排除的组=%s",
            full_attention_groups,
            sliding_window_groups,
            [
                idx
                for idx, group_config in enumerate(self.config.kv_group_configs)
                if not group_config.offload_participating
            ],
        )

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
        # [D2_offload] 诊断计数器（只影响日志）
        self._dbg_store_lines = 0
        self._dbg_lookup_lines = 0
        self._dbg_miss_lines = 0
        self._dbg_store_key_lines = 0
        self._dbg_load_lines = 0
        self._dbg_tally: dict[int, int] = {}
        # 命中判定里第一个被咨询的组（用于每个请求只打一行摘要）
        self._dbg_first_group = self._lookup_groups[0] if self._lookup_groups else -1

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _calc_num_offloadable_tokens(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> int:
        num = min(num_computed_tokens, req_status.req.num_tokens)
        max_offload_tokens = req_status.max_offload_tokens
        if max_offload_tokens is not None:
            num = min(num, max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, req_status.req.num_prompt_tokens)
        return num

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for local_idx, key in enumerate(keys):
            result = self.manager.lookup(key, req_context)
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # Keep only chunks needed to hit the original request, plus
                # decoded chunks.
                chunks_to_skip = max(
                    0,
                    group_state.num_hit_chunks
                    - group_config.sliding_window_size_in_chunks,
                )
                self.manager.touch(
                    group_state.offload_keys[chunks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )

                # For eagle groups, query one extra chunk that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_chunks is not None:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if (
                    num_hit_chunks == 0
                    and group_idx == self._dbg_first_group
                    and self._dbg_lookup_lines < 80
                ):
                    self._dbg_lookup_lines += 1
                    logger.info(
                        "[D2_offload] lookup-summary req=%s group=%d keys=%d "
                        "hit=%s max_hit=%d tally=%s",
                        req_status.req.request_id,
                        group_idx,
                        len(offload_keys),
                        num_hit_chunks,
                        max_hit_size_tokens,
                        sorted(self._dbg_tally.items()),
                    )
                if num_hit_chunks == 0 and self._dbg_miss_lines < 24:
                    # ★ 判别实验：把该组**全部** key 都查一遍，数一数有多少
                    # 其实躺在池子里。present>0 但从头扫描 MISS ⇒ LRU 把前缀链
                    # 头部挤掉了（全有或全无）；present==0 ⇒ key 根本不存在。
                    self._dbg_miss_lines += 1
                    _all_keys = req_status.group_states[group_idx].offload_keys
                    _present = sum(
                        1
                        for _k in _all_keys
                        if self.manager.lookup(_k, req_status.req_context)
                        == LookupResult.HIT
                    )
                    logger.info(
                        "[D2_offload] miss-scan req=%s group=%d scanned=%d "
                        "present=%d head_key=%s chunk_bytes=%d",
                        req_status.req.request_id,
                        group_idx,
                        len(_all_keys),
                        _present,
                        _all_keys[0][:8].hex() if _all_keys else "-",
                        group_config.tokens_per_chunk,
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            num_hit_tokens = self._lookup(req_status)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        # [SWA_pergroup] 每组的 (组内 key 序号, chunk 内 unit 序号) 序列：
        # 把 manager 返回的 chunk→unit id 展开成"一个 GPU block 一个 id"。
        # 标量模式下这些序列不参与（per_group_bpc=False）。
        group_unit_pairs: list[list[tuple[int, int]]] = []
        group_num_keys: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            if not group_config.offload_participating:
                # [D2_offload] 非参与组不进 keys_to_load，也不占 dst block 预算。
                # load spec 仍保留该组占位（worker 断言 len(group_sizes) == 组数），
                # worker 对 group_size==0 会 continue，src/dst 偏移对齐不受影响。
                group_sizes.append(0)
                block_indices.append(0)
                group_unit_pairs.append([])
                group_num_keys.append(0)
                continue
            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            bpc_g = group_config.blocks_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks * bpc_g + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            assert len(offload_keys) >= num_chunks
            pairs: list[tuple[int, int]] = []
            n_keys = 0
            if num_pending_gpu_blocks:
                start_chunk_idx = num_locally_computed_gpu_blocks // bpc_g
                keys_to_load.extend(offload_keys[start_chunk_idx:num_chunks])
                n_keys = num_chunks - start_chunk_idx
                # pending 的第一个 block 在该组 block 序列里的序号 = computed_gpu；
                # 它的 chunk 序号 = pos // bpc_g，chunk 内偏移 = pos % bpc_g。
                for j in range(num_pending_gpu_blocks):
                    pos = num_locally_computed_gpu_blocks + j
                    pairs.append((pos // bpc_g - start_chunk_idx, pos % bpc_g))
            group_unit_pairs.append(pairs)
            group_num_keys.append(n_keys)

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            if self._dbg_load_lines < 40:
                self._dbg_load_lines += 1
                logger.info(
                    "[D2_offload] load-probe req=%s group=%d part=%s "
                    "pending_gpu=%d computed_gpu=%d num_chunks=%d keys=%d",
                    request.request_id,
                    group_config.group_idx,
                    group_config.offload_participating,
                    num_pending_gpu_blocks,
                    num_locally_computed_gpu_blocks,
                    num_chunks,
                    len(keys_to_load),
                )
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit chunks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # chunks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_chunk_idx = num_chunks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        # [SWA_pergroup] unit 模式：把"一个 key 一个 chunk id" 展开成
        # "一个 pending GPU block 一个 unit id"（worker 侧 blocks_per_chunk=1）。
        if self.config.per_group_bpc:
            _flat = list(src_spec.block_ids)
            _out_ids: list[int] = []
            _base = 0
            for _gcfg, _pairs, _n_keys in zip(
                self.config.kv_group_configs, group_unit_pairs, group_num_keys
            ):
                _bpc = _gcfg.blocks_per_chunk
                for _krel, _off in _pairs:
                    _out_ids.append(_flat[_base + _krel * _bpc + _off])
                _base += _n_keys * _bpc
            assert len(_out_ids) == len(dst_block_ids), (
                "[SWA_pergroup] load spec 展开后与 GPU 侧 block 数不一致："
                f"cpu_ids={len(_out_ids)} gpu_blocks={len(dst_block_ids)}"
            )
            assert _base == len(_flat), (
                "[SWA_pergroup] load spec 未把 manager 给的 unit 全部覆盖："
                f"used={_base} total={len(_flat)}"
            )
            src_spec = CPULoadStoreSpec(_out_ids)
        logger.info(
            "[D2_offload] load job req=%s keys=%d group_sizes=%s "
            "src_blocks=%d dst_blocks=%d",
            request.request_id,
            len(keys_to_load),
            group_sizes,
            len(src_spec.block_ids),
            len(dst_block_ids),
        )
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    # [SWA_pergroup] 滑窗组的 chunk 粒度是 per-group 的。
                    start = (
                        group_state.next_stored_chunk_idx
                        * self.config.kv_group_configs[grp_idx].blocks_per_chunk
                    )
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        # [SWA_pergroup] 每个组自己的 blocks_per_chunk（标量模式下全部相同）。
        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        ):
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
                num_tokens_after_batch = req.num_computed_tokens
            elif req.is_finished():
                num_tokens_after_batch = req.num_tokens
            else:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

            num_offloadable_tokens = self._calc_num_offloadable_tokens(
                req_status, num_tokens_after_batch
            )

            # Filter out chunks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if not group_config.offload_participating:
                    # [D2_offload] 非参与组不参与 store（key 列表为空，这里显式跳过）
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )
                if self._dbg_store_lines < 45:
                    self._dbg_store_lines += 1
                    logger.info(
                        "[D2_offload] store-probe req=%s group=%d part=%s "
                        "num_chunks=%d gpu_blocks=%d offload_keys=%d",
                        req_id,
                        group_config.group_idx,
                        group_config.offload_participating,
                        num_chunks,
                        len(group_state.block_ids),
                        len(group_state.offload_keys),
                    )

                start_chunk_idx = group_state.next_stored_chunk_idx
                if num_chunks <= start_chunk_idx:
                    continue
                offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
                # For each chunk, take the last corresponding GPU block. For
                # blocks_per_chunk=3 and GPU block IDs 1 5 6 7 2 4 9 3 8,
                # this selects GPU blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                # [M_bpcfix] 局部名从 `blocks_per_chunk` 改成 `bpc_g`：
                # 原名字在**下一个 loop（spec loop）里被再次使用**，而那里没有
                # 重新赋值 ⇒ 读到的是"最后一个参与组"的残留值（本配置 = SWA 的 1）
                # ⇒ full 组（bpc=8）的 gpu_block_idx/range 按 1 展开、只搬 1/8。
                bpc_g = group_config.blocks_per_chunk  # [SWA_pergroup]
                offload_block_ids = group_state.block_ids[
                    start_chunk_idx * bpc_g
                    + bpc_g
                    - 1 : num_chunks * bpc_g : bpc_g
                ]
                assert len(offload_keys) == len(offload_block_ids)

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA chunks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing chunks queried by _sliding_window_lookup are
                    # reachable. EAGLE/MTP requires one additional chunk that
                    # lookup later drops as its volatile draft tail.
                    abs_chunk_idx = start_chunk_idx + key_idx
                    if not _swa_trim_keep_chunk(
                        group_config,
                        abs_chunk_idx,
                        num_chunks,
                        start_chunk_idx,
                    ):
                        continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            # [SWA_pergroup] manager 给每个 key 的 `bpc_g` 个 unit id
            # （顺序 = chunk 内 block 顺序）；下面按"I 号 block 用第 I 个 unit"展开。
            _flat_units = list(store_output.store_spec.block_ids)
            key_units: dict[OffloadKey, list[int]] = {}
            _pos = 0
            for _k in store_output.keys_to_store:
                _n = self.config.kv_group_configs[
                    get_offload_group_idx(_k)
                ].blocks_per_chunk
                key_units[_k] = _flat_units[_pos : _pos + _n]
                _pos += _n

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            dst_unit_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            # [M_bpcfix] 四件套探针：在 spec loop 内按组记录"真正用到的值"。
            _bpcfix_rows: list[tuple[bool, dict[str, int]]] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                if not group_config.offload_participating:
                    # [D2_offload] 保持 group_sizes/block_indices 与 13 个组一一对应
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )
                start_chunk_idx = group_state.next_stored_chunk_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx

                    if self._dbg_store_key_lines < 12:
                        self._dbg_store_key_lines += 1
                        logger.info(
                            "[D2_offload] store-key req=%s group=%d chunk=%d key=%s",
                            req_id,
                            group_config.group_idx,
                            chunk_idx,
                            offload_key[:8].hex(),
                        )
                    self._dbg_tally[group_config.group_idx] = (
                        self._dbg_tally.get(group_config.group_idx, 0) + 1
                    )
                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    # ★ [M_bpcfix] spec loop 内**必须**重新取本组自己的 bpc：
                    # 收集 loop 里的同名变量是逐组赋值的，loop 结束后停在最后一个
                    # 参与组（本配置 = SWA，bpc=1）⇒ 不重新取值的话 gpu_block_idx 与
                    # range() 都按 1 展开，full 组（bpc=8）每个 chunk 只搬 1 个 block
                    # （其余 7 个 unit 永远不写 ⇒ load 侧读到全 0 行）。
                    bpc_g = group_config.blocks_per_chunk
                    # fail-closed 自检：spec loop 内取到的必须是本组自己的 bpc。
                    assert bpc_g == group_config.blocks_per_chunk, (
                        "[M_bpcfix] bpc 泄漏：loop 内的 %d != group_config 的 %d"
                        "（group=%d）"
                        % (
                            bpc_g,
                            group_config.blocks_per_chunk,
                            group_config.group_idx,
                        )
                    )
                    gpu_block_idx = chunk_idx * bpc_g
                    _units = key_units.get(offload_key, [])
                    if self.config.per_group_bpc:
                        # unit id 表按"I 号 block 用第 I 个 unit"展开 ⇒ 两者必须同长。
                        assert len(_units) == bpc_g, (
                            "[M_bpcfix] unit 数与 bpc 不一致：units=%d bpc=%d"
                            "（group=%d chunk=%d key=%s）"
                            % (
                                len(_units),
                                bpc_g,
                                group_config.group_idx,
                                chunk_idx,
                                offload_key[:8].hex(),
                            )
                        )
                    _BPCFIX_PROBE.record(
                        is_sliding_window,
                        {
                            "group": group_config.group_idx,
                            "true_bpc": group_config.blocks_per_chunk,
                            "gpu_block_idx": gpu_block_idx,
                            "chunk": chunk_idx,
                        },
                        _bpcfix_rows,
                    )
                    for i in range(bpc_g):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        if self.config.per_group_bpc:
                            dst_unit_ids.append(_units[i])
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )

            _BPCFIX_PROBE.log_job(
                req_id, _bpcfix_rows, len(src_block_ids), group_sizes
            )
            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            if self.config.per_group_bpc:
                assert len(dst_unit_ids) == len(src_block_ids), (
                    "[SWA_pergroup] store spec 展开后与 GPU 侧 block 数不一致："
                    f"cpu_ids={len(dst_unit_ids)} gpu_blocks={len(src_block_ids)}"
                )
                dst_spec = CPULoadStoreSpec(dst_unit_ids)
            else:
                dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

            if req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in non_sliding_window_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req_status.finished_signaled = True
            self.manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return bool(self._jobs) or self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
        # create store jobs for the last block(s) on the next schedule step.
        req_status.update_offload_keys()

        # Keep req_status alive: _build_store_jobs will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register non_sliding_window_block_ids so future block reuse triggers
        # a flush via _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from chunk 0.
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
