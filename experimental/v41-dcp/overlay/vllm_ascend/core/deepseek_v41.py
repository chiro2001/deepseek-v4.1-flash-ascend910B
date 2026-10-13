# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and layer-outermost hybrid allocation."""

import os
from dataclasses import dataclass, replace

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.circular_buffer import AscendCircularBufferSpec
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

STATE_RING_ROWS = 32


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(AscendMLAAttentionSpec):
    """INT8 index keys followed by FP16 scales inside each shared slot page."""

    # =====================================================================
    # [V41-DCP 2026-09-29] indexer K cache 的**复制度**。
    #
    # 为什么必须复制：全局 top-k 必须每个 rank 完全一致（否则各 rank 的
    # partial attention 覆盖的键集不同，LSE 合并出来的不是全局 softmax）。
    # A3 上没有任何"分布式 top-k"通道，唯一可行的做法是让每个 rank 都能看到
    # **全量** indexer K（与 SFA-DCP 同构）。代价见
    # `docs/V41-DCP-PROGRESS-20260929.md`：pool 页 +22%，容量 7.88× → ~6.2×。
    #
    # 放在 spec 上（而不是进程内全局）是因为：
    #   · `_cache_plane_sizes` 在 **EngineCore**（调度器）侧被调用来算池子大小；
    #   · `reshape_cache` 在 **worker** 侧被调用来建 view；
    #   · 两边是不同进程，spec 会从 worker 序列化过去。
    #   2026-09-29 实测教训：用进程内全局时 EngineCore 读到 1。
    # =====================================================================
    dcp_world_size: int = 1

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41DraftSWASpec(AscendSlidingWindowMLASpec):
    """DSpark SWA owned by G12, aliasing target slots at distinct block IDs."""

    def __post_init__(self):
        if self.dtype != torch.bfloat16 or self.num_kv_heads != 1 or self.compress_ratio != 1:
            raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41DraftSWASpec)
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(AscendCircularBufferSpec):
    """One private FP32 KV/score ring page for each active request."""

    compress_ratio: int = 1

    def __post_init__(self):
        if self.dtype != torch.float32 or self.block_size != STATE_RING_ROWS or self.compress_ratio != 1:
            raise ValueError("Aurora state requires a 32-row FP32 uncompressed ring")
        if self.num_kv_heads != 1:
            raise ValueError("Aurora state requires one packed KV/score plane")


def is_v41_spec(spec):
    return isinstance(
        spec,
        (
            DeepseekV41FullSpec,
            DeepseekV41IndexerSpec,
            DeepseekV41SWASpec,
            DeepseekV41DraftSWASpec,
            DeepseekV41CompressorStateSpec,
        ),
    )


def _uniform(members, label):
    if not members:
        raise ValueError(f"V4.1 cache group {label} is empty")
    uniform = UniformTypeKVCacheSpecs.from_specs(members)
    if uniform is None:
        raise ValueError(f"Incompatible V4.1 resource layouts in {label}")
    return uniform


@dataclass(frozen=True)
class CachePlacement:
    name: str
    offset: int
    page_size_bytes: int


@dataclass(frozen=True)
class CacheSlot:
    page_size_bytes: int
    placements: tuple[CachePlacement, ...]


def _layer_number(name):
    try:
        return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid V4.1 cache resource name: {name}") from exc


def _cache_plane_sizes(spec):
    rows = spec.storage_block_size * spec.num_kv_heads
    key_bytes = rows * spec.head_size * spec.dtype.itemsize
    if isinstance(spec, DeepseekV41IndexerSpec):
        # [V41-DCP] 复制态：整条序列每个 rank 各留一份 ⇒ 键与 scale 两个面都 ×dcp。
        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer

        repl = max(1, int(getattr(spec, "dcp_world_size", 1) or 1)) if replicate_indexer() else 1
        return key_bytes * repl, rows * repl * spec.scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _draft_layer_number(name):
    try:
        return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Aurora DSpark cache resource name: {name}") from exc


def plan_cache_slots(specs):
    """Place source KV/index tuples, state and SWA in four shared layer slots.

    Sizes come from payloads, never previously padded specs. Different groups
    overlay a slot at distinct live block IDs; a source's KV and index share
    the same ID at disjoint offsets within its page.
    """
    if not all(is_v41_spec(spec) for spec in specs.values()):
        raise ValueError(
            "V4.1 requires explicit target or Aurora DSpark cache specs; foreign resources are unsupported"
        )
    full = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41FullSpec)), key=_layer_number)
    state = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41CompressorStateSpec)), key=_layer_number)
    swa = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    draft = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41DraftSWASpec)), key=_draft_layer_number)
    if draft and list(map(_draft_layer_number, draft)) != [0, 1, 2]:
        raise ValueError("Aurora DSpark requires exactly three ordered draft layers: mtp.0, mtp.1, mtp.2")
    if list(map(_layer_number, full)) != [2, 8, 14, 20]:
        raise ValueError("V4.1 requires KV source layers 2, 8, 14, 20")
    if list(map(_layer_number, state)) != [2, 8, 14]:
        raise ValueError("V4.1 requires state source layers 2, 8, 14")
    if list(map(_layer_number, swa)) != list(range(40)):
        raise ValueError("V4.1 requires exactly 40 ordered SWA resources")

    slots = []
    _state_aliased = set()
    for slot_idx, kv_name in enumerate(full):
        prefix, suffix = kv_name.rsplit(".", 1)
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs.get(index_name)
        kv_spec = specs[kv_name]
        ratio = 2 if slot_idx < len(state) else 1
        if (
            suffix != "long_kv_cache"
            or not isinstance(index_spec, DeepseekV41IndexerSpec)
            or kv_spec.compress_ratio != ratio
            or index_spec.compress_ratio != ratio
            or kv_spec.block_size != index_spec.block_size
        ):
            raise ValueError(f"V4.1 source {prefix} has incompatible KV/index specs")
        # =====================================================================
        # ★★★★★★ [V41-DCP-STATE-SLOT 2026-09-30 13:45] **state ring 何时可以别名**。
        #
        # 原实现无条件把 `state[slot_idx]` 别名进 slot `slot_idx`，并依赖
        # `capacity = max(kv+idx, state, swa)`。这在**不复制**时成立：
        #     kv+idx = 65536+8320 = 73856 ≤ state 131072 = swa 131072
        # ⇒ slot = 131072，`reshape_cache` 的
        #   `sum(plane_sizes) == block_stride`（state 必须**填满**槽位）成立。
        #
        # 但 indexer 复制态（`V41_DCP_REPLICATE_INDEXER=1`）下
        #     kv+idx = 65536 + 8×8320 = 132096 > 131072
        # ⇒ slot 被顶到 132096，state（131072）**不再填满** ⇒ 真机报错
        #   `RuntimeError: Aurora circular state must fill its slot with 32
        #    contiguous FP32 rows`（2026-09-30 12:50 实测）。
        #
        # 该断言不能放宽：state 由 CANN `Compressor` 消费，代码里按
        #     `state_cache[block_table[b], pos % cache_size, :head_dim]` 与
        #     `state_cache[..., head_dim:]` 切片（最后维**必须**恰好 2·head_dim）
        # ⇒ 既不能拉长最后一维做 padding，也不能整块非连续。
        #
        # 因此：**state 放得下就继续别名（不复制时零成本）；放不下就给它一个
        # 独立槽**（容量恰为 state 平面大小，填满 ⇒ 断言仍成立）。
        # 代价：复制态下 pool 从 660480 → 791552 B/块（容量约 −18%），这是
        # "indexer 必须全局可见"的结构性成本，已记录在 RCA 文档。
        # =====================================================================
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        _kv_idx_bytes = kv_bytes + index_bytes
        _state_here = state[slot_idx] if slot_idx < len(state) else None
        _state_bytes_here = (
            sum(_cache_plane_sizes(specs[_state_here])) if _state_here is not None else 0
        )
        _state_fits = _state_bytes_here > 0 and _kv_idx_bytes <= _state_bytes_here
        aliases = ([_state_here] if _state_fits else []) + swa[slot_idx :: len(full)]
        if _state_fits and _state_here is not None:
            _state_aliased.add(_state_here)
        capacity = max(_kv_idx_bytes, *(sum(_cache_plane_sizes(specs[n])) for n in aliases))
        if slot_idx < len(draft):
            draft_name = draft[slot_idx]
            draft_spec = specs[draft_name]
            swa_spec = specs[swa[slot_idx]]
            if (
                draft_spec.block_size != swa_spec.block_size
                or draft_spec.head_size != swa_spec.head_size
                or draft_spec.sliding_window != swa_spec.sliding_window
                or sum(_cache_plane_sizes(draft_spec)) > capacity
            ):
                raise ValueError("Aurora DSpark geometry must match target SWA and fit its existing slot")
            aliases.append(draft_name)
        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))
    # ★ [V41-DCP-STATE-SLOT] 没被别名进 KV 槽的 state ring ⇒ 独立槽（恰为其平面大小）
    _orphan_state = [n for n in state if n not in _state_aliased]
    if _orphan_state:
        _state_slot_capacity = max(sum(_cache_plane_sizes(specs[n])) for n in _orphan_state)
        if any(sum(_cache_plane_sizes(specs[n])) != _state_slot_capacity for n in _orphan_state):
            raise ValueError("V4.1 state ring slots must be uniform when split out")
        slots.append(
            CacheSlot(
                _state_slot_capacity,
                tuple(CachePlacement(n, 0, _state_slot_capacity) for n in _orphan_state),
            )
        )
    names = [p.name for slot in slots for p in slot.placements]
    if len(names) != len(set(names)) or set(names) != set(specs):
        raise ValueError("V4.1 slot placement must cover each resource exactly once")
    return tuple(slots)


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if not any(is_v41_spec(s) for s in specs.values()):
        return None
    slots = plan_cache_slots(specs)
    padded = {
        p.name: replace(specs[p.name], page_size_padded=p.page_size_bytes) for slot in slots for p in slot.placements
    }
    full = {n: s for n, s in padded.items() if isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))}
    state = {n: s for n, s in padded.items() if isinstance(s, DeepseekV41CompressorStateSpec)}
    groups = [_uniform(full, "full"), _uniform(state, "state")]
    swa = sorted((n for n, s in padded.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    groups.extend(
        _uniform({n: padded[n] for n in swa[start : start + len(slots)]}, f"swa{start}")
        for start in range(0, len(swa), len(slots))
    )
    draft = sorted((n for n, s in padded.items() if isinstance(s, DeepseekV41DraftSWASpec)), key=_draft_layer_number)
    if draft:
        groups.append(_uniform({n: padded[n] for n in draft}, "dspark"))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def has_v41_groups(groups):
    return any(
        is_v41_spec(s)
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        for s in g.kv_cache_spec.kv_cache_specs.values()
    )


def cache_slots_from_groups(groups):
    specs = {}
    for group in groups:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("V4.1 requires uniform-type cache groups")
        for name in group.layer_names:
            if name in specs:
                raise ValueError(f"V4.1 resource belongs to multiple cache groups: {name}")
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return plan_cache_slots(specs)


def pool_bytes_per_block(groups):
    return sum(slot.page_size_bytes for slot in cache_slots_from_groups(groups))


def request_blocks(vllm_config, groups):
    # Different logical groups consume different IDs in one global block pool.
    per_group = [
        max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes
            for s in g.kv_cache_spec.kv_cache_specs.values()
        )
        for g in groups
    ]
    # [V41-DCP-DIAG] 块账插桩：DCP 下"每个请求要多少块、池子里有多少块"是容量的全部秘密，
    # 而它分散在 4 个 spec 类的 max_memory_usage_bytes 里，事后无法从日志倒推。
    # 只在 `V41_DCP_DIAG=1` 时打印（默认零开销、零日志噪音）。
    if os.environ.get("V41_DCP_DIAG") == "1":
        _log_block_accounting(vllm_config, groups, per_group)
    return sum(per_group)


def _log_block_accounting(vllm_config, groups, per_group):
    """Print the exact per-group block accounting for the V4.1 global pool."""
    import logging

    logger = logging.getLogger(__name__)
    parallel = vllm_config.parallel_config
    slots = cache_slots_from_groups(groups)
    pool_bytes = sum(slot.page_size_bytes for slot in slots)
    lines = [
        "[V41-DCP-DIAG] pool_bytes_per_block=%d (slots=%s)"
        % (pool_bytes, [s.page_size_bytes for s in slots]),
        "[V41-DCP-DIAG] dcp=%d tp=%d block_size=%d interleave=%d max_model_len=%d"
        % (
            parallel.decode_context_parallel_size,
            parallel.tensor_parallel_size,
            vllm_config.cache_config.block_size,
            parallel.cp_kv_cache_interleave_size,
            vllm_config.model_config.max_model_len,
        ),
        "[V41-DCP-DIAG] max_in_flight_tokens=%s" % getattr(vllm_config, "max_in_flight_tokens", "n/a"),
    ]
    for group, blocks in zip(groups, per_group):
        kinds = {}
        for name, spec in group.kv_cache_spec.kv_cache_specs.items():
            kinds.setdefault(type(spec).__name__, []).append(name)
        detail = []
        for spec in group.kv_cache_spec.kv_cache_specs.values():
            mem = spec.max_memory_usage_bytes(vllm_config)
            detail.append(
                "%s: mem=%d page=%d blocks=%d"
                % (type(spec).__name__, mem, spec.page_size_bytes, (mem + spec.page_size_bytes - 1) // spec.page_size_bytes)
            )
        lines.append(
            "[V41-DCP-DIAG] group(block_size=%d, layers=%d, %s) -> %d blocks | %s"
            % (group.kv_cache_spec.block_size, len(group.layer_names), "+".join(sorted(kinds)), blocks, "; ".join(detail))
        )
    lines.append("[V41-DCP-DIAG] request_blocks(total)=%d" % sum(per_group))
    logger.warning("\n".join(lines))


def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    return num_blocks, [
        KVCacheTensor(
            size=num_blocks * slot.page_size_bytes,
            shared_by=[p.name for p in slot.placements],
            block_stride=slot.page_size_bytes,
        )
        for slot in slots
    ]


def reshape_cache(raw: torch.Tensor, spec, *, num_blocks, offset, block_stride):
    """Create typed per-page views using the containing slot's physical stride."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("V4.1 cache requires contiguous one-dimensional uint8 storage")
    if num_blocks <= 0 or block_stride <= 0 or raw.numel() != num_blocks * block_stride:
        raise ValueError("V4.1 cache backing does not match its declared layout")
    plane_sizes = _cache_plane_sizes(spec)
    if offset < 0 or offset + sum(plane_sizes) > block_stride:
        raise ValueError("V4.1 cache component exceeds its slot page")
    if isinstance(spec, DeepseekV41CompressorStateSpec) and sum(plane_sizes) != block_stride:
        raise ValueError("Aurora circular state must fill its slot with 32 contiguous FP32 rows")

    # [V41-DCP] 复制态的 indexer 平面在**同一个物理块**里放 `dcp` 份子块，
    # 所以行数（token 槽位）要乘 dcp。列/行 stride 不变（子块在块内连续）。
    from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer

    index_rows = (
        spec.storage_block_size * max(1, int(getattr(spec, "dcp_world_size", 1) or 1))
        if isinstance(spec, DeepseekV41IndexerSpec) and replicate_indexer()
        else spec.storage_block_size
    )

    def view(dtype, width, byte_offset, rows=None):
        dtype_size = dtype.itemsize
        storage_offset = raw.storage_offset() + byte_offset
        if storage_offset % dtype_size or block_stride % dtype_size or raw.numel() % dtype_size:
            raise ValueError("V4.1 cache offset/stride is not dtype aligned")
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, rows if rows is not None else spec.storage_block_size, spec.num_kv_heads, width),
            stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, spec.head_size, offset)
    if isinstance(spec, DeepseekV41IndexerSpec):
        return (
            view(spec.dtype, spec.head_size, offset, rows=index_rows),
            view(spec.scale_dtype, spec.scale_dim, offset + plane_sizes[0], rows=index_rows),
        )
    return key


def validate_cache_runtime(vllm_config):
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError("V4.1 cache initialization currently requires model runner V1")
    cudagraph_mode = getattr(
        vllm_config.compilation_config,
        "cudagraph_mode",
        CUDAGraphMode.NONE if vllm_config.model_config.enforce_eager else CUDAGraphMode.FULL,
    )
    if cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise NotImplementedError("V4.1 currently supports only eager or FULL_DECODE_ONLY graph mode")
    speculative = vllm_config.speculative_config
    if speculative is not None:
        use_dspark = getattr(speculative, "use_dspark", None)
        if not callable(use_dspark) or not use_dspark():
            raise NotImplementedError("Aurora supports only DSpark speculative decoding")
        # Verification writes the anchor and up to S speculative rows. After
        # rejection, the earliest needed residual is the verified anchor.
        # It must survive the final 32-row write: S must be strictly below 32.
        if not 0 < speculative.num_speculative_tokens < STATE_RING_ROWS:
            raise ValueError("Aurora DSpark requires 1..31 speculative tokens to preserve FP32 ring residuals")
        per_batch = getattr(speculative, "num_speculative_tokens_per_batch_size", None) or ()
        if any(not 0 <= count <= speculative.num_speculative_tokens for _, _, count in per_batch):
            raise ValueError("Aurora DSpark per-batch speculation must stay within the configured ring-safe maximum")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        # [V41-DCP 2026-09-29] 开发期放开 PP=DCP=PCP=1 这道门，两个档位：
        #   V41_DCP_ALLOW_CAPACITY_PROBE=1 —— 只测「DCP ⇒ KV 池 token 容量」这条内存账；
        #     attention 侧 DCP 执行路径尚未接入，输出**一定错**。
        #   V41_DCP=1 —— 真·DCP 开发档，后续各阶段补丁挂在这个标志下。
        import os as _os

        if _os.environ.get("V41_DCP_ALLOW_CAPACITY_PROBE") != "1" and _os.environ.get("V41_DCP") != "1":
            raise NotImplementedError("V4.1 initial runtime requires PP=DCP=PCP=1")
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "[V41-DCP] PP/DCP/PCP guard bypassed (pp=%s dcp=%s pcp=%s) — DEVELOPMENT BUILD",
            parallel.pipeline_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.prefill_context_parallel_size,
        )

    # [V41-DCP 注] `cp_kv_cache_interleave_size` 的强制与滑窗 admission cap 的
    # DCP 换算都在 `vllm_ascend/platform.py::_validate_parallel_config` 里做。
    # 这里**不要**重复做：worker 侧的 VllmConfig 与 EngineCore 侧是两份对象，
    # 只改 worker 那份不会影响调度器侧的容量核算（2026-09-29 实测踩过）。
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("V4.1 requires the hybrid KV cache manager")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("V4.1 initial cache layout requires BF16")
    if speculative is not None:
        # Aurora's planes are always BF16. Pin the inherited DSV4 draft
        # backend to the same layout, including on hardware where auto is FP8.
        vllm_config.cache_config.cache_dtype = "bfloat16"
