# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and layer-outermost hybrid allocation."""

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
        return key_bytes, rows * spec.scale_dim * spec.scale_dtype.itemsize
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
    _plan: list[dict] = []
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
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + swa[slot_idx :: len(full)]
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        _alias_max = max((sum(_cache_plane_sizes(specs[n])) for n in aliases), default=0)
        capacity = max(kv_bytes + index_bytes, _alias_max)
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
            _alias_max = max(_alias_max, sum(_cache_plane_sizes(draft_spec)))
        _plan.append(
            {
                "kv_name": kv_name,
                "index_name": index_name,
                "kv_bytes": kv_bytes,
                "index_bytes": index_bytes,
                "aliases": tuple(aliases),
                "alias_max": _alias_max,
                "capacity": capacity,
            }
        )
    # =========================================================================
    # ★ [KV32-SLOT-REPACK] index 平面可以挪到**别的 slot 的空闲区**。
    #
    # 动机（32 位页偏移）：`block_id × slot.page_size_bytes` 若 ≥ 2^32 会回绕 ⇒
    #   KV 静默串块（长上下文间歇性偏离，见 docs/KV32-OVERFLOW-DOSE-RESPONSE-*.md）。
    #   判据是**每个 slot 自己的页步长**：现状 ratio-1 槽 = KV 131072 + index 16640
    #   = 147712 ⇒ 块上限 floor(2^32/147712) = 29076；而其余三个槽被 DSpark draft
    #   顶到 131072 ⇒ 上限本可为 floor(2^32/131072) = 32768（**+12.7% 容量**）。
    #
    # 做法：若某槽的 (kv+index) 超过它的别名容量 `alias_max`（即 kv+index 才是
    #   binding 项），就把 index 平面挪到另一个槽的空闲区（该槽 capacity − 已用
    #   ≥ index 平面大小）；原槽容量降回 `alias_max`。
    #
    # 为什么安全：placement → (offset, stride=所在槽 page_size_bytes) 由本函数决定，
    #   `allocate_cache_config` 会把每个 placement 的名字登记到**所在槽**的 backing；
    #   `reshape_cache` 只要求 `offset + 平面大小 ≤ block_stride`。挪走后 index 的
    #   平面大小不变 ⇒ request_blocks / 每请求块需求不变。
    # =========================================================================
    _used = [p["kv_bytes"] + p["index_bytes"] for p in _plan]
    _extra: dict[int, list[tuple[int, int, int]]] = {}
    _moved: set[int] = set()
    for i, p in enumerate(_plan):
        if p["kv_bytes"] + p["index_bytes"] <= p["alias_max"]:
            continue  # index 不 binding，保持原样
        for j, q in enumerate(_plan):
            if j == i or j in _moved or j in _extra:
                continue
            if q["capacity"] - _used[j] >= p["index_bytes"]:
                _extra[j] = [(i, _used[j], p["index_bytes"])]
                _used[j] += p["index_bytes"]
                _moved.add(i)
                break
    for i, p in enumerate(_plan):
        _cap = max(p["kv_bytes"], p["alias_max"]) if i in _moved else p["capacity"]
        placements = [CachePlacement(p["kv_name"], 0, p["kv_bytes"])]
        if i not in _moved:
            placements.append(
                CachePlacement(p["index_name"], p["kv_bytes"], _cap - p["kv_bytes"])
            )
        placements.extend(CachePlacement(name, 0, _cap) for name in p["aliases"])
        for _src, _off, _sz in _extra.get(i, ()):
            placements.append(CachePlacement(_plan[_src]["index_name"], _off, _sz))
        slots.append(CacheSlot(_cap, tuple(placements)))
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
    return sum(
        max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes
            for s in g.kv_cache_spec.kv_cache_specs.values()
        )
        for g in groups
    )


def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    # =========================================================================
    # ★ [V41-KV32-CAP 2026-10-06] **上限语义**：块数取 min(自动 profiling 给出的块数, 32 位安全上限)。
    #
    # 为什么必须这样而不是"钉一个 KV_CACHE_MEMORY_BYTES"：
    #   * 回绕的判据是**每个 slot 自己的页步长**（block_id × slot.page_size_bytes ≥ 2³² 就串块），
    #     所以安全上限可以用**几何自己**算出来：floor(2³² / max_slot_page_size)；
    #   * 钉死一个字节值会在**显存更小的机器上直接 OOM**：实测 A3 每个 rank 可用 KV 是
    #     16.60 GiB，而 A2（910B3）只有 **14.40 GiB**（a2/docs/A2-DEPLOY-NOW.md §B0）⇒
    #     把 A3 上验证过的 16 GiB 常量搬到 A2 = 起不来；
    #   * 上限语义则天然自适应：小显存机器自动 profiling 给得少 ⇒ 取小值，不 OOM；
    #     大显存机器给得多 ⇒ 被夹到安全上限，不放行回绕。
    #
    # 关掉：`V41_KV_MAX_BLOCKS=off`（逃生口，**知道自己在越过 32 位边界**时才用）。
    # 收紧：`V41_KV_MAX_BLOCKS=<N>`（只会更小；写大于安全上限的值会被夹回安全上限）。
    # =========================================================================
    cap = _kv32_safe_blocks(slots)
    if cap is not None and num_blocks > cap:
        print(
            f"[V41-KV32-CAP] 块数 {num_blocks} → {cap}：自动 profiling 给得比 32 位安全上限多。"
            f" 每槽页步长 {[s.page_size_bytes for s in slots]}，上限 = floor(2^32 / {max(s.page_size_bytes for s in slots)})"
            f" = {cap}。要关掉本夹取设 V41_KV_MAX_BLOCKS=off（会放行块号回绕 ⇒ 静默读错）",
            flush=True,
        )
        num_blocks = cap
    return num_blocks, [
        KVCacheTensor(
            size=num_blocks * slot.page_size_bytes,
            shared_by=[p.name for p in slot.placements],
            block_stride=slot.page_size_bytes,
        )
        for slot in slots
    ]


def _kv32_safe_blocks(slots):
    """由几何算出的 32 位页偏移安全块数；返回 None = 明确关掉。

    判据取**所有 slot 里最紧的那个**（每个 slot 的块号都按自己的页步长参与寻址）。
    """
    import os

    raw = os.environ.get("V41_KV_MAX_BLOCKS", "auto").strip().lower()
    # 只有**显式**写 off 才关掉安全网：`0` 按"没设上限"处理（= auto），
    # 免得"手滑写 0"把 32 位回绕的保护静默摘掉。
    if raw in ("off", "none", "disable"):
        return None
    auto = min((1 << 32) // slot.page_size_bytes for slot in slots)
    if raw in ("", "auto", "0"):
        return auto
    try:
        want = int(raw)
    except ValueError as exc:
        raise ValueError(f"V41_KV_MAX_BLOCKS={raw!r} 非法（应为 auto / off / 正整数）") from exc
    if want <= 0:
        return auto
    return min(want, auto)  # 显式值只允许**更收紧**，不允许越过安全上限


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

    def view(dtype, width, byte_offset):
        dtype_size = dtype.itemsize
        storage_offset = raw.storage_offset() + byte_offset
        if storage_offset % dtype_size or block_stride % dtype_size or raw.numel() % dtype_size:
            raise ValueError("V4.1 cache offset/stride is not dtype aligned")
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, spec.storage_block_size, spec.num_kv_heads, width),
            stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, spec.head_size, offset)
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key, view(spec.scale_dtype, spec.scale_dim, offset + plane_sizes[0])
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
        raise NotImplementedError("V4.1 initial runtime requires PP=DCP=PCP=1")
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("V4.1 requires the hybrid KV cache manager")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("V4.1 initial cache layout requires BF16")
    if speculative is not None:
        # Aurora's planes are always BF16. Pin the inherited DSV4 draft
        # backend to the same layout, including on hardware where auto is FP8.
        vllm_config.cache_config.cache_dtype = "bfloat16"
