# SPDX-License-Identifier: Apache-2.0
# [SWA_pergroup] per-group `blocks_per_chunk`：**"块"为单位的池记账**。
#
# 背景（见 a2/logs/017 §6 / 021）：
#   镜像内 CPU 卸载池的一格 = `worker_kv_bytes_per_block × num_copies × blocks_per_chunk`，
#   而 `blocks_per_chunk` 是**全局标量** ⇒ SWA 组的 chunk（1 个 block 就够）也被按 8 个
#   block 记账，且上游 `is_store_reachable_swa_chunk()` 因 `alignment_tokens == tokens_per_chunk`
#   而**空转**。
#
# 本模块的做法（"单位池"）：
#   * 池的**单位（unit）** = 1 个 GPU block（= `worker_kv_bytes_per_block × num_copies` 字节）；
#     于是 `CPUOffloadingSpec.__init__` 用 `blocks_per_chunk=1` 算出来的 `num_blocks`
#     就是 unit 数（`build_offloading_config()` 在 per-group 模式下把
#     `cache.blocks_per_chunk` 设成 1）。
#   * 一个 chunk（= 该组的 `blocks_per_chunk` 个 block）占 `bpc_g` 个 unit；manager 给每个 key
#     分配一组 unit id（顺序 = chunk 内 block 顺序），调度侧把它们展开成"一个 block 一个 id"
#     写进 `CPULoadStoreSpec`（worker 侧 `blocks_per_chunk=1` ⇒ 两边 1:1，skip 恒为 0）。
#   * 组与组之间**共用一个 LRU 池**（不切分容量），淘汰按 **unit** 记账。
#
# 向后兼容：`blocks_per_chunk` 仍是标量时**根本不会走到这里**（spec 用镜像内的原 manager）。

from __future__ import annotations

from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    OffloadKey,
    OffloadingEvent,
    PrepareStoreOutput,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics, CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

BPC_BY_GROUP_KEY = "blocks_per_chunk_by_group"


def bpc_map_from_extra(extra_config) -> dict[int, int] | None:
    """从 `OffloadingConfig.extra_config` 取出 per-group bpc（没有就返回 None）。

    这个 dict 由 `build_offloading_config()` 在**调度侧和 worker 侧各自**算出，
    内容只依赖 `kv_cache_config` ⇒ 两侧逐字一致（这是 unit↔row 映射一致的前提）。
    """
    raw = (extra_config or {}).get(BPC_BY_GROUP_KEY)
    if not raw:
        return None
    return {int(k): int(v) for k, v in raw.items()}


class PerGroupBPCManager(CPUOffloadingManager):
    """`CPUOffloadingManager` 的"单位池"版本：一个 key 占 `bpc_g` 个 unit。"""

    def __init__(
        self,
        num_blocks: int,
        bpc_by_group: dict[int, int],
        *args,
        **kwargs,
    ):
        super().__init__(num_blocks=num_blocks, *args, **kwargs)
        self._bpc_by_group: dict[int, int] = dict(bpc_by_group)
        # block_id(首个 unit) -> 该 key 占用的 unit 列表（顺序 = chunk 内 block 顺序）
        self._units_of_block: dict[int, list[int]] = {}
        self._num_allocated_units: int = 0

    # --- 账 ---

    def bpc_of(self, key: OffloadKey) -> int:
        return self._bpc_by_group.get(get_offload_group_idx(key), 1)

    def free_units(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_units

    def _used_units(self) -> int:
        return self._num_allocated_units - len(self._free_list)

    # --- 池 ---

    @override
    def _get_num_free_blocks(self) -> int:
        return self.free_units()

    @override
    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        blocks: list[BlockStatus] = []
        for key in keys:
            want = self.bpc_of(key)
            units: list[int] = []
            while len(units) < want and self._free_list:
                units.append(self._free_list.pop())
            while len(units) < want:
                units.append(self._num_allocated_units)
                self._num_allocated_units += 1
            assert self._used_units() <= self._num_blocks, (
                "[SWA_pergroup] 单位池溢出："
                f"used={self._used_units()} capacity={self._num_blocks}"
            )
            block = BlockStatus(units[0])
            self._units_of_block[units[0]] = units
            blocks.append(block)
        return blocks

    @override
    def _free_block(self, block: BlockStatus) -> None:
        units = self._units_of_block.pop(block.block_id, None)
        if units is None:
            units = [block.block_id]
        self._free_list.extend(units)

    @override
    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        ids: list[int] = []
        for block in blocks:
            ids.extend(self._units_of_block.get(block.block_id, [block.block_id]))
        return CPULoadStoreSpec(ids)

    # --- store：淘汰按 unit（镜像内原版按 key 数，这里必须换成 unit） ---

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        units_needed = sum(self.bpc_of(k) for k in keys_to_store)
        units_to_free = units_needed - self.free_units()

        to_evict: list[OffloadKey] = []
        if units_to_free > 0:
            # 每个可淘汰条目至少占 1 个 unit ⇒ 这一条只是快速失败。
            if units_to_free > self._num_evictable_cache_blocks:
                return None
            protected = set(keys)
            freed = 0
            while freed < units_to_free:
                evicted = self._policy.evict(1, protected)
                if not evicted:
                    # 部分淘汰已经发生：被淘汰的都是 idle 条目（缓存语义允许丢），
                    # 本轮 store 直接放弃，池子记账保持一致。
                    return None
                for key, block in evicted:
                    freed += len(
                        self._units_of_block.get(block.block_id, [block.block_id])
                    )
                    self._free_block(block)
                    to_evict.append(key)
                    self._num_evictable_cache_blocks -= 1
            assert self._num_evictable_cache_blocks >= 0

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def reset_cache(self) -> None:
        super().reset_cache()
        self._units_of_block.clear()
        self._num_allocated_units = 0

    @override
    def get_stats(self):
        stats = super().get_stats()
        try:
            # 原版口径把"已分配 unit 数"与"可淘汰 key 数"相减（单位混了），
            # 单位池下会偏小甚至为负；这里改成 **unit** 口径。
            usage = (
                self._used_units() / self._num_blocks if self._num_blocks > 0 else 0.0
            )
            stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
        except Exception:
            pass
        return stats



# --------------------------------------------------------------------------- #
# [L3_8card] 从 SWA_pergroup/patch/pgp_hooks.py 原样搬来的"配置面"解析：
#   8 卡是文件挂载（没有 sitecustomize 去 monkeypatch build_offloading_config），
#   所以把这段放进能被两边 import 的模块里，由挂载版 offloading/config.py 调用。
#   逻辑与 021 逐字一致（同一份 resolve_per_group_bpc）。
# --------------------------------------------------------------------------- #
def _unwrap_spec(kv_cache_spec):
    """unwrap UniformTypeKVCacheSpecs：判"是不是滑窗"只看代表 spec。"""
    members = getattr(kv_cache_spec, "kv_cache_specs", None)
    if members:
        return next(iter(members.values()))
    return kv_cache_spec


def _group_kinds(kv_cache_config) -> list[str]:
    from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

    kinds: list[str] = []
    for group in kv_cache_config.kv_cache_groups:
        spec = _unwrap_spec(group.kv_cache_spec)
        if isinstance(spec, SlidingWindowSpec):
            kinds.append("swa")
        elif isinstance(spec, MambaSpec):
            kinds.append("mamba")
        else:
            kinds.append("full")
    return kinds


def resolve_per_group_bpc(user_cfg, kv_cache_config):
    """返回 (标量 bpc 或 None, per-group map 或 None)。与 021 逐字一致。"""
    if user_cfg is None or isinstance(user_cfg, int):
        return user_cfg, None
    if not isinstance(user_cfg, dict):
        raise ValueError(
            "[L3_8card] blocks_per_chunk 必须是整数或 dict，得到 "
            f"{type(user_cfg).__name__}"
        )

    from vllm.v1.kv_cache_interface import MambaSpec

    kinds = _group_kinds(kv_cache_config)
    for group in kv_cache_config.kv_cache_groups:
        spec = _unwrap_spec(group.kv_cache_spec)
        if isinstance(spec, MambaSpec) and getattr(
            spec, "mamba_cache_mode", None
        ) == "align":
            raise ValueError(
                "[L3_8card] 单位池模式不支持 MambaSpec(align)："
                "resolve_mamba_align_size() 假定全局 blocks_per_chunk"
            )

    base = user_cfg.get("default", user_cfg.get("full", 8))
    per_kind = {
        "swa": int(user_cfg.get("swa", base)),
        "full": int(user_cfg.get("full", base)),
    }
    bpc_by_group: dict[int, int] = {}
    for idx, kind in enumerate(kinds):
        if idx in user_cfg:
            value = user_cfg[idx]
        elif str(idx) in user_cfg:
            value = user_cfg[str(idx)]
        else:
            value = per_kind.get(kind, int(base))
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"[L3_8card] group {idx} 的 blocks_per_chunk={value} 必须 > 0"
            )
        bpc_by_group[idx] = value
    return 1, bpc_by_group
