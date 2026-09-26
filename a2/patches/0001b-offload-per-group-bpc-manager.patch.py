# SPDX-License-Identifier: Apache-2.0
# [SWA_pergroup / J_mgrhardening] per-group `blocks_per_chunk`：**块为单位的池记账**
#                              + **J_mgrhardening 的 fail-closed 加固**。
#
# 这份文件 = `a2/publish/0001b-offload-per-group-bpc-manager.patch.py` 的产物
# （`agents/SWA_pergroup/patch/pgp_manager.py`，md5 3b64eb4977f3302ed71e6c759c48740f）
# **逐字复制** + 下面四处**最小加固**。加固全部由 `PGP_MGR_HARDEN` 门控，
# **默认 0 = 逐字回退旧行为**（只多了一份只读记账，见 `PGP_MGR_STATS`）。
#
# 加固的动机（`038` §4/§11、任务书 `041`）：
#   旧实现在三处可以**静默**错：
#     ① `_allocate_blocks` 没有上游的容量 cap（`cpu/manager.py:80-95` 的
#        `num_fresh = min(len(keys), num_blocks - num_allocated)`），free_list 空时会
#        **超发** unit（row id ≥ num_blocks）；
#     ② `_units_of_block[units[0]] = units` 只按首 unit 建索引 ⇒ 首 unit 被复用时**键被覆盖**，
#        旧块的 unit 列表**静默丢失**；
#     ③ `_free_block` 在 `_units_of_block` 里查不到时**兜底** `[block.block_id]` 推回 free_list
#        ⇒ "不知道这行是否还活着"时**静默**把活行还回池子 ⇒ 同一行被两个 block 占用。
#   （`_used_units()` 又会因 free_list 重复而**少算**，所以旧有的 assert 兜不住。）
#
# 加固（PGP_MGR_HARDEN=1；=2 更严）：
#   (a) 容量 cap：`want > num_blocks - _num_allocated_units + len(_free_list)` ⇒ **raise**（不超发）
#   (b) 过期/重复 free：`_units_of_block` 里查不到 ⇒ **raise**（去掉静默兜底）
#   (c) 索引按**完整 unit 集合**（冻结 tuple）+ 反查表 `_owner_of_unit`：
#       分配时"同一行已有活块"或"键已存在" ⇒ **raise**；free 时逐 unit 校验归属
#   (d) 只读计数器（stale_free / dup_unit / oob_unit / over_budget / key_overwrite）：
#       `PGP_MGR_STATS=1` 打开，或 HARDEN=1 时自动开
#
# 语义**不变**的部分（L5 的五条判据依赖它）：`block_id` 仍然是 `units[0]`；
# `_get_load_store_spec()` 仍然返回"按 chunk 内 block 顺序展开"的 unit 平铺表；
# `free_units()`/`get_stats()` 的 unit 口径不变 ⇒ 调度侧展开、worker 侧 1:1 映射全部不动。
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

import os
import traceback
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


# 0 = 逐字旧行为；1 = b+c+a 全开（推荐）；2 = 再严一层（`_get_load_store_spec` 也 fail-closed）
HARDEN_MODE = int(os.environ.get("PGP_MGR_HARDEN", "0") or 0)
STATS_ON = _env_flag("PGP_MGR_STATS", "1" if HARDEN_MODE else "0")
STATS_EVERY = int(os.environ.get("PGP_MGR_STATS_EVERY", "200") or 0)

COUNTERS = (
    "over_budget",  # (a) 分配前 want > 预算
    "stale_free",  # (b) free 时表里没有这个 block_id
    "free_owner_mismatch",  # (b) free 的 unit 归属对不上
    "stale_block_obj",  # (b') free/spec 时 block_id 对应的 BlockStatus **不是**这一个（老对象指向被复用的行）
    "key_overwrite",  # (c) `_units_of_block[units[0]]` 会覆盖已存在的键
    "dup_unit",  # (c) 同一 unit 已有活块
    "oob_unit",  # unit id ≥ num_blocks
    "used_mismatch",  # Σ 活块 unit 数 ≠ _used_units()
)


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
        # block_id(首个 unit) -> 该 key 占用的 unit 集合（**冻结 tuple**，顺序 = chunk 内 block 顺序）
        self._units_of_block: dict[int, tuple[int, ...]] = {}
        # unit -> block_id（反查表）：用来在分配/释放时**立刻**发现"同一行两个块"
        self._owner_of_unit: dict[int, int] = {}
        # block_id -> 当初分配出去的那个 BlockStatus **对象**
        # （`038` §11-3 要的"行级 provenance"：老对象指向被复用的行 ⇒ 命中会读到别人的数据）
        self._block_obj: dict[int, BlockStatus] = {}
        self._num_allocated_units: int = 0
        # (d) 只读计数器
        self._mgr_counters: dict[str, int] = {k: 0 for k in COUNTERS}
        self._mgr_stat_ops: int = 0
        # Σ 活块占用的 unit 数（**增量维护**：`_sanity()` 因此是 O(1)，A2 大池也不怕）
        self._index_units_total: int = 0

    # --- 账 ---

    def bpc_of(self, key: OffloadKey) -> int:
        return self._bpc_by_group.get(get_offload_group_idx(key), 1)

    def free_units(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_units

    def _used_units(self) -> int:
        return self._num_allocated_units - len(self._free_list)

    # --- (d) 计数器 ---

    def _bump(self, name: str, **kv) -> None:
        self._mgr_counters[name] = self._mgr_counters.get(name, 0) + 1
        if STATS_ON:
            print(
                "[J_mgr_hard] cnt=%s n=%d %s"
                % (
                    name,
                    self._mgr_counters[name],
                    " ".join("%s=%s" % (k, v) for k, v in sorted(kv.items())),
                ),
                flush=True,
            )

    def _maybe_stats(self) -> None:
        if not STATS_ON or STATS_EVERY <= 0:
            return
        self._mgr_stat_ops += 1
        if self._mgr_stat_ops % STATS_EVERY == 0:
            print(
                "[J_mgr_hard] stats harden=%d allocated=%d free=%d live_blocks=%d live_units=%d %s"
                % (
                    HARDEN_MODE,
                    self._num_allocated_units,
                    len(self._free_list),
                    len(self._units_of_block),
                    len(self._owner_of_unit),
                    " ".join("%s=%d" % (k, v) for k, v in sorted(self._mgr_counters.items())),
                ),
                flush=True,
            )

    def mgr_hardening_stats(self) -> dict[str, int]:
        """给"上线观测"用的只读快照（HARDEN=0 时也有效，只要 PGP_MGR_STATS=1）。"""
        return dict(self._mgr_counters)

    # --- 索引（(c)：按完整 unit 集合 + 反查） ---

    def _index_insert(self, units: list[int], block: BlockStatus) -> None:
        first = units[0]
        if first in self._units_of_block:
            self._bump("key_overwrite", block_id=first, units=units)
            if HARDEN_MODE:
                raise RuntimeError(
                    "[J_mgr_hard] 单位池索引键被覆盖：block_id=%d 已经在活块表里"
                    "（old=%s new=%s）⇒ 同一行会被两个 block 占用"
                    % (first, self._units_of_block[first], tuple(units))
                )
            # 旧行为 = 直接覆盖 ⇒ 旧条目的 unit 数要从增量账里扣掉（否则 _sanity 会误报）
            self._index_units_total -= len(self._units_of_block[first])
        for u in units:
            owner = self._owner_of_unit.get(u)
            if owner is not None and owner != first:
                self._bump("dup_unit", unit=u, owner=owner, block_id=first, units=units)
                if HARDEN_MODE:
                    raise RuntimeError(
                        "[J_mgr_hard] 同一行被两个活块占用：unit=%d 现在属于 block_id=%d，"
                        "又分配给 block_id=%d（units=%s）"
                        % (u, owner, first, tuple(units))
                    )
        for u in units:
            if u < 0 or u >= self._num_blocks:
                self._bump("oob_unit", unit=u, num_blocks=self._num_blocks, block_id=first)
                if HARDEN_MODE:
                    raise RuntimeError(
                        "[J_mgr_hard] unit id 越界：unit=%d num_blocks=%d"
                        % (u, self._num_blocks)
                    )
            self._owner_of_unit[u] = first
        self._units_of_block[first] = tuple(units)
        self._block_obj[first] = block
        self._index_units_total += len(units)

    def _index_pop(self, block_id: int) -> tuple[int, ...] | None:
        units = self._units_of_block.pop(block_id, None)
        if units is None:
            return None
        for u in units:
            if self._owner_of_unit.get(u) != block_id:
                self._bump(
                    "free_owner_mismatch",
                    unit=u,
                    owner=self._owner_of_unit.get(u),
                    block_id=block_id,
                )
                if HARDEN_MODE:
                    raise RuntimeError(
                        "[J_mgr_hard] free 的 unit 归属对不上：unit=%d 属于 block_id=%s，"
                        "却在释放 block_id=%d"
                        % (u, self._owner_of_unit.get(u), block_id)
                    )
            self._owner_of_unit.pop(u, None)
        self._block_obj.pop(block_id, None)
        self._index_units_total -= len(units)
        return tuple(units)

    def _check_block_obj(self, block: BlockStatus, where: str) -> bool:
        """`block_id` 还活着，但"这个 BlockStatus 对象"是不是当初那一个？

        如果不是（例如老 BlockStatus 被复用时留下），说明**同一个 block_id 有第二个对象**
        ⇒ 命中/释放都会指到别人的行。这是"同一行 → 两个 block"在**对象层**的等价物。
        """
        cur = self._block_obj.get(block.block_id)
        if cur is not None and cur is not block:
            self._bump("stale_block_obj", block_id=block.block_id, where=where)
            return False
        return True

    def _units_len(self, block: BlockStatus) -> int:
        units = self._units_of_block.get(block.block_id)
        if units is None:
            self._bump("stale_free", block_id=block.block_id, where="units_len")
            if HARDEN_MODE:
                raise RuntimeError(
                    "[J_mgr_hard] 活块表里没有 block_id=%d（不知道它是否还活着）"
                    % block.block_id
                )
            return 1
        return len(units)

    def _sanity(self) -> None:
        """Σ 活块 unit 数 vs `_used_units()`：真账目不一致就记数（HARDEN 时 raise）。"""
        total = self._index_units_total
        used = self._used_units()
        if total != used:
            # 增量账可能自己漂了 ⇒ 用一次全量重算把"真值"钉死再报
            real = sum(len(v) for v in self._units_of_block.values())
            self._bump("used_mismatch", index_total=total, real=real, used=used)
            self._index_units_total = real
            if HARDEN_MODE:
                raise RuntimeError(
                    "[J_mgr_hard] 单位池账目不一致：Σ活块unit=%d（增量账 %d）而 "
                    "_used_units()=%d（free_list 有重复 ⇒ 少算）" % (real, total, used)
                )

    # --- 池 ---

    @override
    def _get_num_free_blocks(self) -> int:
        return self.free_units()

    @override
    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        blocks: list[BlockStatus] = []
        for key in keys:
            want = self.bpc_of(key)
            # (a) 上游 `cpu/manager.py:81` 的 cap（L5 丢了它）：
            #     `num_fresh = min(len(keys), num_blocks - num_allocated)` —— 这里按 **unit** 算。
            budget = self._num_blocks - self._num_allocated_units + len(self._free_list)
            if want > budget:
                self._bump(
                    "over_budget",
                    want=want,
                    budget=budget,
                    num_blocks=self._num_blocks,
                    allocated=self._num_allocated_units,
                    free=len(self._free_list),
                )
                if HARDEN_MODE:
                    raise RuntimeError(
                        "[J_mgr_hard] 单位池容量不足而**拒绝超发**：want=%d budget=%d"
                        "（num_blocks=%d allocated=%d free_list=%d）"
                        % (want, budget, self._num_blocks, self._num_allocated_units,
                           len(self._free_list))
                    )
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
            self._index_insert(units, block)
            blocks.append(block)
        self._sanity()
        self._maybe_stats()
        return blocks

    @override
    def _free_block(self, block: BlockStatus) -> None:
        if not self._check_block_obj(block, "free") and HARDEN_MODE:
            raise RuntimeError(
                "[J_mgr_hard] 释放的不是当初分配给 block_id=%d 的那个 BlockStatus"
                "（老对象指向被复用的行 ⇒ 会把别人的行还回池子）" % block.block_id
            )
        units = self._index_pop(block.block_id)
        if units is None:
            # (b) **去掉静默兜底**：不知道这行是否还活着时，必须响亮失败
            self._bump("stale_free", block_id=block.block_id, free=len(self._free_list))
            if HARDEN_MODE:
                raise RuntimeError(
                    "[J_mgr_hard] 过期/重复 free：block_id=%d 不在活块表里"
                    "（旧实现会静默 push [%d] ⇒ 同一行两个块）"
                    % (block.block_id, block.block_id)
                )
            units = (block.block_id,)
        self._free_list.extend(units)

    @override
    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        ids: list[int] = []
        for block in blocks:
            if not self._check_block_obj(block, "spec") and HARDEN_MODE >= 2:
                raise RuntimeError(
                    "[J_mgr_hard] 取/存 spec 时 block_id=%d 的 BlockStatus 不是当初那一个"
                    % block.block_id
                )
            units = self._units_of_block.get(block.block_id)
            if units is None:
                self._bump("stale_free", block_id=block.block_id, where="spec")
                if HARDEN_MODE >= 2:
                    raise RuntimeError(
                        "[J_mgr_hard] 取/存 spec 时活块表里没有 block_id=%d"
                        "（旧实现静默退化成 1 个 unit）" % block.block_id
                    )
                units = (block.block_id,)
            ids.extend(units)
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
                    freed += self._units_len(block)
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
        self._owner_of_unit.clear()
        self._block_obj.clear()
        self._index_units_total = 0
        self._num_allocated_units = 0
        self._sanity()

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
            print("[J_mgr_hard] get_stats 失败:\n" + traceback.format_exc(), flush=True)
        return stats
