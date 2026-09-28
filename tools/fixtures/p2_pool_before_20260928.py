# SPDX-License-Identifier: Apache-2.0
# [K_l1_8card] 本文件 = agents/P2_poolsizing/patch/p2_pool.py 的**逐字节副本**
#   + 末尾追加 patch/src/l1_extra.py（挂载版需要的 compute_weights/worker_rows 等）。
#   仅有的两处定点改写：`prepare_store` 的配额不足分支各加一行日志（算法未动），
#   以及新增的 `_p2_log_short()`；算术部分与 030 逐字节相同。
# [P2_poolsizing] 把 DRAM KV 池的"16 张张量各分 num_blocks 个 slot"改成
# "每张张量只分它自己那组真正需要的行数"。
#
# 依据（见 a2/logs/029 / 030）：
#   * 镜像里 `NPUOffloadingWorker.__init__` 对 `kv_caches.tensors` 里**每一张**张量都分配
#     `(num_cpu_blocks, page_bytes)` ⇒ 宿主 = N × Σpage（16 张全按 N 行算）；
#   * 而 manager 只有**一个** unit 名空间（`CPUOffloadingManager._free_list`），
#     一个 unit 号只属于**一个** (group, chunk) ⇒ 一张张量同一时刻最多用到
#     "该组占的 unit 数"，其余行是结构性闲置；
#   * worker 侧 `compute_sub_block_ptrs()` 只把 unit 号当作**该组张量的行号**用
#     （`base_ptr + block_id * row_stride`），所以"unit 号"在**组内**只要 < 行数即可
#     —— 不需要跨组唯一（组与组之间靠 `CPULoadStoreSpec` 的**分组切片**隔离）。
#
# 于是本模块做两件事：
#   ① 调度侧：按组配额分配 unit（每组的行号从 0 开始），给的 spec 里**回填行号**
#      （`_get_load_store_spec` 解码），worker 侧一行代码都不用改；
#   ② worker 侧：每张张量的行数 = 引用它的组里最大的配额。
#
# 向后兼容：只有 `blocks_per_chunk` 是 dict（= `logs/021` 的 unit 模式）且本补丁开关打开
# 时才生效；否则**一个字节的行为都不变**。

from __future__ import annotations

# ★ 这个模块会被 sitecustomize 在**解释器启动早期**导入 ⇒ 顶层**不能** import vllm
#   （会在 `vllm_ascend.device.device_op` 上撞到循环导入，L1_dummy 实测过）。
#   所有 vllm 符号都在函数体内延迟导入。
from collections.abc import Collection, Iterable
from typing import Any

# unit id = (group << STRIDE_BITS) | row。STRIDE 取 2^20：
# A2 生产口径 58 GiB / 1 MiB/unit = 59,392 unit，离 2^20 还有 17×，够用。
STRIDE_BITS = 20
STRIDE = 1 << STRIDE_BITS
ROW_MASK = STRIDE - 1
MAX_UNITS = STRIDE

WEIGHTS_KEY = "p2_group_weights"
COMP_KEY = "p2_group_component"
BASE_KEY = "p2_group_base"


def encode_unit(group: int, row: int) -> int:
    assert 0 <= row < STRIDE, f"[P2] row={row} 超出 STRIDE={STRIDE}（unit 太多）"
    return (int(group) << STRIDE_BITS) | int(row)


def unit_group(unit_id: int) -> int:
    return int(unit_id) >> STRIDE_BITS


def unit_row(unit_id: int) -> int:
    return int(unit_id) & ROW_MASK


def key_group(key: Any) -> int:
    from vllm.v1.kv_offload.base import get_offload_group_idx

    return get_offload_group_idx(key)


# ---------------------------------------------------------------- 配额（两侧共用）


def split_quota(total: int, weights: dict[int, int]) -> dict[int, int]:
    """把 `total` 个 unit 按权重拆成**每组自己的行数**（两侧必须算出同一个结果）。

    确定性：最大余数法 + 平局按组号小优先 + 权重 > 0 但配额为 0 时从最大配额借 1。
    """
    weights = {int(g): int(w) for g, w in weights.items()}
    quota = {g: 0 for g in weights}
    pos = {g: w for g, w in weights.items() if w > 0}
    if not pos or total <= 0:
        return quota
    s = sum(pos.values())
    remainders: list[tuple[int, int]] = []
    for g, w in sorted(pos.items()):
        num = total * w
        quota[g] = num // s
        remainders.append((num % s, g))
    left = total - sum(quota.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    idx = 0
    while left > 0:
        quota[remainders[idx % len(remainders)][1]] += 1
        left -= 1
        idx += 1
    if total >= len(pos):
        for g in sorted(pos):
            if quota[g] == 0:
                donor = max(sorted(quota), key=lambda k: (quota[k], -k))
                quota[donor] -= 1
                quota[g] += 1
    return quota


def quota_from_config(num_units: int, weights: dict[int, int]) -> dict[int, int]:
    return split_quota(num_units, weights)


def component_bases(
    quota: dict[int, int], comp_of: dict[int, int] | None
) -> dict[int, int]:
    """每组行号的**起始偏移**。

    ★ 为什么需要"分量"（component）：worker 侧 CPU 张量的行号 = unit id 的低位
      （`compute_sub_block_ptrs` 直接拿它当行号），而**同一张张量可以被多个组引用**
      （DSV4.1 里 10 个 SWA 组 + state 组共享 4 张 "alias 容量视图"）。
      两个组只要引用同一张张量，它们的行区间就**必须互不相交**，否则会互相覆盖。
      所以：
        * 同一分量内的组按组号顺序**连续排布**（偏移累加）；
        * 不同分量之间**可以重叠**（它们不引用任何同一张张量）⇒ 省下大量行。
      `comp_of=None`（拿不到可靠的分组关系）⇒ 全部算成一个分量 = 全局累加（保守但安全）。
    """
    bases: dict[int, int] = {}
    if not comp_of:
        acc = 0
        for g in sorted(quota):
            bases[g] = acc
            acc += quota[g]
        return bases
    per_comp: dict[int, list[int]] = {}
    for g in sorted(quota):
        per_comp.setdefault(int(comp_of.get(g, 0)), []).append(g)
    for _c, gs in per_comp.items():
        acc = 0
        for g in gs:
            bases[g] = acc
            acc += quota[g]
    return bases


def alloc_end(quota: dict[int, int], bases: dict[int, int], g: int) -> int:
    """组 g 的行区间上界（开区间）。"""
    return int(bases.get(g, 0)) + int(quota.get(g, 0))


class _GroupEvictFilter:
    """喂给 `CachePolicy.evict(n, protected)` 的**惰性**过滤集。

    `evict()` 只用 `key in protected` 判断（见 lru.py / arc.py），所以传一个只实现
    `__contains__` 的对象即可：**别的组的 key 一律"受保护"** ⇒ 只会淘汰本组的条目。
    这样不用复制整个 LRU/ARC 策略，也不用 O(n) 建列表。
    """

    __slots__ = ("group", "extra")

    def __init__(self, group: int, extra: Collection[OffloadKey] = ()):
        self.group = group
        self.extra = extra

    def __contains__(self, key) -> bool:
        return key in self.extra or key_group(key) != self.group


# ---------------------------------------------------------------- 调度侧 manager


def make_quota_manager(base_cls):
    """基于（可能已被 `logs/021` 补过的）manager 生成"按组配额"版本。"""

    from typing_extensions import override

    from vllm.v1.kv_offload.base import OffloadingEvent, PrepareStoreOutput, ReqContext
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics, CPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

    class P2QuotaManager(base_cls):  # type: ignore[misc, valid-type]
        """每组的 unit 号从 0 开始（= 该组张量的行号），配额见 `self.quota`。"""

        def __init__(
            self,
            num_blocks: int,
            bpc_by_group: dict[int, int],
            weights_by_group: dict[int, int],
            base_by_group: dict[int, int] | None = None,
            *args,
            **kwargs,
        ):
            super().__init__(
                num_blocks=num_blocks, bpc_by_group=bpc_by_group, *args, **kwargs
            )
            self._p2_weights = {int(g): int(w) for g, w in weights_by_group.items()}
            self._p2_quota = quota_from_config(self._num_blocks, self._p2_weights)
            self._p2_base = (
                {int(g): int(b) for g, b in base_by_group.items()}
                if base_by_group
                else component_bases(self._p2_quota, None)
            )
            self._p2_free_rows: dict[int, list[int]] = {
                g: [] for g in self._p2_quota
            }
            self._p2_hi_rows: dict[int, int] = {g: 0 for g in self._p2_quota}
            self._p2_rows_used: dict[int, int] = {g: 0 for g in self._p2_quota}
            self._p2_quota_short = 0
            print(
                "[P2_poolsizing] 按组配额：units=%d weights=%s quota=%s "
                "base=%s (Σquota=%d)"
                % (
                    self._num_blocks,
                    dict(sorted(self._p2_weights.items())),
                    dict(sorted(self._p2_quota.items())),
                    dict(sorted(self._p2_base.items())),
                    sum(self._p2_quota.values()),
                ),
                flush=True,
            )

        # --- 账 ---

        def p2_quota(self) -> dict[int, int]:
            return dict(self._p2_quota)

        def p2_used_rows(self) -> dict[int, int]:
            return dict(self._p2_rows_used)

        def _p2_log_short(self, group: int, deficit: int, abandoned: bool) -> None:
            """配额不足的可观测性（对应 030 §6 的 R3/R4 风险；也是上线监测判据之一）。

            本组配额用尽 ⇒ 只淘汰**本组**条目；若本组没有可淘汰条目，本轮 store 被放弃
            （上游 `prepare_store` 返回 None 的既有语义），**不会**去借别组的配额。
            """
            if self._p2_quota_short <= 20 or self._p2_quota_short % 200 == 0:
                print(
                    "[P2_poolsizing] [K_l1_8card] 配额不足 #%d group=%d deficit=%d "
                    "abandoned=%s quota=%d used=%d"
                    % (
                        self._p2_quota_short,
                        int(group),
                        int(deficit),
                        bool(abandoned),
                        int(self._p2_quota.get(group, 0)),
                        int(self._p2_rows_used.get(group, 0)),
                    ),
                    flush=True,
                )

        def _p2_free_rows_of(self, group: int, want: int) -> list[int]:
            free = self._p2_free_rows.setdefault(group, [])
            out: list[int] = []
            while len(out) < want and free:
                out.append(free.pop())
            cap = self._p2_quota.get(group, 0)
            while len(out) < want:
                local = self._p2_hi_rows.get(group, 0)
                if local >= cap:
                    return out  # 配额用尽（调用方负责先淘汰）
                self._p2_hi_rows[group] = local + 1
                out.append(self._p2_base.get(group, 0) + local)
            return out

        def free_units(self) -> int:
            return sum(
                max(0, self._p2_quota.get(g, 0) - self._p2_rows_used.get(g, 0))
                for g in self._p2_quota
            )

        def _used_units(self) -> int:
            return sum(self._p2_rows_used.values())

        # --- 池 ---

        @override
        def _get_num_free_blocks(self) -> int:
            return self.free_units()

        @override
        def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
            blocks: list[BlockStatus] = []
            for key in keys:
                g = key_group(key)
                want = self.bpc_of(key)
                rows = self._p2_free_rows_of(g, want)
                assert len(rows) == want, (
                    f"[P2] 配额不足：group={g} want={want} got={len(rows)} "
                    f"quota={self._p2_quota.get(g)} used={self._p2_rows_used.get(g)}"
                )
                units = [encode_unit(g, r) for r in rows]
                self._p2_rows_used[g] = self._p2_rows_used.get(g, 0) + want
                self._units_of_block[units[0]] = units
                blocks.append(BlockStatus(units[0]))
            # 兼容父类的 `_num_allocated_units`（本类不再用它记账）
            self._num_allocated_units = self._used_units()
            return blocks

        @override
        def _free_block(self, block: BlockStatus) -> None:
            unit_id = int(block.block_id)
            g = unit_group(unit_id)
            units = self._units_of_block.pop(unit_id, None)
            if units is None:
                units = [unit_id]
            for u in units:
                self._p2_free_rows.setdefault(unit_group(u), []).append(unit_row(u))
                self._p2_rows_used[unit_group(u)] = max(
                    0, self._p2_rows_used.get(unit_group(u), 0) - 1
                )

        @override
        def _get_load_store_spec(
            self,
            keys: Iterable[OffloadKey],
            blocks: Iterable[BlockStatus],
        ) -> CPULoadStoreSpec:
            """★ 关键：给调度/worker 侧的 unit 号一律**解码成组内行号**。

            worker 侧只把它当 `tensor[block_id]` 的行号用，而 `CPULoadStoreSpec` 的
            消费是**按组切片**的（`transfer_async` 逐组取 `cdiv(group_size, 1)` 个），
            所以组内从 0 编号正是它要的东西 ⇒ **worker 的拷贝路径一行都不用改**。
            """
            ids: list[int] = []
            for block in blocks:
                for unit_id in self._units_of_block.get(
                    int(block.block_id), [int(block.block_id)]
                ):
                    ids.append(unit_row(unit_id))
            return CPULoadStoreSpec(ids)

        # --- store：按组淘汰 ---

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

            need: dict[int, int] = {}
            for key in keys_to_store:
                g = key_group(key)
                need[g] = need.get(g, 0) + self.bpc_of(key)

            to_evict: list[OffloadKey] = []
            for g in sorted(need):
                deficit = need[g] - max(
                    0, self._p2_quota.get(g, 0) - self._p2_rows_used.get(g, 0)
                )
                if deficit <= 0:
                    continue
                self._p2_quota_short += 1
                self._p2_log_short(g, deficit, abandoned=False)
                evicted = self._policy.evict(
                    deficit, _GroupEvictFilter(g, set(keys))
                )
                if evicted is None:
                    # 本组（配额内）没有足够的可淘汰条目 ⇒ 本轮放弃 store（上游语义）
                    self._p2_quota_short += 1
                    self._p2_log_short(g, deficit, abandoned=True)
                    return None
                for key, block in evicted:
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

            blocks = self._allocate_blocks(list(keys_to_store))
            assert len(blocks) == len(keys_to_store)
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
            self._p2_free_rows = {g: [] for g in self._p2_quota}
            self._p2_hi_rows = {g: 0 for g in self._p2_quota}
            self._p2_rows_used = {g: 0 for g in self._p2_quota}

        @override
        def get_stats(self):
            stats = super().get_stats()
            try:
                usage = (
                    self._used_units() / self._num_blocks
                    if self._num_blocks > 0
                    else 0.0
                )
                stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
            except Exception:
                pass
            return stats

    return P2QuotaManager


# ---------------------------------------------------------------- worker 侧行数


def rows_per_tensor(
    kv_caches, num_units: int, weights: dict[int, int], base_by_group=None
):
    """每张 canonical 张量要几行 = 引用它的组里最大的**行区间上界**。

    并**校验**：引用同一张张量的组的行区间必须两两不相交（否则会互相覆盖）。
    """
    quota = quota_from_config(num_units, weights)
    bases = (
        {int(g): int(b) for g, b in base_by_group.items()}
        if base_by_group
        else component_bases(quota, None)
    )
    rows = [0] * len(kv_caches.tensors)
    detail: list[tuple[int, int]] = []
    conflicts: list[str] = []
    for g, refs in enumerate(kv_caches.group_data_refs):
        lo, hi = bases.get(g, 0), alloc_end(quota, bases, g)
        tset = sorted({ref.tensor_idx for ref in refs})
        detail.append((g, quota.get(g, 0)))
        for t in tset:
            rows[t] = max(rows[t], hi)
    # 校验：同一张张量的组区间不相交
    per_tensor: dict[int, list[tuple[int, int, int]]] = {}
    for g, refs in enumerate(kv_caches.group_data_refs):
        if quota.get(g, 0) <= 0:
            continue
        for t in sorted({ref.tensor_idx for ref in refs}):
            per_tensor.setdefault(t, []).append(
                (bases.get(g, 0), alloc_end(quota, bases, g), g)
            )
    for t, spans in per_tensor.items():
        spans.sort()
        for (lo1, hi1, g1), (lo2, hi2, g2) in zip(spans, spans[1:]):
            if lo2 < hi1:
                conflicts.append(
                    f"tensor[{t}]: group{g1} [{lo1},{hi1}) 与 group{g2} "
                    f"[{lo2},{hi2}) 重叠"
                )
    return rows, quota, detail, conflicts


# SPDX-License-Identifier: Apache-2.0
# [K_l1_8card] 追加到 `patched/p2_pool.py` 末尾的一段（挂载版 L1 的"引擎侧"部分）。
#
# 来源（**逐字搬运，只改"从哪里拿 cfg.groups"这一处**）：
#   * `compute_weights`   ← `agents/P2_poolsizing/patch/p2_hooks.py::compute_weights`
#     （唯一改动：`cfg.groups` → 形参 `groups`，因为挂载版是在
#      `offloading/config.build_offloading_config()` 内部调用，那里拿到的是
#      OffloadingGroupConfig 的 tuple，还没有 OffloadingConfig 对象）
#   * `compute_components` / `comp_from_env`  ← `p2_hooks.py` 同名函数（逐字）
#   * `components_from_group_refs`            ← `p2_hooks.py` 同名函数（逐字）
#   * `weights_from_extra` / `comp_from_extra` / `bpc_map_from_extra` ← `p2_hooks.py`（逐字）
#
# 为什么要拆成"引擎侧"：8 卡是 **docker -v 单文件挂载**（见 a2/logs/027 §1.1），
# 没有 sitecustomize 去 monkeypatch；所以 030 的三个钩子改成"挂在文件里"：
#   ① 权重/分量  → 这份文件里的 `compute_*`，由挂载版 `offloading/config.py` 调用
#   ② manager    → `make_quota_manager`（在 p2_pool.py 主体里，逐字来自 030）
#   ③ worker 行数 → 本文件的 `worker_rows()`，由挂载版 `native/npu.py` 调用


import os


def _log(msg: str) -> None:
    if os.environ.get("P2_POOL_LOG", "1") == "1":
        print("[P2_poolsizing] [K_l1_8card] " + msg, flush=True)


def patch_enabled() -> bool:
    """`P2_POOL_PATCH`（默认 0 = 逐字回退 027/L5 行为）。"""
    return os.environ.get("P2_POOL_PATCH", "0") == "1"


def struct_log_enabled() -> bool:
    return os.environ.get("P2_STRUCT_LOG", "1") == "1"


def bpc_map_from_extra(extra) -> dict[int, int] | None:
    raw = (extra or {}).get("blocks_per_chunk_by_group")
    if not raw:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def weights_from_extra(extra) -> dict[int, int] | None:
    raw = (extra or {}).get(WEIGHTS_KEY)
    if not raw:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def comp_from_extra(extra) -> dict[int, int] | None:
    raw = (extra or {}).get(COMP_KEY)
    if raw is None:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def _unwrap(spec):
    members = getattr(spec, "kv_cache_specs", None)
    if members:
        return next(iter(members.values()))
    return spec


def _prefix_cacheable(spec) -> bool:
    """与 `pgp_manager` / `pgp_scheduler` 同口径（递归 unwrap）。"""
    members = getattr(spec, "kv_cache_specs", None)
    if members:
        return bool(members) and all(_prefix_cacheable(m) for m in members.values())
    return bool(getattr(spec, "prefix_cacheable", True)) and bool(
        getattr(spec, "participates_in_prefix_caching", True)
    )


def compute_weights(kv_cache_config, groups, bpc_map: dict[int, int]) -> dict[int, int]:
    """每组"每个对齐窗口占多少 unit"。

    ★ 逐字来自 `agents/P2_poolsizing/patch/p2_hooks.py::compute_weights`，
    唯一改动是 `cfg.groups` → `groups` 形参（见本文件头）。规则：
      * `tokens_per_chunk_g = tokens_per_block_g × bpc_g`；
      * 对齐窗口 `W = max_g tokens_per_chunk_g`（8 卡真权重上 = 1024）；
      * 不参与前缀缓存的组（state）权重 **0**；
      * SWA 组：`alignment_chunk_count = W/span_g` ⇒ 每段只留 `sw_chunks + is_eagle` 个 chunk；
      * 其余组：每个对齐段全存（`W/span_g` 个 chunk）。
      单位是 unit（= bpc_g 个 GPU block），所以权重 × bpc_g。
    """
    from vllm.utils.math_utils import cdiv
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    tpb = [g.tokens_per_block for g in groups]
    spans = [tpb[i] * bpc_map.get(i, 1) for i in range(len(groups))]
    alignment_tokens = max(spans) if spans else None

    weights: dict[int, int] = {}
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        span = spans[idx]
        bpc = bpc_map.get(idx, 1)
        spec = _unwrap(group.kv_cache_spec)
        if not _prefix_cacheable(group.kv_cache_spec):
            weights[idx] = 0
            continue
        swa = isinstance(spec, SlidingWindowSpec)
        sw_chunks = cdiv(spec.sliding_window, span) if swa else None
        alignment = None
        if swa and alignment_tokens is not None and alignment_tokens > span:
            per_segment = alignment_tokens // span
            if sw_chunks is not None and sw_chunks < per_segment:
                alignment = per_segment
        per_segment = (
            max(1, alignment_tokens // span) if alignment_tokens is not None else 1
        )
        eagle = 1 if getattr(group, "is_eagle_group", False) else 0
        if alignment is None:
            weights[idx] = bpc * per_segment
        else:
            tail = max(1, min(alignment, (sw_chunks or 1) + eagle))
            weights[idx] = bpc * tail
    return weights


def comp_from_env(num_groups: int) -> dict[int, int] | None:
    raw = os.environ.get("P2_COMP_JSON", "").strip()
    if not raw:
        return None
    try:
        import json

        members: list[list[int]] = json.loads(raw)
        comp: dict[int, int] = {}
        for c_idx, group_list in enumerate(members):
            for g in group_list:
                comp[int(g)] = c_idx
        if sorted(comp) != list(range(num_groups)):
            raise ValueError(
                f"P2_COMP_JSON 必须覆盖全部 {num_groups} 个组，得到 {sorted(comp)}"
            )
        return comp
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"P2_COMP_JSON 解析失败: {raw!r} ({exc})") from exc


def compute_components(kv_cache_config) -> tuple[dict[int, int], str]:
    """★ 组↔张量的**连通分量**（同一分量内的组共享过至少一张张量）。

    逐字来自 `p2_hooks.compute_components`：拿不到可靠信息 ⇒ 返回**单分量**（保守但安全）。
    ⚠️ Ascend 的 slot 打包布局下 `kv_cache_tensors[i].shared_by` 会把所有组并成一个分量
    （030 §3 实测）⇒ 想要完整 1.96× 必须给 `P2_COMP_JSON`（worker 侧按真实
    `group_data_refs` 反向校验，把共享张量的组拆开就**拒绝启动**）。
    """
    hint = comp_from_env(len(kv_cache_config.kv_cache_groups))
    if hint is not None:
        return hint, "来自 P2_COMP_JSON（运行期由 worker 按真实张量关系校验）"
    groups = kv_cache_config.kv_cache_groups
    layer_to_group: dict[str, int] = {}
    for idx, group in enumerate(groups):
        for name in group.layer_names:
            layer_to_group[name] = idx
    parent = list(range(len(groups)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    tensors = list(getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
    if not tensors:
        return {g: 0 for g in range(len(groups))}, "无 kv_cache_tensors ⇒ 单分量"
    seen_any = False
    for kv_tensor in tensors:
        names = list(getattr(kv_tensor, "shared_by", ()) or ())
        gs = sorted({layer_to_group[n] for n in names if n in layer_to_group})
        if not gs:
            return (
                {g: 0 for g in range(len(groups))},
                "有张量的 shared_by 解析不出组 ⇒ 单分量（保守）",
            )
        seen_any = True
        for other in gs[1:]:
            union(gs[0], other)
    if not seen_any:
        return {g: 0 for g in range(len(groups))}, "shared_by 全空 ⇒ 单分量"
    roots: dict[int, int] = {}
    comp: dict[int, int] = {}
    for g in range(len(groups)):
        r = find(g)
        if r not in roots:
            roots[r] = len(roots)
        comp[g] = roots[r]
    return comp, f"分量={len(roots)} 来自 {len(tensors)} 张 kv_cache_tensors"


def components_from_group_refs(group_data_refs) -> list[list[int]]:
    """worker 侧真值：按"是否引用同一张 canonical 张量"做组的连通分量（逐字来自 p2_hooks）。"""
    n = len(group_data_refs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[int, int] = {}
    for g, refs in enumerate(group_data_refs):
        for t in {int(ref.tensor_idx) for ref in refs}:
            if t in owner:
                a, b = find(owner[t]), find(g)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[t] = g
    groups: dict[int, list[int]] = {}
    for g in range(n):
        groups.setdefault(find(g), []).append(g)
    return sorted((sorted(v) for v in groups.values()), key=lambda xs: xs[0])


def log_weights(weights, comp, bpc_map, enabled: bool, why: str = "") -> None:
    """① 的日志（挂载版没有钩子，所以由 `offloading/config.py` 直接调）。"""
    if not struct_log_enabled() and not enabled:
        return
    _log(
        "① 权重: %s (Σ=%d) components=%s (%s) bpc=%s enabled=%s"
        % (
            dict(sorted(weights.items())),
            sum(weights.values()),
            None if comp is None else dict(sorted(comp.items())),
            why,
            dict(sorted(bpc_map.items())),
            enabled,
        )
    )


# --------------------------------------------------------------------------- #
# ③ worker 侧：行数 + 结构日志（把 030 的 p2_hooks.install_create_worker_hook
#    里 `_create_worker_inner()` 的日志/校验部分搬过来，**只在挂载版 npu.py 里调用**）
# --------------------------------------------------------------------------- #


def worker_rows(spec, kv_caches) -> list[int]:
    """返回"每张 canonical 张量要几行"，并把 030 的判据日志打全。

    只读模式（`P2_POOL_PATCH=0`）：`rows = [num_blocks] × tensors`（**逐字回退 027**），
    但仍打"真实分量 / 记账对账 / 结构"三行 —— 这正是本任务要的只读结构臂。
    """
    num_units = int(spec.num_blocks)
    extra = spec.extra_config
    bpc_map = bpc_map_from_extra(extra)
    weights = weights_from_extra(extra)
    enabled = patch_enabled() and bool(weights) and bool(bpc_map)

    pages = [int(t.page_size_bytes) for t in kv_caches.tensors]
    refs = [
        [int(ref.tensor_idx) for ref in group_refs]
        for group_refs in kv_caches.group_data_refs
    ]
    truth = components_from_group_refs(kv_caches.group_data_refs)

    # ★ 判据 A（030 §7.1）：该几何下"页 == 行宽"是否成立（int8 会破坏它）
    _bpc = int(spec.blocks_per_chunk)
    _sum_page = sum(pages)
    _book = getattr(spec.config, "worker_kv_bytes_per_block", None)
    _aligned = spec.kv_bytes_per_chunk
    _log(
        "③ ★ 记账对账: worker_kv_bytes_per_block=%s  Σpage=%d  Σ(page)×bpc=%d  "
        "aligned_kv_bytes_per_chunk(单位池)=%d  ⇒ 旧口径主机/记账 = %d/%s = %.3f×"
        % (
            _book,
            _sum_page,
            _sum_page * _bpc,
            _aligned,
            _sum_page * _bpc,
            _book,
            ((_sum_page * _bpc) / _book) if _book else 0.0,
        )
    )
    _log("③ ★ 真实分量（worker 侧真值，可直接用作 P2_COMP_JSON）：%s" % (truth,))
    if struct_log_enabled() or enabled:
        _log("③ 结构: tensors=%d pages=%s" % (len(pages), pages))
        _log("③ 结构: group→tensor_idx = %s" % refs)
        _log(
            "③ 结构: group→page(copy) = %s"
            % [
                [int(ref.page_size_bytes) for ref in group_refs]
                for group_refs in kv_caches.group_data_refs
            ]
        )
    old = sum(num_units * p * _bpc for p in pages)

    if not enabled:
        _log(
            "③ [只读] P2_POOL_PATCH=%r weights=%s bpc=%s ⇒ 旧口径每张张量 %d 行 "
            "⇒ 宿主 = %d B (%.3f GiB)"
            % (
                os.environ.get("P2_POOL_PATCH"),
                None if not weights else {k: weights[k] for k in sorted(weights)},
                None if not bpc_map else {k: bpc_map[k] for k in sorted(bpc_map)},
                num_units,
                old,
                old / (1 << 30),
            )
        )
        return [num_units] * len(pages)

    comp = comp_from_extra(extra)
    if comp is None:
        # 没有 hint ⇒ 单分量（全局累加）：安全，但只拿到 ~1.29×（030 §3）
        _log("③ !! 没有 p2_group_component（P2_COMP_JSON 未给或未生效）⇒ 单分量（保守）")
    # ★ 反向校验：分量必须**不细于**真实张量关系（更粗 = 更保守 = 安全）
    truth_comp: dict[int, int] = {}
    for ci, members in enumerate(truth):
        for g in members:
            truth_comp[g] = ci
    bad = [
        (g, h)
        for g in truth_comp
        for h in truth_comp
        if g < h
        and truth_comp[g] == truth_comp[h]
        and (comp or {}).get(g) != (comp or {}).get(h)
    ]
    if bad:
        raise RuntimeError(
            "[K_l1_8card][L1] P2_COMP_JSON 把**共享张量**的组拆到了不同分量 —— 会互相覆盖，"
            f"拒绝启动。truth={truth} hint={dict(sorted((comp or {}).items()))} 冲突对={bad[:5]}"
        )

    bases = component_bases(quota_from_config(num_units, weights), comp)
    rows, quota, _detail, conflicts = rows_per_tensor(
        kv_caches, num_units, weights, bases
    )
    if conflicts:
        raise RuntimeError(
            "[K_l1_8card][L1] 行区间冲突（分量推导与实际张量共享关系不一致）:\n  "
            + "\n  ".join(conflicts)
        )
    _log(
        "③ 池子: units=%d Σquota=%d quota=%s base=%s rows=%s"
        % (
            num_units,
            sum(quota.values()),
            dict(sorted(quota.items())),
            dict(sorted(bases.items())),
            rows,
        )
    )
    new = sum(r * p * _bpc for r, p in zip(rows, pages))
    _log(
        "③ 宿主实占: 旧=%d B (%.3f GiB) → 新=%d B (%.3f GiB) = %.2f× 缩小"
        % (old, old / (1 << 30), new, new / (1 << 30), (old / new) if new else 0.0)
    )
    if any(r <= 0 for r in rows):
        _log("③ !! 有张量分到 0 行：rows=%s（该张量无组引用或配额为 0）" % (rows,))
    return rows
