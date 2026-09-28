#!/usr/bin/env python3
"""离线自检：`a2/patches/kv8-offload-pool/p2_pool.py` 的两处**池满缺陷**（2026-09-28）。

为什么要有它：这两处缺陷**只在池接近满时**才走到，而真机验证一次要 6 分钟起服
  + 灌池。离线把它们钉住，后续任何改动都能在 <1 s 内发现回归。

被测的两个缺陷（都在 `make_quota_manager` 生成的 `P2QuotaManager` 里）：

  ① **过淘汰**：`deficit` 是 unit 数，`policy.evict(n)` 的 n 是**条目**数。
     组 g 每个条目占 `bpc_g` 个 unit ⇒ 旧代码淘汰 `deficit` 个条目 =
     释放 `deficit × bpc_g` 个 unit。full 组（bpc=8）⇒ 8 倍过淘汰。
     **判据**：`evict()` 实际收到的 n 必须 == `ceil(deficit / bpc_g)`，
     且释放的 unit 数恰好 ≥ deficit（不多不少到条目粒度）。

  ② **assert 打死引擎**：`_allocate_blocks` 里原来是 `assert len(rows)==want`。
     池满时若"账本口径(quota-used)"与"可分配口径(free+cap-hi)"不一致，
     deficit 会算小 ⇒ 淘汰不足 ⇒ assert ⇒ `EngineCore encountered a fatal error`。
     **判据**：制造这种不一致后，`prepare_store` 必须**返回 None**（上游语义
     "本轮不 store"），**绝不抛异常**；且已分配的半批行要回滚干净。

本文件**不依赖 vllm**：只把 `p2_pool` 真正用到的那几个符号 stub 出来。
跑法：`python3 tools/selftest_p2_pool_quota.py`
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 允许指向别的副本 ⇒ 用同一套判据做**负控**（对未修复的原版必须失败）：
#   P2_POOL_UNDER_TEST=/tmp/p2_pool.orig.py python3 tools/selftest_p2_pool_quota.py
TARGET = pathlib.Path(
    os.environ.get("P2_POOL_UNDER_TEST")
    or (ROOT / "a2/patches/kv8-offload-pool/p2_pool.py")
)

FAILS: list[str] = []
CHECKS = [0]


def check(cond: bool, label: str, detail: str = "") -> None:
    CHECKS[0] += 1
    if cond:
        print(f"  ✓ {label}")
    else:
        print(f"  ✗ {label}  {detail}")
        FAILS.append(label)


# ---------------------------------------------------------------- vllm stubs


class BlockStatus:
    __slots__ = ("ref_cnt", "block_id")

    def __init__(self, block_id: int):
        self.ref_cnt = -1  # -1 = 写入未完成（上游语义）
        self.block_id = int(block_id)

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class CPULoadStoreSpec:
    def __init__(self, block_ids):
        self.block_ids = list(block_ids)


class PrepareStoreOutput:
    def __init__(self, keys_to_store, store_spec, evicted_keys):
        self.keys_to_store = list(keys_to_store)
        self.store_spec = store_spec
        self.evicted_keys = list(evicted_keys)


class OffloadingEvent:
    def __init__(self, keys, medium, removed):
        self.keys = list(keys)
        self.medium = medium
        self.removed = removed


def make_offload_key(block_hash: bytes, group_idx: int):
    return block_hash + group_idx.to_bytes(4, "big", signed=False)


def get_offload_group_idx(key) -> int:
    return int.from_bytes(key[-4:], "big", signed=False)


def _install_stubs() -> None:
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for name in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu"):
        mod(name)
    base = mod("vllm.v1.kv_offload.base")
    base.OffloadingEvent = OffloadingEvent
    base.PrepareStoreOutput = PrepareStoreOutput
    base.ReqContext = object
    base.get_offload_group_idx = get_offload_group_idx
    common = mod("vllm.v1.kv_offload.cpu.common")
    common.CPULoadStoreSpec = CPULoadStoreSpec
    common.CPUOffloadingMetrics = types.SimpleNamespace(
        CPU_CACHE_USAGE_PERC="cpu_cache_usage_perc"
    )
    pol = mod("vllm.v1.kv_offload.cpu.policies.base")
    pol.BlockStatus = BlockStatus
    if "typing_extensions" not in sys.modules:
        try:
            import typing_extensions  # noqa: F401
        except ImportError:  # pragma: no cover
            te = mod("typing_extensions")

            def override(f):
                return f

            te.override = override


def load_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("p2_pool_under_test", TARGET)
    m = importlib.util.module_from_spec(spec)
    sys.modules["p2_pool_under_test"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- fakes


class FakePolicy:
    """最小 LRU：只实现 p2_pool 真正调用的那几个方法，并把 `evict` 的 n 记下来。"""

    def __init__(self):
        self.entries: list[tuple[bytes, BlockStatus]] = []
        self.evict_calls: list[int] = []

    def get(self, key):
        for k, b in self.entries:
            if k == key:
                return b
        return None

    def insert(self, key, block) -> None:
        self.entries.append((key, block))

    def remove(self, key) -> None:
        self.entries = [(k, b) for k, b in self.entries if k != key]

    def clear(self) -> None:
        self.entries.clear()

    def evict(self, n, protected):
        self.evict_calls.append(int(n))
        if n == 0:
            return []
        cands = [(k, b) for k, b in self.entries if k not in protected and b.ref_cnt == 0]
        if len(cands) < n:
            return None
        take = cands[:n]
        for item in take:
            self.entries.remove(item)
        return take


class FakeBase:
    """`PerGroupBPCManager` 的最小替身（只覆盖 p2_pool 用到的接口）。"""

    def __init__(self, num_blocks, bpc_by_group=None, **_kw):
        # 走 stub 模块里的那个（p2_pool 内部是从 vllm.v1.kv_offload.base 延迟导入的）
        from vllm.v1.kv_offload.base import get_offload_group_idx as _g  # noqa: PLC0415

        self._get_group = _g
        self._num_blocks = int(num_blocks)
        self._bpc_by_group = {int(k): int(v) for k, v in (bpc_by_group or {}).items()}
        self._policy = FakePolicy()
        self.counts = None
        self.store_threshold = 1
        self._num_evictable_cache_blocks = 0
        self._num_allocated_units = 0
        self._units_of_block: dict[int, list[int]] = {}
        self._num_write_pending_blocks = 0
        self.allocation_sizes_in_current_batch: list[int] = []
        self.events = None
        self.medium = "CPU"
        self._used_units_override = 0

    def bpc_of(self, key) -> int:
        return self._bpc_by_group.get(self._get_group(key), 1)

    def _get_load_store_spec(self, keys, blocks):
        return CPULoadStoreSpec([b.block_id for b in blocks])

    def get_stats(self):
        return None

    def reset_cache(self) -> None:
        self._policy.clear()


def make_manager(mod, num_blocks: int, weights: dict[int, int], bpc: dict[int, int]):
    cls = mod.make_quota_manager(FakeBase)
    mgr = cls(num_blocks=num_blocks, bpc_by_group=bpc, weights_by_group=weights)
    # 让条目可被淘汰（store 完成后 ref_cnt 0）
    return mgr



def raw_allocatable(mgr, group: int) -> int:
    """**绕过任何打桩**直接读底层行账本：free 列表 + 尚未用过的行。

    为什么不能直接用 mgr._p2_allocatable()：本文件为了制造"账本骗人"的
    场景，把它打桩成了常量 ⇒ 用它做判据就成了自证（第一版就是栽在这里）。
    """
    cap = mgr._p2_quota.get(group, 0)
    free = len(mgr._p2_free_rows.get(group) or ())
    return free + max(0, cap - mgr._p2_hi_rows.get(group, 0))


def mark_ready(mgr) -> None:
    for _k, b in mgr._policy.entries:
        b.ref_cnt = 0
    mgr._num_evictable_cache_blocks = len(mgr._policy.entries)


# ---------------------------------------------------------------- 用例


def case_over_eviction(mod) -> None:
    print("\n[① 过淘汰] full 组 bpc=8：淘汰 n 必须是 ceil(deficit/bpc)，不是 deficit")
    # 额度取成整数倍：num_blocks=153, Σw=9 ⇒ quota[0] = 136 = 17 × 8
    weights = {0: 8, 3: 1}
    bpc = {0: 8, 3: 1}
    mgr = make_manager(mod, 153, weights, bpc)
    print(f"  quota={mgr.p2_quota()}")
    q0 = mgr.p2_quota()[0]

    keys = [make_offload_key(f"k{i:06d}".encode(), 0) for i in range(q0 // 8)]
    out = mgr.prepare_store(keys, None)
    check(out is not None, "填满 full 组：17 条 8-unit 条目都能存下")
    check(
        mgr.p2_used_rows()[0] == q0,
        f"填满后 used[0] 恰好 == quota[0] ({q0})",
        f"used={mgr.p2_used_rows()[0]}",
    )
    mark_ready(mgr)
    check(raw_allocatable(mgr, 0) == 0, "此时可分配行数 == 0（真的满了）")

    mgr._policy.evict_calls.clear()
    new_key = make_offload_key(b"brandnew", 0)
    out2 = mgr.prepare_store([new_key], None)
    check(out2 is not None, "池满后仍能存下 1 条新条目（先淘汰再分配）")
    n_called = mgr._policy.evict_calls[0] if mgr._policy.evict_calls else None
    check(
        n_called == 1,
        "evict() 收到 n == ceil(8 unit / bpc 8) == 1（旧代码会传 8）",
        f"实际收到 {n_called}",
    )
    check(
        out2 is not None and len(out2.evicted_keys) == 1,
        "只淘汰 1 条（旧代码淘汰 8 条 = 64 unit，8 倍过淘汰）",
        f"实际淘汰 {len(out2.evicted_keys) if out2 else 'None'} 条",
    )
    check(
        mgr.p2_used_rows()[0] == q0,
        f"淘汰+分配后 used[0] 回到 quota（{q0}）",
        f"used={mgr.p2_used_rows()[0]}",
    )


def case_no_assert(mod) -> None:
    """② 账本口径说『还有位置』、分配器实际一行都拿不到时，**不许 assert 打死引擎**。

    ★ 这个状态就是生产日志里那一条：
        AssertionError: [P2] 配额不足：group=3 want=1 got=0 quota=51 used=51
      来源【推断】：组内有行被"泄漏"（`hi_rows` 已涨到配额，但那些行既不在 free
      列表里、`used` 也被减掉了）⇒ 旧代码用 `quota - used` 算 deficit 会得到 ≤ 0
      （于是**跳过淘汰**），而 `_p2_free_rows_of` 一行都拿不到 ⇒ assert。
      ⇒ 本用例**直接构造这个状态**，只依赖两个版本都有的字段，
        所以它能对**未修复的原版**给出判别力（负控必须 FAIL）。
    """
    print("\n[② 不 assert] 账本说『还有位置』、分配器说『没有』 ⇒ 返回 None，绝不抛异常")
    weights = {0: 8, 3: 1}
    bpc = {0: 8, 3: 1}
    mgr = make_manager(mod, 153, weights, bpc)
    q0 = mgr.p2_quota()[0]
    keys = [make_offload_key(f"k{i:06d}".encode(), 0) for i in range(q0 // 8)]
    mgr.prepare_store(keys, None)
    mark_ready(mgr)
    check(mgr.p2_used_rows()[0] == q0, f"先填满（used={q0}）")

    # ★ 构造"行泄漏"：账本归零 + free 清空 + hi 保持满配额
    mgr._p2_rows_used[0] = 0
    mgr._p2_free_rows[0] = []
    check(mgr._p2_hi_rows.get(0) == q0, "hi_rows 仍在满配额（= 行已分配但拿不回）")
    check(raw_allocatable(mgr, 0) == 0, "实际可分配行数 = 0")
    check(
        mgr._p2_quota.get(0, 0) - mgr._p2_rows_used.get(0, 0) > 0,
        "旧口径（quota - used）却说『还有位置』（= 缺陷入口）",
    )

    try:
        out = mgr.prepare_store([make_offload_key(b"willfail", 0)], None)
        raised = None
    except Exception as exc:  # noqa: BLE001
        out, raised = None, exc
    check(
        raised is None,
        "prepare_store 没有抛异常（原版在这里抛 AssertionError ⇒ EngineCore 死）",
        f"实际抛出 {type(raised).__name__}: {raised}",
    )
    # 判据不是"必须返回 None" —— 修复后这里能**自愈**：诚实淘汰 1 条旧条目
    # （ceil(8 unit / bpc 8) = 1 条）就够存下新条目了，比直接放弃更好。
    # 真正的判据是：不许崩 + 记账必须自洽。
    if out is None:
        check(True, "返回 None（本轮不 store，上游语义）")
        check(raw_allocatable(mgr, 0) == 0, "行账本未被污染（仍为 0）")
    else:
        check(
            len(out.keys_to_store) == 1,
            "自愈成功：淘汰 1 条旧条目后存下新条目（不是 8 条 = 旧代码的过淘汰）",
            f"实际 keys_to_store={len(out.keys_to_store)}",
        )
        check(
            len(out.evicted_keys) == 1,
            "只淘汰 1 条（旧代码这里会淘汰 8 条）",
            f"实际淘汰 {len(out.evicted_keys)}",
        )
        check(
            mgr.p2_used_rows()[0] == 8 and raw_allocatable(mgr, 0) == 0,
            "记账自洽：used=8、可分配=0（8 行被新条目占走）",
            f"used={mgr.p2_used_rows()[0]} raw_allocatable={raw_allocatable(mgr, 0)}",
        )


def case_partial_rollback(mod) -> None:
    """②b `_allocate_blocks` 半批失败时必须**回滚已分配的行**（否则行会泄漏）。

    直接调 `_allocate_blocks` 绕过淘汰循环，构造"第 1 条够、第 2 条不够"：
    free 列表正好够 1 条（8 行），`hi_rows` 已在满配额 ⇒ 第 2 条必然拿不到行。
      · 原版：`assert` ⇒ AssertionError，**且第 1 条的 8 行永久泄漏**
      · 新版：`_P2QuotaShort`，并把第 1 条的 8 行吐回 free 列表
    """
    print("\n[②b 半批回滚] _allocate_blocks 半批失败 ⇒ 已分配的行必须吐回去")
    mgr = make_manager(mod, 153, {0: 8, 3: 1}, {0: 8, 3: 1})
    q0 = mgr.p2_quota()[0]
    keys = [make_offload_key(f"k{i:06d}".encode(), 0) for i in range(q0 // 8)]
    mgr.prepare_store(keys, None)
    mark_ready(mgr)
    # 造出"正好够 1 条"的 free 列表；hi 保持满配额 ⇒ 第 2 条拿不到行
    mgr._p2_free_rows[0] = list(range(8))
    used_before = mgr.p2_used_rows()[0]
    alloc_before = raw_allocatable(mgr, 0)
    check(alloc_before == 8, f"构造完成：可分配 8 行（正好 1 条）")

    raised = None
    try:
        mgr._allocate_blocks(
            [make_offload_key(b"firstok", 0), make_offload_key(b"secondfail", 0)]
        )
    except Exception as exc:  # noqa: BLE001
        raised = exc
    check(raised is not None, "半批失败确实被拒（没有静默少分配）")
    check(
        type(raised).__name__ == "_P2QuotaShort",
        "失败类型是可控的 _P2QuotaShort（原版是 AssertionError）",
        f"实际 {type(raised).__name__ if raised else None}",
    )
    check(
        raw_allocatable(mgr, 0) == alloc_before,
        f"第 1 条的 8 行已回滚（可分配回到 {alloc_before}）",
        f"实际 {raw_allocatable(mgr, 0)}",
    )
    check(
        mgr.p2_used_rows()[0] == used_before,
        f"used[0] 回到 {used_before}（没有留下半批记账）",
        f"实际 {mgr.p2_used_rows()[0]}",
    )


def case_bpc_of_group(mod) -> None:
    print("\n[③ 辅助函数] _p2_bpc_of_group 取组内常量 bpc，缺项回落 1")
    mgr = make_manager(mod, 153, {0: 8, 3: 1}, {0: 8, 3: 1})
    if not hasattr(mgr, "_p2_bpc_of_group"):
        check(False, "管理器提供 _p2_bpc_of_group（原版没有 ⇒ 负控在此 FAIL）")
        return
    # 新方法必须与"绕过打桩"的原始口径一致（防它自己算错）
    mgr._p2_rows_used[0] = 0
    mgr._p2_free_rows[0] = []
    mgr._p2_hi_rows[0] = mgr._p2_quota[0] // 2
    check(
        mgr._p2_allocatable(0) == raw_allocatable(mgr, 0),
        "_p2_allocatable 与原始行账本口径一致",
        f"方法={mgr._p2_allocatable(0)} raw={raw_allocatable(mgr, 0)}",
    )
    check(mgr._p2_bpc_of_group(0) == 8, "group 0 -> 8")
    check(mgr._p2_bpc_of_group(3) == 1, "group 3 -> 1")
    check(mgr._p2_bpc_of_group(99) == 1, "未声明的组回落 1")


def main() -> int:
    print(f"被测文件：{TARGET}")
    try:
        mod = load_module()
    except Exception as exc:  # noqa: BLE001
        # ★ 绝不静默：加载失败会让后面全部用例"消失"，调用方会把它误读成
        #   "负控通过了"（2026-09-28 实测踩到：mktemp 不带 .py 后缀 ⇒
        #   spec_from_file_location 拿不到 loader ⇒ 这里抛 AttributeError）。
        check(False, "能加载被测文件", f"{type(exc).__name__}: {exc}")
        print(f"\n合计 {CHECKS[0]} 项，失败 {len(FAILS)} 项")
        for f in FAILS:
            print(f"  FAIL: {f}")
        return 2
    for _case in (
        case_over_eviction,
        case_no_assert,
        case_partial_rollback,
        case_bpc_of_group,
    ):
        try:
            _case(mod)
        except Exception as exc:  # noqa: BLE001
            # 未捕获异常本身就是一条 FAIL（负控会走这里）；不要让它掀掉整份报告
            check(False, f"{_case.__name__} 未抛未捕获异常", f"{type(exc).__name__}: {exc}")
    print(f"\n合计 {CHECKS[0]} 项，失败 {len(FAILS)} 项")
    if FAILS:
        for f in FAILS:
            print(f"  FAIL: {f}")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
