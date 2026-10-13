# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decode-context-parallel (DCP) reconciliation patch.

**Only active when `V41_DCP` or `V41_DCP_ALLOW_CAPACITY_PROBE` is set** — the
module returns immediately otherwise, so the packaged A3/CED deployments are
bit-for-bit unaffected.

## 为什么需要这个补丁

## 只有 full-attention（MLA）平面在 DCP 下分片

能分片的前提是「每个 rank 只算自己那段 partial，LSE 合并后等于全局」。
V4.1 有四个平面，只有两个满足：

| 平面 | DCP 语义 | 判据 |
|---|---|---|
| `long_kv`（full MLA，压缩态） | **分片** 1/dcp | `max_memory_usage_bytes` 已除以 dcp；top-k 由 `cmp_sparse_indices` 表达（支持 -1 跳过）——**实测可用** |
| `indexer.k_cache` | **分片**（配合 remap） | 同 full；索引空间是压缩 token |
| `swa`（window 128） | **必须复制** | 见下 |
| `compressor.state_cache` | **复制**（本来就非 full） | `AscendCircularBufferSpec` 不是 `FullAttentionSpec` |

### 滑窗为什么必须复制（【实测】，2026-09-29）

`ori`（未压缩）路径在 A3 上被**硬绑**：

* `ori_sparse_indices`（显式给出可见键集合）是 **A5-only**：
  `sparse_flash_mla_tiling.cpp:1246` → `ori_sparse_indices is only supported on A5`；
* `ori_mask_mode` 必须为 4、`ori_win_left` 必须为 127（metadata EZ0024/EZ0027）；
* `ori_topk_length` 在 A2/A3 是保留参数（传非空即 tiling 失败）。

⇒ 滑窗只能表达成「以本地 KV 末端为右沿、宽 128 的**连续带**」。
分片后每个 rank 只有 1/8 的 KV，该带无法表达「全局窗口 ∩ 本 rank 分片」，
只有 `p ≡ 127 (mod 128)`（窗口恰为一个完整 128-chunk）时才精确。
实测反例：`T=129/130/201`、`dcp=2` → 相对误差 8.4% / 12.4% / 62%。

### 复制滑窗**不贵**（【实测】+【推断】）

滑窗是「最近 128 token + 在飞 token」的**滚动窗口**，**与序列长度无关**：
DCP1 与 DCP8 都是 130 块/组（`cdiv(127 + in_flight, 128) + 1`）。
若改成按序列分片，反而是 `L/dcp/128` = **1024 块**（1M 上下文）、
再加 halo 就是 2048 块 —— 比滚动窗口贵 8~16 倍。
⇒ 复制不仅「唯一可行」，也是**长上下文下最优**。

## 这个模块做什么

`KVCacheCoordinator.__init__` 把同一个 `dcp_world_size` 传给**所有** manager，
于是 `SlidingWindowManager` 也会 `block_size *= dcp`，与「复制态」矛盾。
这里在构造完成后把非 full 组的 manager 归 1（`resolve_group_dcp`）。
上游自己在 `find_longest_cache_hit` 里就是这么区分的
（`dcp_world_size if isinstance(spec, FullAttentionSpec) else 1`），
本模块只是把同一条规则补到分配路径上。
"""

import os

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

logger = init_logger(__name__)


def v41_dcp_active() -> bool:
    return os.environ.get("V41_DCP") == "1" or os.environ.get("V41_DCP_ALLOW_CAPACITY_PROBE") == "1"


def replicate_indexer() -> bool:
    """indexer K cache 是否按 DCP 复制（正确性所需）。

    · 打开（默认）：每个 rank 留一份全量 indexer K ⇒ 8 个 rank 各自独立算出
      **同一份全局 top-k**（零通信）⇒ cmp 路径能正确 LSE 合并。
      代价：index 面 ×dcp ⇒ pool 页 +22% ⇒ 容量 7.88× 降到 ~6.4×。
    · 关闭（**当前默认**）：indexer 按序列分片（沿用上游 1/dcp），容量 7.88×，
      但每个 rank 只能看到自己那 1/8 的候选 ⇒ 全局 top-k 不成立 ⇒ cmp 输出**错误**。
      仅供容量/带宽测量。

    ## ★ 2026-09-29 实测：复制路线有两个障碍，暂不足以交付
    （1）**容量代价 ~22%**：pool_bytes_per_block 540928 → **660480**（真机，
        run dcpcap_0929_132019 的 `[V41-DCP-DIAG]`），容量 7.88× → ~6.45×。
        机制：复制让 index 面 ×dcp，slot 0/1/2 的 `kv+index` 从 73856 涨到
        132096（> SWA 别名 131072），slot 3 涨到 264192。
    （2）**结构性阻碍**：slot 0/1/2 里 compressor state ring（FP32 32 行 ×
        1024 = 131072 B）必须**等长且连续**地填满槽位
        （`reshape_cache` 的 `"Aurora circular state must fill its slot with 32
        contiguous FP32 rows"`），而 `compressor_triton.py:665` 要求
        `state_cache.is_contiguous()`。槽位涨到 132096 后该断言直接否决
        （真机报错原文见 run dcpcap_0929_132019）。
        ⇒ 要让复制可用，必须先把 state ring 迁出该槽位或改 ring 内核的 stride 假设。

    ## 正确且不涨内存的替代路线（下一步）
    分布式 top-k：indexer 保持分片，各 rank 在自己分片上算分数并取 **本地 top-k
    （k = 全局 k = 512）**，all-gather `8×[T,512]` 的 (index, score)，本地归并取全局
    top-512，再走本模块的 remap。
    正确性：全局 top-512 的元素在其本 rank 内排名必 ≤512 ⇒ 本地保留 512 足够（标准结论）。
    代价：每 indexer 层每步 `T×8×512×8B = T×32KB`，8 层 ⇒ `T×256KB`/步
    （T=128 → 32 MB/步；T=8192 → 2 GB/chunk，prefill 需另想办法）。
    """
    return os.environ.get("V41_DCP_REPLICATE_INDEXER", "0") == "1"


def spec_is_dcp_sharded(spec) -> bool:
    """这个 cache 是否按序列在 DCP 下分片。

    与上游 `KVCacheCoordinator.find_longest_cache_hit` 同一条规则：
    只有 `FullAttentionSpec`（及其 MLA 子类）分片；滑窗/环状/包装类都是复制态。
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(spec_is_dcp_sharded(s) for s in spec.kv_cache_specs.values())
    return isinstance(spec, FullAttentionSpec)


def resolve_group_dcp(spec, dcp_world_size: int) -> int:
    """该 group 的 manager/block-table 应当使用的 DCP 度。"""
    return int(dcp_world_size) if spec_is_dcp_sharded(spec) else 1


def _force_replicated_managers(coordinator) -> None:
    groups = coordinator.kv_cache_config.kv_cache_groups
    managers = coordinator.single_type_managers
    # ★★★★★ [V41-KVGROUP-DIAG 2026-10-01 00:20] **每组一块表**（一次性启动日志）。
    #
    # 为什么必须打印：单卡 + 在线证据已把根因锁到"滑窗组的**块表只有 1 个有效项**，
    # 而算子按 `logicalIdx / 128` 逐块寻址"（写侧 SLOTTRACE 实测：T=904 时
    # 位置 0..127 → 块 3，位置 ≥128 → **块 0**；块表 [3,0,0,…]）。
    # 到底是谁算错块数，只能靠这张表判断：
    #   · `spec.block_size` ≠ `manager.block_size` ⇒ 分配与寻址用了两个不同的块大小；
    #   · `manager.dcp_world_size` 仍为 8 ⇒ 这一组的 `_force` 根本没生效。
    # 早退（`len(groups) != len(managers)`）以前是**静默**的，现在显式打出来。
    _diag = []
    for _i, _g in enumerate(groups):
        _sp = _g.kv_cache_spec
        _mgr = managers[_i] if _i < len(managers) else None
        _diag.append(
            "#%d %s spec_bs=%s spec_page=%s mgr=%s mgr_bs=%s mgr_dcp=%s sharded=%s"
            % (
                _i,
                type(_sp).__name__,
                getattr(_sp, "block_size", None),
                getattr(_sp, "page_size_bytes", None),
                type(_mgr).__name__ if _mgr is not None else "<无 manager>",
                getattr(_mgr, "block_size", None),
                getattr(_mgr, "dcp_world_size", None),
                spec_is_dcp_sharded(_sp),
            )
        )
    # ★ 用 print 而不是 logger：实测 `logger.warning` 在这一步**没有出现在
    #   serve.log**（2026-10-01 00:08，run dcpcap_1001_000409），而同一进程里
    #   其它 print/logger 行都在 ⇒ 这里必须以 print 落到 stdout 才能取证。
    print(
        "[V41-KVGROUP-DIAG] n_groups=%d n_managers=%d | %s"
        % (len(groups), len(managers), " || ".join(_diag)),
        flush=True,
    )
    if len(groups) != len(managers):
        print(
            "[V41-KVGROUP-DIAG] ★ groups/managers 数量不一致 ⇒ 复制态归一**整段跳过**"
            "（这正是「滑窗组只分到 1 块」的候选原因）",
            flush=True,
        )
        return
    fixed = []
    for group, manager in zip(groups, managers):
        if spec_is_dcp_sharded(group.kv_cache_spec):
            continue
        spec_block_size = group.kv_cache_spec.block_size
        if manager.dcp_world_size != 1 or manager.block_size != spec_block_size:
            manager.dcp_world_size = 1
            manager.pcp_world_size = 1
            manager.block_size = spec_block_size
            fixed.append((group.kv_cache_spec.__class__.__name__, spec_block_size))
    if fixed:
        logger.warning(
            "[V41-DCP] 复制态 KV group 的 manager 已归一到 DCP=1（spec_class, block_size）：%s",
            fixed,
        )


def apply() -> None:
    if not v41_dcp_active():
        return

    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    original_init = KVCacheCoordinator.__init__
    if getattr(original_init, "_v41_dcp_patched", False):
        return

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        _force_replicated_managers(self)

    patched_init._v41_dcp_patched = True  # type: ignore[attr-defined]
    KVCacheCoordinator.__init__ = patched_init  # type: ignore[method-assign]
    # =====================================================================
    # ★★★★★ [V41-ALLOC-FIX 2026-10-01 00:30] **真正的 hook：工厂函数**。
    #
    # 实测（run `dcpcap_1001_001109`）：`apply()` 打了补丁、那行日志在 9 个进程都出现，
    # 但 `_force_replicated_managers` 的 `[V41-KVGROUP-DIAG]` **一次都没打印**
    # ⇒ 在这条路径上 `KVCacheCoordinator.__init__` 根本没被走到
    # （`get_kv_cache_coordinator()` 直接实例化子类，子类不走被替换的基类 `__init__`）。
    #
    # 后果不是"少打一条日志"，而是**实打实的算错**：滑窗组 manager 的 `block_size`
    # 仍是 `128*dcp = 1024` ⇒ 一个 904-token 请求只分到 **1 块**；而寻址侧
    # （`block_table.py`，已按复制态归一）按 128 逐块寻址 ⇒ 位置 ≥128 的读写
    # 全部落到"块 0"（SLOTTRACE 实测：`ori_bt=[3,0,0,…]`、写侧 `pos≥128 → blk 0`）。
    #
    # ⇒ 这里把**工厂函数**也包一层，保证无论走哪个子类都能拿到 coordinator 实例。
    # =====================================================================
    from vllm.v1.core import kv_cache_coordinator as _kvcc

    _orig_factory = _kvcc.get_kv_cache_coordinator
    if not getattr(_orig_factory, "_v41_dcp_patched", False):
        def _patched_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            coordinator = _orig_factory(*args, **kwargs)
            try:
                _force_replicated_managers(coordinator)
            except Exception as exc:  # noqa: BLE001
                print("[V41-KVGROUP-DIAG] ★ 复制态归一失败：%r" % (exc,), flush=True)
            return coordinator

        _patched_factory._v41_dcp_patched = True  # type: ignore[attr-defined]
        _kvcc.get_kv_cache_coordinator = _patched_factory  # type: ignore[assignment]
        # ★★ [V41-ALLOC-FIX-2 00:45] **必须连"已 import 的那个名字"一起换**。
        #   实测（run `dcpcap_1001_001639`）：只换模块属性时，新加的打印出现了，
        #   但 `[V41-KVGROUP-DIAG]` 依旧为 0 条 ⇒ `kv_cache_manager.py` 里是
        #   `from ... import get_kv_cache_coordinator`（模块级绑定），替换模块属性
        #   改不到它已经绑好的本地名。所以把**两个**位置都替换掉。
        _patched_names = []
        try:
            from vllm.v1.core import kv_cache_manager as _kvcm

            if getattr(_kvcm, "get_kv_cache_coordinator", None) is _orig_factory:
                _kvcm.get_kv_cache_coordinator = _patched_factory  # type: ignore[assignment]
                _patched_names.append("kv_cache_manager")
        except Exception as exc:  # noqa: BLE001
            print("[V41-KVGROUP-DIAG] ★ 替换 kv_cache_manager 的引用失败：%r" % (exc,), flush=True)
        print(
            "[V41-DCP] get_kv_cache_coordinator 已打补丁（复制态组按 DCP=1 分配）| 已替换：%s"
            % (["kv_cache_coordinator"] + _patched_names),
            flush=True,
        )
    # =====================================================================
    # ★★★★★ [V41-ALLOC-FIX-3 2026-10-01 00:55] **钩住 vllm-ascend 自己的
    # coordinator 子类**。
    #
    # 实测（run `dcpcap_1001_002049`）：工厂函数的两个绑定都替换了、打印也出现了，
    # 但 `[V41-KVGROUP-DIAG]` 仍为 0 条。原因在
    # `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:590/663`：
    # DeepSeek-V4 命中 `_is_deepseek_v4_kv_cache_config` 分支后**直接**
    # `return AscendHybridKVCacheCoordinator(...)` —— 既不调用被我们包住的
    # `_orig_get_kv_cache_coordinator`，也不经过基类 `__init__` 那条路。
    # ⇒ 想让「滑窗组 manager 的 block_size 归一」生效，只能钩这个子类本身。
    # =====================================================================
    try:
        from vllm_ascend.patch.platform import patch_kv_cache_coordinator as _akvc

        _wrapped = []
        for _cls_name in ("AscendHybridKVCacheCoordinator", "AscendUnitaryKVCacheCoordinator"):
            _cls = getattr(_akvc, _cls_name, None)
            if _cls is None:
                continue
            _c_init = _cls.__init__
            if getattr(_c_init, "_v41_dcp_patched", False):
                continue

            def _mk(orig):  # type: ignore[no-untyped-def]
                def _pinit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    orig(self, *args, **kwargs)
                    try:
                        _force_replicated_managers(self)
                    except Exception as exc:  # noqa: BLE001
                        print("[V41-KVGROUP-DIAG] ★ 归一失败：%r" % (exc,), flush=True)

                _pinit._v41_dcp_patched = True  # type: ignore[attr-defined]
                return _pinit

            _cls.__init__ = _mk(_c_init)  # type: ignore[method-assign]
            _wrapped.append(_cls_name)
        print("[V41-DCP] ascend coordinator 子类已打补丁：%s" % (_wrapped,), flush=True)
    except Exception as _exc:  # noqa: BLE001
        print("[V41-DCP] ★ ascend coordinator 子类打补丁失败：%r" % (_exc,), flush=True)
    # =====================================================================
    # ★★★★★ [V41-ALLOC-FIX-4 2026-10-01 01:05] **最后一个确定性 hook：
    # `KVCacheManager.__init__`**。
    #
    # 前三轮钩子（基类 `__init__`、工厂函数两处绑定、ascend 子类 `__init__`）的
    # 打印都出现了，但 `[V41-KVGROUP-DIAG]` 仍为 0 条（run `dcpcap_1001_002559`）
    # ⇒ 那些路径**都没有真正构造出我们以为的那个对象**。而 `KVCacheManager`
    # 是这个链路上**唯一确定会被构造**的类（`self.coordinator = get_kv_cache_coordinator(...)`
    # 就在它的 `__init__` 里，`vllm/v1/core/kv_cache_manager.py:151`）。
    # 直接在这里对 `self.coordinator` 做归一，绕开所有间接层。
    # =====================================================================
    try:
        from vllm.v1.core import kv_cache_manager as _kvcm2

        _KM = getattr(_kvcm2, "KVCacheManager", None)
        if _KM is not None and not getattr(_KM.__init__, "_v41_dcp_patched", False):
            _km_init = _KM.__init__

            def _km_pinit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                _km_init(self, *args, **kwargs)
                try:
                    _force_replicated_managers(self.coordinator)
                except Exception as exc:  # noqa: BLE001
                    print("[V41-KVGROUP-DIAG] ★ 归一失败：%r" % (exc,), flush=True)

            _km_pinit._v41_dcp_patched = True  # type: ignore[attr-defined]
            _KM.__init__ = _km_pinit  # type: ignore[method-assign]
            print("[V41-DCP] KVCacheManager.__init__ 已打补丁（分配侧归一）", flush=True)
    except Exception as _exc:  # noqa: BLE001
        print("[V41-DCP] ★ KVCacheManager 打补丁失败：%r" % (_exc,), flush=True)
    logger.warning(
        "[V41-DCP] KVCacheCoordinator 已打补丁：复制态（非 full-attention）KV group "
        "按 DCP=1 分配与寻址 — DEVELOPMENT BUILD"
    )


apply()
