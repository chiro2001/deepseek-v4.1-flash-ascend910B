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
    if len(groups) != len(managers):
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
    logger.warning(
        "[V41-DCP] KVCacheCoordinator 已打补丁：复制态（非 full-attention）KV group "
        "按 DCP=1 分配与寻址 — DEVELOPMENT BUILD"
    )


apply()
