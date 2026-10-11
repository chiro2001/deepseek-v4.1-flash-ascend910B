#!/usr/bin/env python3
"""[ENGRAM-JIT-PLAN] `_local_owner_plan_numpy` 的 numba 实现（host 侧纯 CPU）。

被 `patch_engram_plan_jit.py` 安装到宿主侧 `probe_bneck/` 下，由打过补丁的
`engram_host_ws_opt.localowner_v2.py` 调用。门控 `V41_ENGRAM_JIT=1`（默认 0 = stock）。

## 为什么用「稳定计数排序」而不是 `np.argsort(kind="stable")`

stock 是 `np.argsort(owners, kind="stable")` —— stable 是关键语义：
`order` 决定「本 rank 收到的 my_ids 顺序」，而这个顺序必须与发送方逐位一致。
`owners` 的取值域只有 `size`（= query_group.size，本配置 8）个桶，
所以**计数排序**不仅 O(n)、还天然稳定，且比 argsort 更省常数。

## 逐位语义

* `owners = arr // shard_rows`；`shard_rows = ceil(rows / size)`
  ⇒ `arr < rows` 时恒有 `owners <= size-1`（证明：shard_rows >= rows/size
  ⇒ arr//shard_rows <= (rows-1)*size/rows < size）。
* 越界（`arr < 0 or arr >= rows`）⇒ 返回 -1，由 Python 侧抛 `IndexError`
  （与 stock 的 `raise IndexError("Engram hash ID outside table")` 同型同文本）。
* `starts` = `counts` 的**排他前缀和**（对应 `np.concatenate(([0], cumsum[:-1]))`）。
* `my_ids[j] = arr[order[starts[rank] + j]]`（对应 `arr[idx]`）。

## 一条与 stock 的**有意差异**（不是 bug）

stock 返回 4 元组的第一项 `flat = ids.reshape(-1)`；本实现返回由 `ids` 得到的
**连续 numpy 视图**（`flat_np`）。调用方全部把它丢弃（`_, order, counts, global_ids = planned`
或只取 `p[3]`），所以换成视图反而省掉 stock 每调一次的 torch 拷贝。
为兼容任何未预料的读者，补丁同时把该值包装成 torch 张量返回。
"""
from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit as _njit

    _AVAILABLE = True
except Exception:  # pragma: no cover
    _njit = None
    _AVAILABLE = False

PLAN_JIT = (os.environ.get("V41_ENGRAM_JIT", "0") == "1") and _AVAILABLE


def _njit_safe(**kw):
    if _njit is None:
        def _identity(fn):
            return fn

        return _identity
    return _njit(**kw)


@_njit_safe(cache=True, nogil=True, parallel=False)
def engram_plan_kernel(arr, rows, shard_rows, size, rank,
                       order, counts, starts, cursor, my_ids):
    """稳定计数排序 + 本 rank 切片。返回本 rank 的元素个数，越界返回 -1。

    arr      : int64 连续 1-D（hash id 展平）
    order    : int64[size=arr.shape[0]]，输出：按 owner 分桶后的原始下标（桶内保序）
    counts   : int64[size=size]，输出：每个 owner 的元素个数
    starts   : int64[size]，输出：counts 的排他前缀和
    cursor   : int64[size]，scratch
    my_ids   : int64[size=arr.shape[0]]，输出：本 rank 的 id（按 order 顺序）
    """
    n = arr.shape[0]
    for j in range(size):
        counts[j] = 0
    # 1) 校验 + 计数（顺带把 owners 存进 cursor 复用前先做校验遍历）
    for i in range(n):
        v = arr[i]
        if v < 0 or v >= rows:
            return -1
        counts[v // shard_rows] += 1
    # 2) starts = 排他前缀和
    acc = 0
    for j in range(size):
        starts[j] = acc
        acc += counts[j]
    # 3) 稳定落位
    for j in range(size):
        cursor[j] = starts[j]
    for i in range(n):
        o = arr[i] // shard_rows
        order[cursor[o]] = i
        cursor[o] += 1
    # 4) 本 rank 的切片
    m = counts[rank]
    s = starts[rank]
    for j in range(m):
        my_ids[j] = arr[order[s + j]]
    return m


def flatten_ids(ids):
    """torch CPU tensor -> 连续 1-D numpy 视图（尽量零拷贝）。"""
    a = ids.numpy()
    if a.ndim != 1 or not a.flags.c_contiguous:
        a = np.ascontiguousarray(a).reshape(-1)
    return a


def selftest():
    """合成自检：n=12、size=4、shard_rows=3 ⇒ owners=[0,0,0,1,1,1,2,2,2,3,3,3]。"""
    arr = np.arange(12, dtype=np.int64)
    order = np.zeros(12, dtype=np.int64)
    counts = np.zeros(4, dtype=np.int64)
    starts = np.zeros(4, dtype=np.int64)
    cursor = np.zeros(4, dtype=np.int64)
    my_ids = np.zeros(12, dtype=np.int64)
    m = engram_plan_kernel(arr, 12, 3, 4, 1, order, counts, starts, cursor, my_ids)
    if m != 3:
        raise RuntimeError(f"selftest m={m} (expect 3)")
    if list(counts) != [3, 3, 3, 3] or list(starts) != [0, 3, 6, 9]:
        raise RuntimeError(f"selftest counts={counts.tolist()} starts={starts.tolist()}")
    if list(order) != list(range(12)):
        raise RuntimeError(f"selftest order={order.tolist()}")
    if list(my_ids[:m]) != [3, 4, 5]:
        raise RuntimeError(f"selftest my_ids={my_ids[:m].tolist()}")
    # 越界
    bad = np.array([0, 12], dtype=np.int64)
    if engram_plan_kernel(bad, 12, 3, 4, 0, order[:2], counts, starts, cursor, my_ids[:2]) != -1:
        raise RuntimeError("selftest 越界未报 -1")
    return True
