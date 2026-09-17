#!/usr/bin/env python3
"""[ENGRAM-JIT-KERNEL] PagedNgramHistory.update 的 numba 实现（host 侧纯 CPU）。

被 `patch_engram_jit_hash.py` 安装到 `vllm_ascend/models/deepseek_v41/` 下，
由打过补丁的 `engram_hash.py` 调用。门控 `V41_ENGRAM_JIT=1`（默认 0 = stock）。

## 与 stock 的逐位对齐点

1. **缺页 bail-out**（最容易错的地方）：
   stock 的 `_vectorized_history` 在「镜像里缺页」时返回 None → 回退标量走法。
   dense 数组无法区分「缺页」与「槽位 = -1」，所以用 `page_present` 复刻该判据：
     * 先扫全部 `n*lookback` 个 flat page（已按 `torch.where(in_range, pages, pages[:, :1])`
       把越界槽位重定向到本行 shift0 的页）；
     * 有缺页 → 走标量走法；标量走法真的读到缺页 → 返回该页号，由 Python 侧抛 `KeyError`
       （与 `self.pages[page]` 的 KeyError 同型）。
2. **图像 token**：`compressed.masked_fill(~mask, -1)` → 写进页里当屏障；历史读遇 `<0` 即断。
3. **running-min**：`values.masked_fill(values.cummin(dim=1).values < 0, pad_id)`。
4. **prefill（n >= small_max）**：`active` 逐 shift 收窄；`page_ids` 全量存在性检查
   （与 `torch.stack([self.pages[p] for p in unique])` 一致：缺页即 KeyError。
   仅「报告哪个 page」的顺序从 sorted-unique 变成遍历序）。
5. **整数语义**：全程 int64；`%` 显式做负数补正以匹配 torch 的 floored 语义
   （实测 history 恒 >= 0、multipliers 恒为奇正数 ⇒ rolling 恒非负，补正是防御性冗余）。
6. **token_map 先查表后 mask**：保留 stock 对越界 token id 的 IndexError 行为。
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

ENGRAM_JIT = (os.environ.get("V41_ENGRAM_JIT", "0") == "1") and _AVAILABLE
# 初始页容量（按需增长）；128 槽 × 8B × 4096 = 4 MB
PAGE_CAP_INIT = int(os.environ.get("V41_ENGRAM_JIT_PAGES", "4096"))
_WARNED = [False]


def _njit_safe(**kw):
    if _njit is None:
        def _identity(fn):
            return fn

        return _identity
    return _njit(**kw)


@_njit_safe(cache=True, nogil=True, parallel=False)
def engram_update_kernel(
    token_map, input_ids, positions, request_ids, block_table,
    block_size, n, pad_id, image_token_id, image_pad_token_id,
    pages, page_present, multipliers, primes, offsets,
    lookback, n_heads, small_max,
    out_hashes, out_mask,
    hist, flat, prev, ir, vals, rows, pid, offb, prs, act,
):
    """页写 + n-gram 历史 + 哈希。返回 (err_page, oob_page, fell_back)。

    err_page >= 0 : 标量回退时读到缺页 → 调用方抛 KeyError
    oob_page >= 0 : 页号超出 pages 容量 → 调用方扩容后重跑
    fell_back     : 本次走了标量回退（用于 scalar_history_fallbacks 计数器）
    """
    cap = pages.shape[0]
    n_layers = primes.shape[0]
    n_shifts = primes.shape[1]          # lookback - 1
    err_page = np.int64(-1)
    oob_page = np.int64(-1)

    # ---------- 1) token_map + 图像 mask + 页写 ----------
    for i in range(n):
        tok = input_ids[i]
        comp = token_map[tok]           # 先查表：保留 stock 的越界 IndexError
        if tok == image_token_id or tok == image_pad_token_id:
            out_mask[i] = False
            comp = np.int64(-1)
        else:
            out_mask[i] = True
        pos = positions[i]
        page = block_table[request_ids[i], pos // block_size]
        if page < 0 or page >= cap:
            if page > oob_page:
                oob_page = page
            continue
        if page_present[page] == 0:
            for j in range(block_size):
                pages[page, j] = -1
            page_present[page] = 1
        pages[page, pos % block_size] = comp
    if oob_page >= 0:
        return err_page, oob_page, np.int64(0)

    # ---------- 2) n-gram 历史 ----------
    for r in range(n):
        for c in range(lookback):
            hist[r, c] = pad_id
    fell_back = np.int64(0)
    if n < small_max:
        # 小批（decode）：一次索引全部 (row, shift)，与 _vectorized_history 同构
        nflat = n * lookback
        for i in range(nflat):
            row = i // lookback
            sh = i - row * lookback
            p = positions[row] - sh
            prev[i] = p
            if p >= 0:
                ir[i] = 1
                blk = p // block_size
            else:
                ir[i] = 0
                blk = 0
            flat[i] = block_table[request_ids[row], blk]
        # 越界槽位重定向到本行 shift0 的页
        for i in range(nflat):
            if ir[i] == 0:
                flat[i] = flat[(i // lookback) * lookback]
        missing = 0
        for i in range(nflat):
            if page_present[flat[i]] == 0:
                missing = 1
                break
        if missing == 0:
            for i in range(nflat):
                s = prev[i] % block_size
                if s < 0:
                    s += block_size
                v = pages[flat[i], s]
                if ir[i] == 0 or v < 0:
                    vals[i] = -1
                else:
                    vals[i] = v
            for r in range(n):
                mn = np.int64(0)
                for sh in range(lookback):
                    v = vals[r * lookback + sh]
                    if sh == 0 or v < mn:
                        mn = v
                    if mn < 0:
                        hist[r, sh] = pad_id
                    else:
                        hist[r, sh] = v
        else:
            # 缺页回退：标量走法（遇缺页即上报，Python 侧抛 KeyError）
            fell_back = np.int64(1)
            for r in range(n):
                for sh in range(lookback):
                    p = positions[r] - sh
                    if p < 0:
                        break
                    page = flat[r * lookback + sh]
                    if page_present[page] == 0:
                        err_page = page
                        break
                    tok = pages[page, p % block_size]
                    if tok < 0:
                        break
                    hist[r, sh] = tok
                if err_page >= 0:
                    break
    else:
        # prefill：逐 shift 收窄 active（与 slab 路径同构）
        for r in range(n):
            act[r] = 1
        for sh in range(lookback):
            m = 0
            for r in range(n):
                if act[r] == 1 and positions[r] - sh >= 0:
                    rows[m] = r
                    m += 1
            if m == 0:
                break
            for j in range(m):
                r = rows[j]
                p = positions[r] - sh
                page = block_table[request_ids[r], p // block_size]
                if page < 0 or page >= cap:
                    if page > oob_page:
                        oob_page = page
                    continue
                pid[j] = page
                offb[j] = p % block_size
                if page_present[page] == 0:
                    err_page = page
                    break
            if oob_page >= 0:
                return err_page, oob_page, fell_back
            if err_page >= 0:
                break
            for j in range(m):
                v = pages[pid[j], offb[j]]
                if v >= 0:
                    hist[rows[j], sh] = v
                    prs[j] = 1
                else:
                    prs[j] = 0
            for j in range(m):
                act[rows[j]] = prs[j]

    # ---------- 3) 滚动 XOR + 取模哈希 ----------
    if err_page < 0:
        for r in range(n):
            for lay in range(n_layers):
                roll = hist[r, 0] * multipliers[lay, 0]
                for sh in range(1, lookback):
                    roll = roll ^ (hist[r, sh] * multipliers[lay, sh])
                    base = (sh - 1) * n_heads
                    for h in range(n_heads):
                        pm = primes[lay, sh - 1, h]
                        v = roll % pm
                        if v < 0:
                            v += pm
                        out_hashes[r, lay, base + h] = v + offsets[lay, base + h]
    return err_page, oob_page, fell_back


def selftest():
    """合成小输入跑一遍内核：首次编译 + 正确性自检。

    期望历史：row0=[3,pad,pad,pad]，row1=[5,3,pad,pad]（block_size=4, lookback=4）。
    """
    tm = np.arange(64, dtype=np.int64)
    ii = np.array([3, 5], dtype=np.int64)
    pos = np.array([0, 1], dtype=np.int64)
    req = np.array([0, 0], dtype=np.int64)
    bt = np.zeros((1, 4), dtype=np.int64)
    pages = np.full((8, 4), -1, np.int64)
    present = np.zeros(8, np.uint8)
    mult = np.ones((2, 4), np.int64)
    pr = np.full((2, 3, 8), 7, np.int64)
    off = np.zeros((2, 24), np.int64)
    oh = np.zeros((2, 2, 24), np.int64)
    om = np.zeros(2, np.bool_)
    hist = np.zeros((2, 4), np.int64)
    flat = np.zeros(8, np.int64)
    prev = np.zeros(8, np.int64)
    ir = np.zeros(8, np.uint8)
    vals = np.zeros(8, np.int64)
    rows = np.zeros(2, np.int64)
    pid = np.zeros(2, np.int64)
    offb = np.zeros(2, np.int64)
    prs = np.zeros(2, np.uint8)
    act = np.zeros(2, np.uint8)
    err, oob, _fb = engram_update_kernel(
        tm, ii, pos, req, bt, 4, 2, 1, -1, -2,
        pages, present, mult, pr, off, 4, 8, 16,
        oh, om, hist, flat, prev, ir, vals, rows, pid, offb, prs, act,
    )
    if err >= 0 or oob >= 0:
        raise RuntimeError(f"selftest status err={err} oob={oob}")
    if int(hist[0, 0]) != 3 or int(hist[1, 0]) != 5 or int(hist[1, 1]) != 3:
        raise RuntimeError(f"selftest history mismatch {hist.tolist()}")
    if int(hist[0, 1]) != 1 or int(hist[1, 2]) != 1:
        raise RuntimeError(f"selftest pad mismatch {hist.tolist()}")
    if not (bool(om[0]) and bool(om[1])):
        raise RuntimeError("selftest mask mismatch")
    return True
