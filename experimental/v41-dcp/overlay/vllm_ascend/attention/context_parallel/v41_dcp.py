# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decode-context-parallel (DCP) primitives.

All functions here are pure and side-effect free so they can be unit tested
offline (see `docs/SFA-DCP-PORTING-MANUAL-20260929.md` for the derivation).

## 坐标系统（推导来源：`AscendBlockTable._compute_dcp_slot_mapping`）

写侧对**未压缩**位置 `pos`：

```
owner(pos)  = (pos // I) % dcp                       # I = cp_kv_cache_interleave_size
local(pos)  = (pos // (B*dcp)) * B
            + ((pos % (B*dcp)) // (dcp*I)) * I
            + (pos % I)                              # B = 逻辑 block_size
slot(pos)   = block_table[local(pos) // B] * B + local(pos) % B
```

即：全局按 `B*dcp` 个 token 划一个「超块」，超块内 rank r 取 `dcp` 段各 `I` 个，
压缩成连续的 `local` 索引。`I | B` 时上面的 `owner` 化简为 `(pos // I) % dcp`。

## 压缩平面（compress_ratio = 2）

压缩 token `g` 覆盖未压缩 `[ratio*g, ratio*g+ratio)`。因为 `ratio | I`，
一整组必然落在同一个 `I` 段内 ⇒ 压缩组**不跨 rank**。写侧
`compressed_slot_mapping(slot, ratio) = slot // ratio`（仅当 `(slot+1) % ratio == 0`），
而 `local` 在 `//ratio` 之后正好等于「把 B、I 都除以 ratio」的同一公式：

```
local_compressed(g) = local(pos = ratio*g + ratio - 1) // ratio
                    = (g // (B'*dcp))*B' + ((g % (B'*dcp)) // (dcp*I'))*I' + (g % I')
其中 B' = B/ratio, I' = I/ratio
```

⇒ **压缩域的 remap 就是同一个公式换成 `(B', I')`**，这就是必须让 `ratio | I` 的原因。
"""

from __future__ import annotations

import torch


def uncompressed_owner(pos: torch.Tensor, interleave: int, dcp_size: int) -> torch.Tensor:
    """Rank owning the uncompressed token at ``pos`` (0-based, global)."""
    return (pos // interleave) % dcp_size


def uncompressed_local(pos: torch.Tensor, block_size: int, interleave: int, dcp_size: int) -> torch.Tensor:
    """Rank-local index of the uncompressed token at global ``pos``."""
    super_block = block_size * dcp_size
    return (
        (pos // super_block) * block_size
        + ((pos % super_block) // (dcp_size * interleave)) * interleave
        + (pos % interleave)
    )


def compressed_owner(idx: torch.Tensor, interleave: int, ratio: int, dcp_size: int) -> torch.Tensor:
    """Rank owning compressed token ``idx``."""
    return uncompressed_owner(idx * ratio + (ratio - 1), interleave, dcp_size)


def compressed_local(idx: torch.Tensor, block_size: int, interleave: int, ratio: int, dcp_size: int) -> torch.Tensor:
    """Rank-local compressed index of global compressed token ``idx``."""
    return uncompressed_local(
        idx * ratio + (ratio - 1),
        block_size,
        interleave,
        dcp_size,
    ) // ratio


def remap_sparse_indices(
    indices: torch.Tensor,
    *,
    block_size: int,
    interleave: int,
    ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Remap global compressed top-k indices to this rank's local coordinates.

    ``indices`` is ``[T, K]`` int (or int32) with optional ``-1`` padding.
    Returns the same shape: owned indices rewritten to local coordinates,
    everything else ``-1``, with the valid entries stably compacted to the
    front (the fused kernel stops at the first ``-1`` per row in the compacted
    view, and the compaction keeps the top-k ordering intact).

    ``block_size``/``interleave`` are the **uncompressed** plane values
    (``B`` and ``I``); ``ratio`` converts them internally.  Callers pass
    ``vllm_config.cache_config.block_size`` and
    ``parallel_config.cp_kv_cache_interleave_size`` verbatim.
    """
    if dcp_size <= 1:
        return indices
    if indices.numel() == 0:
        return indices

    idx = indices.to(torch.int64)
    owner = compressed_owner(idx, interleave, ratio, dcp_size)
    local = compressed_local(idx, block_size, interleave, ratio, dcp_size)
    valid = (idx >= 0) & (owner == dcp_rank)
    remapped = torch.where(valid, local, torch.full_like(local, -1))

    # Stable compaction: sort by (rank_within_row) where invalid entries get a
    # large key so they move to the tail while keeping relative order.
    width = indices.shape[-1]
    order = torch.arange(width, device=indices.device).expand_as(remapped)
    keys = order + (~valid).to(torch.int64) * width
    pack_order = torch.argsort(keys, dim=-1, stable=True)
    return torch.gather(remapped, -1, pack_order).to(indices.dtype)


def build_replicated_local_index(
    num_tokens: int,
    *,
    block_size: int,
    interleave: int,
    dcp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-rank local index for *every* global token (``dcp``-replicated planes).

    Used by the indexer K cache: every rank keeps a full copy of the sequence,
    so each rank must write all tokens -- but into its own physical page whose
    capacity is ``block_size * dcp`` token slots per scheduler block, laid out
    as ``local_physical = local_block * (block_size*dcp) + offset_in_block``
    where ``local_block = pos // (block_size*dcp)`` and
    ``offset_in_block = (pos % interleave) + ((pos % (block_size*dcp)) // (dcp*interleave)) * interleave``.
    """
    pos = torch.arange(num_tokens, dtype=torch.int64, device=device)
    super_block = block_size * dcp_size
    return (pos // super_block) * (super_block) + (
        (pos % super_block) // (dcp_size * interleave)
    ) * interleave + (pos % interleave)


def local_compressed_len(
    seq_lens: torch.Tensor,
    *,
    interleave: int,
    ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """本 rank 实际拥有的**压缩行数**（与写侧 `compressed_slot_mapping` 完全同源）。

    写侧：未压缩位置 `pos` 只有在 `(pos+1) % ratio == 0` 且 `owner(pos) == rank`
    时才写出一行压缩槽。因此本 rank 拥有的压缩 token 就是
    `{ g : owner(g·ratio + ratio - 1) == rank }`，而
    `owner(pos) = (pos // I) % dcp` ⇒ 以 `Is = I/ratio` 为块、块号模 dcp 即归属。

    所以：把 `G = L // ratio` 个压缩 token 按 `Is` 切块，块号 `% dcp == rank` 的归本 rank。

    ## 为什么必须算这个（而不是直接用全局长度）

    本 rank 的 long_kv / index_k 物理块只装 **1/dcp** 的序列。若把**全局**压缩长度
    当 `seqused_cmp_kv` / `seqused_k` 传给算子，算子会去读本 rank 的块表里
    **不存在的行** ⇒ 读到 null/邻块数据 ⇒ 返回一个**有限的** LSE ⇒
    在 `Σ e^{L_r}·O_r / Σ e^{L_r}` 合并里按错误权重污染结果。

    ★ 极端情形（实测复现的输出错误）：prompt 只有 17 个 token 时
    `owner(pos) = (pos//32) % 8` ⇒ 只有 rank 0 拥有 KV，rank 1-7 的本地长度是 **0**。
    全局长度会告诉 rank 1-7 "你有 8 行"，它们就会去读空块。
    """
    if dcp_size <= 1:
        return torch.div(seq_lens, ratio, rounding_mode="floor")
    isize = interleave // ratio
    if isize <= 0:
        raise ValueError(
            f"interleave({interleave}) 必须能被 compress_ratio({ratio}) 整除；"
            "否则压缩组会跨 rank（见本模块 docstring 的推导）"
        )
    g_total = torch.div(seq_lens, ratio, rounding_mode="floor")
    cycle = isize * dcp_size
    cycles = torch.div(g_total, cycle, rounding_mode="floor")
    rem = g_total - cycles * cycle
    full_blocks = torch.div(rem, isize, rounding_mode="floor")
    partial = rem - full_blocks * isize
    zero = torch.zeros_like(rem)
    own = cycles * isize + torch.where(
        full_blocks > dcp_rank,
        torch.full_like(rem, isize),
        torch.where(full_blocks == dcp_rank, partial, zero),
    )
    return own
