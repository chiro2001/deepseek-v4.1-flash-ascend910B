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

_V41_DCP_FLAG_CACHE = {"t": None, "v": {}}


def _perf_flags_v41_dcp() -> dict:
    """复用 `dsa_v41._perf_flags` 的**缓存**（每步刷新一次）。

    ★ 性能：本函数在 `remap_sparse_indices` 里每层调用一次；原实现自带
    `os.stat` ⇒ EAGER 解码下每步多出 38 次系统调用。惰性导入避免模块级
    循环依赖（`dsa_v41` 在函数内才导入本模块）。
    """
    from vllm_ascend.attention.dsa_v41 import _perf_flags

    return _perf_flags()


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
    replicated: bool = False,
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

    # =====================================================================
    # ★★★★★★ [V41-REPL-LAYOUT 2026-09-30 14:05] **复制态的行号必须与写侧同源**。
    #
    # 复制态（`V41_DCP_REPLICATE_INDEXER=1`）下**每个 rank 都存全量 indexer K**，
    # 页内布局由 metadata builder 的写侧决定
    # （`dsa_v41.py` 的 `index_is_replicated` 分支，2026-09-30 实测）：
    #     块列    = g // (dcp·B')          （`B' = storage_block_size = block_size/ratio`）
    #     页内偏移 = (g % (dcp·B') // B')·B' + g % B'  ≡  g % (dcp·B')
    # ⇒ 读侧行号必须就是 **`g % (dcp·B')`**。
    #
    # 而原来的读侧用 `compressed_local()`（按 `block_size`/`interleave` 推导的
    # **非复制**长 KV 布局）：
    #     B=128, I=32, dcp=8, ratio=2 ⇒ g=16 → 行 0，g=128 → 行 16
    # 与写侧（g=16 → 行 16，g=128 → 行 128）**不一致** ⇒ 读到的行不是写进去的内容。
    # 实测后果：L=2000/8000/16000 长针答案退化成乱码（'#/issues Minim'），
    # 且同 prompt 两次结果不同（run1 1/6、run2 2/6）。
    #
    # 复制态下所有权归**所有** rank（每个 rank 都有全量副本）⇒ 不再做 owner 过滤。
    # =====================================================================
    if replicated:
        # ---------------------------------------------------------------------
        # ★★★★★★ [V41-REPLMAP-AB 2026-09-30 14:40] **两套坐标系的 A/B 开关**。
        #
        # SMLA 的 `cmp_kv` 读的是 **long_kv 面**（`_native_attention` 里
        # `source_cache = no_compile_layers[long_kv_source_prefix].kv_cache[0]`，
        # 而 long_kv 面是**按序列分片**的，每 rank 只有 1/dcp）。
        # 复制只发生在 **index_k 面**（`_cache_plane_sizes`/`reshape_cache` 只对
        # `DeepseekV41IndexerSpec` 乘 dcp）。两个面的页内行号**不是同一套**：
        #   · index_k（复制面）行号 = g % (dcp·B')        ← 写侧 `face_offset`
        #   · long_kv（分片面）行号 = `compressed_local(g)` ← 写侧 `compressed_slot_mapping`
        # ⇒ 送给 SMLA 的 `cmp_sparse_indices` **必须**是后者。
        #
        # 实测（run dcpcap_0930_140418，T=564 层2）：用复制面行号时
        # `n_valid=79524 / sum=7435541 / max=281` 与 DCP1 **逐一相同**
        # —— 因为 `dcp·B' = 512` 大于当时的最大压缩索引 281，`g % 512 == g`
        # ⇒ 该映射是**恒等**，等于完全没重映射 ⇒ SMLA 拿全局索引去读分片
        # long_kv ⇒ 读到别人的行。
        #
        # 文件开关 `replmap`：`0` = long_kv 局部坐标（**默认，应当是正解**）；
        #                       `1` = index_k 复制面行号（旧行为，用于 A/B）。
        # ---------------------------------------------------------------------
        _replmap_mode = _perf_flags_v41_dcp().get("replmap", "0")
        if _replmap_mode != "1":
            # 正解：把全局压缩索引转到**本 rank long_kv 的局部行号**，
            # 并只保留本 rank 真正拥有的那些（owner 过滤）。
            _idx = indices.to(torch.int64)
            _owner = compressed_owner(_idx, interleave, ratio, dcp_size)
            _local = compressed_local(_idx, block_size, interleave, ratio, dcp_size)
            _valid = (_idx >= 0) & (_owner == dcp_rank)
            _remapped = torch.where(
                _valid, _local, torch.full_like(_local, -1)
            )
        else:
            _storage = max(1, int(block_size) // max(1, int(ratio)))
            _span = _storage * int(dcp_size)
            _idx = indices.to(torch.int64)
            _valid = _idx >= 0
            _local = _idx.remainder(_span)
            _remapped = torch.where(
                _valid, _local, torch.full_like(_local, -1)
            )
        _width = indices.shape[-1]
        _order = torch.arange(_width, device=indices.device).expand_as(_remapped)
        _keys = _order + (~_valid).to(torch.int64) * _width
        _pack = torch.argsort(_keys, dim=-1, stable=True)
        return torch.gather(_remapped, -1, _pack).to(indices.dtype)

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


def local_visible_positions(
    positions: torch.Tensor,
    *,
    interleave: int,
    ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """把**全局** query 位置转换成「本 rank 局部可见上界」编码后的位置。

    ## 这个函数修的是什么

    `prepare_indexer_indices`（`ops/triton/prepare_indexer_indices.py`）用

        visible = (positions + 1) // COMPRESS_RATIO
        valid   = (selected >= 0) & (selected < visible)

    过滤 top-k，其中 `positions` 是**全局** query 位置。

    * indexer K 缓存**复制**时，`selected` 是**全局**压缩索引 ⇒ 比较成立；
    * indexer K 缓存**分片**时（DCP 下每个 rank 只有 1/dcp），`selected` 是
      **本 rank 局部**压缩索引 ⇒ 拿它去和全局界比较**坐标系不一致**。

    rank r>0 的局部索引 `j` 对应全局压缩索引 `g = super(j) + r·Is + (j mod Is)`，
    远大于 `j` 本身 ⇒ `j < 全局界` 几乎恒真 ⇒ **保留大量未来键**，破坏因果性。

    ## 实测影响（离线对拍）
    对 `rank∈[0,8) × p∈12 个位置 × j∈[0,40)` 逐项与"正确判据
    `g(j) <= (p+1)//ratio - 1`"对比：
    * 旧写法（直接比）错判 **1803** 次；
    * 本函数错判 **0** 次。
    最刺眼的例子：rank=5、p=40 时全局界=20，rank 5 的 g 从 80 起（本不该可见），
    旧写法却把 j=0..5 全部判为可见 ⇒ 让位置 40 的 query 看到位置 160+ 的键。

    ## 做法
    令 `vlc = ` 本 rank 在全局界内**实际可见的局部压缩 token 数**（即局部上界），
    再返回 `p' = ratio·vlc − 1`，于是 `(p'+1)//ratio == vlc` ⇒ 过滤退化成
    `j < vlc`，正好是局部坐标系下的正确判据。
    `vlc` 用 `local_compressed_len` 算：它本来就统计"g < G 且属本 rank"的个数。
    """
    p = positions.to(torch.int64)
    g_bound = torch.div(p + 1, ratio, rounding_mode="floor")
    vlc = local_compressed_len(
        g_bound * ratio,
        interleave=interleave,
        ratio=ratio,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    return (ratio * vlc - 1).to(positions.dtype)
