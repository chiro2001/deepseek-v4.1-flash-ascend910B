#!/usr/bin/env python3
"""ENGRAM × 卸载：**精确**修补模块（草案，未合入任何生产文件）。

★ 本文件 = 交付时**会被挂进容器**的那一份（`engram_hash.py` 按 env 门控 import 它）：
      -v $A2F/engram_repair.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_repair.py:ro
  因此 `tests/test_true_tokens_repair.py` 直接 import 的就是**出货件本身**。

用途：把「从 DRAM 取回的前缀」那 1~3 个边界 token 的**真实压缩 id** 回填进
`PagedNgramHistory` 的页镜像，使 `update()` 给出的 n-gram 历史与冷启动
（fill 轮）逐值相同。**不改变 kernel**（`engram_jit_kernel.py` 一行都不动）。

合入方式（见 engram_hash.true_tokens.diff）：
  ① 影子包在 `VLLM_V41_ENGRAM_TRUE_TOKENS != 0` 时挂本文件（缺文件 **die**，不静默降级）；
  ② `engram_hash.py` 顶部按同一个 env 门控 `from .engram_repair import ...`；
  ③ `update()` 多收一个 `prev_tok`，`_engram_update_jit()` 在
     **调用 numba kernel 之前**调 `apply_repairs()`。

★ 为什么必须在 kernel 之前：kernel 的 step-1 对「本步第一次写到的页」会
  `for j in range(block_size): pages[page, j] = -1`（行复位）再写。若把回填放在
  kernel 之后，step-1 的复位会**擦掉**刚回填的槽位。放在 kernel 之前则相反：
  我们先把页标成 `present=1`，kernel 的 step-1 就不会再复位它（见
  engram_jit_kernel.py:90-94 的 `if page_present[page] == 0:` 分支）。

★ 坐标系：本函数的 (page, slot) 计算与 kernel 的**读路径逐字一致**
  （`engram_jit_kernel.py:117` 与 `:155`/`:180`）：
      q    = positions[row] - shift
      page = block_table[request_ids[row], q // block_size]
      slot = q % block_size
  ⇒ 因此本修补**不依赖**对页号语义的任何假设（页号是 SWA 组的物理块号、
    还是 SWA-trim 之后的"段尾块号"都无所谓）：写的地方 == 读的地方。

★ 两种病都治：
  (a) **缺页**（`page_present[page] == 0`）—— 今天会 `raise KeyError(page)`
      （`engram_hash.py:463`）。mode>=1 时回填真值并 `present=1`。
  (b) **陈旧页**（`page_present[page] == 1` 但内容是**上一任占用者**的 token）
      —— 今天**静默算错**（不报错、不计数）。mode>=2 时用真值覆写该槽位。
"""

from __future__ import annotations

import numpy as np


def plan_repair_slots(positions, request_ids, lookback):
    """返回本步需要检查/回填的槽位 ``(rows, shifts)``（**不含 shift=0**）。

    规则：只覆盖「本步 kernel step-1 不会写到」的槽位，即
        q = positions[row] - shift  <  first_pos[req(row)]
    其中 ``first_pos[req]`` = 该请求在本步的**首个**位置。

    为什么只覆盖这些（证明）：
      * vLLM 每步给一个请求调度的 token 是**一段连续 span**，且本步 kernel 的
        step-1 会把这一段里**所有**位置写进镜像；
      * 于是对请求内的第 i 行、shift ≤ i 的槽位：q = p0+i-sh ≥ p0 ⇒ 本步已写；
      * 只有 ``q < p0``（即跨到上一步/上一块的窗口）才可能既不在本步写入集合里、
        又被读者读到 ⇒ 只有这些槽位可能是缺页/陈旧。
      * 代价因此被限死：**每请求 ≤ (lookback-1)·lookback/2 = 6 个槽位**
        （lookback=4 时：第 0 行 3 个 + 第 1 行 2 个 + 第 2 行 1 个），
        **与 batch 大小无关**（不是每行都扫）。由 `test_repair_plan.py` 逐用例核对。

    ★ 若发现某个请求的行**不连续**（上述假设被破坏，例如未来调度器改成
      一个请求多个 span），本函数退化为「该请求所有 shift>=1 的槽位」全扫
      （安全、只是更慢），由 ``tests/test_repair_plan.py`` 用暴力枚举对照验证。
    """
    n = int(len(positions))
    if n == 0:
        return [], []

    first_pos: dict[int, int] = {}
    contiguous = True
    prev_req = None
    prev_pos = None
    for r in range(n):
        req = int(request_ids[r])
        pos = int(positions[r])
        if req != prev_req:
            first_pos[req] = pos
        else:
            if prev_pos is not None and pos != prev_pos + 1:
                contiguous = False
        prev_req, prev_pos = req, pos

    rows: list[int] = []
    shifts: list[int] = []
    for r in range(n):
        pos = int(positions[r])
        req = int(request_ids[r])
        base = first_pos[req] if contiguous else pos  # 退化：q < pos ⇒ 所有 shift>=1
        for sh in range(1, int(lookback)):
            q = pos - sh
            if q < 0:
                continue
            if q < base:
                rows.append(r)
                shifts.append(sh)
    return rows, shifts


def apply_repairs(
    pages,
    present,
    token_map,
    block_table,
    request_ids,
    positions,
    prev_tok,
    block_size,
    lookback,
    image_token_id,
    image_pad_token_id,
    mode,
    stats,
):
    """按**读者坐标**回填（mode>=1）/覆写（mode>=2）镜像槽位。

    参数
    ----
    pages, present : JIT 的稠密镜像（``np.int64[cap, block_size]`` / ``np.uint8[cap]``）
                     —— 即 ``PagedNgramHistory._jit_pages`` / ``._jit_page_present``。
    token_map      : ``PagedNgramHistory.token_map.numpy()``（raw token id -> 压缩 id）。
    block_table    : 本步的 block table（CPU tensor / numpy 均可，按 2-D 索引）。
    prev_tok       : ``np.int64[n, lookback]``，``prev_tok[row, shift]`` = 该行
                     ``positions[row]-shift`` 处的**真实 token id**，``-1`` = 不可得
                     （越界 / 超出该请求已提交长度 / runner 未发布）⇒ 不修。
    mode           : 0 = 只统计（不写任何字节）；1 = 缺页回填；2 = 覆写陈旧槽位。
    stats          : 计数 dict（就地累加，键见下）。

    返回
    ----
    oob_page : 需要扩容的最大页号（>=0），否则 -1。**调用方必须先扩容再调 kernel**
               （否则 kernel 的读路径会越界访问 `page_present`）。

    计数键：``absent``（本会被 KeyError 的槽位）/ ``filled`` / ``mismatch``
    （镜像有值但与真值不同 ⇒ 静默错误）/ ``overwrote`` / ``unavailable`` / ``oob``。
    """
    rows, shifts = plan_repair_slots(positions, request_ids, lookback)
    oob_page = -1
    cap = int(pages.shape[0])
    bs = int(block_size)
    tm = token_map
    # ★ 本步**首次接触**的页：`was_absent=True` 表示"整行由我们初始化" ⇒ 该页后续
    #   槽位必须**直接写**（它们是同一次初始化的一部分），不能走"和镜像比对"的分支
    #   ——否则同一页里第 2/3 个槽位会因为我们刚写的值与被复位掉的 -1 不同而被误记成
    #   `mismatch`（2026-09-22 单测抓到：scenario 1 的 filled 只有 2 而不是 3）。
    first_touch: dict[int, bool] = {}
    for row, sh in zip(rows, shifts):
        q = int(positions[row]) - int(sh)
        if q < 0:
            continue
        page = int(block_table[int(request_ids[row]), q // bs])
        # ★ 与 kernel 的读路径**同款守卫**：负页号/越界一律不动（kernel 那边对
        #   page<0 会静默 pad；对 page>=cap 会走 oob 扩容后重跑）。绝不写到
        #   pages[-1]（numpy 负索引）上去。
        if page < 0:
            continue
        if page >= cap:
            if page > oob_page:
                oob_page = page
            continue
        tok = int(prev_tok[row, sh])
        if tok < 0:
            stats["unavailable"] = stats.get("unavailable", 0) + 1
            continue
        comp = int(tm[tok])
        if tok == int(image_token_id) or tok == int(image_pad_token_id):
            comp = -1  # 与 kernel step-1 的图像 mask 逐字一致（engram_jit_kernel.py:79-81）
        slot = q % bs
        if page not in first_touch:
            fresh = int(present[page]) == 0
            first_touch[page] = fresh
            if fresh and mode >= 1:
                # 与 kernel step-1 的"首次写页"同款：整行复位**一次**再逐槽位写。
                # ★ 复位必须在首次接触时做且只做一次 —— 单测抓过一次回归：
                #   若每个槽位都复位，同一页里前一个槽位会被后一个擦掉。
                pages[page, :] = -1
                present[page] = 1
        if first_touch[page]:
            # 本页的行由我们初始化 ⇒ 这些槽位是同一次初始化的一部分，直接写
            stats["absent"] = stats.get("absent", 0) + 1
            if mode >= 1:
                pages[page, slot] = comp
                stats["filled"] = stats.get("filled", 0) + 1
        else:
            if int(pages[page, slot]) != comp:
                stats["mismatch"] = stats.get("mismatch", 0) + 1
                if mode >= 2:
                    pages[page, slot] = comp
                    stats["overwrote"] = stats.get("overwrote", 0) + 1
    if oob_page >= 0:
        stats["oob"] = stats.get("oob", 0) + 1
    return oob_page


def build_prev_tok(num_tokens, positions, request_ids, lookback, token_ids_cpu):
    """从 runner 发布的 host token 表构造 ``prev_tok``（纯 numpy，零 D2H）。

    ``token_ids_cpu``  : ``int32[max_num_reqs, max_model_len]``（runner 的
                         ``input_batch.token_ids_cpu``，**零拷贝视图**）
    ``num_tokens``     : ``int32[max_num_reqs]``（= ``input_batch.num_tokens_no_spec``，
                         "已提交（不含投机）token 数"；见
                         vllm/v1/worker/gpu_input_batch.py:387 与
                         vllm_ascend/worker/model_runner_v1.py:2915）
    ``request_ids``    : 本步每行的**请求序号**。★ 关键假设（必须 fail-closed 校验）：
                         vLLM 里 **模型侧的行序 == runner input_batch 的行序**
                         （两侧都从 `num_scheduled_tokens` 的顺序推出），因此
                         `request_ids[row]` 直接就是 `token_ids_cpu` 的 batch 行号。
                         校验办法见 engram_hash.true_tokens.diff：把
                         `num_computed` 发布出来，与模型自己算的首位置逐请求比对，
                         不一致就**拒绝修补**（退化为今天的行为）。

    返回 ``(prev_tok, stats)``；``prev_tok[row, 0]`` 恒为 -1（shift=0 由本步
    kernel 自己的写入决定，绝不覆盖）。
    """
    n = int(len(positions))
    lb = int(lookback)
    out = np.full((n, lb), -1, np.int64)
    stats = {"unavailable": 0}
    for r in range(n):
        row = int(request_ids[r])
        ntok = int(num_tokens[row]) if 0 <= row < int(num_tokens.shape[0]) else 0
        for sh in range(1, lb):
            q = int(positions[r]) - sh
            if q < 0 or q >= ntok:
                continue
            tok = int(token_ids_cpu[row, q])
            if tok < 0:  # PLACEHOLDER_TOKEN_ID（vllm.v1.sample.rejection_sampler）等
                continue
            out[r, sh] = tok
    return out, stats
