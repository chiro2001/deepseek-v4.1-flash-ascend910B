#!/usr/bin/env python3
"""★★ 语义测试：镜像缺页/陈旧页 × 真 token 回填（**不需要 NPU、不需要 numba**）。

跑的是**生产里那个 numba kernel 的原文件**
（`dsv41-release/patches/files/engram_jit_kernel.py`，只读 + `NUMBA_DISABLE_JIT=1`）
加上本交付的草案 helper（`patches/engram_hash.repair_helpers.py`）。

四个 arm（★ 全部与"真历史"对照，而不是与"今天的行为"对照）：

  ① **现状 = 降级修复**（`21e7d99` "ENGRAM × 卸载 的 P0 修复"：缺页从 KeyError 降级为 barrier）
     * 场景 1（p = 页首）：历史 = `[cur, pad, pad, pad]` + `miss_rows=1`（可观测的降级）
     * 场景 2（p = 页中）：历史 = `[cur, pad, pad, pad]` 但 **`miss_rows=0` / `err=-1`**
       ⇒ ★★ **没有任何计数器会响** —— 这是"pad 修复"漏掉的那一格
  ② **精确修复**（本交付，mode>=1）：回填真值 ⇒ 历史 = `[cur, 真实 p-1, p-2, p-3]`
  ③ 陈旧页对照（present=1 但内容是上一任占用者的 token）：今天**静默算错**且无计数；
     mode=1 把 `mismatch` 计出来（只测不改）；mode=2 覆写后逐值等于真历史
  ④ 守门：负页号 / 越界页号**绝不写**（numpy 负索引会写到 `pages[-1]`）

★ 兼容老 kernel：若返回 3 元组（`21e7d99` 之前）则 `err>=0` 表示**调用方会 raise KeyError**
（= 崩溃），本测试把这种情况如实标成 `crash`。

用法：`python3 tests/test_true_tokens_repair.py`
"""
from __future__ import annotations

import numpy as np

from _loader import release_kernel, repair_helpers

PAD_ID = 1
IMAGE_TOKEN_ID = -1
IMAGE_PAD_TOKEN_ID = -2
LOOKBACK = 4
N_HEADS = 8
SMALL_MAX = 16  # n < small_max ⇒ kernel 走"小批/decode"分支
BLOCK_SIZE = 4


def _mirror(cap_pages: int):
    return np.full((cap_pages, BLOCK_SIZE), -1, np.int64), np.zeros(cap_pages, np.uint8)


def _run_kernel(kernel, *, positions, block_table, pages, present, input_ids, request_ids):
    """跑生产 kernel，返回 dict。★ 对 3 元组/4 元组两种版本都兼容。"""
    n = len(positions)
    n_layers = 2
    cap_tokens = max(n, 1) + 4
    args = (
        np.arange(64, dtype=np.int64),
        np.array(input_ids, np.int64),
        np.array(positions, np.int64),
        np.array(request_ids, np.int64),
        np.asarray(block_table, np.int64),
        BLOCK_SIZE,
        n,
        PAD_ID,
        IMAGE_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        pages,
        present,
        np.ones((n_layers, LOOKBACK), np.int64),
        np.full((n_layers, LOOKBACK - 1, N_HEADS), 7, np.int64),
        np.zeros((n_layers, (LOOKBACK - 1) * N_HEADS), np.int64),
        LOOKBACK,
        N_HEADS,
        SMALL_MAX,
        np.zeros((cap_tokens, n_layers, (LOOKBACK - 1) * N_HEADS), np.int64),
        np.zeros(cap_tokens, np.bool_),
        np.zeros((cap_tokens, LOOKBACK), np.int64),
        np.zeros(cap_tokens * LOOKBACK, np.int64),
        np.zeros(cap_tokens * LOOKBACK, np.int64),
        np.zeros(cap_tokens * LOOKBACK, np.uint8),
        np.zeros(cap_tokens * LOOKBACK, np.int64),
        np.zeros(cap_tokens, np.int64),
        np.zeros(cap_tokens, np.int64),
        np.zeros(cap_tokens, np.int64),
        np.zeros(cap_tokens, np.uint8),
        np.zeros(cap_tokens, np.uint8),
    )
    hist = args[20]  # 位置参数第 21 项 = hist（0-based 20；原先误取 args[21]=flat）
    try:
        ret = kernel.engram_update_kernel(*args)
    except Exception as exc:  # 老 kernel 的严格臂（不应发生，但如实记录）
        return {"raised": type(exc).__name__, "hist": hist[:n].tolist()}
    return {
        "raised": None,
        "err": int(ret[0]),
        "oob": int(ret[1]),
        "fell_back": int(ret[2]),
        "miss_rows": int(ret[3]) if len(ret) > 3 else None,
        "crash": (len(ret) == 3 and int(ret[0]) >= 0),  # 老 hash 会 raise KeyError(err)
        "hist": hist[:n].tolist(),
    }


def _token_ids_cpu(true_tokens, max_len=16):
    """把真 token 序列铺进 runner 的 host token 表（batch 行 0）。"""
    t = np.zeros((2, max_len), np.int32)
    t[0, : len(true_tokens)] = true_tokens
    return t


# 真 token 序列：位置 0..7
TRUE = [2, 8, 9, 11, 5, 12, 13, 14]
NUM_TOKENS = np.array([8, 0], np.int32)
TOK_CPU = _token_ids_cpu(TRUE)


def main() -> int:
    kernel = release_kernel()
    helpers = repair_helpers()
    fails: list[str] = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'✓' if ok else '✗'}] {name}: got={got} want={want}")
        if not ok:
            fails.append(name)

    def exact(prev_tok, positions, block_table, mode, mirror=None):
        """精确修复臂：helper 回填 → 跑 kernel。mirror 可传入预置（脏）镜像。"""
        pages, present = _mirror(8) if mirror is None else (mirror[0].copy(), mirror[1].copy())
        stats: dict = {}
        oob = helpers.apply_repairs(
            pages,
            present,
            np.arange(64, dtype=np.int64),
            np.asarray(block_table, np.int64),
            np.array([0], np.int64),
            np.array(positions, np.int64),
            prev_tok,
            BLOCK_SIZE,
            LOOKBACK,
            IMAGE_TOKEN_ID,
            IMAGE_PAD_TOKEN_ID,
            mode,
            stats,
        )
        res = _run_kernel(
            kernel,
            positions=positions,
            block_table=block_table,
            pages=pages,
            present=present,
            input_ids=[TRUE[positions[0]]],
            request_ids=[0],
        )
        res["oob_repair"] = oob
        res["stats"] = stats
        return res

    # ------------------------------------------------------------------ 场景 1
    # 页首命中边界：p = 4（页 1 的 slot 0）⇒ 窗口跨到上一页（页 2，未写）
    print("场景 1：p=4（页首）—— 窗口跨到上一页（= logs/073 的 KeyError 形态）")
    positions = [4]
    block_table = [[2, 3]]
    pages0, present0 = _mirror(8)
    today = _run_kernel(
        kernel,
        positions=positions,
        block_table=block_table,
        pages=pages0,
        present=present0,
        input_ids=[TRUE[4]],
        request_ids=[0],
    )
    prev_tok, _ = helpers.build_prev_tok(
        positions=np.array(positions, np.int64),
        request_ids=np.array([0], np.int64),
        num_tokens=NUM_TOKENS,
        token_ids_cpu=TOK_CPU,
        lookback=LOOKBACK,
    )
    check("prev_tok（真 id；shift0 恒 -1，绝不覆盖本步自己的写入）",
          prev_tok.tolist(), [[-1, 11, 9, 8]])
    check("plan（每请求 ≤ lookback-1 = 3 个跨界槽位）",
          helpers.plan_repair_slots(np.array(positions), np.array([0]), LOOKBACK),
          ([0, 0, 0], [1, 2, 3]))
    soft = today["miss_rows"] is not None
    if soft:
        check("① 现状 hist（缺页 ⇒ barrier/pad 顶替）",
              today["hist"], [[TRUE[4], PAD_ID, PAD_ID, PAD_ID]])
        check("① 现状 miss_rows（可观测）", today["miss_rows"], 1)
        check("① 现状 err（软信息，不再致命）", today["err"], 2)
    else:
        check("① 现状 = 老 kernel ⇒ err_page ⇒ KeyError（引擎死）", today["crash"], True)

    x1 = exact(prev_tok, positions, block_table, mode=1)
    check("② 精确版 oob / 回填", (x1["oob_repair"], x1["stats"].get("filled")), (-1, 3))
    check("② 精确版 hist == 真历史", x1["hist"], [[TRUE[4], TRUE[3], TRUE[2], TRUE[1]]])
    check("② 精确版 err（不降级）", x1["err"], -1)
    if soft:
        check("② 精确版 miss_rows == 0（不再降级）", x1["miss_rows"], 0)

    # ------------------------------------------------------------------ 场景 2
    # ★★ 页中：p = 6（页 1 的 slot 2）
    print("场景 2：p=6（页中）—— ★★ 今天**静默**丢 3 个历史槽位、且计数器不响")
    positions2 = [6]
    pages1, present1 = _mirror(8)
    today2 = _run_kernel(
        kernel,
        positions=positions2,
        block_table=block_table,
        pages=pages1,
        present=present1,
        input_ids=[TRUE[6]],
        request_ids=[0],
    )
    check("① 现状 hist（3 个槽位静默变 pad）",
          today2["hist"], [[TRUE[6], PAD_ID, PAD_ID, PAD_ID]])
    if soft:
        check("★★ 现状 miss_rows == 0（★ 降级修复的计数器漏掉这一格）",
              today2["miss_rows"], 0)
        check("★★ 现状 err == -1（也没有任何上报）", today2["err"], -1)
    prev_tok2, _ = helpers.build_prev_tok(
        positions=np.array(positions2, np.int64),
        request_ids=np.array([0], np.int64),
        num_tokens=NUM_TOKENS,
        token_ids_cpu=TOK_CPU,
        lookback=LOOKBACK,
    )
    x2 = exact(prev_tok2, positions2, block_table, mode=1)
    check("② 精确版 hist == 真历史（含**同一页**里的 2 个槽位）",
          x2["hist"], [[TRUE[6], TRUE[5], TRUE[4], TRUE[3]]])
    check("② 精确版回填计数 filled == 3", x2["stats"].get("filled"), 3)
    if soft:
        check("② 精确版 miss_rows == 0 且 err == -1（无降级）",
              (x2["miss_rows"], x2["err"]), (0, -1))

    # ------------------------------------------------------------------ 场景 3
    # 陈旧页：present=1，但那一行是**上一任占用者**的 token
    print("场景 3：陈旧页（present=1，内容是别人的 token）—— 今天静默算错且无计数")
    pages_s, present_s = _mirror(8)
    present_s[2] = 1
    pages_s[2, :] = [0, 20, 21, 22]
    today3 = _run_kernel(
        kernel,
        positions=positions,
        block_table=block_table,
        pages=pages_s,
        present=present_s,
        input_ids=[TRUE[4]],
        request_ids=[0],
    )
    check("① 现状 hist（★ 静默算错：22/21/20 是别人的 token）",
          today3["hist"], [[TRUE[4], 22, 21, 20]])
    if soft:
        check("★★ 现状 miss_rows == 0（★ 静默错误没有任何计数器）", today3["miss_rows"], 0)

    m1 = exact(prev_tok, positions, block_table, mode=1, mirror=(pages_s, present_s))
    check("③ mode=1 只测不改：hist 仍为错的", m1["hist"], [[TRUE[4], 22, 21, 20]])
    check("③ mode=1 把静默错误计出来 mismatch == 3", m1["stats"].get("mismatch"), 3)
    m2 = exact(prev_tok, positions, block_table, mode=2, mirror=(pages_s, present_s))
    check("③ mode=2 覆写后 == 真历史", m2["hist"], [[TRUE[4], TRUE[3], TRUE[2], TRUE[1]]])
    check("③ mode=2 覆写计数 overwrote == 3", m2["stats"].get("overwrote"), 3)

    # ------------------------------------------------------------------ 场景 4
    print("场景 4：越界/负页号守门（numpy 负索引会写到 pages[-1]）")
    for name, bt, want_oob in (("负页号", [[-1, 3]], -1), ("越界页号", [[99, 3]], 99)):
        pages_g, present_g = _mirror(8)
        stats_g: dict = {}
        oob_g = helpers.apply_repairs(
            pages_g,
            present_g,
            np.arange(64, dtype=np.int64),
            np.asarray(bt, np.int64),
            np.array([0], np.int64),
            np.array(positions, np.int64),
            prev_tok,
            BLOCK_SIZE,
            LOOKBACK,
            IMAGE_TOKEN_ID,
            IMAGE_PAD_TOKEN_ID,
            2,
            stats_g,
        )
        check(f"{name}：报 oob（由调用方扩容）且未误写",
              (oob_g, int(present_g.sum())), (want_oob, 0))

    print()
    if fails:
        print(f"✗ {len(fails)} 项不符：{fails}")
        return 1
    print("✓ 全部通过（现状=降级 / 精确修复 / 陈旧页 / 守门 四个 arm 都按要求复现）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
