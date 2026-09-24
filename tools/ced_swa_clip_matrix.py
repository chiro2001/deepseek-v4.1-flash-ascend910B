#!/usr/bin/env python3
"""CED SWA clip 的 CPU 索引矩阵仿真（不碰 NPU，不需要 vLLM）。

与 `tools/ced_swa_clip_verify.py` 的分工：那个脚本按算子源码逐行复算**单个**
请求的 legacy/clip 访存列；本脚本补两块那个脚本没有覆盖的：

  1. **多请求同批**：复刻实现的 `base_pages[:, None] + arange(width)` +
     `gather` 行表，验证第二行及以后的行内 0 也对应它自己的页（简化版
     "取 positions[0] 套所有行" 会在这里错）。
  2. **触发边界全扫**：对 N = 100..2999 逐长度判断 legacy 是否越界，并用
     主 Agent 在真权重线扫出的规则 `N ≥ 2*block_size 且 N % block_size != 0`
     做判据，若有一处不符即失败。

请求长度矩阵还带一组 tiny 计划长度（2000/4000/8000）与真权重 1M 例，作为
后续真机探针的预测值来源。

复刻 `experimental/ced/dsa_v41.py::_native_attention` 的两种视图，按
`npu_sparse_flash_mla` 的 PA 语义（kv 位置 = seqused_ori - q_len + i，
窗口 = [pos-127, pos]，block 列 = pos // block_size）逐 query 求它实际
读到的 block 列号，并和“本请求真正持有的列集合”比较：

  legacy : 整行 block table + 完整 seq_lens  -> 首个 query 会越到保留集合之外
  clip   : rebase 到 replay 起始页 + 相对 seqused_ori -> 只落在保留集合内

判据：`legacy_out_of_hold` 必须 > 0 才说明修复有意义；`clip_out_of_hold` 必须
== 0。退出码 1 表示判据不满足。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

W = 128  # sliding_window / ori_win_left+1


def held_columns(prompt_len: int, block_size: int) -> range:
    """replay 步开始时 D 侧 SWA manager 真正持有的 block 列。

    replay 覆盖 [S, E)（E = N-1），并写回这些 token 的 KV，所以 D 必然持有
    从 `S // block_size` 到 `(E-1) // block_size` 的全部页；此前的页已被
    `remove_skipped_blocks` 换成 null（行内 0）。
    """
    end = prompt_len - 1  # E：P 交接的 prefix
    start = max(0, end - W)  # S：replay 起点
    return range(start // block_size, (end - 1) // block_size + 1)


def kernel_columns(
    prompt_len: int,
    replay_tokens: int,
    block_size: int,
    rebase: bool,
) -> tuple[list[int], int]:
    """返回 (每个 query 实际读到的 block 列, 使用的 seqused_ori)。"""
    end = prompt_len - 1
    start = max(0, end - replay_tokens)
    q_len = end - start
    base_page = start // block_size if rebase else 0
    seqused = end - base_page * block_size if rebase else end
    cols: list[int] = []
    for i in range(q_len):
        local_pos = seqused - q_len + i  # 该 query 在（可能 rebase 过的）KV 里的位置
        left = max(0, local_pos - (W - 1))
        for p in range(left, local_pos + 1):
            # rebase=True 时 p 已经是 rebase 后的相对位置，列号是行内局部列。
            cols.append(p // block_size)
    return cols, seqused


def check(prompt_len: int, replay_tokens: int, block_size: int) -> dict:
    held = set(held_columns(prompt_len, block_size))
    legacy_cols, legacy_len = kernel_columns(prompt_len, replay_tokens, block_size, False)
    clip_cols, clip_len = kernel_columns(prompt_len, replay_tokens, block_size, True)
    base_page = (max(0, prompt_len - 1 - replay_tokens)) // block_size
    clip_abs = {base_page + c for c in clip_cols}
    return {
        "prompt_len": prompt_len,
        "block_size": block_size,
        "held_columns": sorted(held),
        "legacy_read_columns": sorted(set(legacy_cols)),
        "legacy_out_of_hold": sorted(set(legacy_cols) - held),
        "legacy_seqused_ori": legacy_len,
        "clip_local_read_columns": sorted(set(clip_cols)),
        "clip_abs_read_columns": sorted(clip_abs),
        "clip_out_of_hold": sorted(clip_abs - held),
        "clip_seqused_ori": clip_len,
    }


def check_multi(requests: list[int], replay_tokens: int, block_size: int) -> dict:
    """验证修复里的逐请求 gather：多请求同一步 replay 时每行 rebase 到自己的页。

    复刻实现：
        base_pages = first_position // block_size
        local_lens = seq_lens - base_pages * block_size
        width      = ceil(max(local_lens) / block_size)
        columns    = base_pages[:, None] + arange(width)
        row_r      = block_table[r, columns[r]]
    然后按 PA 语义检查每个 query 读到的行内下标都在 [0, width) 内，且换算回绝对
    列号后都在该请求自己持有的页集合里。
    """
    bases, lens, held = [], [], []
    for n in requests:
        end = n - 1
        start = max(0, end - replay_tokens)
        bases.append(start // block_size)
        lens.append(end - bases[-1] * block_size)
        held.append(set(range(start // block_size, (end - 1) // block_size + 1)))
    width = -(-max(lens) // block_size)
    problems = []
    for r, n in enumerate(requests):
        end = n - 1
        start = max(0, end - replay_tokens)
        q_len = end - start
        reads = set()
        for i in range(q_len):
            local_pos = lens[r] - q_len + i
            left = max(0, local_pos - (W - 1))
            for p in range(left, local_pos + 1):
                col = p // block_size
                if not 0 <= col < width:
                    problems.append((r, "row_index_out_of_width", col, width))
                reads.add(bases[r] + col)
        outside = sorted(reads - held[r])
        if outside:
            problems.append((r, "reads_unheld_page", outside, sorted(held[r])))
    return {
        "requests": requests,
        "base_pages": bases,
        "local_lens": lens,
        "width": width,
        "held_pages": [sorted(h) for h in held],
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="1019847,1048576,999,1024,1300,257,8192,8193")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--replay-tokens", type=int, default=128)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    failed = False
    for raw in args.lengths.split(","):
        row = check(int(raw), args.replay_tokens, args.block_size)
        rows.append(row)
        # 判据：clip 视图永不越界；legacy 视图越界当且仅当
        # `N >= 2*block_size 且 N % block_size != 0`（与主 Agent 在真权重线上
        # 扫出的 N≥256 且不整除规则一致；短序列首个 replay query 的窗口左沿
        # 落在页 0 内，本来就安全）。
        expect_legacy = (
            row["prompt_len"] >= 2 * row["block_size"]
            and row["prompt_len"] % row["block_size"] != 0
        )
        ok = (
            not row["clip_out_of_hold"]
            and bool(row["legacy_out_of_hold"]) == expect_legacy
        )
        failed = failed or not ok
        print(
            f"N={row['prompt_len']:>8} held={row['held_columns']} "
            f"legacy_out={row['legacy_out_of_hold']} clip_out={row['clip_out_of_hold']} "
            f"clip_seqused={row['clip_seqused_ori']} expect_legacy_out={expect_legacy} "
            f"{'OK' if ok else 'FAIL'}"
        )
    multi = check_multi([4000, 1019847], args.replay_tokens, args.block_size)
    print(
        f"multi: bases={multi['base_pages']} local_lens={multi['local_lens']} "
        f"width={multi['width']} problems={multi['problems']}"
    )
    failed = failed or bool(multi["problems"])

    # 边界扫描：主 Agent 在真权重线上独立扫出「N ≥ 256 且 N % 128 != 0 才越界」。
    # 这里用同一套整数模型复算 100..2999，若有任何一处与规则不符就失败。
    sweep = {"checked": 0, "mismatches": []}
    for n in range(100, 3000):
        row = check(n, args.replay_tokens, args.block_size)
        expected = n >= 256 and n % args.block_size != 0
        sweep["checked"] += 1
        if bool(row["legacy_out_of_hold"]) != expected or row["clip_out_of_hold"]:
            sweep["mismatches"].append(
                {"N": n, "legacy_out": row["legacy_out_of_hold"], "expected_out": expected,
                 "clip_out": row["clip_out_of_hold"]}
            )
    print(
        f"boundary sweep N=100..2999: checked={sweep['checked']} "
        f"mismatches={len(sweep['mismatches'])}"
    )
    failed = failed or bool(sweep["mismatches"])
    payload = {
        "scope": "CPU integer index simulation of the SMLA PA view",
        "rows": rows,
        "multi_request": multi,
        "boundary_sweep": sweep,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print("wrote", args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
