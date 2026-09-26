#!/usr/bin/env python3
"""把一条腿（arm）的 walk.jsonl 汇总成判据表。

输入：walk.jsonl（ced_knifeedge_walk.py 逐请求落盘）
输出：
  * 每个请求一行：label / n / max_id / 首 token / logprob / top-5 摘要 / 判定
  * 同 n 的分组一致性：同 n 的请求输出是否逐字段一致（单变量判据）
  * 池顶对照：max_id == C-1 的请求 vs 其他请求的输出是否发生可复现变化
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def first_token(row: dict) -> dict:
    return (row.get("response") or {}).get("first_token") or {}


def fingerprint(row: dict) -> tuple:
    ft = first_token(row)
    top5 = tuple((entry.get("token"), entry.get("logprob")) for entry in ft.get("top5") or [])
    return (ft.get("token"), ft.get("logprob"), top5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", required=True)
    parser.add_argument("--pool-blocks", type=int, required=True)
    args = parser.parse_args()

    rows = []
    with open(args.walk, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    pool_max = args.pool_blocks - 1
    print(f"# pool_blocks={args.pool_blocks} 池内最大块号={pool_max} 请求数={len(rows)}")
    print(
        f"{'#':>3s} {'label':22s} {'n':>6s} {'count':>5s} {'min':>7s} {'max':>7s} "
        f"{'tok':>14s} {'logprob':>11s} {'http':>4s} {'wall':>6s}"
    )
    prev_max = None
    for row in rows:
        dump = row.get("dump") or {}
        ft = first_token(row)
        token = str(ft.get("token"))
        print(
            f"{row['index']:3d} {row['label']:22s} {row['n_prompt']:6d} "
            f"{dump.get('count', 0):5d} {dump.get('min', 0):7d} {dump.get('max', 0):7d} "
            f"{token[:14]:>14s} {str(ft.get('logprob')):>11s} "
            f"{row['http_status']:>4d} {row['wall_s']:6.1f}"
        )
        prev_max = dump.get("max")

    print()
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("dump"):
            groups[row["n_prompt"]].append(row)
    print("# 同 n 分组一致性（单变量判据：同 prompt 长度 ⇒ 输出应逐字段一致）")
    for n, items in sorted(groups.items()):
        prints = {fingerprint(r) for r in items}
        maxes = [r["dump"]["max"] for r in items]
        flag = "一致" if len(prints) == 1 else f"不一致（{len(prints)} 种）"
        print(
            f"  n={n:6d} 请求数={len(items)} max_id=[{min(maxes)}..{max(maxes)}] "
            f"输出指纹={flag}"
        )
        if len(prints) > 1:
            for row in items:
                ft = first_token(row)
                print(
                    f"      max={row['dump']['max']:7d} label={row['label']:22s} "
                    f"tok={str(ft.get('token'))[:16]!r} lp={ft.get('logprob')}"
                )

    print()
    print("# 池顶判定")
    top_rows = [r for r in rows if (r.get("dump") or {}).get("max") == pool_max]
    below_rows = [
        r
        for r in rows
        if r.get("dump") and (r.get("dump") or {}).get("max") < pool_max
    ]
    if not top_rows:
        print(f"  ⚠ 没有任何请求的 max_id 触及池顶 {pool_max} ⇒ 本臂未命中边界，不能判否证")
        return 0
    baseline_by_n: dict[int, set] = defaultdict(set)
    for row in below_rows:
        baseline_by_n[row["n_prompt"]].add(fingerprint(row))
    for row in top_rows:
        ft = first_token(row)
        same_n = baseline_by_n.get(row["n_prompt"], set())
        verdict = (
            "与低块号同 n 输出一致 ⇒ 池顶未造成变化"
            if fingerprint(row) in same_n
            else "与低块号同 n 输出不同 ⇒ 疑似阈值效应"
        )
        print(
            f"  max={row['dump']['max']} n={row['n_prompt']} label={row['label']} "
            f"tok={str(ft.get('token'))[:16]!r} lp={ft.get('logprob')} → {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
