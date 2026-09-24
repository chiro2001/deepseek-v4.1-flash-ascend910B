#!/usr/bin/env python3
"""分析 `V41_CED_BLOCK_DUMP_DIR` 落盘的**完整**块列表，判读池阈值假说。

为什么不能用摘要：`[CED-BLOCKS]` 的 `first/last/descents/head` 对"摘要相同但顺序
不同"的两个列表不可区分（2026-09-24 实测：#4 与 #10 摘要完全相同、结果相反）。
本工具直接读完整列表，给出每个请求的：

  * 分配到的块 id 集合的**最大/最小**值、被**重复分配**的块数（同一 id 出现两次
    说明池在请求内部就绕了一圈）、以及**是否触及 ≥ N_total 的越界 id**；
  * 跨请求的**复用**关系：某个请求拿到的块，有多少在之前的请求里出现过
    （区分"连续段"与"绕回段"要靠这个，而不是靠 descents）；
  * 与结果的配对表。

用法：
    python3 tools/ced_block_dump_analysis.py \
        --dump-dir <blockdump 目录> \
        --results  <ced_seq_probe 的 outdir> \
        [--total-blocks N] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def load_dumps(dump_dir: str):
    """返回 [(req 短 id, [block ids...])]，按 g0 文件名里的请求 id 排序。"""
    out = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "*_g0.txt"))):
        name = os.path.basename(path)[:-len("_g0.txt")]
        parts = name.split("_")
        rid = parts[-1] if parts else name
        ids = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ids.append(int(line))
        out.append((rid, ids, path))
    return out


def load_results(results_dir: str):
    """返回 {req 短 id 前缀: {passed, content, completion}}。"""
    table = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.summary.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rid = (data.get("id") or "")[len("chatcmpl-"):][:8]
        content = data.get("content")
        usage = data.get("usage") or {}
        table[rid or os.path.basename(path).split("_")[0]] = {
            "passed": content is not None and "RB9N-6014" in content,
            "content": content,
            "completion": usage.get("completion_tokens"),
        }
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--results", default="", help="ced_seq_probe 的 outdir")
    ap.add_argument("--total-blocks", type=int, default=0,
                    help="D 的 num_blocks；给了才能判越界 id")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dumps = load_dumps(args.dump_dir)
    if not dumps:
        print(f"[blockdump] 在 {args.dump_dir} 找不到 *_g0.txt", file=sys.stderr)
        return 2
    results = load_results(args.results) if args.results else {}

    print(f"请求数={len(dumps)}  total_blocks={args.total_blocks or '未知'}")
    print(f"{'#':>3} {'req':>9} {'n':>6} {'min':>6} {'max':>6} {'dups':>5} "
          f"{'>=total':>7} {'reused':>7} {'result':>7}")
    seen: set[int] = set()
    rows = []
    for index, (rid, ids, _path) in enumerate(dumps, 1):
        n = len(ids)
        uniq = set(ids)
        dups = n - len(uniq)
        hi = max(ids) if ids else -1
        lo = min(ids) if ids else -1
        over = sum(1 for v in ids if v >= args.total_blocks) if args.total_blocks else 0
        reused = len(uniq & seen)
        seen |= uniq
        res = results.get(rid[:8]) or {}
        verdict = "PASS" if res.get("passed") else ("FAIL" if res else "?")
        print(f"{index:>3} {rid[:8]:>9} {n:>6} {lo:>6} {hi:>6} {dups:>5} "
              f"{over:>7} {reused:>7} {verdict:>7}")
        rows.append({"index": index, "req": rid[:8], "n": n, "min": lo, "max": hi,
                     "dups": dups, "over_total": over, "reused": reused,
                     "verdict": verdict})

    print()
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    passes = [r for r in rows if r["verdict"] == "PASS"]
    print(f"通过 {len(passes)} / 失败 {len(fails)}")
    if args.total_blocks:
        print(f"触及 ≥ total_blocks 的请求："
              f"{[r['index'] for r in rows if r['over_total']] or '无'}")
    if fails and passes:
        for key in ("dups", "over_total", "max"):
            fv = [r[key] for r in fails]
            pv = [r[key] for r in passes]
            print(f"  {key}: 失败集 {min(fv)}–{max(fv)}  通过集 {min(pv)}–{max(pv)}")
    print("\n提示：判断是否越界要看 `>=total` 列与 `max` 列，不要用 descents ——")
    print("      摘要在顺序不同的列表上不可区分（同一摘要出过相反结果）。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print(f"已写 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
