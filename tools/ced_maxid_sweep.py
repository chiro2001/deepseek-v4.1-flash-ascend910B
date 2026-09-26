#!/usr/bin/env python3
"""在**同一个 D 实例**上按顺序发不同长度的请求，并逐条读出 D 侧 blockdump 的 max 块号。

用途：把"失败 ⟺ 块号 ≥ B"的 B 在同一实例内夹到 ±1，且把 B 与"池大小 C"分开。

关键前提（已被实测验证，见 docs/CED-PD-BLOCK-BOUND-20260925.md）：
  * 新鲜池上块是**单调向前**发放的：连续同批请求的第 k 条占
    `[1 + (k-1)(c+21), k(c+21)]`，其中 `c = ceil((N-1)/128)`。
    所以第 k 条的 g0 max = 前面所有请求的 (c+21) 之和 + c。
  * 每条请求的块列表由 D 的 `V41_CED_BLOCK_DUMP_DIR` 落一个文件；
    本脚本在发请求前记录已有文件集合，发完后等**新文件**出现，避免读旧 dump。

用法：
  python3 tools/ced_maxid_sweep.py --corpus data/hongloumeng.txt \
    --tokenize http://127.0.0.1:18990/tokenize --url http://127.0.0.1:18992/v1/chat/completions \
    --model deepseek-v41-ced-pd --dump-dir /path/blockdump \
    --targets 1019847,1019847,1019847,653953,655617 --out /path/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ced_length_sweep import EXPECTED, build_prompt, post  # noqa: E402


def existing_dumps(dump_dir: str) -> set[str]:
    try:
        return set(os.listdir(dump_dir))
    except OSError:
        return set()


def wait_new_dump(dump_dir: str, before: set[str], timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        names = existing_dumps(dump_dir) - before
        if names:
            name = sorted(names)[0]
            path = os.path.join(dump_dir, name)
            ids = [int(x) for x in open(path, encoding="utf-8", errors="replace").read().split()]
            if ids:
                return name, ids
        time.sleep(0.5)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenize", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--targets", required=True, help="逗号分隔的 token 目标（按顺序发送）")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--dump-timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    cum = 0

    for index, target in enumerate([int(x) for x in args.targets.split(",") if x.strip()], start=1):
        payload, got = build_prompt(corpus, args.tokenize, args.model, target, 300.0)
        c = -(-(got - 1) // 128)
        before = existing_dumps(args.dump_dir)
        started = time.strftime("%H:%M:%S")
        t0 = time.time()
        status, data, body = post(args.url, payload, args.timeout)
        wall = time.time() - t0
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        usage = data.get("usage") or {}
        passed = isinstance(content, str) and EXPECTED in content
        name, ids = wait_new_dump(args.dump_dir, before, args.dump_timeout)
        max_id = max(ids) if ids else None
        min_id = min(ids) if ids else None
        descents = sum(1 for a, b in zip(ids, ids[1:]) if b < a) if ids else None
        cum += c + 21
        row = {
            "idx": index, "target": target, "prompt_tokens": usage.get("prompt_tokens", got),
            "c": c, "predicted_max": cum - 21, "max_id": max_id, "min_id": min_id,
            "n": len(ids) if ids else 0, "descents": descents, "dump": name,
            "status": status, "completion": usage.get("completion_tokens"),
            "content": content, "passed": passed, "wall_s": round(wall, 1), "start": started,
        }
        rows.append(row)
        print(
            f"[{index}] target={target} prompt={row['prompt_tokens']} c={c} "
            f"max_id={max_id} predicted={row['predicted_max']} n={row['n']} "
            f"{'PASS' if passed else 'FAIL'} completion={row['completion']} "
            f"content={str(content)[:24]!r} wall={row['wall_s']}s",
            flush=True,
        )
        open(os.path.join(args.out, f"req{index:02d}.request.json"), "wb").write(body)
        json.dump(rows, open(os.path.join(args.out, "sweep.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(f"{'idx':>4} {'prompt':>9} {'c':>6} {'max_id':>8} {'pred':>8} {'verdict':>8}")
    for row in rows:
        print(f"{row['idx']:>4} {row['prompt_tokens']:>9} {row['c']:>6} {str(row['max_id']):>8} "
              f"{row['predicted_max']:>8} {'PASS' if row['passed'] else 'FAIL':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
