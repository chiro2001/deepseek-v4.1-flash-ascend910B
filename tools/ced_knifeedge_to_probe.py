#!/usr/bin/env python3
"""把 ced_knifeedge_walk.py 的 walk.jsonl 转成 ced_pair_dump_verdict.py 能吃的探针 JSON。

为什么需要：`/v1/completions` 的响应 id 是 `cmpl-<uuid>-0-<hash>`（不是 chatcmpl-*），
而 dump 文件名正是用这个完整 id；`ced_pair_dump_verdict.py` 的 rid 推导对两者都成立，
只要探针 JSON 里带 `id` 与 `content`/`token_ids` 就能配对。

输出是**单文件 JSON 数组**（不是 JSONL），放在 <out>，可被 --probe-dir 递归扫到。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", action="append", required=True, help="walk.jsonl（可重复）")
    parser.add_argument("--out", required=True, help="目标 .json 文件")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    records = []
    for path in args.walk:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                response = row.get("response") or {}
                request_id = response.get("id")
                if not request_id:
                    continue
                dump = row.get("dump") or {}
                first = response.get("first_token") or {}
                records.append(
                    {
                        "id": request_id,
                        "content": response.get("text"),
                        "token_ids": [first.get("token")] if first.get("token") else None,
                        "label": row.get("label"),
                        "n_prompt": row.get("n_prompt"),
                        "http_status": row.get("http_status"),
                        "dump_max": dump.get("max"),
                        "dump_count": dump.get("count"),
                        "first_logprob": first.get("logprob"),
                        "tag": args.tag,
                    }
                )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[to-probe] 写出 {len(records)} 条 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
