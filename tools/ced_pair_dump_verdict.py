#!/usr/bin/env python3
"""把 [CED-BLOCK-DUMP] 的 g0 块列表与每个请求的结果配对，输出 (max_id, verdict) 表。

用途：验证"失败 ⟺ g0 块号触及某个固定上界 B"这条规律，并把 B 夹到尽可能窄。

输入：
  1) dump 目录（V41_CED_BLOCK_DUMP_DIR），文件名 decode_<rid>_g0.txt，一行一个物理块号；
  2) 探针目录（--probe-dir，可重复），递归扫描其中的 *.json，
     找形如 {"id": "chatcmpl-<rid>-...", "content": ..., "token_ids": [...]} 的记录。

输出：TSV（默认按 max_id 升序）：
  max_id  n  first  last  verdict  completion_len  rid  source

verdict：PASS = 命中任一 --expect 子串；FAIL-NULL = 空回复；FAIL-OTHER = 其它内容。
本脚本只读，不发请求。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def rid_of(chatcmpl_id: str) -> str:
    if chatcmpl_id.startswith("chatcmpl-"):
        parts = chatcmpl_id.split("-")
        if len(parts) > 1:
            return parts[1]
    return chatcmpl_id


def scan_probe_dir(path: str, out: dict[str, dict]) -> None:
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not name.endswith(".json"):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    data = json.load(handle)
            except Exception:  # noqa: BLE001
                continue
            records = []
            if isinstance(data, dict):
                records = [data]
                for key in ("results", "samples", "runs"):
                    value = data.get(key)
                    if isinstance(value, list):
                        records.extend(r for r in value if isinstance(r, dict))
            elif isinstance(data, list):
                records = [r for r in data if isinstance(r, dict)]
            for rec in records:
                cid = rec.get("id") or rec.get("request_id")
                # `chatcmpl-` 是 /v1/chat/completions 的 id；/v1/completions 用
                # `cmpl-<uuid>-<n>-<hash>`，而 dump 文件名正是用这个完整 id，
                # 所以两种前缀都要接收（rid_of 对非 chatcmpl 前缀原样返回）。
                if not isinstance(cid, str) or not cid.startswith(("chatcmpl-", "cmpl-")):
                    continue
                rid = rid_of(cid)
                content = rec.get("content")
                if content is None and isinstance(rec.get("choices"), list) and rec["choices"]:
                    choice = rec["choices"][0]
                    content = (choice.get("message") or {}).get("content")
                tokens = rec.get("token_ids")
                if tokens is None and isinstance(rec.get("choices"), list) and rec["choices"]:
                    tokens = rec["choices"][0].get("token_ids")
                out.setdefault(
                    rid,
                    {
                        "content": content,
                        "token_ids": tokens,
                        "source": os.path.relpath(full, start=os.path.dirname(path.rstrip("/")) or "."),
                    },
                )


def classify(content, token_ids, expects: list[str]) -> str:
    if isinstance(content, str):
        for want in expects:
            if want and want in content:
                return "PASS"
        if content.strip() == "":
            return "FAIL-NULL"
        return "FAIL-OTHER"
    if content is None:
        if isinstance(token_ids, list) and len(token_ids) == 1:
            return f"FAIL-NULL(tok={token_ids[0]})"
        return "FAIL-NULL"
    return "FAIL-OTHER"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--probe-dir", action="append", default=[], required=True)
    ap.add_argument("--expect", action="append", default=["RB9N-6014", "ZQ7K-3341"])
    ap.add_argument("--sort", default="max", choices=["max", "time", "verdict"])
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for path in args.probe_dir:
        scan_probe_dir(path, results)

    rows = []
    missing = 0
    for name in sorted(os.listdir(args.dump_dir)):
        if not (name.startswith("decode_") and name.endswith("_g0.txt")):
            continue
        rid = name[len("decode_") : -len("_g0.txt")]
        ids = []
        with open(os.path.join(args.dump_dir, name), encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ids.append(int(line))
        if not ids:
            continue
        rec = results.get(rid)
        if rec is None:
            missing += 1
            continue
        rows.append(
            {
                "max_id": max(ids),
                "n": len(ids),
                "first": ids[0],
                "last": ids[-1],
                "verdict": classify(rec["content"], rec["token_ids"], args.expect),
                "rid": rid,
                "source": rec["source"],
                "mtime": os.path.getmtime(os.path.join(args.dump_dir, name)),
            }
        )

    if args.sort == "max":
        rows.sort(key=lambda r: (r["max_id"], r["mtime"]))
    elif args.sort == "time":
        rows.sort(key=lambda r: r["mtime"])
    else:
        rows.sort(key=lambda r: (r["verdict"], r["max_id"]))

    print(f"# paired={len(rows)} unmatched_dumps={missing} probe_records={len(results)}")
    print("max_id\tn\tfirst\tlast\tverdict\trid")
    for row in rows:
        print(f"{row['max_id']}\t{row['n']}\t{row['first']}\t{row['last']}\t{row['verdict']}\t{row['rid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
