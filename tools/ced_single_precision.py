#!/usr/bin/env python3
"""Probe one-chip DeepSeek V4.1 tiny logits at CED replay boundaries.

Run the same token pattern against CED PD and a full-40-layer baseline. The
dummy model's near-tied logits make text equality alone a weak numeric test;
the response includes top-logprob values and a same-service repeat.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from itertools import cycle, islice
from pathlib import Path


TOKEN_PATTERN = (28669, 6441, 58603, 693, 85450, 84483, 22089, 320)


def request(base_url: str, payload: dict, timeout: float) -> tuple[int, dict | str, float]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")[:1000], time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v41-ced-tiny")
    parser.add_argument("--lengths", default="2,127,128,129,130,256,512,4096")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) < 1 or args.repeats < 1:
        parser.error("lengths and repeats must be positive")

    rows = []
    failed = False
    for length in lengths:
        tokens = list(islice(cycle(TOKEN_PATTERN), length))
        for repeat in range(args.repeats):
            payload = {
                "model": args.model,
                "prompt": tokens,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 20,
                "seed": 0,
            }
            status, response, elapsed = request(args.base_url, payload, args.timeout)
            choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
            logprobs = choice.get("logprobs") or {}
            row = {
                "length": length,
                "repeat": repeat,
                "http": status,
                "wall_s": round(elapsed, 6),
                "prompt_tokens_reported": (response.get("usage") or {}).get("prompt_tokens")
                if isinstance(response, dict) else None,
                "text": choice.get("text"),
                "token_logprob": (logprobs.get("token_logprobs") or [None])[0],
                "top_logprobs": (logprobs.get("top_logprobs") or [None])[0],
                "finish_reason": choice.get("finish_reason"),
                "error": response.get("error") if isinstance(response, dict) else response,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("length", "repeat", "http", "text", "token_logprob")}, ensure_ascii=False), flush=True)
            if status != 200 or row["prompt_tokens_reported"] != length or row["top_logprobs"] is None:
                failed = True
                break
        if failed:
            break

    result = {"base_url": args.base_url, "model": args.model, "token_pattern": TOKEN_PATTERN, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
