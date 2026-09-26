#!/usr/bin/env python3
"""Exercise the internal CED P role and verify its Mooncake handoff metadata.

The returned text is a transfer marker, never a model answer. This probe does
not contact a decoder or claim end-to-end PD correctness.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def token_count(base_url: str, model: str, prompt: str) -> int:
    data = post(base_url, "/tokenize", {"model": model, "prompt": prompt}, 180)
    return int(data.get("count") or len(data.get("tokens") or []))


def make_prompt(base_url: str, model: str, corpus: str, target: int) -> tuple[str, int]:
    if not corpus:
        raise ValueError("corpus is empty")
    chars = target
    prompt = ""
    count = 0
    for _ in range(3):
        prompt = (corpus * (chars // len(corpus) + 1))[:chars]
        count = token_count(base_url, model, prompt)
        if count <= 0:
            raise RuntimeError("/tokenize returned no tokens")
        if abs(count - target) <= target // 100:
            break
        chars = max(1, round(chars * target / count))
    return prompt, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v41-ced-prefill-only")
    parser.add_argument("--context-tokens", type=int, default=144000)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "data/hongloumeng.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--expect-masked-swa", action="store_true",
                        help="require CED replay metadata and empty upper-layer SWA groups")
    args = parser.parse_args()
    if args.context_tokens < 128:
        parser.error("--context-tokens must be at least 128")
    corpus = args.corpus.read_text(encoding="utf-8")
    prompt, count = make_prompt(args.base_url, args.model, corpus, args.context_tokens)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt + "\n\n请准备 KV 缓存交接。"}],
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "stream": False,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    }
    started = time.monotonic()
    answer = post(args.base_url, "/v1/chat/completions", payload, args.timeout)
    elapsed = time.monotonic() - started
    choices = answer.get("choices") or []
    choice = choices[0] if choices else {}
    transfer = answer.get("kv_transfer_params") or {}
    groups = transfer.get("remote_block_ids") or []
    result = {
        "context_tokens": count,
        "context_target": args.context_tokens,
        "prompt_tokens_reported": (answer.get("usage") or {}).get("prompt_tokens"),
        "wall_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "marker_text": (choice.get("message") or {}).get("content"),
        "marker_token_id": transfer.get("last_token_id"),
        "do_remote_prefill": transfer.get("do_remote_prefill"),
        "kv_groups": len(groups),
        "kv_group_blocks": [len(group) for group in groups],
        "num_prompt_blocks": transfer.get("num_prompt_blocks"),
        "ced_replay_tokens": transfer.get("ced_replay_tokens"),
        "ced_missing_swa_groups": transfer.get("ced_missing_swa_groups"),
        "ced_prefix_tokens": transfer.get("ced_prefix_tokens"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    masked_ok = (
        result["ced_replay_tokens"] == 128
        and tuple(result["ced_missing_swa_groups"] or ()) == (7, 8, 9, 10, 11)
        and len(result["kv_group_blocks"]) >= 12
        and all(result["kv_group_blocks"][idx] == 0 for idx in range(7, 12))
    )
    return 0 if (
        count >= args.context_tokens * 0.95
        and result["finish_reason"] == "length"
        and result["marker_token_id"] == 42
        and result["do_remote_prefill"] is True
        and result["kv_groups"] > 0
        and (not args.expect_masked_swa or masked_ok)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
