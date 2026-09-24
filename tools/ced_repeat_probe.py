#!/usr/bin/env python3
"""CED 1+1 重复探针：同一请求重复 N 次，每轮立刻落盘；代理卡住就重启代理。

与 `tools/ced_single_precision.py` 的区别：
  * 每轮单独写一行 JSONL（不会被后面某个超时整轮吞掉）；
  * 传输层超时/连接错误时按 `--restart-container` 重启本地代理容器后重试该轮；
  * 可选先发 K 个内容完全不同的 filler（用于把 D 的块池推过池尾、制造碎片）；
  * 输出里带 top-k logprob，便于逐轮比对「首 token 是否变」。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from itertools import cycle, islice
from pathlib import Path

TOKEN_PATTERN = (28669, 6441, 58603, 693, 85450, 84483, 22089, 320)


def post(url: str, body: bytes, timeout: float):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.time() - started
    except Exception as error:  # noqa: BLE001
        return -1, json.dumps({"transport_error": repr(error)}).encode(), time.time() - started


def summarize(raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        return {"parse_error": str(error), "raw_head": raw[:200].decode("utf-8", "replace")}
    choices = payload.get("choices") or []
    if not choices:
        return {"choices": []}
    choice = choices[0]
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    toks = (choice.get("logprobs") or {}).get("tokens") or []
    top = (choice.get("logprobs") or {}).get("top_logprobs") or []
    first = None
    if logprobs:
        first = {
            "token": toks[0] if toks else None,
            "logprob": round(logprobs[0], 6),
            "top": {k: round(v, 6) for k, v in list((top[0] if top else {}).items())[:5]},
        }
    return {
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "first_token": first,
        "usage": payload.get("usage"),
    }


def build_body(model: str, length: int, pattern: tuple[int, ...], top_logprobs: int) -> bytes:
    ids = list(islice(cycle(pattern), length))
    return json.dumps(
        {
            "model": model,
            "prompt": ids,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": top_logprobs,
        }
    ).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18962/v1/completions")
    parser.add_argument("--model", default="deepseek-v41-ced-tiny")
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--filler-count", type=int, default=0)
    parser.add_argument("--filler-length", type=int, default=2000)
    parser.add_argument("--restart-container", default="dsv41-ced-tiny-clip1-proxy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = build_body(args.model, args.length, TOKEN_PATTERN, args.top_logprobs)
    filler_body = build_body(args.model, args.filler_length, (31, 7, 101, 4099, 65537), 1)

    def send(payload: bytes, label: str) -> dict:
        for tries in range(3):
            status, raw, wall = post(args.url, payload, args.timeout)
            if status > 0:
                summary = summarize(raw)
                summary.update(http_status=status, wall_s=round(wall, 2))
                return summary
            print(f"[probe] {label} attempt {tries + 1} transport failure: "
                  f"{raw[:120]!r}; restarting proxy", flush=True)
            subprocess.run(["docker", "restart", args.restart_container],
                           capture_output=True, text=True)
            time.sleep(15)
        return {"http_status": -1, "transport_error": "retries exhausted"}

    for i in range(args.filler_count):
        res = send(filler_body, f"filler{i}")
        print(f"[probe] filler {i + 1}/{args.filler_count} http={res.get('http_status')}", flush=True)

    for i in range(1, args.repeats + 1):
        res = send(body, f"repeat{i}")
        res["index"] = i
        res["length"] = args.length
        with args.out.open("a") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
        ft = res.get("first_token") or {}
        print(
            f"[probe] {i:02d}/{args.repeats} http={res.get('http_status')} "
            f"wall={res.get('wall_s')} text={res.get('text')!r} "
            f"lp={ft.get('logprob')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
