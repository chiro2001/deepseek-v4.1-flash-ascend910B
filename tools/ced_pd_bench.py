#!/usr/bin/env python3
"""PD 端到端性能：prefill token/s 与 decode ms/step（流式逐块计时）。

为什么需要单独的工具：`ced_pd_acceptance.py` 的 `tpot_ms` 对**非流式**请求等于
`wall / completion_tokens`，里面含了 prefill，不能当 decode 速率用。
本工具用流式响应，把每一块（每个 token）的到达时间单独记下来：

    ttft        = 首个内容块到达时间       → prefill token/s = prompt_tokens / ttft
    step[i]     = 第 i+1 块与第 i 块的时间差 → decode ms/step  = 中位数(step[1:])

时间戳用 `time.monotonic()` 在收到每个 SSE 块的瞬间取，避免解析开销污染。

用法：
  python3 tools/ced_pd_bench.py \
      --base-url http://127.0.0.1:18992 --tokenize-url http://127.0.0.1:18990 \
      --model deepseek-v41-ced-pd --corpus data/hongloumeng.txt \
      --contexts 32768,144000 --max-tokens 128 --repeat 2 \
      --out results/ced_bench_pd.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ced_pd_acceptance import SYSTEM, count_tokens, slice_for_tokens  # noqa: E402

NEEDLE = "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"
QUESTION = "运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。"


def stream_timed(url: str, payload: dict, timeout: float):
    """返回 (status, marks, n_text_chunks, usage, finish_reason, raw_tail)。

    marks[0] = 请求发出后到**首个内容块**的时间（TTFT）
    marks[i] = 第 i 块与第 i-1 块的间隔
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    first = None
    marks: list[float] = []
    prev = None
    usage = None
    finish = None
    text = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                choice = (chunk.get("choices") or [{}])[0]
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    if first is None:
                        first = now - started
                        marks.append(first)
                    else:
                        marks.append(now - prev)
                    prev = now
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except urllib.error.HTTPError as err:
        return err.code, [], 0, None, None, err.read()[:400].decode("utf-8", "replace")
    except Exception as err:  # noqa: BLE001
        return -1, [], 0, None, None, repr(err)[:400]
    return status, marks, len(text), usage, finish, "".join(text)[:120]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokenize-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="data/hongloumeng.txt")
    ap.add_argument("--contexts", required=True, help="逗号分隔的 token 目标")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="强制生成满 max_tokens 个 token（decode 速率测量必需；"
                         "否则模型答完就停，只统计到 4~7 步）")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()
    rows = []
    print(f"{'ctx_target':>10} {'prompt':>9} {'rep':>3} {'ttft_s':>8} {'prefill_tok/s':>14} "
          f"{'tokens':>7} {'decode_ms/step':>15} {'p10':>8} {'p90':>8} {'decode_tok/s':>13}")
    for target in [int(x) for x in args.contexts.split(",") if x.strip()]:
        base, _ = slice_for_tokens(args.tokenize_url, args.model, corpus, target, 0)
        prompt = base + "\n" + NEEDLE + "\n\n" + QUESTION
        prompt_tokens = count_tokens(args.tokenize_url, args.model, prompt)
        for rep in range(args.repeat):
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if args.ignore_eos:
                payload["ignore_eos"] = True
            status, marks, ntok, usage, finish, tail = stream_timed(
                args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
            )
            if status != 200 or not marks:
                print(f"{target:>10} {prompt_tokens:>9} {rep:>3} FAIL status={status} {tail[:60]!r}")
                rows.append({"target": target, "prompt_tokens": prompt_tokens, "rep": rep,
                             "status": status, "error": tail})
                continue
            ttft = marks[0]
            steps = marks[1:]
            med = statistics.median(steps) * 1000.0 if steps else float("nan")
            # vLLM 在客户端读得慢时会把多个 token 合进一个 SSE 块，于是"块间隔"
            # 会偶尔翻倍，中位数因此偏高。用**真实 token 数**（usage）算平均更稳：
            #   sum(steps) = 首块到末块的时间 = 纯 decode 总时长
            decode_total_s = sum(steps)
            true_decode_tokens = ((usage or {}).get("completion_tokens") or len(steps) + 1) - 1
            avg_ms = 1000.0 * decode_total_s / max(1, true_decode_tokens)
            p10 = sorted(steps)[len(steps) // 10] * 1000.0 if len(steps) >= 10 else med
            p90 = sorted(steps)[-max(1, len(steps) // 10)] * 1000.0 if len(steps) >= 10 else med
            row = {
                "target": target, "prompt_tokens": prompt_tokens, "rep": rep,
                "status": status, "ttft_s": round(ttft, 3),
                "prefill_tok_per_s": round(prompt_tokens / ttft, 1),
                "decode_tokens": len(steps),
                "usage_completion_tokens": (usage or {}).get("completion_tokens"),
                "decode_total_s": round(decode_total_s, 3),
                "decode_ms_per_token_avg": round(avg_ms, 2),
                "decode_tok_per_s_avg": round(1000.0 / avg_ms, 3) if avg_ms > 0 else None,
                "decode_ms_per_step": round(med, 1),
                "decode_ms_p10": round(p10, 1), "decode_ms_p90": round(p90, 1),
                "decode_tok_per_s": round(1000.0 / med, 3) if med == med else None,
                "first_step_ms": round(steps[0] * 1000.0, 1) if steps else None,
                "last_step_ms": round(steps[-1] * 1000.0, 1) if steps else None,
                "usage": usage, "finish_reason": finish, "answer_head": tail,
            }
            rows.append(row)
            print(f"{target:>10} {prompt_tokens:>9} {rep:>3} {ttft:>8.3f} "
                  f"{row['prefill_tok_per_s']:>14.1f} {len(steps):>7} "
                  f"avg={avg_ms:>7.1f} med={med:>7.1f} p90={p90:>6.1f} "
                  f"{row['decode_tok_per_s_avg'] or 0:>10.2f} tok/s")
            if args.out:
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                json.dump(rows, open(args.out, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
