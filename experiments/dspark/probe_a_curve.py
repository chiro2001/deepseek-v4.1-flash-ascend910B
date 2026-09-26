#!/usr/bin/env python3
"""A 的步数曲线：流式解码时每隔 N 个 token 采一次 metrics，看接受长度是否随步数上升。

判据：
  * A 从 ~1.0 升到 ≥1.5（约 128 步后）⇒ 重放步**没有**往草稿窗口写 context KV，
    只有 decode 步在补 —— 即 CED 的重放没接上 DSpark 的上下文写入。
  * A 全程 ~1.08 ⇒ 更严重：草稿窗口写了但读不到（写错页 / 查询侧块表不对）。
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from ced_pd_acceptance import load_corpus, slice_for_tokens  # noqa: E402


def metrics(url):
    out = {}
    with urllib.request.urlopen(url + "/metrics", timeout=30) as fh:
        for line in fh.read().decode().splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18992")
    ap.add_argument("--tokenize-url", default="http://127.0.0.1:18990")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:18991")
    ap.add_argument("--model", default="deepseek-v41-ced-pd")
    ap.add_argument("--corpus", default="data/hongloumeng.txt")
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=24)
    args = ap.parse_args()

    body, ntok = slice_for_tokens(args.tokenize_url, args.model, args.corpus, args.context_tokens)
    payload = {"model": args.model, "prompt": body, "max_tokens": args.max_tokens,
               "temperature": 0.0, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(args.base_url + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    m0 = metrics(args.metrics_url)

    def d(k, cur):
        return cur.get(k, 0.0) - m0.get(k, 0.0)

    rows, n, t0 = [], 0, time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if not raw.startswith(b"data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == b"[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except Exception:
                continue
            if obj.get("choices") and obj["choices"][0].get("text"):
                n += 1
                if n % args.sample_every == 0:
                    cur = metrics(args.metrics_url)
                    dr, ac = d("vllm:spec_decode_num_draft_tokens_total", cur), d(
                        "vllm:spec_decode_num_accepted_tokens_total", cur)
                    rows.append({"tok": n, "A": round(1.0 + 7.0 * ac / dr, 3) if dr else None,
                                 "t_s": round(time.perf_counter() - t0, 1)})
    cur = metrics(args.metrics_url)
    dr, ac = d("vllm:spec_decode_num_draft_tokens_total", cur), d(
        "vllm:spec_decode_num_accepted_tokens_total", cur)
    rows.append({"tok": n, "A": round(1.0 + 7.0 * ac / dr, 3) if dr else None,
                 "t_s": round(time.perf_counter() - t0, 1), "final": True})
    print(json.dumps({"context_tokens": ntok, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
