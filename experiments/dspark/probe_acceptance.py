#!/usr/bin/env python3
"""CED+DSpark 快速探针：跑一条请求，报接受长度 A 与 decode 速度。

判据：A ≈ 1.0 ⇒ 草稿没产出（图/上下文 KV 静默失效）；A ≥ 1.5 ⇒ 草稿在干活。
**ms/step 不能单独当判据**：A≈1.0 时每步只出 1 个 token，ms/step 反而更好看
（reports/draft-graph-negative-control.md 的负控就是这么骗人的）。

复用 tools/ced_pd_acceptance.py 的语料切分与 /tokenize 校准，避免自己造一套。
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from ced_pd_acceptance import load_corpus, post_json, slice_for_tokens  # noqa: E402

NEEDLE = "松江府的密报编号是 QX-7731，签发人姓沈。"


def get(url, timeout=30):
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def metrics(url):
    out = {}
    for line in get(url + "/metrics").splitlines():
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
    ap.add_argument("--metrics-url", default="http://127.0.0.1:18991",
                    help="指标在 D 上（代理不透传 /metrics）")
    ap.add_argument("--model", default="deepseek-v41-ced-pd")
    ap.add_argument("--corpus", default="data/hongloumeng.txt")
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--needle", action="store_true",
                    help="在 prompt 中段插入唯一事实，末尾问它（验正确性）")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    body, ntok = slice_for_tokens(args.tokenize_url, args.model, corpus, args.context_tokens)
    prompt = body
    if args.needle:
        cut = len(body) // 2
        prompt = (
            body[:cut]
            + "\n\n【内部备忘】" + NEEDLE + "\n\n"
            + body[cut:]
            + "\n\n问题：上面内部备忘里的密报编号和签发人姓氏分别是什么？只回答编号与姓氏。\n"
        )

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }
    before = metrics(args.metrics_url)
    t0 = time.perf_counter()
    status, raw, wall, _b = post_json(args.base_url + "/v1/completions", payload, 1800)
    status2, raw2, _w2, _b2 = None, None, None, None
    if status != 200:
        print(f"[FAIL] status={status} body={raw[:400]!r}")
        raise SystemExit(1)
    resp = json.loads(raw)
    # post_json 把耗时藏在返回值第 3 项；再单独量一次总墙钟（含网络）
    dt = time.perf_counter() - t0
    time.sleep(2)
    after = metrics(args.metrics_url)

    def d(k):
        return after.get(k, 0.0) - before.get(k, 0.0)

    draft = d("vllm:spec_decode_num_draft_tokens_total")
    acc = d("vllm:spec_decode_num_accepted_tokens_total")
    n_out = resp["usage"]["completion_tokens"]
    a = (1.0 + 7.0 * acc / draft) if draft else float("nan")  # 与 vLLM 的 Mean acceptance length 同口径
    print(json.dumps({
        "context_tokens": ntok,
        "completion_tokens": n_out,
        "wall_s": round(dt, 2),
        "decode_tok_s": round(n_out / dt, 2),
        "draft_tokens": draft,
        "accepted_tokens": acc,
        "acceptance_len_A": round(a, 3), "spec_tokens": 7,
        "text": resp["choices"][0]["text"][:200],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
