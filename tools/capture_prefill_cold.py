#!/usr/bin/env python3
"""采集**冷 prefill**（前缀缓存未命中）的 profile，并报出 TTFT 与吞吐。

为什么要单独做：`capture_prefill_profile.sh` 走 `bench_concurrency`，而它从语料**开头**
取切片 ⇒ 前几次实验已经把这段灌进 prefix cache，实测 TTFT 只有 0.46 s（32K），
根本不是冷 prefill。本脚本从语料**远端**取切片（可用 `--offset-frac` 指定），
并用随机 nonce 打头，确保任何缓存都命中不了。

用法:
  capture_prefill_cold.py --port 19210 --n 2 --tokens 32768 [--offset-frac 0.6] [--out-tokens 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request


def post(url: str, payload: dict | None = None, timeout: float = 1800.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"{}")


def tokenize(base: str, model: str, text: str) -> int:
    d = post(f"{base}/tokenize", {"model": model, "prompt": text}, timeout=120)
    return int(d.get("count", d.get("token_count", -1)))


def build_prompt(corpus: str, tokens: int, offset_frac: float, nonce: str, base: str, model: str) -> str:
    """二分逼近到 ~tokens 个 token（用服务端 /tokenize 校准）。"""
    n = len(corpus)
    start = int(n * offset_frac) % max(1, n - 200)
    body = corpus[start:] + corpus[:start]
    lo, hi = 0, len(body) - 1
    for _ in range(26):
        mid = (lo + hi) // 2
        t = tokenize(base, model, nonce + body[:mid] + "\n\n请用一句话概括上文。")
        if t <= tokens:
            lo = mid
        else:
            hi = mid
    return nonce + body[:lo] + "\n\n请用一句话概括上文。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19210)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--offset-frac", type=float, default=0.6)
    ap.add_argument("--out-tokens", type=int, default=4)
    ap.add_argument("--corpus", default="/home/l00886679/cedpd-repo/data/hongloumeng.txt")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    model = "deepseek-v41"
    corpus = open(a.corpus, encoding="utf-8").read()
    stamp = f"【会话 {time.time_ns()}】\n"

    prompts = []
    for i in range(a.n):
        frac = (a.offset_frac + i * 0.11) % 1.0
        p = build_prompt(corpus, a.tokens, frac, stamp + f"【第{i}段】\n", base, model)
        prompts.append(p)
    print(f"prompt {len(prompts)} 条（nonce 各自不同，远端切片）", flush=True)

    print("start_profile:", end=" ", flush=True)
    post(f"{base}/start_profile")
    print("ok", flush=True)

    t0 = time.time()

    def fire(i: int) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompts[i]}],
            "max_tokens": a.out_tokens,
            "temperature": 0.0,
        }
        t = time.time()
        d = post(f"{base}/v1/chat/completions", payload)
        return {"i": i, "ttft_s": time.time() - t, "usage": d.get("usage", {})}

    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        res = list(ex.map(fire, range(len(prompts))))
    wall = time.time() - t0

    print("stop_profile:", end=" ", flush=True)
    post(f"{base}/stop_profile", timeout=600)
    print("ok", flush=True)

    tot = sum(r["usage"].get("prompt_tokens", 0) for r in res)
    for r in res:
        print(f"  req{r['i']}: prompt={r['usage'].get('prompt_tokens')} tok  TTFT={r['ttft_s']:.2f} s")
    print(f"合计 prompt {tot} tok / 墙钟 {wall:.2f} s ⇒ **{tot/wall:.0f} tok/s**（含尾 token 与采样，略低于纯 prefill）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
