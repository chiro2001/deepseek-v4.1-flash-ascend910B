#!/usr/bin/env python3
"""N=2 冷 prefill 重测（无 profiler 版）——对齐 prefill_cold.log 的口径。

用法: prefill_n2_noprof.py --port 19210 --n 2 --tokens 32768 --repeat 2
每轮重新生成 nonce（保证冷前缀），不调用 /start_profile。
"""
import argparse, concurrent.futures as cf, json, time, urllib.request

def post(url, payload=None, timeout=1800.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"{}")

def tokenize(base, model, text):
    d = post(f"{base}/tokenize", {"model": model, "prompt": text}, timeout=120)
    return int(d.get("count", d.get("token_count", -1)))

def build_prompt(corpus, tokens, offset_frac, nonce, base, model):
    n = len(corpus)
    start = int(n * offset_frac) % max(1, n - 200)
    body = corpus[start:] + corpus[:start]
    lo, hi = 0, len(body) - 1
    for _ in range(26):
        mid = (lo + hi) // 2
        t = tokenize(base, model, nonce + body[:mid] + "\n\n请用一句话概括上文。")
        if t <= tokens: lo = mid
        else: hi = mid
    return nonce + body[:lo] + "\n\n请用一句话概括上文。"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19210)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--offset-frac", type=float, default=0.6)
    ap.add_argument("--out-tokens", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--corpus", default="/home/l00886679/cedpd-repo/data/hongloumeng.txt")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = "deepseek-v41"
    corpus = open(a.corpus, encoding="utf-8").read()
    print(f"N={a.n} 冷 prefill × {a.repeat} 轮（无 profiler）", flush=True)
    for r in range(a.repeat):
        if r: time.sleep(a.gap)
        stamp = f"【会话 {time.time_ns()}】\n"
        prompts = []
        for i in range(a.n):
            frac = (a.offset_frac + i * 0.11) % 1.0
            prompts.append(build_prompt(corpus, a.tokens, frac, stamp + f"【第{i}段】\n", base, model))
        t0 = time.time()
        def fire(i):
            payload = {"model": model, "messages": [{"role": "user", "content": prompts[i]}],
                       "max_tokens": a.out_tokens, "temperature": 0.0}
            t = time.time()
            d = post(f"{base}/v1/chat/completions", payload)
            return {"i": i, "ttft_s": time.time() - t, "usage": d.get("usage", {})}
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            res = list(ex.map(fire, range(len(prompts))))
        wall = time.time() - t0
        tot = sum(x["usage"].get("prompt_tokens", 0) for x in res)
        ttfts = "  ".join(f"req{x["i"]} TTFT={x["ttft_s"]:.2f}s" for x in res)
        print(f"[run{r}] {ttfts} | {tot} tok / {wall:.2f}s ⇒ {tot/wall:.0f} tok/s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
