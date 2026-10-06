#!/usr/bin/env python3
"""干净的高并发 prefill 聚合吞吐（nonce + 零缓存命中核验）。

用法: tp8_prefill_conc_clean.py <base> <conc> [prompt_tokens] [max_tokens]
"""
import json
import random
import string
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WANT = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
MT = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
CH = string.ascii_letters + string.digits

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = (body[i + 2:] if i >= 0 else body) * 40


def _metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(" ", 1)
        if len(p) == 2:
            nm = p[0].split("{")[0]
            try:
                out[nm] = out.get(nm, 0.0) + float(p[1])
            except ValueError:
                pass
    return out


def tok_count(prompt):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": "deepseek-v41", "prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return int(d.get("count", d.get("token_count", -1)))


lo, hi = 100, min(len(body) // 2, 400000)
mid = hi
for _ in range(26):
    m = (lo + hi) // 2
    n = tok_count(body[:m] + "\n\n请概括上文。")
    if abs(n - WANT) <= max(8, WANT // 200):
        mid = m
        break
    if n < WANT:
        lo = m + 1
    else:
        hi = m - 1
    mid = m

off = max(mid + 1000, len(body) // max(1, CONC))
prompts = []
for k in range(CONC):
    nonce = "".join(random.choice(CH) for _ in range(64))
    prompts.append("[%s-%d]\n%s\n\n请概括上文。" % (nonce, k, body[k * off: k * off + mid]))
ntok = tok_count(prompts[0])
print("conc=%d  每条 ≈ %d token（nonce 保证零缓存命中）" % (CONC, ntok), flush=True)

res = [None] * CONC


def one(k):
    payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if line.startswith("data: ") and "[DONE]" not in line:
                    res[k] = time.perf_counter() - t0
                    return
    except Exception as exc:  # noqa: BLE001
        res[k] = None


a = _metrics()
ths = [threading.Thread(target=one, args=(k,)) for k in range(CONC)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.perf_counter() - t0
b = _metrics()

ok = [r for r in res if r]
pt = b.get("vllm:prompt_tokens_total", 0.0) - a.get("vllm:prompt_tokens_total", 0.0)
pc = b.get("vllm:prompt_tokens_cached_total", 0.0) - a.get("vllm:prompt_tokens_cached_total", 0.0)
tot = sum(ntok for _ in ok)
print("  成功 %d/%d  首token: min %.3fs 中位 %.3fs max %.3fs"
      % (len(ok), CONC, min(ok), sorted(ok)[len(ok) // 2], max(ok)), flush=True)
print("  全部完成墙钟 %.3fs ⇒ 聚合 prefill %,.0f tok/s".replace(",", "") % (wall, tot / wall))
print("  服务端 prompt_tokens Δ=%.0f（缓存 %.0f，命中 %.1f%%）⇒ 清洁度核验"
      % (pt, pc, 100 * pc / pt if pt else 0.0))
print("  对比单请求 7.1K tok/s ⇒ 增益 x%.2f" % (tot / wall / 7100.0))
