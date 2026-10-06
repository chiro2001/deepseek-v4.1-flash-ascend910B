#!/usr/bin/env python3
"""干净的单请求 prefill / TTFT 曲线（nonce 破除 prefix cache + 缓存命中率核验）。

为什么重做：tp8k5 的 PREFIX=1 会让**重复前缀**命中缓存。实测同一 8208-token prompt
第二次只花 0.172s，而冷启约 0.59s ⇒ 之前那组 prefill 数字（4K/16K/64K 用嵌套前缀）
被前面几次测量预热过，不可用。
这里每条 prompt 前面加唯一 random nonce，并用 `prompt_tokens_cached_total` 核验命中≈0。

用法: tp8_prefill_clean.py <base> <len_list> [max_tokens]
"""
import json
import random
import statistics
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
LENS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "4096,16384,65536").split(",")]
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 8
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


def ttft(prompt):
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data: ") and "[DONE]" not in line:
                return time.perf_counter() - t0
    return None


print("%10s %12s %12s %12s %10s %10s" % ("目标tok", "实际tok", "TTFT_s", "tok/s", "缓存tok", "命中%"))
for want in LENS:
    lo, hi = 100, min(len(body), 400000)
    best_mid, best_n = hi, -1
    for _ in range(26):
        mid = (lo + hi) // 2
        n = tok_count(body[:mid] + "\n\n请概括上文。")
        if n < 0:
            break
        if best_n < 0 or abs(n - want) < abs(best_n - want):
            best_mid, best_n = mid, n
        if abs(n - want) <= max(8, want // 200):
            break
        if n < want:
            lo = mid + 1
        else:
            hi = mid - 1
    # 每次都用唯一 nonce ⇒ 前缀永不命中
    nonce = "".join(random.choice(CH) for _ in range(64))
    prompt = "[%s]\n%s\n\n请概括上文。" % (nonce, body[:best_mid])
    ntok = tok_count(prompt)
    a = _metrics()
    t = ttft(prompt)
    b = _metrics()
    pt = b.get("vllm:prompt_tokens_total", 0.0) - a.get("vllm:prompt_tokens_total", 0.0)
    pc = b.get("vllm:prompt_tokens_cached_total", 0.0) - a.get("vllm:prompt_tokens_cached_total", 0.0)
    frac = (100 * pc / pt) if pt > 0 else 0.0
    print("%10d %12d %12.3f %12.1f %10.0f %9.1f%%"
          % (want, ntok, t, ntok / t, pc, frac), flush=True)
    time.sleep(2.0)
