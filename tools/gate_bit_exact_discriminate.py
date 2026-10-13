#!/usr/bin/env python3
"""判别非确定性的来源：是「prefix cache 命中 vs 冷路径」还是「真·非确定」。

第一版（gate_bit_exact.py）以 r00 为基准 ⇒ 全部 FAIL。但 r00 是**冷启全量 prefill**
（实测 15.65 s），r01 之后全部命中 prefix cache（0.6~0.9 s）。
所以至少有两种可能：
  (A) 「缓存命中路径」与「冷路径」数值不同，但**各自内部**是确定的；
  (B) 真·非确定（同一条路径每次也不同）。

本工具输出三张对比：
  1. 每轮 vs **r00**（冷路径基线）
  2. 每轮 vs **r01**（首个缓存命中轮）
  3. 轮间两两：r01..rN 的**所有**配对最大差（只看"缓存路径"内部是否一致）

用法: gate_bit_exact2.py <base> <prompt_tokens> [rounds] [max_tokens]
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MT = int(sys.argv[4]) if len(sys.argv) > 4 else 32
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
CH = string.ascii_letters + string.digits

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = (body[i + 2:] if i >= 0 else body)


def tok_count(prompt):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": "deepseek-v41", "prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return int(d.get("count", d.get("token_count", -1)))


lo, hi = 100, min(len(body) - 1000, 400000)
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

nonce = "".join(random.choice(CH) for _ in range(32))
prompt = "[%s]\n%s\n\n请概括上文。" % (nonce, body[:mid])
print("prompt = %d token，%d 轮（先发 2 条预热请求跨过编译）" % (tok_count(prompt), ROUNDS), flush=True)


def ask():
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True, "logprobs": 1}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.perf_counter() - t0
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return ch["text"], list(lp.get("token_logprobs") or []), dt


# 预热（不计入）
for _ in range(2):
    ask()

runs = [ask() for _ in range(ROUNDS)]
print()
for k, (txt, lp, dt) in enumerate(runs):
    print("  r%02d  %.2fs  %d 个 logprob  %r" % (k, dt, len(lp), txt[:40]), flush=True)


def dmax(a, b):
    n = min(len(a), len(b))
    mx = 0.0
    for j in range(n):
        if a[j] is None or b[j] is None:
            continue
        mx = max(mx, abs(a[j] - b[j]))
    return mx


print()
print("=== (1) 每轮 vs r00 ===")
for k in range(1, ROUNDS):
    print("  r%02d vs r00: max|Δ|=%.3e  文本一致=%s"
          % (k, dmax(runs[k][1], runs[0][1]), runs[k][0] == runs[0][0]))

print()
print("=== (2) 每轮 vs r01 ===")
for k in range(1, ROUNDS):
    print("  r%02d vs r01: max|Δ|=%.3e  文本一致=%s"
          % (k, dmax(runs[k][1], runs[1][1]), runs[k][0] == runs[1][0]))

print()
print("=== (3) r01..rN 两两（只看缓存路径内部）===")
worst = 0.0
worst_pair = None
all_same_txt = True
for a in range(1, ROUNDS):
    for b in range(a + 1, ROUNDS):
        d = dmax(runs[a][1], runs[b][1])
        if d > worst:
            worst, worst_pair = d, (a, b)
        if runs[a][0] != runs[b][0]:
            all_same_txt = False
print("  r01..r%02d 最大配对差 = %.3e（出现在 r%d/r%d）" % (ROUNDS - 1, worst, *(worst_pair or (-1, -1))))
print("  这些轮的生成文本全部一致 = %s" % all_same_txt)
print()
if worst == 0.0 and all_same_txt:
    print("⇒ **缓存路径内部完全确定**；唯一差异是 r00（冷路径）⇒ 属 (A)，与并发/桶无关")
elif worst > 0.0:
    print("⇒ **连缓存路径内部都不确定** ⇒ 属 (B)，真·非确定")
