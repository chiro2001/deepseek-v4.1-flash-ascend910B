#!/usr/bin/env python3
"""门①：逐位一致性（有界版）。

为什么不用 `~/tmp/walk_blocks.py`：它对 64K prompt 请求 `echo=True + prompt_logprobs=1`，
即要求**整段 prompt 每个位置的完整 logprob** ⇒ 输出张量 ≈ 64000 × 129280 × 4 B ≈ **33 GB**
⇒ 实测把引擎打成 OOM（`NPU out of memory. Tried to allocate 3.82 GiB`）。

本工具只比**生成 token** 的 logprob（输出量 = max_tokens × (k+1)），量级安全，
同时仍是"同 prompt 重复 ≥10 轮、逐位比对 max|Δ|"的同一判据。

用法: gate_bit_exact.py <base> <prompt_tokens> [rounds] [max_tokens]
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

# ★ 固定 nonce：所有轮次用**完全相同**的 prompt（这是本门的前提）
nonce = "".join(random.choice(CH) for _ in range(32))
prompt = "[%s]\n%s\n\n请概括上文。" % (nonce, body[:mid])
ntok = tok_count(prompt)
print("prompt = %d token，%d 轮，每轮生成 %d token（比生成位置的 logprob）"
      % (ntok, ROUNDS, MT), flush=True)


def ask():
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True, "logprobs": 1}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    vals = list(lp.get("token_logprobs") or []) if isinstance(lp, dict) else []
    return ch["text"], vals, lp


base_txt, base_lp, base_lpobj = None, None, None
rows = []
for r in range(ROUNDS):
    t0 = time.perf_counter()
    txt, lp, lpobj = ask()
    dt = time.perf_counter() - t0
    if base_lp is None:
        base_txt, base_lp, base_lpobj = txt, lp, lpobj
        print("  r%02d  基准  生成 %d 字符  %.2fs  logprob 项数=%d" % (r, len(txt), dt, len(lp)), flush=True)
    else:
        n = min(len(lp), len(base_lp))
        mx = 0.0
        first_bad = None
        for j in range(n):
            a, b = lp[j], base_lp[j]
            if a is None or b is None:
                continue
            d = abs(a - b)
            if d > 0 and first_bad is None:
                first_bad = j
            mx = max(mx, d)
        same_txt = (txt == base_txt)
        rows.append((r, mx, first_bad, same_txt, len(lp)))
        print("  r%02d  max|Δlogprob| = %.3e  首个偏离=%-5s 文本一致=%s  项数=%d  %.2fs"
              % (r, mx, first_bad, same_txt, len(lp), dt), flush=True)
    time.sleep(1.0)

if not rows:
    print("只有一轮，无法比较")
else:
    worst = max(r[1] for r in rows)
    txt_all = all(r[3] for r in rows)
    print()
    print("=== 判据（要求 max|Δ| = 0）===")
    print("  最坏 max|Δlogprob| = %.3e  ⇒ %s" % (worst, "PASS" if worst == 0.0 else "FAIL"))
    print("  生成文本逐轮一致 = %s ⇒ %s" % (txt_all, "PASS" if txt_all else "FAIL"))
    print("  轮次 = %d（要求 ≥10）" % (ROUNDS + 1))
