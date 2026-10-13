#!/usr/bin/env python3
"""判别：**prefill 本身是否确定**？（零重启、短 prompt、输出有界）

动机：上一轮发现 tp8k5（TP=8）在 temperature=0 下**生成结果非确定**，而 tiny（TP=2）完全确定。
但"生成"包含 prefill（不涉及投机解码）与 decode（涉及投机解码 + 每步 allreduce 累积）。
先要分清抖动发生在哪一段。

做法：用**短 prompt**（几百 token）请求 `prompt_logprobs=1`（每位置只取 top-1 ⇒ 输出很小），
连续跑 R 轮，逐位置比对 prompt_logprobs。
  · prefill 确定 ⇒ 同位置 logprob 完全一致（max|Δ|=0）
  · prefill 不确定 ⇒ 能直接看到差值，且**与投机解码无关**

用法: gate_prefill_det.py <base> [prompt_tokens] [rounds]
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
CH = string.ascii_letters + string.digits

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body


def tok_count(prompt):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": "deepseek-v41", "prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return int(d.get("count", d.get("token_count", -1)))


# 固定 nonce ⇒ 所有轮次 prompt 完全相同（并保证首轮不被缓存）
nonce = "".join(random.choice(CH) for _ in range(24))
seg = body[:1500]
prompt = "[%s]\n%s" % (nonce, seg)
ntok = tok_count(prompt)
print("prompt = %d token，%d 轮（比较每个 prompt 位置的 top-1 logprob）" % (ntok, ROUNDS), flush=True)


def ask():
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": 1,
               "temperature": 0.0, "ignore_eos": True, "prompt_logprobs": 1}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.perf_counter() - t0
    plp = d["choices"][0].get("prompt_logprobs") or []
    # 每个位置是 {token_id_str: {"logprob": f, "rank": i, "decoded_token": s}} 或 None
    # 取 rank==1（或 logprob 最大）的那一项 ⇒ 比较 (token_id, logprob) 是最强的判据
    vals = []
    for ent in plp:
        if not isinstance(ent, dict) or not ent:
            vals.append(None)
            continue
        best_tok, best_lp = None, None
        for tok, info in ent.items():
            if not isinstance(info, dict):
                continue
            lp = info.get("logprob")
            if lp is None:
                continue
            if best_lp is None or lp > best_lp:
                best_tok, best_lp = tok, lp
        vals.append((best_tok, best_lp) if best_lp is not None else None)
    return vals, dt


runs = []
for r in range(ROUNDS):
    vals, dt = ask()
    runs.append(vals)
    print("  r%02d  %.2fs  位置数=%d  非空=%d" % (r, dt, len(vals), sum(1 for v in vals if v is not None)),
          flush=True)
    time.sleep(0.5)

print()


def dmax(a, b):
    """比较 (token_id, logprob) 序列；返回 (最大 logprob 差, 首个偏离位置, 偏离位置数, token 不一致数)。"""
    n = min(len(a), len(b))
    mx = 0.0
    first = None
    cnt = 0
    tok_diff = 0
    for j in range(n):
        if a[j] is None or b[j] is None:
            continue
        if a[j][0] != b[j][0]:
            tok_diff += 1
        d = abs(a[j][1] - b[j][1])
        if d > 0:
            cnt += 1
            if first is None:
                first = j
        mx = max(mx, d)
    return mx, first, cnt, tok_diff


print("=== 每轮 vs r00（同 prompt、同参数）===")
worst = 0.0
for r in range(1, ROUNDS):
    mx, first, cnt, tok_diff = dmax(runs[r], runs[0])
    worst = max(worst, mx)
    print("  r%02d vs r00: max|Δlogprob|=%.3e  首个偏离=%-6s  偏离位置数=%d  **argmax token 不同数=%d**"
          % (r, mx, first, cnt, tok_diff))

print()
if worst == 0.0:
    print("⇒ **prefill 完全确定**（max|Δ|=0）⇒ 抖动出在 decode 段（投机解码 + 每步 allreduce）")
else:
    print("⇒ **prefill 本身就不确定**（max|Δ|=%.3e）⇒ 抖动在主干，与投机解码无关" % worst)
