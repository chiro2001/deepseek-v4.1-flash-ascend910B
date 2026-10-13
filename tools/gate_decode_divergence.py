#!/usr/bin/env python3
"""找出 decode 从**第几个 token** 开始分叉 —— 区分"第一步就抖"还是"累积后抖"。

已知（本轮实测）：同 prompt 的 **prefill 完全确定**（1240 位置 max|Δ|=0）。
而 decode 会分叉。关键问题是**多快分叉**：

  · 第 2 个 token（即第 1 个 decode step）就分叉 ⇒ 单步 allreduce 就不定序；
  · 前面若干 token 一致、之后才分叉 ⇒ 逐步累积的微小抖动被放大。

做法：同一 prompt 连续 R 轮、每轮 `max_tokens=MT`、`logprobs=1`，
逐 token 比较 (logprob, argmax)；给出**首个分叉的 token 序号**（取所有轮的最早值）。

用法: gate_decode_diverge.py <base> [mt] [rounds]
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 48
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
CH = string.ascii_letters + string.digits

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
nonce = "".join(random.choice(CH) for _ in range(24))
prompt = "[%s]\n%s\n\n请概括上文。" % (nonce, body[:1200])


def ask():
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True, "logprobs": 1}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.perf_counter() - t0
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return list(lp.get("token_logprobs") or []), list(lp.get("tokens") or []), dt


print("prompt ≈ %d 字符，每轮生成 %d token，共 %d 轮（先预热 1 次）" % (len(prompt), MT, ROUNDS),
      flush=True)
ask()

runs = [ask() for _ in range(ROUNDS)]
for k, (lpv, tk, dt) in enumerate(runs):
    print("  r%02d  %.2fs  %d 个 token" % (k, dt, len(lpv)), flush=True)

def analyze(idxs, label):
    """以 idxs[0] 为基准，比较 idxs[1:]；同时给出 idxs 内部的**两两**最大差与 token 不同数。"""
    base = idxs[0]
    print()
    print("=== %s（基准 = r%02d）===" % (label, base))
    print("%8s %16s %14s" % ("token#", "max|Δlogprob|", "token 不同轮数"))
    first_bad = None
    for j in range(MT):
        mx = 0.0
        tok_diff = 0
        bv = runs[base][0][j] if j < len(runs[base][0]) else None
        bt = runs[base][1][j] if j < len(runs[base][1]) else None
        for k in idxs[1:]:
            if j >= len(runs[k][0]):
                continue
            v, t = runs[k][0][j], runs[k][1][j]
            if bv is not None and v is not None:
                mx = max(mx, abs(v - bv))
            if bt is not None and t is not None and t != bt:
                tok_diff += 1
        if (mx > 0 or tok_diff > 0) and first_bad is None:
            first_bad = j
        if j < 24 or j % 8 == 0:
            print("%8d %16.4e %14d" % (j, mx, tok_diff))

    # idxs 内部的真·两两（排除"基准特殊"的可能）
    worst = 0.0
    worst_pair = None
    tokpair = 0
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            ia, ib = idxs[a], idxs[b]
            n = min(len(runs[ia][1]), len(runs[ib][1]))
            for j in range(n):
                if runs[ia][1][j] != runs[ib][1][j]:
                    tokpair += 1
                if runs[ia][0][j] is not None and runs[ib][0][j] is not None:
                    d = abs(runs[ia][0][j] - runs[ib][0][j])
                    if d > worst:
                        worst, worst_pair = d, (ia, ib, j)
    print("  首个分叉 token# = %s" % first_bad)
    print("  内部两两：最大 |Δlogprob| = %.4e（r%s/r%s @token%s）；token 不同的 (轮对×位置) 数 = %d"
          % (worst, *(worst_pair or ("-", "-", "-")), tokpair))
    return first_bad, worst


fb_cold, _ = analyze(list(range(ROUNDS)), "全部轮 vs r00（注意 r00 是冷路径）")
if ROUNDS > 2:
    analyze(list(range(1, ROUNDS)), "热轮（r01..）内部")

print()
if fb_cold is None:
    print("⇒ 全部一致")
elif fb_cold == 0:
    print("⇒ 与 r00 相比**第 0 个 token 就分叉**；请看上面的「热轮内部」结论判断是否真非确定")
else:
    print("⇒ 与 r00 相比从第 %d 个 token 起分叉" % fb_cold)
