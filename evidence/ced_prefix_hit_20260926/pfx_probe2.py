#!/usr/bin/env python3
"""前缀缓存**正确性**探针：针在 prompt 里，判据是答案对不对 + 冷/热是否一致。

对每个 prompt 做：冷 → 热 → 热，比较
  * 答案是否等于期望针值（正确性）
  * 冷与热是否给出**完全相同**的答案（缓存路径一致性）
判定用服务端 metrics（响应里的 cached_tokens 恒为 0，不可用）。
"""
import json, sys, time, urllib.request

BASE = sys.argv[1]; TOK = sys.argv[2]; N = int(sys.argv[3])
MODEL = "deepseek-v41-ced-pd"
NEEDLE = "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"
Q = "运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。"
EXPECT = "RB9N-6014"

def post(payload, timeout=1800):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return time.monotonic() - t0, d

def tok(text):
    req = urllib.request.Request(f"{TOK}/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(json.loads(r.read().decode())["count"])

def hits():
    out = {}
    for port in (18990, 18991):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:prefix_cache_hits_total"):
                    out[port] = float(line.rsplit(" ", 1)[1])
    return out

corpus = open("data/hongloumeng.txt", encoding="utf-8", errors="replace").read()
M = "========正文========"
body = corpus[corpus.index(M) + len(M):] if M in corpus else corpus

def build(offset, target):
    lo, hi, best = 1, target * 4, (0, -1)
    while lo <= hi:
        mid = (lo + hi) // 2
        t = body[offset:offset + mid]
        if offset + mid > len(body): t = body[offset:] + body[:mid - (len(body) - offset)]
        # 针插在 80% 深度
        cut = int(len(t) * 0.8)
        user = t[:cut] + "\n" + NEEDLE + "\n" + t[cut:] + "\n\n" + Q
        n = tok(user)
        if best[1] < 0 or abs(n - target) < abs(best[1] - target): best = (mid, n, user)
        if n == target: break
        lo, hi = (mid + 1, hi) if n < target else (lo, mid - 1)
    return best[2], best[1]

for label, off in (("P1", 0), ("P2", len(body)//2)):
    prompt, n = build(off, N)
    print(f"[{label}] prompt={n} tok（含针，插在 80% 深度）")
    prev = None
    for step in ("cold", "hit1", "hit2"):
        h0 = hits()
        wall, d = post({"model": MODEL, "messages": [{"role":"user","content":prompt}],
                        "max_tokens": 16, "temperature": 0.0, "stream": False})
        h1 = hits()
        ans = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
        dh = h1.get(18991,0) - h0.get(18991,0)
        ok = EXPECT in (ans or "")
        same = "" if prev is None else ("相同" if ans == prev else "**不同**")
        print(f"   {step:5s} wall={wall:7.2f}s  D_hits增量={dh:>8.0f}  答案正确={ok}  与上一步{same}"
              f"  ans={str(ans)[:36]!r}")
        prev = ans
