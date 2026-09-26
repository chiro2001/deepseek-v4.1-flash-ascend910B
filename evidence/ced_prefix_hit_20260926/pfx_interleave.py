#!/usr/bin/env python3
"""交错命中测试：检验"D 侧对 hashed 块预清零"会不会破坏**别的**请求的缓存。

顺序：P1冷 → P2冷 → P1热 → P2热 → P1热 → P2热
判据：每一次 P1/P2 都必须答对且与各自冷值逐字节相同。
若 P2 的处理把 P1 在 G7..G11 的缓存块清零了，那么 P1 的第 3 次（热）就会错。
"""
import json, sys, time, urllib.request

BASE, TOK, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
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

def d_hits():
    with urllib.request.urlopen("http://127.0.0.1:18991/metrics", timeout=20) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_hits_total"):
                return float(line.rsplit(" ", 1)[1])
    return 0.0

corpus = open("data/hongloumeng.txt", encoding="utf-8", errors="replace").read()
M = "========正文========"
body = corpus[corpus.index(M) + len(M):] if M in corpus else corpus

def build(off, target):
    lo, hi, best = 1, target*4, None
    while lo <= hi:
        mid = (lo+hi)//2
        t = body[off:off+mid]
        if off+mid > len(body): t = body[off:] + body[:mid-(len(body)-off)]
        cut = int(len(t)*0.8)
        user = t[:cut] + "\n" + NEEDLE + "\n" + t[cut:] + "\n\n" + Q
        n = tok(user)
        if best is None or abs(n-target) < abs(best[1]-target): best = (user, n)
        if n == target: break
        lo, hi = (mid+1, hi) if n < target else (lo, mid-1)
    return best

P = {"P1": build(0, N), "P2": build(len(body)//2, N)}
print("[interleave] " + ", ".join(f"{k}={v[1]}tok" for k,v in P.items()))
first = {}
order = ["P1", "P2", "P1", "P2", "P1", "P2"]
for i, name in enumerate(order, 1):
    prompt, _ = P[name]
    h0 = d_hits()
    wall, d = post({"model": MODEL, "messages":[{"role":"user","content":prompt}],
                    "max_tokens":16, "temperature":0.0, "stream":False})
    h1 = d_hits()
    ans = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
    ok = EXPECT in (ans or "")
    first.setdefault(name, ans)
    kind = "冷" if i <= 2 else "热"
    print(f"  {i} {name}({kind}) wall={wall:7.2f}s D_hits增量={h1-h0:>8.0f} 正确={ok} "
          f"与首次{'相同' if ans == first[name] else '**不同**'} ans={str(ans)[:20]!r}")
