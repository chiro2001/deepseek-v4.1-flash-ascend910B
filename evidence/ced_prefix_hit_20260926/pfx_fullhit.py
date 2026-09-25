#!/usr/bin/env python3
"""**整池命中**探针：构造 N ≡ 1 (mod 128) 的 prompt，使缓存的整块正好覆盖 N-1 个 token。

为什么需要它：CED 的 D 侧在整池命中时 `num_external_tokens == 0`，会走一条与
"部分命中"完全不同的分支（不发 KV 传输）。历史崩溃
（`CED decoder expected 12 KV cache groups without DSpark`）只在**整池命中**时触发，
而 144K/1M 的常规探针大多落在"部分命中"（N-1 不是 128 的整数倍）⇒ 一直没复现。

顺序：冷 → 热 → 热，再换一个同结构的 prompt 重复一遍。
判据：答案正确 + 冷/热逐字节一致 + 命中计数器增长。
"""
import json, sys, time, urllib.request

BASE, TOK = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32769      # 32768 = 256*128
MODEL = "deepseek-v41-ced-pd"
NEEDLE = "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"
Q = "运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。"
EXPECT = "RB9N-6014"

def post(payload, timeout=600):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        return time.monotonic() - t0, d, r.status
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, {"error": e.read()[:200].decode("utf-8", "replace")}, e.code

def tok(text):
    req = urllib.request.Request(f"{TOK}/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(json.loads(r.read().decode())["count"])

def hits(port=18991):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_hits_total"):
                return float(line.rsplit(" ", 1)[1])
    return 0.0

corpus = open("data/hongloumeng.txt", encoding="utf-8", errors="replace").read()
M = "========正文========"
body = corpus[corpus.index(M) + len(M):] if M in corpus else corpus

def build(off, target, tries=12):
    """把**最终 prompt** 校准到正好 target 个 token。"""
    best = None
    for k in range(tries):
        o = (off + k * 41) % max(1, len(body) - 64)
        lo, hi = 1, target * 4
        while lo <= hi:
            mid = (lo + hi) // 2
            t = body[o:o + mid]
            if o + mid > len(body): t = body[o:] + body[:mid - (len(body) - o)]
            cut = int(len(t) * 0.8)
            user = t[:cut] + "\n" + NEEDLE + "\n" + t[cut:] + "\n\n" + Q
            n = tok(user)
            if best is None or abs(n - target) < abs(best[1] - target): best = (user, n)
            if n == target: return best
            lo, hi = (mid + 1, hi) if n < target else (lo, mid - 1)
    return best

print(f"[fullhit] 目标 N={N}，N-1={N-1} = {(N-1)//128}*128 -> 整块覆盖，{(N-1)%128} 余")
for label, off in (("A", 0), ("B", len(body) // 2)):
    prompt, n = build(off, N)
    print(f"[{label}] 实际 prompt={n} tok   (N-1)%128 = {(n-1)%128}")
    first = None
    for step in ("cold", "hit1", "hit2"):
        h0 = hits()
        wall, d, code = post({"model": MODEL,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 16, "temperature": 0.0, "stream": False})
        h1 = hits()
        if code != 200:
            print(f"   {step:5s} wall={wall:7.2f}s  ** HTTP {code} **  {str(d.get('error'))[:110]!r}")
            break
        ans = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
        if first is None: first = ans
        print(f"   {step:5s} wall={wall:7.2f}s D_hits增量={h1-h0:>8.0f} 正确={EXPECT in (ans or '')} "
              f"与冷{'相同' if ans == first else '**不同**'} ans={str(ans)[:24]!r}")
