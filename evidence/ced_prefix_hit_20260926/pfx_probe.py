#!/usr/bin/env python3
"""前缀缓存探针：用**服务端 metrics**（而不是响应里的 cached_tokens）判定命中。

四步：
  A 冷 prefill（新 prompt）        → 期望 hits 不增，wall 大
  B 同 prompt 再来一次              → 期望 hits +N，wall 小
  C 同 prompt 第三次               → 同样命中
  D 换一个新的同长度 prompt（冷）   → 期望 hits 不增，wall 恢复到冷值

判据：`vllm:prefix_cache_hits_total` 与
`vllm:prompt_tokens_by_source_total{source="local_cache_hit"}` 的**增量**。
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18992"
TOK  = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18990"
MODEL = "deepseek-v41-ced-pd"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 144000

def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return time.monotonic() - t0, data

def tok_count(text):
    req = urllib.request.Request(f"{TOK}/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(json.loads(r.read().decode())["count"])

def metrics(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"): continue
        for key in ("prefix_cache_hits_total", "prefix_cache_queries_total",
                    'prompt_tokens_by_source_total{engine="0",model_name="deepseek-v41-ced-pd",source="local_cache_hit"}'):
            if line.startswith("vllm:" + key.split("{")[0]) and key in line:
                try: out[key] = float(line.rsplit(" ", 1)[1])
                except Exception: pass
    return out

corpus = open("data/hongloumeng.txt", encoding="utf-8", errors="replace").read()
MARK = "========正文========"
body_start = corpus.index(MARK) + len(MARK) if MARK in corpus else 0
body = corpus[body_start:]

def make(offset, target):
    lo, hi = 1, target * 4
    best = (0, -1)
    while lo <= hi:
        mid = (lo + hi) // 2
        t = body[offset:offset + mid]
        if offset + mid > len(body): t = body[offset:] + body[:mid - (len(body) - offset)]
        n = tok_count(t + "\n\n运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。")
        if best[1] < 0 or abs(n - target) < abs(best[1] - target): best = (mid, n)
        if n == target: break
        lo, hi = (mid + 1, hi) if n < target else (lo, mid - 1)
    t = body[offset:offset + best[0]]
    if offset + best[0] > len(body): t = body[offset:] + body[:best[0] - (len(body) - offset)]
    return t + "\n\n运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。", best[1]

p_a, n_a = make(0, N)
p_b, n_b = make(len(body) // 2, N)
print(f"[probe] prompt A={n_a} tok, prompt B(不同位置)={n_b} tok  (目标 {N})")

def ask(label, prompt):
    m0p, m0d = metrics(18990), metrics(18991)
    wall, data = post(f"{BASE}/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16, "temperature": 0.0, "stream": False})
    m1p, m1d = metrics(18990), metrics(18991)
    ans = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    u = data.get("usage") or {}
    def d(m0, m1, k): return m1.get(k, 0) - m0.get(k, 0)
    hp = d(m0p, m1p, "prefix_cache_hits_total"); hd = d(m0d, m1d, "prefix_cache_hits_total")
    lp = d(m0p, m1p, 'prompt_tokens_by_source_total{engine="0",model_name="deepseek-v41-ced-pd",source="local_cache_hit"}')
    print(f"  {label:22s} wall={wall:7.2f}s  cached_tokens(响应)={((u.get('prompt_tokens_details') or {}).get('cached_tokens'))!s:>5}"
          f"  P_hits增量={hp:>8.0f}  D_hits增量={hd:>8.0f}  P_local_hit={lp:>8.0f}  ans={str(ans)[:14]!r}")
    return wall

print("[probe] A 冷 prefill")
ask("A cold", p_a)
print("[probe] B 同一 prompt 第二次")
ask("B same#2", p_a)
print("[probe] C 同一 prompt 第三次")
ask("C same#3", p_a)
print("[probe] D 新 prompt（不同位置，同长度）")
ask("D cold-new", p_b)
