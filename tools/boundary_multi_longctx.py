#!/usr/bin/env python3
"""边界测试 v2：每条 prompt 用 /tokenize 精确校准到目标 token 数（不超 max-model-len）。
用法: boundary2.py <port> <target_tokens> <n_conc>
"""
import json, sys, threading, time, urllib.request
sys.path.insert(0, "/home/l00886679/cedpd-repo/tools")
from ced_pd_acceptance import NEEDLE_Q, embed_needles

PORT = int(sys.argv[1]); TGT = int(sys.argv[2]); NC = int(sys.argv[3])
BASE = f"http://127.0.0.1:{PORT}"
CORPUS = open("/home/l00886679/cedpd-repo/data/hongloumeng.txt", encoding="utf-8").read()

def toklen(txt):
    r = urllib.request.Request(f"{BASE}/tokenize",
        data=json.dumps({"model": "deepseek-v41", "prompt": txt}).encode(),
        headers={"Content-Type": "application/json"})
    return len(json.loads(urllib.request.urlopen(r, timeout=900).read())["tokens"])

keys = list(NEEDLE_Q.keys())[:NC]
step = max(1, len(CORPUS) // (NC + 1))
prompts = []
for i, k in enumerate(keys):
    off = i * step
    body = (CORPUS[off:] + CORPUS[:off]) * 3      # 足够长
    lo, hi = 0, len(body)
    for _ in range(24):                            # 二分到 <= TGT
        mid = (lo + hi) // 2
        if toklen(body[:mid]) <= TGT: lo = mid
        else: hi = mid
    body = body[:lo]
    q, ans = NEEDLE_Q[k]
    full = embed_needles(body, [k]) + "\n\n" + q
    n = toklen(full)
    prompts.append((k, full, ans, n))
    print(f"  针{k}: 校准 {n} tok", flush=True)

res = {}
def fire(i, prompt):
    payload = {"model": "deepseek-v41", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 32, "temperature": 0.0}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        txt = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        res[i] = (time.time()-t0, txt, d.get("usage", {}).get("prompt_tokens", -1))
    except urllib.error.HTTPError as e:
        res[i] = (time.time()-t0, f"<HTTP {e.code}: {e.read()[:180].decode(errors='replace')}>", -1)
    except Exception as e:
        res[i] = (time.time()-t0, f"<{type(e).__name__}: {e}>", -1)

print(f"发起 {NC} 条并发 × {TGT} tok（合计 ≈{NC*TGT//128} 块）", flush=True)
ths = [threading.Thread(target=fire, args=(i, p)) for i, (_, p, _, _) in enumerate(prompts)]
T0 = time.time()
for t in ths: t.start()
for t in ths: t.join()
print(f"总墙钟 {time.time()-T0:.1f}s", flush=True)
ok = 0
for i, (k, _, ans, n) in enumerate(prompts):
    dt, txt, pt = res[i]
    hit = ans in txt
    others = [a for kk, (_, a) in NEEDLE_Q.items() if kk != k and a in txt]
    good = hit and not others
    ok += 1 if good else 0
    print(f"  [{'PASS' if good else 'FAIL'}] 针{k} prompt={pt} tok {dt:6.1f}s 含本针={hit} 含他针={others} 答={txt[:60]!r}")
print(f"VERDICT: {ok}/{NC} PASS")
