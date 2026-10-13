#!/usr/bin/env python3
"""单流长请求的步间隔分布：中位 vs 均值 vs 尾部，判断有无"慢步"长尾。
用法: step_tail.py <port> [out_tokens]
"""
import json, sys, time, statistics, urllib.request, random
PORT = int(sys.argv[1]); OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
base = f"http://127.0.0.1:{PORT}"
rnd = random.Random(4242)
body = "\n".join(f"记录{rnd.randint(10**8,10**9)} 值{rnd.random():.6f}" for _ in range(120))
body += "\n\n请用一句话概括上文。"
payload = {"model": "deepseek-v41", "prompt": body, "max_tokens": OUT,
           "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
           "ignore_eos": True}
def metrics_A():
    try:
        for line in urllib.request.urlopen(f"{base}/metrics", timeout=30).read().decode().split("\n"):
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                return float(line.split()[-1])
    except Exception:
        return None
    return None

a0 = metrics_A()
ev = []
req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=900) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        b = line[5:].strip()
        if b == "[DONE]": break
        now = time.perf_counter()
        try: o = json.loads(b)
        except Exception: continue
        if o.get("usage"): continue
        for ch in o.get("choices") or []:
            if ch.get("text"): ev.append(now - t0)
a1 = metrics_A()
g = [ev[i+1]-ev[i] for i in range(len(ev)-1)]
g = [x for x in g if x > 0]
gs = sorted(g)
n = len(gs)
print(f"chunks={len(ev)}  span={ev[-1]:.2f}s")
print(f"步间隔: 中位={statistics.median(g)*1000:.2f}ms  均值={statistics.mean(g)*1000:.2f}ms  "
      f"p90={gs[9*n//10]*1000:.2f}  p99={gs[99*n//100]*1000:.2f}  max={gs[-1]*1000:.2f}")
print(f"  p10={gs[n//10]*1000:.2f}  最小={gs[0]*1000:.2f}")
tail = gs[9*n//10:]
print(f"  最慢 10% 的步: 均值 {statistics.mean(tail)*1000:.2f}ms ⇒ 若把它压到中位，"
      f"整轮可省 {sum(tail)-statistics.median(g)*len(tail):.2f}s")
if a0 is not None and a1 is not None:
    acc = a1 - a0
    print(f"\n/metics 接受 token 增量 = {acc:.0f}，chunks={len(ev)} ⇒ A≈{acc/len(ev):.3f}")
    print(f"  用中位步长算 tok/s = {acc/len(ev)/statistics.median(g):.1f}；"
          f"用均值算 = {acc/len(ev)/statistics.mean(g):.1f}；实测 = {len(ev)/ev[-1]:.1f}")
