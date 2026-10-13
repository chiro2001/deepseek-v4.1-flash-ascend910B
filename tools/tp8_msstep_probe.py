#!/usr/bin/env python3
"""对 tp8k5 施短时定点负载，并采样 /metrics 得到真实 ms/step 与每步产出 token。

用法: tp8_probe.py <base> <conc> [seconds]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
MARK = "正文"


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(" ", 1)
        if len(p) != 2:
            continue
        nm = p[0].split("{")[0]
        try:
            out[nm] = out.get(nm, 0.0) + float(p[1])
        except ValueError:
            pass
    return out


body = open(CORPUS, encoding="utf-8").read()
i = body.find(MARK)
body = body[i + len(MARK):] if i >= 0 else body
# 目标 ~2048 token 的 prompt（中文约 1.5 字符/token ⇒ 取 3000 字符足够）
head = body[:3000]
stop = threading.Event()
counts = [0] * CONC


def one(k):
    payload = {
        "model": "deepseek-v41",
        "prompt": head + "\n\n请用一句话概括上文。",
        "max_tokens": 4096,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if stop.is_set():
                    return
                line = raw.decode("utf-8", "ignore").strip()
                if line.startswith("data: ") and "[DONE]" not in line:
                    counts[k] += 1
    except Exception:
        return


ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(CONC)]
t0 = time.perf_counter()
for t in ths:
    t.start()
time.sleep(6.0)                      # 预热：等 prefill 结束、进入稳态 decode
a = metrics()
ta = time.perf_counter()
time.sleep(DUR)
b = metrics()
tb = time.perf_counter()
stop.set()
dt = tb - ta

keys = [
    ("vllm:spec_decode_num_drafts_total", "drafts(步数)"),
    ("vllm:spec_decode_num_draft_tokens_total", "draft_tokens"),
    ("vllm:spec_decode_num_accepted_tokens_total", "accepted"),
    ("vllm:generation_tokens_total", "gen_tokens"),
]
print("conc=%d  窗口 %.1fs  客户端收到 %d tokens" % (CONC, dt, sum(counts)))
for k, lbl in keys:
    d = b.get(k, 0.0) - a.get(k, 0.0)
    print("  %-16s Δ=%-12.1f  %10.2f/s" % (lbl, d, d / dt))

d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
if d > 0:
    print()
    print("  ⇒ ms/step      = %.3f" % (dt * 1000.0 / d))
    print("  ⇒ 步/秒        = %.2f" % (d / dt))
    print("  ⇒ 每步产出token= %.2f" % (gt / d))
    print("  ⇒ 接受长度     = %.2f" % (acc / d))
    print("  ⇒ 生成吞吐     = %.1f tok/s" % (gt / dt))
