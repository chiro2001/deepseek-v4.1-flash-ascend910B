#!/usr/bin/env python3
"""门③：`[bneck] hp`（引擎侧 ms/step）与 /metrics 派生的 ms/step **同时**测量并比对。

为什么要同时：bneck 的 hp 是**步间隔均值**，服务空闲时会被拉爆（实测 idle 窗口 hp=39805 ms）。
必须在**稳态 decode**窗口内取，并用第二个独立方法交叉验证。

用法: gate3_hp.py <base> <conc> [seconds]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 32
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(" ", 1)
        if len(p) == 2:
            nm = p[0].split("{")[0]
            try:
                out[nm] = out.get(nm, 0.0) + float(p[1])
            except ValueError:
                pass
    return out


prompts = [body[k * 60000: k * 60000 + 4000] + "\n\n请概括上文。" for k in range(CONC)]
stop = threading.Event()


def one(k):
    payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": 4096,
               "temperature": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for _r in resp:
                if stop.is_set():
                    return
    except Exception:
        return


ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(CONC)]
for t in ths:
    t.start()
# 等 running 达到 conc
t_end = time.time() + 180
while time.time() < t_end:
    if metrics().get("vllm:num_requests_running", 0.0) >= CONC - 0.5:
        break
    time.sleep(0.5)
time.sleep(5.0)

a = metrics()
ta = time.perf_counter()
time.sleep(DUR)
b = metrics()
dt = time.perf_counter() - ta
stop.set()
time.sleep(2.0)

d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
rounds = d / CONC
print("conc=%d  窗口 %.1fs" % (CONC, dt))
if rounds > 0:
    print("  [方法A /metrics] ms/step = %.3f   每步 token=%.2f   吞吐=%.1f tok/s"
          % (dt * 1000 / rounds, gt / rounds, gt / dt))
print("  注意窗口起止时间（用于从 serve.log 里切 [bneck] 行）: %.0f → %.0f (epoch s)"
      % (time.time() - dt, time.time()))
