#!/usr/bin/env python3
"""接受长度是否真的随并发上升？—— 用**不同 prompt 切片**排除"相同 prompt"假象。

上半：所有请求用同一条 prompt（复现 tp8_probe.py 的口径）
下半：每个请求用不同的正文切片（互不重叠 ⇒ 无 prefix cache 复用）

用法: tp8_probe3.py <base> <conc> [seconds]
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"


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
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
SEG = 3000
same = body[:SEG]
diff = [body[(k * SEG) % (len(body) - SEG):][:SEG] for k in range(CONC)]

sup = {}


def _post(payload):
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return req


def run(prompts, tag):
    stop = threading.Event()

    def one(k):
        try:
            with urllib.request.urlopen(_post({
                "model": "deepseek-v41", "prompt": prompts[k] + "\n\n请用一句话概括上文。",
                "max_tokens": 8192, "temperature": 0.0, "ignore_eos": True, "stream": True,
            }), timeout=900) as resp:
                for _raw in resp:
                    if stop.is_set():
                        return
        except Exception:
            return

    ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(CONC)]
    for t in ths:
        t.start()
    time.sleep(5.0)
    a = metrics()
    ta = time.perf_counter()
    time.sleep(DUR)
    b = metrics()
    dt = time.perf_counter() - ta
    stop.set()
    time.sleep(1.0)
    d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
    rounds = d / CONC
    if rounds <= 0:
        print("  [%s] 无步数" % tag)
        return
    print("  [%-10s] conc=%-3d ms/step=%6.3f  每步token=%6.2f  接受长度/请求=%.2f  "
          "吞吐=%7.1f tok/s"
          % (tag, CONC, dt * 1000 / rounds, gt / rounds, acc / d, gt / dt))


print("=== conc=%d，窗口 %.0fs ===" % (CONC, DUR))
run([same] * CONC, "same-prompt")
run(diff, "distinct")
