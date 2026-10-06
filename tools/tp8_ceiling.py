#!/usr/bin/env python3
"""tp8k5 的可持续并发上限：逐档加压，看吞吐/步长/KV 容量在哪里拐弯。

为什么重要：KV 容量实测 2,987,509 token，而并发 32 时吞吐仍在涨（1413 tok/s）。
需要知道：(a) 吞吐在哪一档不再涨；(b) 哪一档开始出现排队/超时；(c) KV 何时打满。

每档：
  · nonce + 互不重叠切片 ⇒ 零 prefix cache 命中（并核验）
  · 稳定期后开 20 s 窗口，用 /metrics 差分算步长与吞吐
  · 同时记录 num_requests_running / waiting / kv_cache_usage

用法: tp8_ceiling.py <base> <conc_list> [window_s] [prompt_tokens]
"""
import json
import random
import string
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "32,64,96").split(",")]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
WANT = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
SUFDIR = Path("/home/l00886679/cedpd-repo/data/hlm_local")
CH = string.ascii_letters + string.digits
MAXN = max(CONCS)


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


def tok_count(prompt):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": "deepseek-v41", "prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return int(d.get("count", d.get("token_count", -1)))


body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = (body[i + 2:] if i >= 0 else body) * 30
sufs = [p.read_text().strip() for p in sorted(SUFDIR.glob("*.txt"))] or ["\n\n请概括上文。"]

# 校准到 ~WANT token
lo, hi = 200, min(len(body) // 4, 400000)
mid = hi
for _ in range(24):
    m = (lo + hi) // 2
    n = tok_count(body[:m] + "\n\n" + sufs[0])
    if abs(n - WANT) <= max(8, WANT // 100):
        mid = m
        break
    if n < WANT:
        lo = m + 1
    else:
        hi = m - 1
    mid = m


def make_prompts(tag):
    step = max(mid + 2000, len(body) // MAXN)
    out = []
    for k in range(MAXN):
        nonce = "".join(random.choice(CH) for _ in range(40))
        seg = body[(k * step) % max(1, len(body) - mid):][:mid]
        out.append("[%s-%d-%s]\n%s\n\n%s" % (tag, k, nonce, seg, sufs[k % len(sufs)]))
    return out


def level(conc, prompts):
    stop = threading.Event()
    done = [0]

    def one(k):
        payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": 16384,
                   "temperature": 0.0, "ignore_eos": True, "stream": True}
        req = urllib.request.Request(BASE + "/v1/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for _raw in resp:
                    done[0] += 1
                    if stop.is_set():
                        return
        except Exception:
            return

    ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(conc)]
    for t in ths:
        t.start()
    # 等所有请求都进入 decode
    t_end = time.time() + 240
    while time.time() < t_end:
        m = metrics()
        if m.get("vllm:num_requests_running", 0.0) >= conc - 0.5:
            break
        time.sleep(0.5)
    time.sleep(3.0)

    a = metrics()
    ta = time.perf_counter()
    time.sleep(DUR)
    b = metrics()
    dt = time.perf_counter() - ta
    mid_m = metrics()
    stop.set()
    time.sleep(2.0)

    d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
    rounds = d / conc
    run_m = mid_m.get("vllm:num_requests_running", 0.0)
    wait_m = mid_m.get("vllm:num_requests_waiting", 0.0)
    kv = mid_m.get("vllm:kv_cache_usage_perc", 0.0)
    if rounds <= 0:
        print("conc=%-4d 无步数（可能全部失败）" % conc, flush=True)
        return None
    row = dict(conc=conc, ms_step=dt * 1000 / rounds, tps=gt / dt,
               accept=acc / d, tb=gt / rounds)
    print("conc=%-4d ms/step=%8.3f  每步token=%8.2f  接受=%4.2f  **吞吐=%9.1f tok/s**  "
          "running=%.0f waiting=%.0f kv=%.3f"
          % (conc, row["ms_step"], row["tb"], row["accept"], row["tps"], run_m, wait_m, kv),
          flush=True)
    return row


print("prompt 目标 %d token（实际校准）  并发档=%s  窗口=%.0fs" % (WANT, CONCS, DUR), flush=True)
rows = []
for idx, c in enumerate(CONCS):
    r = level(c, make_prompts("c%d" % idx))
    if r:
        rows.append(r)
    # 等 KV 释放 + 服务空闲
    for _ in range(60):
        m = metrics()
        if m.get("vllm:num_requests_running", 0.0) <= 0.5 and m.get("vllm:num_requests_waiting", 0.0) <= 0.5:
            break
        time.sleep(1.0)
    time.sleep(3.0)

if len(rows) >= 2:
    print()
    best = max(rows, key=lambda r: r["tps"])
    print("最高吞吐出现在 conc=%d：%.1f tok/s（%.3f ms/step）"
          % (best["conc"], best["tps"], best["ms_step"]))
    for r in rows:
        print("  conc=%-4d %9.1f tok/s   %.3f ms/step" % (r["conc"], r["tps"], r["ms_step"]))
