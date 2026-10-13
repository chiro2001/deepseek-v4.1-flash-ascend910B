#!/usr/bin/env python3
"""tp8k5 吞吐/步长曲线（非退化 prompt 集，稳态 /metrics 窗口）。

为什么重建：上一版探针用同一条 prompt，模型会退化成逐字重复
（实测 rep=0.092、尾部反复出现同一句）⇒ 投机解码接受长度被虚高到 4.97/5
⇒ 吞吐被严重高估。这里改用 data/hlm_local/ 的 4 个任务后缀 + 互不重叠的正文切片，
与 tools/bench_concurrency.py 的口径一致，但支持任意并发数。

用法: tp8_curve.py <base> <conc_list> [seconds_per_level]
"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,8,32").split(",")]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
SUFDIR = Path("/home/l00886679/cedpd-repo/data/hlm_local")
SEG = 3000
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


body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
sufs = [p.read_text().strip() for p in sorted(SUFDIR.glob("*.txt"))]

step = len(body) // MAXN
prompts = []
for k in range(MAXN):
    seg = body[k * step: k * step + SEG]
    prompts.append(seg + "\n\n" + sufs[k % len(sufs)])


def level(conc):
    stop = threading.Event()

    def one(k):
        payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": 8192,
                   "temperature": 0.0, "ignore_eos": True, "stream": True}
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                for _raw in resp:
                    if stop.is_set():
                        return
        except Exception:
            return

    ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(conc)]
    for t in ths:
        t.start()
    time.sleep(5.0)
    a = metrics()
    ta = time.perf_counter()
    time.sleep(DUR)
    b = metrics()
    dt = time.perf_counter() - ta
    stop.set()
    time.sleep(1.5)
    d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
    rounds = d / conc
    if rounds <= 0:
        print("conc=%-3d 无步数" % conc, flush=True)
        return None
    row = dict(conc=conc, ms_step=dt * 1000 / rounds, tok_step=gt / rounds,
               accept=acc / d, tps=gt / dt)
    print("conc=%-3d ms/step=%7.3f  每步token=%7.2f  接受长度=%5.2f  吞吐=%8.1f tok/s"
          % (conc, row["ms_step"], row["tok_step"], row["accept"], row["tps"]), flush=True)
    return row


print("prompt 集：%d 条互不重叠切片 × %d 个任务后缀（非退化）" % (MAXN, len(sufs)), flush=True)
rows = []
for c in CONCS:
    r = level(c)
    if r:
        rows.append(r)
    time.sleep(3.0)

if len(rows) >= 2:
    r0, r1 = rows[0], rows[-1]
    print()
    print("从 conc=%d 到 conc=%d：步长 x%.2f，吞吐 x%.2f"
          % (r0["conc"], r1["conc"], r1["ms_step"] / r0["ms_step"], r1["tps"] / r0["tps"]))
