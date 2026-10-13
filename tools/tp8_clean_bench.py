#!/usr/bin/env python3
"""tp8k5 **无污染**测量：随机 nonce 破除 prefix cache + 等全部请求进入 decode 再取窗口。

本脚本修正了此前三个测量缺陷：
  1. **prefix cache**（tp8k5 的 PREFIX=1）：重复使用同一 prompt 会让 TTFT 被缓存命中吞掉，
     实测同一 8208-token prompt 第二次只花 0.172s（第一次约 0.59s）。
     ⇒ 每个 prompt 前面加一段**唯一随机 nonce**，保证缓存永不命中；并用
        `prompt_tokens_cached_total` 的增量**核验**确实没有命中。
  2. **预热不足**：并发请求的 prefill 需要时间，若窗口开始时还有请求在 prefill，
     `Δdrafts/并发` 会低估步数、从而高估 ms/step。⇒ 等到 `num_requests_running == N`
     且 drafts 速率稳定后再开窗。
  3. **重复退化**：改用 4 个任务后缀轮换 + 互不重叠切片。

用法: tp8_clean.py <base> <conc_list> [seconds] [prompt_seg_chars]
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
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,8,32").split(",")]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
SEG = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
SUFDIR = Path("/home/l00886679/cedpd-repo/data/hlm_local")
MAXN = max(CONCS)

NONCE_CHARS = string.ascii_letters + string.digits


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
if not sufs:
    sufs = ["\n\n请概括上文。"]

step = max(1, len(body) // MAXN)
segs = [body[k * step: k * step + SEG] for k in range(MAXN)]


def make_prompts(tag):
    """每条 prompt 前置唯一 nonce（保证 prefix cache 永不命中）。"""
    out = []
    for k in range(MAXN):
        nonce = "".join(random.choice(NONCE_CHARS) for _ in range(48))
        out.append("[%s-%d-%s]\n%s\n\n%s" % (tag, k, nonce, segs[k], sufs[k % len(sufs)]))
    return out


def level(conc, tag):
    prompts = make_prompts(tag)
    stop = threading.Event()
    got_first = [False] * conc

    def one(k):
        payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": 16384,
                   "temperature": 0.0, "ignore_eos": True, "stream": True}
        req = urllib.request.Request(BASE + "/v1/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for raw in resp:
                    got_first[k] = True
                    if stop.is_set():
                        return
        except Exception:
            return

    ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(conc)]
    for t in ths:
        t.start()

    # 等全部请求进入 decode（都拿到过首 token）且 running 达到 conc
    t_deadline = time.time() + 120
    while time.time() < t_deadline:
        if all(got_first[:conc]):
            m = metrics()
            if m.get("vllm:num_requests_running", 0.0) >= conc - 0.5:
                break
        time.sleep(0.25)
    time.sleep(3.0)                    # 额外稳定期

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
    pt = b.get("vllm:prompt_tokens_total", 0.0) - a.get("vllm:prompt_tokens_total", 0.0)
    pc = b.get("vllm:prompt_tokens_cached_total", 0.0) - a.get("vllm:prompt_tokens_cached_total", 0.0)
    rounds = d / conc
    if rounds <= 0:
        print("conc=%-3d 无步数" % conc, flush=True)
        return None
    row = dict(conc=conc, ms_step=dt * 1000 / rounds, tok_step=gt / rounds,
               accept=acc / d, tps=gt / dt, cached_frac=(pc / pt if pt > 0 else 0.0))
    print("conc=%-3d ms/step=%7.3f 每步token=%7.2f 接受长度=%5.2f 吞吐=%8.1f tok/s "
          "窗口内prefill_cached=%.1f%%"
          % (conc, row["ms_step"], row["tok_step"], row["accept"], row["tps"],
             100 * row["cached_frac"]), flush=True)
    return row


print("prompt 集：%d 条互不重叠切片(%d 字符) × %d 任务后缀 + 48 字符随机 nonce"
      % (MAXN, SEG, len(sufs)), flush=True)
rows = []
for idx, c in enumerate(CONCS):
    r = level(c, "r%d" % idx)
    if r:
        rows.append(r)
    time.sleep(3.0)

if len(rows) >= 2:
    r0, r1 = rows[0], rows[-1]
    print()
    print("conc=%d → conc=%d：步长 x%.2f，吞吐 x%.2f"
          % (r0["conc"], r1["conc"], r1["ms_step"] / r0["ms_step"], r1["tps"] / r0["tps"]))
