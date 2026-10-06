#!/usr/bin/env python3
"""tp8k5 无歧义吞吐测量：非流式请求 + usage.completion_tokens 计数。

为什么不用流式：上一版流式解析只数到 5177 token，而 /metrics 记 19283（差 3.7×）
⇒ 客户端计数不可信。这里直接用 `usage.completion_tokens`，它是服务端权威值。
用法: tp8_probe2.py <base> <conc> <max_tokens>
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512

CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
head = body[:3000]

res = [None] * CONC


def one(k):
    payload = {
        "model": "deepseek-v41",
        "prompt": head + "\n\n请用一句话概括上文。",
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            d = json.loads(resp.read())
        dt = time.perf_counter() - t0
        res[k] = (dt, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"])
    except Exception as exc:  # noqa: BLE001
        res[k] = (time.perf_counter() - t0, 0, 0)
        print("  err:", repr(exc)[:120])


ths = [threading.Thread(target=one, args=(k,)) for k in range(CONC)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.perf_counter() - t0

ok = [r for r in res if r and r[1] > 0]
if not ok:
    print("conc=%d  全部失败" % CONC)
    sys.exit(1)
tot = sum(r[1] for r in ok)
dts = [r[0] for r in ok]
print("conc=%-3d prompt=%dtok out/req=%d  墙钟 %.2fs  合计 %d token" %
      (CONC, ok[0][2], MAXTOK, wall, tot))
print("        ⇒ 聚合吞吐 %8.1f tok/s   每请求中位 %6.1f tok/s   最长 %5.2fs" %
      (tot / wall, statistics.median([r[1] / r[0] for r in ok]), max(dts)))
