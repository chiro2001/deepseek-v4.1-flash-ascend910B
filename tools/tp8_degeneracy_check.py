#!/usr/bin/env python3
"""检查生成文本是否退化为重复 —— 这决定"接受长度随并发上升"是不是假象。

用法: tp8_textcheck.py <base> <conc> [max_tokens] [distinct]
"""
import collections
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 256
DISTINCT = (len(sys.argv) > 4 and sys.argv[4] == "distinct")
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
SEG = 3000
prompts = [body[(k * SEG) % (len(body) - SEG):][:SEG] for k in range(CONC)] if DISTINCT else [body[:SEG]] * CONC

res = [None] * CONC


def one(k):
    payload = {"model": "deepseek-v41", "prompt": prompts[k] + "\n\n请用一句话概括上文。",
               "max_tokens": MT, "temperature": 0.0, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            d = json.loads(resp.read())
        res[k] = (time.perf_counter() - t0, d["choices"][0]["text"], d["usage"]["completion_tokens"])
    except Exception as exc:  # noqa: BLE001
        res[k] = (time.perf_counter() - t0, "ERR %r" % (exc,), 0)


ths = [threading.Thread(target=one, args=(k,)) for k in range(CONC)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.perf_counter() - t0

ok = [r for r in res if r and r[2] > 0]
tot = sum(r[2] for r in ok)
print("conc=%d distinct=%s MT=%d  墙钟 %.2fs  合计 %d tok  聚合 %.1f tok/s"
      % (CONC, DISTINCT, MT, wall, tot, tot / wall))


def rep_ratio(s):
    """粗糙的重复度：最常见的 20 字符片段出现次数 / 总片段数"""
    if len(s) < 40:
        return 0.0
    grams = [s[i:i + 20] for i in range(0, len(s) - 20)]
    c = collections.Counter(grams)
    return c.most_common(1)[0][1] / max(1, len(grams))


print()
for k, r in enumerate(res):
    if not r or r[2] == 0:
        print("  req%-2d FAILED %s" % (k, (r[1] if r else "")[:80]))
        continue
    txt = r[1]
    print("  req%-2d %5.2fs  %4d tok  rep=%.3f  %r" % (k, r[0], r[2], rep_ratio(txt), txt[-70:]))
