#!/usr/bin/env python3
"""用 /metrics 的 draft 计数器**无侵入**测 tp8k5 的真实 ms/step。

原理：spec_decode_num_drafts_total 每步 +1/请求（每步起草一次）⇒
      Δ(drafts)/Δt = 步/秒 ⇒ ms/step = Δt/Δdrafts*1000。
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{")[0]
        try:
            out[name] = out.get(name, 0.0) + float(parts[1])
        except ValueError:
            pass
    return out


a = metrics()
t0 = time.perf_counter()
time.sleep(DUR)
b = metrics()
dt = time.perf_counter() - t0

for key, label in (
    ("vllm:spec_decode_num_drafts_total", "drafts"),
    ("vllm:spec_decode_num_draft_tokens_total", "draft_tokens"),
    ("vllm:spec_decode_num_accepted_tokens_total", "accepted"),
    ("vllm:generation_tokens_total", "gen_tokens"),
    ("vllm:request_success_total", "ok_reqs"),
):
    if key in a and key in b:
        d = b[key] - a[key]
        print("%-16s Δ=%-12.1f  %.2f/s" % (label, d, d / dt))
    else:
        print("%-16s (无此指标)" % label)

d = b.get("vllm:spec_decode_num_drafts_total", 0) - a.get("vllm:spec_decode_num_drafts_total", 0)
if d > 0:
    print()
    print("⇒ 步/秒 = %.2f   ms/step = %.3f" % (d / dt, dt * 1000.0 / d))
gt = b.get("vllm:generation_tokens_total", 0) - a.get("vllm:generation_tokens_total", 0)
if gt and d:
    print("⇒ 每步产出 token = %.2f   生成吞吐 = %.1f tok/s" % (gt / d, gt / dt))
