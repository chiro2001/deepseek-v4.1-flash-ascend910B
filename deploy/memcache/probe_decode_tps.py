#!/usr/bin/env python3
"""单发 decode 吞吐探针（用 /metrics 差分，不依赖客户端计时）。

为什么不用客户端 wall time：并发/排队会把 wall 拉长；也不要用 [bneck] hp：
hp 是相邻两次 mark_step 的墙钟差，空闲间隙会被算进去（空闲时能读到 2000ms）。

用法: python3 probe_decode_tps.py [base_url] [n_tokens]
输出: 单发 decode 秒数、tok/s、接受长度 A、ms/step（由 drafts 计数反推）
"""
import json
import sys
import time
import urllib.request
K = ("request_decode_time_seconds_sum", "request_decode_time_seconds_count", "spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total")

def snap(base):
    out = dict.fromkeys(K, 0.0)
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if not line or line.startswith("#"):
                continue
            nm = line.split("{")[0].split(" ")[0]
            for k in K:
                if nm.endswith(k):
                    out[k] = float(line.rsplit(" ", 1)[1])
    return out

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    body = json.dumps({"model": "deepseek-v41", "prompt": "Reply with one long coherent paragraph about rivers.",
                       "max_tokens": ntok, "temperature": 0.0, "ignore_eos": True}).encode()
    a = snap(base)
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()
    wall = time.time() - t0
    b = snap(base)
    dsum = b["request_decode_time_seconds_sum"] - a["request_decode_time_seconds_sum"]
    dcnt = b["request_decode_time_seconds_count"] - a["request_decode_time_seconds_count"]
    ddraft = b["spec_decode_num_drafts_total"] - a["spec_decode_num_drafts_total"]
    dacc = b["spec_decode_num_accepted_tokens_total"] - a["spec_decode_num_accepted_tokens_total"]
    if dcnt != 1:
        print("WARN: 窗口内完成了 %d 个请求（有并发/排队），数值仅供参考" % dcnt)
    per_req = 0.0
    if dsum > 0 and dcnt >= 1:
        per_req = dsum / max(dcnt, 1)
        print("decode_s=%.3f tok_per_s=%.1f wall_s=%.1f" % (per_req, ntok / per_req, wall))
    if ddraft > 0 and per_req > 0:
        print("A=%.2f ms_per_step=%.1f accepted=%d drafts=%d"
              % (1.0 + dacc / ddraft, 1000.0 * per_req / ddraft, dacc, ddraft))

main()
