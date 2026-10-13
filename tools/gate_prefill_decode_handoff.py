#!/usr/bin/env python3
"""定位分叉进入点：prefill 末位 logits vs 首生成 token。

原理：`prompt_logprobs` 给出**prefill 每个位置**的 logprob；`logprobs.token_logprobs[0]`
给出**首生成 token** 的 logprob —— 二者来自**同一份最后位置的 logits**。
在同一个请求里同时取两者、重复多轮比较：

  · `prompt_logprobs` 稳定、`logprobs[0]` 波动 ⇒ 分叉发生在
    **prefill→decode 的交接/采样/verify 路径**，不在主干前向；
  · 两者同时波动 ⇒ 主干（末位 logits）本身就不是确定的。

用法: gate_handoff.py <base> [rounds] [max_tokens]
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
CH = string.ascii_letters + string.digits

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
body = body[i + 2:] if i >= 0 else body
nonce = "".join(random.choice(CH) for _ in range(24))
prompt = "[%s]\n%s" % (nonce, body[:800])      # 约 500~600 token，窗口 128 已被填满


def ask():
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True,
               "logprobs": 1, "prompt_logprobs": 1}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.perf_counter() - t0
    ch = d["choices"][0]
    plp = ch.get("prompt_logprobs") or []
    lp = ch.get("logprobs") or {}

    def top1(ent):
        if not isinstance(ent, dict) or not ent:
            return None
        best = None
        for tok, info in ent.items():
            if isinstance(info, dict) and info.get("logprob") is not None:
                if best is None or info["logprob"] > best[1]:
                    best = (tok, float(info["logprob"]))
        return best

    # 取最后 3 个非空 prompt 位置（末位最接近交接点）
    tail = []
    for ent in reversed(plp):
        t = top1(ent)
        if t is not None:
            tail.append(t)
        if len(tail) == 3:
            break
    tail.reverse()
    return dict(tail=tail,
                gen_lp=[float(x) for x in (lp.get("token_logprobs") or [])],
                gen_tk=list(lp.get("tokens") or []),
                dt=dt)


print("prompt ≈ %d 字符，%d 轮，每轮 max_tokens=%d（同取 prompt_logprobs + logprobs）"
      % (len(prompt), ROUNDS, MT), flush=True)
for _ in range(1):
    ask()                                       # 预热

runs = [ask() for _ in range(ROUNDS)]
for k, r in enumerate(runs):
    print("  r%02d  %.2fs  末位3=(%s)  首生成(%.6f, %r)"
          % (k, r["dt"],
             ", ".join("%s:%.6f" % t for t in r["tail"]),
             r["gen_lp"][0] if r["gen_lp"] else float("nan"),
             (r["gen_tk"][0] if r["gen_tk"] else ""))[:150], flush=True)

print()


def spread(vals):
    vs = [v for v in vals if v is not None]
    return (max(vs) - min(vs)) if vs else 0.0


print("=== 跨轮波动（r01..rN）===")
print("%-34s %14s" % ("量", "max-min"))
for j, name in enumerate(("末位-2", "末位-1", "末位(交接点)")):
    print("%-34s %14.3e" % ("prompt_logprobs " + name,
                            spread([r["tail"][j][1] if len(r["tail"]) > j else None for r in runs])))
for j in range(min(MT, 3)):
    print("%-34s %14.3e" % ("logprobs[%d]（生成）" % j,
                            spread([r["gen_lp"][j] if len(r["gen_lp"]) > j else None for r in runs])))
print()
print("末位 token id 是否跨轮相同:", len({r["tail"][-1][0] for r in runs if r["tail"]}) == 1)
print("首生成 token id 是否跨轮相同:", len({r["gen_tk"][0] for r in runs if r["gen_tk"]}) == 1)
