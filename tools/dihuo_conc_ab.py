#!/usr/bin/env python3
"""地火口径的受控并发对照：同一语料下 conc=1 vs conc=8 的 ms/step 与接受长度。

目的：回答"单流 >90 tok/s、8 并发损失 <20%"是否成立。
关键：① 同一语料（地火正文切片 + dihuo_local 四个任务后缀轮换）；
     ② 每请求独立 nonce 破前缀缓存；
     ③ 用 /metrics 稳态窗口求步数（Δdrafts/并发）与接受长度（Δaccepted/Δdrafts）。

用法: dihuo_conc_ab.py <base> <conc_list> [sec]
"""
import json, sys, threading, time, urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,8").split(",")]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
CORPUS = "/home/l00886679/cedpd-repo/data/dihuo.txt"
SUFDIR = Path("/home/l00886679/cedpd-repo/data/dihuo_local")
SEG = 3000
MAXN = max(CONCS)


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=60).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(" ", 1)
        if len(p) != 2:
            continue
        try:
            out[p[0].split("{")[0]] = out.get(p[0].split("{")[0], 0.0) + float(p[1])
        except ValueError:
            pass
    return out


body = open(CORPUS, encoding="utf-8").read()
sufs = [p.read_text().strip() for p in sorted(SUFDIR.glob("*.txt"))]
# 互不重叠的正文切片：按 MAXN 均分，取最大并发所需的段数
step = len(body) // MAXN
prompts = []
for k in range(MAXN):
    seg = body[k * step: k * step + SEG]
    nonce = "【样本编号 %d-%d】\n" % (k, int(time.time() * 1000) % 100000)
    prompts.append(nonce + seg + "\n\n" + sufs[k % len(sufs)])


def level(conc):
    stop = threading.Event()

    def one(k):
        payload = {"model": "deepseek-v41", "prompt": prompts[k], "max_tokens": 8192,
                   "temperature": 0.0, "ignore_eos": True, "stream": True}
        req = urllib.request.Request(BASE + "/v1/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for _ in resp:
                    if stop.is_set():
                        return
        except Exception:
            return

    ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(conc)]
    for t in ths:
        t.start()
    time.sleep(8.0)
    a = metrics(); ta = time.perf_counter()
    time.sleep(DUR)
    b = metrics(); dt = time.perf_counter() - ta
    stop.set(); time.sleep(2.0)

    d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    gt = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
    rounds = d / conc
    if rounds <= 0:
        print("conc=%-3d 无步数" % conc, flush=True); return None
    ms_step = dt * 1000 / rounds
    tok_step = gt / rounds            # 全批每步 token（= conc × 每路每步）
    accept = acc / d                  # 接受长度 A
    a_out = tok_step / conc           # 每路每步产出 = 1 + A
    tps = gt / dt                     # 聚合
    per = tps / conc                  # 每路
    row = dict(conc=conc, ms_step=ms_step, tok_step=tok_step, accept=accept,
               a_out=a_out, agg=tps, per=per)
    print("conc=%-3d ms/step=%7.3f  每路每步token=%5.2f (A=%4.2f)  每路=%7.2f tok/s  聚合=%8.1f tok/s"
          % (conc, ms_step, a_out, accept, per, tps), flush=True)
    return row


print("地火语料：%d 条互不重叠切片 × %d 个任务后缀（dihuo_local）" % (MAXN, len(sufs)), flush=True)
rows = []
for c in CONCS:
    r = level(c)
    if r: rows.append(r)
    time.sleep(4.0)

if len(rows) >= 2:
    r0, r1 = rows[0], rows[-1]
    print()
    print("=== 对照（conc=%d → conc=%d）===" % (r0["conc"], r1["conc"]), flush=True)
    print("步长膨胀      x%.3f  (%6.2f → %6.2f ms)" % (r1["ms_step"]/r0["ms_step"], r0["ms_step"], r1["ms_step"]))
    print("A_out 变化    x%.3f  (%5.2f → %5.2f)" % (r1["a_out"]/r0["a_out"], r0["a_out"], r1["a_out"]))
    print("每路速度变化  x%.3f  (%6.1f → %6.1f tok/s)  ⇒ 损失 %.1f%%"
          % (r1["per"]/r0["per"], r0["per"], r1["per"], 100*(1-r1["per"]/r0["per"])))
    print("聚合吞吐变化  x%.3f  (%6.1f → %6.1f tok/s)" % (r1["agg"]/r0["agg"], r0["agg"], r1["agg"]))
    print("A 归一化步长代价：若 A_out 不变，每路损失 = 1 - 1/%.3f = %.1f%%"
          % (r1["ms_step"]/r0["ms_step"], 100*(1-1/(r1["ms_step"]/r0["ms_step"]))))
