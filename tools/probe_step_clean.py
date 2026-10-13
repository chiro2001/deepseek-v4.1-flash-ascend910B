#!/usr/bin/env python3
"""干净的单流步长探针：max_tokens 足够大（不结束）、且校验 A 在正常区间（非重复退化）。

用法: clean_probe.py <base> <dur_s> [rounds]
"""
import json, sys, threading, time, urllib.request
BASE = sys.argv[1]; DUR = float(sys.argv[2]); ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CORPUS = "/home/l00886679/cedpd-repo/data/dihuo.txt"
import pathlib
SUFS = sorted(pathlib.Path("/home/l00886679/cedpd-repo/data/dihuo_local").glob("*.txt"))
MAXTOK = 100000   # 关键：足够大，保证窗口内请求不结束

def M():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    o = {}
    for ln in t.splitlines():
        if not ln or ln.startswith("#"): continue
        p = ln.rsplit(" ", 1)
        if len(p) != 2: continue
        try: o[p[0].split("{")[0]] = o.get(p[0].split("{")[0], 0.0) + float(p[1])
        except ValueError: pass
    return o

body = open(CORPUS, encoding="utf-8").read()

def one_round(i):
    # 每轮换切片与任务后缀，避免重复退化
    seg = body[(i * 3000) % (len(body) - 3000): (i * 3000) % (len(body) - 3000) + 3000]
    suf = SUFS[i % len(SUFS)].read_text().strip()
    prompt = "【%d-%d】\n%s\n\n%s" % (i, int(time.time()*1000) % 99999, seg, suf)
    stop = threading.Event()
    def run():
        pl = {"model":"deepseek-v41","prompt":prompt,"max_tokens":MAXTOK,
              "temperature":0.0,"ignore_eos":True,"stream":True}
        r = urllib.request.Request(BASE+"/v1/completions", data=json.dumps(pl).encode(),
                                   headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(r, timeout=1800) as resp:
                for _ in resp:
                    if stop.is_set(): return
        except Exception: return
    th = threading.Thread(target=run, daemon=True); th.start()
    time.sleep(8.0)
    a = M(); ta = time.perf_counter(); time.sleep(DUR); b = M(); dt = time.perf_counter() - ta
    stop.set(); time.sleep(2.0)
    d = b["vllm:spec_decode_num_drafts_total"] - a["vllm:spec_decode_num_drafts_total"]
    gt = b["vllm:generation_tokens_total"] - a["vllm:generation_tokens_total"]
    acc = b["vllm:spec_decode_num_accepted_tokens_total"] - a["vllm:spec_decode_num_accepted_tokens_total"]
    fin = b.get("vllm:request_success_total", 0) - a.get("vllm:request_success_total", 0)
    if d <= 0: return None
    A = acc / d
    ms = dt * 1000 / d
    ok = "✅" if (fin == 0 and 1.2 <= A <= 3.2) else "⚠️污染"
    print(f"  第{i+1}轮 ms/step={ms:7.3f}  A={A:5.3f}  tokens/step={gt/d:5.3f}  吞吐={gt/dt:6.1f}  结束={fin:.0f} {ok}", flush=True)
    return ms, A, fin

rows = []
for i in range(ROUNDS):
    r = one_round(i)
    if r: rows.append(r)
    time.sleep(3)
good = [m for m, A, f in rows if f == 0 and 1.2 <= A <= 3.2]
if good:
    import statistics as st
    print(f"\n干净样本 n={len(good)}：p50 ms/step = {st.median(good):.3f}  (min {min(good):.3f} / max {max(good):.3f})")
else:
    print("\n没有干净样本")
