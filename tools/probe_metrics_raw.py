#!/usr/bin/env python3
"""打印原始计数器增量，诊断 ms/step 两个口径的分歧。"""
import json, sys, threading, time, urllib.request
BASE = sys.argv[1]; CONC = int(sys.argv[2]); DUR = float(sys.argv[3])
CORPUS = "/home/l00886679/cedpd-repo/data/dihuo.txt"
SUF = "/home/l00886679/cedpd-repo/data/dihuo_local/continue.txt"
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
suf = open(SUF, encoding="utf-8").read().strip()
seg = body[:3000]
prompts = ["【%d-%d】\n%s\n\n%s" % (k, int(time.time()*1000)%99999, seg, suf) for k in range(CONC)]
stop = threading.Event()
def one(k):
    pl = {"model":"deepseek-v41","prompt":prompts[k],"max_tokens":8192,
          "temperature":0.0,"ignore_eos":True,"stream":True}
    r = urllib.request.Request(BASE+"/v1/completions", data=json.dumps(pl).encode(),
                               headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            for _ in resp:
                if stop.is_set(): return
    except Exception: return
ths=[threading.Thread(target=one,args=(k,),daemon=True) for k in range(CONC)]
for t in ths: t.start()
time.sleep(6.0)
a=M(); ta=time.perf_counter(); time.sleep(DUR); b=M(); dt=time.perf_counter()-ta
stop.set()
keys=["vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_accepted_tokens_total",
      "vllm:generation_tokens_total","vllm:prompt_tokens_total",
      "vllm:num_requests_running","vllm:request_success_total"]
print(f"窗口 dt={dt:.2f}s conc={CONC}")
for k in keys:
    if k in a or k in b:
        print(f"  Δ{k.split(':')[1]:<44} {b.get(k,0)-a.get(k,0):>12.1f}")
d=b["vllm:spec_decode_num_drafts_total"]-a["vllm:spec_decode_num_drafts_total"]
gt=b["vllm:generation_tokens_total"]-a["vllm:generation_tokens_total"]
acc=b["vllm:spec_decode_num_accepted_tokens_total"]-a["vllm:spec_decode_num_accepted_tokens_total"]
rounds=d/CONC
if rounds>0:
    print(f"\n  rounds(=Δdrafts/conc) = {rounds:.1f}  ⇒ ms/step = {dt*1000/rounds:.3f}")
    print(f"  tokens/round = {gt/rounds:.3f}   A = {acc/d:.3f}   吞吐 = {gt/dt:.1f} tok/s")
