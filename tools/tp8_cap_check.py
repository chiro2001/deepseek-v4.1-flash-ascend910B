#!/usr/bin/env python3
"""验证 MAX_SEQS 是否是硬上限：加压 conc=64，反复采样 num_requests_running。"""
import json, random, string, sys, threading, time, urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 64
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
SUFDIR = Path("/home/l00886679/cedpd-repo/data/hlm_local")
CH = string.ascii_letters + string.digits

def m():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    o = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line: continue
        p = line.rsplit(" ", 1)
        if len(p) == 2:
            nm = p[0].split("{")[0]
            try: o[nm] = o.get(nm, 0.0) + float(p[1])
            except ValueError: pass
    return o

body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文"); body = (body[i+2:] if i>=0 else body)*30
sufs = [p.read_text().strip() for p in sorted(SUFDIR.glob("*.txt"))] or ["\n\n概括。"]
mid = 2000
prompts = []
step = max(mid+2000, len(body)//CONC)
for k in range(CONC):
    nonce = "".join(random.choice(CH) for _ in range(40))
    prompts.append("[%s-%d]\n%s\n\n%s" % (nonce, k, body[(k*step) % max(1,len(body)-mid):][:mid], sufs[k%len(sufs)]))

stop = threading.Event()
def one(k):
    p = {"model":"deepseek-v41","prompt":prompts[k],"max_tokens":16384,"temperature":0.0,"ignore_eos":True,"stream":True}
    r = urllib.request.Request(BASE+"/v1/completions", data=json.dumps(p).encode(), headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            for _ in resp:
                if stop.is_set(): return
    except Exception: return

ths=[threading.Thread(target=one,args=(k,),daemon=True) for k in range(CONC)]
for t in ths: t.start()

samples=[]
for _ in range(40):
    time.sleep(1.0)
    mm = m()
    samples.append((mm.get("vllm:num_requests_running",0), mm.get("vllm:num_requests_waiting",0),
                    mm.get("vllm:kv_cache_usage_perc",0)))
stop.set(); time.sleep(1.5)

print("conc=%d  40 次采样（running, waiting, kv）:" % CONC)
for k in range(0, 40, 4):
    print("   t=%2ds  " % (k+1) + "  ".join("run=%.0f wait=%.0f kv=%.3f" % s for s in samples[k:k+2]))
run_max = max(s[0] for s in samples)
wait_max = max(s[1] for s in samples)
print()
print("running 最大 = %.0f   waiting 最大 = %.0f   kv 最大 = %.3f"
      % (run_max, wait_max, max(s[2] for s in samples)))
print("⇒ 并发硬上限 = %.0f（MAX_SEQS）；超出部分排队" % run_max)
