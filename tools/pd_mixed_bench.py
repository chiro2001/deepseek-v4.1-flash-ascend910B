#!/usr/bin/env python3
"""PD 混布：真 32K prefill 对并发 decode 的伤害（每个 run 用**互不重叠的语料窗口**）。
用法: dup2.py <port> <conc> <pf_tokens> <off1,off2,...>   # off 单位=千字符，间距>=300
"""
import concurrent.futures as cf, json, sys, threading, time, urllib.request
PORT=int(sys.argv[1]); CONC=int(sys.argv[2]); PF=int(sys.argv[3]); OFFS=[int(x)*1000 for x in sys.argv[4].split(",")]
URL="http://127.0.0.1:%d/v1/chat/completions"%PORT; TOK="http://127.0.0.1:%d/tokenize"%PORT
_c=open("/home/l00886679/cedpd-repo/data/hongloumeng.txt",encoding="utf-8").read()
def count(p):
    r=urllib.request.Request(TOK,data=json.dumps({"model":"deepseek-v41","prompt":p}).encode(),headers={"Content-Type":"application/json"})
    return int(json.loads(urllib.request.urlopen(r,timeout=600).read())["count"])
def calib(s,t):
    lo,hi,best=1,len(s),s[:1]
    while lo<hi:
        mid=(lo+hi)//2
        if count(s[:mid])<t: lo,best=mid+1,s[:mid]
        else: hi=mid
    return best
def post(prompt,mt,ig=True):
    body=json.dumps({"model":"deepseek-v41","messages":[{"role":"user","content":prompt+"\n\n请续写。"}],
                     "max_tokens":mt,"temperature":0.0,"ignore_eos":ig}).encode()
    r=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(r,timeout=3600).read())
    return time.time()-t0, d.get("usage",{}).get("completion_tokens",0)
def batch(ps,mt):
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=len(ps)) as ex: tk=sum(ex.map(lambda p: post(p,mt)[1],ps))
    return time.time()-t0,tk
# decode prompt 固定在语料开头；prefill 用 OFFS 指定的窗口（与开头不重叠）
dec=[calib(_c[0:60000],1200) for _ in range(4)]
for off in OFFS:
    win=_c[off:off+300000]
    pf=calib(win,PF); ntok=count(pf+"\n\n请续写。")
    w1,t1=batch(dec[:CONC],512)
    res={}
    th=threading.Thread(target=lambda: res.__setitem__("pf",post(pf,1,ig=False))); th.start()
    time.sleep(0.5)
    w2,t2=batch(dec[:CONC],512); th.join()
    print("[off=%dK] prefill=%d tok 基线%.1f | 混布%.1f (背景TTFT=%.2fs) | %+.1f%%  零和预测=%.1f%%"
          %(off//1000,ntok,t1/w1,t2/w2,res["pf"][0],100.0*(t2/w2)/(t1/w1)-100.0, -100.0*res["pf"][0]/w2),flush=True)
