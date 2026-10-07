#!/usr/bin/env python3
"""最干净的 CBP 判据：同一对请求，串行总时长 vs 并行总时长。
   P = T_serial / T_parallel
   完美并行 ⇒ P = (T1+T2)/max(T1,T2) ≈ 2
   无并行   ⇒ P = 1
避免接受长度波动（两臂用完全相同的请求）。
"""
import json, time, threading, urllib.request, statistics, sys, os
CORPUS="/home/l00886679/cedpd-repo/data/hongloumeng.txt"
txt=open(CORPUS,encoding="utf-8",errors="ignore").read()
A="http://127.0.0.1:19400/v1/completions"; B="http://127.0.0.1:19401/v1/completions"
def post(url, idx, out=96):
    body={"model":"deepseek-v41","prompt":txt[100000+idx*137:100000+idx*137+3000],
          "max_tokens":out,"temperature":0}
    req=urllib.request.Request(url,data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as r: json.load(r)
    return time.time()-t0
# 预热
for i in range(2): post(A,i,16); post(B,i+10,16)
print("预热完成\n")
REP=int(os.environ.get("REP","4"))
print("%-5s %10s %10s %12s %12s %8s"%("rep","T1(A)","T2(B)","串行 T1+T2","并行 max","P"))
Ps=[]; th=[]
for i in range(REP):
    ia, ib = 100+i, 200+i
    t1=post(A,ia); t2=post(B,ib)
    ser=t1+t2
    res={}
    def run(t,u,x): res[t]=post(u,x)
    th1=threading.Thread(target=run,args=("A",A,ia)); th2=threading.Thread(target=run,args=("B",B,ib))
    th1.start(); th2.start(); th1.join(); th2.join()
    par=max(res["A"],res["B"])
    P=ser/par; Ps.append(P)
    print("%-5d %10.3f %10.3f %12.3f %12.3f %8.3f"%(i+1,t1,t2,ser,par,P))
m=statistics.median(Ps)
print("\n中位 P = %.3f"%m)
print("理论上限（两请求等长）= 2.000 ; 无并行 = 1.000")
print("⇒ 并行效率 = %.1f%%"%((m-1)*1.0*100))
# 换算成 tax：tax = T_serial/2 / T_par_per... 等价于 2/P
print("⇒ tax(2) = 2/P = %.3f   （判据 S(2)=1.137 ⇒ %s）"%(2/m, "✅ 过线" if 2/m<1.137 else "❌ 未过线"))
