#!/usr/bin/env python3
"""S1 分解臂：把"event 成本"从"依赖成本"里剥出来。

臂（K 个 AIC/AIV 单元）：
  1 ser_1stream   两算子全在 root，严格交替（真串行基准）
  2 aiv_only      root 只跑 AIC；AIV 在 s2，**不等**（完全不相关）
  3 wait1way      s2 每步 wait AIC（fork），但不 join 回 root
  4 join_full     fork + join（= 现状形态）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
W=bf16(1024,5120); G=bf16(5120)
K=int(os.environ.get("K","128"))
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
aic=lambda i: torch.matmul(XA[i],W)
aiv=lambda i: torch_npu.npu_rms_norm(XV[i],G)[0]

def cap(kind,K):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ea=[torch.npu.Event() for _ in range(K)]; ev=[torch.npu.Event() for _ in range(K)]
    def body():
        if kind=="ser":
            for i in range(K): aic(i); aiv(i)
        elif kind=="aiv_only":
            with torch.npu.stream(s2):
                for i in range(K): aiv(i)
            for i in range(K): aic(i)
            root.wait_stream(s2)
        else:
            for i in range(K):
                aic(i); ea[i].record(root)
                with torch.npu.stream(s2):
                    if kind=="wait1way": s2.wait_event(ea[i])
                    aiv(i); ev[i].record(s2)
                if kind=="join_full" and i==K-1: root.wait_event(ev[i])
            if kind=="wait1way":
                with torch.npu.stream(s2): pass
                torch.npu.current_stream().wait_stream(s2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=40
def meas(g):
    for _ in range(5): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
print("K=%d 单元"%K)
print("%-14s %11s %11s %9s"%("臂","makespan(ms)","每单元(µs)","相对串行"))
base=None; res={}
for kind,lab in [("ser","1 单流交替"),("aiv_only","2 完全独立"),("wait1way","3 只 fork"),("join_full","4 fork+join")]:
    try: g=cap(kind,K)
    except Exception as e:
        print("%-14s CAPTURE FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    dt=meas(g)
    if base is None: base=dt
    res[lab]=dt
    print("%-14s %11.3f %11.2f %8.3fx"%(lab,dt,dt/K*1000,base/dt))
print("\n分解（µs/单元）：")
s=res.get("1 单流交替"); a=res.get("2 完全独立"); w=res.get("3 只 fork"); j=res.get("4 fork+join")
if s and a: print("  理想两流（无任何同步）      = %.2f" % (a/K*1000))
if s and w: print("  + 每步 fork（record+wait）  = %.2f  ⇒ fork 成本 %.2f" % (w/K*1000, (w-a)/K*1000))
if w and j: print("  + 每步 join                 = %.2f  ⇒ join 成本 %.2f" % (j/K*1000, (j-w)/K*1000))
if s and j: print("  串行基准                    = %.2f  ⇒ 两流实现净开销 %.2f" % (s/K*1000, (j-s)/K*1000))
json.dump(res,open("/tmp/shunt_event.json","w"),indent=1)
