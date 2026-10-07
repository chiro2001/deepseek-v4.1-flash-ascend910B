#!/usr/bin/env python3
"""S1 争用判定：两流上不去（离 max(a,v) 很远）是"HBM 带宽争用"还是"同步/调度开销"？

做法：固定 AIV 工作量，改变 AIC 算子的"权重规模"（决定是否打满 HBM）：
  big ：(48,1024)@(1024,5120)  bf16, W=10.5MB —— 每算子重读 10.5MB
  small：(48,1024)@(1024,512)  bf16, W=1.0MB  —— 权重可驻留
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
K=int(os.environ.get("K","128"))
G=bf16(5120)
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
Wbig=bf16(1024,5120); Wsmall=bf16(1024,512)

def build(W, kind, K, seg=32):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    nseg=(K+seg-1)//seg
    ef=[torch.npu.Event() for _ in range(nseg)]; ej=[torch.npu.Event() for _ in range(nseg)]
    def body():
        if kind=="serial":
            for i in range(K):
                torch.matmul(XA[i],W); torch_npu.npu_rms_norm(XV[i],G)[0]
        elif kind=="aic_only":
            for i in range(K): torch.matmul(XA[i],W)
        elif kind=="aiv_only":
            for i in range(K): torch_npu.npu_rms_norm(XV[i],G)[0]
        else:
            for s in range(nseg):
                lo,hi=s*seg,min((s+1)*seg,K)
                ef[s].record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(ef[s])
                    for i in range(lo,hi): torch_npu.npu_rms_norm(XV[i],G)[0]
                    ej[s].record(s2)
                for i in range(lo,hi): torch.matmul(XA[i],W)
                root.wait_event(ej[s])
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
print("K=%d"%K)
print("%-8s %-12s %11s %11s %11s %9s %9s"%("W","臂","makespan","每单元µs","理论µs","实测/理论","加速"))
for wl,W in (("big(10.5MB)",Wbig),("small(1MB)",Wsmall)):
    r={}
    for kind in ("aic_only","aiv_only","serial","two_stream"):
        r[kind]=meas(build(W,kind,K))
    ideal=max(r["aic_only"],r["aiv_only"])
    print("%-8s %-12s %11.3f %11.2f"%(wl,"aic_only",r["aic_only"],r["aic_only"]/K*1000))
    print("%-8s %-12s %11.3f %11.2f"%(wl,"aiv_only",r["aiv_only"],r["aiv_only"]/K*1000))
    print("%-8s %-12s %11.3f %11.2f"%(wl,"serial",r["serial"],r["serial"]/K*1000))
    print("%-8s %-12s %11.3f %11.2f %11.2f %8.2fx %8.2fx"%(wl,"two_stream",r["two_stream"],
          r["two_stream"]/K*1000, ideal/K*1000, r["two_stream"]/ideal, r["serial"]/r["two_stream"]))
    print()
