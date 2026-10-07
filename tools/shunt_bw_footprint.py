#!/usr/bin/env python3
"""决定性实验：两流上不去，是 HBM 争用还是 L2 争用？

变量 = AIC 权重的**足迹**（决定数据从 L2 还是 HBM 来）：
  shared   —— 所有单元共用 1 份权重 (10.5 MB) ⇒ 常驻 L2
  distinct —— 每单元独立权重 (64 × 10.5 MB = 672 MB) ⇒ 超出 192 MiB L2，必须走 HBM
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
K=int(os.environ.get("K","64"))
G=bf16(5120)
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
WSHARED=bf16(1024,5120)
WDIST=[bf16(1024,5120) for _ in range(K)]

def build(mode, kind, seg=16):
    Ws = [WSHARED]*K if mode=="shared" else WDIST
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    nseg=(K+seg-1)//seg
    ef=[torch.npu.Event() for _ in range(nseg)]; ej=[torch.npu.Event() for _ in range(nseg)]
    def body():
        if kind=="serial":
            for i in range(K):
                torch.matmul(XA[i],Ws[i]); torch_npu.npu_rms_norm(XV[i],G)[0]
        elif kind=="aic_only":
            for i in range(K): torch.matmul(XA[i],Ws[i])
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
                for i in range(lo,hi): torch.matmul(XA[i],Ws[i])
                root.wait_event(ej[s])
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=30
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000

# 参考：纯 HBM 带宽标定
big=bf16(512,1024,1024)   # 1 GiB
src=bf16(512,1024,1024)
def bwtest():
    for _ in range(3): big.copy_(src)
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(5): big.copy_(src)
    torch.npu.synchronize(); dt=(time.perf_counter()-t0)/5
    return dt
dt=bwtest(); gb=big.numel()*2/1e9*2   # 读+写
print("参考：大块 copy 实测量 %.2f ms ⇒ %.0f GB/s（读+写）"%(dt*1000, gb/dt))
print("\nK=%d 单元"%K)
print("%-10s %-12s %10s %10s %10s %9s %9s"%("足迹","臂","makespan","µs/单元","理论µs","实测/理论","vs串行"))
out={}
for mode,note in (("shared","10.5MB(L2)"),("distinct","672MB(HBM)")):
    r={}
    for kind in ("aic_only","aiv_only","serial","two_stream"):
        r[kind]=meas(build(mode,kind))
    ideal=max(r["aic_only"],r["aiv_only"])
    for kind in ("aic_only","aiv_only","serial"):
        print("%-10s %-12s %10.3f %10.2f"%(note,kind,r[kind],r[kind]/K*1000))
    ts=r["two_stream"]
    print("%-10s %-12s %10.3f %10.2f %10.2f %8.2fx %8.2fx"%(note,"two_stream",ts,ts/K*1000,ideal/K*1000,ts/ideal,r["serial"]/ts))
    out[mode]={k:v for k,v in r.items()}
    print()
json.dump(out,open("/tmp/shunt_bw.json","w"),indent=1)
