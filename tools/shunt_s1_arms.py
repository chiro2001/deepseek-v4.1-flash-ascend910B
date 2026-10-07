#!/usr/bin/env python3
"""S1 实测臂 v3（正确结构）：分段 fork/join vs 逐算子 fork/join。

关键规则（本轮实测）：**所有侧流必须在 capture_end 之前 join 回根流**，
否则报 "capture model contains a stream that was not joined to the original stream"。

臂：
  perop   —— 每个 AIV 算子一次 fork+join（= 最朴素做法）
  seg_k   —— 每 k 个 AIC/AIV 单元做一次 fork/join（分段侧流）
  ideal   —— 全链一次 fork（数学上界）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
W=bf16(1024,5120); G=bf16(5120)
K=int(os.environ.get("K","128"))
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
def aic(i): return torch.matmul(XA[i],W)
def aiv(i): return torch_npu.npu_rms_norm(XV[i],G)[0]

def cap_perop(K):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    e=[torch.npu.Event() for _ in range(K)]; ejoin=torch.npu.Event()
    def body():
        for i in range(K):
            e[i].record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e[i]); aiv(i)
            aic(i)
        with torch.npu.stream(s2):
            ejoin.record(s2)          # ★ 收尾必须 join 回根流
        root.wait_event(ejoin)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

def cap_seg(K, seg):
    """每 seg 个单元：1 次 fork → s2 上跑 seg 个 AIV → 1 次 join；root 上跑 seg 个 AIC。"""
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    nseg=(K+seg-1)//seg
    ef=[torch.npu.Event() for _ in range(nseg)]; ej=[torch.npu.Event() for _ in range(nseg)]
    def body():
        for s in range(nseg):
            lo,hi=s*seg,min((s+1)*seg,K)
            ef[s].record(root)
            with torch.npu.stream(s2):
                s2.wait_event(ef[s])
                for i in range(lo,hi): aiv(i)
                ej[s].record(s2)
            for i in range(lo,hi): aic(i)
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

print("K=%d 单元"%K)
res={}
g=cap_perop(K); res["perop"]=meas(g)
print("%-18s %10.3f ms %9.2f µs/单元" % ("perop(逐算子)", res["perop"], res["perop"]/K*1000))
base=res["perop"]
print("\n%-18s %10s %9s %9s"%("seg(分段)","makespan","µs/单元","相对 perop"))
for seg in (2,4,8,16,32,K):
    try: gg=cap_seg(K,seg)
    except Exception as ex:
        print("%-18s FAIL %s"%(seg,str(ex).replace("\n"," ")[:60])); continue
    dt=meas(gg); res["seg%d"%seg]=dt
    print("%-18s %10.3f %9.2f %8.3fx" % (("seg=%d"%seg) if seg<K else "seg=全链", dt, dt/K*1000, base/dt))
json.dump(res,open("/tmp/shunt_arms.json","w"),indent=1)
