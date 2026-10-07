#!/usr/bin/env python3
"""陷阱检查：NPUGraph 是否会对"相同输入+相同算子"做去重（CSE）？"""
import torch, torch_npu, time
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
W = bf16(1024,5120); G = bf16(5120)
K = 64
# 两种输入布局
same_x  = bf16(M,1024)                                   # 所有单元共用
dist_x  = [bf16(M,1024) for _ in range(K)]               # 每单元独立
same_a  = bf16(M,5120)
dist_a  = [bf16(M,5120) for _ in range(K)]

def cap(mode, K):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for i in range(K):
            if mode=="same": torch.matmul(same_x, W)
            else:            torch.matmul(dist_x[i], W)
            if mode=="same": torch_npu.npu_rms_norm(same_a, G)[0]
            else:            torch_npu.npu_rms_norm(dist_a[i], G)[0]
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=40
print("%-14s %12s %14s"%("输入布局","makespan(ms)","每算子(µs)"))
for mode in ("same","dist"):
    g=cap(mode,K)
    for _ in range(5): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); dt=(time.perf_counter()-t0)/REP*1000
    print("%-14s %12.3f %14.2f"%(mode, dt, dt/(2*K)*1000))
