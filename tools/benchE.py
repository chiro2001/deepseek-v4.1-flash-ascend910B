#!/usr/bin/env python3
"""跨流 event 代价：细粒度依赖切链是否可行。"""
import torch, torch_npu, time
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
A=bf16(M,1024); B=bf16(1024,5120); X=bf16(M,5120); G=bf16(5120)
f_aic=lambda: torch.matmul(A,B)
f_aiv=lambda: torch_npu.npu_rms_norm(X,G)[0]
def cap(chain_len, use_event):
    """chain_len 对 (AIC,AIV) 交替；use_event=True 时用 cross-stream event 串起来。"""
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ev=[torch.npu.Event() for _ in range(8)]
    def body():
        ws=torch.npu.Stream()
        with torch.npu.stream(ws):
            for _ in range(chain_len):
                f_aic(); f_aiv()
        torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
        with torch.npu.graph(g, stream=root):
            if not use_event:
                for _ in range(chain_len):
                    f_aic(); f_aiv()
            else:
                for i in range(chain_len):
                    f_aic()
                    e=ev[i%8]; e.record(root); s2.wait_event(e)
                    with torch.npu.stream(s2):
                        f_aiv()
                    e2=ev[(i+4)%8]; e2.record(s2); root.wait_event(e2)
    body(); return g
REP=100
print("%-26s %12s %12s"%("配置","每链(ms)","每步µs"))
for L in (16,48):
    for ue in (False,True):
        try:
            g=cap(L,ue)
        except Exception as e:
            print("%-26s FAIL %s"%(f"L={L} event={ue}",str(e).replace(chr(10),' ')[:70])); continue
        for _ in range(5): g.replay()
        torch.npu.synchronize(); t0=time.perf_counter()
        for _ in range(REP): g.replay()
        torch.npu.synchronize(); dt=(time.perf_counter()-t0)/REP*1000
        print("%-26s %12.3f %12.2f"%(f"L={L} event={ue}",dt,dt/L*1000))
