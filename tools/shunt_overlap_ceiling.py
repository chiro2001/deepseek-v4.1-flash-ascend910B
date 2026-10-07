#!/usr/bin/env python3
"""两个补充实验：
  ① 序列长度对"自由漂移净收益"的影响（16/32/64 层）
  ② Shunt 的理想形态：cube 类工作 ∥ vector 类工作（不切批！）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
m=int(os.environ.get("MBATCH","24"))
TEMPLATE=["mix","rms","dq","mix","rms","dqs","mix","rms","mix","dqs","rms","mm","mm","dqs","rms",
          "mix","rms","mm","dqs","rms","dqs","dqs","dqs","dqs","mix","mix","mix","dqs","dqs","rms","rms","rms"]
OK={"mix":"mix","mm":"mm","rms":"rms","dq":"dqs","dqs":"dqs"}
def mk(M):
    bf=lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
    f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
    return dict(M=M,mm_a=bf(M,1024),mm_w=bf(1024,5120),rms_x=bf(M,5120),rms_g=bf(5120),
                dq_x=bf(M,1280),mix_x=bf(M,5120),
                row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
                exp=torch.topk(f32(M,384),6).indices.to(torch.int32))
F={"mix":lambda b: torch_npu.npu_moe_init_routing(b["mix_x"],b["row"],b["exp"],b["M"]),
   "mm":lambda b: torch.matmul(b["mm_a"],b["mm_w"]),
   "rms":lambda b: torch_npu.npu_rms_norm(b["rms_x"],b["rms_g"])[0],
   "dqs":lambda b: torch_npu.npu_dynamic_quant(b["dq_x"])[0]}
REP=20
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
def cap(fn):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): fn(root)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): fn(root)
    return g

print("=== ① 序列长度对自由漂移净收益的影响（两批，端点同步）===")
print("%-8s %11s %11s %11s %10s"%("层数","单批","两批串行","两批两流","两流/串行"))
r1={}
for L in (8,16,32,64):
    seq=[OK[t] for t in TEMPLATE]*L
    B1,B2,B3=mk(2*m),mk(m),mk(m)
    for b in (B1,B2,B3):
        for k in set(seq): F[k](b)
    torch.npu.synchronize()
    def f_one(root):
        for k in seq: F[k](B1)
    def f_ser(root):
        for k in seq: F[k](B2)
        for k in seq: F[k](B3)
    def f_two(root):
        s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
        e1.record(root)
        with torch.npu.stream(s2):
            s2.wait_event(e1)
            for k in seq: F[k](B3)
            e2.record(s2)
        for k in seq: F[k](B2)
        root.wait_event(e2)
    a=meas(cap(f_one)); c=meas(cap(f_ser)); b=meas(cap(f_two))
    r1[L]=(a,c,b)
    print("%-8d %11.3f %11.3f %11.3f %9.3fx"%(L,a,c,b,c/b))

print("\n=== ② Shunt 理想形态：cube 类 ∥ vector 类（同一批，不切批）===")
L=32
seq=[OK[t] for t in TEMPLATE]*L
CUBE=[k for k in seq if k in ("mix","mm")]
VEC =[k for k in seq if k in ("rms","dqs")]
print("  cube 类 %d 个 (mix+mm) | vector 类 %d 个 (rms+dq)"%(len(CUBE),len(VEC)))
B=mk(2*m)
for k in set(seq): F[k](B)
torch.npu.synchronize()
def f_ser(root):
    for k in CUBE+VEC: F[k](B)
def f_two(root):
    s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
    e1.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(e1)
        for k in VEC: F[k](B)
        e2.record(s2)
    for k in CUBE: F[k](B)
    root.wait_event(e2)
a=meas(cap(f_ser)); b=meas(cap(f_two))
print("  串行 %.3f ms → 两流 %.3f ms  =  %.3fx"%(a,b,a/b))
json.dump({"len_sweep":r1,"ideal":[a,b]},open("/tmp/shunt_ceiling.json","w"),indent=1)
