#!/usr/bin/env python3
"""决定性实验：一个 batch vs 两个 batch（各自一条流，只在首尾同步）。

算子顺序取自真实 trace（armF_r6_base 主流，按 core/block 映射到可用代理算子）：
  mix  ← MIX_AIC 24blk（HcPre/SparseFlashMla/GroupedMatmul）
  mm   ← AI_CORE 20~23blk（MatMulV2/V3）
  rms  ← AI_VECTOR_CORE 48blk（RmsNorm/HcPost/MoeGating/Unpermute）
  dqs  ← AI_VECTOR_CORE 4~16blk（DynamicQuant/Cast/Neg/Small）

臂：
  A) one_batch   —— 单个 M=2m 的 batch 跑完整模板（= 现状 vLLM）
  B) two_stream  —— 两个 M=m 的 batch，各自一条流，**只在首尾 fork/join**（用户设想）
  C) two_serial  —— 两个 M=m 的 batch 串在同一流上
"""
import torch, torch_npu, time, os, json, sys
torch.npu.set_device(0); dev="npu:0"
LAYERS = int(os.environ.get("LAYERS","16"))
m = int(os.environ.get("MBATCH","24"))     # 每个小 batch 的 token 数

# 一层模板（core 类型 + 用哪种代理 + 形状）
# t: mix / mm / rms / dqs
TEMPLATE = ["mix","rms","dq","mix","rms","dqs","mix","rms","mix","dqs","rms","mm","mm","dqs","rms",
            "mix","rms","mm","dqs","rms","dqs","dqs","dqs","dqs","mix","mix","mix","dqs","dqs","rms","rms","rms"]
OPKIND = {"mix":"mix","mm":"mm","rms":"rms","dq":"dqs","dqs":"dqs"}
SEQ = [OPKIND[t] for t in TEMPLATE] * LAYERS

def make_batch(M):
    """为一个 batch 建好所有张量"""
    bf=lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
    f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
    return dict(
        M=M,
        mm_a=bf(M,1024), mm_w=bf(1024,5120),
        rms_x=bf(M,5120), rms_g=bf(5120),
        dq_x=bf(M,1280),
        mix_x=bf(M,5120),
        row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
        exp=torch.topk(f32(M,384),6).indices.to(torch.int32),
    )
def op_mix(b):  return torch_npu.npu_moe_init_routing(b["mix_x"],b["row"],b["exp"],b["M"])
def op_mm(b):   return torch.matmul(b["mm_a"],b["mm_w"])
def op_rms(b):  return torch_npu.npu_rms_norm(b["rms_x"],b["rms_g"])[0]
def op_dqs(b):  return torch_npu.npu_dynamic_quant(b["dq_x"])[0]
FUN={"mix":op_mix,"mm":op_mm,"rms":op_rms,"dqs":op_dqs}

B1 = make_batch(2*m)   # 大 batch
B2 = make_batch(m); B3 = make_batch(m)  # 两个小 batch
for b in (B1,B2,B3):
    for k in set(SEQ): FUN[k](b)
torch.npu.synchronize()

def emit(b, seq): 
    for k in seq: FUN[k](b)

def build(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        if arm=="one_batch":
            emit(B1, SEQ)
        elif arm=="two_serial":
            emit(B2, SEQ); emit(B3, SEQ)
        else:  # two_stream：只在首尾同步
            s2=torch.npu.Stream(); ef=torch.npu.Event(); ej=torch.npu.Event()
            ef.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(ef)
                emit(B3, SEQ)
                ej.record(s2)
            emit(B2, SEQ)
            root.wait_event(ej)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=int(os.environ.get("REP","25"))
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000

print("层数=%d  每层 %d 算子  每流 %d 算子  | 大 batch M=%d  小 batch M=%d ×2"%(LAYERS,len(TEMPLATE),len(SEQ),2*m,m))
print("算子构成: mix=%d mm=%d rms=%d dqs=%d"%(SEQ.count("mix"),SEQ.count("mm"),SEQ.count("rms"),SEQ.count("dqs")))
print("\n%-16s %12s %12s %10s"%("臂","makespan(ms)","每层µs","相对单批"))
res={}; base=None
for arm,lab in (("one_batch","A 单批 M=%d"%(2*m)),("two_stream","B 两流(首尾同步)"),("two_serial","C 两批串行")):
    try: g=build(arm)
    except Exception as e:
        print("%-16s FAIL %s"%(lab,str(e).replace("\n"," ")[:70])); continue
    dt=meas(g); res[arm]=dt
    if arm=="one_batch": base=dt
    print("%-16s %12.3f %12.2f %10s"%(lab,dt,dt/LAYERS*1000,("%.3fx"%(base/dt)) if base else "-"))
json.dump(res,open("/tmp/shunt_two_batch.json","w"),indent=1)
