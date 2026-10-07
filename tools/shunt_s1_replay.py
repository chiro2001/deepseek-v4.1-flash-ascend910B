#!/usr/bin/env python3
"""S1 实测臂 v2：真实算子的"依赖密度 → 收益"曲线。

臂：
  pure_aic / pure_aiv      —— 纯 AIC / 纯 AIV 链，量出 a、v（图模式真实设备时长）
  alt_join=k (k=1,2,4,...) —— AIC/AIV 交替，每 k 个单元插一次 join（只保留 1/k 的依赖边）
  alt_join=0               —— 无依赖（数学上界）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M = 48
def bf16(*s): return torch.randn(*s, dtype=torch.bfloat16, device=dev)
W = bf16(1024,5120); G = bf16(5120)
K = int(os.environ.get("K","128"))
XS_A = [bf16(M,1024) for _ in range(K)]
XS_V = [bf16(M,5120) for _ in range(K)]
def aic(i): return torch.matmul(XS_A[i], W)
def aiv(i): return torch_npu.npu_rms_norm(XS_V[i], G)[0]

def cap(kind, K, join_every=None):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ea=[torch.npu.Event() for _ in range(K)]; ev=[torch.npu.Event() for _ in range(K)]
    def body():
        if kind=="aic":
            for i in range(K): aic(i)
        elif kind=="aiv":
            for i in range(K): aiv(i)
        else:
            for i in range(K):
                aic(i); ea[i].record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(ea[i]); aiv(i); ev[i].record(s2)
                je = join_every
                if je == 0:
                    if i == K-1: root.wait_event(ev[i])     # 收尾必须 join
                elif je and (i % je == je-1):
                    root.wait_event(ev[i])
                elif je is None:
                    root.wait_event(ev[i])
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

REP=40
def meas(g, nop):
    for _ in range(5): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize()
    return (time.perf_counter()-t0)/REP*1000

print("K=%d（每单元 1×MatMulV2(AI_CORE) + 1×RmsNorm(AI_VECTOR)）" % K)
ga = cap("aic",K); gv = cap("aiv",K)
A = meas(ga,K); V = meas(gv,K)
a = A/K*1000; v = V/K*1000
print("纯 AIC 链 %8.3f ms ⇒ a = %.2f µs/算子" % (A, a))
print("纯 AIV 链 %8.3f ms ⇒ v = %.2f µs/算子" % (V, v))
print("预测：serial=%.1fµs/单元  理想上界=%.1fµs/单元  上界加速=%.3f×"
      % (a+v, max(a,v), (a+v)/max(a,v)))

print("\n%-20s %11s %11s %9s %9s" % ("臂","makespan(ms)","每单元(µs)","加速比","预测"))
base=None; res={}
for je in [None,2,4,8,16,32,64,0]:
    lab = "join=1(现状)" if je is None else ("join=0(上界)" if je==0 else "join=%d"%je)
    try:
        g = cap("alt",K,je)
    except Exception as e:
        print("%-20s CAPTURE FAIL %s" % (lab, str(e).replace("\n"," ")[:60])); continue
    dt = meas(g,2*K)
    if base is None: base = dt
    k = 1 if je is None else (999 if je==0 else je)
    pred = (a+v)/(a + v/k) if k<900 else (a+v)/a
    res[lab]=dt
    print("%-20s %11.3f %11.2f %8.3fx %8.3fx" % (lab, dt, dt/K*1000, base/dt, pred))
json.dump(res, open("/tmp/shunt_s1.json","w"), indent=1)
