#!/usr/bin/env python3
"""U3 探针：主流 AIV 窗口里 AIC 在干什么；全卡 AIC 空闲窗口里有什么在跑。"""
import csv, sys, statistics
from collections import defaultdict, Counter

D = sys.argv[1]
AIC = {"AI_CORE", "MIX_AIC"}
AIV = {"AI_VECTOR_CORE", "MIX_AIV"}
COMM = {"COMMUNICATION"}
AICPU = {"AI_CPU"}

rows = []
with open(D, newline="") as fh:
    for r in csv.DictReader(fh):
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception:
            continue
        core = (r.get("Accelerator Core") or "").strip()
        rows.append((st, du, st + du, core, (r.get("Stream ID") or "").strip(),
                     (r.get("Name") or "")[:38]))
rows.sort()
print("[diag] n=%d  span=%.0f..%.0f (delta %.0f)" % (len(rows), rows[0][0], rows[-1][0], rows[-1][0]-rows[0][0]))
print("[diag] cores:", Counter(r[3] for r in rows).most_common(8))
print("[diag] streams:", Counter(r[4] for r in rows).most_common(10))

anchors = sorted(r[0] for r in rows if r[5].startswith("HcPre"))
PER = 86
if len(anchors) < 3*PER + 10:
    print("[warn] HcPre anchors=%d 太少" % len(anchors)); sys.exit(1)
per_diffs = [anchors[i+PER]-anchors[i] for i in range(len(anchors)-PER)]
STEP = statistics.median(per_diffs)
print("[diag] HcPre anchors=%d  step(每 %d 个 HcPre)=%.1f raw = %.3f ms" % (len(anchors), PER, STEP, STEP/1000))
k0 = PER*3
S, E = anchors[k0], anchors[k0+3*PER]
NSTEP = 3
WIN = E - S
sel = [r for r in rows if r[0] < E and r[2] > S]
print("[diag] window %.0f..%.0f  n_ops=%d" % (S, E, len(sel)))

def merge(iv):
    if not iv: return []
    iv = sorted(iv); out=[list(iv[0])]
    for s,e in iv[1:]:
        if s <= out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out

def inter(A,B):
    i=j=0; tot=0.0
    while i<len(A) and j<len(B):
        s=max(A[i][0],B[j][0]); e=min(A[i][1],B[j][1])
        if e>s: tot+=e-s
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return tot

def span(iv): return sum(e-s for s,e in iv)

MAIN = "109"
def by(cs, stream=None, notstream=None):
    return merge([(r[0],r[2]) for r in sel if r[3] in cs
                  and (stream is None or r[4]==stream)
                  and (notstream is None or r[4]!=notstream)])

mainAIC=by(AIC,MAIN); mainAIV=by(AIV,MAIN)
allAIC=by(AIC); allAIV=by(AIV); allCOMM=by(COMM); allAICPU=by(AICPU)
sideAIC=by(AIC,None,MAIN)

print("\n=== 步均值（%d 步窗口，raw/1000=ms）===" % NSTEP)
for nm, iv in [("主流AIC",mainAIC),("主流AIV",mainAIV),("全卡AIC",allAIC),("全卡AIV",allAIV),
               ("全卡COMM",allCOMM),("全卡AICPU",allAICPU),("侧流AIC",sideAIC)]:
    v=span(iv)/NSTEP/1000
    print(f"  {nm:10s} {v:8.3f} ms/步  ({span(iv)/NSTEP/WIN*100:5.1f}%)")

if mainAIV:
    print(f"\n=== 主流AIV 窗口里谁在跑（{span(mainAIV)/NSTEP/1000:.3f} ms/步）===")
    for nm, iv in [("全卡AIC(任意流)",allAIC),("侧流AIC",sideAIC),
                   ("其它流AIV",by(AIV,None,MAIN)),("COMM",allCOMM),("AICPU",allAICPU)]:
        o=inter(mainAIV,iv)/NSTEP
        print(f"  ∩ {nm:16s} {o/1000:7.3f} ms/步  ({o/span(mainAIV)*100:5.1f}% of AIV)")
else:
    print("\n[warn] 主流上没有 AIV（检查 Stream ID 口径）")

idle=[]; cur=S
for s,e in allAIC:
    if s>cur: idle.append((cur,min(s,E)))
    cur=max(cur,e)
if cur<E: idle.append((cur,E))
idle=merge(idle)
print(f"\n=== 全卡 AIC 空闲窗口（{span(idle)/NSTEP/1000:.3f} ms/步）里谁在跑 ===")
for nm, iv in [("AIV",allAIV),("COMM",allCOMM),("AICPU",allAICPU)]:
    o=inter(idle,iv)/NSTEP
    print(f"  ∩ {nm:10s} {o/1000:7.3f} ms/步  ({o/span(idle)*100:5.1f}% of idle)")
busy_other=merge((allAIV or [])+(allCOMM or [])+(allAICPU or []))
print(f"  → 三者都没有 = {(span(idle)-inter(idle,busy_other))/NSTEP/1000:.3f} ms/步（真空转）")

print("\n=== 侧流 AIC 构成 ===")
agg=defaultdict(lambda:[0.0,0])
for r in sel:
    if r[3] in AIC and r[4]!=MAIN:
        agg[(r[4],r[5])][0]+=r[1]; agg[(r[4],r[5])][1]+=1
for (st,nm),(t,c) in sorted(agg.items(), key=lambda x:-x[1][0])[:10]:
    print(f"  s{st:<4} {nm:<40} {t/NSTEP/1000:8.3f} ms/步  n={c/NSTEP:.1f}")

print("\n=== 主流构成（stream 109）===")
agg2=Counter()
for r in sel:
    if r[4]==MAIN: agg2[(r[3],r[5])]+=1
print("  n_ops/步 = %.1f" % (sum(agg2.values())/NSTEP))
