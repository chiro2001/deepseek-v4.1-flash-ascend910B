import csv, glob, os, statistics
from collections import Counter
base="/opt/dsv41/results/ab_mkc_1001_215615/prof_cap8"
cands=sorted(glob.glob(base+"/dp0_pp0_tp0_dcp0_ep0_rank0_*_ascend_pt"), key=os.path.getmtime)
f=cands[-1]+"/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
rows=[]
for r in csv.DictReader(open(f,newline="")):
    try:
        st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        aic=float(r.get("aicore_time(us)") or 0); aiv=float(r.get("aiv_time(us)") or 0)
    except: continue
    rows.append((st,st+du,(r.get("Stream ID") or "").strip(),aic,aiv))
rows.sort()
# 取最密的一段：以 200ms 滑窗找 kernel 最多的区域
W=200000.0
best=(0,0,0)
j=0
for i in range(len(rows)):
    while rows[j][0] < rows[i][0]-W: j+=1
    if i-j > best[0]: best=(i-j, rows[i][0]-W, i)
n,S,i2 = best
E = rows[i2][1]
sel=[x for x in rows if x[1]>S and x[0]<E]
sp=(E-S)/1000
ev=[]
for x in sel: ev.append((x[0],1)); ev.append((x[1],-1))
ev.sort(key=lambda y:(y[0],-y[1]))
cur=0; last=ev[0][0]; dual=0.0; any1=0.0; peak=0
for t,dd in ev:
    if cur>=1: any1+=t-last
    if cur>=2: dual+=t-last
    cur+=dd; peak=max(peak,cur); last=t
sa=sum(x[3] for x in sel); sv=sum(x[4] for x in sel)
print("=== 取 device 最忙的 200ms 窗口 ===")
print("  kernel %d 个, 窗口 %.1f ms"%(len(sel),sp))
print("  有 kernel 在跑的时间   : %.1f ms (%.1f%%)"%(any1/1000, any1/(sp*1000)*100))
print("  **≥2 kernel 同时(真并发)**: %.1f ms (%.1f%% of 窗口, %.1f%% of 活跃)"%(
    dual/1000, dual/(sp*1000)*100, dual/any1*100 if any1 else 0))
print("  峰值并发 %d"%peak)
print("  AIC 忙 %.1f%%  AIV 忙 %.1f%%  两者之和 %.1f%%"%(sa/(sp*1000)*100, sv/(sp*1000)*100, (sa+sv)/(sp*1000)*100))
# 双流交叉：找出并发时刻都在哪些 stream
both=Counter()
ev2=[]
for x in sel: ev2.append((x[0],1,x[2])); ev2.append((x[1],-1,x[2]))
ev2.sort(key=lambda y:(y[0],-y[1]))
cur_s=Counter(); last=ev2[0][0]
for t,dd,s in ev2:
    if sum(cur_s.values())>=2:
        top=tuple(sorted([k for k,v in cur_s.items() if v>0])[:2])
        both[top]+=t-last
    cur_s[s]+=dd; last=t
print("\n  并发时刻的 stream 组合 Top5:")
for k,v in both.most_common(5):
    print("     %-20s %.1f ms"%(str(k),v/1000))
