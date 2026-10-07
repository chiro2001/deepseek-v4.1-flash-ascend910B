#!/usr/bin/env python3
"""S1 解析臂：用真实 trace 计算三种调度下的 makespan。

读 kernel_details.csv 的主流(stream 109)算子序列（含真实时长），计算：
  ① serial      —— 单流按原顺序（= 现状）
  ② split_dep   —— 按 AIC/AIV 分两条流，但保留"相邻算子真依赖"（event 串起来）
  ③ split_nodep —— 按 AIC/AIV 分两条流，**无任何依赖**（数学上界，非合法调度）

用法: shunt_trace_schedule.py <ASCEND_PROFILER_OUTPUT> [NSTEP]
"""
import csv, statistics, sys
from collections import defaultdict

D = sys.argv[1]
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
AIC = {"AI_CORE", "MIX_AIC"}
AIV = {"AI_VECTOR_CORE", "MIX_AIV"}
COMM = {"COMMUNICATION"}
AICPU = {"AI_CPU"}

rows = []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception:
            continue
        rows.append((st, du, (r.get("Accelerator Core") or "").strip(),
                     (r.get("Stream ID") or "").strip(), (r.get("Name") or "")))
rows.sort()

anchors = sorted(r[0] for r in rows if r[4].startswith("HcPre"))
PER = 86
STEP = statistics.median([anchors[i+PER]-anchors[i] for i in range(len(anchors)-PER)])
k0 = PER * 5
S, E = anchors[k0], anchors[k0 + NSTEP*PER]
seq = [r for r in rows if r[3] == "109" and r[0] < E and (r[0] + r[1]) > S]
seq.sort(key=lambda x: x[0])
print("步长 %.1f us | 窗口 %d 步 | 主流算子 %d 个 (%.1f/步)"
      % (STEP, NSTEP, len(seq), len(seq)/NSTEP))

def cls(c):
    if c in AIC:   return "AIC"
    if c in AIV:   return "AIV"
    if c in COMM:  return "COMM"
    if c in AICPU: return "AICPU"
    return "OTHER"

tot = defaultdict(float); cnt = defaultdict(int)
for st, du, core, _, _ in seq:
    k = cls(core); tot[k] += du; cnt[k] += 1
serial = sum(tot.values())
print("\n=== 资源构成（主流）===")
for k in ("AIC", "AIV", "COMM", "AICPU", "OTHER"):
    if tot[k]:
        print("  %-6s %8.3f ms/步 (%4.1f%%)  n=%6.1f/步" % (k, tot[k]/NSTEP/1000, tot[k]/serial*100, cnt[k]/NSTEP))

# ---- ② split_dep：两流 + 相邻依赖 ----
free = {"A": 0.0, "B": 0.0}
dep_ready = 0.0
last_finish = 0.0
for st, du, core, _, _ in seq:
    k = cls(core)
    s = "A" if k == "AIC" else "B"
    start = max(free[s], dep_ready)
    fin = start + du
    free[s] = fin
    dep_ready = fin            # 下一个算子必须等这一个完成（相邻真依赖）
    last_finish = fin
split_dep = last_finish

# ---- ③ split_nodep：两流 + 无依赖（上界）----
sa = sum(du for st, du, core, _, _ in seq if cls(core) == "AIC")
sb = sum(du for st, du, core, _, _ in seq if cls(core) != "AIC")
split_nodep = max(sa, sb)

# 切换次数（AIC<->非AIC）
sw = sum(1 for i in range(1, len(seq)) if (cls(seq[i][2]) == "AIC") != (cls(seq[i-1][2]) == "AIC"))

print("\n=== 三种调度的 makespan（%d 步）===" % NSTEP)
print("  ① serial       %9.3f ms   （= 现状，1.00×）" % (serial/1000))
print("  ② split_dep    %9.3f ms   （%.3f×）" % (split_dep/1000, serial/split_dep))
print("  ③ split_nodep  %9.3f ms   （%.3f×，数学上界）" % (split_nodep/1000, serial/split_nodep))
print("\n  各流负载: AIC 侧 %.3f ms | 非AIC 侧 %.3f ms" % (sa/1000, sb/1000))
print("  AIC<->非AIC 切换 %d 次 / %d 步 = %.1f 次/步" % (sw, NSTEP, sw/NSTEP))
print("  ⇒ 每步有 %.1f 个依赖边必须跨流，若全部保留则与串行等价" % (sw/NSTEP))
print("  ⇒ 只有\"删掉依赖边\"才能兑现 ③ 与 ② 之间的 %.3f ms 差额" % ((split_dep-split_nodep)/1000))
