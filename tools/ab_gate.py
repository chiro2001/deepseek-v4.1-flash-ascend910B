#!/usr/bin/env python3
"""交替 A/B 门禁：把两组 run 的步时（[bneck] hp）做交替配对比较。
为什么要这个：本机 run-to-run 漂移达 ±5%，而算子级改动只有 2-10µs（约 0.5-4%），
单跑两次会被漂移淹没（gmm1 线实测：单跑曾给出 -9.8µs 的**假象**，交替 12 轮后中位是 -4.95µs）。

用法: ab_gate.py <runA1,runA2,...> -- <runB1,runB2,...> [--n 6]
  · A 臂 = 基线，B 臂 = 改动臂
  · 每个 run 从 `[bneck] hp` 提取按 batch size 分组的中位数
  · 输出逐轮配对差 + 中位 + 符号检验（多少轮为负/正）

判据（建议）：
  1) 配对差中位 < 0
  2) ≥ 2/3 的轮次为负
  3) 所有负轮次的差 > 该 run 自身的 p90-p10 噪声量级
"""
import re, statistics, sys, os

BASE = "/home/l00886679/cedpd-repo/results"

def load_hp(run, nmax=60):
    """返回 {n: p50_hp_ms}（只取 n<=nmax，排除 prefill chunk）"""
    p = f"{BASE}/{run}/serve.log"
    agg = {}
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "[bneck]" not in line:
                continue
            mn = re.search(r"\bn=(\d+)", line)
            mh = re.search(r"\bhp=([\d.]+)", line)
            if not (mn and mh):
                continue
            n = int(mn.group(1))
            if n > nmax:
                continue
            agg.setdefault(n, []).append(float(mh.group(1)))
    return {n: statistics.median(v) for n, v in agg.items() if len(v) >= 5}

def main():
    if "--" not in sys.argv:
        print(__doc__); return 2
    i = sys.argv.index("--")
    A = sys.argv[1:i][0].split(",") if i > 1 else []
    rest = sys.argv[i+1:]
    nmax = 60
    if "--n" in rest:
        j = rest.index("--n"); nmax = int(rest[j+1]); rest = rest[:j] + rest[j+2:]
    B = rest[0].split(",") if rest else []
    if not A or not B:
        print(__doc__); return 2
    da = [(r, load_hp(r, nmax)) for r in A]
    db = [(r, load_hp(r, nmax)) for r in B]
    print(f"A 臂: {A}")
    print(f"B 臂: {B}")
    print(f"{'run':28s} " + " ".join(f"n={n:<4d}" for n in (6, 12, 18, 24, 36, 48)))
    for tag, ds in (("A", da), ("B", db)):
        for r, d in ds:
            if d is None:
                print(f"[{tag}] {r:26s} <缺>"); continue
            row = " ".join(f"{d.get(n, float('nan')):7.2f}" for n in (6, 12, 18, 24, 36, 48))
            print(f"[{tag}] {r:26s} {row}")
    # 配对差（按轮次对齐：A[i] vs B[i]）
    print("\n逐轮配对差（B - A，负数 = B 更快）:")
    deltas = []
    for k in range(min(len(da), len(db))):
        ra, dA = da[k]; rb, dB = db[k]
        if dA is None or dB is None:
            continue
        common = [n for n in dA if n in dB]
        if not common:
            continue
        dd = [dB[n] - dA[n] for n in common]
        med = statistics.median(dd)
        deltas.append(med)
        print(f"  轮{k+1}: {ra} -> {rb}   配对差中位 = {med:+7.3f} ms  ({len(common)} 个 n 档)")
    if deltas:
        neg = sum(1 for x in deltas if x < 0)
        print(f"\n配对差中位 = {statistics.median(deltas):+.3f} ms   "
              f"均值 = {statistics.mean(deltas):+.3f}   "
              f"负轮次 = {neg}/{len(deltas)}")
        print("判据建议：中位 < 0 且 负轮次 ≥ 2/3 才算 B 更优")
        print("★ [STABLE-BUCKET] 上面的中位数可能被过渡桶带偏（n=18 等）。")
        print("  请先看分桶样本数与 p10/p90，只在样本≥64 且 p90 与 p10 同量级的桶上判决：")
        print("      python3 tools/hp_buckets.py <基线run> <候选run>")
    return 0

if __name__ == "__main__":
    sys.exit(main())
