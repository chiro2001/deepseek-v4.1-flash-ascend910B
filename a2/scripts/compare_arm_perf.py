#!/usr/bin/env python3
"""相位感知的**臂间性能对比** —— 专门防"拿 A 臂的 fill 相位比 B 臂的 replay 相位"这类误判。

## 为什么需要它（2026-09-23 00:xx 主代理自己踩了一次）

我看 `TRUE_TOKENS=0` 与 `=1` 两条臂时，拿**旧臂早期的** `d2h=14.2ms` 去比**新臂晚期的** `d2h=63.4ms`，
得出"精确修复带来 4.5× 每步回退"的结论 —— **完全错了**。同 steps 对齐后两条臂
`d2h≈63ms`（steps=200: 63.4 vs 62.7；steps=800: 63.5 vs 63.2），**没有回退**。
同理 `A`：旧臂 **fill 相位**是 1.15–1.32，**replay2 相位**才是 3.2；
我拿新臂的 fill（1.24）去比旧臂的 replay2（3.2），又一次比错了相位。

⇒ 本工具强制**先按相位分组、再比**，且：
  · `bneck` 的 `d2h/hp` 按 **steps 对齐**取同一步（它们**随 steps 增长**，不能跨步比）；
  · 每个指标都标出**它是哪个相位的**；相位不可比时**拒绝给结论**（打印原因）。

用法：
    python3 compare_arm_perf.py --a <armA_dir> --b <armB_dir> [--label-a X --label-b Y]
    # 目录里应有 <tag>.client.json（要 serve.log 就再加 --log-a/--log-b）

退出码：0 = 比出来了；1 = 有回退超过阈值（默认 10%）；2 = 用法/数据不足
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REGRESS_PCT = 10.0   # 超过这个百分比才算"值得关注的回退"


def load_client(d: str):
    f = glob.glob(os.path.join(d, "*.client.json"))
    if not f:
        return None, d
    return json.load(open(f[0])), f[0]


def phase_of(round_obj):
    t = (round_obj.get("tag") or "").lower()
    return t if t else "?"


def rnd(x, n=2):
    try:
        return round(float(x), n)
    except Exception:  # noqa: BLE001
        return None


def client_table(d, label):
    """把每个相位摊平：TTFT / total / tok_s / wall / gen_tokens。"""
    rounds = d.get("rounds") or []
    out = {}
    for i, r in enumerate(rounds):
        ph = phase_of(r) or f"round{i}"
        ttft = r.get("ttft") or {}
        tot = r.get("total") or {}
        out[ph] = {
            "idx": i,
            "wall_s": rnd(r.get("wall_s")),
            "ok": r.get("requests_ok"),
            "failed": r.get("requests_failed"),
            "gen_tokens": r.get("gen_tokens"),
            "tok_per_s": rnd(r.get("tok_per_s")),
            "ttft_p50_ms": rnd(ttft.get("p50_ms")),
            "ttft_mean_ms": rnd(ttft.get("mean_ms")),
            "ttft_p90_ms": rnd(ttft.get("p90_ms")),
            "total_p50_ms": rnd(tot.get("p50_ms")),
        }
    return out


def bneck_at_steps(log: str, steps: int, tp: str = "TP0_EP0"):
    """取某 rank 在**恰好 steps** 那一步的 bneck 读数。★ 这些量随 steps 增长，必须对齐后再比。"""
    if not log or not os.path.exists(log):
        return None
    best = None
    for ln in open(log, errors="replace"):
        if "bneck" not in ln or tp not in ln:
            continue
        m = re.search(r"steps=(\d+)", ln)
        if not m or int(m.group(1)) != steps:
            continue
        g = re.search(r"d2h=([\d.]+) hash=([\d.]+) hp=([\d.]+)", ln)
        if g:
            best = {"d2h_ms": float(g.group(1)), "hash_ms": float(g.group(2)),
                    "hp_ms": float(g.group(3))}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 臂输出目录（含 *.client.json）")
    ap.add_argument("--b", required=True, help="B 臂输出目录")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--log-a", default="")
    ap.add_argument("--log-b", default="")
    ap.add_argument("--steps", default="200,400,800",
                    help="用逗号分隔的 steps 采样点（bneck 对齐用）")
    a = ap.parse_args()

    da, fa = load_client(a.a)
    db, fb = load_client(a.b)
    if da is None or db is None:
        print(f"缺 client.json：a={fa} b={fb}", file=sys.stderr)
        return 2

    print("=" * 92)
    print(f"臂间对比（**按相位分组**；不同相位之间不给结论 —— 见文件头那条教训）")
    print(f"  {a.label_a}: {fa}")
    print(f"  {a.label_b}: {fb}")
    print("=" * 92)

    ta, tb = client_table(da, a.label_a), client_table(db, a.label_b)
    phases = [p for p in ta if p in tb]
    if not phases:
        print("两个 client.json **相位名不重合** ⇒ 拒绝给结论（这本身就是'不能比'的证据）")
        print(f"  A 的相位：{sorted(ta)}")
        print(f"  B 的相位：{sorted(tb)}")
        return 2

    hdr = f"{'相位':<10} {'指标':<16} {a.label_a:>14} {a.label_b:>14} {'变化':>12}  判"
    print(hdr)
    print("-" * 92)
    worst = 0.0
    for ph in phases:
        for key, unit, better in [("ttft_p50_ms", "ms", "low"), ("ttft_mean_ms", "ms", "low"),
                                  ("wall_s", "s", "low"), ("tok_per_s", "tok/s", "high"),
                                  ("total_p50_ms", "ms", "low")]:
            va, vb = ta[ph].get(key), tb[ph].get(key)
            if va is None or vb is None or va == 0:
                continue
            pct = (vb - va) / va * 100.0
            # 对"越低越好"的指标，正 pct = 变差；对"越高越好"，负 pct = 变差
            worse = pct if better == "low" else -pct
            worst = max(worst, worse)
            flag = "✓" if worse <= REGRESS_PCT else ("⚠ 回退" if worse > 0 else "✓ 变好")
            print(f"{ph:<10} {key:<16} {va:>14} {vb:>14} {pct:>+11.1f}%  {flag} ({unit})")
        # 每相位也报一下 round 的内部一致性
        print(f"{ph:<10} {'(ok/failed)':<16} {str(ta[ph]['ok'])+'/'+str(ta[ph]['failed']):>14} "
              f"{str(tb[ph]['ok'])+'/'+str(tb[ph]['failed']):>14}")
    print("-" * 92)

    # ---- bneck 按 steps 对齐 ----
    if a.log_a and a.log_b:
        print("bneck 按 **same steps** 对齐（这些量随 steps 增长，跨步比会得出假回退）：")
        for s in [int(x) for x in a.steps.split(",") if x.strip().isdigit()]:
            ra = bneck_at_steps(a.log_a, s)
            rb = bneck_at_steps(a.log_b, s)
            if ra and rb:
                line = f"  steps={s:<6}"
                for k in ("d2h_ms", "hp_ms", "hash_ms"):
                    line += f" {k}: {ra[k]:>8.2f} → {rb[k]:>8.2f}"
                print(line)
            else:
                print(f"  steps={s:<6} 一侧缺该步读数（A={'有' if ra else '无'} B={'有' if rb else '无'}）"
                      " ⇒ 不比较")
    else:
        print("（没给 --log-a/--log-b ⇒ 跳过 bneck 对比；★ 若要评判'每步开销'必须给，"
              "并按 same steps 比）")

    print("=" * 92)
    print(f"最大回退 = {worst:+.1f}%（阈值 {REGRESS_PCT:.0f}%）")
    if worst > REGRESS_PCT:
        print("✗ 有超过阈值的回退，见上面标 '⚠ 回退' 的行")
        return 1
    print("✓ 同相位内没有超过阈值的回退")
    return 0


if __name__ == "__main__":
    sys.exit(main())
