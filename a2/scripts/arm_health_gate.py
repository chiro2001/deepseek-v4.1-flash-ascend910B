#!/usr/bin/env python3
"""arm_health_gate.py —— ★★★ 任何性能臂的「行为健康闸」（`logs/123` 的机械化）

为什么需要它（本轮实测事故，`a2/logs/123-...md`）：
  `r8-g2g3` 臂的 `ms/step` 看起来**没变**（8K 28.008 vs 参考 27.905），若只看它就会得出
  "G2+G3 无收益"的**完全错误**结论 —— 真相是**投机解码的接受率被打崩了**：
  引擎自报 position-0 从 0.62–0.93 掉到 **0.036**、位置 1–4 **全归零**、
  `Mean acceptance length` 1.04（健康 2.80–2.98），而 **tok/s 掉了 2.7×**。
  ⇒ 每步时间由固定开销主导：**draft 被拒不会让步变快，只会让"出 token"变慢**。
  ⇒ 凡改到 attention / KV / 图输入的臂，**必须**同时过这道闸。

数据来源：vLLM 自己的 `metrics.py` 行（**零成本、已在 serve.log 里**）：
  `... Mean acceptance length: 2.81, ... Per-position acceptance rate: 0.931, 0.535, ...,
   Avg Draft acceptance rate: 36.2%`（同段还有 `Accepted: N tokens, Drafted: M tokens`）

★★★ **2026-09-23 07:5x 重要修正（本脚本第一版有相位污染缺陷）**：
  第一版只读 `tail -1`（**最后一个 metrics 窗口**）⇒ **会被相位污染**。
  实测事故：`r8-g3only` 的**最后一个窗口**读数是 `mean=1.25 / position-0=0.245`（看起来崩了），
  但**同一份日志的其余窗口**是 `mean 2.97 / position-0 0.811` —— 后者与**未改一个字节的基线臂**同档。
  而真正崩掉的 `r8-g2g3` 是 **41 个窗口全都低**（中位 1.04 / 0.042）。
  ⇒ **修法：读全部窗口，用「中位 + 聚合」双口径判**（并报 n），`tail` 只作辅助显示。
  ★ 这是本仓同族第 N 次「拿单个窗口当整段」的错（参见 `logs/102` 的"分步分母混相位"）。

判据（**用「聚合」口径为主，中位为辅助**）：
  H0 日志里**必须存在**该行（不存在 ⇒ 按 FAIL 处理：要么没跑 spec，要么日志被截断）
  H0b ★★ **流量不足 ⇒ "无法判定"（rc=65），不是 FAIL** —— 判据：`Σdrafted < 200` 或 `n < 3`
      （典型：**刚起服、压测还没跑**。实测 2026-09-23：`r8-safe-levers` 起服后 1 个窗口、Σdrf=15 ⇒ 旧版误判 FAIL）
  H1 **聚合 A = Σaccepted/Σdrafted + 1 ≥ 1.10**  ← ★ 主判据（抗样本少、抗相位）
  H2 聚合接受率 `Σaccepted/Σdrafted ≥ 0.10`（与 H1 等价，保留为可读显示）
  H3 `Σaccepted > 0` 且 `Σdrafted > 0`
  H4（可选，给了 --ref）**聚合 A** 比参考臂低不超过 --tol（默认 0.10）
  ★ n < 5 时提示"样本少"，但**不因此判 FAIL**（聚合口径本身就是为此设计的）

★★ 为什么主判据必须是「聚合 A」而不是「中位 A」或「最后一窗口」——**实测三臂对照**（同一天同一探针）：
  | 臂 | 最后一窗口 A | 中位 A | **聚合 A** | 真判 |
  |---|---:|---:|---:|---|
  | 基线（未改一字节） | 2.96 | 2.205 | **1.200** | ✅ |
  | `g3only`（n=3） | 2.97 | 1.50 | **1.142** | ✅ 与基线同档 |
  | `g3only2`（n=5） | — | 2.76 | **1.235** | ✅ 同档 |
  | **`g2g3`（n=41）** | 1.05 | 1.04 | **1.009** | ❌ **真崩** |
  ⇒ 只有**聚合 A** 把"真崩的"与"同档的"分开；中位在 n=3 时会误杀。

用法：
  python3 arm_health_gate.py <serve.log|run_dir>
  python3 arm_health_gate.py <serve.log> --ref <另一个 serve.log>
  python3 arm_health_gate.py <serve.log> --json out.json

退出码：0 = 全过；9 = 至少一条 FAIL（★ **该臂性能读数作废，别记进对照表**）；64 = 用法/文件错误
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def resolve(p: str) -> str:
    if os.path.isdir(p):
        cand = os.path.join(p.rstrip("/"), "serve.log")
        if os.path.isfile(cand):
            return cand
        hits = sorted(glob.glob(os.path.join(p.rstrip("/"), "**", "serve.log"), recursive=True))
        if hits:
            return hits[-1]
        return cand
    return p


LINE_RE = re.compile(r"Per-position acceptance rate")
MEAN_RE = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
POS_RE = re.compile(r"Per-position acceptance rate:\s*([0-9.,\s]+?)(?:,?\s*Avg|$)")
ACC_RE = re.compile(r"Accepted:\s*(\d+)")
DRF_RE = re.compile(r"Drafted:\s*(\d+)")


def _parse_line(line: str):
    m = MEAN_RE.search(line)
    mean = float(m.group(1)) if m else None
    m = POS_RE.search(line)
    pos = [float(x) for x in m.group(1).split(",") if x.strip()] if m else []
    m = ACC_RE.search(line); acc = int(m.group(1)) if m else 0
    m = DRF_RE.search(line); drf = int(m.group(1)) if m else 0
    return {"mean": mean, "pos": pos, "accepted": acc, "drafted": drf, "raw": line.strip()}


def _median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_arm(path: str):
    """读**全部** SpecDecoding 窗口（纯流式，O(1) 内存），返回整段口径的汇总或 None。

    ★ 为什么不只看最后一行：metrics 行是**滚动窗口**（每 ~10s 一行），单看哪一行都会被相位污染。
      实测（2026-09-23）：同一份 `r8-g3only` 日志，最后一行 `mean=1.25/pos0=0.245`，
      而其余窗口是 `mean=2.97/pos0=0.811` ⇒ 单窗口会得出相反结论。
    """
    wins = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if LINE_RE.search(line):
                    wins.append(_parse_line(line))
    except OSError as exc:
        print(f"⛔ 读不了 {path}: {exc}", file=sys.stderr)
        return None
    if not wins:
        return None
    means = [w["mean"] for w in wins if w["mean"] is not None]
    p0s = [w["pos"][0] for w in wins if w["pos"]]
    sac = sum(w["accepted"] for w in wins)
    sdr = sum(w["drafted"] for w in wins)
    return {
        "n": len(wins),
        "mean_median": _median(means),
        "mean_min": min(means) if means else None,
        "mean_max": max(means) if means else None,
        "pos0_median": _median(p0s),
        "pos0_min": min(p0s) if p0s else None,
        "pos0_max": max(p0s) if p0s else None,
        "sum_accepted": sac,
        "sum_drafted": sdr,
        "agg_ratio": (sac / sdr) if sdr else 0.0,
        "agg_a": (sac / sdr + 1.0) if sdr else 1.0,
        "last": wins[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--json", default=None, dest="json_out")
    args = ap.parse_args()

    tgt = resolve(args.target)
    if not os.path.isfile(tgt):
        print(f"⛔ 找不到 serve.log：{tgt}", file=sys.stderr)
        return 64

    t = parse_arm(tgt)
    v = 0
    print("== 行为健康闸 ==")
    print(f"  serve.log = {tgt}")

    if t is None:
        print("  ⛔ H0 FAIL：日志里**没有** 'Per-position acceptance rate' 行")
        print("     ⇒ 要么本臂没跑过带 spec 的请求，要么日志被截断 ⇒ **无法判定，按 FAIL 处理**")
        print("     （★ 别跳过：这条正是 'r8-g2g3' 事故里唯一能看出问题的信号）")
        v = 1
    elif t["sum_drafted"] < 200 or t["n"] < 3:
        # ★★ H0b：流量不足 ⇒ 明确报"无法判定"，**不要**报 FAIL（否则会误杀刚起服的臂）
        print(f"  ⚠⚠ H0b **流量不足，无法判定**（n={t['n']} 窗口、Σdrafted={t['sum_drafted']}）")
        print(f"     聚合 A={t['agg_a']:.3f}、聚合接受率={t['agg_ratio']:.3f} —— 但这**不足以**下结论")
        print("     ⇒ 典型原因：**刚起服、压测/quote 还没跑**（此时窗口少、drafted 也少）")
        print("     ⇒ 处置：**等压测跑完再来**；若压测已经跑过 ⇒ 才是真的没跑 spec")
        print("     ★ 判据：`Σdrafted ≥ 200` 且 `n ≥ 3` 才判生判死")
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump({"serve_log": tgt, "target": t, "verdict": "INSUFFICIENT_TRAFFIC"},
                          fh, ensure_ascii=False, indent=2)
            print(f"  （JSON 已写 {args.json_out}）")
        return 65
    else:
        print(f"  窗口数 n={t['n']}"
              + ("   ⚠ n<5：样本少、仅供参考" if t["n"] < 5 else ""))
        print(f"  中位口径：mean={t['mean_median']}（min {t['mean_min']} / max {t['mean_max']}）"
              f"  position-0={t['pos0_median']}（min {t['pos0_min']} / max {t['pos0_max']}）")
        print(f"  聚合口径：Σacc={t['sum_accepted']} / Σdrf={t['sum_drafted']} = {t['agg_ratio']:.3f}")
        print(f"  （参考：最后一窗口 mean={t['last']['mean']} / position-0={t['last']['pos'][0] if t['last']['pos'] else None}"
              f" —— ★ 单窗口会被相位污染，不作为判据）")
        print(f"  ★ 聚合 A = {t['agg_a']:.3f}   ← 主判据（基线 1.200 ｜ G3 1.142/1.235 ｜ 事故 1.009）")
        if t["agg_a"] < 1.10:
            print(f"  ⛔ H1 FAIL: 聚合 A={t['agg_a']:.3f} < 1.10"); v = 1
        else:
            print(f"  ✓ H1 聚合 A={t['agg_a']:.3f}")
        if t["sum_accepted"] <= 0 or t["sum_drafted"] <= 0:
            print(f"  ⛔ H3 FAIL: acc={t['sum_accepted']} drf={t['sum_drafted']}（spec 可能根本没跑）"); v = 1
        elif t["agg_ratio"] < 0.10:
            print(f"  ⛔ H2 FAIL: 聚合接受率 {t['agg_ratio']:.3f} < 0.10"); v = 1
        else:
            print(f"  ✓ H2 聚合接受率 {t['agg_ratio']:.3f}（Σacc={t['sum_accepted']} / Σdrf={t['sum_drafted']}）")
        print(f"  （辅助口径：中位 A={t['mean_median']}、中位 position-0={t['pos0_median']}。"
              f"★ 中位在 n 小时会误杀，故不作判据）")

    r = None
    if args.ref:
        ref = resolve(args.ref)
        if not os.path.isfile(ref):
            print(f"  ⚠ H4 SKIP: 找不到参考臂 {ref}")
        else:
            r = parse_arm(ref)
            if r is None:
                print(f"  ⚠ H4 SKIP: 读数缺失（ref={r is not None}）")
            else:
                d = r["agg_a"] - t["agg_a"]
                print(f"  参考臂 = {ref}（n={r['n']} 聚合 A={r['agg_a']:.3f}）")
                if d > args.tol:
                    print(f"  ⛔ H4 FAIL: 聚合 A 比参考低 {d:.3f} > tol {args.tol}"); v = 1
                else:
                    print(f"  ✓ H4 相对参考臂聚合 A 差 {d:+.3f}（tol {args.tol}）")

    print()
    if v == 0:
        print("✅ 健康闸全过 ⇒ 本臂的性能读数**可用于对照**")
    else:
        print("⛔⛔ 健康闸 FAIL ⇒ ★ **本臂性能读数作废，别记进对照表**")
        print("   处置：先二分归因（哪个改动打崩了 draft），再重跑；")
        print("   参考 logs/123：'ms/step 没变' 与 '行为已回退' 可以同时成立。")

    if args.json_out:
        out = {
            "serve_log": tgt,
            "target": t,
            "ref": r,
            "verdict": "PASS" if v == 0 else "FAIL",
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"  （JSON 已写 {args.json_out}）")
    return v * 9


if __name__ == "__main__":
    sys.exit(main())
