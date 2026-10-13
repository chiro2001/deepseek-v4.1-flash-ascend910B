#!/usr/bin/env python3
"""两个 `bench_concurrency.py` JSON 的配对对比（基线 vs 候选）。

为什么要单做一个：`total_decode` 会同时被"步时"和"接受长度 A"影响，
只看它会误判（本轮就遇到过：A 掉但步时也掉了，方向相反）。
所以这里**同时**给出 ms/step 与 A，并对每个并发档给出相对变化。

ms/step 推导（只用两个可信量，不用墙钟）：
    单流吞吐 = 单请求每秒被接受的 token 数 = A × (步/秒)
    ⇒ ms/step = 1000 × A / per_stream_med
（A = accept_len，含被接受的首 token，与 /metrics 一致。实测 N=1：A=2.78、101.9 tok/s
 ⇒ 27.3 ms/step，与 `[bneck] hp` 中位 26.2–26.3 ms 吻合，故该式可用。）

用法: bench_delta.py <baseline.json> <candidate.json> [--label-b B] [--label-a A]
"""

from __future__ import annotations

import argparse
import json


def load(path: str) -> dict[float, dict]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {float(r["conc"]): r for r in doc["rows"]}, doc


def ms_per_step(row: dict) -> float:
    a = row.get("accept_len") or 0.0
    rate = row.get("per_stream_med") or 0.0
    if a <= 0 or rate <= 0:
        return float("nan")
    return 1000.0 * a / rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="candidate")
    args = ap.parse_args()

    base, _ = load(args.baseline)
    cand, _ = load(args.candidate)

    print(f"A = {args.label_a}   B = {args.label_b}")
    print(f"A: {args.baseline}")
    print(f"B: {args.candidate}")
    print()
    head = (
        f"{'N':>4}{'A tok/s':>10}{'B tok/s':>10}{'Δ%':>8}"
        f"{'A ms/step':>11}{'B ms/step':>11}{'Δ ms':>8}"
        f"{'A agg':>9}{'B agg':>9}{'Δ%':>8}{'A A_len':>9}{'B A_len':>9}"
    )
    print(head)
    for n in sorted(set(base) & set(cand)):
        a, b = base[n], cand[n]
        sA, sB = a["per_stream_med"], b["per_stream_med"]
        gA, gB = a["total_decode"], b["total_decode"]
        mA, mB = ms_per_step(a), ms_per_step(b)
        print(
            f"{int(n):>4}{sA:>10.1f}{sB:>10.1f}{(sB/sA-1)*100:>7.1f}%"
            f"{mA:>11.2f}{mB:>11.2f}{mB-mA:>8.2f}"
            f"{gA:>9.1f}{gB:>9.1f}{(gB/gA-1)*100:>7.1f}%"
            f"{a.get('accept_len',0):>9.2f}{b.get('accept_len',0):>9.2f}"
        )
        if a.get("fail") or b.get("fail"):
            print(f"     ⚠️ fail: A={a.get('fail')} B={b.get('fail')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
