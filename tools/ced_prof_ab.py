#!/usr/bin/env python3
"""两个 profiler 的**逐算子 A/B 对照**，并把差异换算成 ms/step。

为了排除"批大小/请求数不同"的干扰，本工具只按**每步次数**归一：
    per_step = count ÷ steps
    us/step  = sum(duration) ÷ steps
其中 steps 由调用方给出（本项目用 HcPre 次数 ÷ 每步次数精确数出，
见 experiments/dspark/step_period.py）。

用法：
  python3 tools/ced_prof_ab.py a.csv:80 b.csv:86 --label "SPEC0 vs SPEC7"
  # 也可加时间窗（秒，相对各自窗口起点）只比 decode 稳态段：
  python3 tools/ced_prof_ab.py a.csv:80:4.2:13.2 b.csv:86:0.5:9.5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load(spec: str):
    """spec = path:per_step[:t0[:t1]]

    步数由 HcPre 次数精确数出（每步 per_step 次），所以能给出**真实的
    ms/step**，而不是"窗口 ÷ 步数"那种被 prefill/尾部污染的近似。
    """
    parts = spec.split(":")
    path, per_step = parts[0], int(parts[1])
    t0 = float(parts[2]) if len(parts) > 2 and parts[2] else None
    t1 = float(parts[3]) if len(parts) > 3 and parts[3] else None

    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("Start Time(us)") or "").strip()
            if not v:
                continue
            try:
                rows.append((float(v), (r.get("Name") or ""), float(r["Duration(us)"])))
            except (ValueError, KeyError, TypeError):
                continue
    if not rows:
        raise SystemExit(f"{path}: 没有可解析的行")
    origin = min(x[0] for x in rows)
    if t0 is not None:
        rows = [x for x in rows if (x[0] - origin) / 1e6 >= t0]
    if t1 is not None:
        rows = [x for x in rows if (x[0] - origin) / 1e6 <= t1]
    if not rows:
        raise SystemExit(f"{path}: 时间窗内为空")

    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "us": 0.0})
    for _t, name, dur in rows:
        short = name.split("_")[0] if "_" in name else name
        agg[short]["n"] += 1
        agg[short]["us"] += dur

    n_hcpre = sum(v["n"] for k, v in agg.items() if k == "HcPre")
    steps = n_hcpre / per_step if per_step else 0
    if steps <= 0:
        raise SystemExit(f"{path}: 窗口内 HcPre={n_hcpre}，推不出步数")
    lo = min(x[0] for x in rows)
    hi = max(x[0] + x[2] for x in rows)
    return {
        "path": path, "per_step": per_step, "agg": agg,
        "window_s": (hi - lo) / 1e6, "rows": len(rows),
        "steps": steps, "ms_per_step": (hi - lo) / 1000.0 / steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=0.05,
                    help="只显示 |Δ ms/step| 大于此值的行")
    ap.add_argument("--names-a", default="A")
    ap.add_argument("--names-b", default="B")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    print(f"== {args.label or (args.a + '  vs  ' + args.b)}")
    for tag, nm, X in (("A", args.names_a, A), ("B", args.names_b, B)):
        print(f"   {tag}[{nm}] window={X['window_s']:.3f}s rows={X['rows']} "
              f"steps={X['steps']:.1f} -> {X['ms_per_step']:.2f} ms/step")

    all_names = set(A["agg"]) | set(B["agg"])
    rows = []
    for name in all_names:
        a = A["agg"].get(name, {"n": 0, "us": 0.0})
        b = B["agg"].get(name, {"n": 0, "us": 0.0})
        # 「每步分担的 core 时」= 该算子总耗时 ÷ 步数。
        # 不能按 per_step 计数归——有些算子（metadata / 通信）的每步次数
        # 与层数无关，按计数归会判错。
        a_ms = a["us"] / 1000.0 / A["steps"]
        b_ms = b["us"] / 1000.0 / B["steps"]
        rows.append((name, a["n"] / A["steps"], b["n"] / B["steps"], a_ms, b_ms, b_ms - a_ms))

    rows.sort(key=lambda r: -abs(r[5]))
    tot_a = sum(r[3] for r in rows)
    tot_b = sum(r[4] for r in rows)
    print(f"\n{'op':<36}{'n/step A':>9}{'n/step B':>9}{'A ms/step':>11}{'B ms/step':>11}{'Δ ms/step':>11}")
    shown = 0
    for name, na, nb, am, bm, dl in rows:
        if abs(dl) < args.min_delta and shown >= args.top:
            break
        if abs(dl) < args.min_delta:
            continue
        print(f"{name[:36]:<36}{na:>9.2f}{nb:>9.2f}{am:>11.3f}{bm:>11.3f}{dl:>+11.3f}")
        shown += 1
        if shown >= args.top:
            break
    print(f"\n   累加 core 时/步：A={tot_a:.2f} ms  B={tot_b:.2f} ms  Δ={tot_b-tot_a:+.2f} ms")
    print(f"   步周期：        A={A['ms_per_step']:.2f} ms  B={B['ms_per_step']:.2f} ms  "
          f"Δ={B['ms_per_step']-A['ms_per_step']:+.2f} ms")
    print(f"   core 占比：     A={100*tot_a/A['ms_per_step']:.1f}%  B={100*tot_b/B['ms_per_step']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
