#!/usr/bin/env python3
"""把 SPEC>0 的每个 step 拆成 **target 段 / 草稿段 / 同步段**。

原理：DSpark 的草稿层每步处理 `batch × num_query_per_req` 行，而
`num_query_per_req = num_speculative_tokens`（`sample_from_anchor=True`）；
target 每步处理 `batch × (1 + num_speculative_tokens)` 行。于是行数 (M)
天然把两者分开：

    batch=4, SP=7 → target M=32，draft M=28
    batch=3       → target M=24，draft M=21
    batch=2       → target M=16，draft M=14
    batch=1       → target M=8，draft M=7

（M 从 `Input Shapes` 的第一个数取；同一算子在 target 与草稿里都会出现，
所以只能按 M 分，不能按算子名分。）

用法：
  python3 tools/ced_prof_split.py kd.csv:per_step:t0:t1 \\
      --target-m 32 --draft-m 28 --label "B SPEC=7 batch4"
  python3 tools/ced_prof_split.py kd.csv:80:1.9 --target-m 32 --label "A SPEC=0 batch4"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def get_m(shape: str) -> int | None:
    s = (shape or "").strip().strip('"')
    if not s:
        return None
    head = s.split(",")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="csv:per_step[:t0[:t1]]")
    ap.add_argument("--target-m", required=True,
                    help="逗号分隔的 M 列表。注意不同算子的 M 定义不同："
                         "dense/HcPre 是 num_tokens；MoE grouped matmul 的首维是"
                         "「token × 专家数 × 2」之类的展开维，所以要把同一批大小的"
                         "所有形态都列进来。")
    ap.add_argument("--draft-m", default=None, help="同上，草稿侧")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    parts = args.spec.split(":")
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
                rows.append((float(v), (r.get("Name") or "").split("_")[0],
                             float(r["Duration(us)"]),
                             get_m(r.get("Input Shapes") or ""),
                             (r.get("Accelerator Core") or "").strip()))
            except (ValueError, KeyError, TypeError):
                continue
    origin = min(x[0] for x in rows)
    if t0 is not None:
        rows = [x for x in rows if (x[0] - origin) / 1e6 >= t0]
    if t1 is not None:
        rows = [x for x in rows if (x[0] - origin) / 1e6 <= t1]

    n_hcpre = sum(1 for x in rows if x[1] == "HcPre")
    steps = n_hcpre / per_step
    lo = min(x[0] for x in rows)
    hi = max(x[0] + x[2] for x in rows)
    ms_per_step = (hi - lo) / 1000.0 / steps

    buckets: dict[str, dict] = {
        "target": defaultdict(lambda: {"n": 0, "us": 0.0}),
        "draft": defaultdict(lambda: {"n": 0, "us": 0.0}),
        "other": defaultdict(lambda: {"n": 0, "us": 0.0}),
    }
    tms = {int(x) for x in str(args.target_m).split(",") if x.strip()}
    dms = ({int(x) for x in str(args.draft_m).split(",") if x.strip()}
           if args.draft_m else set())
    for _t, name, dur, m, _core in rows:
        if m in tms:
            b = "target"
        elif m in dms:
            b = "draft"
        else:
            b = "other"
        buckets[b][name]["n"] += 1
        buckets[b][name]["us"] += dur

    print(f"== {args.label or path}")
    print(f"   window={((hi-lo)/1e6):.3f}s  steps={steps:.1f}  "
          f"**{ms_per_step:.2f} ms/step**  (target M∈{sorted(tms)}"
          + (f", draft M∈{sorted(dms)}" if dms else "") + ")")
    tot = 0.0
    for b in ("target", "draft", "other"):
        us = sum(v["us"] for v in buckets[b].values())
        ms = us / 1000.0 / steps
        tot += ms
        n = sum(v["n"] for v in buckets[b].values())
        print(f"\n   [{b}] {ms:>7.2f} ms/step  ({100*ms/ms_per_step:>5.1f}% of step)  "
              f"kernels/step={n/steps:.1f}")
        for name, v in sorted(buckets[b].items(), key=lambda kv: -kv[1]["us"])[:10]:
            print(f"        {name[:44]:<44}{v['n']/steps:>8.2f}/step"
                  f"{v['us']/1000/steps:>10.3f} ms/step")
    print(f"\n   合计 core 时/步 = {tot:.2f} ms   step 周期 = {ms_per_step:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
