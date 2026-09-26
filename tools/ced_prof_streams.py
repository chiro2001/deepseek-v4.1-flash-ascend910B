#!/usr/bin/env python3
"""从 torch_npu profiler 的 task_time.csv 分析**流（stream）级行为**。

背景与用法见 docs/CED-PD-PROFILING-20260925.md。要拿到 task_time.csv：

  1. 服务带 `PROFILE=1`（= 给 vllm 传 `--profiler-config {"profiler":"torch",...}`）启动；
  2. `curl -XPOST .../start_profile` → 发你要测的请求 → `curl -XPOST .../stop_profile`；
  3. **离线**分析（不需要停服务，也可以在别的机器上做）：

     docker exec <容器> python3 -c "
     from torch_npu.profiler.profiler import analyse
     analyse('/opt/dsv41/results/<run>/prof/<rank>_ascend_pt')"

     产物在 `<rank>_ascend_pt/ASCEND_PROFILER_OUTPUT/`，其中
     `task_time.csv` 每行一个 device task，带 `stream_id` / `kernel_type` /
     `task_time(us)` / `task_start(us)`。这就是"流行为"的原始证据。

本脚本给出四张表：
  A. 每个流的任务数 / 忙时 / 占窗口比 / 主要 kernel 类型
  B. 每个流的主要 kernel **名字**（用来认领"这个流在干什么"）
  C. 两个流之间的重叠（用于判断某条流是否被隐藏在主计算后面）
  D. 主流上的空洞（>--gap-ms），以及每个空洞里其它流是否在忙

用法：
  python3 tools/ced_prof_streams.py task_time.csv --label "基线 P (ms=1)"
  python3 tools/ced_prof_streams.py task_time.csv --label X \
      --compute-stream 47 --other allreduce --other-stream 10
"""

from __future__ import annotations

import argparse
import collections
import csv


def load(path: str):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
        if not (r.get("task_start(us)") or "").strip():
            continue
        try:
            r["_s"] = int(r["stream_id"])
            r["_a"] = float(r["task_start(us)"])
            r["_d"] = float(r["task_time(us)"])
        except (TypeError, ValueError):
            continue
        r["_e"] = r["_a"] + r["_d"]
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_time_csv")
    ap.add_argument("--label", default="")
    ap.add_argument("--top", type=int, default=6, help="表 B 每流显示几个 kernel 名")
    ap.add_argument("--compute-stream", type=int, default=None,
                    help="主流 stream id（表 D 用它算空洞）")
    ap.add_argument("--other", default="allreduce",
                    help="表 C 用来做重叠判定的 kernel 名子串（小写匹配）")
    ap.add_argument("--other-stream", type=int, default=0, help="0=不限流")
    ap.add_argument("--gap-ms", type=float, default=5.0)
    args = ap.parse_args()

    rows = load(args.task_time_csv)
    if not rows:
        print("没有可解析的 task")
        return 1
    t0 = min(r["_a"] for r in rows)
    t1 = max(r["_e"] for r in rows)
    win = (t1 - t0) / 1e6

    by = collections.defaultdict(list)
    for r in rows:
        by[r["_s"]].append(r)

    print("== %s" % (args.label or args.task_time_csv))
    print("   窗口 %.1f s   任务数 %d   流数 %d" % (win, len(rows), len(by)))

    print("\n[A] 按流汇总")
    print("   %7s %9s %10s %8s  %s" % ("stream", "tasks", "busy_s", "duty", "top types"))
    for s, rs in sorted(by.items(), key=lambda kv: -sum(x["_d"] for x in kv[1])):
        busy = sum(x["_d"] for x in rs) / 1e6
        types = collections.Counter(x["kernel_type"] for x in rs)
        print("   %7d %9d %10.2f %7.1f%%  %s" % (
            s, len(rs), busy, 100 * busy / win,
            ",".join("%s:%d" % (k, v) for k, v in types.most_common(3))))

    print("\n[B] 每流的 kernel 名（前 %d）" % args.top)
    for s in sorted(by):
        names = collections.Counter(x["kernel_name"] for x in by[s])
        print("   流 %3d (%6d): %s" % (
            s, len(by[s]),
            ", ".join("%s:%d" % (k, v) for k, v in names.most_common(args.top) if k != "N/A")[:150]))

    if args.compute_stream is not None:
        bin_us = 1000.0
        nb = int((t1 - t0) / bin_us) + 1

        def mask(pred):
            m = bytearray(nb)
            for r in rows:
                if not pred(r):
                    continue
                a = int((r["_a"] - t0) / bin_us)
                b = int((r["_e"] - t0) / bin_us)
                for i in range(max(0, a), min(nb - 1, b) + 1):
                    m[i] = 1
            return m

        comp = mask(lambda r: r["_s"] == args.compute_stream
                    and r["kernel_type"].startswith(("AI_", "MIX_")))
        oth = mask(lambda r: args.other.lower() in r["kernel_name"].lower()
                   and (args.other_stream == 0 or r["_s"] == args.other_stream))
        both = sum(1 for i in range(nb) if comp[i] and oth[i])
        cs, os_ = sum(comp), sum(oth)

        print("\n[C] 重叠：compute(流%d) vs '%s'" % (args.compute_stream, args.other))
        print("   compute 忙 %.1f s (%.1f%%)   other 忙 %.1f s (%.1f%%)   重叠 %.1f s" % (
            cs * bin_us / 1e6, 100 * cs / nb, os_ * bin_us / 1e6, 100 * os_ / nb, both * bin_us / 1e6))
        print("   other 被 compute 覆盖 %.1f%%  → 越接近 100%% 说明越被隐藏" % (
            100 * both / max(os_, 1)))

        gaps, run = [], None
        for i, v in enumerate(comp):
            if not v and run is None:
                run = i
            elif v and run is not None:
                gaps.append((run, i)); run = None
        if run is not None:
            gaps.append((run, nb))
        gaps = [g for g in gaps if (g[1] - g[0]) * bin_us / 1000 > args.gap_ms]
        gaps.sort(key=lambda g: -(g[1] - g[0]))
        print("\n[D] compute 流上的空洞 (>%.0f ms)：共 %d 段，合计 %.1f s" % (
            args.gap_ms, len(gaps), sum(b - a for a, b in gaps) * bin_us / 1e6))
        for a, b in gaps[:8]:
            ov = sum(1 for i in range(a, b) if oth[i])
            print("     %8.1f ms @%7.2fs   other 占 %3.0f%%" % (
                (b - a) * bin_us / 1000, a * bin_us / 1e6, 100 * ov / max(b - a, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
