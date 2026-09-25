#!/usr/bin/env python3
"""小请求标定：打印每个 n 的 dump 条数、min/max 与游标推进量。

用于（a）验证 c(n) = ceil((n-1)/128)、（b）给序列相位加一个"桥"请求。
只发 HTTP + 读 dump，不写服务端状态。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_walker(path: Path):
    spec = importlib.util.spec_from_file_location("ced_knifeedge_walk", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Args:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", required=True, help="逗号分隔的 prompt 长度列表")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18962/v1/completions")
    parser.add_argument("--model", default="deepseek-v41-ced-tiny")
    parser.add_argument("--walker", default=None, help="ced_knifeedge_walk.py 路径")
    args = parser.parse_args()

    walker_path = Path(args.walker) if args.walker else Path(__file__).with_name(
        "ced_knifeedge_walk.py"
    )
    walker = load_walker(walker_path)
    runner_args = _Args()
    runner_args.url = args.url
    runner_args.model = args.model
    runner_args.out = args.out
    runner_args.dump_dir = args.dump_dir
    runner_args.dump_timeout = 90.0
    runner_args.top_logprobs = 5
    runner_args.max_tokens = 1
    runner_args.timeout = 600.0
    runner = walker.Runner(runner_args)

    print(f"{'n':>7s} {'count':>5s} {'min':>7s} {'max':>7s} {'c(n)':>5s} {'step':>6s}")
    previous_max = None
    for raw in args.ns.split(","):
        length = int(raw)
        record = runner.send(length, "cprobe")
        dump = record.get("dump")
        if not dump:
            print(f"{length:7d}  no dump")
            continue
        step = "" if previous_max is None else dump["max"] - previous_max
        print(
            f"{length:7d} {dump['count']:5d} {dump['min']:7d} {dump['max']:7d} "
            f"{walker.c_of_n(length):5d} {str(step):>6s}"
        )
        previous_max = dump["max"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
