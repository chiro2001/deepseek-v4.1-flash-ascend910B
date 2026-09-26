#!/usr/bin/env python3
"""找出「只在失败请求窗口里出现」的日志模板（CED D 偶发故障定位）。

做法：把一份 serve.log 按 request id 切成时间窗（每个请求从它第一行日志到下一条
request 的第一行之间），把行内容归一化成模板（去掉时间戳、request id、数字、
pid/rank），再统计每个模板出现在哪些请求里。最后输出：

  * 只出现在失败请求集合里的模板（噪声候选：该窗口多出来的行）；
  * 只出现在通过请求集合里的模板（缺失候选：失败请求少做的步骤）。

这不是根因证明，只是把"人肉翻 40 万行日志"变成一次可复核的差集。

用法：
    python3 tools/ced_log_template_diff.py --log <serve.log> \
        --fail 087326c3,1edbcdc8,<...> [--pass <...>] [--min-count 1]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
REQ_RE = re.compile(r"chatcmpl-[0-9a-fA-F-]{8,}")
RANK_RE = re.compile(r"\(Worker_TP\d+_EP\d+ pid=\d+\)|\(EngineCore pid=\d+\)|\(APIServer pid=\d+\)")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def template(line: str) -> str:
    text = TS_RE.sub("<TS>", line)
    text = REQ_RE.sub("<REQ>", text)
    text = RANK_RE.sub("<WHO>", text)
    text = NUM_RE.sub("<N>", text)
    return text.strip()[:180]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--fail", required=True, help="逗号分隔的失败请求 id 前缀（8 位足够）")
    ap.add_argument("--pass", dest="pass_ids", default="",
                    help="逗号分隔的通过请求 id 前缀；留空则用所有出现在日志里、"
                         "且不在 --fail 里的请求")
    ap.add_argument("--min-count", type=int, default=1, help="模板至少要出现这么多次才报告")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    fail_ids = [x.strip() for x in args.fail.split(",") if x.strip()]
    pass_ids = [x.strip() for x in args.pass_ids.split(",") if x.strip()]

    # 按 request id 首次出现的位置切窗
    windows: list[tuple[str, list[str]]] = []
    current_id, current_lines = None, []
    order: list[str] = []
    with open(args.log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = REQ_RE.search(line)
            if match:
                rid = match.group(0)[len("chatcmpl-"):][:8]
                if rid != current_id:
                    if current_id is not None:
                        windows.append((current_id, current_lines))
                    current_id, current_lines = rid, []
                    if rid not in order:
                        order.append(rid)
            current_lines.append(line)
    if current_id is not None:
        windows.append((current_id, current_lines))

    def is_fail(rid: str) -> bool:
        return any(rid.startswith(prefix) for prefix in fail_ids)

    def is_pass(rid: str) -> bool:
        if pass_ids:
            return any(rid.startswith(prefix) for prefix in pass_ids)
        return not is_fail(rid)

    fail_windows = [w for w in windows if is_fail(w[0])]
    pass_windows = [w for w in windows if is_pass(w[0])]
    print(f"窗口总数={len(windows)}  失败窗口={len(fail_windows)}  通过窗口={len(pass_windows)}")
    if not fail_windows or not pass_windows:
        print("失败或通过窗口为空，无法做差集", file=sys.stderr)
        return 2

    def counts(selected):
        table: dict[str, int] = defaultdict(int)
        for _rid, lines in selected:
            seen = set()
            for line in lines:
                seen.add(template(line))
            for tpl in seen:
                table[tpl] += 1
        return table

    fail_counts = counts(fail_windows)
    pass_counts = counts(pass_windows)

    only_fail = [(t, c) for t, c in fail_counts.items()
                 if t not in pass_counts and c >= args.min_count]
    only_pass = [(t, c) for t, c in pass_counts.items()
                 if t not in fail_counts and c >= args.min_count]
    only_fail.sort(key=lambda x: -x[1])
    only_pass.sort(key=lambda x: -x[1])

    print(f"\n=== 只在失败窗口出现的模板（{len(only_fail)} 个）===")
    for tpl, count in only_fail[: args.top]:
        print(f"  [{count}/{len(fail_windows)}] {tpl}")
    print(f"\n=== 只在通过窗口出现的模板（{len(only_pass)} 个）===")
    for tpl, count in only_pass[: args.top]:
        print(f"  [{count}/{len(pass_windows)}] {tpl}")

    print("\n=== 说明 ===")
    print("上面两类都是**线索而不是结论**：窗口切分按 request id 首现位置，")
    print("跨请求的公共行会落在先出现的那个窗口里；需要结合时间戳复核。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
