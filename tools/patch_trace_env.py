#!/usr/bin/env python3
"""在 shadow 包的 serve_a2.sh 里加两条只读探针 env 透传（fail-closed）。

用法：python3 patch_trace_env.py <serve_a2.sh 路径>

锚点：`-e V41_CED_CAPTURE_DECODE=...` 那一行；必须恰好命中一次，否则不写盘。
"""

from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = "-e V41_CED_CAPTURE_DECODE="
# 需要存在的 env 透传行（幂等：缺哪条补哪条；全部存在则跳过）。
WANTED = (
    '  -e V41_CED_BLOCK_TRACE="${V41_CED_BLOCK_TRACE:-0}" \\\n',
    '  -e V41_ENGRAM_HIST_TRACE_POS="${V41_ENGRAM_HIST_TRACE_POS:-}" \\\n',
    '  -e V41_CED_SWA_TRACE="${V41_CED_SWA_TRACE:-0}" \\\n',
)


def main() -> int:
    path = Path(sys.argv[1])
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if ANCHOR in line]
    if len(hits) != 1:
        print(f"[patch-trace-env] 锚点命中 {len(hits)} 次，fail-closed 不写盘", file=sys.stderr)
        return 2
    text = "".join(lines)
    missing = [item for item in WANTED if item.strip() not in text]
    if not missing:
        print("[patch-trace-env] 三条 env 透传都在，跳过")
        return 0
    index = hits[0]
    for offset, item in enumerate(missing):
        lines.insert(index + 1 + offset, item)
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[patch-trace-env] 已在第 {index + 1} 行后插入 {len(missing)} 条 env 透传")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
