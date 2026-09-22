#!/usr/bin/env python3
"""[S_graphfix] 把 `R_8card_int8/scripts/run_arm_r8.sh` 机械改造成本任务的臂运行器。

只动 6 处（全部带锚点唯一性断言；R 的文件一变就停，不做模糊匹配）：
  1. 新增 `SG`（本任务目录）；
  2. 证据目录默认落到 `/agents/S_graphfix/{out,logs}`（不写 R 的产物目录）；
  3. `R8_KV8_DIR_C/D` 允许用 `SG_PKG_RING/SG_PKG_D` 覆盖 ⇒ 换成**我的影子包**；
  4. 把 `R8_GRAPH_SAFE` 传进 serve_a2.sh（→ 容器内 `VLLM_V41_KV8_GRAPH_SAFE`）；
  5. 挂载源自检的 grep 里加 `S_graphfix/pkgs`；
  6. meta 里记 `R8_GRAPH_SAFE` 与**我的 dsa_v41.py md5**（证据链）。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPL = [
    (
        "1 新增 SG 变量",
        "ROOT=${ROOT:-$HOME/projects/dsv41-upstream-pr}\nR8=${R8:-$ROOT/agents/R_8card_int8}\n",
        "ROOT=${ROOT:-$HOME/projects/dsv41-upstream-pr}\n"
        "R8=${R8:-$ROOT/agents/R_8card_int8}\n"
        "SG=${SG:-$ROOT/agents/S_graphfix}\n",
    ),
    (
        "2 证据目录默认到 S_graphfix",
        'OUT=${OUT:-$R8/out/$TAG}\nLOGD=${LOGD:-$R8/logs}\n',
        'OUT=${OUT:-$SG/out/$TAG}\nLOGD=${LOGD:-$SG/logs}\n',
    ),
    (
        "3 影子包可覆盖 + 4 传 R8_GRAPH_SAFE",
        '  R8_INT8_TIER="$TIER" R8_DSA_SRC="${DSA_SRC:-auto}" \\\n'
        '  R8_KV8_DIR_C="$X/pkg-ring" R8_KV8_DIR_D="$X/pkg-kv8pf" \\\n',
        '  R8_INT8_TIER="$TIER" R8_DSA_SRC="${DSA_SRC:-auto}" \\\n'
        '  R8_KV8_DIR_C="${SG_PKG_RING:-$X/pkg-ring}" R8_KV8_DIR_D="${SG_PKG_D:-$X/pkg-kv8pf}" \\\n'
        '  R8_GRAPH_SAFE="${R8_GRAPH_SAFE:-0}" \\\n',
    ),
    (
        "5 挂载源自检加 S_graphfix/pkgs",
        'grep -c "R_8card_int8/patched\\|X_integrate/pkg-"',
        'grep -c "R_8card_int8/patched\\|X_integrate/pkg-\\|S_graphfix/pkgs"',
    ),
    (
        "5b SG_CMP_LEGACY / SG_TRACE_PPR 透传进容器",
        '  R8_GRAPH_SAFE="${R8_GRAPH_SAFE:-0}" \\\n',
        '  R8_GRAPH_SAFE="${R8_GRAPH_SAFE:-0}" \\\n'
        '  SG_CMP_LEGACY="${SG_CMP_LEGACY:-0}" SG_TRACE_PPR="${SG_TRACE_PPR:-0}" \\\n',
    ),
    (
        "6a meta 记 R8_GRAPH_SAFE",
        '  echo "R8_APC_TRACE=$R8_APC_TRACE R8_APC_TRACE_LIMIT=$R8_APC_TRACE_LIMIT"\n',
        '  echo "R8_APC_TRACE=$R8_APC_TRACE R8_APC_TRACE_LIMIT=$R8_APC_TRACE_LIMIT"\n'
        '  echo "R8_GRAPH_SAFE=${R8_GRAPH_SAFE:-0} SG_PKG_D=${SG_PKG_D:-（未设：用 X_integrate/pkg-kv8pf）}"\n',
    ),
    (
        "6b meta 记我的 dsa md5",
        '  ( cd "$X/pkg-kv8pf/shadow/vllm_ascend" && md5sum attention/dsa_v41.py '
        "attention/kv8_prefill_triton.py 2>/dev/null )\n",
        '  ( cd "$X/pkg-kv8pf/shadow/vllm_ascend" && md5sum attention/dsa_v41.py '
        "attention/kv8_prefill_triton.py 2>/dev/null )\n"
        '  echo "--- [S_graphfix] 本臂实际挂的 dsa_v41.py（SG_PKG_D）---"\n'
        '  ( cd "${SG_PKG_D:-$X/pkg-kv8pf}/shadow/vllm_ascend" && md5sum attention/dsa_v41.py 2>/dev/null )\n',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="R 的 run_arm_r8.sh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = Path(args.src).read_text()
    text = raw
    for name, old, new in REPL:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"[adapt_runner] 锚点 {name} 出现 {n} 次（期望 1）")
        text = text.replace(old, new, 1)
        print(f"  OK   锚点 {name}")
    for marker in ("SG_PKG_D", "R8_GRAPH_SAFE", "S_graphfix/pkgs", "SG/out/$TAG"):
        if marker not in text:
            raise SystemExit(f"[adapt_runner] 自检失败：缺 {marker}")
    Path(args.out).write_text(text)
    print(f"[adapt_runner] src md5 = {hashlib.md5(raw.encode()).hexdigest()}（{args.src}）")
    print(f"[adapt_runner] out md5 = {hashlib.md5(text.encode()).hexdigest()}（{args.out}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
