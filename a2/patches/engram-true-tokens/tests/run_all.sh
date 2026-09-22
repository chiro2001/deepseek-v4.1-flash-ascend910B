#!/usr/bin/env bash
# 本地语义测试（不需要 NPU、不需要 numba、不写发布树）
# 用法：bash a2/agents/Engram_exactfix/tests/run_all.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rc=0
for t in test_true_tokens_repair.py test_repair_plan.py; do
    echo "=============================================================="
    echo "== $t"
    echo "=============================================================="
    ( cd "$HERE" && python3 "$t" ) || rc=1
done
echo "=============================================================="
[ "$rc" = 0 ] && echo "✓ run_all：全部通过" || echo "✗ run_all：有失败项"
exit "$rc"
