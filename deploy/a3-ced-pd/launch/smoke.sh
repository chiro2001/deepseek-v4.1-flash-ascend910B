#!/usr/bin/env bash
# 144K 四针冒烟：正确性最小判据。
#
# 判据不是"HTTP 200" —— 而是四针答案**精确命中**：
#   A → ZQ7K-3341   B → VX2M-8890   C → HT4P-5527   D → RB9N-6014
# 这四条是 needle 探针注入的唯一事实，与 SPEC=0 交付口径逐字节相同。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
. "$HERE/_common.sh"

CTX=${CTX:-144000}
OUT=${OUT:-$PKG/results/smoke_144k_$(date +%Y%m%d_%H%M%S).json}
CORPUS=${CORPUS:-$PKG/data/hongloumeng.txt}

[ -f "$CORPUS" ] || { echo "缺语料 $CORPUS（可用 tools/fetch_corpus.sh 下载，或 --corpus 换一份）" >&2; exit 2; }

say "144K 四针  proxy=:${PROXY_PORT:-18992}  tokenize=:${PD_PREFILL_PORT}"
python3 "$PKG/tools/ced_pd_acceptance.py" \
  --base-url "http://127.0.0.1:${PROXY_PORT:-18992}" \
  --tokenize-url "http://127.0.0.1:${PD_PREFILL_PORT}" \
  --model "$SERVED_NAME" \
  --corpus "$CORPUS" \
  --mode needle --context-tokens "$CTX" \
  --out "$OUT"
rc=$?
say "退出码 = $rc（0 = 四针全过） 证据：$OUT"
exit $rc
