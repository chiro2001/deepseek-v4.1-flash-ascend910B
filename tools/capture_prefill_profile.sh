#!/usr/bin/env bash
# 采一段**prefill**（不是 decode）的 profile：/start_profile → 发长 prompt → /stop_profile。
#
# 为什么单独采：`prof_conc.sh` 用的是 1K prompt + 160 输出，采到的主体是 decode。
# 但 ③ prefill 是交付的弱项之一（8 die 上 7,200–7,350 tok/s），
# 而"prefill 到底受限于哪个硬件单元"**没有**现成的数据 —— 只有 decode 的。
# 只有拿到 prefill 的资源账，才能判断 8 die 上还有没有杠杆。
#
# 用法: bash tools/capture_prefill_profile.sh <port> <并发> <prompt_tokens> [输出tokens]
# 例:   bash tools/capture_prefill_profile.sh 19210 2 32768 8
set -uo pipefail

PORT=${1:-19210}
CONC=${2:-2}
PROMPT_TOK=${3:-32768}
OUT_TOK=${4:-8}
REPO=$HOME/cedpd-repo
HOST=http://127.0.0.1:$PORT

say() { echo "[cap-prefill] $*"; }

code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -XPOST "$HOST/start_profile")
if [ "$code" != "200" ]; then
  say "⛔ /start_profile 返回 $code（服务需以 PROFILE=1 启动）"
  exit 1
fi
say "profile 已开始；发 $CONC 条 × $PROMPT_TOK tok（输出仅 $OUT_TOK）"

python3 "$REPO/tools/bench_concurrency.py" \
  --base-url "$HOST" --concurrency "$CONC" \
  --prompt-tokens "$PROMPT_TOK" --output-tokens "$OUT_TOK" \
  --repeats 1 --spec-tokens 5 2>&1 | tail -8

code=$(curl -s -o /dev/null -w '%{http_code}' -m 600 -XPOST "$HOST/stop_profile")
say "/stop_profile 返回 $code"
latest=$(ls -dt "$REPO"/results/*/prof 2>/dev/null | head -1)
say "最近含 prof 的 run: ${latest:-未找到}"
[ -n "${latest:-}" ] && find "$latest" -maxdepth 3 -name 'op_summary*.csv' 2>/dev/null | head -3
exit 0
