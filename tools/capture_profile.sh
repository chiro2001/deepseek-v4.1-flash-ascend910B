#!/usr/bin/env bash
# 采一段可分析的 decode-profile：/start_profile → 打几只请求 → /stop_profile。
#
# 为什么要有这个脚本：本轮所有"独占贡献 / 真空闲 / 空闲归因"分析都要求
# **profile 覆盖的是稳态 decode 步**（不是一个混了 prefill 的长窗口）。
# 手工敲容易漏掉 start/stop，或者采样期间的并发与要复现的口径不一致。
#
# 用法:
#   bash tools/capture_profile.sh <port> <tag> [concurrency] [prompt_tokens] [max_tokens] [seconds]
# 例:
#   bash tools/capture_profile.sh 19210 armF_n8 8 1024 256 40
#
# 依赖：run 目录由服务端 $OUT 决定（默认 ~/cedpd-repo/results/<RUN_ID>），
#       /start_profile 落盘位置会在最后打印（服务日志里带 prof 目录）。
set -uo pipefail

PORT=${1:?port}
TAG=${2:?tag}
CONC=${3:-8}
PROMPT_TOK=${4:-1024}
MAX_TOK=${5:-256}
SECONDS_RUN=${6:-40}

HOST=http://127.0.0.1:${PORT}
OUT_DIR=${PROFILE_DIR:-$HOME/cedpd-repo/results/profile_${TAG}_$(date +%m%d_%H%M%S)}

say() { echo "[cap-profile] $*"; }

code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -XPOST "$HOST/start_profile")
if [ "$code" != "200" ]; then
  say "⛔ /start_profile 返回 $code（需要服务以 PROFILE=1 启动）"
  exit 1
fi
say "profile 已开始（$TAG，并发 $CONC，prompt $PROMPT_TOK，max_tokens $MAX_TOK）"

python3 "$HOME/cedpd-repo/tools/bench_concurrency.py" \
  --base-url "$HOST" --concurrency "$CONC" \
  --prompt-tokens "$PROMPT_TOK" --output-tokens "$MAX_TOK" --repeats 1 2>&1 | tail -14

say "跑满 ${SECONDS_RUN}s 稳态 decode…"
sleep "$SECONDS_RUN"

code=$(curl -s -o /dev/null -w '%{http_code}' -m 120 -XPOST "$HOST/stop_profile")
say "/stop_profile 返回 $code"

say "落盘目录（服务侧 \$OUT/prof）："
latest=$(ls -dt "$HOME"/cedpd-repo/results/*/prof 2>/dev/null | head -1)
say "  最近一个有 prof 的 run: ${latest:-未找到}"
[ -n "${latest:-}" ] && find "$latest" -maxdepth 3 -name 'op_summary*.csv' 2>/dev/null | head -3
exit 0
