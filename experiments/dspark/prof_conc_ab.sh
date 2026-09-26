#!/usr/bin/env bash
# 并发 N 的 profiler 采集（**带热身**）。
#
# 顺序（热身必须在 start_profile **之前**）：
#   1) 预热 prompt 校准（tokenize）
#   2) 热身 W 轮：同并发、同 prompt、短输出，不计入
#   3) 等空闲
#   4) start_profile
#   5) 正式采集（同并发、满输出）
#   6) stop_profile、等落盘
#
# ★ 不接 `| tail`：那会把"热身N/N、start_profile、stop_profile"这些
#   **时序关键行**截掉，事后无法证明采集窗口是干净的（已踩过一次）。
#
# 用法：TAG=conc4_sk1 bash prof_conc_ab.sh
set -uo pipefail
PKG=/home/l00886679/tmp/20260926/dspark
TAG=${TAG:-conc4}
D_URL=${D_URL:-http://127.0.0.1:18991}
TOKENS=${TOKENS:-256}
CONC=${CONC:-4}
PROMPT=${PROMPT:-2048}
WARM=${WARM:-2}
WARM_TOK=${WARM_TOK:-64}
cd "$PKG" || exit 1
echo "[conc-prof] $(date -Is) TAG=$TAG conc=$CONC prompt=$PROMPT tokens=$TOKENS warm=$WARM"

# 热身 + start/stop_profile 都由 python 侧按其严格顺序执行（--profile-url）
env PYTHONUNBUFFERED=1 python3 prof_capture_conc.py \
  --base-url http://127.0.0.1:18992 --tokenize-url http://127.0.0.1:18990 \
  --model deepseek-v41-ced-pd --corpus data/hongloumeng.txt \
  --concurrency "$CONC" --prompt-tokens "$PROMPT" --max-tokens "$TOKENS" \
  --warmup-rounds "$WARM" --warmup-tokens "$WARM_TOK" \
  --profile-url "$D_URL" \
  --out "results/prof_${TAG}.json" 2>&1

for i in $(seq 1 60); do
  n=$(sudo -n ls -d /opt/dsv41/results/*/prof/*_ascend_pt 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] && { echo "  prof dirs=$n (${i}0s)"; break; }
  sleep 10
done
echo "[conc-prof] DONE $(date -Is)"
