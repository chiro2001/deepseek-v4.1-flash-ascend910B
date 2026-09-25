#!/usr/bin/env bash
# 在 D 上采一段 profiler：start → 一次 144K decode → stop。
# 用法：TAG=graph144k CONTEXT=144000 TOKENS=128 bash prof_capture.sh
set -uo pipefail
PKG=/home/l00886679/tmp/20260926/dspark
TAG=${TAG:-prof}
CONTEXT=${CONTEXT:-144000}
TOKENS=${TOKENS:-128}
D_URL=${D_URL:-http://127.0.0.1:18991}
cd "$PKG" || exit 1
echo "[prof] $(date -Is) TAG=$TAG ctx=$CONTEXT tokens=$TOKENS"
curl -sS -XPOST "$D_URL/start_profile" -o /dev/null -w "  start_profile=%{http_code}\n" || exit 1
env PYTHONUNBUFFERED=1 python3 tools/ced_pd_bench.py \
  --base-url http://127.0.0.1:18992 --tokenize-url http://127.0.0.1:18990 \
  --model deepseek-v41-ced-pd --corpus data/hongloumeng.txt \
  --contexts "$CONTEXT" --max-tokens "$TOKENS" --ignore-eos --repeat 1 \
  --out "results/prof_${TAG}.json" 2>&1 | tail -3
sleep 3
curl -sS -XPOST "$D_URL/stop_profile" -o /dev/null -w "  stop_profile=%{http_code}\n"
# 等落盘
for i in $(seq 1 60); do
  n=$(docker exec dsv41-ced-d4b bash -lc "ls -d /opt/dsv41/results/*/prof/*_ascend_pt 2>/dev/null | wc -l")
  echo "  prof dirs=$n (${i}0s)"
  [ "$n" -ge 1 ] && break
  sleep 10
done
docker exec dsv41-ced-d4b bash -lc 'ls -la /opt/dsv41/results/*/prof/ | head -30'
echo "[prof] DONE $(date -Is)"
