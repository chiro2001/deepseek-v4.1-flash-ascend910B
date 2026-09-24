#!/usr/bin/env bash
# 逐请求跑同一请求 N 次；每次结束后把 cache 快照目录改名保留。
#
# 用法：
#   RD=<run 目录> TEMPLATE=<请求 json> COUNT=8 bash ced_layer_trace_sequence.sh
#
# 可选环境变量（用于在同一次 D 实例里跑多组实验，避免互相覆盖）：
#   OUTDIR     证据落盘目录（默认 $RD/probe/layer_trace_seq）
#   TAG_PREFIX 文件名前缀（默认 lt，形如 lt1/lt2/...）
#   SNAP_PREFIX 快照改名前缀（默认 snap_after_，形如 snap_after_1）
#   URL        目标 chat/completions（默认本机 proxy 18992）
set -uo pipefail

RD=${RD:?需要设置 RD（run 目录）}
TEMPLATE=${TEMPLATE:?需要设置 TEMPLATE（请求 JSON）}
COUNT=${COUNT:-8}
URL=${URL:-http://127.0.0.1:18992/v1/chat/completions}
OUT=${OUTDIR:-"$RD/probe/layer_trace_seq"}
TAG_PREFIX=${TAG_PREFIX:-lt}
SNAP_PREFIX=${SNAP_PREFIX:-snap_after_}

mkdir -p "$OUT"
for i in $(seq 1 "$COUNT"); do
  echo "[layer-seq] 第 $i/$COUNT 次 tag=${TAG_PREFIX}${i} $(date -Is)"
  python3 "$RD/tools/ced_seq_probe.py" \
    --template "$TEMPLATE" --url "$URL" \
    --outdir "$OUT" --tag "${TAG_PREFIX}${i}" --count 1 --timeout 1800
  if [ -d "$RD/snap" ]; then
    mv "$RD/snap" "$RD/${SNAP_PREFIX}$i"
  fi
done
echo "[layer-seq] 完成 $(date -Is)"
