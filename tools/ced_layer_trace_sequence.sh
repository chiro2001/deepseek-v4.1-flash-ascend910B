#!/usr/bin/env bash
# 逐请求跑同一 1M 请求 N 次；每次结束后把 cache 快照目录改名保留。
# 用法：RD=<run 目录> TEMPLATE=<请求 json> COUNT=8 bash ced_layer_trace_sequence.sh
set -uo pipefail

RD=${RD:?需要设置 RD（run 目录）}
TEMPLATE=${TEMPLATE:?需要设置 TEMPLATE（请求 JSON）}
COUNT=${COUNT:-8}
URL=${URL:-http://127.0.0.1:18992/v1/chat/completions}
OUT="$RD/probe/layer_trace_seq"

mkdir -p "$OUT"
for i in $(seq 1 "$COUNT"); do
  echo "[layer-seq] 第 $i/$COUNT 次 $(date -Is)"
  python3 "$RD/tools/ced_seq_probe.py" \
    --template "$TEMPLATE" --url "$URL" \
    --outdir "$OUT" --tag "lt$i" --count 1 --timeout 1800
  if [ -d "$RD/snap" ]; then
    mv "$RD/snap" "$RD/snap_after_$i"
  fi
done
echo "[layer-seq] 完成 $(date -Is)"
