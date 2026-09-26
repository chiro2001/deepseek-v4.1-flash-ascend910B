#!/usr/bin/env bash
# A3 单机 1P1D 的本地负载均衡代理入口。
# 只负责启动官方示例 proxy；P/D 服务由 scripts/serve_a3_pd.sh 启动。
set -euo pipefail

IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
NAME=${NAME:-dsv41-pd-v41-proxy}
HOST=${HOST:-127.0.0.1}
PROXY_PORT=${PROXY_PORT:-18552}
PREFILL_HOST=${PREFILL_HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-18550}
DECODE_HOST=${DECODE_HOST:-127.0.0.1}
DECODE_PORT=${DECODE_PORT:-18551}
DRY_RUN=${DRY_RUN:-0}

args=(run -d --name "$NAME" --network host --entrypoint python3 "$IMAGE"
  /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
  --host "$HOST" --port "$PROXY_PORT"
  --prefiller-hosts "$PREFILL_HOST" --prefiller-ports "$PREFILL_PORT"
  --decoder-hosts "$DECODE_HOST" --decoder-ports "$DECODE_PORT")

printf '[a3-pd-proxy] docker'
printf ' %q' "${args[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi
command -v docker >/dev/null 2>&1 || { echo "[a3-pd-proxy][FAIL] 找不到 docker" >&2; exit 2; }
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "[a3-pd-proxy][FAIL] 容器名已存在：$NAME；请换 NAME 或先按实验记录停止旧代理。" >&2
  exit 3
fi
exec docker "${args[@]}"
