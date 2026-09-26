#!/usr/bin/env bash
# 官方 load_balance_proxy：客户端只连它（PD 分离对客户端透明）。
#
# ⚠️ proxy 的名字**必须以 dsv41-ced 开头**才被 stop_all.sh 覆盖，
#    但 stop_all.sh 又必须把它排除（否则每次重启都拆掉代理重新建）。
#    当前约定：proxy 名字含 "proxy"，stop_all.sh 用 `grep -v proxy` 排除。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
. "$HERE/_common.sh"

export PROXY_PORT=${PROXY_PORT:-18992}
export HOST=${HOST:-127.0.0.1}
export PREFILL_HOST=${PREFILL_HOST:-127.0.0.1}
export PREFILL_PORT=${PREFILL_PORT:-$PD_PREFILL_PORT}
export DECODE_HOST=${DECODE_HOST:-127.0.0.1}
export DECODE_PORT=${DECODE_PORT:-$PD_DECODE_PORT}
export NAME=${NAME:-dsv41-ced-pd-proxy-$PROXY_PORT}

say "proxy name=$NAME port=$PROXY_PORT  P=$PREFILL_HOST:$PREFILL_PORT  D=$DECODE_HOST:$DECODE_PORT"
cd "$PKG"
exec bash scripts/serve_a3_pd_proxy.sh
