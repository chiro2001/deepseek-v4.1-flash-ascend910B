#!/usr/bin/env bash
# 起 CED 的 P（prefill）：chip 0–7，只跑 layer 0–19 + layer-20 全局源。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
. "$HERE/_common.sh"

export RUN_ID=${RUN_ID:-ced_p_$(date +%Y%m%d_%H%M%S)}
export NAME=${NAME:-dsv41-ced-p-$RUN_ID}
export PORT=$PD_PREFILL_PORT
export KV_PORT=$PD_PREFILL_KV_PORT
export DEVS=$PD_PREFILL_DEVS

# P 侧：**不开**推测解码（架构性不可行 —— DSpark 取目标层 37/38/39 的残差，
# 而 P 在第 20 层就 break）。P 的响应被代理丢弃，只有它写出的 KV 有用。
export SPEC=0 DRAFT_GRAPH=0
export STATIC_KERNEL=${STATIC_KERNEL:-0}   # P 侧未单变量验过，保守取 0

say "P  role=prefill devs='$DEVS' port=$PORT kv_port=$KV_PORT patch_mode=$PATCH_MODE"
cd "$PKG"
exec bash scripts/serve_a3_ced_pd.sh prefill
