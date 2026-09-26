#!/usr/bin/env bash
# 起 CED 的 D（decode）：chip 8–15，128-token 有界重放 + 全 40 层 + DSpark。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
. "$HERE/_common.sh"

export RUN_ID=${RUN_ID:-ced_d_$(date +%Y%m%d_%H%M%S)}
export NAME=${NAME:-dsv41-ced-d-$RUN_ID}
export PORT=$PD_DECODE_PORT
export KV_PORT=$PD_DECODE_KV_PORT
export DEVS=$PD_DECODE_DEVS

# ---- ★ DSpark（D 侧）----
export SPEC=${SPEC:-1}
export SP_TOKENS=${SP_TOKENS:-7}
export DRAFT_GRAPH=${DRAFT_GRAPH:-1}
# 显式放行 D 侧开 SPEC（默认拒绝；P/D 是两个独立引擎实例，per-instance 生效）
export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-1}

# ---- ★ 图模式的两个前提（launcher 已 fail-closed，这里显式写出来）----
export CED_EXPERIMENTAL_GRAPH=${CED_EXPERIMENTAL_GRAPH:-1}
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-1}
export V41_CED_SWA_CLIP=${V41_CED_SWA_CLIP:-1}

# ---- 本形态实测有效的两个（见 docs/CED-PD-STATIC-KERNEL-20260926.md）----
export STATIC_KERNEL=${STATIC_KERNEL:-1}     # −4.4 ms/step（−9.6%）
export V41_SLOT_MAP_FUSED=${V41_SLOT_MAP_FUSED:-on}

say "D  role=decode  devs='$DEVS' port=$PORT kv_port=$KV_PORT patch_mode=$PATCH_MODE"
say "   spec=$SPEC sp=$SP_TOKENS draft_graph=$DRAFT_GRAPH static_kernel=$STATIC_KERNEL"
cd "$PKG"
exec bash scripts/serve_a3_ced_pd.sh decode
