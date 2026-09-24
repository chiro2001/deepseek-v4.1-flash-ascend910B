#!/usr/bin/env bash
# a3-22 tiny 1+1 的 D 角色启动器（单卡，CED decode，图模式）。
#
# 与 8+8 真权重线对齐的 D 臂：GRAPH=1 EAGER=0 + prompt-tail eager
# + metadata inline + MULTISTREAM=0 DSA_OVERLAP=0；只读探针全开
# （块形态 / SWA 裁剪 / 层快照位置可选）。
#
# 安全：chip0/chip1 一律拒绝；卡被别人占用时必须显式 ALLOW_BUSY=1，
# 调用方要先自己通过 AICore/HBM 安全门。
set -euo pipefail

PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL=${MODEL:-$HOME/projects/dsv41-ced-singlechip/model-tiny}
DEVS=${DEVS:-5}
PORT=${PORT:-18961}
KV_PORT=${KV_PORT:-19061}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

for dev in $DEVS; do
  case "$dev" in
    0|1) echo "[tiny-d][FAIL] A3-22 chip0/chip1 已预留，不能使用（DEVS=$DEVS）" >&2; exit 2 ;;
    ''|*[!0-9]*) echo "[tiny-d][FAIL] DEVS 含非法项：'$dev'" >&2; exit 2 ;;
  esac
done

[ -f "$MODEL/config.json" ] || { echo "[tiny-d][FAIL] 缺少模型：$MODEL/config.json" >&2; exit 2; }
[ -f "$PKG/scripts/serve_a3.sh" ] || { echo "[tiny-d][FAIL] 缺少包：$PKG" >&2; exit 2; }

cd "$PKG"
export MODEL DEVS PORT
export NAME=${NAME:-dsv41-ced-tiny-swa-clip-d-$STAMP}
export RUN_ID=${RUN_ID:-ced_tiny_swa_clip_d_$STAMP}
export SERVED_NAME=${SERVED_NAME:-deepseek-v41-ced-tiny}
export TP=1 DP=1 PATCH_MODE=mount
export ENGRAM=0 ENGRAM_DEVICE_INDEX=0 VISION=0 CPU_BIND=0
export SPEC=0 NSPEC=0 DRAFT_GRAPH=0 PREFIX=0
export KV_DTYPE=bfloat16 LOAD_FORMAT=dummy QUANTIZATION=none SEED=0 DROPCACHE=0
export STATIC_KERNEL=0 NPUGRAPH_EX=1
export MAX_LEN=${MAX_LEN:-8192} MAX_SEQS=${MAX_SEQS:-4} BAT_TOKENS=${BAT_TOKENS:-1024}
export GPU_UTIL=${GPU_UTIL:-0.5} KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-1073741824}
export BLOCK=${BLOCK:-128}

# 图模式 D 臂（与 8+8 一致）
export GRAPH=1 EAGER=0
export V41_CED_ROLE=decode
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1
export V41_CED_METADATA_INLINE=1
export MULTISTREAM=0 DSA_OVERLAP=0

# 只读探针 + 修复开关
export V41_CED_BLOCK_TRACE=${V41_CED_BLOCK_TRACE:-1}
export V41_CED_SWA_TRACE=${V41_CED_SWA_TRACE:-1}
export V41_CED_SWA_CLIP=${V41_CED_SWA_CLIP:-1}
export V41_CED_LAYER_SNAPSHOT_POS=${V41_CED_LAYER_SNAPSHOT_POS:-}
export V41_CED_LAYER_SNAPSHOT_DIR=${V41_CED_LAYER_SNAPSHOT_DIR:-}
export V41_CED_LAYER_SNAPSHOT_LAYERS=${V41_CED_LAYER_SNAPSHOT_LAYERS:-0,1,2,19,20,38,39}
export V41_CED_SNAPSHOT_POS=${V41_CED_SNAPSHOT_POS:-}
export V41_CED_SNAPSHOT_DIR=${V41_CED_SNAPSHOT_DIR:-}

_kv_config=$(printf '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' "$KV_PORT")
export KV_ARGS_EXTRA="--kv-transfer-config $_kv_config"

echo "[tiny-d] pkg=$PKG devs='$DEVS' port=$PORT kv_port=$KV_PORT name=$NAME"
echo "[tiny-d] max_len=$MAX_LEN kv_mem=$KV_CACHE_MEMORY_BYTES graph=$GRAPH eager=$EAGER clip=$V41_CED_SWA_CLIP"
exec bash scripts/serve_a3.sh
