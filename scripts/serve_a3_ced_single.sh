#!/usr/bin/env bash
# A3-22 tiny V4.1 CED-PD: one chip per role, dummy weights, BF16 KV.
# The tiny config keeps 40 layers and the 12 production-shaped cache groups.
# It tests replay geometry and numerical consistency, not real-weight quality.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
role=${1:-}
case "$role" in
  prefill)
    DEVS=${DEVS:-6}
    PORT=${PORT:-18960}
    KV_PORT=${KV_PORT:-19060}
    KV_ROLE=kv_producer
    CED_ROLE=prefill
    ;;
  decode)
    DEVS=${DEVS:-7}
    PORT=${PORT:-18961}
    KV_PORT=${KV_PORT:-19061}
    KV_ROLE=kv_consumer
    CED_ROLE=decode
    ;;
  baseline)
    DEVS=${DEVS:-7}
    PORT=${PORT:-18963}
    KV_PORT=
    KV_ROLE=
    CED_ROLE=
    ;;
  *) echo "用法：$0 prefill|decode|baseline" >&2; exit 2 ;;
esac

# 可用卡：0/1（用户 2026-09-25 明确划给本实验的 1+1）与 6/7（原空闲卡）。
# 其余卡一律拒绝，避免踩到同事的进程。
[ "$DEVS" = 0 ] || [ "$DEVS" = 1 ] || [ "$DEVS" = 6 ] || [ "$DEVS" = 7 ] || {
  echo "本单卡实验仅使用 A3-22 的 Phy-ID 0/1/6/7；收到 DEVS=$DEVS" >&2
  exit 2
}

MODEL=${MODEL:-$HOME/projects/dsv41-ced-singlechip/model-tiny}
if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f "$MODEL/config.json" ]; then
  echo "缺少 tiny V4.1 config：$MODEL/config.json" >&2
  exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-ced_single_${role}_${stamp}}
NAME=${NAME:-dsv41-ced-single-${role}-${stamp}}
if [ -n "${CED_SNAPSHOT_POS:-}" ]; then
  export V41_CED_SNAPSHOT_POS=$CED_SNAPSHOT_POS
  export V41_CED_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/snapshots"
fi
if [ -n "${CED_H20_SNAPSHOT_POS:-}" ]; then
  export V41_CED_H20_SNAPSHOT_POS=$CED_H20_SNAPSHOT_POS
  export V41_CED_H20_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/h20_snapshots"
fi
if [ -n "${CED_LAYER_SNAPSHOT_POS:-}" ]; then
  export V41_CED_LAYER_SNAPSHOT_POS=$CED_LAYER_SNAPSHOT_POS
  export V41_CED_LAYER_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/layer_snapshots"
fi
if [ "${CED_CAPTURE_DECODE:-0}" = 1 ]; then
  export V41_CED_CAPTURE_DECODE=1
fi
export MODEL DEVS PORT KV_PORT RUN_ID NAME
export SERVED_NAME=${SERVED_NAME:-deepseek-v41-ced-tiny}
export TP=1 DP=1 V41_CED_ROLE=$CED_ROLE
if [ -n "$KV_ROLE" ]; then
  KV_CONFIG=$(printf '{"kv_connector":"MooncakeHybridConnector","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' "$KV_ROLE" "$KV_PORT")
  export KV_ARGS_EXTRA="--kv-transfer-config $KV_CONFIG"
else
  export KV_ARGS_EXTRA=
fi
export LOAD_FORMAT=dummy QUANTIZATION=none SEED=0
export ENGRAM=0 ENGRAM_DEVICE_INDEX=0 VISION=0 SPEC=0 DRAFT_GRAPH=0
export KV_DTYPE=bfloat16 CPU_BIND=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export DROPCACHE=0
export MAX_LEN=${MAX_LEN:-8192} MAX_SEQS=${MAX_SEQS:-4} BAT_TOKENS=${BAT_TOKENS:-1024}
export GPU_UTIL=${GPU_UTIL:-0.5} KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-1073741824}
export PREFIX=0 WAIT_READY=${WAIT_READY:-0}

cd "$HERE/.."
exec bash scripts/serve_a3.sh
