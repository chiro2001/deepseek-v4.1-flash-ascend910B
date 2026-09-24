#!/usr/bin/env bash
# 在 A3-21 用带只读探针的 shadow 包起 CED P 或 D（2026-09-24 块相位定位）。
#
# 用法（在 a3-21 上）：
#   bash launch_ced_trace_a3.sh prefill
#   bash launch_ced_trace_a3.sh decode
#
# 探针：
#   V41_CED_BLOCK_TRACE=1          打印每个请求在各 KV cache group 上的物理块形态
#                                  （n/first/last/descents，绕回时 descents>0）
#   V41_ENGRAM_HIST_TRACE_POS=<pos> D 打印尾 token 的 4-gram 镜像来源
#
# 该脚本只设置环境变量并调用包内 scripts/serve_a3_pd.sh，不改任何模型逻辑。
set -euo pipefail

role=${1:-}
case "$role" in
  prefill|decode) ;;
  *) echo "用法：$0 prefill|decode" >&2; exit 2 ;;
esac

PKG=${PKG:-/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace}
MODEL=${MODEL:-/home/l00886679/models/out/v41-flat-verify3}
STAMP=${STAMP:-20260924_trace}

cd "$PKG"

export MODEL SERVED_NAME=deepseek-v41-ced-pd TP=8 DP=1
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export SPEC=0 PREFIX=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export PATCH_MODE=mount WAIT_READY=1
export V41_CED_ROLE=$role V41_CED_BLOCK_TRACE=1

if [ "$role" = "prefill" ]; then
  export NAME=${NAME:-dsv41-ced-trace-p-$STAMP} RUN_ID=${RUN_ID:-ced_trace_p_$STAMP}
  export PORT=18990 KV_PORT=19090 DEVS="0 1 2 3 4 5 6 7"
  # 与 20260924-070428 实例一致：P 保持多流打开。
  export MULTISTREAM=1 DSA_OVERLAP=1
else
  export NAME=${NAME:-dsv41-ced-trace-d-$STAMP} RUN_ID=${RUN_ID:-ced_trace_d_$STAMP}
  export PORT=18991 KV_PORT=19091 DEVS="8 9 10 11 12 13 14 15"
  # 与 metadata-inline 诊断臂一致：D 关多流、保留 FULL decode 图、prompt 尾步 eager。
  export MULTISTREAM=0 DSA_OVERLAP=0 GRAPH=1 EAGER=0
  export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_METADATA_INLINE=1
  export V41_ENGRAM_HIST_TRACE_POS=${V41_ENGRAM_HIST_TRACE_POS:-1019846}
  # 逐层数值探针（针对最终 token 步）与 cache 快照（可针对 replay 窗口内的位置）分开。
  #   TRACE_POS：逐层 digest 的位置（默认 1019846 = 最后一个 prompt token）
  #   SNAP_POS ：cache 快照的位置（默认同 TRACE_POS；设为 1019845 可抓 replay 步写入的值）
  export TRACE_POS=${TRACE_POS:-1019846}
  export SNAP_POS=${SNAP_POS:-$TRACE_POS}
  export V41_CED_LAYER_SNAPSHOT_POS=$TRACE_POS
  export V41_CED_LAYER_SNAPSHOT_LAYERS=${V41_CED_LAYER_SNAPSHOT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}
  export V41_CED_LAYER_SNAPSHOT_DIR=/opt/dsv41/results/$RUN_ID/layer_trace
  export V41_CED_SNAPSHOT_POS=$SNAP_POS
  export V41_CED_SNAPSHOT_DIR=/opt/dsv41/results/$RUN_ID/snap
fi

echo "[trace-launch] role=$role pkg=$PKG name=$NAME port=$PORT devs='$DEVS'"
echo "[trace-launch] block_trace=$V41_CED_BLOCK_TRACE engram_trace_pos=${V41_ENGRAM_HIST_TRACE_POS:-} layer_trace_pos=${V41_CED_LAYER_SNAPSHOT_POS:-}"
echo "[trace-launch] snap_pos=${V41_CED_SNAPSHOT_POS:-}"
exec bash scripts/serve_a3_pd.sh "$role"
