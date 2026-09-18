#!/usr/bin/env bash
# DeepSeek-V4.1 W4A8 统一实验起服脚本（8 卡形态）
# 开关：MODEL TP DP PORT SERVED_NAME MAX_LEN MAX_SEQS BAT_TOKENS GPU_UTIL BLOCK KV_DTYPE
#   GRAPH EAGER PREFIX SPEC SP_TOKENS ENGRAM ENGRAM_STORAGE
#   NPUGRAPH_EX STATIC_KERNEL CPU_BIND MULTISTREAM DSA_OVERLAP MC2 MC2_HIER
#   FUSED_MC2 MC2_ALG REDUCE_SAMPLE CAPTURE_SIZES
#   LOADER_MT LAZY VISION CHAT_TEMPLATE PROFILE PROFILE_DIR HCCL_BUFFSIZE EXTRA
set -uo pipefail
# A2 适配：原版硬编码 A3-node1 的路径（只用于 PROFILE_DIR 默认值），改为从包位置推导。
H=${H:-$HOME}; P=${P:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL=${MODEL:-$H/models/out/v41-w4a8-dspark}
TP=${TP:-8}; DP=${DP:-1}; PORT=${PORT:-8000}; SERVED_NAME=${SERVED_NAME:-deepseek-v41}
MAX_LEN=${MAX_LEN:-8192}; MAX_SEQS=${MAX_SEQS:-32}; BAT_TOKENS=${BAT_TOKENS:-8192}
GPU_UTIL=${GPU_UTIL:-0.90}; BLOCK=${BLOCK:-128}; KV_DTYPE=${KV_DTYPE:-bfloat16}
GRAPH=${GRAPH:-1}; EAGER=${EAGER:-0}; PREFIX=${PREFIX:-0}
SPEC=${SPEC:-1}; SP_TOKENS=${SP_TOKENS:-7}; SPEC_EAGER=${SPEC_EAGER:-1}
ENGRAM=${ENGRAM:-0}; ENGRAM_STORAGE=${ENGRAM_STORAGE:-int8}
NPUGRAPH_EX=${NPUGRAPH_EX:-1}; STATIC_KERNEL=${STATIC_KERNEL:-0}
CPU_BIND=${CPU_BIND:-1}; MULTISTREAM=${MULTISTREAM:-0}; DSA_OVERLAP=${DSA_OVERLAP:-1}
MC2=${MC2:-0}; MC2_HIER=${MC2_HIER:-0}
FUSED_MC2=${FUSED_MC2:-0}; MC2_ALG=${MC2_ALG:-}; REDUCE_SAMPLE=${REDUCE_SAMPLE:-0}
WEIGHT_NZ=${WEIGHT_NZ:-}
FORCE_EPLB=${FORCE_EPLB:-0}; DSA_CP=${DSA_CP:-0}; ENGRAM_HOST_RESTORE=${ENGRAM_HOST_RESTORE:-0}
CAPTURE_SIZES=${CAPTURE_SIZES:-}
LOADER_MT=${LOADER_MT:-1}; LAZY=${LAZY:-1}
VISION=${VISION:-0}; CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
PROFILE=${PROFILE:-0}; PROFILE_DIR=${PROFILE_DIR:-$P/logs/prof}
EXTRA=${EXTRA:-}

b() { if [ "$1" = "1" ]; then echo true; else echo false; fi; }
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

EXTRA_KEYS=""
[ "$ENGRAM" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"engram_storage\":\"$ENGRAM_STORAGE\""
[ "$FUSED_MC2" != "0" ] && EXTRA_KEYS="$EXTRA_KEYS,\"enable_fused_mc2\":$FUSED_MC2"
[ -n "$MC2_ALG" ] && EXTRA_KEYS="$EXTRA_KEYS,\"mc2_comm_alg\":\"$MC2_ALG\""
[ "$REDUCE_SAMPLE" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"enable_reduce_sample\":true"
[ -n "$WEIGHT_NZ" ] && EXTRA_KEYS="$EXTRA_KEYS,\"weight_nz_mode\":$WEIGHT_NZ"
[ "$FORCE_EPLB" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"enable_force_eplb\":true"
[ "$DSA_CP" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"enable_dsa_cp\":true"
[ "${MIX_PLACEMENT:-0}" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"mix_placement\":true"
[ -n "${DRAFT_WINDOW:-}" ] && EXTRA_KEYS="$EXTRA_KEYS,\"draft_window_size\":$DRAFT_WINDOW"
[ "${SFA_C8:-0}" = "1" ] && EXTRA_KEYS="$EXTRA_KEYS,\"enable_sparse_sfa_c8\":true"

AC=$(printf '{"enable_engram":%s,"enable_cpu_binding":%s,"ascend_compilation_config":{"enable_npugraph_ex":%s,"enable_static_kernel":%s},"multistream_overlap_shared_expert":%s,"multistream_dsv4_dsa_overlap":%s,"enable_prefill_mc2":%s,"enable_mc2_hierarchy_comm":%s%s}' \
  "$(b "$ENGRAM")" "$(b "$CPU_BIND")" "$(b "$NPUGRAPH_EX")" "$(b "$STATIC_KERNEL")" \
  "$(b "$MULTISTREAM")" "$(b "$DSA_OVERLAP")" "$(b "$MC2")" "$(b "$MC2_HIER")" "$EXTRA_KEYS")

ARGS=(serve "$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED_NAME"
  --tensor-parallel-size "$TP" --enable-expert-parallel
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype "$KV_DTYPE" --quantization ascend
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BAT_TOKENS"
  --block-size "$BLOCK" --gpu-memory-utilization "$GPU_UTIL"
  --additional-config "$AC")
[ "$DP" -gt 1 ] && ARGS+=(--data-parallel-size "$DP" --data-parallel-size-local "$DP")
if [ "$GRAPH" = "1" ]; then
  if [ -n "$CAPTURE_SIZES" ]; then
    ARGS+=(--compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [$CAPTURE_SIZES]}")
  else
    ARGS+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
  fi
else
  [ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
fi
if [ "$PREFIX" = "1" ]; then ARGS+=(--enable-prefix-caching); else ARGS+=(--no-enable-prefix-caching); fi
if [ "$SPEC" = "1" ]; then
  if [ "$SPEC_EAGER" = "1" ]; then SE=true; else SE=false; fi
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SP_TOKENS,\"enforce_eager\":$SE}")
fi
# [LOADER-MT] dummy load 不接受 model-loader-extra-config（vllm 直接 raise ValueError），
# 且 dummy 模式下本来也没有权重要并发加载，所以此时跳过。
if [ "$LOADER_MT" = "1" ] && [ "${LOAD_FORMAT:-}" != "dummy" ]; then
  ARGS+=(--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}')
fi
[ "$LAZY" = "1" ] && ARGS+=(--safetensors-load-strategy lazy)
# [LOAD_FORMAT] "dummy" 时不读权重，只按 checkpoint 的 shape/dtype 建模型。
# 注意：dummy 下 Engram 的 host 路径会被 model.py:654 主动跳过
# （engram_history 保持 None），所以 **Engram 读取无法用 dummy 测**。
[ -n "${LOAD_FORMAT:-}" ] && ARGS+=(--load-format "$LOAD_FORMAT")
if [ "$VISION" = "1" ]; then ARGS+=(--limit-mm-per-prompt '{"image": 1}'); else ARGS+=(--limit-mm-per-prompt '{"image": 0}'); fi
[ -n "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template "$CHAT_TEMPLATE")
# [LOG_REQUESTS] 端到端请求日志（进 serve.log，带长度上限避免炸日志）
[ "${LOG_REQUESTS:-0}" = "1" ] && ARGS+=(--enable-log-requests --max-log-len "${MAX_LOG_LEN:-4096}")

if [ "$PROFILE" = "1" ]; then mkdir -p "$PROFILE_DIR"; ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false}"); fi
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

echo "[serve-v2] model=$(basename "$MODEL") tp=$TP dp=$DP port=$PORT graph=$GRAPH prefix=$PREFIX spec=$SPEC(sp=$SP_TOKENS) kv=$KV_DTYPE vision=$VISION"
echo "[serve-v2] npugraph_ex=$NPUGRAPH_EX static=$STATIC_KERNEL cpu_bind=$CPU_BIND multistream=$MULTISTREAM dsa=$DSA_OVERLAP fused_mc2=$FUSED_MC2 mc2_alg=$MC2_ALG reduce_sample=$REDUCE_SAMPLE loader_mt=$LOADER_MT lazy=$LAZY max_len=$MAX_LEN bat=$BAT_TOKENS seqs=$MAX_SEQS"
echo "[serve-v2] cmd: vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
