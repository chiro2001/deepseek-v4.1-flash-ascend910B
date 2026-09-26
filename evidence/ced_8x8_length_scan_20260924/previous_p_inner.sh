#!/usr/bin/env bash
set -uo pipefail
cd /workspace
export MODEL="/home/l00886679/models/out/v41-flat-verify3" TP=8 DP=1 PORT=18790 SERVED_NAME="deepseek-v41-ced-pd"
export MAX_LEN=147456 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_DTYPE=bfloat16 GRAPH=1 EAGER=0 PREFIX=0 SPEC=0 SP_TOKENS=7
if [ "0" = "1" ]; then export SPEC_EAGER=0; else export SPEC_EAGER=1; fi
export ENGRAM=1 ENGRAM_STORAGE=int8 VISION=1
export NPUGRAPH_EX=1 STATIC_KERNEL=0 CPU_BIND=0
export MULTISTREAM=1 DSA_OVERLAP=1 FUSED_MC2=0 MC2=0 MC2_HIER=0 REDUCE_SAMPLE=0
export LOADER_MT=1 LAZY=1
export V41_KV_TIER=off
export V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner
export CAPTURE_SIZES="1,2,3,4,8,12,16,20,24,32"
export ASCEND_MAX_OP_CACHE_SIZE=-1
# [PROFILE] 透传 profiler 开关；PROFILE_DIR 指向本次 run 的结果目录（宿主可见），
# 这样 /stop_profile 一落盘就能直接分析，不用再 docker cp。
export PROFILE=0
export PROFILE_DIR=/opt/dsv41/results/ced_p_8b59d03/prof
export KV_ARGS_EXTRA="${KV_ARGS_EXTRA:-}"
# [OPS-SWITCHES] 三个排障开关，**默认全关**（发布口径）。
# 排查长上下文/精度问题时把它们打开很有用：
#   VLLM_SERVER_DEV_MODE=1  → 额外挂出 12 个运维端点（/reset_prefix_cache /pause
#                             /resume /sleep /wake_up /collective_rpc /server_info …），
#                             vLLM 自己会打一条 "Development endpoints are enabled!" 安全告警。
#   LOG_REQUESTS=1          → 把请求级 I/O 写进 serve.log（长度上限 MAX_LOG_LEN）。
#   PROBE=1                 → 稀疏状态插针，见上。
export VLLM_SERVER_DEV_MODE=0
export V41_PROBE_DIR=/opt/dsv41/probe
export LOG_REQUESTS=0
export MAX_LOG_LEN=4096

if [ "1" = "1" ]; then
  export EXTRA='--tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --tool-call-parser=deepseek_v41 --enable-auto-tool-choice --default-chat-template-kwargs={"enable_thinking":false}'
else
  export EXTRA='--tokenizer-mode=deepseek_v4 --default-chat-template-kwargs={"enable_thinking":false}'
fi
if [ -n "" ]; then
  export V41_ENGRAM_WITH_DUMMY=1 V41_DUMMY_WO_A_FIX=1
fi
if [ "0" = "1" ]; then
  export DSPARK_DRAFT_METADATA_MODE=sync
  # ★ 必须同时设这个，否则 draft 图静默无 attention（见上面 DRAFT_GRAPH 注释）。
  export DSPARK_GRAPH_CAPTURE_METADATA=1
fi
echo "[a2] run_id=ced_p_8b59d03 port=18790 static=0 sptok=7 capture_sizes=1,2,3,4,8,12,16,20,24,32 mseqs=4 prefix=0 moeag=1 local_owner=fast pgo=0 load_format=${LOAD_FORMAT:-real} draft_graph=0 moe_zero=0 moe_nf=0"
md5sum /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hbm.py \
       /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py 2>/dev/null
exec bash /opt/dsv41/scripts/serve_v2.sh
