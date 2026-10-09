#!/usr/bin/env bash
# Isolated tiny milestone: activation/overlap baseline plus validated goal20 combo.
set -e
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH="/work/operator_opt:/work/operator_opt/baseline${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=/work/operator_opt/runtime/tmp
mkdir -p "$TMPDIR"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export V41_DUMMY_WO_A_FIX=1
export TINY_PERF_ARM=both
export TINY_PERF_HC_ENABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export OPT_ACT_ARM=overlap
export OPT_TEST_OVERLAP=1
export GOAL20_ARM=${GOAL20_ARM:-combo}
case "$GOAL20_ARM" in baseline|hcstatic|hcpost|route|gmm1|combo) ;; *) exit 2 ;; esac
export VLLM_CACHE_ROOT=/work/operator_opt/runtime/goal20_service_cache
export LOCAL_WORLD_SIZE=1
exec python -m vllm.entrypoints.openai.api_server \
  --model /model --tokenizer /model --load-format dummy --dtype bfloat16 \
  --host 0.0.0.0 --port 18971 --served-model-name dsv41-tiny-prof-20261009 \
  --tensor-parallel-size 1 --enable-expert-parallel --seed 0 \
  --worker-cls goal20_worker.Goal20Worker \
  --no-async-scheduling --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --block-size 128 --gpu-memory-utilization 0.70 --kv-cache-memory-bytes 4294967296 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --additional-config '{"enable_engram":false,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"multistream_dsv4_dsa_overlap":true}'
