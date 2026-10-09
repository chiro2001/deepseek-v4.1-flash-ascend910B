#!/usr/bin/env bash
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export V41_DUMMY_WO_A_FIX=1
export PYTHONPATH="/work/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TINY_PERF_ARM=${TINY_PERF_ARM:-both}
export UP950_CANDIDATE=${UP950_CANDIDATE:-indexer}
export UP950_ARM=${UP950_ARM:-fused}
if [[ "$UP950_CANDIDATE" == "none" ]]; then
  tiny_worker_cls=tiny_perf_worker.TinyPerfWorker
elif [[ "$UP950_CANDIDATE" == "indexer" ]]; then
  tiny_worker_cls=lane_worker.Upstream950Worker
else
  echo "Only verified Indexer fusion is supported by the restored tiny service" >&2
  exit 2
fi
export VLLM_DISABLE_COMPILE_CACHE=1
exec python -m vllm.entrypoints.openai.api_server \
  --model /model --tokenizer /model --load-format dummy --dtype bfloat16 \
  --host 0.0.0.0 --port 18973 --served-model-name dsv41-tiny-upstream950-20261009 \
  --tensor-parallel-size 1 --enable-expert-parallel --seed 0 \
  --worker-cls "$tiny_worker_cls" \
  --no-async-scheduling --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --block-size 128 --gpu-memory-utilization 0.70 --kv-cache-memory-bytes 4294967296 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --additional-config '{"enable_engram":false,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"multistream_dsv4_dsa_overlap":false}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/work/results/service_prof","torch_profiler_with_stack":false}'
