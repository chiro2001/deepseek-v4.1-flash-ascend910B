#!/usr/bin/env bash
# Run inside the authorized tiny container after host chip6 preflight.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
qkv_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
qkv_runtime=${QKV_LANE_RUNTIME_ROOT:?Set a private runtime directory}
qkv_tag=${QKV_LANE_RUN_TAG:?Set a new run tag}
test "${ASCEND_RT_VISIBLE_DEVICES:?}" = 6
test "${HCCL_DETERMINISTIC:?}" = strict
export QKV_LANE_PHYSICAL_CHIP=6
export QKV_LANE_RUNTIME_NAMESPACE=host
export PYTHONPATH="$qkv_source${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$qkv_runtime/tmp/$qkv_tag"
export ASCEND_CACHE_PATH="$qkv_runtime/cache/$qkv_tag/ascend"
export TRITON_CACHE_DIR="$qkv_runtime/cache/$qkv_tag/triton"
export XDG_CACHE_HOME="$qkv_runtime/cache/$qkv_tag/xdg"
mkdir -p "$TMPDIR" "$ASCEND_CACHE_PATH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
exec python "$qkv_source/bench_qkv_tiny.py" "$@"
