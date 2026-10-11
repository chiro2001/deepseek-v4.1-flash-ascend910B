#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
native_copy_gate_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
native_copy_runtime_root=${NATIVE_COPY_RUNTIME_ROOT:?Set a new private runtime directory}
export ASCEND_CACHE_PATH="$native_copy_runtime_root/cache/ascend"
export TRITON_CACHE_DIR="$native_copy_runtime_root/cache/triton"
export TMPDIR="$native_copy_runtime_root/tmp"
mkdir -p "$ASCEND_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"
exec python -u "$native_copy_gate_root/gate_sdk.py" "$@"
