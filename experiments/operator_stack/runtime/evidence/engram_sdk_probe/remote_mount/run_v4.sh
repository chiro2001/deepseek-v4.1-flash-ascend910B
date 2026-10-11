#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_CACHE_PATH=/work/engram_sdk_probe_20261011_c4_v1/cache/ascend
export TRITON_CACHE_DIR=/work/engram_sdk_probe_20261011_c4_v1/cache/triton
export TMPDIR=/work/engram_sdk_probe_20261011_c4_v1/tmp
mkdir -p "$ASCEND_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"
cd /work/engram_sdk_probe_20261011_c4_v1
exec python source_v4/probe_sdk.py "$@"
