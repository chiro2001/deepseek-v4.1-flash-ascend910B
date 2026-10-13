#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
dp_matrix_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dp_matrix_source=${STACK_SRC_ROOT:?explicit immutable best source required}
dp_matrix_generic=${DP_MATRIX_GENERIC_PACKAGE:?explicit pinned generic-v7 package required}
dp_matrix_tmp=${DP_MATRIX_TMPDIR:?explicit task TMPDIR required}
mkdir -p "$dp_matrix_tmp"
export TMPDIR="$dp_matrix_tmp"
export PYTHONPATH="$dp_matrix_source/stack:$dp_matrix_source/operator:$dp_matrix_source/operator/baseline:$dp_matrix_source/up950:$(dirname "$dp_matrix_here"):$dp_matrix_generic${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${DP_MATRIX_LAYOUT:-}" == "dp4-tp2-dcp2" ]]; then
    dp_matrix_dcp2_bootstrap=${DP_MATRIX_DCP2_BOOTSTRAP:?explicit DCP2 bootstrap required}
    test -f "$dp_matrix_dcp2_bootstrap/sitecustomize.py"
    export PYTHONPATH="$dp_matrix_dcp2_bootstrap:$PYTHONPATH"
fi
if [[ -n "${DP_MATRIX_NUMA_INTERLEAVE_NODES:-}" ]]; then
    # No fallback: a denied policy must fail before model loading, not silently
    # reproduce the NUMA-local OOM exposure under a different fingerprint.
    exec numactl --interleave="$DP_MATRIX_NUMA_INTERLEAVE_NODES" python3 -u -m dp_matrix.service "$@"
fi
exec python3 -u -m dp_matrix.service "$@"
