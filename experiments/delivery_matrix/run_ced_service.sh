#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
ced_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ced_source=${STACK_SRC_ROOT:?}
mkdir -p "${DELIVERY_TMPDIR:?}"
export TMPDIR="$DELIVERY_TMPDIR"
export PYTHONPATH="$ced_here/ced_bootstrap:$ced_source/stack:$ced_source/operator:$ced_source/operator/baseline:$ced_source/up950:$(dirname "$ced_here")${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -u -m delivery_matrix.ced_service "$@"
