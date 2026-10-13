#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
delivery_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
delivery_source=${STACK_SRC_ROOT:?explicit frozen source required}
delivery_tmp=${DELIVERY_TMPDIR:?explicit task temp directory required}
mkdir -p "$delivery_tmp"
export TMPDIR="$delivery_tmp"
export PYTHONPATH="$delivery_source/stack:$delivery_source/operator:$delivery_source/operator/baseline:$delivery_source/up950:$(dirname "$delivery_here")${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${DELIVERY_DCP_OVERLAY:-}" ]; then
  export PYTHONPATH="$delivery_here/bootstrap:$PYTHONPATH"
fi
exec python3 -u -m delivery_matrix.service "$@"
