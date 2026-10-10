#!/usr/bin/env bash
set -e
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
stack_source_root=${STACK_SRC_ROOT:-/work/src}
if [ -n "${STACK_EXPECTED_CHIPS:-}" ]; then
  [ "$ASCEND_RT_VISIBLE_DEVICES" = "$STACK_EXPECTED_CHIPS" ] || exit 2
else
  case "$ASCEND_RT_VISIBLE_DEVICES" in 4|8,9,10,11,12,13,14,15) ;; *) exit 2 ;; esac
fi
export PYTHONPATH="$stack_source_root/stack:$stack_source_root/operator:$stack_source_root/operator/baseline:$stack_source_root/up950${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$stack_source_root/runtime/tmp"
mkdir -p "$TMPDIR"
stack_script=$1
shift
exec python -u "$stack_source_root/stack/$stack_script" "$@"
