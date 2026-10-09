#!/usr/bin/env bash
# Run inside this experiment's existing container; preserve CANN's ACL path.
set -e
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=${GOAL20_CHIP:-4}
case "$ASCEND_RT_VISIBLE_DEVICES" in 4|6) ;; *) echo 'Unsupported isolated chip' >&2; exit 2 ;; esac
export PYTHONPATH="/work/operator_opt:/work/operator_opt/baseline${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=/work/operator_opt/runtime/tmp
mkdir -p "$TMPDIR"
script_name=$1
shift
exec python -u "/work/operator_opt/$script_name" "$@"
