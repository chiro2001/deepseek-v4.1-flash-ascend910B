#!/usr/bin/env bash
set -e
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
quality_source_root=${STACK_SRC_ROOT:?}
[ "${ASCEND_RT_VISIBLE_DEVICES:?}" = '8,9,10,11,12,13,14,15' ] || exit 2
[ "${STACK_PHYSICAL_CHIPS:?}" = "$ASCEND_RT_VISIBLE_DEVICES" ] || exit 2
export PYTHONPATH="$quality_source_root/stack:$quality_source_root/operator:$quality_source_root/operator/baseline:$quality_source_root/up950${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$quality_source_root/runtime/tmp"
mkdir -p "$TMPDIR"
quality_module=$1
shift
case "$quality_module" in
  parallel_tiny.dspark_general_quality_api_cpu.bootstrap|parallel_tiny.dspark_general_quality_launcher_cpu.evaluate_when_ready) ;;
  *) exit 2 ;;
esac
exec python -u -m "$quality_module" "$@"
