#!/usr/bin/env bash
# Run UNDER ced_npu_lock.py: retain the locks for the entire P container life.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NAME:?Set a unique ced-dram-p-* container name}"
case "$NAME" in ced-dram-p-*) ;; *) echo "Unexpected experiment container name" >&2; exit 2 ;; esac
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Experiment container already exists: $NAME" >&2
  exit 3
fi
started=0
cleanup() {
  if [ "$started" = 1 ]; then
    docker stop --time 30 "$NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
export RUN_ID=${RUN_ID:-$NAME}
mkdir -p "$PKG/results/$RUN_ID"
started=1
bash "$PKG/scripts/serve_a3_ced_dram.sh"
while [ "$(docker inspect --format '{{.State.Running}}' "$NAME" 2>/dev/null || true)" = true ]; do
  sleep 5
done
echo "[CED-SUPERVISOR] P container stopped: $NAME"
exit 1
