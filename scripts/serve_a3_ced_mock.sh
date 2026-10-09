#!/usr/bin/env bash
# Run under ced_npu_lock.py --devs <one free chip>; no model or vLLM engine.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MOCK_DEV:?Set one free Phy-ID for mock consumption}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Run this script under ced_npu_lock.py}"
[ "$ASCEND_RT_VISIBLE_DEVICES" = "$MOCK_DEV" ] \
  || { echo "Mock device differs from the locked visibility mapping" >&2; exit 2; }
IMAGE=${IMAGE:-local/dsv41-a3-ced-pd:v3}
NAME=${NAME:-ced-dram-mock-$(date +%Y%m%d_%H%M%S)}
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Mock container name already exists: $NAME" >&2
  exit 3
fi
# Foreground docker wait keeps the lock supervisor alive until consumption
# stops; the runtime creates only a device context and a bounded host buffer.
exec docker run --rm --name "$NAME" --privileged --network host \
  --device "/dev/davinci$MOCK_DEV" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v "$PKG/tools/ced_mock_decode.py:/opt/ced_mock_decode.py:ro" \
  --entrypoint bash "$IMAGE" -lc \
  'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; exec python3 /opt/ced_mock_decode.py --device 0 --buffer-mib 64'
