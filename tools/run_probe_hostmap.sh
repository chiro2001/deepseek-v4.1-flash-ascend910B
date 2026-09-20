#!/usr/bin/env bash
# Run the Engram host-mapping capability probe inside the server's image.
#
#   IMAGE=<server image> DEV=<free chip> bash run_probe_hostmap.sh
#
# Defaults target A2: the A2 image built by build_image.sh and chip 0.
# Nothing here is A2-specific -- the same command answers the question on A3.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=${IMAGE:-dsv41-a2:v8}
DEV=${DEV:-0}
DOCKER=${DOCKER:-docker}

echo "[run-probe] image=$IMAGE chip=$DEV"
echo "[run-probe] （只读探测：起一个临时容器，跑完即删；不占卡超过几秒）"

exec $DOCKER run --rm -u 0 --privileged --ipc=host --network=host \
  -v /home:/home \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -e ASCEND_RT_VISIBLE_DEVICES="$DEV" \
  -v "$HERE/probe_a2_hostmap.py:/probe_a2_hostmap.py:ro" \
  -v "$HERE/engram_device_index.py:/engram_device_index.py:ro" \
  "$IMAGE" bash -lc '
    set -u
    # probe imports engram_device_index from the script directory
    mkdir -p /probe && cp /probe_a2_hostmap.py /engram_device_index.py /probe/
    cd /probe
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    python3 -u probe_a2_hostmap.py --chip '"$DEV"'
    rc=$?
    echo
    case "$rc" in
      0) echo "[run-probe] RESULT: SUPPORTED      —— A2 可启用 device-index" ;;
      3) echo "[run-probe] RESULT: NOT SUPPORTED  —— 保持 ENGRAM_DEVICE_INDEX=0（host 路径）" ;;
      *) echo "[run-probe] RESULT: PROBE ERROR (rc=$rc) —— 把上面输出发回来" ;;
    esac
    exit $rc
  '
