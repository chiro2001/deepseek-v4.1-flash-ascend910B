#!/usr/bin/env bash
# a3-22 的 /home 偶发 chdir/路径查找失败（瞬时），这里在本地反复重试后再执行 arm。
# 用法：bash <绝对路径>/ced_knifeedge_arm_retry.sh <TAG> <C> <BRIDGE_N> <UNIFORM_N> <STOP_MAX>
set -uo pipefail
PKG=${PKG:-/home/l00886679/projects/dsv41-ced-singlechip/pkg-swa-clip}
for attempt in $(seq 1 40); do
  if cd "$PKG" 2>/dev/null && [ -f tools/ced_knifeedge_arm.sh ]; then
    echo "[retry] cd+脚本可见（第 $attempt 次尝试）"
    exec bash tools/ced_knifeedge_arm.sh "$@"
  fi
  sleep 3
done
echo "[retry][FAIL] 40 次仍无法进入 $PKG" >&2
exit 9
