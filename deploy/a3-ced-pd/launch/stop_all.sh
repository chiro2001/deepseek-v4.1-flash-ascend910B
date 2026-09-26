#!/usr/bin/env bash
# 停掉本形态的容器。**按前缀匹配，且排除 proxy**（proxy 名字也以 dsv41-ced 开头）。
set -uo pipefail
DOCKER=${DOCKER:-docker}
echo "[a3-ced-pd] 当前相关容器："
$DOCKER ps -a --format '{{.Names}}\t{{.Status}}' | grep -E '^dsv41-(ced|pfx|base|pdv41)' | grep -v proxy || true
$DOCKER ps -a --format '{{.Names}}' | grep -E '^dsv41-(ced|pfx|base|pdv41)' | grep -v proxy \
  | xargs -r $DOCKER rm -f >/dev/null 2>&1 || true
echo "[a3-ced-pd] 已停（proxy 保留，需要时手工 docker rm -f）"
