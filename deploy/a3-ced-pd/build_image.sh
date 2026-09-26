#!/usr/bin/env bash
# 构建 A3 CED-PD 工作镜像：官方基础镜像 + 我们的 1 层。
#
# 用法（在 a3-21 或任何有基础镜像的机器上）：
#   bash deploy/a3-ced-pd/build_image.sh
#   BASE_IMAGE=<你的基底> TAG=<你的tag> bash deploy/a3-ced-pd/build_image.sh
#
# 做三件事：
#   ① build_payload.sh 从仓库组装 payload（payload 清单见 PAYLOAD.md）
#   ② docker build
#   ③ 构建期冒烟 + 产出 BUILD_INFO.txt（在镜像内 /opt/dsv41/BUILD_INFO.txt）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
BASE_IMAGE=${BASE_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
TAG=${TAG:-local/dsv41-a3-ced-pd:v1}
DOCKER=${DOCKER:-docker}
say() { printf '\033[1m[build]\033[0m %s\n' "$*"; }

say "仓库   = $PKG"
say "基底   = $BASE_IMAGE"
say "目标   = $TAG"

$DOCKER image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
  || { echo "[build] FATAL: 本地没有基底镜像 $BASE_IMAGE" >&2; exit 20; }

# ① payload（先删旧树，避免上一次的残留混进镜像）
say "① 组装 payload"
bash "$HERE/build_payload.sh" "$HERE/payload"

# 源码 revision（可复算：别人能知道这层是在哪个 commit 上切的）
REV=$(cd "$PKG" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(cd "$PKG" && git status --porcelain 2>/dev/null | wc -l)
REV="${REV}+${DIRTY}dirty"
say "repo_rev = $REV"

# ② 构建
say "② docker build"
$DOCKER build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "REPO_REV=$REV" \
  -t "$TAG" "$HERE"

# ③ 冒烟 + 指纹
say "③ 构建后核对"
$DOCKER run --rm --entrypoint bash "$TAG" -lc 'cat /opt/dsv41/BUILD_INFO.txt'
say "镜像 = $TAG"
say "下一步：bash deploy/a3-ced-pd/verify_consistency.sh   # 与仓库逐文件比对"
