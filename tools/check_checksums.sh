#!/usr/bin/env bash
# =============================================================================
# check_checksums.sh —— 发布包校验和一致性自检（10 秒，不需要 docker）
#
#   bash tools/check_checksums.sh          # 检查，退出码 0=通过 1=不一致
#   bash tools/check_checksums.sh --emit   # 额外打印应内联进 build_image.sh 的 chk() 行
#   PKG_ROOT=/path/to/pkg bash tools/check_checksums.sh   # 换包根（容器内/部署副本用）
#
# 为什么需要：`scripts/build_image.sh` 的镜像内自检曾**硬编码**补丁 md5，
# v7→v8 改了 `engram_hbm.py`/`model.py` 忘了同步 ⇒ 用户 build image 到最后一步才
# 报 "FAIL md5 .../model.py"。本脚本把这类问题**提前到 10 秒内**、并覆盖
# 「Dockerfile 装了但 chk 没列」「chk 列了但 Dockerfile 没装」两种漏项。
#
# 判据见 tools/check_checksums.py 头部：三重信息各自唯一权威来源，其余全部推导。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT=${PKG_ROOT:-$(cd "$HERE/.." && pwd)}

PY=$(command -v python3 || true)
[ -n "$PY" ] || { echo "[chk][FAIL] 找不到 python3" >&2; exit 2; }

ARGS=()
for a in "$@"; do
  case "$a" in
    --emit) ARGS+=(--emit-chk) ;;
    *) ARGS+=("$a") ;;
  esac
done

echo "[chk] 包根: $PKG_ROOT"
"$PY" "$HERE/check_checksums.py" \
  --dockerfile "$PKG_ROOT/Dockerfile" \
  --payload    "$PKG_ROOT/patches/files" \
  --sums       "$PKG_ROOT/patches/MD5SUMS" \
  "${ARGS[@]}"
rc=$?
if [ "$rc" = "0" ]; then
  echo "[chk] 通过 ✅  （scripts/build_image.sh 起服时就是用这份载荷字节现算期望 md5，不做手工同步）"
else
  echo "[chk] 失败 ❌（见上；修完再跑）" >&2
fi
exit "$rc"
