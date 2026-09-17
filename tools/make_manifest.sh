#!/usr/bin/env bash
# 重新生成 MANIFEST.sha256（发布前跑一次）
#
# 口径：**只收录 git 追踪的文件**（= 仓库里真正会发布的内容）。
#   这样 .gitignore 里排除掉的东西（运行时产物、受版权保护的语料等）
#   自动不会进 MANIFEST，干净 clone 后 sha256sum -c 必然可过。
#   不在 git 仓库里时退回用 find，并沿用下面的排除规则。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

if git -C "$PKG" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SRC=$(git -C "$PKG" ls-files -z \
        | tr '\0' '\n' \
        | grep -v -e '^MANIFEST\.sha256$' -e '^$' \
        | LC_ALL=C sort)
  MODE="git ls-files"
else
  SRC=$(find . -type f \
        -not -path './results/*' \
        -not -path './cache/*' \
        -not -path './.git/*' \
        -not -name 'MANIFEST.sha256' \
        -not -name '.last_run_id' \
        -not -name 'data/dihuo.txt' \
        -not -path '*/__pycache__/*' \
        -not -name '*.pyc' \
        | LC_ALL=C sort)
  MODE="find"
fi

printf '%s\n' "$SRC" | sed 's|^|./|' | xargs -r sha256sum > MANIFEST.sha256

echo "[manifest] 来源=$MODE  文件数=$(wc -l < MANIFEST.sha256)  -> $PKG/MANIFEST.sha256"
