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

# ⚠️ 陷阱提醒（2026-09-20 踩过一次）：本脚本读的是**工作区**内容。
# 如果你只提交了**一部分**文件（其余留在工作区未提交），本脚本会把未提交文件的
# **工作区**哈希写进 MANIFEST —— 而干净 `git archive HEAD` / clone 拿到的是 HEAD 的
# 旧内容 ⇒ `sha256sum -c MANIFEST.sha256` 会报那些文件 FAILED。
# 所以：要么先把改动**全部**提交，要么确保工作区干净后再跑本脚本。
if git -C "$PKG" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  _dirty=$(git -C "$PKG" status --porcelain 2>/dev/null | grep -v '^??' | grep -v '^.. MANIFEST\.sha256$' | wc -l)
  if [ "$_dirty" -gt 0 ]; then
    echo "[manifest][WARN] 工作区有 $_dirty 个已跟踪文件处于**未提交**状态：" >&2
    git -C "$PKG" status --porcelain 2>/dev/null | grep -v '^??' | head -10 >&2
    echo "[manifest][WARN] 本 MANIFEST 记录的是**工作区**内容 ⇒ 对 HEAD 解包会不匹配。" >&2
    echo "[manifest][WARN] 若只想描述已提交的内容，请先提交全部改动，或用：" >&2
    echo "[manifest][WARN]   git ls-tree -r --name-only HEAD | while read f; do \\" >&2
    echo "[manifest][WARN]     h=\$(git cat-file blob HEAD:\$f | sha256sum | cut -d' ' -f1); \\" >&2
    echo "[manifest][WARN]     printf '%s  ./%s\\n' \"\$h\" \"\$f\"; done > MANIFEST.sha256" >&2
  fi
fi
