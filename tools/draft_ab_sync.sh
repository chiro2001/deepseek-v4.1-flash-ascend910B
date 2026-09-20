#!/usr/bin/env bash
# 本地 <-> A3-node1 的**小文件**同步（禁止大文件走 ssh）。用法：
#   bash tools/draft_ab_sync.sh up    # 本地 -> A3-node1:~/projects/dsv41-release/lite-runs/draft-ab/
#   bash tools/draft_ab_sync.sh down  # A3-node1 -> 本地
# 只同步 lite-runs/draft-ab/ 下的文本产物；>1 MB 的文件请走 COS。
set -euo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCAL="$PKG/lite-runs/draft-ab"
TOOLS="$PKG/tools"
REMOTE='user@A3-node1:~/projects/dsv41-release/lite-runs/draft-ab'
DIR=${1:-up}

mkdir -p "$LOCAL"
case "$DIR" in
  up)
    tar czf - -C "$LOCAL" . | ssh A3-node1 'mkdir -p ~/projects/dsv41-release/lite-runs/draft-ab && tar xzf - -C ~/projects/dsv41-release/lite-runs/draft-ab'
    tar czf - -C "$TOOLS" draft_ab_launch.sh draft_ab_run.sh draft_ab_sync.sh \
      | ssh A3-node1 'mkdir -p ~/projects/dsv41-release/tools && tar xzf - -C ~/projects/dsv41-release/tools'
    echo "[sync] up: $(find "$LOCAL" -type f | wc -l) files (+3 tools)"
    ;;
  down)
    ssh A3-node1 'cd ~/projects/dsv41-release/lite-runs/draft-ab 2>/dev/null && tar czf - .' | tar xzf - -C "$LOCAL"
    echo "[sync] down: $(find "$LOCAL" -type f | wc -l) files"
    ;;
  *) echo "用法: $0 up|down" >&2; exit 2 ;;
esac
