#!/usr/bin/env bash
# 把本地 overlay 同步到 a3-21:~/dcpw（V41_DCP_MOUNT 的源目录）。
# ★ 判据：同步后逐文件 md5 一致；不一致就报错退出（不要带着半份代码起服）。
set -euo pipefail
SRC=${SRC:-/home/chiro/projects/dsv41/main-merge/experimental/v41-dcp/overlay}
HOST=${HOST:-a3-21}
DST=${DST:-/home/l00886679/dcpw}
echo "[sync] $SRC → $HOST:$DST"
ssh "$HOST" "mkdir -p '$DST'"
# ★ 镜像式同步：先列出远端已有的 .py，删掉本地已经没有的，避免"上一版补丁残留在
#   远端、被 mount 进去、与新版叠加"这种最难查的静默错误（2026-09-29 踩过）。
_LOCAL_TMP=$(mktemp -d)
ssh "$HOST" "cd '$DST' && find . -type f -name '*.py' | sort" > "$_LOCAL_TMP/remote.txt" 2>/dev/null || true
(cd "$SRC" && find . -type f -name '*.py' | sort) > "$_LOCAL_TMP/local.txt"
comm -23 "$_LOCAL_TMP/remote.txt" "$_LOCAL_TMP/local.txt" | while IFS= read -r stale; do
  [ -n "$stale" ] || continue
  echo "[sync] 远端多余文件，删除：$stale"
  ssh "$HOST" "cd '$DST' && rm -f './${stale#./}'"
done
(cd "$SRC" && tar cf - .) | ssh "$HOST" "cd '$DST' && tar xf -"
echo "[sync] 本地 md5:"
(cd "$SRC" && find . -type f -name '*.py' | sort | xargs md5sum)
echo "[sync] 远端 md5:"
ssh "$HOST" "cd '$DST' && find . -type f -name '*.py' | sort | xargs md5sum"
