#!/usr/bin/env bash
# =============================================================================
# check_publish_links.sh —— **发布仓的日志索引不许有死链**（机械门）。
#
# 为什么要它：2026-09-22 15:5x–16:2x 连续踩了**三次**同一类坑：
#   ① 主代理把 README 里 001/002/003 的「待落盘 / 进行中」改成真实链接，
#      却忘了把这四份文件加进 `prepare_publish.sh` 的 MAP
#      ⇒ **发布仓立刻出现 4 条死链**（工作区完全看不出来，因为工作区里文件都在）；
#   ② 同一次，064/065 两份新日志也漏了 MAP；
#   ③ 更早还漏过 004–024 一整批（那次是 20 处死链）。
#
# ⇒ 判据：**在发布仓里**，`a2/logs/README.md` 的每个 `](NNN-*.md)` 链接都必须有对应文件。
#    （反过来也查：发布仓里有日志文件却没被索引 ⇒ 只警告不失败。）
#
# 用法：
#   bash a2/scripts/check_publish_links.sh            # 检查（默认仓库 = ../dsv41-release）
#   REL=/path/to/release bash a2/scripts/check_publish_links.sh
# 退出码：0 = 无死链；2 = 有死链（会逐条打印）。
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2=$(cd "$HERE/.." && pwd)
REL=${REL:-$A2/../dsv41-release}

IDX=$REL/a2/logs/README.md
[ -f "$IDX" ] || { echo "[links] 找不到发布仓索引：$IDX" >&2; exit 64; }

echo "[links] 检查发布仓：$REL/a2/logs/"

REFS=$(grep -oE '\]\([0-9]{3}-[a-z0-9-]+\.md\)' "$IDX" | sed 's/](\(.*\))/\1/' | sort -u)
total=0; miss=0
while IFS= read -r r; do
  [ -z "$r" ] && continue
  total=$((total + 1))
  if [ ! -f "$REL/a2/logs/$r" ]; then
    echo "  ⛔ 死链: logs/$r"
    miss=$((miss + 1))
  fi
done <<< "$REFS"
echo "[links] 索引引用 $total 个，死链 $miss"

# 反向：发布仓里有日志却没被索引（只警告 —— 可能是尚未登记的中间产物）
un=0
for f in "$REL"/a2/logs/[0-9][0-9][0-9]-*.md; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  grep -q "($b)" "$IDX" || { echo "  ⚠ 未被索引: logs/$b"; un=$((un + 1)); }
done
echo "[links] 未被索引 $un（仅警告）"

if [ "$miss" -gt 0 ]; then
  echo "[links] ⛔ 失败：先把这些文件加进 a2/scripts/prepare_publish.sh 的 MAP，再重跑 PUBLISH=1"
  exit 2
fi
echo "[links] ✓ 通过（发布仓日志索引无死链）"
