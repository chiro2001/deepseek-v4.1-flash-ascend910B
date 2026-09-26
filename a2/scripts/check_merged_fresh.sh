#!/usr/bin/env bash
# =============================================================================
# check_merged_fresh.sh —— ★ 检查"合并件是否已经过期"（**不占卡、不启容器**）
#
# 为什么需要它（2026-09-22 22:xx 连续踩两次）：
#   8 卡 runner 在 `TIER != B` 时会用 `patched/model_merged.py`（生产版 + KV8 的 3 处 hunk）
#   **顶替**影子包挂的那份 `models/deepseek_v41/model.py`。
#   而它是**派生件**：只要 `--prod`（或 --img/--kv8）换了，它**立刻过期**，
#   但**没有任何东西会告诉你** —— 表现形式就是"某个功能静默消失"。本轮实测两次：
#     · 第一次：合并件基于**旧 prod** ⇒ **Engram 的 `set_engram_row_tokens` 整段丢掉**（logs/081）
#     · 第二次：prod 后来被换成带 `[0]` 修复的版本 ⇒ 合并件里**又变回没有 `[0]` 的那个 bug**
#               （会让第一个请求抛 `TypeError`，logs/077 后续/`p3b`）
#
# 做法：用**同一套输入**重新合并一次到临时文件，与已安装的那份逐字节比对。
#   ★ 合并器自带自证（diff 回放镜像版必须逐字节等于 kv8 版、锚点必须唯一命中），
#     所以"重新合并"本身是可信操作，不是猜。
#
# 用法：
#   bash check_merged_fresh.sh --img <镜像原版 model.py> \
#        --kv8 <pkg-kv8pf 版> --prod <影子包 patches/files/model.py> \
#        --installed <patched/model_merged.py> [--merger <merge_model.py>]
#
# 退出码：0 = 新鲜（逐字节相同）；1 = **已过期**（打印差异行数 + 关键标志位对比）；2 = 用法/文件缺失
# =============================================================================
set -uo pipefail

IMG=""; KV8=""; PROD=""; INST=""; MERGER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --img)       IMG=${2:-}; shift 2 ;;
    --kv8)       KV8=${2:-}; shift 2 ;;
    --prod)      PROD=${2:-}; shift 2 ;;
    --installed) INST=${2:-}; shift 2 ;;
    --merger)    MERGER=${2:-}; shift 2 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "[mfresh] ✗ 未知参数：$1" >&2; exit 2 ;;
  esac
done

for pair in "img:$IMG" "kv8:$KV8" "prod:$PROD" "installed:$INST"; do
  k=${pair%%:*}; v=${pair#*:}
  [ -n "$v" ] || { echo "[mfresh] ✗ 缺 --$k" >&2; exit 2; }
  [ -f "$v" ] || { echo "[mfresh] ✗ 文件不存在（--$k）：$v" >&2; exit 2; }
done
[ -n "$MERGER" ] || { echo "[mfresh] ✗ 缺 --merger <merge_model.py>" >&2; exit 2; }
[ -f "$MERGER" ] || { echo "[mfresh] ✗ 合并器不存在：$MERGER" >&2; exit 2; }

md5f() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }

echo "=============================================================="
echo "合并件新鲜度检查（不占卡）"
printf "  img       %s  %s\n" "$(md5f "$IMG")"  "$IMG"
printf "  kv8       %s  %s\n" "$(md5f "$KV8")"  "$KV8"
printf "  prod      %s  %s\n" "$(md5f "$PROD")" "$PROD"
printf "  installed %s  %s\n" "$(md5f "$INST")" "$INST"
echo "=============================================================="

TMPD=$(mktemp -d "${TMPDIR:-$HOME/tmp}/mfresh.XXXXXX") || exit 2
trap 'rm -rf "$TMPD"' EXIT
FRESH="$TMPD/model_merged.fresh.py"

if ! python3 "$MERGER" --img "$IMG" --kv8 "$KV8" --prod "$PROD" --out "$FRESH" >"$TMPD/merge.log" 2>&1; then
  echo "[mfresh] ✗ 重新合并失败（合并器自己的门没通过）—— 下面它的输出：" >&2
  sed 's/^/    /' "$TMPD/merge.log" >&2
  exit 2
fi

want=$(md5f "$FRESH"); got=$(md5f "$INST")
if [ "$want" = "$got" ]; then
  echo "[mfresh] ✓ 新鲜：合并件 == 用当前输入重新合并的结果（md5 $got）"
  echo "[mfresh]   上面合并器自证已通过（diff 回放镜像版逐字节相同 + 锚点唯一命中）"
  exit 0
fi

echo "[mfresh] ⛔ **合并件已过期** —— 已安装的那份与"用当前输入现算"的结果不一致" >&2
echo "            installed = $got" >&2
echo "            fresh     = $want" >&2
_nd=$(diff -u "$INST" "$FRESH" | grep -c '^[+-][^+-]' || true)
echo "            差异行数  = $_nd" >&2
echo "" >&2
echo "   ★ 关键标志位对比（installed vs fresh）：" >&2
for pat in "set_engram_row_tokens" "swa_plane_kwargs" "long_kv_plane_kwargs"; do
  a=$(grep -c "$pat" "$INST" 2>/dev/null || echo 0)
  b=$(grep -c "$pat" "$FRESH" 2>/dev/null || echo 0)
  mark=" "; [ "$a" != "$b" ] && mark="★"
  printf "     %s %-30s installed=%-4s fresh=%-4s\n" "$mark" "$pat" "$a" "$b" >&2
done
# ★ 单独处理"调用点是否取 [0]"：**比行内容，不比计数**。
#   （只比计数是抓不到的：带不带 `[0]` 都是 1 行 —— 我自己刚踩过这个弱判据。）
_a=$(grep -m1 "return build_prev_tok(" "$INST" 2>/dev/null | tr -d ' ')
_b=$(grep -m1 "return build_prev_tok(" "$FRESH" 2>/dev/null | tr -d ' ')
case "$_a" in *")[0]"*) _as="有";; *) _as="★无";; esac
case "$_b" in *")[0]"*) _bs="有";; *) _bs="★无";; esac
_mark=" "; [ "$_as" != "$_bs" ] && _mark="★"
printf "     %s %-30s installed=%-4s fresh=%-4s  ← 按**行内容**判（不是计数）\n" \
       "$_mark" "调用点取 [0]" "$_as" "$_bs" >&2
if [ "$_mark" = "★" ]; then
  echo "         installed: ${_a:-<无此调用>}" >&2
  echo "         fresh    : ${_b:-<无此调用>}" >&2
  echo "         ⇒ ★ 这一格正是 2026-09-22 的 TypeError 事故（元组被当 prev_tok 传下去）" >&2
fi
echo "" >&2
echo "   ⇒ 处置：用当前输入重新合并并安装（旧的先备份），然后**重跑这一臂**：" >&2
echo "        cp -a <installed> <installed>.stale-\$(date +%H%M%S)" >&2
echo "        python3 $MERGER --img $IMG --kv8 $KV8 --prod $PROD --out <installed>" >&2
echo "   ★ 不要跳过这一步：合并件过期 = 某个功能**静默消失**（可能是 Engram 接线，也可能是刚修的 bugfix）。" >&2
exit 1
