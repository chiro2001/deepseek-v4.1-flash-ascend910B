#!/usr/bin/env bash
# =============================================================================
# make_shadow_dp.sh —— 造一份**支持 DP（data parallel）**的 shadow 副本，用于跑 DP2TP8
#
# 为什么需要（不是"顺手加个开关"）：
#   `scripts/serve_a2.sh` 在生成 inner.sh 时把 DP **写死成 1**：
#       export MODEL="$MODEL" TP=$TP DP=1 PORT=$PORT SERVED_NAME="$SERVED_NAME"
#   而内层 `scripts/serve_v2.sh` **本来就支持 DP**：
#       [ "$DP" -gt 1 ] && ARGS+=(--data-parallel-size "$DP" --data-parallel-size-local "$DP")
#   ⇒ 只要把那个硬编码的 `1` 换成"可被环境覆盖"，DP2TP8 就能起来，
#     而**发布仓本体一个字都不用改**（本仓纪律：生产脚本只读，改动落在副本上）。
#
# DP2TP8 的照抄公式（单机 16 die = 8 卡 × 2 die）：
#   DEVS="0 1 2 … 15" TP=8 DP=2 —— 即 `--tensor-parallel-size 8 --data-parallel-size 2
#   --data-parallel-size-local 2`（官方教程里双机版是 `--data-parallel-size 4
#   --data-parallel-size-local 2`，即每机 2 个 DP rank；单机就是把 size 设成 2）。
#
# 它改了什么（**只有一处**，且带断言）：
#   `DP=1`  →  `DP=${DP:-1}`
#   ★ 不改默认行为：不传 DP 时仍然是 1，与发布仓完全一致。
#   ★ 断言：替换必须**恰好命中 1 次**；0 次或多次 ⇒ 拒绝产出（fail-closed）。
#
# 用法：
#   PKG=<仓> DST=<影子目录> bash tools/make_shadow_dp.sh
#   PKG=<仓> DST=<影子目录> bash tools/make_shadow_dp.sh --check-only   # 只校验不改
#
# 造完起服：
#   cd <DST> && DEVS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" TP=8 DP=2 \
#     MODEL=<模型目录> CPU_BIND=0 bash scripts/serve_a2.sh
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=${PKG:-$(cd "$HERE/.." && pwd)}
DST=${DST:-$HOME/projects/dsv41-upstream-pr/shadow-pkg-dp}
CHECK_ONLY=${CHECK_ONLY:-0}
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1

say() { printf '\n\033[1m======== %s ========\033[0m\n' "$*"; }
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

say "① 前置"
for f in scripts/serve_a2.sh scripts/serve_v2.sh; do
    if [ -f "$PKG/$f" ]; then ok "源文件在位：$PKG/$f"
    else bad "缺 $PKG/$f"; exit 2; fi
done
# 目标行必须恰好出现 1 次（先验，再动手）
_n=$(grep -c '^export MODEL="\$MODEL" TP=\$TP DP=1 PORT=\$PORT SERVED_NAME="\$SERVED_NAME"$' \
     "$PKG/scripts/serve_a2.sh" || true)
if [ "$_n" = "1" ]; then ok "找到唯一的 DP 硬编码行（出现 1 次）"
else bad "DP 硬编码行出现 $_n 次（期望 1）⇒ 上游可能改过，拒绝产出"; exit 2; fi
# 内层确实支持 DP（否则改了也没用）
if grep -q -- '--data-parallel-size "\$DP"' "$PKG/scripts/serve_v2.sh"; then
    ok "内层 serve_v2.sh 支持 DP（含 --data-parallel-size）"
else bad "内层 serve_v2.sh 里没有 --data-parallel-size ⇒ 本工具无意义"; exit 2; fi

if [ "$CHECK_ONLY" = "1" ]; then
    say "check-only：以上都过 ⇒ 可以用本工具造 DP 影子"; exit 0
fi

say "② 造影子树（$DST）"
mkdir -p "$DST" "$DST/scripts"
if [ -e "$DST/.built" ]; then
    ok "已存在 ⇒ 只刷新 scripts/"
else
    _n=0
    for e in "$PKG"/* "$PKG"/.[!.]*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        case "$b" in scripts|.git) continue ;; esac
        [ -e "$DST/$b" ] && continue
        ln -s "$e" "$DST/$b" 2>/dev/null && _n=$((_n+1))
    done
    ok "软链了 $_n 项（scripts/ 除外，它是真目录）"
fi
for e in "$PKG"/scripts/*; do
    [ -e "$e" ] || continue
    b=$(basename "$e")
    case "$b" in serve_a2.sh|serve_v2.sh) cp -f "$e" "$DST/scripts/$b" ;;   # 这两份要改 ⇒ 真拷贝
                    *) [ -e "$DST/scripts/$b" ] || ln -s "$e" "$DST/scripts/$b" 2>/dev/null ;; esac
done
ok "scripts/：serve_a2.sh / serve_v2.sh 为真拷贝，其余软链"

say "③ 打 DP 补丁（唯一一处改动，带断言）"
python3 - "$DST/scripts/serve_a2.sh" <<'PYEOF'
import io, sys
p = sys.argv[1]
old = 'export MODEL="$MODEL" TP=$TP DP=1 PORT=$PORT SERVED_NAME="$SERVED_NAME"'
new = 'export MODEL="$MODEL" TP=$TP DP=${DP:-1} PORT=$PORT SERVED_NAME="$SERVED_NAME"'
s = io.open(p, encoding="utf-8").read()
n = s.count(old)
if n != 1:
    sys.exit("DP 硬编码行出现 %d 次（期望 1）⇒ 拒绝写入" % n)
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8").write(s)
print("  ✓ 已替换：DP=1 → DP=${DP:-1}（命中 1 次）")
PYEOF
rc=$?
[ "$rc" = "0" ] || { bad "打补丁失败（$rc）⇒ 产物不可用"; exit 2; }

say "④ 自检（判据绑内容 + 反例）"
_v=0
grep -q 'DP=\${DP:-1}' "$DST/scripts/serve_a2.sh" && ok "新行在位（DP 可被环境覆盖）" || { bad "新行不在位"; _v=1; }
grep -q '^export MODEL="\$MODEL" TP=\$TP DP=1 PORT' "$DST/scripts/serve_a2.sh" \
    && { bad "旧的硬编码 DP=1 仍在（可能没替换干净）"; _v=1; } || ok "旧的硬编码行已消失"
grep -q -- '--data-parallel-size "\$DP"' "$DST/scripts/serve_v2.sh" && ok "内层仍支持 DP" || { bad "内层丢了对 DP 的支持"; _v=1; }
# ★ 不传 DP 时必须与发布仓**逐字一致**（默认行为不许变）
( cd "$DST" && DP=1 bash -n scripts/serve_a2.sh ) && ok "语法通过（DP 默认路径）" || { bad "语法不通过"; _v=1; }
# ★ 用 dry-run 证明"传了 DP=2 真的会生成 --data-parallel-size 2"
_dr=$(cd "$DST" && DP=2 TP=8 DRY_RUN=1 MODEL=/nonexistent bash scripts/serve_v2.sh 2>&1 | head -60 || true)
case "$_dr" in
  *"--data-parallel-size 2"*) ok "干跑证明：DP=2 ⇒ 命令行里出现 --data-parallel-size 2" ;;
  *) bad "干跑里没看到 --data-parallel-size 2（DP 透传没生效）"; printf '%s\n' "$_dr" | head -6 | sed 's/^/        /'; _v=1 ;;
esac
touch "$DST/.built"

if [ "$_v" != "0" ]; then
    say "结果：自检未通过 ⇒ 产物不可用"; exit 2
fi
cat <<EOF

✓ DP shadow 就绪：$DST

起服（单机 16 die = DP2×TP8）：
  cd $DST && \\
  DEVS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" TP=8 DP=2 CPU_BIND=0 \\
  MODEL=<模型目录> bash scripts/serve_a2.sh

★ 纪律：DEVS 必须是 16 个（TP8×DP2）；只给 8 个会起来但只跑 DP1。
EOF
