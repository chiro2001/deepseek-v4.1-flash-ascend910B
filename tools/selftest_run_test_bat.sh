#!/usr/bin/env bash
# =============================================================================
# selftest_run_test_bat.sh —— 守护「`run_test.sh` 的 prefill batch 默认值」这条链
#
# 为什么需要（2026-09-23 用户实测暴露）：
#   长上下文乱码的根因修复是 **`BAT_TOKENS 2048 -> 8192`**（`8eb2613`），改的是
#   **`scripts/serve_a2.sh`**；但 `scripts/run_test.sh` **显式**把 `BAT_TOKENS`
#   传给 `serve_a2.sh` ⇒ **它的默认值会覆盖模板**。而它长期是 2048
#   ⇒ ★ **用"标准验证入口"验证长上下文精度 = 在一个已知会退化的配置上验证**。
#   本脚本把这条链**钉死**（并且能抓住任何一环被改回去）。
#
# 它查什么（全部是**静态可判**，不需要起服、不需要 NPU）：
#   ① `run_test.sh` 的 `BAT_TOKENS` 默认 == **8192**
#   ② `run_test.sh` 确实把它**传下去**（有 `BAT_TOKENS="$BAT_TOKENS"` 那行）
#   ③ `serve_a2.sh` 的默认也是 **8192**（模板那一侧）
#   ④ 与 KV 门槛自洽：门槛是按 `BAT=8192/GPU_UTIL=0.92` 的 2,823,080 定的
#      （若有人把默认改回 2048，KV 会涨到 ~4.15M ⇒ 门槛虽仍过，但精度门失效 ⇒ ① 会先拦住）
#   ⑤ ★ **负控**：把默认改回 2048 的一份**临时副本**必须被判 FAIL（证明本自测有效）
#
# 用法： bash tools/selftest_run_test_bat.sh
# 退出码：0 = 全过；9 = 有失败
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/.." && pwd)

V=0; F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }

# 取值：在完整串 `BAT_TOKENS=${BAT_TOKENS:-NNN}` 上取 `:-` 与 `}` 之间的数字。
# ★ 不能用 `grep -oE '[0-9]+$'` —— 值后紧跟 `}`，`$` 锚不匹配（实测踩到，
#   表现是"读不到"却仍打印 OK 的假象）。
bat_default_of() {
    grep -oE 'BAT_TOKENS=\$\{BAT_TOKENS:-[0-9]+\}' "$1" 2>/dev/null | head -1 \
        | sed -n 's/.*:-\{0,1\}\([0-9]\{1,\}\)}.*/\1/p'
}

RT=$PKG/scripts/run_test.sh
SV=$PKG/scripts/serve_a2.sh
[ -f "$RT" ] || { echo "⛔ 缺 $RT" >&2; exit 9; }
[ -f "$SV" ] || { echo "⛔ 缺 $SV" >&2; exit 9; }

say "① run_test.sh 的 prefill batch 默认值必须 == 8192"
_rt=$(bat_default_of "$RT")
if [ "${_rt:-}" = "8192" ]; then
    ok "run_test.sh BAT_TOKENS 默认 = $_rt"
else
    bad "run_test.sh BAT_TOKENS 默认 = ${_rt:-读不到}（期望 8192；2048 是长上下文退化的已知开关）"
fi

say "② run_test.sh 必须**真的把它传下去**（否则默认值只是摆设）"
if grep -qE 'BAT_TOKENS="\$BAT_TOKENS"' "$RT"; then
    ok "找到显式透传：BAT_TOKENS=\"\$BAT_TOKENS\""
else
    bad "没找到把 BAT_TOKENS 传给 serve_a2.sh 的那行 ⇒ 默认值可能根本没生效"
fi

say "③ 模板侧（serve_a2.sh）默认也必须是 8192（两处不许分叉）"
_sv=$(bat_default_of "$SV")
if [ "${_sv:-}" = "8192" ]; then
    ok "serve_a2.sh BAT_TOKENS 默认 = $_sv"
else
    bad "serve_a2.sh BAT_TOKENS 默认 = ${_sv:-读不到}（期望 8192）"
fi
if [ "${_rt:-x}" = "${_sv:-y}" ]; then
    ok "两处默认一致（$_rt）"
else
    bad "两处默认**分叉**：run_test=$_rt vs serve_a2=$_sv ⇒ 验证入口会覆盖模板"
fi

say "④ 与 KV 门槛自洽（门槛是按 BAT=8192/GPU_UTIL=0.92 的 2,823,080 定的）"
_kvmin=$(sed -n 's/^KV_MIN=\${KV_MIN:-\([0-9]\{1,\}\)}.*/\1/p' "$RT" | head -1)
if [ -n "${_kvmin:-}" ] && [ "${_kvmin:-0}" -le 2823080 ] 2>/dev/null; then
    ok "KV_MIN=$_kvmin ≤ 8192 口径的实测 2,823,080（默认配置不会必然判 FAIL）"
else
    bad "KV_MIN=${_kvmin:-读不到} 高于 8192 口径的 2,823,080 ⇒ 默认配置会必然判 FAIL"
fi

say "⑤ ## 负控：把默认改回 2048 的副本必须被判 FAIL（证明本自测真的有效）"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/scripts"
sed 's/BAT_TOKENS=${BAT_TOKENS:-8192}/BAT_TOKENS=${BAT_TOKENS:-2048}/' "$RT" > "$T/scripts/run_test.sh"
cp "$SV" "$T/scripts/serve_a2.sh"
_neg=$(bat_default_of "$T/scripts/run_test.sh")
if [ "${_neg:-}" = "2048" ]; then
    ok "负控夹具生效（临时副本的默认确实被改成 2048）"
else
    bad "负控夹具没生效（读到 ${_neg:-空}）⇒ 本项无法证明自测有效性"
fi
if [ "${_neg:-}" != "8192" ]; then
    ok "负控判定：非 8192 会被 ① 拦住（判据有效）"
else
    bad "负控判定失败：2048 竟然算通过"
fi

echo
echo "=============== 通过 $V 条 / 失败 $F 条 ==============="
if [ "$F" = "0" ] && [ "$V" -ge 7 ]; then echo "✅ 自测全过"; exit 0; fi
echo "❌ 不合格"; exit 9
