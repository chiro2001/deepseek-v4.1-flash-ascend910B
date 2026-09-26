#!/usr/bin/env bash
# 本地语义测试（不需要 NPU、不需要 numba、不写发布树）
# 用法：bash a2/agents/Engram_exactfix/tests/run_all.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rc=0

# ---------------------------------------------------------------- 门 0：★ py_compile
# 为什么要有它（2026-09-22 实测踩到）：交付的 `model_runner_v1.patched.py` 第一版里
# `global _ENGRAM_ROW_TOKENS_DISABLED` 写在 `except` 里，而函数体前面**先读**了它
# ⇒ `SyntaxError: name ... is used prior to global declaration` ⇒ **import 期崩、起服必挂**。
# ★ 陷阱：`ast.parse()` 与"只 grep 关键行"**都能过**，只有 `py_compile`/`compile()`
#   的 **symtable 阶段**会报。⇒ 交付件一律先过这道门（三个文件 + 两个 diff 的应用结果）。
echo "=============================================================="
echo "== 门 0：py_compile 全部交付件（AST 检查会漏掉这一类）"
echo "=============================================================="
# 交付件既可能在开发布局（patches/ 子目录），也可能在发布布局（同目录）⇒ 两处都找
_pt="$HERE/../patches"
[ -d "$_pt" ] || _pt="$HERE/.."
# ★ 不用 /tmp（本仓纪律）：落一个 mktemp 目录，跑完删掉
_tmp=$(mktemp -d "${TMPDIR:-$HOME/tmp}/dsv41-pyc.XXXXXX") || exit 2
trap 'rm -rf "$_tmp"' EXIT
for f in model_runner_v1.patched.py model.patched.py engram_hash.patched.py engram_repair.py; do
    if [ ! -f "$_pt/$f" ]; then
        echo "  [✗] 缺交付件 $_pt/$f"; rc=1; continue
    fi
    if python3 -m py_compile "$_pt/$f" 2>"$_tmp/err"; then
        echo "  [✓] $f"
    else
        echo "  [✗] $f py_compile 失败："; sed 's/^/       /' "$_tmp/err"; rc=1
    fi
done
rm -rf "$_tmp"; trap - EXIT

for t in test_callsite_contract.py test_true_tokens_repair.py test_repair_plan.py; do
    echo "=============================================================="
    echo "== $t"
    echo "=============================================================="
    ( cd "$HERE" && python3 "$t" ) || rc=1
done
echo "=============================================================="
[ "$rc" = 0 ] && echo "✓ run_all：全部通过" || echo "✗ run_all：有失败项"
exit "$rc"
