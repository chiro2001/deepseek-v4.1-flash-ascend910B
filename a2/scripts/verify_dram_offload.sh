#!/usr/bin/env bash
# verify_dram_offload.sh —— A2/A3 上一条命令验证「DRAM 卸载真的在干活」
#
# 为什么有它：`scripts/run_test.sh` 的 7 项里**没有卸载检查**；
#   `a2/scripts/check_4axis_acceptance.py` 是**判决器**（要你先备好证据），
#   而"制造卸载"的压测客户端一直在工作区（`bench/kv_offload_client.py`）**没进发布仓**。
#   本脚本把「压测 + 判据 + 留证据」串成一条命令（Python 侧 = `tests/verify_dram_offload.py`）。
#
# 它**只发 HTTP 请求**：不起容器、不改配置、不碰 NPU 设备。生产的服务照常在跑。
#
# 用法：
#   bash a2/scripts/verify_dram_offload.sh                      # 默认 32K 前缀 ×1 请求
#   PROMPT_TOKENS=131072 bash a2/scripts/verify_dram_offload.sh  # 更大前缀（更容易触发卸载）
#   PORT=8100 bash a2/scripts/verify_dram_offload.sh             # 换端口
#   DRY=1 bash a2/scripts/verify_dram_offload.sh                 # 只看连接与当前计数器，不发请求
#   PLAT=a3 bash a2/scripts/verify_dram_offload.sh               # ★ A3：端口 8020 + 模型名 deepseek-v41
#
# 退出码：0 = 硬判据全过；9 = 有硬判据未过；64 = 前置不给（连不上/缺脚本）
#
# ★★ 平台：`PLAT=a2|a3`（默认 a2）。只改两个平台不同的默认值，显式给的一律优先：
#     PLAT | 端口 | 模型名（API body 的 "model" 字段 —— 填错会被服务端 400，很容易误读成"服务坏了"）
#     a2   | 8077 | deepseek-v4-flash
#     a3   | 8020 | deepseek-v41      ← `scripts/serve_a3.sh` 的 SERVED_NAME 默认
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/../.." && pwd)

PLAT=${PLAT:-a2}
case "$PLAT" in
  a2) _def_port=8077; _def_model=deepseek-v4-flash ;;
  a3) _def_port=8020; _def_model=deepseek-v41 ;;
  *)  printf '⛔ PLAT 只能是 a2|a3，得到 %s\n' "$PLAT" >&2; exit 64 ;;
esac
PORT=${PORT:-$_def_port}
URL=${URL:-http://127.0.0.1:$PORT}
MODEL=${MODEL_NAME:-$_def_model}
echo "[verify_dram_offload] PLAT=$PLAT  URL=$URL  model=$MODEL  （a2=8077/deepseek-v4-flash，a3=8020/deepseek-v41）"
PROMPT_TOKENS=${PROMPT_TOKENS:-32768}
MAX_TOKENS=${MAX_TOKENS:-16}
PROMPTS=${PROMPTS:-1}
CONC=${CONC:-1}
TIMEOUT_S=${TIMEOUT_S:-3600}
DRY=${DRY:-0}
OUT=${OUT:-$HOME/dram_offload_verify_$(date +%Y%m%d_%H%M%S).json}
PY=$PKG/tests/verify_dram_offload.py

say() { printf '\n==== %s ====\n' "$*"; }
die() { printf '\n⛔ %s\n' "$*" >&2; exit 64; }

[ -f "$PY" ] || die "缺 $PY（先 git pull）"

say "① 前置检查（只读）"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$URL/health" 2>/dev/null || true)
[ "$code" = "200" ] || die "容器内 $URL/health 不通（code=${code:-无}）。★ 若生产端口不是 $PORT，用 PORT=<端口> 指定"
echo "  ✓ $URL/health = 200"
echo "  · served model 名候选：$(curl -s -m 8 "$URL/v1/models" 2>/dev/null | head -c 200)"

if [ "$DRY" = "1" ]; then
    say "DRY=1：只打印当前卸载相关计数器（不发任何请求）"
    curl -s -m 10 "$URL/metrics" | grep -E '^vllm:(kv_offload|external_prefix_cache|prefix_cache)' | head -20
    exit 0
fi

say "② 压测 + 判据（fill → reset_prefix_cache → replay）"
echo "  前缀 ${PROMPT_TOKENS} token × ${PROMPTS} 请求，max_tokens=${MAX_TOKENS}，并发 ${CONC}"
python3 "$PY" --base-url "$URL" --model "$MODEL" \
    --prompts "$PROMPTS" --prompt-tokens "$PROMPT_TOKENS" \
    --max-tokens "$MAX_TOKENS" --concurrency "$CONC" --timeout "$TIMEOUT_S" \
    --out "$OUT"
rc=$?

say "③ 产物与后续可核命令"
echo "  · 本次报告（含 before/mid/after 三份 metrics 原始行）：$OUT"
_sl=$(ls -td "$HOME"/projects/dsv41-upstream-pr/shadow-pkg/results/*/ 2>/dev/null | head -1)
if [ -n "$_sl" ]; then
    echo "  · 起服日志（最近一次）：${_sl}serve.log"
    echo "  · 交给判决器（同一份 JSON 可直接当 --client 用，字段名与 A3 客户端同形）："
    echo "      python3 $PKG/a2/scripts/check_4axis_acceptance.py \\"
    echo "          --log ${_sl}serve.log --client $OUT"
fi

exit $rc
