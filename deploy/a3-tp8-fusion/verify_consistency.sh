#!/usr/bin/env bash
# =============================================================================
# verify_consistency.sh —— 核心判据：容器里装的 = 包里那一份（逐文件 sha256）
#
# 为什么必须有：本包有两个交付面
#   ① baked（install.sh 拷进容器真实路径）
#   ② mount（起服时 -v 挂仓库文件）
# 两者"概念上一致"没有意义 —— 必须逐字节一致，否则同一个实验号
# 在两种形态下跑出不同结果，而没人看得出来。
#
# 本脚本比对 **容器 vs payload**。要同时验 baked 与 mount 两个面，就跑两次：
#   bash verify_consistency.sh <container>                  # 比容器
#   bash verify_consistency.sh --mode mount <container>     # 比挂载源（仓库文件）
#
# 退出码：0 = 全部一致；1 = 有差异（逐条打印）。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=baked
CT=""
# ★ 两种布局都要能识别：
#   * 仓库里：deploy/a3-tp8-fusion/{payload/,verify_consistency.sh}
#   * 发布包里：包根直接就是 payload（./{opp,so,py}/verify_consistency.sh）
if [ -d "$HERE/payload/opp" ]; then DEFAULT_REF="$HERE/payload"; else DEFAULT_REF="$HERE"; fi
PAYLOAD="$DEFAULT_REF"
REF="$DEFAULT_REF"

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)    MODE="$2"; shift 2 ;;
        --payload) PAYLOAD="$2"; REF="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) CT="$1"; shift ;;
    esac
done
[ -n "$CT" ] || { echo "用法: bash verify_consistency.sh [--mode baked|mount] <container>" >&2; exit 2; }
[ -d "$REF" ] || { echo "[FAIL] 找不到 payload 目录 $REF（先跑 build_payload.sh）" >&2; exit 2; }
[ -f "$REF/PAYLOAD.sha256" ] || { echo "[FAIL] $REF 里没有 PAYLOAD.sha256" >&2; exit 2; }

SO_NAME=vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
ASC=/vllm-workspace/vllm-ascend/vllm_ascend
OPP=/vllm-workspace/3out_opp

pass=0; fail=0
ok()  { pass=$((pass+1)); printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad() { fail=$((fail+1)); printf '  \033[31m✗\033[0m %s\n' "$*"; }

echo "== 一致性校验  container=$CT  mode=$MODE  ref=$REF"

if [ "$MODE" = mount ]; then
    # mount 形态：只挂 2 个 .py（二进制仍走 baked，因为 .so 必须落在 site-packages 里）。
    # 所以这里比对的是「仓库里的挂载源」vs「payload 里的同名件」。
    # 仓库根可用 REPO_ROOT 覆盖（默认从脚本位置往上两级）。
    SRC_REPO="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"
    echo "   (mount 形态：比对挂载源 vs payload；repo=$SRC_REPO)"
    for pair in "py/model.py:experimental/fusion-3out/integration/patched/model.py" \
                "py/dsa_v41.py:experimental/fusion-3out/integration/patched/dsa_v41.py"; do
        rel="${pair%%:*}"; sub="${pair#*:}"
        f="$SRC_REPO/$sub"
        if [ ! -f "$f" ]; then bad "挂载源缺失: $sub"; continue; fi
        want="$(awk -v r="$rel" '$2==r{print $1}' "$REF/PAYLOAD.sha256" | head -1)"
        got="$(sha256sum "$f" | awk '{print $1}')"
        [ "$want" = "$got" ] && ok "$rel  <=  $(basename "$sub")" \
                             || bad "$rel  期望 ${want:0:12}… 实得 ${got:0:12}…"
    done
    echo "   注：二进制（opp/ 与 .so）不走 mount，仅 baked；这两个 .py 是唯一可挂载的件。"
else
    docker inspect "$CT" >/dev/null 2>&1 || { echo "[FAIL] 找不到容器 $CT" >&2; exit 2; }
    echo "   源码位置：容器内真实路径"
    while read -r want rel; do
        [ -n "$rel" ] || continue
        case "$rel" in
            opp/*)  tgt="$OPP/${rel#opp/}" ;;
            so/*)   tgt="$ASC/$SO_NAME" ;;
            py/model.py)   tgt="$ASC/models/deepseek_v4/model.py" ;;
            py/dsa_v41.py) tgt="$ASC/attention/dsa_v41.py" ;;
            *) continue ;;   # 文档与脚本不落在容器里
        esac
        if ! docker exec "$CT" test -f "$tgt" 2>/dev/null; then
            bad "$rel → $tgt  （容器内不存在）"
            continue
        fi
        got="$(docker exec "$CT" sha256sum "$tgt" 2>/dev/null | awk '{print $1}')"
        [ "$want" = "$got" ] && ok "$rel" || bad "$rel  期望 ${want:0:12}… 实得 ${got:0:12}…"
    done < "$REF/PAYLOAD.sha256"
fi

echo "----"
echo "  一致 $pass 项，不一致 $fail 项"
if [ "$fail" != 0 ]; then
    echo "  ❌ 校验失败：两个交付面不一致"
    exit 1
fi
echo "  ✅ 校验通过：是同一批文件"
