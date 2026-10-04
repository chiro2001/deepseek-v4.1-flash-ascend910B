#!/usr/bin/env bash
# 验证"自定义 vendor 覆盖镜像 vendor"是否真的生效，并做性能/正确性 A/B。
#
# 背景：只设 ASCEND_CUSTOM_OPP_PATH **不会**覆盖已有算子的 kernel ——
#   vllm_ascend.utils.bootstrap_custom_op_env() 会把镜像自带 vendor 路径**前插**
#   （utils.py:323-332）。唯一可行路径是起服前把 vendor 复制覆盖镜像路径
#   （serve_a2.sh 的 [OPP-OVERRIDE] 段，由 V41_HC_OPP_PKG 触发）。
# 本脚本的第 0 步就是**证明覆盖真的生效**（对比容器内 .o 的 md5），否则后面的
#   性能数字全是 stock 的，会得出"改动无效"的错误结论。
#
# 用法：
#   RUN_ID=<本次 run> HOST_VENDOR=<宿主 vendor 目录（含 vendors/custom_transformer）> \
#   REL_OBJ=<相对 vendor 的 .o 路径> \
#   bash tools/verify_opp_vendor.sh [conc_list] [out_tokens] [repeats]
set -u
cd "$(dirname "$0")/.." || exit 1
: "${RUN_ID:?需要 RUN_ID=<本次 run id>}"
: "${HOST_VENDOR:?需要 HOST_VENDOR=<宿主 vendor 目录>}"
: "${REL_OBJ:?需要 REL_OBJ=<相对 vendor 的 .o 路径，如 op_impl/ai_core/tbe/kernel/ascend910_93/<op>/<file>.o>}"
CONC=${1:-1,8}
OUT=${2:-256}
REPS=${3:-3}
NAME=${NAME:-dsv41-tp8k5}
PORT=${PORT:-19210}
BASE="http://127.0.0.1:$PORT"
IMG_VENDOR=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

echo "== [0] 覆盖生效断言（最关键）=="
HOST_MD5=$(md5sum "$HOST_VENDOR/vendors/custom_transformer/$REL_OBJ" 2>/dev/null | awk '{print $1}')
CTR_MD5=$(sudo docker exec "$NAME" bash -lc "md5sum $IMG_VENDOR/$REL_OBJ 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
echo "  宿主:     ${HOST_MD5:-<缺>}"
echo "  容器内:   ${CTR_MD5:-<缺>}"
if [ -z "$HOST_MD5" ] || [ "$HOST_MD5" != "$CTR_MD5" ]; then
  echo "  ✗ FAIL：容器内 vendor 与宿主不一致 ⇒ 自定义 kernel **没生效**，后续性能数字无意义"
  exit 2
fi
echo "  ✓ 一致（自定义 kernel 已真正加载）"

echo "== [1] 就绪 =="
for i in $(seq 1 60); do
  c=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null); c=${c:-000}
  [ "$c" = "200" ] && { echo "  health=200 after $((i*15))s"; break; }
  sleep 15
done
[ "$c" = "200" ] || { echo "  ✗ FAIL health=$c"; tail -5 "results/$RUN_ID/serve.log"; exit 3; }

echo "== [2] 性能 N=$CONC × $REPS rep =="
python3 tools/bench_concurrency.py --base-url "$BASE" --model deepseek-v41 \
  --metrics-url "$BASE/metrics" --concurrency "$CONC" --prompt-tokens 1024 \
  --output-tokens "$OUT" --repeats "$REPS" \
  --label "opp-$(basename "$HOST_VENDOR")" \
  --json-out "$HOME/tmp/opp_bench_${RUN_ID}.json" 2>&1 | tail -14

echo "== [3] 正确性 regress2 =="
timeout 900 python3 "$HOME/tmp/regress2.py" "$PORT" 2>&1 | tail -8
echo "== done =="
