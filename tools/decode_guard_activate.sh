#!/usr/bin/env bash
# [DECODE-GUARD-ACTIVATE] 直连 decode 半边，检验请求边界护栏。
#
# 用法（在跑着 CED P/D 的机器上，例如 a3-21）：
#   bash tools/decode_guard_activate.sh <D_PORT> [SERVE_LOG]
#   默认 D_PORT=18991，SERVE_LOG 缺省则自动从 /opt/dsv41/results 里取最新的
#   *decode*/serve.log。
#
# 为什么需要它：2026-09-27 00:01 一条**直连 decode 半边**的普通请求让 D 自己去
# prefill，撞上 CED 固定 128-token replay 守卫，worker 里 raise ⇒ EngineCore
# 退出 ⇒ 整个 D 实例死掉（要重载 ~20 分钟权重）。
#
# 判据（三条同时成立才算护栏生效）：
#   ① 那条请求返回 400 + code=ced_decode_role_requires_kv_transfer_params；
#   ② 紧接着 /health 仍是 200（**引擎还活着**，这就是事故的负控）；
#   ③ serve.log 出现 `[V41-DECODE-GUARD] rejected`。
#
# ⚠️ 这条脚本会**故意**发出事故形状的请求。护栏没装好的实例上跑它 = 把实例打死，
#    所以它先做自检：③ 的 middleware loaded 痕迹必须先存在，否则直接拒绝执行。
set -uo pipefail
PORT=${1:-18991}
BASE="http://127.0.0.1:$PORT"
LOG=${2:-}

if [ -z "$LOG" ]; then
  LOG=$(ls -t /opt/dsv41/results/*/serve.log 2>/dev/null | head -1)
fi

echo "[guard] D=$BASE  log=${LOG:-<未找到>}"

# ---- 前置自检：护栏必须已在运行（否则这条脚本本身就是事故复现器）----
if [ -z "$LOG" ] || ! [ -f "$LOG" ]; then
  echo "[guard][拒绝执行] 找不到 serve.log（用它判断护栏是否已加载）；请显式传第 2 个参数。" >&2
  exit 2
fi
if ! grep -qa "V41-DECODE-GUARD.*middleware loaded\|decode API guard: ON" "$LOG"; then
  echo "[guard][拒绝执行] 这个实例上的护栏**没有加载**（日志里没有 middleware loaded / guard ON）。" >&2
  echo "[guard]            此时发事故形状的请求会真的打死引擎，本脚本不做这件事。" >&2
  exit 3
fi
echo "[guard] ✓ 前置：护栏已加载"

echo
echo "=== ① 事故形状：直连 D，无 kv_transfer_params，prompt > 128 token ==="
code=$(curl -s -o /tmp/.decode_guard_probe.json -w '%{http_code}' --max-time 60 \
  -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-ced-pd","messages":[{"role":"user","content":"计算 1+1 是多少？ 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。 请详细说明。"}],"max_tokens":8,"temperature":0}' 2>/dev/null)
code=${code:-000}
printf '  HTTP %s\n' "$code"
grep -o '"code":"[^"]*"' /tmp/.decode_guard_probe.json 2>/dev/null | head -1
ok1=0
[ "$code" = "400" ] && grep -q "ced_decode_role_requires_kv_transfer_params" /tmp/.decode_guard_probe.json && ok1=1

echo
echo "=== ② 引擎是否还活着（事故的判据）==="
sleep 3
h=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/health" 2>/dev/null); h=${h:-000}
m=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/v1/models" 2>/dev/null); m=${m:-000}
printf '  /health -> %s   /v1/models -> %s\n' "$h" "$m"
ok2=0
[ "$h" = "200" ] && [ "$m" = "200" ] && ok2=1

echo
echo "=== ③ 拦截日志 ==="
grep -a "V41-DECODE-GUARD" "$LOG" | tail -2 | sed 's/^/  /'
ok3=0
grep -qa "V41-DECODE-GUARD] rejected" "$LOG" && ok3=1

echo
if [ "$ok1" = 1 ] && [ "$ok2" = 1 ] && [ "$ok3" = 1 ]; then
  echo "[guard] ✅ 护栏生效：拦下 400 + 引擎存活 + 有可观测痕迹"
  exit 0
fi
echo "[guard] ❌ 有判据未通过：①=$ok1 ②=$ok2 ③=$ok3" >&2
[ "$ok2" = 0 ] && echo "[guard]    ⚠️ 引擎已经不响应 —— 这条实例需要重启。" >&2
exit 1
