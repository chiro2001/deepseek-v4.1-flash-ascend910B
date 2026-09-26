#!/usr/bin/env bash
# [GUARD-NEGCTRL] 复现事故形状：直连 decode 半边发一条**没有 kv_transfer_params**
# 的生成请求（prompt > 128 token）。期望：HTTP 400，且引擎活着。
set -uo pipefail
D=http://127.0.0.1:18991
PROXY=http://127.0.0.1:18992
echo "=== 0) 前置：D 与代理都活着吗 ==="
printf '  D /health      -> %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $D/health)"
printf '  D /v1/models   -> %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $D/v1/models)"

echo
echo "=== 1) 事故形状：直连 D，无 kv_transfer_params，长 prompt ==="
python3 - <<'PY'
import json, urllib.request, urllib.error
body = {
  "model": "deepseek-v41-ced-pd",
  "messages": [{"role": "user", "content": "计算 1+1 是多少？" + " 请详细说明。" * 60}],
  "max_tokens": 8, "temperature": 0,
}
req = urllib.request.Request("http://127.0.0.1:18991/v1/chat/completions",
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("  !! 期望 400，实际 HTTP", r.status, "—— 护栏没生效")
    print("  body:", r.read()[:300])
except urllib.error.HTTPError as e:
    print("  HTTP", e.code)
    print("  body:", e.read().decode()[:400])
except Exception as e:
    print("  异常（不是护栏行为）:", type(e).__name__, e)
PY

echo
echo "=== 2) 引擎是否还活着（这就是事故的判据）==="
sleep 3
printf '  D /health      -> %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $D/health)"
printf '  D /v1/models   -> %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $D/v1/models)"
printf '  进程数         -> %s\n' "$(docker exec dsv41-ced-d4b bash -lc 'ps -ef | grep -c "[v]llm serve"')"

echo
echo "=== 3) 拦截日志（判据：可观测痕迹）==="
grep -a "V41-DECODE-GUARD" /home/l00886679/tmp/20260926/dspark/results/ced_d4b_guarded_0927_001851/serve.log | tail -3
