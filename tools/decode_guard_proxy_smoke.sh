#!/usr/bin/env bash
# 经代理 18992 的端到端冒烟：证明加了护栏后，正常 P→D 转发链路没被破坏。
set -uo pipefail
echo "=== 纯文本（经 18992 代理） ==="
curl -s --max-time 300 -X POST http://127.0.0.1:18992/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-ced-pd","messages":[{"role":"user","content":"只回答一个词：1+1=?"}],"max_tokens":16,"temperature":0}' \
  | head -c 600
echo
echo
echo "=== D 侧是否收到带 kv_transfer_params 的转发（护栏放行痕迹）==="
grep -ac "V41-DECODE-GUARD" /home/l00886679/tmp/20260926/dspark/results/ced_d4b_guarded_0927_001851/serve.log
