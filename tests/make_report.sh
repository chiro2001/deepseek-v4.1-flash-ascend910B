#!/usr/bin/env bash
# 汇总一次 run_test.sh 的产物为 REPORT.md
#   bash make_report.sh <结果目录> <run_id> <模型路径> <local_owner_模式>
set -uo pipefail
OUT=${1:-.}; RUN_ID=${2:-unknown}; MODEL=${3:-unknown}; LO=${4:-unknown}
JQ() { python3 -c "$1" 2>/dev/null || echo "n/a"; }

{
echo "# A2 验收报告 — $RUN_ID"
echo
echo "- 时间：$(date '+%F %T')"
echo "- 模型：\`$MODEL\`"
echo "- local-owner：**$LO**"
echo "- 镜像指纹："
sed 's/^/  - /' "$OUT/BUILD_INFO.txt" 2>/dev/null || echo "  - (无)"
echo
echo "## 1. 关键指标"
echo
echo "| 指标 | 值 | 门槛 | 判定 |"
echo "|---|---|---|---|"

KV=$(grep -oE "kv_tokens=[0-9]+" "$OUT/env.txt" 2>/dev/null | cut -d= -f2)
KV=${KV:-0}
if [ "$KV" -gt 3145728 ]; then KVJ="✅ PASS"; else KVJ="❌ FAIL"; fi
echo "| GPU KV cache tokens | ${KV} | > 3,145,728 | $KVJ |"

for tag in 8k 128k; do
  f=$(ls "$OUT"/p42_t4_quote_*_quote_${tag}.jsonl 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  line=$(python3 - "$f" <<'PY' 2>/dev/null
import json, statistics as st, sys
recs=[json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
ok=[r for r in recs if r.get("metrics_ok") and not r.get("error")]
if not ok: print("n/a|n/a|n/a|0"); raise SystemExit
m=lambda k: st.median([r[k] for r in ok if r.get(k) is not None])
ms,a,tps=m("ms_per_step"),m("accept_length"),m("decode_tok_s")
print(f"{ms:.2f}|{a:.3f}|{tps:.1f}|{len(ok)}")
PY
)
  IFS='|' read -r ms a tps n <<< "$line"
  echo "| ${tag} ms/step | ${ms} | — | — |"
  echo "| ${tag} 接受长度 A | ${a} | — | — |"
  echo "| ${tag} decode tok/s | ${tps} | — | — |"
done

if [ -f "$OUT/vision.json" ]; then
  v=$(python3 -c "
import json;d=json.load(open('$OUT/vision.json'))
print(d.get('pass_n') or d.get('pass') or '?', d.get('cases') or d.get('total') or '?')" 2>/dev/null)
  vp=${v%% *}; vt=${v##* }
  if [ "${vp:-0}" != "?" ] && [ "${vp:-0}" -ge 19 ] 2>/dev/null; then vj="✅ PASS"; else vj="❌ FAIL"; fi
  echo "| Vision | ${vp}/${vt} | ≥ 19/23 | $vj |"
fi

if [ -f "$OUT/gsm8k.json" ]; then
  g=$(python3 -c "
import json;d=json.load(open('$OUT/gsm8k.json'))
print(d.get('acc') or d.get('accuracy') or '?')" 2>/dev/null)
  echo "| GSM8K-200 | ${g} | ≥ 0.95 参考 | 参考 |"
fi

# --- 多 batch / 多轮对话（MODE=prod 或 multibatch_session.sh 产出 summary.json）---
if [ -f "$OUT/summary.json" ]; then
  echo
  echo "## 1b. 多 batch / 多轮对话（summary.json）"
  echo
  python3 - "$OUT/summary.json" <<'PY' 2>/dev/null || echo "（解析失败，见原始 summary.json）"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
mt, cc, mx = d.get("multiturn"), d.get("concurrency"), d.get("mixed")
print("| 块 | 结果 | 判据 | 判定 |")
print("|---|---|---|---|")
if mt:
    print(f"| A 多轮对话（{mt['rounds']} 轮，末次 prompt={mt.get('final_prompt_tokens')} tok） "
          f"| 轮内逐字召回 {mt['needle_hits']}/{mt['needle_total']}，全长召回 "
          f"{sum(mt['recall'].values())}/3 | 全部命中 | "
          f"{'✅' if mt['needle_hits']==mt['needle_total'] and all(mt['recall'].values()) else '❌'} |")
if cc:
    print(f"| B 并发逐 item 比对（conc={cc['conc']}，n={cc['n']}） "
          f"| 串行 {cc['serial_ok']}/{cc['n']}，并发 {cc['conc_ok']}/{cc['n']}，"
          f"逐项不一致 **{len(cc['mismatch'])}** | 0 项不一致 | "
          f"{'✅' if not cc['mismatch'] else '❌'} |")
if mx:
    print(f"| C 长短交错（{mx['long_ctx']} + {mx['n_short']} 短请求） "
          f"| 基线 {mx['baseline_ok']}/{mx['n_short']}，混合 {mx['mixed_ok']}/{mx['n_short']}，"
          f"不一致 **{len(mx['mismatch'])}** | 0 项不一致 | "
          f"{'✅' if not mx['mismatch'] else '❌'} |")
print()
print(f"- 机器可读判据：`{d.get('verdict')}`")
PY
fi

echo
echo "## 2. 服务日志关键行"
echo
echo '```'
grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency[^,]*|Available KV cache memory[^,]*" \
  "$OUT/serve.log" 2>/dev/null | head -5
echo '```'
echo
echo "## 3. 已知预期（来自 A3 实测，A2 需自行比对）"
echo
echo "| 项 | A3(A3-node2 front8, 910C) | 说明 |"
echo "|---|---|---|"
echo "| 8K ms/step | 35.8 | static_kernel=1 + mtpq + gate(CHUNK=0) + local-owner |"
echo "| 128K ms/step | 36.1–37.1 | 同上 |"
echo "| 128K A | 2.87–2.96 | Engram-on |"
echo "| 128K tok/s | 78–82 | 单流 |"
echo "| KV tokens | 3,557,471 | util=0.94 |"
echo "| Vision | 23/23 | qrot 修复后的 vision 分片 |"
echo "| GSM8K-200 | 197–199 | chat 模式 |"
echo
echo "> A2 是 8×910B3，设备侧更慢但 verify 更快（A2 报告：无投机 18.9 ms/round）；"
echo "> **不要期待与 A3 完全一致**，重点是看趋势与门槛。"
echo
echo "## 4. 原始产物"
echo
ls -la "$OUT" | sed 's/^/    /'
} > "$OUT/REPORT.md"

cat "$OUT/REPORT.md"
