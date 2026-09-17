#!/usr/bin/env bash
# quote 口径单流性能测量（T4）：由 run_test.sh 调用，也可单独跑。
#   TAG=8k TOKENS=8192 URL=http://127.0.0.1:8100 bash t_quote.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
TAG=${TAG:-quote}
TOKENS=${TOKENS:-8192}
MAXTOK=${MAXTOK:-256}
REPEATS=${REPEATS:-2}
URL=${URL:-http://127.0.0.1:8100}
OUTDIR=${OUTDIR:-$PKG/results}
PREFIX=${PREFIX:-$PKG/data/hongloumeng.txt}
SUFFIX=${SUFFIX:-$PKG/data/suffix_quote.txt}
OUTJ="$OUTDIR/p42_t4_quote_${TOKENS}_${TAG}.jsonl"
PY=${PYHOST:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python3

code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/health" || true)
[ "$code" = "200" ] || { echo "[t_quote] ABORT: 服务不健康 ($code) at $URL"; exit 3; }
[ -f "$PREFIX" ] || { echo "[t_quote] ABORT: 缺语料 $PREFIX"; exit 3; }
[ -f "$SUFFIX" ] || { echo "[t_quote] ABORT: 缺 suffix $SUFFIX"; exit 3; }

rm -f "$OUTJ"
echo "[t_quote] tag=$TAG tokens=$TOKENS max_tokens=$MAXTOK repeats=$REPEATS"
for i in $(seq 1 "$REPEATS"); do
  echo "[t_quote] ---- point $i/$REPEATS $(date '+%F %T') ----"
  "$PY" "$HERE/p15_stream_curve_filefiller.py" \
    --base-url "$URL" --model deepseek-v41 \
    --tokens "$TOKENS" --max-tokens "$MAXTOK" \
    --warmup-tokens 1 --warmup-output-tokens 24 \
    --prefix-file "$PREFIX" --suffix-file "$SUFFIX" \
    --corpus-label "p42_quote_${TOKENS}_r${i}" --out "$OUTJ" \
    --max-running 1.0 --sample-interval 0.05 --idle-timeout 1800 2>&1 | tail -4 | sed 's/^/    /'
done

"$PY" - "$OUTJ" "$TOKENS" <<'PYEOF'
import json, statistics as st, sys
path, target = sys.argv[1], int(sys.argv[2])
recs = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
ok = [r for r in recs if r.get("metrics_ok") and not r.get("error")]
print(f"[t_quote] points={len(recs)} usable={len(ok)} target_tokens={target}")
if not ok:
    print("  !! 无可用测点 —— 看上面的错误"); sys.exit(4)
def med(name):
    v = [r[name] for r in ok if r.get(name) is not None]
    return st.median(v) if v else None
ms, a, tps = med("ms_per_step"), med("accept_length"), med("decode_tok_s")
for name in ("prompt_tokens_actual","ttft_s","prefill_tok_s","ms_per_step",
             "tok_per_step","decode_tok_s","accept_length","n_sse_chunks"):
    m = med(name)
    if m is not None: print(f"  {name}: median={m}")
if ms and a:
    print(f"  >> ms/step={ms:.2f}  A={a:.3f}  tok/s={tps:.1f}  (1000/ms*A={1000/ms*a:.1f})")
exc = [r.get("exclusive_check", {}) for r in recs]
print("  exclusive_ok:", all(e.get("ok") for e in exc if e))
PYEOF
