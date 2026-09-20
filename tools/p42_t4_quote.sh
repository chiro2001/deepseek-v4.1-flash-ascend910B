#!/usr/bin/env bash
# p42_t4_quote.sh -- T4 (512Ki, 逐字引用 quote 口径) measurement against an ALREADY
# RUNNING server (e.g. the one P42 leaves up with KEEP_UP=1).  It does NOT start or
# stop any container; it only issues single-stream requests and samples /metrics.
#
#   TAG=delta_b1024 TOKENS=524288 REPEATS=2 bash logs/perf/p42_t4_quote.sh
#   # 8K 对照点（同口径）：
#   TAG=delta_b1024 TOKENS=8192  REPEATS=1 bash logs/perf/p42_t4_quote.sh
#
# env: TAG TOKENS MAXTOK REPEATS URL PREFIX SUFFIX OUTDIR
set -uo pipefail
# [SERVED_NAME] API 请求里的 `"model"` 字段。与起服时的 --served-model-name 一致；
# 默认 deepseek-v41（向后兼容）。改服务名时同步设它，否则请求会 404。
SERVED_NAME=${SERVED_NAME:-deepseek-v41}
P=${P:-/home/user/projects/dsv41}
OUTDIR=${OUTDIR:-$P/logs/perf/p36_phase0}
TAG=${TAG:-delta_b1024}
TOKENS=${TOKENS:-524288}
MAXTOK=${MAXTOK:-448}
REPEATS=${REPEATS:-2}
URL=${URL:-http://127.0.0.1:8001}
PREFIX=${PREFIX:-$P/data/hongloumeng.txt}
SUFFIX=${SUFFIX:-$P/a2_package/data/hlm/suffix_quote.txt}
OUTJ=$OUTDIR/p42_t4_quote_${TOKENS}_${TAG}.jsonl
mkdir -p "$OUTDIR"

code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$URL/health" || true)
if [ "$code" != "200" ]; then echo "[p42t4] ABORT: server not healthy ($code) at $URL"; exit 3; fi
[ -f "$PREFIX" ] || { echo "[p42t4] ABORT: missing prefix $PREFIX"; exit 3; }
[ -f "$SUFFIX" ] || { echo "[p42t4] ABORT: missing suffix $SUFFIX"; exit 3; }
rm -f "$OUTJ"
echo "[p42t4] tag=$TAG tokens=$TOKENS max_tokens=$MAXTOK repeats=$REPEATS prefix=$(basename "$PREFIX") suffix=$(basename "$SUFFIX")"
for i in $(seq 1 "$REPEATS"); do
  echo "[p42t4] ---- point $i/$REPEATS $(date '+%F %T') ----"
  # [PATH-FIX 2026-09-18] 这里原来写死 `$P/logs/perf/p15_stream_curve_filefiller.py`，
  # 那是历史工作区的布局；在本交付包里该文件位于 `tests/`。
  # 改为按「脚本自身所在包根 → $P → 历史路径」依次探测，找不到就明确报错。
  FILLER=""
  for _c in \
      "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/p15_stream_curve_filefiller.py" \
      "$P/tests/p15_stream_curve_filefiller.py" \
      "$P/logs/perf/p15_stream_curve_filefiller.py" ; do
    [ -f "$_c" ] && { FILLER="$_c"; break; }
  done
  if [ -z "$FILLER" ]; then
    echo "[p42t4] ABORT: 找不到 p15_stream_curve_filefiller.py。" >&2
    echo "  期望它在 <包根>/tests/ 下（当前 P=$P）。可用 P=<包根> 显式指定。" >&2
    exit 3
  fi
  PYHOST=${PYHOST:-python3}
  "$PYHOST" "$FILLER" \
    --base-url "$URL" --model "$SERVED_NAME" \
    --tokens "$TOKENS" --max-tokens "$MAXTOK" \
    --warmup-tokens 1 --warmup-output-tokens 24 \
    --prefix-file "$PREFIX" --suffix-file "$SUFFIX" \
    --corpus-label "p42_quote_${TOKENS}_r${i}" --out "$OUTJ" \
    --max-running 1.0 --sample-interval 0.25 --idle-timeout 900 2>&1 | tail -6
done

python3 - "$OUTJ" "$TOKENS" <<'PY'
import json, statistics, sys
path, target = sys.argv[1], int(sys.argv[2])
recs = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
ok = [r for r in recs if r.get("metrics_ok") and not r.get("error")]
print(f"[p42t4] points={len(recs)} usable={len(ok)} target_tokens={target}")
def col(name):
    return [r[name] for r in ok if r.get(name) is not None]
for name in ("prompt_tokens_actual", "ttft_s", "prefill_tok_s", "ms_per_step", "tok_per_step",
             "decode_tok_s", "accept_length", "n_sse_chunks", "e2e_tok_s"):
    vals = col(name)
    if not vals:
        print(f"  {name}: n/a"); continue
    med = statistics.median(vals)
    print(f"  {name}: median={med} min={min(vals)} max={max(vals)} n={len(vals)}")
exc = [r.get("exclusive_check", {}) for r in recs]
print("  exclusive_ok:", all(e.get("ok") for e in exc if e), [e.get("max_running") for e in exc])
warn = [w for r in recs for w in (r.get("warnings") or [])]
if warn:
    print("  warnings:", warn[:6])
PY
