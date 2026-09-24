#!/usr/bin/env bash
# CED tiny 1+1 的 SWA-clip A/B 编排（在 a3-22 上跑；只停/起我们自己的容器）。
#
# 一次调用跑一个 arm：
#   CLIP=1 DEVS=7 bash tools/ced_tiny_clip_ab.sh arm_clip_on    # 修复臂（默认）
#   CLIP=0 DEVS=7 bash tools/ced_tiny_clip_ab.sh arm_clip_off   # 旧行为对照臂
#
# 每个 arm 会：起 D（图模式）→ 起 proxy → 发 filler 把 D 的块池推过池尾 →
# 对同一请求重复 N 次（逐轮落盘）→ 摘 [CED-BLOCKS]（descents）与
# [CED-SWA-CLIP] → 跑完停掉本 arm 的容器。产物落在 results/<tag>/。
#
# 安全：chip0/chip1 直接拒绝；DEVS 必须显式给出。HBM/他人进程的守护交给
# tools/ced_phy_watchdog.py（单独起，见 evidence README）。
set -euo pipefail

PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TAG=${1:?用法: CLIP=1 DEVS=7 $0 <tag>}
CLIP=${CLIP:-1}
DEVS=${DEVS:?必须显式给 DEVS（例如 DEVS=7），脚本不替你选卡}
PORT=${PORT:-18961}
KV_PORT=${KV_PORT:-19061}
PROXY_PORT=${PROXY_PORT:-18962}
PREFILL_PORT=${PREFILL_PORT:-18960}
REPEATS=${REPEATS:-40}
FILLERS=${FILLERS:-0}
FILLER_LEN=${FILLER_LEN:-2000}
MODEL_LEN=${MODEL_LEN:-2000}
OUT=${OUT:-$PKG/results/$TAG}

for dev in $DEVS; do
  case "$dev" in
    0|1) echo "[ab][FAIL] chip0/chip1 预留，不能用（DEVS=$DEVS）" >&2; exit 2 ;;
    ''|*[!0-9]*) echo "[ab][FAIL] DEVS 非法项：'$dev'" >&2; exit 2 ;;
  esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
RN="ced_tiny_ab_${TAG}_${STAMP}"
DN="dsv41-ced-tiny-ab-d-${TAG}-${STAMP}"
PN="dsv41-ced-tiny-ab-p-${TAG}-${STAMP}"
XN="dsv41-ced-tiny-ab-proxy-${TAG}-${STAMP}"
mkdir -p "$OUT"
echo "[ab] tag=$TAG clip=$CLIP devs='$DEVS' out=$OUT"
printf '{"clip": %s, "devs": "%s"}\n' "$CLIP" "$DEVS" > "$OUT/arm.json"

cd "$PKG"
ALLOW_BUSY=1 DEVS="$DEVS" PORT="$PORT" KV_PORT="$KV_PORT" \
  V41_CED_SWA_CLIP="$CLIP" V41_CED_SWA_TRACE=1 V41_CED_BLOCK_TRACE=1 \
  RUN_ID="$RN" NAME="$DN" \
  nohup bash tools/launch_ced_tiny_d.sh > "$OUT/d_launch.log" 2>&1 &

P_STARTED=0
if ! curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PREFILL_PORT/health"; then
  ALLOW_BUSY=1 DEVS="${P_DEVS:-6}" PORT="$PREFILL_PORT" KV_PORT="${P_KV_PORT:-19060}" \
    V41_CED_BLOCK_TRACE=1 RUN_ID="ced_tiny_ab_p_${STAMP}" NAME="$PN" \
    nohup bash scripts/serve_a3_ced_single.sh prefill > "$OUT/p_launch.log" 2>&1 &
  P_STARTED=1
else
  echo "[ab] 复用已在跑的 P（$PREFILL_PORT）"
fi

cleanup() {
  echo "[ab] cleanup"
  docker stop -t 30 "$XN" "$DN" >/dev/null 2>&1 || true
  if [ "$P_STARTED" = "1" ]; then docker stop -t 60 "$PN" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

for i in $(seq 1 120); do
  pd=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$PREFILL_PORT/health" || true)
  dd=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$PORT/health" || true)
  if [ "$pd" = "200" ] && [ "$dd" = "200" ]; then echo "[ab] P/D ready after $((i * 10))s"; break; fi
  if [ "$i" = "120" ]; then
    echo "[ab][FAIL] P/D 未就绪（P=$pd D=$dd）" >&2
    tail -5 "$OUT/d_launch.log" >&2
    exit 4
  fi
  sleep 10
done

NAME="$XN" PROXY_PORT="$PROXY_PORT" PREFILL_PORT="$PREFILL_PORT" DECODE_PORT="$PORT" \
  bash scripts/serve_a3_pd_proxy.sh > "$OUT/proxy.log" 2>&1
sleep 10

python3 tools/ced_repeat_probe.py \
  --url "http://127.0.0.1:$PROXY_PORT/v1/completions" \
  --length "$MODEL_LEN" --repeats "$REPEATS" --filler-count "$FILLERS" \
  --filler-length "$FILLER_LEN" --restart-container "$XN" \
  --out "$OUT/repeat.jsonl" 2>&1 | tee "$OUT/repeat.log"

LOG="$PKG/results/$RN/serve.log"
if [ -f "$LOG" ]; then
  grep "\[CED-BLOCKS\] role=decode" "$LOG" | tail -n $((REPEATS + FILLERS + 5)) > "$OUT/block_trace.txt" || true
  grep "\[CED-SWA-CLIP\]" "$LOG" | tail -n 40 > "$OUT/clip_trace.txt" || true
fi

python3 - "$OUT" <<'PY'
import json, re, sys
out = sys.argv[1]
rows = [json.loads(line) for line in open(f"{out}/repeat.jsonl") if line.strip()]
first = [r.get("first_token", {}).get("logprob") for r in rows]
texts = [r.get("text") for r in rows]
desc = []
try:
    for line in open(f"{out}/block_trace.txt"):
        m = re.search(r"g0:\(n=(\d+) first=(\d+) last=(\d+) descents=(\d+)\)", line)
        if m:
            desc.append(int(m.group(4)))
except FileNotFoundError:
    pass
summary = {
    "tag": out.rsplit("/", 1)[-1],
    "requests": len(rows),
    "http_200": sum(1 for r in rows if r.get("http_status") == 200),
    "unique_texts": sorted({t for t in texts if t is not None}),
    "unique_first_logprobs": sorted({lp for lp in first if lp is not None}),
    "g0_descents": desc,
    "g0_descents_positive": sum(1 for d in desc if d > 0),
}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("判据：碎片 arm 需 g0_descents_positive>0、http_200==requests，且 successful 与 fragmented 两臂 unique_texts 一致")
PY
echo "[ab] arm $TAG 完成，产物在 $OUT"
