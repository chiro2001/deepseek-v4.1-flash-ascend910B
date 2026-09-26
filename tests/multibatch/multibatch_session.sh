#!/usr/bin/env bash
# =============================================================================
# multibatch_session.sh -- 用**生产口径**起服，跑「多 batch / 多轮对话」三块验证。
#
#   生产口径 = MAX_SEQS=32 + PREFIX=1（A2 真机：`--max-num-seqs 32` + prefix caching ON）
#   历史性能口径 = MAX_SEQS=1 + PREFIX=0 —— **两者不可混比**（A/ms 都不是一回事）。
#
# 用法（宿主机，在包根目录）：
#   MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq TAG=mbg_p1 bash tests/multibatch/multibatch_session.sh
#   MODEL=... TAG=mbg_p0 PREFIX=0 bash tests/multibatch/multibatch_session.sh
#
# 变量：
#   MODEL      必填（模型目录）      CONC    并发 item 数（默认 = MAX_SEQS，即 32 时取 8）
#   MAX_SEQS   默认 32（生产）       ROUNDS  多轮对话轮数（默认 8）
#   PREFIX     默认 1（生产）        LONG_CTX [C] 长请求 token 数（默认 131072；0=跳过）
#   TAG        run_id / 输出目录名   SKIP    传给 multibatch_gate.py 的块（如 "A,C"）
#   STOP_FIRST 默认 0；=1 时才 `docker rm -f $NAME`（避免误杀别人正在跑的服务）
#   DOCKER     默认自动探测（docker / sudo -n docker）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"

MODEL=${MODEL:-}
TAG=${TAG:-mbg_p1}
MAX_SEQS=${MAX_SEQS:-32}
PREFIX=${PREFIX:-1}
CONC=${CONC:-8}
ROUNDS=${ROUNDS:-8}
LONG_CTX=${LONG_CTX:-131072}
SKIP=${SKIP:-}
NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8100}
STOP_FIRST=${STOP_FIRST:-0}
URL="http://127.0.0.1:$PORT"
OUT=$PKG/results/$TAG
LOG=$OUT/serve.log
WAIT_MIN=${WAIT_MIN:-40}

say() { printf '[mbg %s] %s\n' "$(date +%T)" "$*"; }

[ -n "$MODEL" ] || { echo "必须设置 MODEL=<模型目录>"; exit 2; }

DOCKER=${DOCKER:-docker}
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo -n docker"
$DOCKER info >/dev/null 2>&1 || { echo "无法访问 docker"; exit 2; }

mkdir -p "$OUT"
say "===== $TAG MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX CONC=$CONC ROUNDS=$ROUNDS ====="

if [ "$STOP_FIRST" = "1" ]; then
  say "STOP_FIRST=1 ⇒ 停掉旧容器 $NAME"
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 4
fi

# ---------- 起服（生产口径）----------
# 注意事项：MAX_SEQS=32 时 CAPTURE_SIZES 会扩到 1,2,3,4,6,8,12,16,20,24,32,40,48,96,192（15 桶），
# 首次起服的图捕获比单流口径慢得多（每桶 ~10-30 s）。serve_a2.sh 会自动推导，**不要手写**。
MODEL="$MODEL" NAME="$NAME" PORT="$PORT" MAX_SEQS="$MAX_SEQS" PREFIX="$PREFIX" \
  RUN_ID="$TAG" OUT="$OUT" LOG="$LOG" WAIT_READY=1 READY_TIMEOUT=${READY_TIMEOUT:-3000} \
  bash "$PKG/scripts/serve_a2.sh" || { say "$TAG 起服失败"; exit 1; }

code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)
code=${code:-000}   # 不写 `|| echo 000`：那会拼成 000000
[ "$code" = "200" ] || { say "$TAG 未就绪（health=$code）"; tail -8 "$LOG" | sed 's/^/    /'; exit 1; }

say "$TAG READY degrade=$(grep -ac 'static_kernel.py:650' "$LOG") $(grep -m1 -o 'GPU KV cache size: [0-9,]*' "$LOG")"
grep -E "max-num-seqs|enable_prefix_caching|Capture|capture_sizes" "$LOG" 2>/dev/null | head -4 | sed 's/^/    /'
say "口径确认：$(grep -m1 'serve_a2\] 口径' "$LOG" || echo '（见 serve_cmd.txt）')"
sleep 20

# ---------- 三块验证 ----------
say "--- [A] 多轮对话 / [B] 并发逐 item / [C] 长短交错 ---"
PYHOST=${PYHOST:-$(command -v python3)}
"$PYHOST" "$PKG/tests/multibatch/multibatch_gate.py" --base "$URL" --model deepseek-v41 \
  --out "$OUT" --rounds "$ROUNDS" --conc "$CONC" --long-ctx "$LONG_CTX" \
  ${SKIP:+--skip "$SKIP"} 2>&1 | tee -a "$OUT/mbg.log" | sed 's/^/  /'
RC=${PIPESTATUS[0]}
say "$TAG done rc=$RC out=$OUT"
exit "$RC"
