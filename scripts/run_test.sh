#!/usr/bin/env bash
# =============================================================================
# 【命令 ②】一次执行就开始测试推理
#
#   MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/run_test.sh
#
# 流程（全自动，失败即停并打印原因）：
#   [0] 预检     docker / 镜像 / 模型目录完整性 / 芯片占用
#   [1] 起服     serve_a2.sh（TP8 + 图模式主干 + Engram int8 host + DSpark S=5 + Vision）
#   [2] 等就绪   每 15 s 报进度，35 min 超时
#   [3] 必查     ✓ static_kernel 未被静默降级（static_kernel.py:650 == 0）
#                ✓ Engram local-owner validate 自检
#                ✓ KV 容量 > $KV_MIN（默认 2,800,000，随默认 GPU_UTIL=0.92 而定）
#   [4] 性能     quote 口径单流：8K / 32K（MODE=full 再加 128K）
#               记录 ms/step、接受长度 A、tok/s
#   [5] 视觉     23 例图文问答（≥19 通过为达标；需要官方图片目录）
#   [6] 精度     GSM8K-200（RUN_GSM8K=1 或 MODE=full）
#   [7] 报告     results/<run_id>/REPORT.md + 全部原始 jsonl
#
# 常用变量：
#   MODEL       必填；IMAGE 默认 dsv41-a2:v6；PORT 默认 8100
#   MODE        quick(默认)=8K+32K+vision | full=+128K+GSM8K | prod=生产口径 + 多 batch 三块
#               prod = MAX_SEQS=32 + PREFIX=1（A2 真机口径）：只跑 必查三项 + [A]多轮/[B]并发/[C]长短交错，
#               **不跑 quote 性能**（生产口径的 ms/A 与单流口径不可比）。两臂对照见
#               tests/multibatch/run_prod_both.sh（PREFIX=1 与 PREFIX=0 各一个会话）。
#   PREFIX      0(默认，性能口径) | 1(生产口径：prefix caching ON)
#   MAX_SEQS    默认 4；MODE=prod 时默认抬到 32（等于生产）
#   GPU_UTIL    默认 0.92（留 activation 余量；0.94 会让 prefill 慢 2.1×）
#   LOAD_FORMAT dummy = 只测时延（**A 恒为 1.0，不能用于精度/容量判据**）
#   PYTHON_PGO  默认 1（编译好的 libpython，宿主 CPU 弱时更值；自动降级）
#   MOE_ZERO / DRAFT_GRAPH 默认 0 —— **未验证**，只在实验时打开（见 CHANGELOG）
#   OFFICIAL_DIR 官方 checkpoint 目录（vision 用例图片从这里取）
#   ENC_DIR      官方 `encoding` 目录（GSM8K 的 chat 模板；默认自动找
#                ~/models/DeepSeek-V4.1-Flash/encoding，找不到就跳过 GSM8K）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

MODEL=${MODEL:-}
IMAGE=${IMAGE:-dsv41-a2:v6}
PORT=${PORT:-8100}
MODE=${MODE:-quick}
TP=${TP:-8}
# 见 serve_a2.sh 的 §[MEM-HEADROOM]：0.94 会让真实 prefill 的 activation
# 贴住显存上限、forward 慢 2.1×；0.92 是实测的拐点，KV 少 8.6% 但 prefill 快 6~7×。
GPU_UTIL=${GPU_UTIL:-0.92}
MAX_LEN=${MAX_LEN:-1048576}
# 口径：quick/full = **无前缀缓存**的单流性能口径；prod = 生产口径（max-num-seqs 32 + prefix ON）
# 两者**不可混比**（A/ms 都不是一回事），所以这里按 MODE 给不同默认值，显式传参优先。
# 注意：`serve_*.sh` 的用户默认现在是 PREFIX=1（生产形态），所以性能臂必须**显式**
# 用 NO_PREFIX=1 关掉，不能靠"不写就是关"。
MAX_SEQS=${MAX_SEQS:-}
PREFIX=${PREFIX:-}
BAT_TOKENS=${BAT_TOKENS:-2048}
STATIC_KERNEL=${STATIC_KERNEL:-1}
SP_TOKENS=${SP_TOKENS:-5}
LOCAL_OWNER=${LOCAL_OWNER:-fast}
PYTHON_PGO=${PYTHON_PGO:-1}
LOAD_FORMAT=${LOAD_FORMAT:-}
MOE_ZERO=${MOE_ZERO:-0}
DRAFT_GRAPH=${DRAFT_GRAPH:-0}
OFFICIAL_DIR=${OFFICIAL_DIR:-}
RUN_GSM8K=${RUN_GSM8K:-0}
NAME=${NAME:-dsv41-a2}
CHIPS=${CHIPS:-"0 1 2 3 4 5 6 7"}
REPEATS=${REPEATS:-2}
RUN_ID=${RUN_ID:-a2_$(date +%Y%m%d_%H%M%S)}
OUT=$PKG/results/$RUN_ID
LOG=$OUT/serve.log
mkdir -p "$OUT"

say()  { printf '\n\033[1m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
die()  { bad "$*"; say "中止。日志：$LOG"; exit 1; }

choose_py() {
  for c in "$HOME/venvs/lmeval311/bin/python" "$HOME/venvs/lmeval/bin/python" \
           /home/*/venvs/lmeval311/bin/python "$(command -v python3)"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo python3
}
# [PYHOST-OVERRIDE] v5 是无条件覆盖，export PYHOST=... 会被静默忽略。
# v6：显式传入优先（用户想指定带 datasets 的解释器时就靠这个）。
PYHOST=${PYHOST:-$(choose_py)}

# ============================== [MODE=prod] 生产口径 ==============================
# A2 真机：`--max-num-seqs 32` + prefix caching ON。v3 起服器沿用的是**单流口径**
# （MAX_SEQS=1、无 prefix），所以"生产形态"（decode 队列里随时插入新请求 +
# 128K chunked prefill 与短请求 decode 混进同一个 batch step）此前从未跑过。
if [ "$MODE" = "prod" ]; then
  MAX_SEQS=${MAX_SEQS:-32}
  PREFIX=${PREFIX:-1}
  echo "prod 口径：MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX（quote 性能段跳过）" >> "$OUT/driver.log"
else
  MAX_SEQS=${MAX_SEQS:-4}
  # 性能口径：显式关掉 prefix caching（等价 PREFIX=0）。
  if [ -z "$PREFIX" ]; then
    NO_PREFIX=1
  fi
fi

# ============================== [0] 预检 ==============================
say "预检"
DOCKER="docker"
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo -n docker"
$DOCKER info >/dev/null 2>&1 || die "无法访问 docker（试过 docker 与 sudo -n docker）"
ok "docker 可用"
$DOCKER image inspect "$IMAGE" >/dev/null 2>&1 || die "镜像 $IMAGE 不存在，请先：bash scripts/build_image.sh"
ok "镜像 $IMAGE 存在"
[ -n "$MODEL" ] || die "必须指定 MODEL=<模型目录>"
[ -d "$MODEL" ] || die "模型目录不存在：$MODEL"
ok "模型目录：$MODEL"

if [ -x "$PKG/tools/check_model_dir.sh" ]; then
  bash "$PKG/tools/check_model_dir.sh" "$MODEL" | sed 's/^/  /' | tee -a "$OUT/driver.log"
  CHK_RC=${PIPESTATUS[0]}
  [ "$CHK_RC" = "1" ] && [ "${SKIP_MODEL_CHECK:-0}" != "1" ] \
    && die "模型自检不通过（见上）。修好后再跑，或 SKIP_MODEL_CHECK=1 强制继续"
fi

ENGRAM_ON=1
python3 - "$MODEL" <<'PY' || ENGRAM_ON=0
import json, os, sys
m = sys.argv[1]
try:
    c = json.load(open(os.path.join(m, "config.json")))
    tc = c.get("text_config") or {}
    layers = list(tc.get("engram_layer_ids") or [])
    has_w = os.path.exists(os.path.join(m, "engram_extra.safetensors")) or os.path.exists(os.path.join(m, "engram_int8"))
    sys.exit(0 if (layers and has_w) else 1)
except Exception:
    sys.exit(1)
PY
if [ "$ENGRAM_ON" = "1" ]; then ok "检测到 Engram（enable_engram=true）"
else bad "模型目录无 Engram ⇒ 本次 enable_engram=false"; fi
echo "engram_on=$ENGRAM_ON" >> "$OUT/env.txt"

if [ -z "$OFFICIAL_DIR" ]; then
  for c in "$HOME/models/DeepSeek-V4.1-Flash" /home/*/models/DeepSeek-V4.1-Flash; do
    [ -d "$c" ] && { OFFICIAL_DIR="$c"; break; }
  done
fi
[ -n "$OFFICIAL_DIR" ] && [ -d "$OFFICIAL_DIR" ] && ok "官方目录：$OFFICIAL_DIR" \
  || bad "未找到官方 checkpoint 目录（视觉用例会跳过）；可用 OFFICIAL_DIR=... 指定"

if command -v npu-smi >/dev/null 2>&1; then
  nested=0
  hbm_line=$(npu-smi info 2>/dev/null | grep -E "^[|] [01] +[0-9]+ " | head -16)
  while read -r line; do
    chip=$(echo "$line" | awk '{print $2}')
    used=$(echo "$line" | grep -oE "[0-9]+ */ *65536" | head -1 | tr -dc '0-9' | head -c4)
    if [ -n "${used:-}" ] && [ "${used:-0}" -gt 20000 ] 2>/dev/null; then
      case " $CHIPS " in *" $chip "*) nested=$((nested+1));; esac
    fi
  done <<< "$hbm_line"
  [ "$nested" -gt 0 ] && die "目标芯片（$CHIPS）里已有 $nested 个 HBM 占用 >20 GB，请先停掉其它服务"
  ok "目标芯片（$CHIPS）HBM 空闲"
else
  bad "没有 npu-smi（本脚本应在**宿主机**上执行，不要在容器内跑）"
fi

SINGLE_STREAM_NOTE="真权重"
[ "$LOAD_FORMAT" = "dummy" ] && SINGLE_STREAM_NOTE="dummy（A 恒 1.0，仅测时延）"

# ============================== [1][2][3] 起服 + 就绪 + 必查项 ==============================
say "起服（模式：$SINGLE_STREAM_NOTE；sptok=$SP_TOKENS；mseqs=$MAX_SEQS；prefix=$PREFIX；PGO=$PYTHON_PGO）"
MODEL="$MODEL" IMAGE="$IMAGE" NAME="$NAME" PORT="$PORT" TP="$TP" GPU_UTIL="$GPU_UTIL" \
  MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" BAT_TOKENS="$BAT_TOKENS" STATIC_KERNEL="$STATIC_KERNEL" \
  SP_TOKENS="$SP_TOKENS" LOCAL_OWNER="$LOCAL_OWNER" PYTHON_PGO="$PYTHON_PGO" LOAD_FORMAT="$LOAD_FORMAT" \
  MOE_ZERO="$MOE_ZERO" DRAFT_GRAPH="$DRAFT_GRAPH" ENGRAM="$ENGRAM_ON" VISION=1 \
  PREFIX="$PREFIX" NO_PREFIX="${NO_PREFIX:-0}" \
  CPUSET="${CPUSET:--1}" MEMS="${MEMS:--1}" CPU_BIND="${CPU_BIND:-1}" \
  SKCACHE_GC="${SKCACHE_GC:-1}" CACHE="${CACHE:-}" MOE_NF="${MOE_NF:-0}" \
  RUN_ID="$RUN_ID" OUT="$OUT" LOG="$LOG" WAIT_READY=1 \
  bash "$HERE/serve_a2.sh" || die "serve_a2.sh 失败（见上）"

# --- 必查 ① static_kernel 静默降级 ---
SK_HITS=$(grep -ac "static_kernel.py:650" "$LOG" 2>/dev/null || echo 0)
if [ "${SK_HITS:-0}" != "0" ]; then
  bad "static_kernel 被静默降级（static_kernel.py:650 命中 ${SK_HITS} 次）—— 结果不可信"
  grep -n "static_kernel.py:650" "$LOG" | head -3 | sed 's/^/    /'
  SK_PASS=0
else
  ok "static_kernel 未被降级（static_kernel.py:650 命中 0 次）"
  SK_PASS=1
fi
echo "static_kernel_degrade_hits=$SK_HITS" >> "$OUT/env.txt"

# --- 必查 ② local-owner validate ---
ask() { curl -s -m "${2:-180}" "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"deepseek-v41\",\"prompt\":$1,\"max_tokens\":${3:-16},\"temperature\":0.0}"; }
if [ "$ENGRAM_ON" = "1" ]; then
  say "Engram local-owner 自检（validate → 全对则切 fast）"
  $DOCKER exec "$NAME" bash -lc "printf '%s' validate > /tmp/v41_engram_localowner" 2>/dev/null || true
  ask '"The capital of France is"' 180 16 >/dev/null 2>&1 || true
  sleep 2
  if grep -q "local-owner] VALIDATE OK" "$LOG"; then
    ok "validate 通过 → 切 fast（最快路径）"
    $DOCKER exec "$NAME" bash -lc "printf '%s' fast > /tmp/v41_engram_localowner"
    LO_EFF=fast
  elif grep -q "local-owner] VALIDATE FAILED" "$LOG"; then
    bad "validate 失败 → 退回 on（功能正确，慢约 1 ms/step）"
    $DOCKER exec "$NAME" bash -lc "printf '%s' on > /tmp/v41_engram_localowner"
    LO_EFF=on
  else
    bad "未看到 validate 结果 → 保持 $LOCAL_OWNER"
    LO_EFF=$LOCAL_OWNER
  fi
  echo "$LO_EFF" > "$OUT/local_owner_effective.txt"
else
  LO_EFF=n/a; echo "n/a" > "$OUT/local_owner_effective.txt"
fi

# --- 必查 ③ KV 容量 ---
# 门槛跟着**默认 GPU_UTIL** 走，否则默认配置会必然判 FAIL：
#   GPU_UTIL=0.92（现默认）→ 约 2.82M tokens   ← 门槛取 2,800,000
#   GPU_UTIL=0.94（旧默认）→ 3,088,412 tokens（>3M），但长 prompt 的 prefill 慢 6~7×
# 若你的场景确实要卡 3Mi 门槛，显式传 KV_MIN=3145728 并同时设 GPU_UTIL=0.94。
KV_MIN=${KV_MIN:-2800000}
say "容量检查（门槛 $KV_MIN tokens；GPU_UTIL=$GPU_UTIL）"
KV=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | tr -dc '0-9')
KV=${KV:-0}
if [ "$KV" -gt "$KV_MIN" ]; then
  ok "KV $KV tokens > $KV_MIN"
  KV_PASS=1
else
  bad "KV $KV tokens ≤ $KV_MIN"
  KV_PASS=0
  echo "     提示：默认 GPU_UTIL=0.92 下 KV 约 2.82M 是**有意取舍**（换 prefill 快 6~7×）。"
  echo "           要更大 KV 请设 GPU_UTIL=0.94，但长 prompt 首 token 会从 1.1 s 涨到 8 s。"
fi
printf 'kv_tokens=%s\nkv_pass=%s\nmax_seqs=%s\nprefix=%s\nmode=%s\n' \
  "$KV" "$KV_PASS" "$MAX_SEQS" "$PREFIX" "$MODE" >> "$OUT/env.txt"

# ============================== [4] 多 batch / 多轮对话（MODE=prod）==========
if [ "$MODE" = "prod" ]; then
  say "多 batch / 多轮对话三块验证（生产口径 MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX）"
  "$PYHOST" "$PKG/tests/multibatch/multibatch_gate.py" \
      --base "http://127.0.0.1:$PORT" --model deepseek-v41 --out "$OUT" \
      --rounds "${MBG_ROUNDS:-8}" --conc "${MBG_CONC:-8}" \
      --long-ctx "${MBG_LONG_CTX:-131072}" ${MBG_SKIP:+--skip "$MBG_SKIP"} \
      2>&1 | tail -30 | sed 's/^/  /' | tee -a "$OUT/driver.log"
  MBG_RC=${PIPESTATUS[0]}
  [ "$MBG_RC" = "0" ] && ok "多 batch 三块全部 PASS" \
    || bad "多 batch 有 FAIL 项（rc=$MBG_RC）—— 看 $OUT/summary.json 的 verdict"
fi

# ============================== [4b] 性能（单流口径）==============================
quote() { # ctx tag
  local ctx=$1 tag=$2
  say "quote 口径 ${ctx} tokens（${REPEATS} 发）…"
  P=$PKG TAG="$tag" TOKENS="$ctx" MAXTOK=$([ "$ctx" -ge 131072 ] && echo 256 || echo 256) \
    REPEATS="$REPEATS" URL="http://127.0.0.1:$PORT" OUTDIR="$OUT" \
    PREFIX="$PKG/data/hongloumeng.txt" SUFFIX="$PKG/data/hlm/suffix_quote.txt" \
    PYHOST="$PYHOST" bash "$PKG/tests/t_quote.sh" 2>&1 | tail -14 | sed 's/^/  /' | tee -a "$OUT/driver.log"
}
if [ "$MODE" != "prod" ]; then
  quote 8192 quote_8k
  quote 32768 quote_32k
  [ "$MODE" = "full" ] && quote 131072 quote_128k
fi

# ============================== [5] 视觉 ==============================
if [ -n "$OFFICIAL_DIR" ] && [ -d "$OFFICIAL_DIR/inference/examples/images" ]; then
  say "视觉 23 例 …"
  "$PYHOST" "$PKG/tests/t_vision.py" \
      --server "http://127.0.0.1:$PORT" --model deepseek-v41 \
      --images-dir "$OFFICIAL_DIR/inference/examples/images" \
      --out "$OUT/vision.json" 2>&1 | tail -10 | sed 's/^/  /' | tee -a "$OUT/driver.log"
else
  bad "跳过视觉（未找到官方图片目录：$OFFICIAL_DIR/inference/examples/images）"
fi

# ============================== [6] GSM8K（可选）==============================
if [ "$RUN_GSM8K" = "1" ] || [ "$MODE" = "full" ]; then
  say "GSM8K-200 …"
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYHOST" "$PKG/tests/t_gsm8k.py" --base "http://127.0.0.1:$PORT" \
    --limit 200 --conc 4 ${ENC_DIR:+--enc-dir "$ENC_DIR"} \
    --out "$OUT/gsm8k.json" 2>&1 | tail -6 | sed 's/^/  /' | tee -a "$OUT/driver.log"
fi

# ============================== [7] 报告 ==============================
say "生成报告"
bash "$PKG/tests/make_report.sh" "$OUT" "$RUN_ID" "$MODEL" "$LO_EFF" 2>&1 | tail -40
say "完成。产出目录：$OUT"
echo
echo "  报告：      less $OUT/REPORT.md"
echo "  服务仍在跑：$DOCKER exec -it $NAME bash"
echo "  停止服务：  $DOCKER rm -f $NAME"
