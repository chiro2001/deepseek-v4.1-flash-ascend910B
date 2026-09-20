#!/usr/bin/env bash
# =============================================================================
# 【附着测试】服务已经在跑 ⇒ 只跑 [3]–[7]，**不起容器、不删容器**
#
#   PORT=8077 bash tools/attach_test.sh
#
# 为什么需要它：run_test.sh 的 [0] 预检会在目标芯片 HBM 占用 >20 GB 时直接 die
# （服务已起时必然命中），而且 [1] 无条件调 serve_a2.sh，后者开头就 `docker rm -f`
# ⇒ 会把你正在跑的服务杀掉重建。[3]–[7] 其实都是纯客户端，本脚本把它们摘出来。
#
# 做的事（与 run_test.sh 逐条同口径）：
#   [3] 必查   static_kernel 未降级 / Engram local-owner validate / KV > 3Mi
#              （日志里若有 DRAFT_GRAPH 指纹则一并打印）
#   [4] 性能   quote 单流 8K / 32K（MODE=full 再加 128K）
#   [5] 视觉   23 例（需官方图片目录）
#   [6] 精度   GSM8K-200（MODE=full 或 RUN_GSM8K=1）
#   [7] 报告   results/<run_id>/REPORT.md
#
# 常用变量：
#   PORT       服务端口（默认 8100）—— 你起服务时传的那个
#   NAME       容器名（默认 dsv41-a2）—— 只用于（a）local-owner validate（b）日志兜底
#   SLOG       服务日志路径；默认自动找最新的 results/*/serve.log
#   MODEL      模型目录（仅用于 [0'] Engram 探测与报告文案；不传则跳过探测）
#   OUT/RUN_ID 结果目录；默认 results/attach_<时间戳>，**不覆盖原目录**
#   MODE       quick(默认)=8K+32K | full=+128K+GSM8K
#   SKIP       逗号分隔，跳过某几段：quote,vision,gsm8k,report,lo（local-owner）
#   REPEATS    quote 每点发数（默认 2，与 run_test.sh 一致）
#   DOCKER     默认自动探测 docker / sudo -n docker；纯客户端段不需要它
#   PYHOST     跑 python 测试的解释器（需带 datasets 5.0.1）；默认自动找
#   OFFICIAL_DIR / ENC_DIR  官方 checkpoint / encoding 目录
#
# 注意：本脚本**只读**服务与容器，唯一的写操作是
#   ① 往结果目录写 jsonl/report ② 按需写容器内 /tmp/v41_engram_localowner（可 SKIP=lo 关掉）
# =============================================================================
set -uo pipefail
# [SERVED_NAME] API 请求里的 model 字段；默认 deepseek-v41（向后兼容）。
SERVED_NAME=${SERVED_NAME:-deepseek-v41}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

PORT=${PORT:-8100}
NAME=${NAME:-dsv41-a2}
SLOG=${SLOG:-}
MODEL=${MODEL:-}
MODE=${MODE:-quick}
SKIP=${SKIP:-}
REPEATS=${REPEATS:-2}
RUN_ID=${RUN_ID:-attach_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$PKG/results/$RUN_ID}
OFFICIAL_DIR=${OFFICIAL_DIR:-}
ENC_DIR=${ENC_DIR:-}
RUN_GSM8K=${RUN_GSM8K:-0}

say()  { printf '\n\033[1m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
note() { printf '  \033[33m·\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
die()  { bad "$*"; say "中止。结果目录：$OUT"; exit 1; }

skipped() { case ",${SKIP}," in *",$1,"*) return 0;; *) return 1;; esac; }

mkdir -p "$OUT"

# ---------- 解释器 ----------
choose_py() {
  for c in "$HOME/venvs/lmeval311/bin/python" "$HOME/venvs/lmeval/bin/python" \
           /home/*/venvs/lmeval311/bin/python "$(command -v python3)"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo python3
}
PYHOST=${PYHOST:-$(choose_py)}

# ---------- docker（可选） ----------
DOCKER=""
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then DOCKER="docker"
  elif sudo -n docker info >/dev/null 2>&1; then DOCKER="sudo -n docker"; fi
fi

# ============================== [0'] 附着预检 ==============================
say "附着预检（服务应已在运行；本脚本不会起/删任何容器）"

# 注意：curl 的 -w '%{http_code}' 在连接失败时**自己就会打印 000** 并以非 0 退出；
# 若再 `|| echo 000` 会得到 "000\n000"（v6 的 run_test.sh:176 就是踩了这个）。
code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
code=${code:-000}
if [ "$code" = "200" ]; then ok "服务健康：http://127.0.0.1:$PORT/health (200)"
else
  bad "服务不健康（http=$code）—— 确认 PORT=$PORT，且服务已就绪"
  note "若刚起服还在编译 static kernel，等就绪后再跑本脚本"
  exit 2
fi

if [ -z "$SLOG" ]; then
  SLOG=$(ls -t "$PKG"/results/*/serve.log 2>/dev/null | head -1)
fi
[ -n "$SLOG" ] && [ -f "$SLOG" ] && ok "服务日志：$SLOG" \
  || bad "未找到服务日志（SLOG=... 指定）。涉及日志的判据会跳过。"

if [ -n "$DOCKER" ] && [ -n "$NAME" ]; then
  if $DOCKER inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
    ok "容器在跑：$NAME"
  else
    bad "容器 $NAME 不在跑（docker exec 相关的检查会跳过；可 NAME=... 指定）"
    DOCKER=""
  fi
fi
[ -z "$DOCKER" ] && note "无可用 docker ⇒ 跳过 local-owner validate 与容器内指纹"

ENGRAM_ON=1
if [ -n "$MODEL" ] && [ -d "$MODEL" ] && [ -f "$MODEL/config.json" ]; then
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
  [ "$ENGRAM_ON" = "1" ] && ok "检测到 Engram（enable_engram=true）" \
    || bad "模型目录无 Engram ⇒ 本次 enable_engram=false"
elif [ -n "$MODEL" ] && [ -d "$MODEL" ]; then
  note "$MODEL 下没有 config.json（可能传的不是模型根目录）⇒ 跳过探测，按 enable_engram=true 处理"
else
  note "未传 MODEL ⇒ 跳过 Engram 探测，按 enable_engram=true 处理（有 Engram 就是它）"
fi
echo "attach=1" >> "$OUT/env.txt"
echo "engram_on=$ENGRAM_ON" >> "$OUT/env.txt"
echo "port=$PORT" >> "$OUT/env.txt"
echo "slog=$SLOG" >> "$OUT/env.txt"

if [ -z "$OFFICIAL_DIR" ]; then
  for c in "$HOME/models/DeepSeek-V4.1-Flash" /home/*/models/DeepSeek-V4.1-Flash; do
    [ -d "$c" ] && { OFFICIAL_DIR="$c"; break; }
  done
fi
[ -n "$OFFICIAL_DIR" ] && [ -d "$OFFICIAL_DIR" ] && ok "官方目录：$OFFICIAL_DIR" \
  || bad "未找到官方 checkpoint 目录（视觉用例会跳过）；可用 OFFICIAL_DIR=... 指定"

# ============================== [3] 必查三项 ==============================
say "必查 ① static_kernel 静默降级"
SK_PASS=1
if [ -n "$SLOG" ] && [ -f "$SLOG" ]; then
  # grep -c 无匹配时打印 0 **且退出码 1** ⇒ 绝不能写成 `|| echo 0`（会得到 "0\n0"，
  # 于是 "0\n0" != "0" 成立，把"没降级"这个**正常结果**误判成降级）。
  SK_HITS=$(grep -ac "static_kernel.py:650" "$SLOG" 2>/dev/null || true)
  SK_HITS=${SK_HITS:-0}
  if [ "$SK_HITS" != "0" ]; then
    bad "static_kernel 被静默降级（命中 ${SK_HITS} 次）—— 结果不可信"
    grep -n "static_kernel.py:650" "$SLOG" | head -3 | sed 's/^/    /'
    SK_PASS=0
  else
    ok "static_kernel 未被降级（命中 0 次）"
  fi
  echo "static_kernel_degrade_hits=$SK_HITS" >> "$OUT/env.txt"
else
  bad "无日志可查 ⇒ 该项跳过（未判 PASS）"
  echo "static_kernel_degrade_hits=n/a" >> "$OUT/env.txt"
fi

ask() { curl -s -m "${2:-180}" "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",\"prompt\":$1,\"max_tokens\":${3:-16},\"temperature\":0.0}"; }

say "必查 ② Engram local-owner 自检（validate → 全对则切 fast）"
LO_EFF=unknown
if skipped lo; then
  note "SKIP 里含 lo ⇒ 跳过（服务保持当前 local-owner 模式）"
elif [ "$ENGRAM_ON" != "1" ]; then
  LO_EFF=n/a; note "无 Engram ⇒ 跳过"
elif [ -z "$DOCKER" ]; then
  note "无 docker ⇒ 跳过（不影响其它项）"
else
  $DOCKER exec "$NAME" bash -lc "printf '%s' validate > /tmp/v41_engram_localowner" 2>/dev/null || true
  ask '"The capital of France is"' 180 16 >/dev/null 2>&1 || true
  sleep 2
  if [ -n "$SLOG" ] && grep -q "local-owner] VALIDATE OK" "$SLOG"; then
    ok "validate 通过 → 切 fast（最快路径）"
    $DOCKER exec "$NAME" bash -lc "printf '%s' fast > /tmp/v41_engram_localowner"
    LO_EFF=fast
  elif [ -n "$SLOG" ] && grep -q "local-owner] VALIDATE FAILED" "$SLOG"; then
    bad "validate 失败 → 退回 on（功能正确，慢约 1 ms/step）"
    $DOCKER exec "$NAME" bash -lc "printf '%s' on > /tmp/v41_engram_localowner"
    LO_EFF=on
  else
    bad "未看到 validate 结果（日志里没有 VALIDATE OK/FAILED）→ 保持现状"
    LO_EFF=unknown
  fi
fi
echo "$LO_EFF" > "$OUT/local_owner_effective.txt"

say "必查 ③ KV 容量（门槛 3,145,728 tokens = 3Mi）"
KV_PASS=0
if [ -n "$SLOG" ] && [ -f "$SLOG" ]; then
  KV=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$SLOG" | tail -1 | tr -dc '0-9')
  KV=${KV:-0}
  if [ "$KV" -gt 3145728 ]; then ok "KV $KV tokens > 3Mi"; KV_PASS=1
  else bad "KV $KV tokens ≤ 3Mi"; KV_PASS=0; fi
else
  KV=0; bad "无日志可查 ⇒ KV 容量未知"
fi
printf 'kv_tokens=%s\nkv_pass=%s\nmax_seqs=%s\nprefix=%s\nmode=%s\n' \
  "$KV" "$KV_PASS" "${MAX_SEQS:-n/a}" "${PREFIX:-n/a}" "$MODE" >> "$OUT/env.txt"

# --- 附加：DRAFT_GRAPH 生效性指纹（只有日志里出现过才打印）---
if [ -n "$SLOG" ] && [ -f "$SLOG" ] && grep -q "Wrapping draft model with ACLGraphWrapper" "$SLOG" 2>/dev/null; then
  say "附加：DRAFT_GRAPH 生效性指纹"
  WP=$(grep -c "Wrapping draft model with ACLGraphWrapper" "$SLOG" 2>/dev/null || true)
  CAP=$(grep -c "dspark-graph-capture" "$SLOG" 2>/dev/null || true)
  [[ "${WP:-0}" =~ ^[0-9]+$ ]] || WP=0
  [[ "${CAP:-0}" =~ ^[0-9]+$ ]] || CAP=0
  [ "$WP" -ge 8 ] && ok "Wrapping draft model = $WP（期望 8，每 rank 一次）" \
                  || bad "Wrapping draft model = $WP（期望 8）"
  if [ "$CAP" -gt 0 ]; then ok "dspark-graph-capture 打印 $CAP 次 ⇒ 捕获期真的建了 draft attention metadata"
  else bad "dspark-graph-capture = 0 ⇒ 缺 DSPARK_GRAPH_CAPTURE_METADATA=1，本轮 A / tok·s 作废"; fi
  printf 'draft_wrap=%s\ndspark_capture=%s\n' "$WP" "$CAP" >> "$OUT/env.txt"
fi

# 报告需要 $OUT/serve.log（make_report.sh 会从它抽关键行）—— 软链过来，不动原日志
if [ -n "$SLOG" ] && [ -f "$SLOG" ]; then ln -sfn "$SLOG" "$OUT/serve.log"; fi
# BUILD_INFO.txt 也顺带取一份（报告里显示镜像指纹）
if [ -n "$DOCKER" ]; then
  $DOCKER exec "$NAME" bash -lc 'cat /opt/dsv41/BUILD_INFO.txt 2>/dev/null' > "$OUT/BUILD_INFO.txt" 2>/dev/null || true
fi

# ============================== [4] 性能（单流 quote）==============================
if skipped quote; then
  say "性能 quote —— SKIP"
else
  quote() { # ctx tag
    local ctx=$1 tag=$2
    say "quote 口径 ${ctx} tokens（${REPEATS} 发）…"
    P=$PKG TAG="$tag" TOKENS="$ctx" MAXTOK=256 \
      REPEATS="$REPEATS" URL="http://127.0.0.1:$PORT" OUTDIR="$OUT" \
      PREFIX="$PKG/data/hongloumeng.txt" SUFFIX="$PKG/data/hlm/suffix_quote.txt" \
      PYHOST="$PYHOST" bash "$PKG/tests/t_quote.sh" 2>&1 | tail -14 | sed 's/^/  /' | tee -a "$OUT/driver.log"
  }
  quote 8192 quote_8k
  quote 32768 quote_32k
  [ "$MODE" = "full" ] && quote 131072 quote_128k
fi

# ============================== [5] 视觉 ==============================
if skipped vision; then
  say "视觉 —— SKIP"
elif [ -n "$OFFICIAL_DIR" ] && [ -d "$OFFICIAL_DIR/inference/examples/images" ]; then
  say "视觉 23 例 …"
  "$PYHOST" "$PKG/tests/t_vision.py" \
      --server "http://127.0.0.1:$PORT" --model "$SERVED_NAME" \
      --images-dir "$OFFICIAL_DIR/inference/examples/images" \
      --out "$OUT/vision.json" 2>&1 | tail -10 | sed 's/^/  /' | tee -a "$OUT/driver.log"
else
  bad "跳过视觉（未找到官方图片目录：${OFFICIAL_DIR:-<未指定>}/inference/examples/images）"
fi

# ============================== [6] GSM8K ==============================
if skipped gsm8k; then
  say "GSM8K —— SKIP"
elif [ "$RUN_GSM8K" = "1" ] || [ "$MODE" = "full" ]; then
  say "GSM8K-200 …"
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYHOST" "$PKG/tests/t_gsm8k.py" --base "http://127.0.0.1:$PORT" \
    --limit 200 --conc 4 ${ENC_DIR:+--enc-dir "$ENC_DIR"} \
    --out "$OUT/gsm8k.json" 2>&1 | tail -6 | sed 's/^/  /' | tee -a "$OUT/driver.log"
else
  note "GSM8K 未跑（MODE=$MODE；要跑用 MODE=full 或 RUN_GSM8K=1）"
fi

# ============================== [7] 报告 ==============================
if skipped report; then
  say "报告 —— SKIP"
else
  say "生成报告"
  bash "$PKG/tests/make_report.sh" "$OUT" "$RUN_ID" "${MODEL:-<附着测试，未传 MODEL>}" "$LO_EFF" 2>&1 | tail -40
fi

say "完成。产出目录：$OUT"
echo
echo "  报告：      less $OUT/REPORT.md"
echo "  服务未受影响（仍在跑）：$DOCKER exec -it $NAME bash"
echo "  要停服务：  $DOCKER rm -f $NAME"
