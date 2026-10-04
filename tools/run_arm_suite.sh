#!/usr/bin/env bash
# 一个臂的完整跑批：停旧 → 清图缓存 → 起服 → 等 ready → 基准 → 采 profile → 摘要。
#
# 为什么要有它：单次起服 ~20 min，人工盯守容易漏掉"健康检查没等到就开测"这类错误，
# 且四个维度的口径必须每次一致（并发列表、prompt 长度、reps）。
#
# 用法:
#   bash tools/run_arm_suite.sh <launcher脚本> <tag> [BENCH=1] [PROF=1] [ACCEPT=1]
# 例:
#   bash tools/run_arm_suite.sh ~/tmp/launch_armF.sh armF_base 1 1 1
#
# 安全闸：若检测到别的基准在跑（boundary_n2.py / bench_concurrency.py），直接退出。
set -uo pipefail

LAUNCHER=${1:?launcher}
TAG=${2:?tag}
BENCH=${3:-1}
PROF=${4:-1}
ACCEPT=${5:-0}

PORT=${PORT:-19210}
NAME=${NAME:-dsv41-tp8k5}
# [BUGFIX 2026-10-05] 导出步骤要用镜像名，但本脚本**不定义** IMAGE（它由各 launcher 自己 export）。
# 之前直接引用 $IMAGE，在 `set -u` 下会以 "IMAGE: unbound variable" **中止整个脚本** ——
# 症状是：基准与 profile 都跑完了，**验收被静默跳过**。这里给一个可覆盖的默认值。
IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
REPO=$HOME/cedpd-repo
OUT=$REPO/results/$TAG
HOST=http://127.0.0.1:$PORT

say() { echo "[arm-suite] $*"; }

if pgrep -f "boundary_n2.py|boundary_n.py" >/dev/null 2>&1; then
  say "⛔ 检测到 boundary 压测仍在跑，先等它结束（这是安全闸，不是错误）"
  exit 3
fi

say "停掉旧容器 $NAME"
docker rm -f "$NAME" >/dev/null 2>&1 || true
say "清 npugraph 图缓存（避免跨臂复用旧图）"
if [ -d "$REPO/cache/npugraph" ]; then
  mv "$REPO/cache/npugraph" "$REPO/cache/npugraph.bak_$(date +%m%d_%H%M%S)" 2>/dev/null || true
fi
mkdir -p "$REPO/cache/npugraph"

say "起服（RUN_ID=$TAG，launcher=$LAUNCHER）"
cd "$REPO" || exit 1
RUN_ID="$TAG" setsid nohup bash "$LAUNCHER" > "$HOME/tmp/${TAG}_launch.log" 2>&1 < /dev/null &

say "等 ready（最多 40 min，每 30 s 探一次）"
ok=0
for i in $(seq 1 80); do
  sleep 30
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$HOST/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; say "ready（第 $((i*30)) s）"; break; fi
  [ $((i % 4)) -eq 0 ] && say "  …$((i*30))s health=$code"
done
if [ "$ok" != "1" ]; then
  say "⛔ 起服未在 40 min 内 ready；日志尾部："
  tail -30 "$HOME/tmp/${TAG}_launch.log"
  exit 1
fi

sleep 20
say "健康检查："
for c in /health /v1/models; do
  printf '  %-12s %s\n' "$c" "$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$HOST$c")"
done

if [ "$BENCH" = "1" ]; then
  # 用小样本时把并发/rep 显式收窄（例如只为生成 `[bneck] hp` 样本做单变量 A/B，
  # 不需要完整四维度扫描）。默认保持交付口径。
  BENCH_CONC=${BENCH_CONC:-1,2,4,8,16}
  BENCH_REPS=${BENCH_REPS:-4}
  BENCH_OUT=${BENCH_OUT:-256}
  say "基准（并发 $BENCH_CONC / 1024-$BENCH_OUT / $BENCH_REPS rep）…"
  python3 "$REPO/tools/bench_concurrency.py" --base-url "$HOST" \
    --concurrency "$BENCH_CONC" --prompt-tokens 1024 --output-tokens "$BENCH_OUT" --repeats "$BENCH_REPS" \
    --spec-tokens 5 --label "$TAG" --json-out "$HOME/tmp/${TAG}_bench.json" \
    > "$HOME/tmp/${TAG}_bench.log" 2>&1
  tail -22 "$HOME/tmp/${TAG}_bench.log"
fi

if [ "$PROF" = "1" ]; then
  say "采 profile（start → 并发 1/4/8 各一段 → stop）…"
  bash "$HOME/tmp/prof_conc.sh" "$PORT" "1,4,8" 160 > "$HOME/tmp/${TAG}_prof.log" 2>&1 || true
  tail -6 "$HOME/tmp/${TAG}_prof.log"

  # ★ [MSPROF-EXPORT] torch_npu 的导出**不是**在 /stop_profile 时同步完成的；
  #   若之后很快清容器，`ASCEND_PROFILER_OUTPUT/` 不会生成，只剩几个 GB 的原始数据。
  #   实测：raw 数据在 stop 后立刻就是完整的（有 end_info.done），可以在**任意**
  #   带 CANN 的容器里事后补跑 analyse（不需要 NPU）。这里就用一次性容器补跑。
  say "补跑 msprof 导出（rank0 的每次捕获各 1–2 min）…"
  cat > /tmp/prof_analyze_$$.py <<'PYEOF'
import sys
import torch_npu  # noqa
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1], max_process_number=16)
print("ANALYSE_DONE")
PYEOF
  for d in $(ls -d "$OUT"/prof/dp0_pp0_tp0_* 2>/dev/null); do
    [ -d "$d" ] || continue
    if sudo -n test -d "$d" 2>/dev/null; then :; fi
    docker run --rm -v "$OUT/prof":/pf -v "/tmp/prof_analyze_$$.py":/a.py:ro "$IMAGE" \
      bash -lc "python3 /a.py /pf/$(basename "$d")" >> "$HOME/tmp/${TAG}_profexport.log" 2>&1 || true
  done
  say "导出完成；CSV 在 <run>/prof/*_ascend_pt/ASCEND_PROFILER_OUTPUT/（root 属主，用 sudo 读）"
  sudo -n chmod -R a+rX "$OUT/prof" 2>/dev/null || true
  ls -d "$OUT"/prof/*/ASCEND_PROFILER_OUTPUT 2>/dev/null | head -3
fi

if [ "$ACCEPT" = "1" ]; then
  say "验收（ced_pd_acceptance --mode all @144K）…"
  mkdir -p "$OUT"
  python3 "$REPO/tools/ced_pd_acceptance.py" \
    --base-url "$HOST" --tokenize-url "$HOST" --model "${SERVED_NAME:-deepseek-v41}" \
    --corpus "$REPO/data/hongloumeng.txt" --mode all --context-tokens 144000 \
    --out "$OUT/accept_144k.json" > "$HOME/tmp/${TAG}_accept.log" 2>&1 || true
  grep -aE "PASS|FAIL|通过|失败|✔|✘" "$HOME/tmp/${TAG}_accept.log" | tail -16
fi

say "hp（host 每步开销，来自 [bneck]，中位/中位±）"
python3 "$HOME/tmp/hpstat.py" "$TAG" 2>/dev/null || true

say "完成：$TAG"
say "  launch log: $HOME/tmp/${TAG}_launch.log"
say "  bench  log: $HOME/tmp/${TAG}_bench.log"
