#!/usr/bin/env bash
# 池大小阈值实验：验证「D 的 num_blocks 与 P 的 num_blocks 的相对大小」是否是
# 1M 偶发失败的判别量。
#
# 已知（全部同一 prompt、串行、temperature=0、MAX_SEQS=4、CLIP=1）：
#   D num_blocks = 30082（当前 P 为 29721，D 大 361）→ 每 4 个请求失败一次
#   D num_blocks = 30200（比 P 大 479）              → 每 4 个请求失败一次
#   D num_blocks = 19494（远小于 P）                 → 12/12 全过
# 因此候选判别量是「D 的池 > P 的池」，而不是池的绝对大小。
#
# 本脚本依次跑两个配置（都 8 个请求）：
#   A) D 池 ≈ 29000（**小于** P 的 29721）→ 若假说成立，应全过
#   B) D 池 ≈ 30500（**大于** P 的 29721）→ 若假说成立，应出现每 4 个失败
#
# ★ 每块字节数由两个实测点反推约 541,030 B（10545000000B→19494 块、
#   15.16GiB→30082 块）。字节数→块数不是精确线性，所以脚本会把两端实际
#   num_blocks 打出来核对，按**实测值**而不是目标值判读。
#
# 用法（a3-21）：setsid nohup bash ced_pool_threshold_experiment.sh > /tmp/pt_exp.log 2>&1 &
set -uo pipefail

T=/home/chiro/tmp/ced_numeric/pkg_d7953e5_trace
T=/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace
TEMPLATE=${TEMPLATE:-/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/repeat8_D4.request.json}
COUNT=${COUNT:-8}

say() { echo "[pool-thr] $(date '+%H:%M:%S') $*"; }

run_arm() {
  local label=$1 pool_bytes=$2
  local run_id="ced_trace_d_${label}"
  local name="dsv41-ced-trace-d-${label}"
  local R="$T/results/$run_id"

  say "=== $label: 池 $pool_bytes 字节 ==="
  docker rm -f dsv41-ced-trace-d-ms4sp >/dev/null 2>&1 || true
  sleep 12

  cat > /tmp/pt_inner.sh <<EOF
set -e
cd $T
export MODEL=/home/l00886679/models/out/v41-flat-verify3 SERVED_NAME=deepseek-v41-ced-pd TP=8 DP=1
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_CACHE_MEMORY_BYTES=$pool_bytes
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export SPEC=0 PREFIX=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export PATCH_MODE=mount WAIT_READY=1
export V41_CED_ROLE=decode V41_CED_BLOCK_TRACE=1 V41_CED_SWA_TRACE=1 V41_CED_SWA_CLIP=1
export V41_CED_BLOCK_DUMP_DIR=/opt/dsv41/results/$run_id/blockdump
export MULTISTREAM=0 DSA_OVERLAP=0 GRAPH=1 EAGER=0
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_METADATA_INLINE=1
export NAME=$name RUN_ID=$run_id
export PORT=18991 KV_PORT=19091
export DEVS="8 9 10 11 12 13 14 15"
bash scripts/serve_a3_pd.sh decode
EOF
  nohup bash /tmp/pt_inner.sh > "$T/../pt_${label}_launch.log" 2>&1 &

  say "$label 等就绪"
  local ready=0
  for _ in $(seq 1 160); do
    curl -sf -m 5 -o /dev/null http://127.0.0.1:18991/health && { ready=1; break; }
    sleep 15
  done
  [ "$ready" = "1" ] || { say "$label FAIL: 40 分钟未就绪"; return 2; }

  say "$label 配置核对"
  grep -ao "max-num-seqs [0-9]*" "$R/serve.log" | head -1
  grep -a "num_blocks:" "$R/serve.log" | head -1
  say "$label P 侧 num_blocks（对照）"
  grep -a "num_blocks:" /home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace/results/ced_trace_p_20260924_trace/serve.log | head -1

  mkdir -p "$R/tools" "$R/probe"
  cp /tmp/ced_seq_probe.py /tmp/ced_layer_trace_sequence.sh "$R/tools/" 2>/dev/null || true

  say "$label 发 $COUNT 个 1M"
  cd "$R"
  RD="$R" TEMPLATE="$TEMPLATE" COUNT="$COUNT" TAG_PREFIX="${label}_" \
    OUTDIR="$R/probe/seq" SNAP_PREFIX="${label}_snap_" \
    bash "$R/tools/ced_layer_trace_sequence.sh"
  say "$label 完成"
}

run_arm below 15700000000   # ≈29000 块（< P 的 29721，D 比 P 小）
run_arm above 16500000000   # ≈30500 块（> P 的 29721，D 比 P 大）
say "全部完成"
