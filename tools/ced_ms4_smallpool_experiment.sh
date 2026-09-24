#!/usr/bin/env bash
# 变量分离实验：MAX_SEQS=4 + 小 KV 池（相对标准臂只改池大小）。
#
# 背景：标准臂（MAX_SEQS=4，池 30082 块）每 4 个长请求失败一次；
#       MAX_SEQS=1 + 小池（19494 块）连续 12 个全过。本脚本把 MAX_SEQS 恢复成 4，
#       只保留小池，用来判定"小池"是不是决定性因素。
#
# 用法（在 a3-21 上）：
#   COUNT=12 nohup bash ced_ms4_smallpool_experiment.sh > sp4_exp.log 2>&1 &
#
# 只在所有前置就绪后才发请求；任何一步失败都打印并退出（不静默继续）。
set -uo pipefail

T=/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace
RUN_ID=${RUN_ID:-ced_trace_d_ms4_smallpool}
NAME=${NAME:-dsv41-ced-trace-d-ms4sp}
POOL_BYTES=${POOL_BYTES:-10545000000}
COUNT=${COUNT:-12}
TEMPLATE=${TEMPLATE:-/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/repeat8_D4.request.json}
R="$T/results/$RUN_ID"

say() { echo "[ms4sp] $*"; }

say "停掉旧的 D"
docker rm -f dsv41-ced-trace-d-smallpool dsv41-ced-trace-d-ms1 >/dev/null 2>&1 || true
sleep 12

say "起 D：MAX_SEQS=4 + 池 $POOL_BYTES 字节"
cat > /tmp/ms4sp_inner.sh <<EOF
set -e
cd $T
export MODEL=/home/l00886679/models/out/v41-flat-verify3 SERVED_NAME=deepseek-v41-ced-pd TP=8 DP=1
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_CACHE_MEMORY_BYTES=$POOL_BYTES
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export SPEC=0 PREFIX=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export PATCH_MODE=mount WAIT_READY=1
export V41_CED_ROLE=decode V41_CED_BLOCK_TRACE=1 V41_CED_SWA_TRACE=1 V41_CED_SWA_CLIP=1
export MULTISTREAM=0 DSA_OVERLAP=0 GRAPH=1 EAGER=0
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_METADATA_INLINE=1
export NAME=$NAME RUN_ID=$RUN_ID
export PORT=18991 KV_PORT=19091
export DEVS="8 9 10 11 12 13 14 15"
bash scripts/serve_a3_pd.sh decode
EOF
nohup bash /tmp/ms4sp_inner.sh > "$T/../ms4sp_launch.log" 2>&1 &

say "等 D 就绪（最多 40 分钟）"
ready=0
for _ in $(seq 1 160); do
  if curl -sf -m 5 -o /dev/null http://127.0.0.1:18991/health; then ready=1; break; fi
  sleep 15
done
if [ "$ready" != "1" ]; then
  say "FAIL: D 40 分钟内未就绪，退出（不发请求）"
  exit 2
fi
say "D 就绪"

say "核对关键配置"
grep -ao "max-num-seqs [0-9]*" "$R/serve.log" | head -1
grep -a "num_blocks:" "$R/serve.log" | head -1
docker inspect "$(docker ps -qf name=$NAME)" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "V41_CED_SWA_CLIP|KV_CACHE_MEMORY_BYTES" | head -3

mkdir -p "$R/tools" "$R/probe"
cp /tmp/ced_seq_probe.py /tmp/ced_layer_trace_sequence.sh "$R/tools/" 2>/dev/null || true
[ -f "$R/tools/ced_seq_probe.py" ] || { say "FAIL: 探针脚本缺失"; exit 2; }

say "发 $COUNT 个 1M 请求"
cd "$R"
RD="$R" TEMPLATE="$TEMPLATE" COUNT="$COUNT" TAG_PREFIX=sp4_ \
  OUTDIR="$R/probe/sp4_seq" SNAP_PREFIX=sp4_snap_ \
  bash "$R/tools/ced_layer_trace_sequence.sh"
say "完成"
