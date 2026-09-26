#!/usr/bin/env bash
# 因果实验：把 **P 的 KV 池调大**到超过 D，看原本必失败的 D 配置是否转为通过。
#
# 已知（同一 1M 请求、串行、MAX_SEQS=4、CLIP=1、P=29721）：
#   D=30200 / 30082 / 29850（都 > P）→ 每 4 个长请求失败，且**失败的那个请求正是
#        第一个把块号推进到 [P.num_blocks, D.num_blocks) 区间的请求**
#   D=29024 / 19494（都 < P）       → 16/16、12/12 全过
# 但"D>P 关系"与"D 绝对大小≈30k"仍混淆。**唯一能把两者分开的办法是移动 P 的池**：
# 本脚本把 P 调到 ≈30500 块（> D 的 29850），其余完全不变。
#   若 D 转为通过 ⇒ 关系因果成立；
#   若 D 仍失败   ⇒ 是 D 的绝对大小，与 P 无关。
#
# 用法（a3-21）：setsid nohup env COUNT=8 bash ced_p_enlarge_experiment.sh > /tmp/pe_exp.log 2>&1 &
set -uo pipefail

T=/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace
TEMPLATE=${TEMPLATE:-/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/repeat8_D4.request.json}
COUNT=${COUNT:-8}
D_TARGET_BLOCKS=${D_TARGET_BLOCKS:-29850}
P_KBYTES=${P_KBYTES:-16500000000}     # ≈30500 块（540932 B/块）
BYTES_PER_BLOCK=540932

say() { echo "[p-enlarge] $(date '+%H:%M:%S') $*"; }

workers_on_d() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | awk -F'|' '/VLLMWorker/ {split($2,a," "); phy=a[1]*2+a[2]; if (phy>=8 && phy<=15) c++} END{print c+0}'
}

workers_on_p() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | awk -F'|' '/VLLMWorker/ {split($2,a," "); phy=a[1]*2+a[2]; if (phy<=7) c++} END{print c+0}'
}

wait_workers() {  # $1=函数名 $2=标签
  for _ in $(seq 1 72); do
    n=$($1)
    [ "$n" = "0" ] && { say "$2 已释放"; return 0; }
    sleep 5
  done
  say "$2 警告：仍有 $n 个 worker"
}

# 四道硬门（与池阈值实验同一套）
verify_container() {
  local name=$1 run_id=$2 started=$3 want_min=$4 what=$5
  local cid state log mtime nb
  cid=$(docker ps -qf "name=^/${name}$")
  [ -n "$cid" ] || { say "FAIL $what: 容器 $name 不在运行"; return 1; }
  state=$(docker inspect "$cid" --format '{{.State.Status}}')
  [ "$state" = "running" ] || { say "FAIL $what: 状态=$state"; return 1; }
  log="$T/results/$run_id/serve.log"
  [ -f "$log" ] || { say "FAIL $what: 无 serve.log（说明不是本次启动的）"; return 1; }
  mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
  [ "$mtime" -ge "$started" ] || { say "FAIL $what: serve.log 陈旧"; return 1; }
  grep -q "num_blocks:" "$log" || { say "FAIL $what: 日志无 num_blocks"; return 1; }
  nb=$(grep -a "num_blocks:" "$log" | head -1 | grep -o '[0-9]\+')
  say "$what 校验通过：num_blocks=$nb（要求 ≥$want_min）"
  if [ -n "$want_min" ] && [ "$nb" -lt "$want_min" ]; then
    say "FAIL $what: num_blocks=$nb 未达到要求 $want_min"
    return 1
  fi
  return 0
}

# ---------- 1) 停旧 D 与旧 P ----------
names=$(docker ps -a --format '{{.Names}}' | grep -E '^dsv41-ced-trace-d' || true)
[ -n "$names" ] && { say "删除旧 D：$(echo "$names" | tr '\n' ' ')"; echo "$names" | xargs -r docker rm -f >/dev/null 2>&1; }
pnames=$(docker ps -a --format '{{.Names}}' | grep -E '^dsv41-ced-trace-p' || true)
[ -n "$pnames" ] && { say "删除旧 P：$(echo "$pnames" | tr '\n' ' ')"; echo "$pnames" | xargs -r docker rm -f >/dev/null 2>&1; }
wait_workers workers_on_d "D 的 chip8-15"
wait_workers workers_on_p "P 的 chip0-7"

# ---------- 2) 起**放大池**的 P ----------
P_RUN=ced_trace_p_bigpool
P_NAME=dsv41-ced-trace-p-bigpool
cat > /tmp/pe_p.sh <<EOF
set -e
cd $T
export MODEL=/home/l00886679/models/out/v41-flat-verify3 SERVED_NAME=deepseek-v41-ced-pd TP=8 DP=1
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_CACHE_MEMORY_BYTES=$P_KBYTES
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export SPEC=0 PREFIX=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export PATCH_MODE=mount WAIT_READY=1
export V41_CED_ROLE=prefill V41_CED_BLOCK_TRACE=1
export MULTISTREAM=1 DSA_OVERLAP=1
export NAME=$P_NAME RUN_ID=$P_RUN
export PORT=18990 KV_PORT=19090
export DEVS="0 1 2 3 4 5 6 7"
bash scripts/serve_a3_pd.sh prefill
EOF
say "起 P（池 ≈$((P_KBYTES / BYTES_PER_BLOCK)) 块）"
p_started=$(date +%s)
nohup bash /tmp/pe_p.sh > "$T/../pe_p_launch.log" 2>&1 &
for _ in $(seq 1 160); do
  sleep 15
  docker ps -qf "name=^/${P_NAME}$" >/dev/null 2>&1 && curl -sf -m 5 -o /dev/null http://127.0.0.1:18990/health && break
done
verify_container "$P_NAME" "$P_RUN" "$p_started" "$((D_TARGET_BLOCKS + 50))" "P" || { say "中止：P 未达标"; exit 2; }

# ---------- 3) 起 D=29850（原本必失败的配置） ----------
D_RUN=ced_trace_d_pbig
D_NAME=dsv41-ced-trace-d-pbig
D_KBYTES=$(( D_TARGET_BLOCKS * BYTES_PER_BLOCK ))
cat > /tmp/pe_d.sh <<EOF
set -e
cd $T
export MODEL=/home/l00886679/models/out/v41-flat-verify3 SERVED_NAME=deepseek-v41-ced-pd TP=8 DP=1
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_CACHE_MEMORY_BYTES=$D_KBYTES
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export SPEC=0 PREFIX=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 NPUGRAPH_EX=1
export PATCH_MODE=mount WAIT_READY=1
export V41_CED_ROLE=decode V41_CED_BLOCK_TRACE=1 V41_CED_SWA_TRACE=1 V41_CED_SWA_CLIP=1
export V41_CED_BLOCK_DUMP_DIR=/opt/dsv41/results/$D_RUN/blockdump
export MULTISTREAM=0 DSA_OVERLAP=0 GRAPH=1 EAGER=0
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_METADATA_INLINE=1
export NAME=$D_NAME RUN_ID=$D_RUN
export PORT=18991 KV_PORT=19091
export DEVS="8 9 10 11 12 13 14 15"
bash scripts/serve_a3_pd.sh decode
EOF
say "起 D（目标 $D_TARGET_BLOCKS 块）"
d_started=$(date +%s)
nohup bash /tmp/pe_d.sh > "$T/../pe_d_launch.log" 2>&1 &
for _ in $(seq 1 160); do
  sleep 15
  docker ps -qf "name=^/${D_NAME}$" >/dev/null 2>&1 && curl -sf -m 5 -o /dev/null http://127.0.0.1:18991/health && break
done
verify_container "$D_NAME" "$D_RUN" "$d_started" "" "D" || { say "中止：D 未就绪"; exit 3; }

# ---------- 4) 发请求 ----------
R="$T/results/$D_RUN"
mkdir -p "$R/tools" "$R/probe"
cp /tmp/ced_seq_probe.py /tmp/ced_layer_trace_sequence.sh "$R/tools/" 2>/dev/null || true
say "发 $COUNT 个 1M"
( cd "$R" && RD="$R" TEMPLATE="$TEMPLATE" COUNT="$COUNT" TAG_PREFIX=pe_ \
    OUTDIR="$R/probe/seq" SNAP_PREFIX=pe_snap_ \
    bash "$R/tools/ced_layer_trace_sequence.sh" )
say "完成"
