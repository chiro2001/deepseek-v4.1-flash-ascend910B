#!/usr/bin/env bash
# 池大小近阈值双臂实验（决定"D 池 > P 池"是否即触发条件）。
#
# ★★ 为什么重写：上一版 `ced_pool_threshold_experiment.sh` 有**静默降级 bug** ——
#    它只 `docker rm` 了上一臂的容器名，没有删更早的实例；而就绪检查只 curl
#    18991/health，**旧容器替新容器回答了健康检查**，于是"above 臂"从未启动，
#    它的 8 个请求全部发给了仍在运行的 below 实例（实测 above 容器
#    `No such object`，below 容器 StartedAt=01:43 一直存活）。
#    本版因此加了四道硬门（见 verify_new_container）。
#
# 已知对照（同一 1M 请求、串行、temperature=0、MAX_SEQS=4、CLIP=1）：
#   D=30082（> P 29721）→ 每 4 个长请求失败
#   D=30200（> P）      → 每 4 个失败
#   D=19494（< P）      → 12/12 通过
#   D=29024（< P）      → 16/16 通过
# 但"C>P"与"C≈30k"仍混淆。本脚本在 P=29721 两侧各取一个**尽量贴近**的点：
#   arm_hi:  D≈29850（> P 仅 +129）→ 若"D>P"成立，应出现每 4 个失败
#   arm_lo:  D≈29600（< P −121）   → 应全过
# 两点只差约 250 块，可把"关系"与"绝对大小"分开。
#
# 用法（a3-21）：
#   setsid nohup env COUNT=8 bash ced_pool_arms_experiment.sh > /tmp/pa_exp.log 2>&1 &
set -uo pipefail

T=/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace
TEMPLATE=${TEMPLATE:-/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/repeat8_D4.request.json}
COUNT=${COUNT:-8}
D_PREFIX=dsv41-ced-trace-d

say() { echo "[pool-arms] $(date '+%H:%M:%S') $*"; }

# 逐块字节数由两个实测点反推：15700000000→29024、10545000000→19494
#   ⇒ 540,932 B/块；本脚本按此把目标块数换算成字节。
bytes_for_blocks() { echo $(( $1 * 540932 )); }

cleanup_d_containers() {
  # 删掉**所有**我们的 D 容器（含更早的实例），这是上一版漏掉的关键一步。
  local names
  names=$(docker ps -a --format '{{.Names}}' | grep -E "^${D_PREFIX}" || true)
  if [ -n "$names" ]; then
    say "删除旧 D 容器：$(echo "$names" | tr '\n' ' ')"
    echo "$names" | xargs -r docker rm -f >/dev/null 2>&1 || true
  fi
  # 等卡真正释放（VLLMWorker 退出），最多 5 分钟
  for _ in $(seq 1 60); do
    n=$(npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | grep -c "VLLMWorker")
    [ "$n" = "0" ] && { say "卡已释放（VLLMWorker=0）"; return 0; }
    sleep 5
  done
  say "警告：仍有 $n 个 VLLMWorker 未退出，继续尝试启动（serve_a3.sh 会自行判断）"
}

# 四道硬门：确认**新容器**在服务，而不是旧容器在回答端口
verify_new_container() {
  local name=$1 run_id=$2 started_epoch=$3
  local cid state log
  cid=$(docker ps -qf "name=^/${name}$")
  if [ -z "$cid" ]; then
    say "FAIL: 容器 $name 不在运行 —— 端口上的健康检查一定来自别的进程"
    return 1
  fi
  state=$(docker inspect "$cid" --format '{{.State.Status}}')
  [ "$state" = "running" ] || { say "FAIL: $name 状态=$state"; return 1; }

  log="$T/results/$run_id/serve.log"
  [ -f "$log" ] || { say "FAIL: 新容器没有 serve.log（$log）—— 说明它不是本次启动的"; return 1; }
  # 日志必须在本次启动**之后**被写过
  local mtime
  mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
  if [ "$mtime" -lt "$started_epoch" ]; then
    say "FAIL: serve.log mtime($mtime) 早于本次启动($started_epoch) —— 陈旧日志"
    return 1
  fi
  # 新容器必须已经打印出自己的 num_blocks（配置生效的证据）
  grep -q "num_blocks:" "$log" || { say "FAIL: 新容器日志里没有 num_blocks"; return 1; }

  say "新容器校验通过：cid=${cid:0:12} state=running"
  grep -a "num_blocks:" "$log" | head -1
  grep -ao "max-num-seqs [0-9]*\|max_model_len [0-9]*" "$log" | head -3
  return 0
}

run_arm() {
  local label=$1 target_blocks=$2
  local pool_bytes run_id name
  pool_bytes=$(bytes_for_blocks "$target_blocks")
  run_id="ced_trace_d_${label}"
  name="dsv41-ced-trace-d-${label}"
  local R="$T/results/$run_id"

  say "=== $label: 目标 $target_blocks 块（$pool_bytes 字节）==="
  cleanup_d_containers

  cat > /tmp/pa_inner.sh <<EOF
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
  local started_epoch
  started_epoch=$(date +%s)
  nohup bash /tmp/pa_inner.sh > "$T/../pa_${label}_launch.log" 2>&1 &

  say "$label 等就绪（只在**新容器**起来后才算就绪）"
  local ready=0
  for _ in $(seq 1 160); do
    sleep 15
    if docker ps -qf "name=^/${name}$" >/dev/null 2>&1 \
       && curl -sf -m 5 -o /dev/null http://127.0.0.1:18991/health; then
      ready=1; break
    fi
  done
  if [ "$ready" != "1" ]; then
    say "$label FAIL: 40 分钟内新容器未就绪（看一眼 pa_${label}_launch.log）"
    return 2
  fi
  verify_new_container "$name" "$run_id" "$started_epoch" || return 3

  mkdir -p "$R/tools" "$R/probe"
  cp /tmp/ced_seq_probe.py /tmp/ced_layer_trace_sequence.sh "$R/tools/" 2>/dev/null || true

  say "$label 发 $COUNT 个 1M"
  ( cd "$R" && RD="$R" TEMPLATE="$TEMPLATE" COUNT="$COUNT" TAG_PREFIX="${label}_" \
      OUTDIR="$R/probe/seq" SNAP_PREFIX="${label}_snap_" \
      bash "$R/tools/ced_layer_trace_sequence.sh" )
  say "$label 完成"
}

run_arm hi 29850
run_arm lo 29600
say "全部完成"
