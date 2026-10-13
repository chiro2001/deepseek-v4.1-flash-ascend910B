#!/usr/bin/env bash
# =============================================================================
# DCP Stage-A2「容量探针」：放开 V4.1 的 PP=DCP=PCP=1 门，只回答一个问题——
#   **TP8/DP1 + `--decode-context-parallel-size 8` 时，vLLM 报出的 KV 池 token 容量
#     是不是基线的 ~8 倍？**
#
# ★ 这一臂**一定算错**（attention 侧 DCP 执行路径还没实现）。它的唯一价值是
#   把「DCP ⇒ 内存账 ×8」从【推断】变成【实测】，并暴露 DCP 打开后**第一条真实的报错**。
# =============================================================================
set -uo pipefail
PKG=${PKG:-$HOME/cedpd-repo}
MODEL=${MODEL:-/home/l00886679/models/out/v41-flat-verify3}
# ★ 安全网：mount 模式下 `-v SRC:DST` 若 SRC 不存在，docker 会在宿主机**创建目录**，
#   容器里就变成一个同名目录，导入时报奇怪的错。起服前先断言。
DCPMOUNT=${DCPMOUNT:-$HOME/dcpw}
[ -f "$DCPMOUNT/vllm_ascend/core/deepseek_v41.py" ] || { echo "[cap][FAIL] 缺 $DCPMOUNT/vllm_ascend/core/deepseek_v41.py"; exit 2; }
CHIPS=${CHIPS:-"8 9 10 11 12 13 14 15"}
PORT=${PORT:-19210}
NAME=${NAME:-dsv41-dcpcap}
DCP=${DCP:-8}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-5368709120}
GPU_UTIL=${GPU_UTIL:-0.85}
STAMP=${STAMP:-$(date +%m%d_%H%M%S)}
OUT=${OUT:-$PKG/results/dcpcap_${STAMP}}
mkdir -p "$OUT"
# ★ 宿主侧输出**不能**写进 serve.log：serve_a2.sh 里有 `: > "$LOG"`，
#   会把宿主侧的诊断行整段截掉（2026-09-29 踩过一次，误以为挂载没生效）。
exec > >(tee -a "$OUT/harness.log") 2>&1
echo "[cap] $(date -Is) STAMP=$STAMP DCP=$DCP OUT=$OUT"

busy_devices() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | grep -E "^\| [0-9]+ +[0-9]+ +\|" | awk '{print $2*2+$3}' | sort -n | uniq | tr '\n' ' '
}

# ★ 2026-09-29 新增：**只看"有没有进程"不够**。实测高 8 张（8-15）上
#   `npu-smi` 进程表是空的，但 HBM 只剩 50.12/61.28 GiB（别人进程退出后驱动没回收），
#   于是 `serve_a3.sh` 直接报
#     ValueError: Free memory on device (50.12/61.28 GiB) on startup is less than
#                 desired GPU memory utilization (0.85, 52.09 GiB)
#   ⇒ 必须按**空闲 HBM** 选卡。阈值默认 55 GiB（util=0.85 需要 52.09 GiB，
#   留 ~3 GiB 余量给驱动自身开销）。
# ★ 阈值取 53 而不是 55：`aclnn` 的硬门槛是 `gpu_util * total = 0.85 * 61.27 = 52.08 GiB`。
#   设备 2 有 6.85 GB 驱动级残留（npu-smi 报 53.1 GiB），既往多轮 0-7 起服都成功。
#   取 55 会把设备 2 排除，导致 0-7 整块不可用 —— 而 8-15 现在有 11 GiB 残留（50~52 GiB），
#   两块都选不出来。
MIN_FREE_GIB=${MIN_FREE_GIB:-53}
free_enough_devices() {
  # 取每行**最后两个**空白分隔 token 作 used/total（used 可能带尾随 "/"）；
  # total=0 的表头行自然跳过。设备号用**出现顺序**（第 k 个数据行 = 设备 k-1）：
  # npu-smi 按 NPU0..7、每 NPU 2 die 的固定顺序输出，正好是 0..15。
  # ★ 不要用 `$2*2+$3`：实测部分行 npu-smi 列会错位，会算出不存在的设备号 17。
  npu-smi info 2>/dev/null | awk -v min="$MIN_FREE_GIB" '
    BEGIN { dev = 0 }
    /^\| [0-9]+ +[0-9]+ +\|/ {
      line = $0
      sub(/\|[ \t]*$/, "", line); sub(/[ \t]+$/, "", line)
      # ★ 这一行有**两组** `x / y`：前面是 `AICore(%) / Memory-Usage(MB)`（值是 0/0），
      #   后面才是 `HBM-Usage used / total`。而且空格不一致：同一份输出里既有
      #   `3123 / 65536` 也有 `11141/ 65536`。
      #   踩过两个坑（都白等了一轮 10 分钟起服）：
      #     · 取 `a[n-1]` 当 used：遇到 `used / total` 时 a[n-1] 是孤立的 "/" ⇒ used=0
      #       ⇒ 把**总容量**当成空闲 ⇒ 误判所有卡都 64 GiB 空闲
      #       ⇒ 选到只剩 50 GiB 的卡，起服报
      #         `Free memory on device (50.33/61.27 GiB) ... less than desired utilisation`。
      #     · 不归一化空格：`0    0    / 0` 会被拆成多段，同样算错。
      #   ⇒ 先去掉 `/` 两侧空格，再取**最后一个**含 `/` 的 token 作为 used/total。
      gsub(/[ \t]+\//, "/", line); gsub(/\/[ \t]+/, "/", line)
      n = split(line, a, /[ \t]+/)
      k = 0
      for (i = n; i >= 1; i--) if (a[i] ~ /\//) { k = i; break }
      if (k == 0) next
      split(a[k], p, "/")
      used = p[1] + 0
      tot = p[2] + 0
      if (tot > 0) {
        if ((tot - used) / 1024.0 >= min) printf "%d ", dev
        dev++
      }
    }'
}

# 空闲且 HBM 够的设备列表（两者取交）
usable_devices() {
  _b=" $(busy_devices) "
  _f=" $(free_enough_devices) "
  for d in $CHIP_PREF_ORDER; do
    case "$_b" in *" $d "*) continue ;; esac
    case "$_f" in *" $d "*) echo -n "$d " ;; esac
  done
}

# ★ 选卡顺序必须是**递增**的。实测：给 `8 9 10 11 14 15 0 1`（跨 NPU 且回绕）
#   容器里直接 `aclInit error code is 107001 / Invalid device ID`，白等一轮。
#   策略：① 先要完整的 [8..15] 块；② 再要 [0..7] 块；
#         ③ 都不完整时退到"可用集合里取前 8 张并**升序**"（不保证连续，但保证递增）。
pick_block() {
  _usable=" $(usable_devices) "
  for start in 8 0; do
    _ok=1
    for i in 0 1 2 3 4 5 6 7; do
      d=$((start + i))
      case "$_usable" in *" $d "*) ;; *) _ok=0; break ;; esac
    done
    if [ "$_ok" = "1" ]; then
      echo "$start $((start+1)) $((start+2)) $((start+3)) $((start+4)) $((start+5)) $((start+6)) $((start+7))"
      return 0
    fi
  done
  set -- $(usable_devices)
  [ "$#" -ge 8 ] || return 1
  echo "$1 $2 $3 $4 $5 $6 $7 $8" | tr ' ' '\n' | sort -n | tr '\n' ' ' | sed 's/ $//'
}

# 只看选卡结果、不起服（`DRY_SELECT=1 bash dcp_stage_capacity.sh`）。
# 2026-09-29 加：选卡逻辑踩过两次坑（HBM 不够、设备号回绕），
# 每次验证都要等一整轮 10 分钟起服，太贵。
CHIP_PREF_ORDER=${CHIP_PREF_ORDER:-"8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7"}

if [ "${DRY_SELECT:-0}" = "1" ]; then
  echo "busy      = $(busy_devices)"
  echo "free>=${MIN_FREE_GIB}G = $(free_enough_devices)"
  echo "usable    = $(usable_devices)"
  echo "pick_block= $(pick_block || echo '<凑不出 8 张>')"
  exit 0
fi
# AUTO_CHIPS=1：从 16 个设备里自动挑第一组连续 8 个空闲的（a3-21 是共用机，
# 固定 CHIPS 会被别人的临时 `acl_bw`/`python` 反复挡住）。
# ★ 2026-09-29：优先用**高 8 张（8-15）**。低 8 张（0-7）被别的账号的
#   `acl_bw` 带宽测试反复抢占（实测：选好卡后 20 秒内就被占，fail-closed 的
#   serve_a3.sh 直接退出）。高 8 张历史上更干净。
#   顺序：8-15 → 0-7 → 任意连续 8 张。
AUTO_CHIPS=${AUTO_CHIPS:-1}
for w in $(seq 1 240); do
  _allbusy="$(busy_devices)"
  if [ "$AUTO_CHIPS" = "1" ]; then
    _free="$(usable_devices)"
    if _picked="$(pick_block)"; then CHIPS="$_picked"; _busy=""; break; fi
    _busy="usable内无连续8张：'$_free'（机器忙='$_allbusy'，MIN_FREE_GIB=$MIN_FREE_GIB）"
  else
    _busy=""
    for d in $CHIPS; do case " $_allbusy " in *" $d "*) _busy="$_busy $d";; esac; done
    [ -z "$_busy" ] && break
  fi
  [ $((w % 2)) -eq 1 ] && echo "[cap]   第${w}次 机器忙=$_allbusy $_busy"
  sleep 30
done
[ -n "$_busy" ] && { echo "[cap][FAIL] 等不到 8 张空闲卡：$_busy"; exit 3; }
echo "[cap] 选中 CHIPS='$CHIPS'（机器忙=$_allbusy）"

run_serve() {
  docker rm -f "$NAME" >/dev/null 2>&1
  (
  cd "$PKG" || exit 1
  export MODEL PATCH_MODE=mount V41_DCP_MOUNT="$DCPMOUNT"
  # ★ 这个 env 必须**显式透传进容器**（serve_a2.sh 只带一批白名单 `-e`）。
# ★ 必须用 `${VAR:-default}`：写死会把调用方通过 `env DCP_EXTRA_ENV=...` 传进来的
#   额外开关**静默吞掉**（2026-09-29 踩过：加了 V41_DCP_IDX_DIAG=1 但容器里没有，
#   白等一轮 12 分钟起服才发现）。
  DCP_EXTRA_ENV=${DCP_EXTRA_ENV:-"V41_DCP_ALLOW_CAPACITY_PROBE=1 V41_DCP_DIAG=1"}
  export DCP_EXTRA_ENV
  export TP=8 DP=1 DEVS="$CHIPS" PORT="$PORT" NAME="$NAME"
  # BAT_TOKENS 是**容量第一杠杆**：SWA 复制态的 admission cap =
  # cdiv(127 + async_batches*BAT, 128) + 1，10 个 SWA 组按它预留。
  # 解析式（a2sim-ref/v41_capacity_sweep.py，已被真机逐位验证）：
  #   BAT=8192 + async → 4.08×     BAT=2048 + no-async → 7.88×
  export SPEC=0 MAX_SEQS=16 ENGRAM_DEVICE_INDEX=0 DROPCACHE=0
  export BAT_TOKENS=${BAT_TOKENS:-8192}
  export GPU_UTIL="$GPU_UTIL" KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES"
  export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
  export KV_ARGS_EXTRA="--decode-context-parallel-size $DCP${EXTRA_KV_ARGS:+ $EXTRA_KV_ARGS}"
  export RUN_ID="dcpcap_${STAMP}"
  exec bash scripts/serve_a3.sh
  ) > "$OUT/serve.log" 2>&1
}

# ★ a3-21 是共用机：别人会在我们选定卡之后**立刻**起 `acl_bw` 之类的任务。
#   `serve_a3.sh` 是 fail-closed（发现占用就退），所以这里必须重试而不是白等。
#   判据落在 `serve_a3` 的实际退出原因上，不是"我传了 CHIPS"。
attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[cap] 起服尝试 #$attempt，CHIPS='$CHIPS'"
  run_serve &
  _pid=$!
  sleep 20
  if kill -0 "$_pid" 2>/dev/null && grep -aq "起容器" "$OUT/serve.log" 2>/dev/null; then
    echo "[cap] 起服已提交（PID $_pid）"
    break
  fi
  wait "$_pid" 2>/dev/null
  if grep -aq "选中的卡里有正在被占用的" "$OUT/serve.log" 2>/dev/null; then
    _allbusy="$(busy_devices)"
    _free="$(usable_devices)"
    if ! _picked="$(pick_block)"; then
      echo "[cap][FAIL] 可用设备里凑不出连续 8 张：usable='$_free'（机器忙='$_allbusy'，MIN_FREE_GIB=$MIN_FREE_GIB）"
      exit 3
    fi
    CHIPS="$_picked"
    echo "[cap] 卡被别人抢了，重新选：CHIPS='$CHIPS'"
    [ "$attempt" -lt 12 ] || { echo "[cap][FAIL] 重试 12 次仍被抢"; exit 3; }
    sleep 20
    continue
  fi
  echo "[cap][FAIL] 起服失败（非选卡原因）："; tail -20 "$OUT/serve.log" | tr -d '\000'; exit 4
done

# 等「KV 容量行」或「引擎退出」，最多 50 min
for i in $(seq 1 300); do
  sleep 10
  if grep -aq "GPU KV cache size" "$OUT/serve.log" 2>/dev/null; then
    echo "[cap] 拿到容量行（$((i*10))s）"; break
  fi
  st=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)
  if [ "$st" = "false" ]; then echo "[cap] 容器已退出（$((i*10))s）"; break; fi
  [ $((i % 12)) -eq 0 ] && echo "[cap]   $((i*10))s log=$(du -k "$OUT/serve.log" 2>/dev/null | cut -f1)KB"
done

{
  echo "=== 命令行 ==="; grep -a "cmd: vllm serve" "$OUT/serve.log" | tail -1
  echo "=== 容量 ==="; grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$OUT/serve.log" | tail -3
  grep -aoE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$OUT/serve.log" | tail -2
  echo "=== DCP 相关 ==="; grep -aiE "dcp|decode.context.parallel|interleave" "$OUT/serve.log" | tail -20
  echo "=== 第一个异常 ==="; grep -anE "Error|error|Traceback|assert" "$OUT/serve.log" | head -20
} 2>&1 | tee "$OUT/evidence.txt"
echo "[cap] $(date -Is) OUT=$OUT"
