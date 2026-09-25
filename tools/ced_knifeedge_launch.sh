#!/usr/bin/env bash
# 刀锋实验（a3-22 chip0=P / chip1=D / proxy 18962）的起停编排。
#
#   bash tools/ced_knifeedge_launch.sh start-all <TAG> <C>   # P(默认池)+D(池=C 块)+proxy
#   bash tools/ced_knifeedge_launch.sh start-d   <TAG> <C>   # 只重启 D（保留 P/proxy）
#   bash tools/ced_knifeedge_launch.sh start-p   <TAG>
#   bash tools/ced_knifeedge_launch.sh start-proxy <TAG>
#   bash tools/ced_knifeedge_launch.sh stop-d                # 只停 D
#   bash tools/ced_knifeedge_launch.sh stop-all              # 停我们全部容器
#   bash tools/ced_knifeedge_launch.sh health
#
# 安全边界：只操作名字带 dsv41-ced-single 的容器；只用 Phy-ID 0/1；不 kill 任何
# 非本实验进程。D 池字节数 = C × PAGE_BYTES（实测每块 540928 B = 4 个寻址段的
# block_len 之和：3×131072 + 147712）。
set -uo pipefail

PKG=${PKG:-$HOME/projects/dsv41-ced-singlechip/pkg-swa-clip}
MODEL=${MODEL:-$HOME/projects/dsv41-ced-singlechip/model-tiny}
PAGE_BYTES=${PAGE_BYTES:-540928}
P_DEV=${P_DEV:-0}
D_DEV=${D_DEV:-1}
P_PORT=${P_PORT:-18960}
D_PORT=${D_PORT:-18961}
P_KV_PORT=${P_KV_PORT:-19060}
D_KV_PORT=${D_KV_PORT:-19061}
PROXY_PORT=${PROXY_PORT:-18962}
MAX_LEN=${MAX_LEN:-65536}
P_KV_BYTES=${P_KV_BYTES:-1073741824}
DOCKER=${DOCKER:-docker}

say() { printf '[knife] %s\n' "$*"; }

health_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$1/health" || true; }

our_names() {
  $DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^dsv41-ced-single' || true
}

d_names() {
  $DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^dsv41-ced-single-decode' || true
}

chip_procs() {
  # 只读：打印 NPU 进程表（Phy-ID 0/1 的占用核对用），不动任何进程。
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | head -20
}

cmd_health() {
  say "P($P_PORT)=$(health_code "$P_PORT") D($D_PORT)=$(health_code "$D_PORT") proxy($PROXY_PORT)=$(health_code "$PROXY_PORT")"
  say "容器：$($DOCKER ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | grep -E '^dsv41-ced-single' | tr '\n' ' ')"
}

wait_health() {
  local port=$1 timeout_s=$2 name=$3 i code
  for i in $(seq 1 "$timeout_s"); do
    code=$(health_code "$port")
    if [ "$code" = "200" ]; then say "$name 就绪（${i}s）"; return 0; fi
    sleep 1
  done
  say "$name 未就绪（最后 code=$code）"
  return 1
}

stop_d() {
  local names code
  names=$(d_names)
  if [ -n "$names" ]; then
    say "删除 D 容器：$(echo "$names" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    $DOCKER rm -f $names >/dev/null 2>&1 || true
  fi
  sleep 3
  code=$(health_code "$D_PORT")
  say "D 端口 $D_PORT 现在 = $code（必须非 200，确认没有旧容器替新容器答 health）"
  [ "$code" != "200" ] || { say "端口仍被占用，停止"; return 1; }
}

start_d() {
  local tag=${1:?start-d <TAG> <C>} c=${2:?start-d <TAG> <C>}
  local stamp d_bytes d_run d_name
  stamp=$(date +%Y%m%d_%H%M%S)
  d_bytes=$((c * PAGE_BYTES))
  d_run="${tag}_d"
  d_name="dsv41-ced-single-decode-${tag}-${stamp}"
  mkdir -p "$PKG/results/$d_run"
  printf '{"tag": "%s", "pool_blocks": %s, "page_bytes": %s, "d_kv_bytes": %s, "max_len": %s}\n' \
    "$tag" "$c" "$PAGE_BYTES" "$d_bytes" "$MAX_LEN" > "$PKG/results/$d_run/arm.json"
  say "arm=$tag C=$c D_KV_CACHE_MEMORY_BYTES=$d_bytes"
  stop_d || return 1
  say "chip$D_DEV 上的进程（起服前核对）："
  chip_procs | sed 's/^/    /'
  cd "$PKG" || return 1
  setsid nohup env \
    DEVS="$D_DEV" PORT="$D_PORT" KV_PORT="$D_KV_PORT" MODEL="$MODEL" \
    MAX_LEN="$MAX_LEN" MAX_SEQS=4 BAT_TOKENS=1024 GPU_UTIL=0.5 \
    KV_CACHE_MEMORY_BYTES="$d_bytes" LOAD_FORMAT=dummy QUANTIZATION=none SEED=0 \
    V41_CED_KVGEOM=1 V41_CED_BLOCK_TRACE=1 \
    V41_CED_BLOCK_DUMP_DIR="/opt/dsv41/results/$d_run/blockdump" \
    RUN_ID="$d_run" NAME="$d_name" \
    bash scripts/serve_a3_ced_single.sh decode > "results/$d_run/d_launch.log" 2>&1 &
  say "D 已提交启动：$d_name（日志 results/$d_run/d_launch.log）"
  wait_health "$D_PORT" 300 "D" || { tail -20 "$PKG/results/$d_run/serve.log" 2>/dev/null; return 1; }
}

start_p() {
  local tag=${1:?start-p <TAG>}
  local stamp p_run p_name names
  if [ "$(health_code "$P_PORT")" = "200" ]; then
    say "复用已在跑的 P（$P_PORT）"
    return 0
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  p_run="${tag}_p"
  p_name="dsv41-ced-single-prefill-${tag}-${stamp}"
  mkdir -p "$PKG/results/$p_run"
  names=$($DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^dsv41-ced-single-prefill' || true)
  [ -n "$names" ] && $DOCKER rm -f $names >/dev/null 2>&1 || true
  cd "$PKG" || return 1
  setsid nohup env \
    DEVS="$P_DEV" PORT="$P_PORT" KV_PORT="$P_KV_PORT" MODEL="$MODEL" \
    MAX_LEN="$MAX_LEN" MAX_SEQS=4 BAT_TOKENS=1024 GPU_UTIL=0.5 \
    KV_CACHE_MEMORY_BYTES="$P_KV_BYTES" LOAD_FORMAT=dummy QUANTIZATION=none SEED=0 \
    V41_CED_KVGEOM=1 V41_CED_BLOCK_TRACE=1 \
    RUN_ID="$p_run" NAME="$p_name" \
    bash scripts/serve_a3_ced_single.sh prefill > "results/$p_run/p_launch.log" 2>&1 &
  say "P 已提交启动：$p_name（日志 results/$p_run/p_launch.log）"
  wait_health "$P_PORT" 300 "P" || { tail -20 "$PKG/results/$p_run/serve.log" 2>/dev/null; return 1; }
}

start_proxy() {
  local tag=${1:?start-proxy <TAG>}
  local stamp x_name code names
  if [ "$(health_code "$PROXY_PORT")" = "200" ]; then
    say "复用已在跑的 proxy（$PROXY_PORT）"
    return 0
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  x_name="dsv41-ced-single-proxy-${tag}-${stamp}"
  names=$($DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^dsv41-ced-single-proxy' || true)
  [ -n "$names" ] && $DOCKER rm -f $names >/dev/null 2>&1 || true
  mkdir -p "$PKG/results/${tag}_d"
  ( cd "$PKG" && NAME="$x_name" PROXY_PORT="$PROXY_PORT" \
      PREFILL_PORT="$P_PORT" DECODE_PORT="$D_PORT" \
      bash scripts/serve_a3_pd_proxy.sh > "results/${tag}_d/proxy.log" 2>&1 )
  sleep 5
  code=$(health_code "$PROXY_PORT")
  say "proxy health=$code（$x_name）"
}

main() {
  case "${1:-}" in
    start-d)     shift; start_d "$@" ;;
    start-p)     shift; start_p "$@" ;;
    start-proxy) shift; start_proxy "$@" ;;
    start-all)   shift; local tag=${1:?} c=${2:?}; start_p "$tag" && start_d "$tag" "$c" && start_proxy "$tag" ;;
    stop-d)      stop_d ;;
    stop-all)
      local names
      names=$(our_names)
      if [ -n "$names" ]; then
        say "删除：$(echo "$names" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        $DOCKER rm -f $names >/dev/null 2>&1 || true
      fi
      sleep 3
      cmd_health ;;
    health) cmd_health ;;
    *) echo "用法：$0 start-all <TAG> <C> | start-p <TAG> | start-d <TAG> <C> | start-proxy <TAG> | stop-d | stop-all | health" >&2; exit 2 ;;
  esac
}

main "$@"
