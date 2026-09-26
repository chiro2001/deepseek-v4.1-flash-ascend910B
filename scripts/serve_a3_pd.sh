#!/usr/bin/env bash
# DeepSeek V4.1 Flash：A3 单机 1P1D、双 TP8、BF16 KV 的基础 PD 角色入口。
#
# 这是完整模型的两实例 PD：P/D 各自加载全模型并使用 8 张卡；P 不是只执行
# 前 20 层。脚本只配置已经实测的 MooncakeHybridConnector 基线，不打开 DRAM
# KV 池、KV8 或 draft 入图。
#
# 用法（两个终端分别执行）：
#   MODEL=/path/to/model bash scripts/serve_a3_pd.sh prefill
#   MODEL=/path/to/model bash scripts/serve_a3_pd.sh decode
#
# 先检查参数且不碰 Docker：
#   DRY_RUN=1 MODEL=/path/to/model bash scripts/serve_a3_pd.sh prefill
#
# 可覆盖的角色参数：DEVS、PORT、KV_PORT、NAME、RUN_ID；也可用
# PD_PREFILL_DEVS / PD_DECODE_DEVS、PD_PREFILL_PORT / PD_DECODE_PORT、
# PD_PREFILL_KV_PORT / PD_DECODE_KV_PORT。默认卡号和端口对应 A3-21 的已测口径。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
role=${1:-}

case "$role" in
  prefill)
    _default_devs=${PD_PREFILL_DEVS:-"0 1 2 3 4 5 6 7"}
    _default_port=${PD_PREFILL_PORT:-18550}
    _default_kv_port=${PD_PREFILL_KV_PORT:-18650}
    _kv_role=kv_producer
    _default_name=dsv41-pd-v41-prefill
    ;;
  decode)
    _default_devs=${PD_DECODE_DEVS:-"8 9 10 11 12 13 14 15"}
    _default_port=${PD_DECODE_PORT:-18551}
    _default_kv_port=${PD_DECODE_KV_PORT:-18651}
    _kv_role=kv_consumer
    _default_name=dsv41-pd-v41-decode
    ;;
  *)
    echo "用法：$0 prefill|decode" >&2
    exit 2
    ;;
esac

MODEL=${MODEL:-}
if [ -z "$MODEL" ] && [ "${DRY_RUN:-0}" = "1" ]; then
  MODEL=/nonexistent/DRY_RUN_MODEL
fi
if [ -z "$MODEL" ]; then
  echo "[a3-pd][FAIL] 必须设置 MODEL=<完整 DeepSeek V4.1 W4A8 模型目录>" >&2
  exit 2
fi

TP=${TP:-8}
DP=${DP:-1}
if [ "$TP" != 8 ] || [ "$DP" != 1 ]; then
  echo "[a3-pd][FAIL] 该基线固定 TP=8、DP=1；当前 TP=$TP DP=$DP" >&2
  echo "                 需要 DP 或其他 TP 形态时请另做单变量验证。" >&2
  exit 2
fi

DEVS=${DEVS:-$_default_devs}
PORT=${PORT:-$_default_port}
KV_PORT=${KV_PORT:-$_default_kv_port}
_stamp=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-pdv41_${role}_${_stamp}}
NAME=${NAME:-${PD_NAME:-${_default_name}-${_stamp}}}
if [ "$NAME" = "dsv41-a3" ]; then
  echo "[a3-pd][FAIL] PD 角色不能使用默认容器名 dsv41-a3；请给 P/D 各自唯一的 NAME。" >&2
  exit 2
fi

if [ "${DRY_RUN:-0}" != "1" ] && command -v docker >/dev/null 2>&1 \
    && docker inspect "$NAME" >/dev/null 2>&1; then
  echo "[a3-pd][FAIL] 容器名已存在：$NAME；为避免 serve_a2.sh 清理旧容器，请换 NAME。" >&2
  exit 3
fi

# vLLM 的 kv-transfer-config 由 serve_a2.sh → inner.sh → serve_v2.sh 透传。
# JSON 必须保持紧凑，否则兼容展开会把 JSON 内空格拆成额外 CLI 参数。
KV_CONFIG=$(printf '{"kv_connector":"MooncakeHybridConnector","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}' "$_kv_role" "$KV_PORT")

export MODEL IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
export PATCH_MODE=${PATCH_MODE:-mount} NAME PORT SERVED_NAME=${SERVED_NAME:-deepseek-v41-pd}
export TP DP DEVS RUN_ID
# [BIND-HOST] P/D 两个半边都只应由**本机**的负载均衡代理访问（18992 本身绑在
#   127.0.0.1，跨机访问走隧道而不是直连半边）⇒ 默认只监听回环。
#   把半边暴露到 0.0.0.0 的后果有实测先例：2026-09-27 00:01，一条直连 decode
#   的普通请求让 EngineCore 退出、整个 D 实例死掉（见 v41_decode_guard.py 头注释）。
#   确实需要跨机直连时显式 HOST=0.0.0.0，并同时处理来源限制/鉴权。
export HOST=${HOST:-127.0.0.1}
# [MM-LIMIT] 多图上限：**P 与 D 必须同值**（两边都在 API 层校验同一份请求体，
#   D 更小的话请求会在 D 上被 400）。默认 4 张；要收更多图就 P/D 同时调大。
#   0 张图（纯文本部署）用 VISION=0 表达，不是把这里设成 0。
MM_LIMIT_IMAGES=${MM_LIMIT_IMAGES:-4}
case "$MM_LIMIT_IMAGES" in
  ''|*[!0-9]*)
    echo "[a3-pd][FAIL] MM_LIMIT_IMAGES 必须是正整数，当前 '$MM_LIMIT_IMAGES'" >&2
    exit 2 ;;
  0)
    echo "[a3-pd][FAIL] MM_LIMIT_IMAGES=0 会让所有带图的请求被 400；纯文本请用 VISION=0。" >&2
    exit 2 ;;
esac
export MM_LIMIT_IMAGES
export KV_ARGS_EXTRA="--kv-transfer-config $KV_CONFIG"

# 下面是 2026-09-23 A3-21 完整模型验收时的基线：
# BF16 KV、Engram host 路径、CPU_BIND=0、prefix cache 关闭、DSpark eager。
export KV_DTYPE=${KV_DTYPE:-bfloat16} ENGRAM=${ENGRAM:-1} ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
export CPU_BIND=${CPU_BIND:-0} PYTHON_PGO=${PYTHON_PGO:-0}
export MAX_LEN=${MAX_LEN:-147456} MAX_SEQS=${MAX_SEQS:-4} BAT_TOKENS=${BAT_TOKENS:-8192} GPU_UTIL=${GPU_UTIL:-0.92}
export SPEC=${SPEC:-1} SP_TOKENS=${SP_TOKENS:-7} DRAFT_GRAPH=${DRAFT_GRAPH:-0} PREFIX=${PREFIX:-0}
export STATIC_KERNEL=${STATIC_KERNEL:-1} NPUGRAPH_EX=${NPUGRAPH_EX:-1} DROPCACHE=${DROPCACHE:-0}
export VISION=${VISION:-1} WAIT_READY=${WAIT_READY:-0} DRY_RUN=${DRY_RUN:-0}

if [ "$DRY_RUN" = "1" ]; then
  grep -qF 'export KV_ARGS_EXTRA="\${KV_ARGS_EXTRA:-}"' "$PKG/scripts/serve_a2.sh" \
    || { echo "[a3-pd][FAIL] scripts/serve_a2.sh 没有 KV_ARGS_EXTRA 透传入口" >&2; exit 2; }
  grep -qF 'ARGS+=($KV_ARGS_EXTRA)' "$PKG/scripts/serve_v2.sh" \
    || { echo "[a3-pd][FAIL] scripts/serve_v2.sh 没有 KV_ARGS_EXTRA CLI 入口" >&2; exit 2; }
  echo "[a3-pd-dry] role=$role devs='$DEVS' port=$PORT kv_port=$KV_PORT name=$NAME"
  echo "[a3-pd-dry] tp=$TP dp=$DP kv_dtype=$KV_DTYPE engram_device_index=$ENGRAM_DEVICE_INDEX cpu_bind=$CPU_BIND"
  echo "[a3-pd-dry] host=$HOST mm_limit_images=$MM_LIMIT_IMAGES（P/D 同值；客户端 --max-images 应一致）"
  echo "[a3-pd-dry] kv_args=$KV_ARGS_EXTRA"
  echo "[a3-pd-dry] draft_graph=$DRAFT_GRAPH prefix=$PREFIX max_len=$MAX_LEN max_seqs=$MAX_SEQS bat=$BAT_TOKENS"
fi

cd "$PKG"
exec bash scripts/serve_a3.sh
