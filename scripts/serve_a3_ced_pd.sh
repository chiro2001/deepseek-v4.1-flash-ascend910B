#!/usr/bin/env bash
# A3 real-weight CED-PD roles. Both roles load full weights; the prefill role
# executes layers 0..19 plus the layer-20 global source, and the decode role
# installs the bounded-replay scheduler and attention implementation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
role=${1:-}
case "$role" in
  prefill|decode) ;;
  *) echo "用法：$0 prefill|decode" >&2; exit 2 ;;
esac

if [ -n "${V41_CED_ROLE:-}" ] && [ "$V41_CED_ROLE" != "$role" ]; then
  echo "[a3-ced][FAIL] V41_CED_ROLE=$V41_CED_ROLE 与角色 $role 不一致" >&2
  exit 2
fi
for setting in "SPEC:${SPEC:-0}" "PREFIX:${PREFIX:-0}" "DRAFT_GRAPH:${DRAFT_GRAPH:-0}"; do
  key=${setting%%:*}
  value=${setting#*:}
  if [ "$value" != 0 ]; then
    echo "[a3-ced][FAIL] $key=$value；当前 CED replay 原型要求 $key=0" >&2
    exit 2
  fi
done

stamp=$(date +%Y%m%d_%H%M%S)
export RUN_ID=${RUN_ID:-ced_${role}_${stamp}}
export V41_CED_ROLE=$role SPEC=0 PREFIX=0 DRAFT_GRAPH=0 PATCH_MODE=mount
export STATIC_KERNEL=${STATIC_KERNEL:-0}
if [ "$role" = decode ]; then
  # Both arms are diagnostic until the graph-mode corruption is fixed.
  # Never turn the accurate but slower eager arm into an implicit delivery.
  case "${CED_DIAGNOSTIC_EAGER:-0}:${CED_EXPERIMENTAL_GRAPH:-0}" in
    1:0)
      export GRAPH=${GRAPH:-0} EAGER=${EAGER:-1}
      if [ "$GRAPH" != 0 ] || [ "$EAGER" != 1 ]; then
        echo "[a3-ced][FAIL] CED_DIAGNOSTIC_EAGER=1 要求 GRAPH=0 EAGER=1" >&2
        exit 2
      fi
      echo "[a3-ced][WARN] D eager 仅供正确性和定位基线，不是性能交付配置" >&2
      ;;
    0:1)
      export GRAPH=${GRAPH:-1} EAGER=${EAGER:-0}
      if [ "$GRAPH" != 1 ] || [ "$EAGER" != 0 ]; then
        echo "[a3-ced][FAIL] CED_EXPERIMENTAL_GRAPH=1 要求 GRAPH=1 EAGER=0" >&2
        exit 2
      fi
      echo "[a3-ced][WARN] D 图模式仅供定位；真实权重短针在此模式 2/2 失败" >&2
      ;;
    *)
      echo "[a3-ced][FAIL] CED D 尚无可交付配置：eager 仅供诊断，图模式短针 2/2 乱码。定位时显式设置 CED_DIAGNOSTIC_EAGER=1 或 CED_EXPERIMENTAL_GRAPH=1" >&2
      exit 2
      ;;
  esac
fi
export SERVED_NAME=${SERVED_NAME:-deepseek-v41-ced-pd}
export NAME=${NAME:-dsv41-ced-${role}-${stamp}}

# [CED-D-POOL-GUARD] 2026-09-25：D 侧的 KV 池若放大到 29084 块以上，长上下文请求
# 会偶发静默空回答（HTTP 200 + completion_tokens=1 + token_ids=[1]）。实测边界
# T ∈ (29077, 29084]（见 docs/CED-PD-BLOCK-BOUND-20260925.md）：
#   C=29077/29078 → 池顶 29076/29077，全过
#   C=29128/29129 → 池顶 29127/29128，必失败
#   C=29600 同一实例内 max=29084 失败、max=29063/27482 通过
# 在找到 32 位截断的确切位置之前，decode 角色的池一律钳到 T-1 = 29077 块，
# 换算成 KV_CACHE_MEMORY_BYTES = 29078 × 540928（每块实测 540928 B）。
if [ "$role" = decode ]; then
  CED_D_MAX_POOL_BLOCKS=${CED_D_MAX_POOL_BLOCKS:-29077}
  CED_D_BYTES_PER_BLOCK=${CED_D_BYTES_PER_BLOCK:-540928}
  ced_pool_cap=$(( (CED_D_MAX_POOL_BLOCKS + 1) * CED_D_BYTES_PER_BLOCK ))
  if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && [ "$KV_CACHE_MEMORY_BYTES" -gt "$ced_pool_cap" ]; then
    echo "[a3-ced][WARN] KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES 会让 D 池超过安全块数" >&2
    echo "[a3-ced][WARN] 钳到 $ced_pool_cap（=$((CED_D_MAX_POOL_BLOCKS + 1)) 块，池顶 $CED_D_MAX_POOL_BLOCKS）" >&2
    export KV_CACHE_MEMORY_BYTES=$ced_pool_cap
  elif [ -z "${KV_CACHE_MEMORY_BYTES:-}" ]; then
    export KV_CACHE_MEMORY_BYTES=$ced_pool_cap
    echo "[a3-ced] D 池按安全上限设置：$KV_CACHE_MEMORY_BYTES B（$((CED_D_MAX_POOL_BLOCKS + 1)) 块）"
  fi
fi
if [ -n "${CED_SNAPSHOT_POS:-}" ]; then
  export V41_CED_SNAPSHOT_POS=$CED_SNAPSHOT_POS
  export V41_CED_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/snapshots"
fi
if [ -n "${CED_H20_SNAPSHOT_POS:-}" ]; then
  export V41_CED_H20_SNAPSHOT_POS=$CED_H20_SNAPSHOT_POS
  export V41_CED_H20_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/h20_snapshots"
fi
if [ -n "${CED_LAYER_SNAPSHOT_POS:-}" ]; then
  export V41_CED_LAYER_SNAPSHOT_POS=$CED_LAYER_SNAPSHOT_POS
  export V41_CED_LAYER_SNAPSHOT_DIR="/opt/dsv41/results/$RUN_ID/layer_snapshots"
fi
if [ "${CED_CAPTURE_DECODE:-0}" = 1 ]; then
  export V41_CED_CAPTURE_DECODE=1
fi
echo "[a3-ced] role=$V41_CED_ROLE name=$NAME max_len=${MAX_LEN:-147456} spec=$SPEC prefix=$PREFIX graph=${GRAPH:-1} eager=${EAGER:-0}"
exec bash "$HERE/serve_a3_pd.sh" "$role"
