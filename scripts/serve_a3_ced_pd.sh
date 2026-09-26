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
# [CED-DSPARK] 2026-09-26：SPEC / DRAFT_GRAPH 的硬门按角色拆开。
#
#   * prefill（P）：**永远**要求 SPEC=0。DSpark 的 aux hidden state 取自目标层
#     37/38/39，而 P 在第 20 层 break —— 这三层的残差在 P 上物理不存在，
#     不是配置问题。见 docs/CED-PD-DSPARK-ANALYSIS-20260926.md §1。
#   * decode（D）：允许 SPEC=1 + DRAFT_GRAPH=0/1，但必须显式设
#     V41_CED_ALLOW_DSPARK=1。默认拒绝，与 PREFIX 同模式。
case "$role" in
  prefill)
    for setting in "SPEC:${SPEC:-0}" "DRAFT_GRAPH:${DRAFT_GRAPH:-0}"; do
      key=${setting%%:*}
      value=${setting#*:}
      if [ "$value" != 0 ]; then
        echo "[a3-ced][FAIL] prefill 角色要求 $key=0（当前 $key=$value）" >&2
        echo "[a3-ced][FAIL] DSpark 需要目标层 37/38/39，P 只跑 0..19，属架构性不可行。" >&2
        exit 2
      fi
    done
    ;;
  decode)
    if [ "${SPEC:-0}" != 0 ] || [ "${DRAFT_GRAPH:-0}" != 0 ]; then
      if [ "${V41_CED_ALLOW_DSPARK:-0}" != "1" ]; then
        echo "[a3-ced][FAIL] D 侧 SPEC=$SPEC DRAFT_GRAPH=$DRAFT_GRAPH 是实验臂；" >&2
        echo "[a3-ced][FAIL] 要跑请显式设 V41_CED_ALLOW_DSPARK=1。" >&2
        exit 2
      fi
      if [ "${SPEC:-0}" != 1 ]; then
        echo "[a3-ced][FAIL] 当前分支只验证过 SPEC=1（DSpark 单模型草稿）；当前 SPEC=$SPEC" >&2
        exit 2
      fi
      if [ "${DRAFT_GRAPH:-0}" != 0 ] && [ "${DRAFT_GRAPH:-0}" != 1 ]; then
        echo "[a3-ced][FAIL] DRAFT_GRAPH 只能是 0 或 1；当前 $DRAFT_GRAPH" >&2
        exit 2
      fi
      echo "[a3-ced][WARN] V41_CED_ALLOW_DSPARK=1：D 侧 DSpark（SPEC=$SPEC DRAFT_GRAPH=$DRAFT_GRAPH）属实验臂" >&2
    fi
    ;;
esac

stamp=$(date +%Y%m%d_%H%M%S)
export RUN_ID=${RUN_ID:-ced_${role}_${stamp}}
# [PATCH_MODE] 默认 mount（官方基础镜像 + `-v` 挂我们的文件）。
# 装了 `local/dsv41-a3-ced-pd:*` 工作镜像后可以设 PATCH_MODE=baked 完全不用挂载
# —— 见 deploy/a3-ced-pd/。这里**不再强制 mount**，否则烘好的镜像用不上。
export V41_CED_ROLE=$role PATCH_MODE=${PATCH_MODE:-mount}
if [ "$role" = prefill ]; then
  # P 的 SPEC/DRAFT_GRAPH 由上面的硬门保证为 0，这里显式定稿。
  export SPEC=0 DRAFT_GRAPH=0
else
  # D 侧保留调用方传入的值（默认 0），放行与否已在上面的门里判过。
  export SPEC=${SPEC:-0} DRAFT_GRAPH=${DRAFT_GRAPH:-0}
  # DSpark 的 eager 草稿在 SPEC=1 时必须让引擎知道；其余情况保持默认。
  if [ "$SPEC" != 0 ]; then
    export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-0}
  fi
fi
# [CED-PREFIX-EXPERIMENT] 2026-09-26：`PREFIX=1` 原先是硬门（直接 exit 2）。
# 基线口径的前缀缓存已在真机上验证**可用且正确**（144,000 tok 命中、命中答案与冷
# 路径逐字节相同，见 evidence/ced_prefix_hit_20260926/），所以"CED 能不能开缓存"
# 值得实测，而不是停在推断上。
#
# 打开方式（显式）：V41_CED_ALLOW_PREFIX=1 + PREFIX=1
#   ⚠️ 这是**实验臂**。文档 CED-PD-CACHE-HIT-PLAN-20260925.md §2 列了三处代码级前提
#   （调度器边界断言 / D 侧对 hashed 块预清零 / 上半层 SWA 的残留），
#   其中任一处没处理干净都会**静默算错**，所以结果不能当交付口径。
if [ "${PREFIX:-0}" != "0" ]; then
  if [ "${V41_CED_ALLOW_PREFIX:-0}" != "1" ]; then
    echo "[a3-ced][FAIL] PREFIX=$PREFIX；CED 原型默认要求 PREFIX=0。" >&2
    echo "[a3-ced][FAIL] 要跑缓存命中实验臂，显式设 V41_CED_ALLOW_PREFIX=1。" >&2
    exit 2
  fi
  echo "[a3-ced][WARN] V41_CED_ALLOW_PREFIX=1：CED 开前缀缓存属实验臂，结果不可当交付证据" >&2
  echo "[a3-ced][WARN] 需先处理 CED-PD-CACHE-HIT-PLAN-20260925.md §2 的三处前提" >&2
else
  export PREFIX=0
fi
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
      # [CED-GRAPH-PREREQ] 2026-09-25 加：图模式**必须**带
      # V41_CED_GRAPH_PROMPT_TAIL_EAGER=1，否则会按"图模式裸跑"的坏路径执行
      # 并输出乱码（形态：HTTP 200、completion_tokens 打满 max_tokens、
      # 无 finish_reason、含 <|box|>）。
      #
      # 踩过一次：`launch_ced_full.sh` 只设了 CED_EXPERIMENTAL_GRAPH=1、
      # 忘了这个开关，于是 144K 多轮 3/3 全乱码 —— 看起来像 CED 的缺陷，
      # 实际是启动参数不全（见 docs/CED-PD-GRAPH-PREREQ-20260925.md）。
      #
      # ⚠️ 这里**不再**要求 V41_CED_METADATA_INLINE：2026-09-25 核实，
      # 当前分支上没有任何代码读它（只出现在脚本与文档里）；
      # device-metadata 路径已无条件启用（dsa_v41.py::enable_device_metadata
      # 直接置 True）。通过 21/21 验收的那台 D 日志里
      # `[CED-META] inline metadata` 出现 **0** 次 ⇒ 它从来不是真判据。
      if [ "${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-0}" != "1" ]; then
        echo "[a3-ced][FAIL] CED_EXPERIMENTAL_GRAPH=1 还要求 V41_CED_GRAPH_PROMPT_TAIL_EAGER=1" >&2
        echo "[a3-ced][FAIL] 缺它会静默输出乱码（实测 144K 多轮 3/3 全错）。" >&2
        if [ "${V41_CED_ALLOW_BARE_GRAPH:-0}" = "1" ]; then
          echo "[a3-ced][WARN] V41_CED_ALLOW_BARE_GRAPH=1：已放行裸图模式，结果不可当正确性证据" >&2
        else
          echo "[a3-ced][FAIL] 若确实要裸跑图模式做诊断，设 V41_CED_ALLOW_BARE_GRAPH=1。" >&2
          exit 2
        fi
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

# [CED-POOL-GUARD] 2026-09-25：**两个角色都**适用 —— 池一旦让
# `num_blocks × 每块页步长` 越过 2³²，算子读到的块地址会 32 位回绕到别的块上，
# 长上下文请求随即静默变成"HTTP 200 + completion_tokens=1 + token_ids=[1]"。
# 详见 docs/CED-PD-BLOCK-BOUND-20260925.md §5.1.2 / §5.1.4。
#
# 每块页步长的最大值出现在槽位 3：layer-20 C1 KV(131072) + INT8 index K(16384)
# + FP16 scales(256) = 147712 B。回绕边界 ⌊2³²/147712⌋ = 29076。
#
# 判据用**页尾**（整页必须落在 4 GiB 内），所以是
#     num_blocks ≤ ⌊2³² / 147712⌋ = 29076
# 而不是"最大块号 ≤ 29076"。两者差一块：num_blocks=29077 时块 29076 的
# 最后 54528 B 已经回绕（实测它"通过"只是因为回绕落点恰为恒零的 null block 0）。
#
# 实测边界是在 D 侧定的（D 侧的 22 条配对样本里 D_max 是完美判别量），
# 但 P 侧默认池 29721 块同样越界（29721 × 147712 = 4.44 GB > 2³²），
# 而且我们**没有**证明 P 侧为什么不受影响 ⇒ 两个角色一律按同一上界钳位，
# 不能靠"P 看起来没事"来放行。要复现越界行为需显式设
# V41_CED_ALLOW_32BIT_OVERFLOW=1（连接器侧同一开关）。
#
# 这里只是**配置侧的预防**；真正的强制校验在
# experimental/ced/mooncake_hybrid_connector.py 的 [CED-32BIT-GUARD]，
# 那里拿得到 worker 实际注册的 stride，且 num_blocks 已经定稿。
if [ "${V41_CED_ALLOW_32BIT_OVERFLOW:-0}" != "1" ]; then
  CED_MAX_NUM_BLOCKS=${CED_MAX_NUM_BLOCKS:-29076}
  CED_D_BYTES_PER_BLOCK=${CED_D_BYTES_PER_BLOCK:-540928}
  ced_pool_cap=$(( CED_MAX_NUM_BLOCKS * CED_D_BYTES_PER_BLOCK ))
  if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && [ "$KV_CACHE_MEMORY_BYTES" -gt "$ced_pool_cap" ]; then
    echo "[a3-ced][WARN] KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES 会让 $role 池超过 4 GiB 寻址上界" >&2
    echo "[a3-ced][WARN] 钳到 $ced_pool_cap（num_blocks=$CED_MAX_NUM_BLOCKS，最大可用块号 $((CED_MAX_NUM_BLOCKS - 1))）" >&2
    export KV_CACHE_MEMORY_BYTES=$ced_pool_cap
  elif [ -z "${KV_CACHE_MEMORY_BYTES:-}" ]; then
    export KV_CACHE_MEMORY_BYTES=$ced_pool_cap
    echo "[a3-ced] $role 池按 4 GiB 上界设置：$KV_CACHE_MEMORY_BYTES B（num_blocks=$CED_MAX_NUM_BLOCKS）"
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
