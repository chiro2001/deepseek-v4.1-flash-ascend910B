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
# [CED-DSPARK] SPEC / DRAFT_GRAPH 的硬门按角色拆开。
#
#   * prefill（P）：**永远**要求 SPEC=0。DSpark 的 aux hidden state 取自目标层
#     37/38/39，而 P 在第 20 层 break —— 这三层的残差在 P 上物理不存在，
#     不是配置问题。见 docs/CED-PD-DSPARK-ANALYSIS-20260926.md §1。
#   * decode（D）：**默认 SPEC=1 + DRAFT_GRAPH=1**（2026-09-27 起为交付口径，
#     实测见 docs/CED-PD-CACHE-HIT-PLAN-20260925.md §13）。
#
# 显式 `V41_CED_ALLOW_DSPARK=0` ⇒ D 侧退回 SPEC=0/DRAFT_GRAPH=0。
#   必须保留这个"显式关"：`patches/files/model.py` 用**同一个 env** 做引擎侧
#   的门（默认已改为 1）。若这里不认它，设了 0 的人会拿到 SPEC=1，然后在模型
#   构造时被引擎侧的门拒掉 —— 报错点离原因很远。
if [ "${V41_CED_ALLOW_DSPARK:-1}" = "0" ]; then
  export SPEC=0 DRAFT_GRAPH=0
fi
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
    # ⚠️ 全块统一用 `${VAR:-1}`：默认路径下这两个变量**可能真的未设置**，
    #    而本脚本是 `set -u`。裸写或混用 `:-0` 会让"不带任何 env 启动 D"
    #    直接崩（2026-09-27 加默认值时踩过两次：裸 `$SPEC` → unbound；
    #    内层仍用 `:-0` → 把默认值判成非法）。`bash -n` 两种都查不出来，
    #    由 tools/selftest_ced_defaults.sh 抓到。
    spec=${SPEC:-1}
    draft=${DRAFT_GRAPH:-1}
    if [ "$spec" != 0 ] || [ "$draft" != 0 ]; then
      if [ "$spec" != 1 ]; then
        echo "[a3-ced][FAIL] 当前分支只验证过 SPEC=1（DSpark 单模型草稿）；当前 SPEC=$spec" >&2
        exit 2
      fi
      if [ "$draft" != 0 ] && [ "$draft" != 1 ]; then
        echo "[a3-ced][FAIL] DRAFT_GRAPH 只能是 0 或 1；当前 DRAFT_GRAPH=$draft" >&2
        exit 2
      fi
      echo "[a3-ced] D 侧 DSpark：SPEC=$spec DRAFT_GRAPH=$draft（交付口径）"
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
  # D 侧默认 = 交付口径（DSpark 开）；取值合法性已在上面的门里判过。
  export SPEC=${SPEC:-1} DRAFT_GRAPH=${DRAFT_GRAPH:-1}
  # DSpark 的草稿在 SPEC=1 时必须让引擎知道（`model.py` 用同一个 env 做门）。
  if [ "$SPEC" != 0 ]; then
    export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-1}
  fi
fi
# [CED-PREFIX] 前缀缓存**默认开**（2026-09-27 起为交付口径）。
#
# 转正依据：144K 常规/整池/交错 + 1M 整池 + 1M 部分命中→整段命中全部正确，
# 命中答案与冷路径逐字节相同（≈16–18×）；另修掉三处会打死引擎的问题
# （12-group 形状、P 侧命中回退、D 侧空接收），且三者在真机上都有可观测触发
# 痕迹。见 docs/CED-PD-CACHE-HIT-PLAN-20260925.md §11–§13 与
# evidence/ced_prefix_hit_20260926/。
#
# 关掉：`PREFIX=0`（显式）。旧写法 `V41_CED_ALLOW_PREFIX=0` 也认。
if [ "${V41_CED_ALLOW_PREFIX:-1}" = "0" ]; then
  export PREFIX=0
fi
export PREFIX=${PREFIX:-1}
# [MAX_LEN] 上下文窗口默认 **1M**（2026-09-27 起与 deploy 形态同口径）。
#   原先这里沿用 serve_a3_pd.sh 的 147456(144K)，于是"用脚本起"和
#   "用 deploy/launch 起"拿到的窗口不同 —— 1M 是这套 CED-PD 形态的
#   已验证能力（144K/1M 常规·整池·交错命中均通过），不该只在一种交付面开放。
#   ⚠️ 1M 需要 KV 池够大：本形态默认 KV_CACHE_MEMORY_BYTES 对应 num_blocks=29076
#      ⇒ 29076×128 = 3.72M tokens 容量。`MAX_SEQS=4` 时**四路同时满 1M 会超出池**，
#      引擎会自行限流（不会崩，但别假定 4×1M 一定同时进得来）。
#   收窄窗口：显式 `MAX_LEN=<更小的值>`。
export MAX_LEN=${MAX_LEN:-1048576}
# [STATIC_KERNEL] 按角色取交付口径默认：D=1（−4.4 ms/step、−9.6%）、
#   P=0（P 侧没做过单变量，保守）。与 deploy/a3-ced-pd/launch/serve_{p,d}.sh
#   以及 2026-09-27 验证过的配置逐项一致 —— 避免"脚本默认 ≠ 交付默认"。
if [ "$role" = decode ]; then
  export STATIC_KERNEL=${STATIC_KERNEL:-1}
else
  export STATIC_KERNEL=${STATIC_KERNEL:-0}
fi
if [ "$role" = decode ]; then
  # [CED-GRAPH-DEFAULT] 交付口径 = **图模式**（GRAPH=1 EAGER=0）。
  #   原先两个臂都必须显式选（裸跑会 exit 2），理由是"图模式短针 2/2 乱码"。
  #   那条乱码已由 [CED-SWA-CLIP] 修掉，并通过 144K/1M 四针 21/21 验收
  #   ⇒ 图模式现在是交付口径，把它设成默认；eager 仍是**显式的**诊断臂
  #   （`CED_DIAGNOSTIC_EAGER=1`），不会被隐式选中。
  #   ⚠️ 只在没点名 eager 时才默认：否则 `CED_DIAGNOSTIC_EAGER=1` 会同时踩到
  #      两个开关，落到下面的 `*)` 分支被拒。
  if [ "${CED_DIAGNOSTIC_EAGER:-0}" != "1" ]; then
    export CED_EXPERIMENTAL_GRAPH=${CED_EXPERIMENTAL_GRAPH:-1}
  fi
  # 图模式的硬前提（漏了会静默乱码，见下方 [CED-GRAPH-PREREQ]）。
  export V41_CED_GRAPH_PROMPT_TAIL_EAGER=${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-1}
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
      echo "[a3-ced] D 图模式（交付口径）：GRAPH=1 EAGER=0，prompt-tail eager 已就位"
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
echo "[a3-ced] role=$V41_CED_ROLE name=$NAME max_len=${MAX_LEN:-1048576} spec=$SPEC prefix=$PREFIX graph=${GRAPH:-1} eager=${EAGER:-0}"
exec bash "$HERE/serve_a3_pd.sh" "$role"
