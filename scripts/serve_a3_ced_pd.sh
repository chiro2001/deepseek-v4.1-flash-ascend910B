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
# ============================================================================
# [SPEC_MODE] D 侧推测解码（DSpark）的**三个可选档位** —— 单一开关，互斥可选。
#
#   SPEC_MODE=on       ★ 默认（= 现行交付口径）：全开 SPEC，固定 K=SP_TOKENS
#                          ⇒ SPEC=1 DRAFT_GRAPH=1
#   SPEC_MODE=off        全关 SPEC（纯自回归，无草稿）
#                          ⇒ SPEC=0 DRAFT_GRAPH=0（serve_v2 连 --speculative-config 都不加）
#   SPEC_MODE=dynamic    动态 K：按**当时并发数**切 1 ↔ K=0
#                          ⇒ SPEC=1 DRAFT_GRAPH=1 + SP_SCHEDULE + 上游降级门的豁免
#
# P 侧**恒为 off**，与 SPEC_MODE 无关：DSpark 的 aux hidden state 取自目标层
#   37/38/39，而 P 在第 20 层 break —— 这三层的残差在 P 上物理不存在，
#   不是配置问题。见 docs/CED-PD-DSPARK-ANALYSIS-20260926.md §1。
#   显式在 P 上要求 on/dynamic ⇒ **fail-closed**（不静默降级，否则人会以为开了）。
#
# 向后兼容（三个旧开关仍然认，且**只在 SPEC_MODE 未给时**参与推断）：
#   `V41_CED_ALLOW_DSPARK=0` ⇒ off      （patches/files/model.py 用同一个 env 做引擎侧门）
#   `V41_CED_DYNAMIC_SPEC=1` ⇒ dynamic  （旧写法，缺 FULL_GRAPHS 会在 serve_a2 的 die 处失败）
#   `SPEC=0`                 ⇒ off ；`SPEC=1` ⇒ on
# 若 SPEC_MODE 与上面任一个**同时给且矛盾** ⇒ fail-closed（不静默择一）。
# ============================================================================
_mode=${SPEC_MODE:-}
_mode_explicit=0
[ -n "$_mode" ] && _mode_explicit=1
_legacy_off=0; [ "${V41_CED_ALLOW_DSPARK:-1}" = "0" ] && _legacy_off=1
_legacy_dyn=0; [ "${V41_CED_DYNAMIC_SPEC:-0}" = "1" ] && _legacy_dyn=1
_spec_given=0;  [ -n "${SPEC:-}" ] && _spec_given=1
_draft_given=0; [ -n "${DRAFT_GRAPH:-}" ] && _draft_given=1

# ★ 取值合法性：不论走哪条路径都要判。少了这条，`DRAFT_GRAPH=2` 会被下面的
#   `export DRAFT_GRAPH=1` **静默覆盖**成合法值 —— 用户以为传进去了。
#   （2026-09-28 实测：改写成三档后漏了这条，被 tools/selftest_ced_defaults.sh
#     的负控当场抓住；那条负控是 2026-09-27 加的，正好覆盖这个回归类。）
if [ "$_spec_given" = "1" ]; then
  case "${SPEC}" in
    0|1) ;;
    *) echo "[a3-ced][FAIL] SPEC=$SPEC 非法（只能是 0 或 1）。" >&2; exit 2 ;;
  esac
fi
if [ "$_draft_given" = "1" ]; then
  case "${DRAFT_GRAPH}" in
    0|1) ;;
    *) echo "[a3-ced][FAIL] DRAFT_GRAPH=$DRAFT_GRAPH 非法（只能是 0 或 1）。" >&2; exit 2 ;;
  esac
fi

if [ "$_legacy_off" = "1" ] && [ "$_legacy_dyn" = "1" ]; then
  echo "[a3-ced][FAIL] V41_CED_ALLOW_DSPARK=0 与 V41_CED_DYNAMIC_SPEC=1 互斥。" >&2
  echo "  前者要全关 SPEC、后者要开动态 K —— 请只留一个，或改用 SPEC_MODE=off|on|dynamic。" >&2
  exit 2
fi

if [ "$_mode_explicit" = "1" ]; then
  case "$_mode" in
    off|on|dynamic) ;;
    *) echo "[a3-ced][FAIL] SPEC_MODE='$_mode' 非法；只能是 off|on|dynamic。" >&2; exit 2 ;;
  esac
  # 显式 SPEC_MODE 与旧开关/显式 SPEC 矛盾 ⇒ 拒绝（否则"我传了 A 生效的是 B"）
  _bad=""
  [ "$_legacy_off" = "1" ] && [ "$_mode" != "off" ] && _bad="$_bad V41_CED_ALLOW_DSPARK=0"
  [ "$_legacy_dyn" = "1" ] && [ "$_mode" != "dynamic" ] && _bad="$_bad V41_CED_DYNAMIC_SPEC=1"
  if [ "$_mode" = "off" ] && [ "$_spec_given" = "1" ] && [ "${SPEC}" != "0" ]; then
    _bad="$_bad SPEC=$SPEC"
  fi
  if [ "$_mode" != "off" ] && [ "$_spec_given" = "1" ] && [ "${SPEC}" != "1" ]; then
    _bad="$_bad SPEC=$SPEC"
  fi
  if [ "$_mode" = "off" ] && [ "$_draft_given" = "1" ] && [ "${DRAFT_GRAPH}" != "0" ]; then
    _bad="$_bad DRAFT_GRAPH=$DRAFT_GRAPH"
  fi
  if [ "$_mode" != "off" ] && [ "$_draft_given" = "1" ] && [ "${DRAFT_GRAPH}" != "1" ]; then
    _bad="$_bad DRAFT_GRAPH=$DRAFT_GRAPH"
  fi
  if [ -n "$_bad" ]; then
    echo "[a3-ced][FAIL] SPEC_MODE=$_mode 与这些显式设置矛盾：$_bad" >&2
    echo "  ⇒ 拒绝起服：静默择一会让你以为生效的是另一个档（本仓同族事故已多次）。" >&2
    echo "     请去掉矛盾项，或只留 SPEC_MODE=off|on|dynamic。" >&2
    exit 2
  fi
else
  # 未给 SPEC_MODE ⇒ 按旧开关/显式 SPEC 推断；都没有再用角色默认
  if   [ "$_legacy_off" = "1" ]; then _mode=off
  elif [ "$_legacy_dyn" = "1" ]; then
    # ★ legacy 动态档要求 SPEC=1；若同时显式给了 SPEC=0，解析结果会**静默**
    #   把它改回 1 ⇒ "我传了 0 生效的是 1"。⇒ fail-closed。
    if [ "$_spec_given" = "1" ] && [ "${SPEC}" != "1" ]; then
      echo "[a3-ced][FAIL] V41_CED_DYNAMIC_SPEC=1（动态档）要求 SPEC=1，但显式给了 SPEC=$SPEC" >&2
      echo "  ⇒ 拒绝起服（否则你的 SPEC=$SPEC 会被静默改成 1）。" >&2
      echo "     要全关 SPEC 请用 SPEC_MODE=off（或 V41_CED_ALLOW_DSPARK=0）。" >&2
      exit 2
    fi
    _mode=dynamic
  elif [ "$_spec_given" = "1" ]; then
    case "${SPEC}" in
      0) _mode=off ;;
      1) _mode=on ;;
      *) echo "[a3-ced][FAIL] SPEC=$SPEC 非法（只能是 0/1）；或改用 SPEC_MODE=off|on|dynamic。" >&2
         exit 2 ;;
    esac
  elif [ "$role" = "decode" ]; then _mode=on     # D 的交付口径默认 = 全开
  else _mode=off                                  # P 恒 off
  fi
fi

case "$role" in
  prefill)
    # ★ 保留旧的严格性：P 上显式给 SPEC/DRAFT_GRAPH 非零 ⇒ 拒绝，**不静默降级**
    #   （旧代码的 `for setting in SPEC/DRAFT_GRAPH` 门就是这个行为；静默降级会
    #    让人以为 P 在跑 DSpark）。
    if [ "$_spec_given" = "1" ] && [ "${SPEC}" != "0" ]; then
      echo "[a3-ced][FAIL] prefill 角色要求 SPEC=0（当前 SPEC=$SPEC）。" >&2
      echo "  DSpark 需要目标层 37/38/39，P 只跑 0..19，属架构性不可行。" >&2
      exit 2
    fi
    if [ "$_draft_given" = "1" ] && [ "${DRAFT_GRAPH}" != "0" ]; then
      echo "[a3-ced][FAIL] prefill 角色要求 DRAFT_GRAPH=0（当前 DRAFT_GRAPH=$DRAFT_GRAPH）。" >&2
      exit 2
    fi
    if [ "$_mode_explicit" = "1" ] && [ "$_mode" != "off" ]; then
      echo "[a3-ced][FAIL] SPEC_MODE=$_mode 在 prefill 角色上不可用（P 恒 off）。" >&2
      echo "  DSpark 的 aux hidden state 取自目标层 37/38/39，P 只跑 0..19 —— 架构性不可行。" >&2
      echo "  ⇒ P 请用 SPEC_MODE=off（或不给 SPEC_MODE）。" >&2
      exit 2
    fi
    _mode=off
    ;;
esac

case "$_mode" in
  off)
    export SPEC=0 DRAFT_GRAPH=0
    export V41_CED_DYNAMIC_SPEC=0
    echo "[a3-ced][SPEC_MODE] $role：**全关 SPEC**（SPEC=0 DRAFT_GRAPH=0，无草稿、无 --speculative-config）"
    ;;
  on)
    export SPEC=1 DRAFT_GRAPH=1
    export V41_CED_DYNAMIC_SPEC=0
    export SP_TOKENS=${SP_TOKENS:-7}
    # DSpark 的草稿在 SPEC=1 时必须让引擎知道（`model.py` 用同一个 env 做门）
    export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-1}
    echo "[a3-ced][SPEC_MODE] $role：**全开 SPEC**（SPEC=1 DRAFT_GRAPH=1，固定 K=$SP_TOKENS）"
    ;;
  dynamic)
    if [ "$role" != "decode" ]; then
      echo "[a3-ced][FAIL] SPEC_MODE=dynamic 只对 decode 角色有意义（P 恒 SPEC=0）" >&2
      exit 2
    fi
    export SPEC=1 DRAFT_GRAPH=1
    export SP_TOKENS=${SP_TOKENS:-7}
    export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-1}
    export V41_CED_DYNAMIC_SPEC=1
    # ★ 上游 MRV1 在 dynamic SD 时会把 cudagraph_mode 降级为 PIECEWISE，而 V4.1 的
    #   cache 只支持 eager / FULL_DECODE_ONLY ⇒ 不豁免就在模型构造期炸
    #   （serve_a2.sh 的 die 会拦住"只给 SP_SCHEDULE 不给豁免"的写法）。
    #   本档**自动**把它设上：单一开关要能直接用，否则人人踩这个坑。
    #   代价 = 主动放弃一道上游保护 ⇒ 必须用正确性探针验收（见 docs §9）。
    export V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=${V41_CED_DYNAMIC_SPEC_FULL_GRAPHS:-1}
    if [ "$V41_CED_DYNAMIC_SPEC_FULL_GRAPHS" != "1" ]; then
      echo "[a3-ced][FAIL] SPEC_MODE=dynamic 需要 V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1" >&2
      echo "  当前=${V41_CED_DYNAMIC_SPEC_FULL_GRAPHS}。它关掉上游的 PIECEWISE 降级；" >&2
      echo "  设 0 会在模型构造期失败（V4.1 只支持 eager / FULL_DECODE_ONLY）。" >&2
      echo "  ⇒ 去掉这个显式设置，或显式给 1 并接受"须用正确性探针验收"的代价。" >&2
      exit 2
    fi
    echo "[a3-ced][SPEC_MODE] $role：**动态 K**（SPEC=1 DRAFT_GRAPH=1，按请求数切 K）" >&2
    echo "[a3-ced][SPEC_MODE]   ★ 高风险档：豁免了上游降级保护 ⇒ 必须用 144K/1M 正确性探针验收" >&2
    ;;
esac

# [DYNAMIC-SPEC] 动态档的调度表与图模式前提（只有 dynamic 会走到这里）。
#
#   默认表 = `1,1,7;2,8,0`：
#       batch=1  → K=7（低并发走推测，单流吞吐 1.77×）
#       batch≥2  → K=0（关推测，纯自回归；实测并发 2 时自回归已 1.23× 领先）
#   ⚠️ 表是按**请求数**查的，不是"用户并发"。想用别的阈值就显式给
#      `SP_SCHEDULE='1,1,7;2,8,0'`（分号分隔，闭区间）。
#   ⚠️ 这条路的交叉点是在 **2K prompt** 上测出来的（docs §4）。长上下文负载下
#      每步固定开销大得多，交叉点可能移动 —— 上线前应在自己的负载上复测。
#   ⚠️ **未测过并发 ≥2 的吞吐口径**：docs §11.4 只有 ms/step（并发 2 两流
#      32.2/32.5），"decode tok/s"那一列是空的。用它做容量规划前先补测。
if [ "${V41_CED_DYNAMIC_SPEC:-0}" = "1" ]; then
  export SP_SCHEDULE=${SP_SCHEDULE:-1,1,7;2,8,0}
  if [ "${CED_DIAGNOSTIC_EAGER:-0}" = "1" ]; then
    echo "[a3-ced][FAIL] SPEC_MODE=dynamic 与 CED_DIAGNOSTIC_EAGER=1 互斥（eager 没有图可切）" >&2
    exit 2
  fi
  export CED_EXPERIMENTAL_GRAPH=${CED_EXPERIMENTAL_GRAPH:-1}
  export V41_CED_GRAPH_PROMPT_TAIL_EAGER=${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-1}
  echo "[a3-ced] D 侧 dynamic spec：SP_SCHEDULE='$SP_SCHEDULE'（按请求数切 K）"
fi

# ★ 非动态档却带着 SP_SCHEDULE ⇒ serve_a2.sh 的 `if [ -n "$SP_SCHEDULE" ]`
#   会把动态路径整条拉起来（打补丁、按并发切 K），而你选的是 on/off。
#   这是"我选的是 A、生效的是 B"的典型 ⇒ fail-closed。
if [ "$_mode" != "dynamic" ] && [ -n "${SP_SCHEDULE:-}" ]; then
  echo "[a3-ced][FAIL] SPEC_MODE=$_mode 但设了 SP_SCHEDULE='$SP_SCHEDULE'。" >&2
  echo "  只要 SP_SCHEDULE 非空，serve_a2.sh 就会拉起动态 K 的整条路径（含打补丁），" >&2
  echo "  与你选的档位矛盾。⇒ 要动态 K 请用 SPEC_MODE=dynamic；否则清掉 SP_SCHEDULE。" >&2
  exit 2
fi

# [SELFTEST-HOOK] 只解析并打印 SPEC_MODE 的结果后退出 —— 供 selfcheck 的
#   正控/负控矩阵使用（生产不会设这个变量）。
# 取值：1 = 本层解析后退出；2 = 同上（由 deploy launcher 透传过来时用，语义相同）。
# 判据统一成"非空且非 0 即生效"，避免"传 2 却发现钩子只认 1"这种自己给自己挖的坑。
if [ -n "${V41_SPEC_MODE_CHECK_ONLY:-}" ] && [ "${V41_SPEC_MODE_CHECK_ONLY}" != "0" ]; then
  echo "SPEC_MODE_RESOLVED mode=$_mode role=$role spec=${SPEC:-} draft=${DRAFT_GRAPH:-}"\
" dyn=${V41_CED_DYNAMIC_SPEC:-} full_graphs=${V41_CED_DYNAMIC_SPEC_FULL_GRAPHS:-}"\
" schedule=${SP_SCHEDULE:-}"
  exit 0
fi

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
