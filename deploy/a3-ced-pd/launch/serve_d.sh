#!/usr/bin/env bash
# 起 CED 的 D（decode）：chip 8–15，128-token 有界重放 + 全 40 层 + DSpark。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
. "$HERE/_common.sh"

export RUN_ID=${RUN_ID:-ced_d_$(date +%Y%m%d_%H%M%S)}
export NAME=${NAME:-dsv41-ced-d-$RUN_ID}
export PORT=$PD_DECODE_PORT
export KV_PORT=$PD_DECODE_KV_PORT
export DEVS=$PD_DECODE_DEVS

# ---- ★ DSpark（D 侧）：三档选择，见 scripts/serve_a3_ced_pd.sh 的 [SPEC_MODE] ----
#
#   SPEC_MODE=on       ★ 默认（= 交付口径）：全开 SPEC，固定 K=SP_TOKENS
#   SPEC_MODE=off      全关 SPEC（纯自回归）
#   SPEC_MODE=dynamic  动态 K（按请求数切 1 ↔ K=0；豁免上游降级门，高风险档）
#
# ★ 必须**只在没给 SPEC_MODE 时**才默认 SPEC/DRAFT_GRAPH ——
#   否则 `SPEC_MODE=off` 会被这里的 `SPEC=1` 覆盖，两处默认不一致 ⇒
#   要么起不来（新的矛盾门），要么"选了 off 跑的是 on"（本仓同族事故）。
#   取值合法性与矛盾组合由 scripts/serve_a3_ced_pd.sh fail-closed 统一裁定。
if [ -z "${SPEC_MODE:-}" ]; then
  export SPEC=${SPEC:-1}
  export DRAFT_GRAPH=${DRAFT_GRAPH:-1}
  # 显式放行 D 侧开 SPEC（默认拒绝；P/D 是两个独立引擎实例，per-instance 生效）
  export V41_CED_ALLOW_DSPARK=${V41_CED_ALLOW_DSPARK:-1}
fi
export SPEC_MODE=${SPEC_MODE:-}
export SP_TOKENS=${SP_TOKENS:-7}

# ---- ★ 图模式的两个前提（launcher 已 fail-closed，这里显式写出来）----
export CED_EXPERIMENTAL_GRAPH=${CED_EXPERIMENTAL_GRAPH:-1}
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-1}
export V41_CED_SWA_CLIP=${V41_CED_SWA_CLIP:-1}

# ---- 本形态实测有效的两个（见 docs/CED-PD-STATIC-KERNEL-20260926.md）----
export STATIC_KERNEL=${STATIC_KERNEL:-1}     # −4.4 ms/step（−9.6%）
export V41_SLOT_MAP_FUSED=${V41_SLOT_MAP_FUSED:-on}

say "D  role=decode  devs='$DEVS' port=$PORT kv_port=$KV_PORT patch_mode=$PATCH_MODE"
say "   spec_mode=${SPEC_MODE:-<未给:按旧开关/SPEC 推断，默认 on>} spec=${SPEC:-} sp=$SP_TOKENS draft_graph=${DRAFT_GRAPH:-} static_kernel=$STATIC_KERNEL"

# [SELFTEST-HOOK] 只打印本 launcher 定下来的 SPEC 相关 env 后退出 —— 供
#   tools/selftest_spec_mode.sh 验证**交付面与角色脚本一致**（生产不会设这个变量）。
#   为什么需要：launcher 里一句 `export SPEC=${SPEC:-1}` 就能把 `SPEC_MODE=off`
#   顶掉，而那种不一致在真机上表现为"我选了 off，跑的是 on"。
if [ "${V41_SPEC_MODE_CHECK_ONLY:-0}" != "0" ] && [ -n "${V41_SPEC_MODE_CHECK_ONLY:-}" ]; then
  echo "SERVE_D_RESOLVED spec_mode=${SPEC_MODE:-} spec=${SPEC:-} draft=${DRAFT_GRAPH:-} allow_dspark=${V41_CED_ALLOW_DSPARK:-}"
  # =1：只报 launcher 这一层定的 env。
  # =2：**继续穿透**到角色脚本 ⇒ 验的是"用户敲的命令链最终解析出什么"
  #     （交付面 + 角色脚本两个默认值的交互，正是最容易两处分叉的地方）。
  if [ "${V41_SPEC_MODE_CHECK_ONLY}" = "2" ]; then
    cd "$PKG"
    exec bash scripts/serve_a3_ced_pd.sh decode
  fi
  exit 0
fi

cd "$PKG"
exec bash scripts/serve_a3_ced_pd.sh decode
