#!/usr/bin/env bash
# =============================================================================
# selftest_spec_mode.sh —— A3 CED-PD 的 **SPEC_MODE 三档选择**离线自检
#
# 被测对象：`scripts/serve_a3_ced_pd.sh` 的 [SPEC_MODE] 解析块。
#
# 三个档位（D 侧）：
#   on       全开 SPEC（固定 K=SP_TOKENS）    SPEC=1 DRAFT_GRAPH=1     ★默认
#   off      全关 SPEC（纯自回归）            SPEC=0 DRAFT_GRAPH=0
#   dynamic  动态 K（按请求数切 1 ↔ K=0）     SPEC=1 DRAFT_GRAPH=1 + SP_SCHEDULE
#                                            + 上游降级门豁免
# P 侧恒 off（架构性：DSpark 取目标层 37/38/39，P 在 layer 20 break）。
#
# 为什么必须有这个自检：三档之间**只差几个 env**，而"我选了 A、生效的是 B"
#   是本仓反复栽的一类事故（默认值两处不一致、转发漏一层、补丁没挂上…）。
#   这些错 `bash -n` 一个都抓不到，真机验证一次要起两个实例（~20 分钟）。
#
# 判据绑在**脚本自己解析出来的三元组**上（靠 `V41_SPEC_MODE_CHECK_ONLY=1`
#   钩子，在生产路径上不会设），不绑"我传了哪个 env"。
#
# 跑法：`bash tools/selftest_spec_mode.sh`
# =============================================================================
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
# 允许指向别的副本 ⇒ 用同一套判据做**负控**（对"拆掉门"的版本必须失败）：
#   SERVE_CED_PD_UNDER_TEST=/tmp/serve_ced_pd_noGate.sh bash tools/selftest_spec_mode.sh
SCRIPT=${SERVE_CED_PD_UNDER_TEST:-scripts/serve_a3_ced_pd.sh}
[ -f "$SCRIPT" ] || { echo "缺 $SCRIPT"; exit 2; }

n_ok=0
n_bad=0

# run_case <expect> <role> <e_mode> <e_spec> <e_draft> <e_dyn> <e_full> <e_sched> <envs> <label>
#   expect = ok|fail；e_* = 期望值（ok 时校验，空串=不校验）
run_case() {
  local expect=$1 role=$2 e_mode=$3 e_spec=$4 e_draft=$5 e_dyn=$6 e_full=$7 e_sched=$8
  local envs=${9:-} label=${10:-}
  local out rc line
  # shellcheck disable=SC2086  # envs 刻意按空格拆分（每个都是一个 VAR=VAL）
  out=$(env V41_SPEC_MODE_CHECK_ONLY=1 $envs bash "$SCRIPT" "$role" 2>&1); rc=$?
  line=$(printf '%s\n' "$out" | grep -m1 '^SPEC_MODE_RESOLVED' || true)

  _fail() {
    echo "  ✗ [$label] $1"
    echo "      env='$envs' role=$role rc=$rc"
    printf '%s\n' "$out" | grep -E 'FAIL|SPEC_MODE_RESOLVED' | sed 's/^/      | /' | head -4
    n_bad=$((n_bad + 1))
  }

  if [ "$expect" = "fail" ]; then
    if [ "$rc" = "0" ]; then
      _fail "应该拒绝（rc!=0）却放行了"
      return
    fi
    if ! printf '%s\n' "$out" | grep -q '\[a3-ced\]\[FAIL\]'; then
      _fail "拒绝了但没有 [a3-ced][FAIL] 说明（会让人看不出原因）"
      return
    fi
    echo "  ✓ [$label] 按预期拒绝（rc=$rc）"
    n_ok=$((n_ok + 1))
    return
  fi

  if [ "$rc" != "0" ]; then
    _fail "应该放行（rc=0）却被拒"
    return
  fi
  if [ -z "$line" ]; then
    _fail "没有打印 SPEC_MODE_RESOLVED（钩子没生效？）"
    return
  fi
  local bad=""
  _chk() { # $1=字段名 $2=期望（空=不校验；"-"=必须为空）
    local k=$1 want=$2 got
    [ -z "$want" ] && return 0
    got=$(printf '%s\n' "$line" | sed -n "s/.* $k=\([^ ]*\).*/\1/p")
    if [ "$want" = "-" ]; then
      [ -z "$got" ] || bad="$bad ${k}=${got}(期望为空/未设置)"
      return 0
    fi
    [ "$got" = "$want" ] || bad="$bad ${k}=${got}(期望${want})"
  }
  _chk mode "$e_mode"; _chk spec "$e_spec"; _chk draft "$e_draft"
  _chk dyn "$e_dyn";  _chk full_graphs "$e_full"; _chk schedule "$e_sched"
  if [ -n "$bad" ]; then
    _fail "解析结果不符：$bad"
    return
  fi
  echo "  ✓ [$label] $line"
  n_ok=$((n_ok + 1))
}

echo "== D 侧（decode）：三档正控 =="
run_case ok decode on      1 1 0 ""  ""                ""                      "默认 = 全开 SPEC"
run_case ok decode on      1 1 0 ""  ""                "SPEC_MODE=on"          "SPEC_MODE=on"
run_case ok decode off     0 0 0 ""  ""                "SPEC_MODE=off"         "SPEC_MODE=off"
run_case ok decode dynamic 1 1 1 1 "1,1,7;2,8,0"      "SPEC_MODE=dynamic"     "SPEC_MODE=dynamic"
run_case ok decode dynamic 1 1 1 1 "1,1,3"            "SPEC_MODE=dynamic SP_SCHEDULE=1,1,3" "dynamic 自定义表"

echo "== D 侧：非法值 / 矛盾组合必须 fail-closed =="
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=bogus"                        "非法档位名"
run_case ok   decode on  1 1 0 ""  "" "SPEC_MODE="                        "空串档位 = 未给（落默认 on）"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=off SPEC=1"                   "off 与 SPEC=1 矛盾"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=on SPEC=0"                    "on 与 SPEC=0 矛盾"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=off DRAFT_GRAPH=1"            "off 与 DRAFT_GRAPH=1 矛盾"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=on DRAFT_GRAPH=0"             "on 与 DRAFT_GRAPH=0 矛盾"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=on V41_CED_DYNAMIC_SPEC=1"    "on 与 legacy 动态开关矛盾"
run_case ok   decode off 0 0 0 ""  "" "SPEC_MODE=off V41_CED_ALLOW_DSPARK=1" "legacy 变量的**默认值**不算矛盾（只有 =0 才算）"
run_case fail decode "" "" "" "" "" "" "SPEC=2"                                 "SPEC 非法值"
run_case fail decode "" "" "" "" "" "" "DRAFT_GRAPH=2"                          "DRAFT_GRAPH 非法值（漏判会被静默覆盖成 1）"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=on DRAFT_GRAPH=2"             "on + DRAFT_GRAPH=2"
run_case fail decode "" "" "" "" "" "" "V41_CED_ALLOW_DSPARK=0 V41_CED_DYNAMIC_SPEC=1" "两个 legacy 开关互斥"
run_case fail decode "" "" "" "" "" "" "V41_CED_DYNAMIC_SPEC=1 SPEC=0"          "legacy 动态档 + 显式 SPEC=0（会静默覆盖）"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=on SP_SCHEDULE=1,1,7;2,8,0"   "非动态档却带 SP_SCHEDULE"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=dynamic CED_DIAGNOSTIC_EAGER=1" "动态档 + eager 互斥"
run_case fail decode "" "" "" "" "" "" "SPEC_MODE=dynamic V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=0" "动态档但显式关掉降级豁免"

echo "== D 侧：legacy 兼容（不改变历史行为）=="
run_case ok decode off     0 0 0 ""  "" "V41_CED_ALLOW_DSPARK=0"    "legacy: ALLOW_DSPARK=0 ⇒ off"
run_case ok decode off     0 0 0 ""  "" "SPEC=0"                   "legacy: SPEC=0 ⇒ off"
run_case ok decode on      1 1 0 ""  "" "SPEC=1"                   "legacy: SPEC=1 ⇒ on"
run_case ok decode dynamic 1 1 1 1 "1,1,7;2,8,0" "V41_CED_DYNAMIC_SPEC=1" "legacy: DYNAMIC_SPEC=1 ⇒ dynamic（自动补豁免）"

echo "== P 侧（prefill）：恒 off，且显式非零要拒绝 =="
run_case ok prefill off    0 0 0 "" "" ""              "默认 ⇒ off"
run_case ok prefill off    0 0 0 "" "" "SPEC_MODE=off" "SPEC_MODE=off"
run_case ok prefill off    0 0 0 "" "" "SPEC=0 DRAFT_GRAPH=0" "显式 0（= serve_p.sh 的写法）"
run_case fail prefill "" "" "" "" "" "" "SPEC_MODE=on"      "P 上要求 on ⇒ 拒绝"
run_case fail prefill "" "" "" "" "" "" "SPEC_MODE=dynamic" "P 上要求 dynamic ⇒ 拒绝"
run_case fail prefill "" "" "" "" "" "" "SPEC=1"            "P 上显式 SPEC=1 ⇒ 拒绝（保留旧严格性）"
run_case fail prefill "" "" "" "" "" "" "DRAFT_GRAPH=1"     "P 上显式 DRAFT_GRAPH=1 ⇒ 拒绝"

echo "== 交付面一致性：deploy launcher 不得顶掉 SPEC_MODE =="
# launcher 里一句 `export SPEC=${SPEC:-1}` 就能把 SPEC_MODE=off 顶掉。
# 判据：给了 SPEC_MODE 时，launcher 必须**不设** SPEC/DRAFT_GRAPH（留给角色脚本裁定）。
LAUNCHER=deploy/a3-ced-pd/launch/serve_d.sh
if [ -f "$LAUNCHER" ]; then
  lcase() { # <expect_spec_mode> <expect_spec> <expect_draft> <envs> <label>
    local w_mode=$1 w_spec=$2 w_draft=$3 envs=${4:-} label=${5:-}
    local out rc line bad=""
    # shellcheck disable=SC2086
    out=$(env V41_SPEC_MODE_CHECK_ONLY=1 MODEL=/nonexistent/check-only $envs \
          bash "$LAUNCHER" 2>&1); rc=$?
    line=$(printf '%s\n' "$out" | grep -m1 '^SERVE_D_RESOLVED' || true)
    if [ "$rc" != "0" ] || [ -z "$line" ]; then
      echo "  ✗ [$label] launcher 没跑通（rc=$rc）"
      printf '%s\n' "$out" | tail -3 | sed 's/^/      | /'
      n_bad=$((n_bad + 1)); return
    fi
    _l() { local k=$1 want=$2 got; got=$(printf '%s\n' "$line" | sed -n "s/.* $k=\([^ ]*\).*/\1/p")
           if [ "$want" = "-" ]; then [ -z "$got" ] || bad="$bad ${k}=${got}(期望未设置)"
           else [ "$got" = "$want" ] || bad="$bad ${k}=${got}(期望${want})"; fi; }
    _l spec_mode "$w_mode"; _l spec "$w_spec"; _l draft "$w_draft"
    if [ -n "$bad" ]; then
      echo "  ✗ [$label] $bad"; echo "      | $line"; n_bad=$((n_bad + 1)); return
    fi
    echo "  ✓ [$label] $line"; n_ok=$((n_ok + 1))
  }
  lcase ""         1 1 ""                      "launcher 默认：仍设 SPEC=1 DRAFT_GRAPH=1（= 交付口径）"
  lcase off        - - "SPEC_MODE=off"         "launcher + SPEC_MODE=off：**不设** SPEC/DRAFT_GRAPH"
  lcase on         - - "SPEC_MODE=on"          "launcher + SPEC_MODE=on：不设（同样留给角色脚本）"
  lcase dynamic    - - "SPEC_MODE=dynamic"     "launcher + SPEC_MODE=dynamic：不设"
  lcase ""         0 0 "SPEC=0 DRAFT_GRAPH=0"  "launcher legacy：显式 0 原样透传"
else
  echo "  ✗ 缺 $LAUNCHER"; n_bad=$((n_bad + 1))
fi

echo "== 全链路：用户敲的命令 → 最终解析（launcher + 角色脚本两层默认值交互）=="
# 两层各有一个默认值，最容易在这里分叉（launcher 的 SPEC=1 顶掉 SPEC_MODE=off，
# 或角色脚本的默认把 launcher 的显式值吃掉）。判据用钩子 =2 穿透到角色脚本。
if [ -f "$LAUNCHER" ]; then
  ccase() { # <expect_mode> <expect_spec> <expect_draft> <expect_dyn> <expect_full> <envs> <label>
    local w_mode=$1 w_spec=$2 w_draft=$3 w_dyn=$4 w_full=$5 envs=${6:-} label=${7:-}
    local out rc line bad=""
    # shellcheck disable=SC2086
    out=$(env V41_SPEC_MODE_CHECK_ONLY=2 MODEL=/nonexistent $envs \
          bash "$LAUNCHER" 2>&1); rc=$?
    line=$(printf '%s\n' "$out" | grep -m1 '^SPEC_MODE_RESOLVED' || true)
    if [ -z "$line" ]; then
      echo "  ✗ [$label] 全链路没走到角色脚本（rc=$rc）"
      printf '%s\n' "$out" | grep -E 'FAIL|RESOLVED' | head -3 | sed 's/^/      | /'
      n_bad=$((n_bad + 1)); return
    fi
    _c() { local k=$1 want=$2 got
           got=$(printf '%s\n' "$line" | sed -n "s/.* $k=\([^ ]*\).*/\1/p")
           [ "$got" = "$want" ] || bad="$bad ${k}=${got}(期望${want})"; }
    _c mode "$w_mode"; _c spec "$w_spec"; _c draft "$w_draft"
    _c dyn "$w_dyn"; _c full_graphs "$w_full"
    if [ -n "$bad" ]; then
      echo "  ✗ [$label] $bad"; echo "      | $line"; n_bad=$((n_bad + 1)); return
    fi
    echo "  ✓ [$label] $line"; n_ok=$((n_ok + 1))
  }
  # ★ 参数是 7 个：mode spec draft dyn full envs label。
  #   第一版这里多写了一个空参数 ⇒ envs 收到空串 ⇒ `on` 那条**假通过**
  #   （跑到默认档、恰好与期望一致），而 `off` 那条才把它暴露出来。
  ccase on      1 1 0 "" "SPEC_MODE=on"      "全链路 on"
  ccase off     0 0 0 "" "SPEC_MODE=off"     "全链路 off"
  ccase dynamic 1 1 1 1 "SPEC_MODE=dynamic"  "全链路 dynamic"
  ccase on      1 1 0 "" ""                  "全链路默认（不给 SPEC_MODE）"
else
  echo "  ✗ 缺 $LAUNCHER"; n_bad=$((n_bad + 1))
fi

echo
echo "合计 $((n_ok + n_bad)) 项，失败 $n_bad 项"
if [ "$n_bad" != "0" ]; then
  exit 1
fi
echo "全部通过 ✅"
