#!/usr/bin/env bash
# =============================================================================
# selftest_kv32_scope.sh —— KV32 池守卫「作用域」离线自检（不占卡、不起容器）
#
# 被测对象：`scripts/serve_a2.sh` 的 [KV32-POOL-GUARD] 解析块 + 自检钩子。
#
# 背景（docs/KV32-POOL-GUARD-SCOPE-20260929.md）：守卫原先在**所有形态**下生效，
#   且"未设置 KV_CACHE_MEMORY_BYTES ⇒ 直接 pin 到 29076 块" ⇒ 非 CED 部署
#   （A2 单实例、OffloadingConnector、验证入口）被强制 pin，vLLM 因此**跳过
#   自动显存 profiling**、`GPU_UTIL` 对 KV 池不再生效。本自检锁住修正后的语义。
#
# 为什么必须有它：各档之间**只差几个 env**，而"我选了 A、生效的是 B"是本仓反复
#   栽的一类事故。这些错 `bash -n` 一个都抓不到，真机验证一次要起 8 卡（~20 分钟）。
#
# 判据绑在**脚本自己解析出来的三元组**上（`V41_KV32_GUARD_CHECK_ONLY=1` 钩子，
#   生产路径不会设它），不绑"我传了哪个 env"。
#
# 为什么需要 docker 桩：钩子在守卫尾、**所有 docker 动作之前**，但脚本在到达
#   守卫前会先 `docker info` + `docker image inspect`（离线机器上没有镜像）。
#   用 `PATH` 前置一个只会 `exit 0` 的桩，脚本就能一路走到钩子并退出；
#   全程不会真起容器。`DRY_RUN=1` 那条路走不到守卫（它的出口在守卫之前），
#   所以这里不用 DRY_RUN。
#
# 跑法：`bash tools/selftest_kv32_scope.sh`
#   SERVE_A2_UNDER_TEST=<别的副本> bash tools/selftest_kv32_scope.sh   # 手工负控
#   KV32_SKIP_NEG=1 bash tools/selftest_kv32_scope.sh                  # 跳过内置负控
#
# 退出码：0 = 全过；1 = 有失败。
# =============================================================================
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
SCRIPT=${SERVE_A2_UNDER_TEST:-scripts/serve_a2.sh}
[ -f "$SCRIPT" ] || { echo "缺 $SCRIPT"; exit 2; }

CAP=15728022528          # 29076 × 540928
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/model" "$TMP/bin"
cat > "$TMP/bin/docker" <<'STUB'
#!/bin/sh
# 只用于离线自检：任何调用都成功且无副作用。
exit 0
STUB
chmod +x "$TMP/bin/docker"

n_ok=0
n_bad=0
EXTRA_ENV=()

# run_case <expect> <e_scope> <e_pin> <e_enforce> <e_pinned> <e_bytes> <label>
#   expect = ok|fail；e_* = 期望值（ok 时校验；bytes 用 <unset> 表示未设置）
#   额外 env 放在全局数组 EXTRA_ENV（值里可以带空格，例如连接器 JSON）
run_case() {
  local expect=$1 e_scope=$2 e_pin=$3 e_enforce=$4 e_pinned=$5 e_bytes=$6
  local label=${7:-}
  local out rc line
  out=$(env PATH="$TMP/bin:$PATH" MODEL="$TMP/model" V41_KV32_GUARD_CHECK_ONLY=1 \
        "${EXTRA_ENV[@]}" bash "$SCRIPT" 2>&1); rc=$?
  line=$(printf '%s\n' "$out" | grep -m1 '^KV32_RESOLVED' || true)

  _fail() {
    echo "  ✗ [$label] $1"
    printf '      env=%s rc=%s\n' "$(printf '%q ' "${EXTRA_ENV[@]}")" "$rc"
    printf '%s\n' "$out" | grep -E 'KV32_RESOLVED|\[KV32\]' | sed 's/^/      | /' | head -4
    n_bad=$((n_bad + 1))
  }

  if [ "$expect" = "fail" ]; then
    if [ "$rc" = "0" ] && printf '%s\n' "$line" | grep -q "bytes=$e_bytes\b"; then
      _fail "回归未被抓住（结果与期望一致）"
      return
    fi
    echo "  ✓ [$label] 按预期未通过（负控有效）"
    n_ok=$((n_ok + 1))
    return
  fi

  [ "$rc" = "0" ] || { _fail "应该放行（rc=0）却被拒"; return; }
  [ -n "$line" ] || { _fail "没有打印 KV32_RESOLVED（钩子没生效/位置不可达）"; return; }
  local bad=""
  case "$line" in *" scope=$e_scope"*) ;; *) bad="$bad scope(期望 $e_scope)";; esac
  case "$line" in *" pin=$e_pin"*) ;; *) bad="$bad pin(期望 $e_pin)";; esac
  case "$line" in *" enforce=$e_enforce"*) ;; *) bad="$bad enforce(期望 $e_enforce)";; esac
  case "$line" in *" pinned=$e_pinned"*) ;; *) bad="$bad pinned(期望 $e_pinned)";; esac
  [ -n "$e_bytes" ] && case "$line" in *" bytes=$e_bytes"*) ;; *) bad="$bad bytes(期望 $e_bytes)";; esac
  [ -z "$bad" ] || { _fail "解析结果不符：$bad"; return; }
  echo "  ✓ [$label] $line"
  n_ok=$((n_ok + 1))
}

echo "== 正控：作用域解析（被检脚本 $SCRIPT）=="
EXTRA_ENV=()
run_case ok auto 0 1 0 '<unset>' \
  'case1 无任何标记 ⇒ 非 CED 不 pin（回到自动 profiling）'

EXTRA_ENV=(V41_CED_ROLE=decode)
run_case ok auto 1 1 1 "$CAP" \
  'case2 CED 角色 ⇒ pin 到上界'

EXTRA_ENV=(KV_ARGS_EXTRA='--kv-transfer-config {"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer"}')
run_case ok auto 1 1 1 "$CAP" \
  'case3 普通 PD（Mooncake 连接器）⇒ pin'

EXTRA_ENV=(V41_KV32_POOL_GUARD=on)
run_case ok on 1 1 1 "$CAP" \
  'case4 强制 on ⇒ pin'

EXTRA_ENV=(V41_KV32_POOL_GUARD=off V41_CED_ROLE=decode)
run_case ok off 0 0 0 '<unset>' \
  'case5 显式 off + CED 角色 ⇒ 完全不干预'

EXTRA_ENV=(KV_CACHE_MEMORY_BYTES=15700000000)
run_case ok auto 0 1 0 15700000000 \
  'case6 显式小于上界 ⇒ 原样保留'

EXTRA_ENV=(KV_CACHE_MEMORY_BYTES=16500000000 V41_CED_ROLE=decode)
run_case ok auto 1 1 0 "$CAP" \
  'case7 显式超界 + CED ⇒ 钳到上界（pin 策略仍为 1，但不是靠 pin 得到的）'

EXTRA_ENV=(KV_CACHE_MEMORY_BYTES=16500000000)
run_case ok auto 0 1 0 "$CAP" \
  'case8 显式超界 + 非 CED ⇒ 同样钳位（模型级风险，与形态无关）'

EXTRA_ENV=(V41_KV32_POOL_GUARD=bogus)
run_case ok auto 0 1 0 '<unset>' \
  'case9 非法档位 ⇒ 告警并按 auto'

EXTRA_ENV=(V41_CED_ALLOW_32BIT_OVERFLOW=1 V41_CED_ROLE=decode)
run_case ok auto 0 0 0 '<unset>' \
  'case10 旧逃生口 ⇒ 等价 off（不改 scope，只关 enforce）'

EXTRA_ENV=()

echo "== 正控：起服后池上界复核（从被检脚本抽取 kv32_pool_blocks_from_log，喂合成日志）=="
# 按两行标记从**被检脚本本体**抽函数 —— 测的是真 artifact，不是抄一份。
_fn=$(sed -n '/^kv32_pool_blocks_from_log() {/,/^# --- \[KV32\] 辅助函数结束/p' "$SCRIPT" | sed '$d')
if [ -z "$_fn" ]; then
  echo "  ✗ 抽不到 kv32_pool_blocks_from_log（标记行被改动？）"
  n_bad=$((n_bad + 1))
else
  eval "$_fn"
  BPB=540928
  CAP_BLOCKS=$(( CAP / BPB ))
  mk() {  # mk <gib>...  → 合成日志，字段布局与真日志逐字一致（取 $(NF-1) 才是数字）
    : > "$TMP/log.txt"
    local i=0 g
    for g in "$@"; do
      printf '(Worker_TP%d_EP%d pid=100%d) INFO 09-23 23:09:02 [worker.py:628] Available KV cache memory: %s GiB\n' \
        "$i" "$i" "$i" "$g" >> "$TMP/log.txt"
      i=$((i + 1))
    done
  }
  _of() { mk "$@"; kv32_pool_blocks_from_log "$TMP/log.txt" "$BPB"; }

  # (1) 语义：A2 历史 profiling 值必须**不判越界**（否则会误拦正常运行）
  _b=$(_of 14.40)
  if [ "${_b:-0}" -le "$CAP_BLOCKS" ]; then
    echo "  ✓ [a2-hist-14.40 不判越界] ${_b} 块 ≤ 上界 ${CAP_BLOCKS}"; n_ok=$((n_ok + 1))
  else
    echo "  ✗ [a2-hist-14.40 不判越界] ${_b} 块 > 上界 ${CAP_BLOCKS}（会误拦正常运行）"; n_bad=$((n_bad + 1))
  fi

  # (2) 语义：CED P 侧越界现场必须**被判越界**（这条是本节存在的理由）
  _b=$(_of 15.16)
  if [ "${_b:-0}" -gt "$CAP_BLOCKS" ]; then
    echo "  ✓ [pd-over-15.16 判越界] ${_b} 块 > 上界 ${CAP_BLOCKS}"; n_ok=$((n_ok + 1))
  else
    echo "  ✗ [pd-over-15.16 判越界] ${_b} 块 ≤ 上界 ${CAP_BLOCKS}（漏报＝静默空答风险）"; n_bad=$((n_bad + 1))
  fi

  # (3) 语义：多 rank 必须取 **min**（vLLM 最终 num_blocks 取各 rank 最小值）
  _bmin=$(_of 13.59); _bmix=$(_of 15.16 13.59 14.00)
  if [ "$_bmix" = "$_bmin" ]; then
    echo "  ✓ [multi-rank 取 min] 混排 ${_bmix} == 仅 13.59 的 ${_bmin}"; n_ok=$((n_ok + 1))
  else
    echo "  ✗ [multi-rank 取 min] 混排 ${_bmix} != 仅 13.59 的 ${_bmin}（取成了 max？）"; n_bad=$((n_bad + 1))
  fi

  # (4) 回归锁：一个具体数值，防公式/除数被无声改动
  if [ "$(_of 14.40)" = "28583" ]; then
    echo "  ✓ [数值锁 14.40 GiB → 28583 块]"; n_ok=$((n_ok + 1))
  else
    echo "  ✗ [数值锁 14.40 GiB → 28583 块] 实得 $(_of 14.40)"; n_bad=$((n_bad + 1))
  fi

  # (5) pinned 路径没有该行 ⇒ 必须跳过（不能把"没有数据"当成"安全"）
  mk
  if kv32_pool_blocks_from_log "$TMP/log.txt" "$BPB" >/dev/null 2>&1; then
    echo "  ✗ [无数据 ⇒ 跳过] 却算出了块数（会把 pinned 路径误判）"; n_bad=$((n_bad + 1))
  else
    echo "  ✓ [无数据 ⇒ 跳过]"; n_ok=$((n_ok + 1))
  fi
fi

if [ "${KV32_SKIP_NEG:-0}" != "1" ]; then
  echo
  echo "== 负控：把 auto 默认改回 on（复现「所有形态都 pin」的回归）=="
  NEG="$TMP/serve_a2_regressed.sh"
  sed 's/^_kv32_scope=\${V41_KV32_POOL_GUARD:-auto}$/_kv32_scope=${V41_KV32_POOL_GUARD:-on}/' \
    "$SCRIPT" > "$NEG"
  if cmp -s "$NEG" "$SCRIPT"; then
    echo "  ✗ 负控夹具没造出来（sed 没命中 auto 默认行）—— 自检自身失效"
    n_bad=$((n_bad + 1))
  else
    _saved=${SERVE_A2_UNDER_TEST:-}
    SERVE_A2_UNDER_TEST="$NEG"
    EXTRA_ENV=()
    run_case fail auto 0 1 0 '<unset>' \
      'negative 回归版（默认 on）在 case1 上必须不通过'
    SERVE_A2_UNDER_TEST=$_saved
  fi
fi

echo
echo "结果：$n_ok 通过，$n_bad 失败"
[ "$n_bad" = "0" ] || exit 1
exit 0
