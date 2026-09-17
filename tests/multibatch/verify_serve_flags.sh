#!/usr/bin/env bash
# =============================================================================
# verify_serve_flags.sh -- 启动器开关矩阵烟测（12 组合，**不碰 docker / 不占卡**）
#
#   原理：`DRY_RUN=1` 让 scripts/serve_a2.sh 真正展开**全部**变量后打印并退出。
#   这能抓到 `bash -n` 抓不到、只有展开时才暴露的错（如 `set -u` 下的变量顺序 bug：
#   A3-node1 上 `MOE_ZERO: unbound variable` 就是这样暴露的）。
#
# 用法：bash tests/multibatch/verify_serve_flags.sh
# 判据：最后一行 `[flags] pass=12 fail=0`；任一组合 FAIL 会打印 unbound/error/Traceback 前 3 行。
#
# ⚠️ 它**不会**验证开关在 NPU 上是否真的生效（那要起服）；它只保证"启动器不会因参数组合直接
#    崩在解析阶段"。真正的生效检查见 EXPECTED_PERF.md / REPRO.md 的"判据"列。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
SERVE="$PKG/scripts/serve_a2.sh"

pass=0; fail=0
run() {
  local combo="$1" out rc
  out=$(env DRY_RUN=1 MODEL=${MODEL:-/nonexistent/DRY_RUN_MODEL} $combo bash "$SERVE" 2>&1); rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "\[a2-dry\] OK"; then
    pass=$((pass+1)); printf '[ok  ] %s\n' "${combo:-<默认>}"
  else
    fail=$((fail+1)); printf '[FAIL] %s (rc=%d)\n' "${combo:-<默认>}" "$rc"
    printf '%s\n' "$out" | grep -iE "unbound|error|Traceback|FAIL" | head -3 | sed 's/^/        /'
  fi
}

# ---- 12 组合：① 默认  ②③④ MoE 三个负结果臂  ⑤ DSpark 图  ⑥⑦ 生产/压力口径 ----
#      ⑧ SP_TOKENS 变体  ⑨ dummy  ⑩ 全关基线  ⑪ HCCL 诊断  ⑫ 组合极限 ----
run ""
run "MOE_ZERO=1"
run "MOE_NF=1"
run "MOE_NF=1 MOE_ZERO=1"
run "DRAFT_GRAPH=1"
run "MAX_SEQS=32 PREFIX=1"
run "MAX_SEQS=32 PREFIX=0"
run "SP_TOKENS=7 MAX_SEQS=8"
run "LOAD_FORMAT=dummy MAX_SEQS=1"
run "QLI_NOCAND=0 ROPE_IDXSEL=0 MOE_MASK=0 O_PROJ_2D=0 ENGRAM_JIT=0 LOCAL_OWNER=on"
run "HCCL_DET=strict"
run "HCCL_DET=true DRAFT_GRAPH=1 MOE_NF=1 MOE_ZERO=1 PATCH_MODE=mount MAX_SEQS=4 PREFIX=1"
echo "[flags] pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
