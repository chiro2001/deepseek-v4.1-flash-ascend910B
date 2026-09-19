#!/usr/bin/env bash
# 验证：把 patch 系列 am 到「干净 base checkout」后，产出文件与线上工作树逐字节一致。
# 用法（容器内）： bash /tmp/verify_series.sh
set -euo pipefail

SRC=/vllm-workspace/vllm-ascend
BASE=46856f89e79c3011401e33663c60da37cd486d53
# 每次用**独立的 worktree 目录**：固定路径在反复运行时会被 git 记成 "prunable"，
# 下一次 `git worktree add` 直接以 128 失败（没有任何提示）。
VWT=${VWT:-/tmp/verify-va-$$}
PATCHES=${PATCHES:-/tmp/rel-out-va}

declare -A EXPECT=(
  [vllm_ascend/ascend_forward_context.py]=6cccd4259bd65c907ef9d9dd42a83dca
  [vllm_ascend/attention/dsa_v1.py]=9a36e709b0937589eab05c5316a62591
  [vllm_ascend/models/deepseek_v41/engram_gate.py]=146010cac42261e9dc4380699e156252
  [vllm_ascend/models/deepseek_v41/engram_hash.py]=3a842bbb6d0dd783c65087ccef347370
  [vllm_ascend/models/deepseek_v41/engram_hbm.py]=02ba2b7c258ff16663cd316b69c44fb8
  [vllm_ascend/models/deepseek_v41/engram_jit_kernel.py]=1add256a203d7f6dfd98874c575ce24a
  [vllm_ascend/models/deepseek_v41/engram_plan_kernel.py]=0be62d7775374b0167a54f5b393a65ac
  [vllm_ascend/models/deepseek_v41/engram_device_index.py]=9076716bafd14d8210e674e3489be1ee
  [vllm_ascend/models/deepseek_v41/engram_graph.py]=adc0bd8683cded42ea4e45cb9a67e654
  [vllm_ascend/models/deepseek_v41/indexer.py]=f61f242df4f060106ce1bf4500ff5844
  [vllm_ascend/models/deepseek_v41/model.py]=d22eec4c7401a2f6a37f99c4898b964d
  [vllm_ascend/ops/rope_dsv4.py]=6a19890850ac7cb41c535b070c2dfbf6
  [vllm_ascend/ops/fused_moe/token_dispatcher.py]=a695735ae3e03096a432468eb9ad6b83
)

cd "$SRC"; git worktree prune; rm -rf "$VWT"
git worktree add --detach "$VWT" "$BASE" >/dev/null 2>&1 || {
  echo "[verify][FAIL] git worktree add 失败：$VWT（先 git worktree prune）" >&2; exit 2; }
cd "$VWT"
git config user.name "verify"; git config user.email "verify@local"

echo "=== git am ==="
git am --keep-cr "$PATCHES"/*.patch >/dev/null
echo "am ok, $(git log --oneline "$BASE"..HEAD | wc -l) commits"

fail=0
for f in "${!EXPECT[@]}"; do
  got=$(md5sum "$VWT/$f" | cut -d' ' -f1)
  if [ "$got" = "${EXPECT[$f]}" ]; then
    printf 'OK   %s\n' "$f"
  else
    printf 'FAIL %s\n  want %s\n  got  %s\n' "$f" "${EXPECT[$f]}" "$got"; fail=1
  fi
done

# ---------------------------------------------------------------- 与整文件形态比对
#
# ⚠️ 不要拿**活着的容器**当基准：容器是按门控 env 挂载的，例如 `indexer.py`
#    只在 CAND_MODE≠0 时才挂（默认 0 ⇒ 容器里是 stock 版）。
#    权威载荷是发布包的 `patches/files/`，与 Dockerfile 的落位表一一对应。
PAYLOAD=${PAYLOAD:-/tmp/pkg/files}
A=vllm_ascend
MAP=(
  "ascend_forward_context.py:$A/ascend_forward_context.py"
  "dsa_v1.py:$A/attention/dsa_v1.py"
  "rope_dsv4.py:$A/ops/rope_dsv4.py"
  "token_dispatcher_moemask.py:$A/ops/fused_moe/token_dispatcher.py"
  "engram_gate.py:$A/models/deepseek_v41/engram_gate.py"
  "engram_hbm.py:$A/models/deepseek_v41/engram_hbm.py"
  "engram_hash.py:$A/models/deepseek_v41/engram_hash.py"
  "engram_jit_kernel.py:$A/models/deepseek_v41/engram_jit_kernel.py"
  "engram_plan_kernel.py:$A/models/deepseek_v41/engram_plan_kernel.py"
  "engram_device_index.py:$A/models/deepseek_v41/engram_device_index.py"
  "engram_graph.py:$A/models/deepseek_v41/engram_graph.py"
  "model.py:$A/models/deepseek_v41/model.py"
  "indexer.py:$A/models/deepseek_v41/indexer.py"
)
echo "=== 与发布包 patches/files/ 直接 diff（应为空）payload=$PAYLOAD ==="
if [ -d "$PAYLOAD" ]; then
  for pair in "${MAP[@]}"; do
    src=${pair%%:*}; tgt=${pair#*:}
    diff -q "$PAYLOAD/$src" "$VWT/$tgt" || fail=1
  done
else
  echo "SKIP: 找不到载荷目录 $PAYLOAD ⇒ 无法证明两种形态等价（本次判 FAIL）"
  fail=1
fi

echo "=== py_compile ==="
python3 -m py_compile \
  "$VWT"/vllm_ascend/ascend_forward_context.py \
  "$VWT"/vllm_ascend/attention/dsa_v1.py \
  "$VWT"/vllm_ascend/ops/rope_dsv4.py \
  "$VWT"/vllm_ascend/ops/fused_moe/token_dispatcher.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_gate.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_hbm.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_hash.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_jit_kernel.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_plan_kernel.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_device_index.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/engram_graph.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/indexer.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/model.py && echo "py_compile ok" || fail=1

[ "$fail" = 0 ] && echo "=== VERIFY: ALL PASS ===" || { echo "=== VERIFY: FAIL ==="; exit 1; }
