#!/usr/bin/env bash
# 验证：把 patch 系列 am 到「干净 base checkout」后，产出文件与线上工作树逐字节一致。
# 用法（容器内）： bash /tmp/verify_series.sh
set -euo pipefail

SRC=/vllm-workspace/vllm-ascend
BASE=46856f89e79c3011401e33663c60da37cd486d53
VWT=/tmp/verify-va
PATCHES=/tmp/rel-out-va

declare -A EXPECT=(
  [vllm_ascend/ascend_forward_context.py]=6cccd4259bd65c907ef9d9dd42a83dca
  [vllm_ascend/attention/dsa_v1.py]=9a36e709b0937589eab05c5316a62591
  [vllm_ascend/models/deepseek_v41/engram_gate.py]=146010cac42261e9dc4380699e156252
  [vllm_ascend/models/deepseek_v41/engram_hash.py]=3a842bbb6d0dd783c65087ccef347370
  [vllm_ascend/models/deepseek_v41/engram_hbm.py]=6f227a749aa6ba6f1290446611202028
  [vllm_ascend/models/deepseek_v41/engram_jit_kernel.py]=1add256a203d7f6dfd98874c575ce24a
  [vllm_ascend/models/deepseek_v41/engram_plan_kernel.py]=0be62d7775374b0167a54f5b393a65ac
  [vllm_ascend/models/deepseek_v41/indexer.py]=f61f242df4f060106ce1bf4500ff5844
  [vllm_ascend/models/deepseek_v41/model.py]=5b7c45261e2d63838b9e4a25f87ad350
  [vllm_ascend/ops/rope_dsv4.py]=6a19890850ac7cb41c535b070c2dfbf6
  [vllm_ascend/ops/fused_moe/token_dispatcher.py]=a695735ae3e03096a432468eb9ad6b83
)

cd "$SRC"; git worktree prune; rm -rf "$VWT"
git worktree add --detach "$VWT" "$BASE" >/dev/null 2>&1
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

echo "=== 与线上工作树直接 diff（应为空）==="
for f in "${!EXPECT[@]}"; do
  diff -q "$SRC/$f" "$VWT/$f" || fail=1
done

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
  "$VWT"/vllm_ascend/models/deepseek_v41/indexer.py \
  "$VWT"/vllm_ascend/models/deepseek_v41/model.py && echo "py_compile ok" || fail=1

[ "$fail" = 0 ] && echo "=== VERIFY: ALL PASS ===" || { echo "=== VERIFY: FAIL ==="; exit 1; }
