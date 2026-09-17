#!/usr/bin/env bash
# 在容器内生成 vllm-ascend 的 patch 系列（带 git 历史，git am 可复现）
# 用法（容器内）： bash /tmp/make_series_va.sh <输出目录>
set -euo pipefail

SRC=/vllm-workspace/vllm-ascend
BASE=46856f89e79c3011401e33663c60da37cd486d53
WT=/tmp/rel-va
OUT=${1:-/tmp/rel-out-va}

echo "[series] src=$SRC base=$BASE wt=$WT out=$OUT"
cd "$SRC"

# --- 0) 前置校验：工作树必须就是我们要发布的那些文件 ---
git -C "$SRC" worktree prune
rm -rf "$WT"
mkdir -p "$OUT"
rm -f "$OUT"/*.patch

git worktree add --detach "$WT" "$BASE" >/dev/null 2>&1
echo "[series] worktree ok"

# --- 1) 把改动文件搬进 worktree（未跟踪的新文件也要） ---
FILES=(
  vllm_ascend/attention/dsa_v1.py
  vllm_ascend/ops/rope_dsv4.py
  vllm_ascend/models/deepseek_v41/indexer.py
  vllm_ascend/ascend_forward_context.py
  vllm_ascend/ops/fused_moe/token_dispatcher.py
  vllm_ascend/models/deepseek_v41/engram_gate.py
  vllm_ascend/models/deepseek_v41/engram_hbm.py
  vllm_ascend/models/deepseek_v41/engram_hash.py
  vllm_ascend/models/deepseek_v41/engram_jit_kernel.py
  vllm_ascend/models/deepseek_v41/engram_plan_kernel.py
  vllm_ascend/models/deepseek_v41/model.py
)
for f in "${FILES[@]}"; do
  cp -f "$SRC/$f" "$WT/$f"
done

cd "$WT"
git config user.name  "chiro2001"
git config user.email "chiro2001@163.com"

commit() { # commit <subject> <files...>   —— 正文从 stdin 读
  local subj="$1"; shift
  local body
  body="$(cat)"
  git add -- "$@"
  git commit -q -F - <<EOF
$subj

$body
EOF
  echo "[series] $(git log --oneline -1)"
}

# --- 2) 八个逻辑提交（顺序即依赖顺序） ---

commit "perf(moe): dispatch/combine over AllGather when TP=EP" \
  vllm_ascend/ascend_forward_context.py <<'EOF'
当 TP=EP（本例 TP=8 + --enable-expert-parallel）时，MoE 的 dispatch/combine
走 AllGather 路径，比逐 token 的标量开销更容易被 token 数摊薄。

门控：V41_MOE_COMM_ALLGATHER=1（默认 0 = 行为与上游一致）
实测（A3-node1，128K 单流，同会话配对 A/B）：
  128K -4.25 ms/step、32K -1.35、8K -1.23
  KV 池 3.39M -> 4.16M tokens（同一 gpu-memory-utilization）
  输出与上游逐字节一致
证据：reports/engram-sync-optimization.md、F2_ENGRAM_SYNC_MEASURED.md
EOF

commit "perf(moe): range-compare expert mask, optional invalid-row zeroing" \
  vllm_ascend/ops/fused_moe/token_dispatcher.py <<'EOF'
expert_map[topk_ids] != -1 在标准 EP 映射（本地专家连成一段）下等价于
(topk_ids >= first) & (topk_ids < last)，省掉 aclnnIndex 的
Index + IndexCheck 两个大 kernel。掩码本身保留：expanded_row_idx 里的 -1
会让 unpermute 读到未写入的行。

门控：V41_MOE_MASK_RANGE=1（默认 0）
      配合 V41_MOE_ZERO_INVALID=1 时额外把无效行清零（实验项，未端到端验证）
实测：-0.51 ms/step；精度 GSM8K 100/100、Vision 23/23
安全边界：EPLB（动态/静态）开启时自动回落到原路径。
EOF

commit "perf(rope): fuse cos/sin table index selection" \
  vllm_ascend/ops/rope_dsv4.py <<'EOF'
cos/sin 取表链由 6 个 kernel 收敛到 2 个。

门控：V41_ROPE_IDXSEL=1（默认 0）
实测：-0.45 ~ -0.62 ms/pass
EOF

commit "perf(indexer): fast path when QLI has no candidate" \
  vllm_ascend/models/deepseek_v41/indexer.py <<'EOF'
QLI 在没有候选时仍会走一遍去重/排序链，直接短路。

门控：V41_QLI_NO_CANDIDATE=1（默认 0）
      V41_FORCE_CAND_MODE=0/3/4 仅为诊断用（0 = stock 等价）
实测：单算子 99.3 us -> 50.3 us => -0.49 ms/step
EOF

commit "perf(attention): 2D wo_a matmul and dummy-shape guard" \
  vllm_ascend/attention/dsa_v1.py <<'EOF'
F3：wo_a 的退化 batch matmul 改为 2D matmul；
另有 V41_DUMMY_WO_A_FIX 处理 dummy batch 下的 wo_a 形状（默认 0）。

门控：V41_O_PROJ_2D=1（默认 0）、V41_DUMMY_WO_A_FIX（默认 0）
实测：-0.31 ~ -0.76 ms/step
EOF

commit "perf(engram): chunked gate without 2048-row padding" \
  vllm_ascend/models/deepseek_v41/engram_gate.py <<'EOF'
gate 按 chunk 分块计算，去掉固定 2048 行的 padding；同时把同一函数里
重复的 .float() 合并（V41_ENGRAM_GATE_HOIST，默认 0 = 保持两次独立 cast）。

门控：V41_ENGRAM_GATE_CHUNK=<int>（空/0 = stock 路径）
      V41_ENGRAM_GATE_MAX_TOKENS=2048 与 chunk 配合使用
实测：8K 下 -1.56 ms，KV 反而更省
注意：线上验证配置用的是 CHUNK=0（stock gate）+ MAX_TOKENS=2048，
      分块路径的收益需按你自己的 batch 形状复测。
EOF

commit "feat(engram): host-resident INT8 tables with local-owner fast path" \
  vllm_ascend/models/deepseek_v41/engram_hbm.py \
  vllm_ascend/models/deepseek_v41/model.py \
  vllm_ascend/models/deepseek_v41/engram_plan_kernel.py <<'EOF'
Engram INT8 表常驻 host（DRAM），并加 local-owner 快速路径
（走 numpy/numba 的 plan 分支，省一次 metadata all_gather 与 ids all_to_all）。
engram_plan_kernel.py 是该分支的 sidecar（目标目录必须同放）。

门控：V41_ENGRAM_HOST_RESIDENT=1（仅在 storage_format=int8 生效）
      V41_ENGRAM_LOCAL_OWNER=on + V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner
      V41_ENGRAM_REUSE_EP_GROUP=1
      V41_ENGRAM_ROUTE_PROBE=1 仅为每 N 步打印一次 route 分解（默认关）
证据：F2_ENGRAM_SYNC_MEASURED.md、reports/engram-sync-optimization.md
EOF

commit "perf(engram): numba JIT for hash and plan kernels" \
  vllm_ascend/models/deepseek_v41/engram_hash.py \
  vllm_ascend/models/deepseek_v41/engram_jit_kernel.py <<'EOF'
hash / plan 两条 host 侧热路径改成 numba JIT（sidecar：engram_jit_kernel.py）。
首次运行会编译一次，之后走 numba 磁盘缓存（建议挂 NUMBA_CACHE_DIR）。

门控：V41_ENGRAM_JIT=1（默认 0；numba 不可用时自动回落到纯 Python）
实测：hash 0.427 -> 0.076 ms/step；plan 0.261 -> 0.068 ms/step
EOF

# numbering: 0001..0008 from base（保留真实 commit hash，便于追溯）
git format-patch --no-signature -o "$OUT" "$BASE" >/dev/null
echo "[series] produced:"
ls -1 "$OUT"
git log --oneline "$BASE"..HEAD
