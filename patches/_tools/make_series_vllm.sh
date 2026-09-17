#!/usr/bin/env bash
# 在容器内生成 vllm（core）的 patch 系列：admission gate
# 用法（容器内）： bash /tmp/make_series_vllm.sh <输出目录>
set -euo pipefail

SRC=/vllm-workspace/vllm
BASE=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
WT=/tmp/rel-vllm
OUT=${1:-/tmp/rel-out-vllm}

echo "[series] src=$SRC base=$BASE wt=$WT out=$OUT"
cd "$SRC"
git worktree prune
rm -rf "$WT"
mkdir -p "$OUT"; rm -f "$OUT"/*.patch
git worktree add --detach "$WT" "$BASE" >/dev/null 2>&1

cp -f "$SRC/vllm/v1/core/sched/scheduler.py" "$WT/vllm/v1/core/sched/scheduler.py"

cd "$WT"
git config user.name  "chiro2001"
git config user.email "chiro2001@163.com"
git add -- vllm/v1/core/sched/scheduler.py
git commit -q -F - <<'EOF'
feat(scheduler): admission gate to protect decode from prefill starvation

长上下文场景下 prefill 会长期占住调度步，decode 被饿死（表现为
"prefill-only step #N ... deferred_decode_reqs=1" 连续上百步、首 token
之后长时间不出字）。admission gate 在同一个 step 里只放 prefill、把
decode 延后到下一个 step 之前的门控步，保证 decode 有稳定节拍。

门控：VLLM_ADMISSION_GATE=1（默认关；只认环境变量，避免
      --additional-config 的 extra="forbid" 拒绝既有部署）
观测：日志打印 [admission_gate] prefill-only step #N (step=M)
      用于确认门控真的在起作用（配合 tools/negative_control 使用）。
EOF
echo "[series] $(git log --oneline -1)"

git format-patch --no-signature -o "$OUT" "$BASE" >/dev/null
ls -1 "$OUT"
