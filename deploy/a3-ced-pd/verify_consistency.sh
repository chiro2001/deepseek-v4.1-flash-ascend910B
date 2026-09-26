#!/usr/bin/env bash
# ★ 核心判据：**镜像里装的文件** 与 **我们的源** 逐字节相同。
#
# 为什么必须有这一步：本部署形态有两个交付面
#   ① GitHub 形态（PATCH_MODE=mount：现场 `-v` 挂仓库文件）
#   ② 镜像层 patch 形态（PATCH_MODE=baked：文件烘在镜像真实路径）
# "概念上一致"没有意义 —— 必须**逐字节一致**，否则同一个实验号在两种形态下
# 会跑出不同结果，而没人看得出来。
#
# 校验链：
#   仓库 ──build_payload.sh(纯 cp)──► payload/ ──docker COPY──► 镜像
#   本脚本比对 **镜像 vs 源**，源可以是仓库（--mode repo，默认）或 payload（--mode payload）。
#   在 a3-21 构建时一般只有 payload，用 --mode payload；两边都跑一遍最稳。
#
# 用法：
#   bash deploy/a3-ced-pd/verify_consistency.sh                       # 镜像 vs 仓库
#   bash deploy/a3-ced-pd/verify_consistency.sh --mode payload        # 镜像 vs payload/
#   bash deploy/a3-ced-pd/verify_consistency.sh --image <tag> --payload <dir>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
IMAGE=${TAG:-local/dsv41-a3-ced-pd:v1}
PAYLOAD="$HERE/payload"
MODE=repo
DOCKER=${DOCKER:-docker}
A=/vllm-workspace/vllm-ascend/vllm_ascend

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)    MODE=$2; shift 2 ;;
    --image)   IMAGE=$2; shift 2 ;;
    --payload) PAYLOAD=$2; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done

pass=0; fail=0
ok()  { pass=$((pass+1)); printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad() { fail=$((fail+1)); printf '  \033[31m✗\033[0m %s\n' "$*"; }

case "$MODE" in
  repo)    SRC=repo ;;
  payload) SRC=payload ;;
  *) echo "MODE 只能是 repo 或 payload，当前 $MODE" >&2; exit 2 ;;
esac
echo "== 一致性校验  image=$IMAGE  source=$MODE  src_root=$([ "$SRC" = repo ] && echo "$PKG" || echo "$PAYLOAD")"

$DOCKER image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FATAL: 本地没有镜像 $IMAGE" >&2; exit 20; }

# 一次性取出镜像内的 md5 表（避免几十次 docker exec）
_tmp=$(mktemp); trap 'rm -f "$_tmp"' EXIT
$DOCKER run --rm --entrypoint bash "$IMAGE" -lc "
  for f in \
    models/deepseek_v41/engram_hbm.py \
    models/deepseek_v41/engram_hash.py \
    models/deepseek_v41/engram_gate.py \
    models/deepseek_v41/engram_jit_kernel.py \
    models/deepseek_v41/engram_plan_kernel.py \
    models/deepseek_v41/engram_device_index.py \
    models/deepseek_v41/engram_graph.py \
    models/deepseek_v41/model.py \
    models/deepseek_v41/indexer.py \
    ascend_forward_context.py \
    ops/rope_dsv4.py \
    worker/block_table.py \
    ops/fused_moe/token_dispatcher.py \
    attention/dsa_v41.py \
    distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py ; do
      printf 'IMG  %s  %s\n' \"\$(md5sum $A/\$f | cut -d' ' -f1)\" \"\$f\"
  done
  for p in dsa_v1.py dspark_proposer.py llm_base_proposer.py ; do
      printf 'IMG  %s  %s\n' \"\$(md5sum /opt/dsv41/patches/draft/\$p | cut -d' ' -f1)\" \"DRAFT/\$p\"
  done
  for p in admission_gate.patch ced_scheduler_replay.patch ced_scheduler_prefill_hit.patch ced_runner_prompt_tail.patch ; do
      printf 'IMG  %s  %s\n' \"\$(md5sum /opt/dsv41/\$p | cut -d' ' -f1)\" \"PATCH/\$p\"
  done
  for s in serve_a2.sh serve_v2.sh serve_a3.sh serve_a3_pd.sh serve_a3_pd_proxy.sh serve_a3_ced_pd.sh serve_a3_ced_single.sh run_test.sh ; do
      printf 'IMG  %s  %s\n' \"\$(md5sum /opt/dsv41/scripts/\$s | cut -d' ' -f1)\" \"SCRIPT/\$s\"
  done
" > "$_tmp" 2>/dev/null || { echo "FATAL: 读镜像内文件失败" >&2; exit 21; }
img_md5() { awk -v k="$1" '$3==k{print $2}' "$_tmp" | head -1; }

# ★ 映射表（与 PAYLOAD.md 一致）：<镜像内键> <仓库相对路径> <payload 相对路径>
MAP=$(cat <<'EOF'
models/deepseek_v41/engram_hbm.py	patches/files/engram_hbm.py	ascend/engram_hbm.py
models/deepseek_v41/engram_hash.py	patches/files/engram_hash.py	ascend/engram_hash.py
models/deepseek_v41/engram_gate.py	patches/files/engram_gate.py	ascend/engram_gate.py
models/deepseek_v41/engram_jit_kernel.py	patches/files/engram_jit_kernel.py	ascend/engram_jit_kernel.py
models/deepseek_v41/engram_plan_kernel.py	patches/files/engram_plan_kernel.py	ascend/engram_plan_kernel.py
models/deepseek_v41/engram_device_index.py	patches/files/engram_device_index.py	ascend/engram_device_index.py
models/deepseek_v41/engram_graph.py	patches/files/engram_graph.py	ascend/engram_graph.py
models/deepseek_v41/model.py	patches/files/model.py	ascend/model.py
models/deepseek_v41/indexer.py	patches/files/indexer.py	ascend/indexer.py
ascend_forward_context.py	patches/files/ascend_forward_context.py	ascend/ascend_forward_context.py
ops/rope_dsv4.py	patches/files/rope_dsv4.py	ascend/rope_dsv4.py
worker/block_table.py	patches/files/block_table.py	ascend/block_table.py
ops/fused_moe/token_dispatcher.py	patches/files/token_dispatcher_moemask.py	ascend/token_dispatcher.py
attention/dsa_v41.py	experimental/ced/dsa_v41.py	ced/dsa_v41.py
distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py	experimental/ced/mooncake_hybrid_connector.py	ced/mooncake_hybrid_connector.py
DRAFT/dsa_v1.py	patches/files/draft/dsa_v1.py	draft/dsa_v1.py
DRAFT/dspark_proposer.py	patches/files/draft/dspark_proposer.py	draft/dspark_proposer.py
DRAFT/llm_base_proposer.py	patches/files/draft/llm_base_proposer.py	draft/llm_base_proposer.py
PATCH/admission_gate.patch	patches/admission_gate.patch	patches/admission_gate.patch
PATCH/ced_scheduler_replay.patch	experimental/ced/core_scheduler_replay.patch	patches/core_scheduler_replay.patch
PATCH/ced_scheduler_prefill_hit.patch	experimental/ced/core_scheduler_prefill_hit.patch	patches/core_scheduler_prefill_hit.patch
PATCH/ced_runner_prompt_tail.patch	experimental/ced/core_model_runner_prompt_tail.patch	patches/core_model_runner_prompt_tail.patch
SCRIPT/serve_a2.sh	scripts/serve_a2.sh	scripts/serve_a2.sh
SCRIPT/serve_v2.sh	scripts/serve_v2.sh	scripts/serve_v2.sh
SCRIPT/serve_a3.sh	scripts/serve_a3.sh	scripts/serve_a3.sh
SCRIPT/serve_a3_pd.sh	scripts/serve_a3_pd.sh	scripts/serve_a3_pd.sh
SCRIPT/serve_a3_pd_proxy.sh	scripts/serve_a3_pd_proxy.sh	scripts/serve_a3_pd_proxy.sh
SCRIPT/serve_a3_ced_pd.sh	scripts/serve_a3_ced_pd.sh	scripts/serve_a3_ced_pd.sh
SCRIPT/serve_a3_ced_single.sh	scripts/serve_a3_ced_single.sh	scripts/serve_a3_ced_single.sh
SCRIPT/run_test.sh	scripts/run_test.sh	scripts/run_test.sh
EOF
)

n_map=0
while IFS=$'\t' read -r key repo_rel pl_rel; do
  [ -n "$key" ] || continue
  n_map=$((n_map+1))
  if [ "$SRC" = repo ]; then f="$PKG/$repo_rel"; else f="$PAYLOAD/$pl_rel"; fi
  if [ ! -f "$f" ]; then bad "$key   源缺失：$f"; continue; fi
  im=$(img_md5 "$key")
  if [ -z "$im" ]; then bad "$key   镜像里读不到"; continue; fi
  if [ "$im" = "$(md5sum "$f" | cut -d' ' -f1)" ]; then ok "$key"; else bad "$key   镜像≠源"; fi
done <<< "$MAP"

echo
echo "-- 源码基线（仅 repo 模式可比）"
if [ "$SRC" = repo ]; then
  _imgrev=$($DOCKER run --rm --entrypoint bash "$IMAGE" -lc \
    'grep -m1 "^repo_rev=" /opt/dsv41/BUILD_INFO.txt | cut -d= -f2' 2>/dev/null | tr -d '\r')
  _localrev="$(cd "$PKG" && git rev-parse --short HEAD 2>/dev/null)+$(cd "$PKG" && git status --porcelain | wc -l)dirty"
  if [ -n "$_imgrev" ] && [ "$_imgrev" = "$_localrev" ]; then
    ok "repo_rev=$_imgrev"
  else
    printf '  \033[33m~\033[0m repo_rev  镜像=%s  仓库=%s（不一致 ⇒ 两边不是同一提交）\n' "${_imgrev:-?}" "$_localrev"
  fi
else
  echo "  （payload 模式跳过：镜像里的 repo_rev 只在构建时写入，payload 里没有它）"
fi

echo
if [ "$fail" = "0" ]; then
  printf '\033[32m== PASS\033[0m  逐文件一致 %d/%d（source=%s）\n' "$pass" "$n_map" "$SRC"
  exit 0
fi
printf '\033[31m== FAIL\033[0m  一致 %d，不一致/缺失 %d（source=%s）\n' "$pass" "$fail" "$SRC"
exit 1
