#!/usr/bin/env bash
# ★ 核心判据：**镜像里装的文件** 与 **仓库里的文件** 逐字节相同。
#
# 为什么必须有这一步：本部署形态有两个交付面
#   ① GitHub 上的启动器（PATCH_MODE=mount：现场 `-v` 挂仓库文件）
#   ② 镜像层 patch 包（PATCH_MODE=baked：文件烘在镜像真实路径）
# 两边"概念上一致"没有意义 —— 必须是**逐字节一致**，否则同一个实验号
# 在两种形态下会跑出不同结果，而没人看得出来。
#
# 判据（任一不过即 FAIL）：
#   ① 15 个 vllm-ascend 件：镜像内 md5 == 仓库源 md5
#   ② 3 个暂存件（/opt/dsv41/patches/draft/）：同上
#   ③ 4 个 .patch：同上
#   ④ 8 个起服脚本：同上
#   ⑤ 镜像内 /opt/dsv41/BUILD_INFO.txt 的 repo_rev 与本仓 HEAD 对得上
#
# 用法：
#   bash deploy/a3-ced-pd/verify_consistency.sh [镜像tag]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
IMAGE=${1:-${TAG:-local/dsv41-a3-ced-pd:v1}}
DOCKER=${DOCKER:-docker}
A=/vllm-workspace/vllm-ascend/vllm_ascend

pass=0; fail=0
ok()   { pass=$((pass+1)); printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { fail=$((fail+1)); printf '  \033[31m✗\033[0m %s\n' "$*"; }

echo "== 一致性校验：$IMAGE  vs  $PKG"

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

# ---- ① + ③ + ④：仓库 → 容器目标 的映射表（与 PAYLOAD.md 一致）----
check() {                 # check <容器内相对键> <仓库源文件>
  local k="$1" src="$2" im
  [ -f "$PKG/$src" ] || { bad "$src 在仓库里不存在"; return; }
  im=$(img_md5 "$k")
  [ -n "$im" ] || { bad "$k 在镜像里读不到"; return; }
  if [ "$im" = "$(md5sum "$PKG/$src" | cut -d' ' -f1)" ]; then ok "$k"; else bad "$k  镜像≠仓库"; fi
}

echo "-- ① vllm-ascend 件（15）"
check models/deepseek_v41/engram_hbm.py        patches/files/engram_hbm.py
check models/deepseek_v41/engram_hash.py       patches/files/engram_hash.py
check models/deepseek_v41/engram_gate.py       patches/files/engram_gate.py
check models/deepseek_v41/engram_jit_kernel.py patches/files/engram_jit_kernel.py
check models/deepseek_v41/engram_plan_kernel.py patches/files/engram_plan_kernel.py
check models/deepseek_v41/engram_device_index.py patches/files/engram_device_index.py
check models/deepseek_v41/engram_graph.py      patches/files/engram_graph.py
check models/deepseek_v41/model.py             patches/files/model.py
check models/deepseek_v41/indexer.py           patches/files/indexer.py
check ascend_forward_context.py                patches/files/ascend_forward_context.py
check ops/rope_dsv4.py                         patches/files/rope_dsv4.py
check worker/block_table.py                    patches/files/block_table.py
check ops/fused_moe/token_dispatcher.py        patches/files/token_dispatcher_moemask.py
check attention/dsa_v41.py                     experimental/ced/dsa_v41.py
check distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py experimental/ced/mooncake_hybrid_connector.py

echo "-- ② 暂存件（draft，3）"
check DRAFT/dsa_v1.py            patches/files/draft/dsa_v1.py
check DRAFT/dspark_proposer.py   patches/files/draft/dspark_proposer.py
check DRAFT/llm_base_proposer.py patches/files/draft/llm_base_proposer.py

echo "-- ③ vLLM core 补丁（4）"
check PATCH/admission_gate.patch               patches/admission_gate.patch
check PATCH/ced_scheduler_replay.patch         experimental/ced/core_scheduler_replay.patch
check PATCH/ced_scheduler_prefill_hit.patch    experimental/ced/core_scheduler_prefill_hit.patch
check PATCH/ced_runner_prompt_tail.patch       experimental/ced/core_model_runner_prompt_tail.patch

echo "-- ④ 起服脚本（8）"
for s in serve_a2.sh serve_v2.sh serve_a3.sh serve_a3_pd.sh \
         serve_a3_pd_proxy.sh serve_a3_ced_pd.sh serve_a3_ced_single.sh run_test.sh; do
  check "SCRIPT/$s" "scripts/$s"
done

echo "-- ⑤ 源码基线"
_imgrev=$($DOCKER run --rm --entrypoint bash "$IMAGE" -lc \
  'grep -m1 "^repo_rev=" /opt/dsv41/BUILD_INFO.txt | cut -d= -f2' 2>/dev/null | tr -d '\r')
_localrev="$(cd "$PKG" && git rev-parse --short HEAD 2>/dev/null)+$(cd "$PKG" && git status --porcelain | wc -l)dirty"
if [ -n "$_imgrev" ] && [ "$_imgrev" = "$_localrev" ]; then
  ok "repo_rev=$_imgrev"
else
  # 不算 FAIL：构建后仓库又提交了是正常事。但要**显式打出来**，别让人以为同版。
  printf '  \033[33m~\033[0m repo_rev  镜像=%s  仓库=%s（不一致 ⇒ 两边不是同一提交，请自行判断）\n' \
    "${_imgrev:-?}" "$_localrev"
fi

echo
if [ "$fail" = "0" ]; then
  printf '\033[32m== PASS\033[0m  逐文件一致 %d/%d\n' "$pass" "$((pass+fail))"
  exit 0
fi
printf '\033[31m== FAIL\033[0m  一致 %d，不一致/缺失 %d\n' "$pass" "$fail"
exit 1
