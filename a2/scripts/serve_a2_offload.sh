#!/usr/bin/env bash
# =============================================================================
# serve_a2_offload.sh —— A2（8×910B3）DRAM KV 卸载一键起服
#
# 设计原则：**不改** dsv41-release/scripts/serve_a2.sh（那是生产脚本）。
#
# 依赖 **shadow-pkg**（一份把 serve_a2.sh 打了两处注入补丁的副本）：
#   * 它认 `KV_ARGS_EXTRA`（把卸载参数带进容器）
#   * 它认 `OFFLOAD_SCHED_PATCH` / `OFFLOAD_NPU_WORKER_PATCH`（挂两个补丁）
# 本脚本把 `a2/patches/` 的四个文件复制进 shadow-pkg 的补丁目录，再调它的 serve_a2.sh。
#
# 用法：
#   bash a2/scripts/serve_a2_offload.sh              # 32K 场景（OFFLOAD_GB=16）
#   OFFLOAD_GB=48 MAX_LEN=131072 bash a2/scripts/serve_a2_offload.sh   # 128K 场景
#
# ★ 起服后**必须先跑自检**（脚本末尾会打印命令），否则可能白等 20 分钟。
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)          # dsv41-release/

# ---------------------------------------------------------------- 参数
# ★ 模型路径不硬编码（发布包不留真实账号名）；必须由调用方给
MODEL=${MODEL:?请设 MODEL=<模型目录>}
IMAGE=${IMAGE:-}
GPU_UTIL=${GPU_UTIL:-0.90}
PORT=${PORT:-8077}
SERVED_NAME=${SERVED_NAME:-deepseek-v4-flash}

# 池子：按 A2 的宿主余量（442 GiB）与 x6.945 乘数定
#   32K x 16 并发  -> 16 GiB（宿主 ≈111 GiB）
#   128K x 16 并发 -> 48 GiB（宿主 ≈333 GiB）
OFFLOAD_GB=${OFFLOAD_GB:-16}
MAX_LEN=${MAX_LEN:-40960}
MAX_SEQS=${MAX_SEQS:-16}
BAT_TOKENS=${BAT_TOKENS:-2048}

# 池后端：registered（aclrtHostRegister，推荐）/ pageable / pinned
NPU_OFFLOAD_HOST_MEM=${NPU_OFFLOAD_HOST_MEM:-registered}

# per-group blocks_per_chunk：SWA=1（细粒度）、full=8
#   ★ 这一项让「SWA 只保留每 1024 token 段的尾块」生效，池子需求降 4.89x
BLOCKS_PER_CHUNK=${BLOCKS_PER_CHUNK:-'{"default":8,"swa":1}'}

# ★ 必须：否则撞 tokens_per_block=32 % tokens_per_hash=128
PREFIX_MATCH_UNIT=${PREFIX_MATCH_UNIT:-32}

ENGRAM=${ENGRAM:-0}      # ★ 首版建议 0（Engram + 卸载池 曾撞 207001）
DRY=${DRY:-0}

echo "=============================================================="
echo "A2 DRAM KV 卸载起服"
echo "=============================================================="
echo "  模型          : $MODEL"
echo "  池子          : ${OFFLOAD_GB} GiB（宿主实占 ≈$((OFFLOAD_GB * 7)) GiB）"
echo "  上下文/并发   : ${MAX_LEN} / ${MAX_SEQS}"
echo "  池后端        : $NPU_OFFLOAD_HOST_MEM"
echo "  blocks_per_chunk: $BLOCKS_PER_CHUNK"
echo "  prefix_match_unit: $PREFIX_MATCH_UNIT"
echo "  ENGRAM        : $ENGRAM"
echo "-------------------------------------------------------------"

# ---------------------------------------------------------------- 补丁
# 三个补丁文件（本目录的 ../patches/）
PDIR="$REPO/a2/patches"
for f in 0001-offload-scheduler.patch.py 0001b-offload-per-group-bpc-manager.patch.py \
         0001c-offload-per-group-bpc-hooks.patch.py 0002-offload-cpu-pool-host-registered.patch.py; do
    if [ ! -f "$PDIR/$f" ]; then
        echo "✗ 缺补丁 $PDIR/$f" >&2
        exit 2
    fi
done
echo "✓ 四个补丁文件已就位（md5 见 a2/patches/README.md）"

# shadow-pkg 的补丁目录（serve_a2.sh 的 PATCH_MODE=mount 从这里挂）
SHADOW=${SHADOW_PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
PATCHDIR="$SHADOW/patches/files/offload_dsv41"
if [ ! -d "$SHADOW" ]; then
    echo "⚠ 找不到 shadow-pkg（$SHADOW）—— 请设 SHADOW_PKG=<路径>" >&2
    echo "  （补丁必须经 shadow-pkg 挂载；不要直接改 dsv41-release/scripts/serve_a2.sh）" >&2
    exit 2
fi
mkdir -p "$PATCHDIR"
cp "$PDIR/0001-offload-scheduler.patch.py"                  "$PATCHDIR/scheduler.py"
cp "$PDIR/0001b-offload-per-group-bpc-manager.patch.py"     "$PATCHDIR/pgp_manager.py"
cp "$PDIR/0001c-offload-per-group-bpc-hooks.patch.py"       "$PATCHDIR/pgp_hooks.py"
cp "$PDIR/0002-offload-cpu-pool-host-registered.patch.py"   "$PATCHDIR/cpu_npu.py"
echo "✓ 补丁已复制到 $PATCHDIR"

# ---------------------------------------------------------------- 起服
export OFFLOAD_SCHED_PATCH=1
export OFFLOAD_NPU_WORKER_PATCH=1
export NPU_OFFLOAD_HOST_MEM
export PREFIX_MATCH_UNIT
export ENGRAM

KV_ARGS="--prefix-match-unit $PREFIX_MATCH_UNIT \
--kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((OFFLOAD_GB * 1073741824)),\"blocks_per_chunk\":$BLOCKS_PER_CHUNK,\"spec_name\":\"NPUOffloadingSpec\",\"spec_module_path\":\"vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu\"}}"

echo "-------------------------------------------------------------"
echo "★ 起服后必须跑这三条自检（任一为 0 就停，别压测）："
echo "    grep -c 'P1_pinned.*ret=0'            <serve.log>   # 期望 8"
echo "    grep -c 'D2_offload'                  <serve.log>   # 期望 >0"
echo "    grep -c 'alignment_chunk_count.*8'    <serve.log>   # 期望 >0（per-group 生效）"
echo "-------------------------------------------------------------"

if [ "$DRY" = "1" ]; then
    echo "[DRY] 将要执行："
    echo "  OFFLOAD_GB=$OFFLOAD_GB MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS \\"
    echo "  KV_ARGS_EXTRA='$KV_ARGS' \\"
    echo "  bash scripts/serve_a2.sh"
    exit 0
fi

cd "$REPO"
MODEL="$MODEL" IMAGE="$IMAGE" GPU_UTIL="$GPU_UTIL" PORT="$PORT" \
SERVED_NAME="$SERVED_NAME" MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" \
BAT_TOKENS="$BAT_TOKENS" \
KV_ARGS_EXTRA="$KV_ARGS" \
    bash scripts/serve_a2.sh
