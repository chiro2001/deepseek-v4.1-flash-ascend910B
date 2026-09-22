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

# ---------------------------------------------------------------- ★ 档位
# 档 A（默认，8 卡真权重已验，logs/027）：
#   128K x 16 并发 -> OFFLOAD_GB=56（57,344 unit = 实测需求 48,064 的 1.193x）
#                     宿主实占 392.4 GiB（A2 余量 442 GiB 的 88%，⚠️ 紧）
# 档 B（+L1，内存砍半）：加 P2_POOL_PATCH=1 + P2_COMP_JSON=<见 §2.2>
#                     同样 OFFLOAD_GB=56 -> 宿主约 203.6 GiB（45%）
#   32K x 16 并发  -> OFFLOAD_GB=10
OFFLOAD_GB=${OFFLOAD_GB:-56}
MAX_LEN=${MAX_LEN:-131072}
MAX_SEQS=${MAX_SEQS:-16}
BAT_TOKENS=${BAT_TOKENS:-2048}

# ★ L1（池张量按需分配行数）—— ★★ 8 卡真权重实测 1.9895x（392.35 -> 197.21 GiB）
#   defaults 到这里 = 档 B（推荐）；置 0 即回档 A
#   P2_POOL_PATCH=1 时必须同时给对的 P2_COMP_JSON（与"张量数"匹配，给错会 fail-closed）
P2_POOL_PATCH=${P2_POOL_PATCH:-1}
# 16 张量几何的【实测·worker 侧真值】分量（8 卡真权重，8 rank x 9 行一致）
P2_COMP_JSON=${P2_COMP_JSON:-'[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]'}

# ★ 池分配器加固（logs/041）：默认关；=1 把三种静默失败变成响亮 raise
PGP_MGR_HARDEN=${PGP_MGR_HARDEN:-0}
PGP_MGR_STATS=${PGP_MGR_STATS:-0}

# ---------------------------------------------------------------- ★ int8 档（logs/047）
# 档 C（容量 ×1.4655）：SWA 页 INT8 + ring16 + APC 对齐
# 档 D（容量 ×1.9133）：再加 KV8 双平面 + prefill 融合
# ★ 前置：VLLM_V41_APC_ALIGN=3 是它们能正确工作的前提（否则 D/F 几何会翻 token）
KV8_SWA=${KV8_SWA:-0}        # 1 = SWA 页 INT8（档 C 起）
KV8_RING_FP16=${KV8_RING_FP16:-0}  # 1 = state ring FP32→FP16（档 C 的必需前置）
KV8_FULL=${KV8_FULL:-0}      # 1 = long-KV 也 INT8（档 D）
KV8_PREFILL=${KV8_PREFILL:-0}  # 1 = prefill 融合 kernel（档 D）
# ★ APC 对齐：0 = 旧行为；3 = 段栅格（推荐，档 C/D 必开）
APC_ALIGN=${APC_ALIGN:-0}

# ★ 开了 int8 就必须同时开 APC 对齐（否则 D/F 几何会翻 token，logs/047）
if { [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; } \
   && [ "$APC_ALIGN" = "0" ]; then
    echo "⚠⚠ 你开了 int8 但 APC_ALIGN=0 ⇒ 自动置 3（D/F 几何否则会翻 token，logs/047）" >&2
    APC_ALIGN=3
fi

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
if [ "$P2_POOL_PATCH" = "1" ]; then
    echo "  池子          : ${OFFLOAD_GB} GiB（★ 档 B 宿主实占 ≈197 GiB，8 卡实测 1.9895x）"
else
    echo "  池子          : ${OFFLOAD_GB} GiB（档 A 宿主实占 ≈392 GiB，8 卡实测）"
fi
echo "  上下文/并发   : ${MAX_LEN} / ${MAX_SEQS}"
echo "  池后端        : $NPU_OFFLOAD_HOST_MEM"
echo "  blocks_per_chunk: $BLOCKS_PER_CHUNK"
echo "  prefix_match_unit: $PREFIX_MATCH_UNIT"
echo "  ENGRAM        : $ENGRAM"
echo "  L1 (P2_POOL_PATCH): $P2_POOL_PATCH${P2_COMP_JSON:+  comp=$P2_COMP_JSON}"
echo "  加固 PGP_MGR_HARDEN: $PGP_MGR_HARDEN（stats=$PGP_MGR_STATS）"
if [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; then
    if [ "$KV8_FULL" = "1" ]; then
        echo "  ★ int8 档 D: SWA=$KV8_SWA ring16=$KV8_RING_FP16 full=$KV8_FULL prefill=$KV8_PREFILL（容量 ×1.9133）"
    else
        echo "  ★ int8 档 C: SWA=$KV8_SWA ring16=$KV8_RING_FP16（容量 ×1.4655）"
    fi
    echo "  ★ APC_ALIGN=$APC_ALIGN（3 = 段栅格；档 C/D 的必需前提）"
else
    echo "  int8            : 关（档 B，容量 ×1.000 + L1）"
fi
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
# ★ 可选项（默认关；不改默认行为）
[ "$P2_POOL_PATCH" = "1" ] && export P2_POOL_PATCH=1 && export P2_WORKER_ROWS=1
case "$P2_POOL_PATCH" in
  1) : "${P2_COMP_JSON:?★ 开 L1 时必须给 P2_COMP_JSON（与张量数匹配，给错会 fail-closed）}"; export P2_COMP_JSON ;;
esac
export PGP_MGR_HARDEN PGP_MGR_STATS

export VLLM_V41_KV8_SWA="$KV8_SWA" \
       VLLM_V41_RING_FP16="$KV8_RING_FP16" \
       VLLM_V41_KV8="$KV8_FULL" \
       VLLM_V41_KV8_PREFILL="$KV8_PREFILL" \
       VLLM_V41_APC_ALIGN="$APC_ALIGN"

KV_ARGS="--prefix-match-unit $PREFIX_MATCH_UNIT \
--kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((OFFLOAD_GB * 1073741824)),\"blocks_per_chunk\":$BLOCKS_PER_CHUNK,\"spec_name\":\"NPUOffloadingSpec\",\"spec_module_path\":\"vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu\"}}"

echo "-------------------------------------------------------------"
echo "★ 起服后必须跑这三条自检（任一为 0 就停，别压测）："
echo "    grep -c 'P1_pinned.*ret=0'            <serve.log>   # 期望 8"
echo "    grep -c 'D2_offload'                  <serve.log>   # 期望 >0"
echo "    grep -c 'alignment_chunk_count.*8'    <serve.log>   # 期望 >0（per-group 生效）"
echo "  ★ 跑 L1 时再加两条："
echo "    grep -c 'P2_poolsizing'               <serve.log>   # 期望 >0（L1 生效）"
echo "    grep -a 'P2_WORKER_HOST_BYTES'        <serve.log>   # ★ 宿主实占（档 B 应 ≈1.927x 更省）"
echo "  ★ 上线后监测（logs/027 判据 1 + logs/040 判据 2）："
echo "    kv_offload_block_removed_total{medium=\"CPU\"} == 0"
echo "    units_ratio >= 1.05（工作集 = cpu_cache_usage_perc x num_units）"
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
