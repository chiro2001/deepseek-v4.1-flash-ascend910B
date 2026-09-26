#!/usr/bin/env bash
# 本部署形态的公共参数（**这些值是 A3-21 上实测通过的**，别随手改）。
#
# 改任何一个都要重新走一遍硬门（见 ../README.md §4）+ 144K/1M 四针。
# shellcheck shell=bash

# ---- 必填 ----
: "${MODEL:?必须设 MODEL=<完整 DeepSeek-V4.1-Flash W4A8+DSpark 模型目录>}"

# ---- 部署形态 ----
export DSV41_DEPLOY_FORM=a3-ced-pd
# mount = 仓库 + -v 挂载（默认）；baked = 工作镜像烘好的文件
export PATCH_MODE=${PATCH_MODE:-mount}

# ---- 角色与卡 ----
export PD_PREFILL_DEVS=${PD_PREFILL_DEVS:-"0 1 2 3 4 5 6 7"}
export PD_DECODE_DEVS=${PD_DECODE_DEVS:-"8 9 10 11 12 13 14 15"}
export PD_PREFILL_PORT=${PD_PREFILL_PORT:-18990}
export PD_DECODE_PORT=${PD_DECODE_PORT:-18991}
export PD_PREFILL_KV_PORT=${PD_PREFILL_KV_PORT:-19090}
export PD_DECODE_KV_PORT=${PD_DECODE_KV_PORT:-19091}

# ---- 共同参数（两侧一致）----
export SERVED_NAME=${SERVED_NAME:-deepseek-v41-ced-pd}
export MAX_LEN=${MAX_LEN:-1048576}        # 1M 上下文
export MAX_SEQS=${MAX_SEQS:-4}
export BAT_TOKENS=${BAT_TOKENS:-8192}
export GPU_UTIL=${GPU_UTIL:-0.92}
export BLOCK=${BLOCK:-128}
export KV_DTYPE=${KV_DTYPE:-bfloat16}     # ★ BF16 KV，不是 int8
export ENGRAM=${ENGRAM:-1}
export ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
export CPU_BIND=${CPU_BIND:-0}            # ★ A3 共用机必需：关内部 NUMA 绑核/迁移
export DROPCACHE=${DROPCACHE:-0}          # ★ A3 共用机：不清整机 page cache
export VISION=${VISION:-1}
export NPUGRAPH_EX=${NPUGRAPH_EX:-1}

# ---- ★ 池钳位：4 GiB 页步长上界（漏了 = 1M 静默空答）----
# 槽位 3 页步长 147712 B，⌊2³²/147712⌋ = 29076。判据用**页尾**。
export CED_MAX_NUM_BLOCKS=${CED_MAX_NUM_BLOCKS:-29076}
export KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-15728022528}   # 29076 × 540928

# ---- ★ D 侧必须关多流（漏了 = 长上下文静默算错，实测 0/4）----
export MULTISTREAM=${MULTISTREAM:-0}
export DSA_OVERLAP=${DSA_OVERLAP:-0}
# 前缀缓存：**默认开**（2026-09-27 起交付口径）。关掉用 PREFIX=0。
# 依据 = 144K/1M 常规·整池·交错命中全部正确、冷热逐字节一致（≈16–18×），
# 且三处会打死引擎的问题已修并有真机触发痕迹。
# 见 docs/CED-PD-CACHE-HIT-PLAN-20260925.md §11–§13。
export PREFIX=${PREFIX:-1}

say() { printf '\033[1m[a3-ced-pd]\033[0m %s\n' "$*"; }
