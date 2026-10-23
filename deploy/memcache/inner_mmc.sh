#!/usr/bin/env bash
# MemCache(AscendStoreConnector, kv_both) + DSpark 混布起服脚本（单机/跨机池通用）
# 用法（容器内）：MODEL=/weights/.../model bash inner_mmc.sh
set -uo pipefail
export MODEL=${MODEL:?need MODEL path}
export TP=${TP:-8} DP=${DP:-1} PORT=${PORT:-8100} SERVED_NAME=${SERVED_NAME:-deepseek-v41}
export MAX_LEN=${MAX_LEN:-1048576} MAX_SEQS=${MAX_SEQS:-32} BAT_TOKENS=${BAT_TOKENS:-8192}
export MMC_LOCAL_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH:-/root/mmc/mmc-local.conf}
MMC_LIB=$(python3 -c 'import os,memcache_hybrid;print(os.path.dirname(memcache_hybrid.__file__)+"/lib")')
export LD_LIBRARY_PATH="$MMC_LIB:${LD_LIBRARY_PATH:-}"
KVC='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both"}'
export EXTRA="--tokenizer-mode deepseek_v41 --kv-transfer-config $KVC"
exec bash /opt/dsv41/scripts/serve_v2.sh
