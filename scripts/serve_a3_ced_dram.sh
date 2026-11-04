#!/usr/bin/env bash
# Experimental real CED P + native DRAM cache; decode is a separate mock/real D.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CED_DRAM=1
export V41_CED_MOCK_ENABLE=${V41_CED_MOCK_ENABLE:-1}
export DEVS=${DEVS:-"6 7 8 9 10 11 12 13"}
export CED_DRAM_GB=${CED_DRAM_GB:-64}
export CED_DRAM_PENDING_BLOCKS=${CED_DRAM_PENDING_BLOCKS:-16384}
export PREFIX=1 V41_CED_ALLOW_PREFIX=1 V41_CED_P_HIT_DIAG=1
export ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
export CPU_BIND=0 DROPCACHE=0
export MULTISTREAM=0 DSA_OVERLAP=0
export SPEC=0 DRAFT_GRAPH=0
export MAX_LEN=${MAX_LEN:-1048576} MAX_SEQS=${MAX_SEQS:-4}
export PATCH_MODE=${PATCH_MODE:-mount}
export PORT=${PORT:-19190} KV_PORT=${KV_PORT:-19290}
export SERVED_NAME=${SERVED_NAME:-deepseek-v41-ced-dram-p}
export NAME=${NAME:-ced-dram-p-$(date +%Y%m%d_%H%M%S)}
exec bash "$HERE/serve_a3_ced_pd.sh" prefill
