#!/usr/bin/env bash
# scal 二分：每个臂一个**独立进程**（同进程连续捕获会触发 NPU 分配器断言）
set -uo pipefail
V=${V:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}
export ASCEND_CUSTOM_OPP_PATH="$V"
export LD_LIBRARY_PATH="$V/op_api/lib:${LD_LIBRARY_PATH:-}"
OUT=${OUT:-/work/out}
mkdir -p "$OUT"
ARMS=${ARMS:-none seq_lens opt_cpu positions input_ids dflash_hs ctx_pos qslot ctxslot tok_idx ALL}
for a in $ARMS; do
  echo "===== ARM $a ====="
  timeout 1200 python3 /work/draft_ab.py --out "$OUT" --stage scal --max-tokens 2048 --arm "$a" \
    > "$OUT/S_arm_$a.log" 2>&1
  echo "rc=$?"
  grep -a "\[scal\] " "$OUT/S_arm_$a.log" | grep -aE "arm=|equal=" | tail -3
done
