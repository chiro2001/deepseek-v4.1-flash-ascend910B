#!/usr/bin/env bash
# 在容器内跑全部判定实验（一条命令出全套原始输出）。
#   docker exec <容器> bash -lc 'cd /work && bash draft_ab_run.sh'
set -uo pipefail
V=${V:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}
export ASCEND_CUSTOM_OPP_PATH="$V"
export LD_LIBRARY_PATH="$V/op_api/lib:${LD_LIBRARY_PATH:-}"
OUT=${OUT:-/work/out}
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "===== RUN $name : $* ====="
  timeout 2400 python3 /work/draft_ab.py --out "$OUT" "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
  grep -aE "\[eager\] run|\[graph\] replay run|\[ctrl\]|\[fresh\]|\[kv\]|\[fix\]|\[dump\]|\[addr\]|==>" "$OUT/$name.log" || true
}

run C1_both   --stage both  --max-tokens 2048 --repeat 3 "$@"
run C2_ctrl   --stage ctrl  --max-tokens 2048 --repeat 3
run C3_fresh  --stage fresh --max-tokens 2048
run C4_kv     --stage kv    --max-tokens 2048
run C5_fix    --stage fix   --variants bind_sl_qsl
run C6_fixall --stage fix   --variants bind_all
run C7_addr   --stage addr  --max-tokens 2048
echo "===== 全部完成，原始日志在 $OUT ====="
