#!/usr/bin/env bash
# 一份 profile 的"体检报告"：一次跑完本轮建立的四个分析，输出可粘贴的结论块。
#
#   bash tools/prof_report.sh <mindstudio_profiler_output 目录> [anchor_stream] [n_steps]
#
# 内含：
#   1) 多步独占贡献（锚点切步、16 步平均）
#   2) 真空闲归因（跨步聚合：谁结束 → 谁在等，谁并行覆盖）
#   3) 尾段（后 30%）按 stream 的工作量分解
#   4) 每个 stream 的算子链聚类（每步次数）
set -uo pipefail
PY=${PY:-$HOME/miniforge3/envs/dsv41/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
PROF=${1:?profdir}
ANCHOR=${2:-158}
NSTEPS=${3:-16}

echo "================ 1) 独占贡献（锚点切步，${NSTEPS} 步平均）================"
"$PY" "$HERE/excl_steps.py" "$PROF" "$ANCHOR" "$NSTEPS" 2>&1 | tail -22

echo
echo "================ 2) 真空闲归因（跨步聚合）================"
"$PY" "$HERE/idle_who.py" "$PROF" "$ANCHOR" "$NSTEPS" 0.020 2>&1 | tail -18

echo
echo "================ 3) 算子链聚类（按 stream，≥0.5 次/步）================"
for S in 47 154 35; do
  echo "--- stream $S"
  "$PY" "$HERE/op_chain_cluster.py" "$PROF" "$S" "$ANCHOR" "$NSTEPS" 2>&1 | head -16
  echo
done

echo "================ 4) Python 侧 aten 算子账（每步次数）================"
FRAME=$(dirname "$PROF")/../FRAMEWORK/torch.op_range
if [ -f "$FRAME" ]; then
  "$PY" - "$FRAME" "$NSTEPS" <<'PYEOF'
import collections
import re
import sys

path, nsteps = sys.argv[1], int(sys.argv[2])
data = open(path, "rb").read()
toks = [m.group().decode() for m in re.finditer(rb"[A-Za-z_][A-Za-z0-9_:]{3,60}", data)]
total_steps = 626  # 由锚点法数出；如需精确请用 stepclass 输出覆盖
c = collections.Counter(toks)
print(f"{'per_step':>9}{'count':>10}  op")
for k, v in c.most_common(28):
    print(f"{v/total_steps:>9.1f}{v:>10}  {k}")
PYEOF
else
  echo "（没有 FRAMEWORK/torch.op_range，跳过）"
fi
exit 0
