#!/usr/bin/env bash
# 边界测试结束后自动接棒：记录结果 → 恢复 TP8 交付实例（armF 基线）→ 跑 armH A/B。
#
# 为什么这样做：单次起服 ~20 min，人工等会白占 8 die。本脚本把"等 → 起服 → 基准"串起来，
# 并**先**把边界测试的判决落盘（避免它被后续日志淹没）。
#
# 用法: nohup bash tools/autochain_after_boundary.sh > ~/tmp/autochain.log 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "[autochain $(date +%H:%M:%S)] $*"; }

say "等 boundary 压测结束（每 30 s 探一次）…"
while pgrep -f "boundary_n2.py|boundary_n.py" >/dev/null 2>&1; do sleep 30; done
sleep 10

say "边界测试已结束。结果摘录："
tail -30 "$HOME/tmp/boundary16.log" | sed 's/^/    /'

say "开始 armF 基线（含 profile + 验收）"
bash "$HERE/run_arm_suite.sh" "$HOME/tmp/launch_armF.sh" armF_r6_base 1 1 1

say "开始 armH（IDS64_HOIST=1 + PAD_SKIP=1）"
bash "$HERE/run_arm_suite.sh" "$HOME/tmp/launch_armH.sh" armH_r6_flags 1 1 1

say "配对对比（armF vs armH）："
/home/l00886679/miniforge3/envs/dsv41/bin/python "$HERE/bench_delta.py" \
  "$HOME/tmp/armF_r6_base_bench.json" "$HOME/tmp/armH_r6_flags_bench.json" \
  --label-a armF_base --label-b armH_flags || true

say "检查 PGO 产物是否能在 A3 镜像里加载（不合格就跳过该臂）"
if docker run --rm --entrypoint bash \
     -v "$HOME/cedpd-repo/optim/pgo/libpython3.12.so.1.0:/usr/local/python3.12.13/lib/libpython3.12.so.1.0:ro" \
     quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
     -lc "python3 -c 'import sys; print(\"PYOK\", sys.version.split()[0])'" 2>/dev/null | grep -q PYOK; then
  say "PGO 产物可加载 ⇒ 开始 armP"
  bash "$HERE/run_arm_suite.sh" "$HOME/tmp/launch_armP.sh" armP_r6_pgo 1 1 1
  say "配对对比（armF vs armP）："
  /home/l00886679/miniforge3/envs/dsv41/bin/python "$HERE/bench_delta.py" \
    "$HOME/tmp/armF_r6_base_bench.json" "$HOME/tmp/armP_r6_pgo_bench.json" \
    --label-a armF_base --label-b armP_pgo || true
else
  say "⛔ PGO 产物在 A3 镜像里加载失败（glibc/ABI 不匹配）⇒ 跳过 armP，需用 A3 镜像重编"
fi

say "全部完成"
