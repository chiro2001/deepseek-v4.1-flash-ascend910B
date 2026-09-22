#!/usr/bin/env bash
# [S_graphfix] 往影子包的 `serve_a2.sh` 里加**一行** export（幂等，默认关）：
#   export VLLM_V41_KV8_GRAPH_SAFE='${R8_GRAPH_SAFE:-0}'
# 位置：紧跟在 [R8-INT8] 的运行期开关块之后（那块由 R_8card_int8 的
# `patch_serve_a2_int8.sh` 写入；本脚本必须先跑它）。
# ★ 默认 0 ⇒ 对任何不设 R8_GRAPH_SAFE 的臂**零影响**（含 R 正在跑的臂）。
set -euo pipefail
PKG=${PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
A="$PKG/scripts/serve_a2.sh"
[ -f "$A" ] || { echo "缺 $A" >&2; exit 2; }

if grep -q "SG_TRACE_PPR" "$A"; then
  echo "[patch_serve_sg] 已经打过（SG_TRACE_PPR 在）"
  grep -n "VLLM_V41_KV8_GRAPH_SAFE\|SG_CMP_LEGACY\|SG_TRACE_PPR" "$A"
  exit 0
fi
ANCHOR="export R8_APC_TRACE_LIMIT='\${R8_APC_TRACE_LIMIT:-40}'"
grep -qF "$ANCHOR" "$A" || { echo "[patch_serve_sg] 缺锚点（先跑 R 的 patch_serve_a2_int8.sh）" >&2; exit 3; }
python3 - "$A" <<'PY'
import pathlib, sys
a = pathlib.Path(sys.argv[1])
sa = a.read_text()
anchor = "export R8_APC_TRACE_LIMIT='${R8_APC_TRACE_LIMIT:-40}'\n"
add = (
    "# [S_graphfix] int8 KV8 读侧的图兼容路径（默认关；见 a2/logs/049）。\n"
    "#   把 decode（含 spec-decode）的 KV8 重建改成 host 上界驱动 ⇒ 图捕获期无 D2H。\n"
    "export VLLM_V41_KV8_GRAPH_SAFE='${R8_GRAPH_SAFE:-0}'\n"
    "#   SG_CMP_LEGACY=1 = 诊断臂：只修窗口(SWA)面，long-KV 面留在捕获期路径上\n"
    "#   （用来实测『捕获期冻结的页数』是响亮失败还是静默读错）。默认 0。\n"
    "export SG_CMP_LEGACY='${SG_CMP_LEGACY:-0}'\n"
    "#   SG_TRACE_PPR=1 = 只读探针：打印捕获期/replay 的 mcs 与 ppr（热路径，限 24 行）。\n"
    "export SG_TRACE_PPR='${SG_TRACE_PPR:-0}'\n"
)
assert sa.count(anchor) == 1, f"锚点出现 {sa.count(anchor)} 次"
sa = sa.replace(anchor, anchor + add, 1)
for mark in (
    "export VLLM_V41_KV8_GRAPH_SAFE=",
    "export SG_CMP_LEGACY=",
    "export SG_TRACE_PPR=",
):
    assert sa.count(mark) == 1, f"{mark} 出现 {sa.count(mark)} 次"
a.write_text(sa)
print("[patch_serve_sg] OK：已插入 3 行 export")
PY
grep -n "VLLM_V41_KV8_GRAPH_SAFE\|SG_CMP_LEGACY\|SG_TRACE_PPR" "$A"
