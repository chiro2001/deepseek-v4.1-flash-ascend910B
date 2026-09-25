#!/usr/bin/env bash
# 跑一个完整臂：起新鲜池的 D（C 块）→ 桥请求 → 同长度序列精确停在池顶 → 出判据表。
#
#   bash tools/ced_knifeedge_arm.sh <TAG> <C> <BRIDGE_N> <UNIFORM_N> <STOP_MAX>
#
# 设计依据（A1/B1 实测）：新鲜池 + 同长度大请求序列的 max 逐条精确推进
# stride = c(n)+21，c(n)=ceil((n-1)/128)；先发一个不同长度的桥把相位对齐，
# 就能让第 K 条请求**恰好**落在池顶 C-1。序列里所有请求 prompt 相同，
# 只有块号不同 ⇒ 单变量对照。
set -uo pipefail

PKG=${PKG:-$HOME/projects/dsv41-ced-singlechip/pkg-swa-clip}
TOOLS=${TOOLS:-$PKG/tools}
TAG=${1:?用法: $0 <TAG> <C> <BRIDGE_N> <UNIFORM_N> <STOP_MAX>}
C=${2:?}
BRIDGE_N=${3:?}
UNIFORM_N=${4:?}
STOP_MAX=${5:?}
# 默认给足条数：序列会在 max 触到 stop_max 时自动停，条数给大不会多跑。
COUNT=${COUNT:-250}

cd "$PKG" || exit 1
OUT="$PKG/results/${TAG}_d"
# 注意：绝不能 rm -rf "$OUT" —— 调用方的 stdout 日志就重定向在这个目录里，
# 删目录会把日志变成"已 unlink 的 inode"（踩过一次）。只清本臂的两个产物。
mkdir -p "$OUT"
rm -f "$OUT/series.jsonl"

echo "[arm $TAG] 起 D（C=$C，池顶=$((C-1))）"
# 该机 /home 在高负载下偶发路径查找失败 ⇒ 所有脚本一律用绝对路径调用。
bash "$TOOLS/ced_knifeedge_launch.sh" start-d "$TAG" "$C" > "$OUT/start.log" 2>&1 || {
  echo "[arm $TAG][FAIL] 起服失败"; tail -20 "$OUT/start.log"; exit 3; }
grep -E "num_blocks=" "$OUT/serve.log" | tail -1 | sed -E 's/^.*\[CED-KVGEOM\]/[KVGEOM]/'

export TMPDIR=/home/l00886679/tmp/knifeedge
mkdir -p "$TMPDIR"

if [ "$BRIDGE_N" != "0" ]; then
  echo "[arm $TAG] 桥请求 n=$BRIDGE_N"
python3 "$TOOLS/ced_knifeedge_cprobe.py" --ns "$BRIDGE_N" \
    --dump-dir "$OUT/blockdump" --out "$OUT/series.jsonl" 2>/dev/null | tail -1
else
  echo "[arm $TAG] 无桥（纯同长度序列，落点更干净）"
  : > "$OUT/series.jsonl"
fi

echo "[arm $TAG] 同长度序列 n=$UNIFORM_N，stop_max=$STOP_MAX"
python3 "$TOOLS/ced_knifeedge_walk.py" series \
  --n "$UNIFORM_N" --count "$COUNT" --stop-max "$STOP_MAX" --overhead 21 \
  --controls 2 --controls-n "$UNIFORM_N" \
  --dump-dir "$OUT/blockdump" --out "$OUT/series.jsonl" --dump-timeout 90 2>&1 | tail -6

echo "[arm $TAG] 判据表"
python3 "$TOOLS/ced_knifeedge_report.py" --walk "$OUT/series.jsonl" --pool-blocks "$C" 2>&1 | tail -12
echo "[arm $TAG] 完成：$OUT/series.jsonl"
