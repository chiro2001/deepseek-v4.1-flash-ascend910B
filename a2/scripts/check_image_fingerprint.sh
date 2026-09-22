#!/usr/bin/env bash
# =============================================================================
# check_image_fingerprint.sh —— **10 秒**回答：「我手上这个镜像，和当前发布包差哪些文件？」
#
# 为什么需要它：A2 默认 `PATCH_MODE=baked` ⇒ 容器里跑的是**镜像里烘焙的那份**补丁。
#   于是"我改好的补丁到底进没进镜像"这件事，如果靠"起服 → 等 30 分钟 → 压测"才知道，
#   成本和风险都太高（本日已栽过一次：`074` 的档 B 静默无卸载）。
#
# 它做什么（**只读**：只 `docker run` 一个临时容器跑 md5sum，不起服务、不占卡、不写任何东西）：
#   1. 由 `tools/check_checksums.py --emit-chk` 推出**当前发布包**期望的镜像内 md5；
#   2. 进镜像把同样这批路径的 md5 打出来；
#   3. 逐行对比，给出「一致 / 不一致 / 缺文件」，并把**关键两项**（engram_hash / engram_jit_kernel）
#      单独高亮 —— 它们是 ENGRAM×卸载 P0（`a2/logs/075`）的载体。
#
# 用法（在 dsv41-release 根目录）：
#   bash a2/scripts/check_image_fingerprint.sh                     # 默认查 dsv41-a2:v8
#   IMAGE=dsv41-a2:v9 bash a2/scripts/check_image_fingerprint.sh   # 查 v9
#   bash a2/scripts/check_image_fingerprint.sh local/xxx:tag       # 位置参数优先
#
# 退出码：0 = 全部一致；1 = 有差异；2 = 用法/环境错误（镜像不存在等）
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

IMAGE=${1:-${IMAGE:-dsv41-a2:v8}}
ASCEND_PKG=${ASCEND_PKG:-/vllm-workspace/vllm-ascend/vllm_ascend}

die() { echo "[imgfp] ✗ $*" >&2; exit 2; }

command -v docker >/dev/null 2>&1 || die "找不到 docker"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "本地没有镜像 $IMAGE（先 build，或检查名字；本脚本**不会**去 registry 拉取）"
[ -f "$REPO/tools/check_checksums.py" ] || die "缺 $REPO/tools/check_checksums.py（请在 dsv41-release 里跑）"

# ---------------------------------------------------------------- 1) 期望值
CHK=$(mktemp)
trap 'rm -f "$CHK"' EXIT
#
# --emit-chk 输出形如（缩进不定）：
#     inst models/deepseek_v41/engram_hash.py   240c5a04…  <- engram_hash.py
#   ⇒ 第 2 字段 = 镜像内相对路径，第 3 字段 = 期望 md5。只取 inst/newf 两类。
# ★ 该工具会把同一份清单**打印两遍**（一份带 `<- 来源` 后缀、一份不带）⇒ 必须 `sort -u` 去重，
#   否则后面 `for p in $paths` 会拿到重复项甚至错字段（2026-09-22 实测踩过）。
python3 "$REPO/tools/check_checksums.py" --emit-chk 2>/dev/null \
  | awk '/^[[:space:]]*(inst|newf)[[:space:]]/ {print $3 "\t" $2}' | sort -u > "$CHK" || true
[ -s "$CHK" ] || die "拿不到期望 md5（check_checksums.py --emit-chk 输出为空）"
_n=$(wc -l < "$CHK")

# ---------------------------------------------------------------- 2) 镜像内的值
# 一次容器启动取完，避免 N 次启动开销；路径不存在时 md5sum 会报错并返回非零，忽略即可。
#   ★ 只取第 2 字段（相对路径），并折成**空格分隔的一行** —— 换行会破坏 `for p in` 的语义。
_paths=$(cut -f2 "$CHK" | tr '\n' ' ')
_out=$(docker run --rm --entrypoint bash "$IMAGE" -lc "
  for p in $_paths; do
    if [ -f \"$ASCEND_PKG/\$p\" ]; then md5sum \"$ASCEND_PKG/\$p\"; else echo 'MISSING  $ASCEND_PKG/'\$p; fi
  done" 2>/dev/null)
[ -n "$_out" ] || die "容器里读不到任何文件（ASCEND_PKG=$ASCEND_PKG 对不对？）"

echo "=============================================================="
echo "镜像指纹对比"
echo "  镜像        : $IMAGE"
echo "  包内期望    : $REPO（$_n 项）"
echo "  镜像内根    : $ASCEND_PKG"
echo "=============================================================="

# ---------------------------------------------------------------- 3) 对比
rc=0; same=0; diff=0; miss=0
while read -r want path; do
    got=$(printf '%s\n' "$_out" | awk -v p="$ASCEND_PKG/$path" '$2==p {print $1; exit}')
    if [ -z "$got" ]; then
        printf '  MISSING  %s\n' "$path"; miss=$((miss+1)); rc=1
    elif [ "$got" = "$want" ]; then
        printf '  ✓        %s\n' "$path"; same=$((same+1))
    else
        printf '  ✗ 不一致  %s\n             镜像=%s\n             包内=%s\n' "$path" "$got" "$want"
        diff=$((diff+1)); rc=1
    fi
done < "$CHK"

echo "--------------------------------------------------------------"
echo "  一致 $same / 不一致 $diff / 缺文件 $miss"

# ---------------------------------------------------------------- 4) 关键两项高亮
echo "--------------------------------------------------------------"
_k1=$(printf '%s\n' "$_out" | awk -v p="$ASCEND_PKG/models/deepseek_v41/engram_hash.py" '$2==p{print $1}')
_k2=$(printf '%s\n' "$_out" | awk -v p="$ASCEND_PKG/models/deepseek_v41/engram_jit_kernel.py" '$2==p{print $1}')
echo "★ ENGRAM×卸载 P0 修复（a2/logs/075）载体："
echo "    engram_hash.py        镜像=${_k1:-<缺>}"
echo "    engram_jit_kernel.py  镜像=${_k2:-<缺>}"
if [ "$_k1" = "240c5a0444a1434a33d80341ea50e1d3" ] && [ "$_k2" = "6668d3fe3c6333b47e9dde3df7cfe03a" ]; then
    echo "    ⇒ ✅ 带修复（ENGRAM=1 + 卸载 可以起服）"
else
    echo "    ⇒ ⛔ 不带修复：ENGRAM=1 + 卸载 会在 replay 轮 KeyError(2486) 引擎死（logs/073）"
    echo "       修法：IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh 然后用 IMAGE=dsv41-a2:v9 起服"
fi
echo "=============================================================="
exit $rc
