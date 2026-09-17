#!/usr/bin/env bash
# =============================================================================
# 选卡助手（只读）—— 打印每张 NPU 的占用情况与进程属主，供决定 DEVS=...
#
#   bash tools/list_chips.sh              # 全量
#   bash tools/list_chips.sh --free        # 只输出空闲卡的 device 号（可直接当 DEVS 候选）
#
# 判据：以 **npu-smi 进程表**为准（无进程 = 空闲）；HBM 用量一并显示供参考。
#       另单独提示"无进程但 HBM 仍被占着"的卡（多为别人退出的残留/驱动预留）。
# 该脚本**不做任何写操作**，不碰 docker、不碰进程。
# =============================================================================
set -uo pipefail

MODE=${1:-}

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "找不到 npu-smi（本脚本要在 NPU 宿主机上跑）" >&2
  exit 1
fi

RAW=$(npu-smi info 2>/dev/null)

# npu-smi 输出用 '|' 分列。两种行：
#   信息行  | <a> <b> | <bus-id> | AICore%  Mem  used/total(MB)  HBM used/total |   -> NF=5
#   进程行  | <npu> <chip> | <pid> | <name> | <mem MB> | <pid in container> |      -> NF=6
#
# ⚠️ 两台机器的列语义不同（实测）：
#   A3-node1: 信息行是 (chip%2, 全局chip)  —— 如 "0 8" 表示全局第 8 个 chip
#   A3-node2: 信息行是 (NPU, 该NPU内chip)  —— 如 "4 0" 表示 NPU4 的 chip0
#   共同点：**枚举第 k 行就是 davinci k**。所以信息表只取序号，不解析列语义。
#   进程表的 (npu, chip) 两台一致：device = npu*2 + chip。

# --- 1) 每个 chip 的 HBM 用量（序号 = device 号） ---
HBM=$(printf '%s\n' "$RAW" | awk -F'|' '
  NF>=5 && $3 ~ /[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]/ {
    s=$4; gsub(/[^0-9\/]/,"",s)                     # 只留数字与斜杠
    n=split(s, p, "/")
    if (n >= 2) printf "%d\t%d\t%d\n", seq, p[n-1]+0, p[n]+0
    seq++
  }')

# --- 2) 进程行 ---
PROC=$(printf '%s\n' "$RAW" | awk -F'|' '
  NF>=6 && $3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
    split($2, a, /[[:space:]]+/); npu=a[2]+0; chip=a[3]+0
    pid=$3; gsub(/[^0-9]/,"",pid)
    name=$4; gsub(/^[[:space:]]+|[[:space:]]+$/,"",name)
    mem=$5; gsub(/[^0-9]/,"",mem)
    printf "%d\t%s\t%s\t%s\n", npu*2+chip, pid, name, mem
  }')

# --- 3) 给每个 pid 找属主 ---
owner_of() {
  local pid=$1 u
  u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$u" ] || u="<已退出或不可见>"
  printf '%s' "$u"
}

VER=$(printf '%s\n' "$RAW" | awk 'NR==2{print $3}')
echo "=================== NPU 概览（npu-smi $VER）==================="
echo "  device = 你在 DEVS / ASCEND_RT_VISIBLE_DEVICES 里用的编号"
echo
printf '%-8s %-11s %-7s %s\n' "device" "HBM(MB)" "占用" "进程（pid/属主/名称/内存MB）"
echo "-------------------------------------------------------------------------------------"

FREE_LIST=""
RESIDUAL=""
while IFS=$'\t' read -r dev used total; do
  [ -n "${dev:-}" ] || continue
  procs=$(printf '%s\n' "$PROC" | awk -v d="$dev" -F'\t' '$1==d {print}')
  detail=""; busy="空闲"
  if [ -n "$procs" ]; then
    busy="占用"
    while IFS=$'\t' read -r _d pid name mem; do
      [ -n "${pid:-}" ] || continue
      detail="$detail ${pid}/$(owner_of "$pid")/${name}/${mem}MB"
    done <<<"$procs"
  fi
  printf '%-8s %-11s %-7s %s\n' "$dev" "$used" "$busy" "${detail:-（无进程）}"
  if [ "$busy" = "空闲" ]; then
    FREE_LIST="$FREE_LIST $dev"
    [ "$used" -ge 4096 ] 2>/dev/null && RESIDUAL="$RESIDUAL $dev（HBM ${used}MB 未释放）"
  fi
done <<<"$HBM"

echo
echo "=================== 空闲（npu-smi 进程表里无进程）==================="
if [ -n "$FREE_LIST" ]; then
  echo "DEVS 候选：$FREE_LIST"
  _n=$(printf '%s\n' $FREE_LIST | wc -l)
  echo "（共 $_n 张；本服务默认 TP=8，需要 8 张）"
  [ -n "$RESIDUAL" ] && {
    echo
    echo "⚠️ 以下卡无进程但 HBM 仍被占着（可能是别人退出的残留或驱动预留，起服前留意）："
    printf '   %s\n' $RESIDUAL
  }
  if [ "$_n" -ge 8 ]; then
    _first8=$(printf '%s\n' $FREE_LIST | head -8 | tr '\n' ' ')
    echo
    echo "可直接用："
    echo "  DEVS=\"${_first8% }\" MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/serve_a3.sh"
  fi
else
  echo "（没有无进程的卡；请查看上面的占用详情）"
fi

if [ "$MODE" = "--free" ]; then
  echo
  printf '%s\n' $FREE_LIST | grep -v '^$' || true
fi
