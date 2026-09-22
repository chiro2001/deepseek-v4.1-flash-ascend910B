#!/usr/bin/env bash
# =============================================================================
# check_artifact_identity.sh —— 交付件「身份台账」的机械门
#
# 起因（2026-09-22）：`attention/dsa_v41.py` 在两小时内出现过 5 个 md5，
# 而"已过"的结论悄悄挂到了没跑过的文件上。见 `a2/publish/ARTIFACT-IDENTITY.md` §0。
#
# 用法：
#   bash a2/scripts/check_artifact_identity.sh            # 打印现盘 md5 + 台账对照
#   bash a2/scripts/check_artifact_identity.sh --strict   # 有"未在 PASS 臂上跑过"的件 ⇒ exit 2
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2=$(cd "$HERE/.." && pwd)
STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

# ★ 两种布局都要能跑（脱敏发布时目录名会变）：
#   工作区：a2/publish/*.py            a2/scripts/*.sh
#   发布仓： a2/patches/*.py（+ patches/kv8-graphsafe/）
if [ -d "$A2/publish" ]; then
    PL=publish; PATCHES=publish
else
    PL=patches; PATCHES=patches
fi

# 台账（见 publish/ARTIFACT-IDENTITY.md §1）：
#   路径|期望 md5|状态（PASS / 未确认 / 作废）
LEDGER=(
  # ★★ 2026-09-22 13:0x：档 C（`sg-c-c-graph-b`）与档 D（`sg-c-d-graph`）**都在这个 md5 上跑过且全绿**
  #    ⇒ 状态从"未确认"升为 PASS。见 publish/ARTIFACT-IDENTITY.md §1.1。
  "$PATCHES/kv8-graphsafe/dsa_v41.py|94aeebb757d6d5708268754481a05e0a|PASS"
  "$PATCHES/kv8-graphsafe/apply_graphsafe.py|4be07bea6cc3127eb8715a1da81f583a|PASS"
  "$PATCHES/kv8-graphsafe/adapt_runner.py|d8e8864ea60ccc7e92d5d82b9e7050af|PASS"
  "$PATCHES/kv8-graphsafe/patch_serve_sg.sh||PASS"
  "$PATCHES/0001-offload-scheduler.patch.py|79001c2671fdbdcd8386cd4684ed4761|PASS"
  "$PATCHES/0001-8card-offload-scheduler.patch.py|f3a7a0053fc6c639150fdde2a2509a63|PASS"
  "$PATCHES/0002-offload-cpu-pool-host-registered.patch.py|2c161a791fe99f17cce2e1139ffbdc3c|PASS"
  # ★ ②c 的补丁：单 die 三问已过（`054`）；★ 8 卡端到端在 c0 排队 ⇒ 标"未确认"（门会挡住 --strict 发布，符合事实）
  "$PATCHES/0004-draft-block64.patch.py|6d29845ea0d7abc432591d69db7fad17|未确认"
  "$PATCHES/0001b-offload-per-group-bpc-manager.patch.py|9f11c9ac0de0d77fbe6a212e42a9966a|PASS"
  "$PATCHES/0001c-offload-per-group-bpc-hooks.patch.py|af2fefb8337fdf9fe1c5e55518f665b8|PASS"
  # ★ 只在发布仓里有的历史件（见 ARTIFACT-IDENTITY.md §1.4）；工作区没有 ⇒ 用 [ -f ] 兜
  "$PATCHES/0003b-kv8-fuse-triton-kernels.py|6ce00b8f6fdd9ba4ad5935876601f8d6|PASS|optional"
  "$PATCHES/0003-kv8-fused-rebuild-triton.patch|27edecc675110511d520f9e9af74c6ed|PASS|optional"
  "scripts/a2_one_shot_probe.sh|40e495e83d198393ea544215dbc4fd50|PASS"
  "scripts/serve_a2_offload.sh||PASS"
)

echo "=============================================================="
echo "交付件身份台账检查    $([ "$STRICT" = 1 ] && echo '★ strict' || echo '（不带 --strict 只打印）')"
echo "  A2 目录 : $A2"
echo "=============================================================="

blocked=0
for row in "${LEDGER[@]}"; do
    IFS='|' read -r rel want state opt <<<"$row"
    f="$A2/$rel"
    if [ ! -f "$f" ]; then
        if [ "${opt:-}" = "optional" ]; then
            printf '·  %-56s （本布局没有，跳过 — 见 ARTIFACT-IDENTITY.md §1.4）\n' "$rel"
        else
            printf '✗  %-56s **文件不存在**\n' "$rel"
            blocked=1
        fi
        continue
    fi
    have=$(md5sum "$f" | awk '{print $1}')
    mark="✓"
    note=""
    if [ -n "$want" ] && [ "$want" != "$have" ]; then
        mark="⛔"
        note="  ← 台账写的是 $want"
        blocked=1
    fi
    case "$state" in
        PASS)   : ;;
        未确认) mark="⚠"; note="$note  ← ★ 未在任何 PASS 臂上跑过"; [ "$STRICT" = 1 ] && blocked=1 ;;
        作废)   mark="⛔"; note="$note  ← 已作废"; blocked=1 ;;
    esac
    printf '%s  %-56s %s  [%s]%s\n' "$mark" "$rel" "$have" "$state" "$note"
done

echo "-------------------------------------------------------------"
if [ -n "$(ls "$A2/$PATCHES"/*.py 2>/dev/null)" ]; then
    echo "★ $PATCHES/ 下**未登记**在本脚本 LEDGER 里的件（如果是新件，请补登记）："
    for f in "$A2/$PATCHES"/*.py "$A2/$PATCHES"/kv8-graphsafe/*.py; do
        [ -f "$f" ] || continue
        rel=${f#"$A2/"}
        found=0
        for row in "${LEDGER[@]}"; do
            [ "${row%%|*}" = "$rel" ] && found=1 && break
        done
        [ "$found" = 0 ] && echo "    … $rel  ($(md5sum "$f" | awk '{print $1}'))"
    done
fi

echo "-------------------------------------------------------------"
if [ "$blocked" != 0 ]; then
    echo "⛔ 有件「没有在 PASS 臂上跑过」或与台账不符。"
    echo "   处理：① 跑对应判据臂并在 arm.out 里记下 md5；② 更新 publish/ARTIFACT-IDENTITY.md 与本脚本 LEDGER。"
    echo "   ★ 规则：**发布件只允许取「某条 PASS 臂的 arm.out 里记过」的那个 md5**。"
    [ "$STRICT" = 1 ] && exit 2
    exit 0
fi
echo "✓ 全部对得上。"
