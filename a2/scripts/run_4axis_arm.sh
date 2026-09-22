#!/usr/bin/env bash
# =============================================================================
# run_4axis_arm.sh —— ★ 四轴同开（A3 8 卡）的**一次性启动脚本**，自带三道前置门
#
# 目标（用户目标的第一句）：把四个指标**同时**使能并跑通
#   ① ENGRAM=1（硬约束）② DRAM 卸载 ③ int8 KV 档 C ④ draft 入图
#
# 为什么需要这个包装：这三道门**今天各踩了一次**，任何一道漏掉都会白烧 20–40 分钟
#   G1 合并件新鲜度（a2/logs/081）—— 派生件过期 = 功能静默消失 / bugfix 静默回退
#   G2 dsa 必须指向 graphsafe 包（a2/logs/082）—— 否则捕获期 EE1016 必炸
#   G3 先看 dmesg（a2/logs/080）—— OOM 会伪装成"代码挂了"，把有效臂误判成失败
#
# 用法（在 A3-node1 上）
#   bash run_4axis_arm.sh                 # 用默认值起臂
#   TAG=my-4axis bash run_4axis_arm.sh    # 自定义臂名
#   DRY=1 bash run_4axis_arm.sh           # ★ 只跑三道门 + 打印将执行的命令，**不起臂**
#
# 退出码：0 = 门全过（并已开始起臂）；64 = 某道门没过（**不要**记成判据失败）
# =============================================================================
set -uo pipefail

ROOT=${ROOT:-$HOME/projects/dsv41-upstream-pr}
R8=${R8:-$ROOT/agents/R_8card_int8}
S=${S:-$ROOT/agents/S_graphfix}
X=${X:-$ROOT/agents/X_integrate}
PKG=${PKG:-$ROOT/shadow-pkg}

# ---- 四轴的配置（可用环境变量覆盖）----
TAG=${TAG:-r8-4axis}
TIER=${TIER:-C}                       # 档 C = SWA int8 + ring16
GRAPH=${GRAPH:-1}
EAGER=${EAGER:-0}
ENGRAM=${ENGRAM:-1}                   # ★ 硬约束
DRAFT_GRAPH=${DRAFT_GRAPH:-1}
# ★★★ 2026-09-22 23:3x **必修（配置）**：必须显式 `ENGRAM_DEVICE_INDEX=0`
#   ① A2 生产就是 0（用户 09-20/09-21 的启动命令）；`a2/scripts/serve_a2_offload.sh` 也默认 0
#   ② 更关键：`_prepare_engram_device()` **不调用** `self.engram_history.update()`
#      ⇒ device-index 一开，**`075`/`077` 修的整条 host 路径（含 pageless 修复、
#         TRUE_TOKENS 精确修补、mismatch 计数）全被绕过** ⇒ 跑出来的"通过"**不代表**
#         我们要验收的那条路（典型"判据没覆盖需求"）
#   ③ `logs/069` 实测：A3 上 device-index 打开 ⇒ Engram 表注册 183 GiB ⇒ `EH0012` + 池拿不到预算
#   实测对照（同一天的两条臂）：`p3b2` 是 DEVICE-INDEX=0 ⇒ `ENGRAM-TRUE-TOKENS` 8 行；
#                                `r8-4axis`（未设 ⇒ auto）是 **1** ⇒ `ENGRAM-TRUE-TOKENS` **0 行**
ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
OFFLOAD_BYTES=${OFFLOAD_BYTES:-23068672000}   # ≈21.5 GiB 记账（够触发取回，又留内存余量，见 logs/080）
MAX_TOKENS=${MAX_TOKENS:-128}         # ★ 128 才有有效的投机读数（mt=1 时 A 恒 ~1.5）
PROMPTS=${PROMPTS:-16}
PROMPT_TOKENS=${PROMPT_TOKENS:-131072}
REPLAY_PROMPT_TOKENS=${REPLAY_PROMPT_TOKENS:-65536}
# ★★★ 2026-09-22 23:1x：**必须 3 轮**（`logs/083 §2`）。
#   跨运行的 sha 比对已证无效（同配置两次独立运行 sha 就不同）⇒ 唯一可用的
#   "逐字可复现"判据是**同运行内**的最后两轮（两轮都走池取回），那需要 rounds>=3。
#   轮次语义（bench/kv_offload_client.py:348-404）：rounds[0]=fill，rounds[1..]=每轮 reset 后的重放。
ROUNDS=${ROUNDS:-3}
PORT=${PORT:-8050}

# ---- G1/G2 用的输入路径 ----
IMG_MODEL=${IMG_MODEL:-$HOME/tmp/20260922/merge4axis/model.img.py}
KV8_MODEL=${KV8_MODEL:-$X/pkg-kv8pf/shadow/vllm_ascend/models/deepseek_v41/model.py}
PROD_MODEL=${PROD_MODEL:-$PKG/patches/files/model.py}
INSTALLED_MERGED=${INSTALLED_MERGED:-$R8/patched/model_merged.py}
MERGER=${MERGER:-$R8/patch/merge_model.py}
GRAPHSAFE_DSA=${GRAPHSAFE_DSA:-$S/pkgs/pkg-kv8pf/shadow/vllm_ascend/attention/dsa_v41.py}

DRY=${DRY:-0}
die() { echo "" >&2; echo "⛔ [4axis] $*" >&2; echo "   ⇒ 这是**前置门失败**，不是判据失败；修完再起，别把这一臂记进对照表。" >&2; exit 64; }
say() { echo "[4axis] $*"; }

echo "=============================================================="
echo "四轴同开启动脚本（ENGRAM=1 × 卸载 × 档C int8 × draft入图）"
echo "  TAG=$TAG TIER=$TIER GRAPH=$GRAPH ENGRAM=$ENGRAM DRAFT_GRAPH=$DRAFT_GRAPH"
echo "  ★ ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX （必须 0：否则 host 路径整段被绕过，见脚本头注释）"
echo "  OFFLOAD_BYTES=$OFFLOAD_BYTES ($((OFFLOAD_BYTES/1073741824)) GiB)"
echo "=============================================================="

# ---------------------------------------------------------------- G1 合并件新鲜度
say "G1 合并件新鲜度（logs/081）"
for f in "$MERGER" "$IMG_MODEL" "$KV8_MODEL" "$PROD_MODEL" "$INSTALLED_MERGED"; do
  [ -f "$f" ] || die "G1: 缺文件 $f"
done
_cf=$(ls "$HOME"/projects/dsv41-release/a2/scripts/check_merged_fresh.sh \
        "$HOME"/tmp/20260922/merge4axis/scripts/check_merged_fresh.sh 2>/dev/null | head -1)
[ -n "$_cf" ] || die "G1: 找不到 check_merged_fresh.sh（从发布仓 a2/scripts/ 取）"
if bash "$_cf" --img "$IMG_MODEL" --kv8 "$KV8_MODEL" --prod "$PROD_MODEL" \
        --installed "$INSTALLED_MERGED" --merger "$MERGER" >/dev/null 2>&1; then
  say "G1 ✓ 合并件新鲜：$(md5sum "$INSTALLED_MERGED" | cut -d' ' -f1)"
else
  bash "$_cf" --img "$IMG_MODEL" --kv8 "$KV8_MODEL" --prod "$PROD_MODEL" \
      --installed "$INSTALLED_MERGED" --merger "$MERGER" >&2
  die "G1: 合并件已过期（上面有差异明细）"
fi

# ---------------------------------------------------------------- G2 graphsafe 接线
say "G2 dsa 必须指向 graphsafe 包（logs/082）"
[ -f "$GRAPHSAFE_DSA" ] || die "G2: 缺 graphsafe dsa：$GRAPHSAFE_DSA"
_rb=$(grep -c "rows_bound" "$GRAPHSAFE_DSA")
_gs=$(grep -c "VLLM_V41_KV8_GRAPH_SAFE" "$GRAPHSAFE_DSA")
_md=$(md5sum "$GRAPHSAFE_DSA" | cut -d' ' -f1)
if [ "$_rb" -ge 1 ] && [ "$_gs" -ge 1 ]; then
  say "G2 ✓ graphsafe dsa：md5=$_md rows_bound=$_rb 开关=$_gs"
else
  # ★ 必须用**精确模式**判（`rows_bound` 单独数）—— `max_query_len` 在非 graphsafe 文件里
  #   本来就有 4 处，混着 grep 会得出"已经修了"的假结论（logs/082 §1 的判据口径教训）。
  die "G2: $GRAPHSAFE_DSA 看起来不是 graphsafe 版（rows_bound=$_rb 开关=$_gs，期望都 >=1）"
fi
say "G2 ✓ 起臂时会带：DSA_SRC=D R8_KV8_DIR_D=$S/pkgs/pkg-kv8pf（⇒ shadow 打印 'dsa=D=带 role 分键'）"

# ---------------------------------------------------------------- G3 dmesg（OOM）
say "G3 先看 dmesg（logs/080：OOM 会伪装成代码挂了）"
_oom=$( { sudo -n dmesg -T 2>/dev/null || dmesg -T 2>/dev/null; } | grep -icE "killed process" || true)
_avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
say "  近端 dmesg 里 'Killed process' 条数 = ${_oom:-0}（>0 只说明**历史上**发生过，要看时间戳是不是本次）"
say "  MemAvailable = ${_avail} GiB（建议 >= 1400 再起，见 logs/080 §4.2）"
if [ "${_avail:-0}" -lt 900 ]; then
  die "G3: 可用内存只有 ${_avail} GiB —— 8 卡 + Engram 表 + 池很可能 OOM。等邻居闲下来或调小 OFFLOAD_BYTES"
fi

# ---------------------------------------------------------------- 起臂
CMD=(env TAG="$TAG" TIER="$TIER" GRAPH="$GRAPH" EAGER="$EAGER"
     ENGRAM="$ENGRAM" DRAFT_GRAPH="$DRAFT_GRAPH"
     ENGRAM_DEVICE_INDEX="$ENGRAM_DEVICE_INDEX"
     OFFLOAD_BYTES="$OFFLOAD_BYTES" MAX_TOKENS="$MAX_TOKENS"
     DSA_SRC=D R8_KV8_DIR_D="$S/pkgs/pkg-kv8pf"
     PROMPTS="$PROMPTS" PROMPT_TOKENS="$PROMPT_TOKENS"
     REPLAY_PROMPT_TOKENS="$REPLAY_PROMPT_TOKENS" ROUNDS="$ROUNDS"
     bash "$R8/scripts/run_arm_r8.sh")
echo "--------------------------------------------------------------"
say "三道门全过。将执行："
printf '   %s\n' "${CMD[*]}"
echo "--------------------------------------------------------------"
if [ "$DRY" = "1" ]; then
  say "DRY=1 ⇒ 只跑到这里（不含起臂）"
  exit 0
fi
echo ""
say "★ 起服期必须看到这四条（缺一就停，别等压测）："
say "   1) [R8-INT8] ... dsa=D=带 role 分键   （不是 dsa=C=原样）"
say "   2) KV8_GRAPH_SAFE=1"
say "   3) model.py 用**合并版**（md5 应为 $(md5sum "$INSTALLED_MERGED" | cut -d' ' -f1)）"
say "   4) 捕获期 EE1016 = 0"
say "   ★★ 5) 容器内 ENGRAM_DEVICE_INDEX 必须是 0，且日志里 **不应** 出现 [DEVICE-INDEX]："
say "        docker exec <ctr> sh -c 'env | grep ENGRAM_DEVICE'" 
say "        grep -ac 'DEVICE-INDEX' <serve.log>    # ★ 必须 0（=1 说明走了 device 路径，host 路径被绕过）"
say "        grep -ac 'ENGRAM-TRUE-TOKENS' <serve.log>  # ★ 应 >0（证明修补代码真的在跑）"
say "★ 压测后跑自然语言判据（验收标准的一条）："
say "   python3 a2/scripts/text_correctness_probe.py --base-url http://127.0.0.1:${PORT} --model deepseek-v41 --out <证据>"
say "★ 以及**同运行内**的逐字可复现判据（替代跨运行 sha，见 logs/083 §2）："
say "   python3 a2/scripts/check_same_run_replay.py $HOME/projects/dsv41-upstream-pr/agents/R_8card_int8/out/$TAG/*.client.json"
say "★ 最后用**验收判决器**一次判目标里那 7 条（缺证据的项会标'未验'，不算通过）："
say "   python3 a2/scripts/check_4axis_acceptance.py \\"
say "     --log <serve.log> \\"
say "     --client $HOME/projects/dsv41-upstream-pr/agents/R_8card_int8/out/$TAG/*.client.json \\"
say "     --metrics $HOME/projects/dsv41-upstream-pr/agents/R_8card_int8/out/$TAG/*.metrics_after.txt \\"
say "     --container $TAG --text-probe-json <textprobe.json>"
echo ""
exec "${CMD[@]}"
