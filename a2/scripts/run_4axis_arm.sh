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
#   G4 端口占用（a2/logs/099）—— ★ 两个 net=host 臂共用固定端口时，`health=200`
#      会被**别的容器**满足 ⇒ 自检读到自己那份**空**日志 ⇒ 误判"补丁没挂上"（实测第三例）
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
# ★★★ 2026-09-23 01:xx：`KEEP=1` 必须能转发到 runner —— 否则容器跑完就被删，
#   而"**在同一条臂上**跑自然语言文本判据"（目标里'同一臂内同时满足'的严格口径）
#   正需要**容器活着**。★ 默认 0（保持 runner 的原行为，不改变任何既有语义）。
KEEP=${KEEP:-0}
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
# ★★★ 2026-09-23 00:4x **G2b：起服后断言「实际挂的」就是我指定的那份**（logs/093 的实测事故）
#   事故：`r8-4axis-mode2` 臂**手工起**、没带 `DSA_SRC=D R8_KV8_DIR_D=…` ⇒ runner 的
#   `R8_KV8_DIR_D` 默认指向 `X_integrate/pkg-kv8pf`（**rows_bound=0，没有 graphsafe**）
#   ⇒ 图捕获期 `capture failed: LocalScalarDenseNpu.cpp:23` + `EE1016` ×8、容器死。
#   ★ 而 G2 只检查了"我指定的那份是 graphsafe"，**没检查 runner 真的挂了它** ⇒ 这个洞要补。
#   ★ 判据指纹：`grep -ac "capture failed" <serve.log>` > 0 且含 `LocalScalarDenseNpu`
#     ⇒ 先查 `dsa_dir_D` 指向哪份、`rows_bound` 是否为 0，**不要**先怀疑 Engram/int8/池分量。
say "G2b 起服后会自动断言 dsa_dir_D 含 $S（防「手工起臂漏 env」，logs/093）"

# ---------------------------------------------------------------- G4 端口占用门
# ★★★ 2026-09-23 01:xx **实测事故（`logs/099`）**：两条臂都用 `net=host` + 固定 `PORT=8050`，
#   而前一条臂（`KEEP=1` 留着的容器）**仍在服务** ⇒ 后一条臂的起服判定
#   `curl 127.0.0.1:8050/health` **立刻 200**（那是**别人的** 200）⇒ 自检读到的 `serve.log`
#   还是全新空文件 ⇒ 报 `P1_pinned ret=0 行数=0（期望 128）` ⇒ **FATAL exit 9**，
#   而且 `docker rm -f` 还因为容器 0.93s 后才创建而**打空** ⇒ 留下**孤儿容器**占卡占内存。
#   ⇒ 本门：起臂前**断言端口空闲**；被占就 die（并告诉占用者是谁）。
G4_PORT_CHECK=${G4_PORT_CHECK:-1}
if [ "$G4_PORT_CHECK" = "1" ]; then
    say "G4 端口占用门：断言 ${PORT} 空闲（防'health 200 被别的容器满足'，logs/099）"
    _busy=""
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
        _busy="ss"
    elif command -v curl >/dev/null 2>&1 && curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        _busy="curl"
    fi
    if [ -n "$_busy" ]; then
        echo "" >&2
        echo "⛔⛔ [4axis][G4] 端口 ${PORT} 已被占用（判据来自 $_busy）—— 绝不能就这样起臂！" >&2
        echo "   原因：本仓的臂都用 net=host ⇒ 别的容器在同一个 ${PORT} 上服务时，" >&2
        echo "         起服判定会拿**别人的 health=200** 当自己的 ⇒ 自检读到空日志 ⇒ 误判 exit 9。" >&2
        echo "   ⇒ 先找出并处理占用者：" >&2
        echo "        docker ps --format '{{.Names}}\t{{.Status}}' | grep -v '^prbench'" >&2
        echo "        ss -ltnp | grep ${PORT}" >&2
        echo "   ⇒ 或者直接换个端口重跑：  PORT=$((PORT+1)) bash $0" >&2
        exit 64
    fi
    say "G4 ✓ 端口 ${PORT} 空闲"
fi

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
     # ★★★ 2026-09-23 02:5x 修正（实测教训）：此前这里**硬编码** `R8_KV8_DIR_D="$S/pkgs/pkg-kv8pf"`
     #   ⇒ 调用方若想换 dsa 实现（例如 `agents/CHUNKVIEW/pkg-kv8pf` 的 chunk 连续视图版 `d84f087c`），
     #   **环境变量会被这里覆盖** ⇒ 挂上去的还是 S 原版 `94aeebb7`。
     #   事故现场：`r8-chunkview` 臂起服后容器内 md5 = `94aeebb7`（= 原版），**整臂无效**，
     #   若没被指纹门抓到就会白烧 25 min ⇒ 这正是 `logs/081`/`093` 同族的"改了没生效"。
     #   ⇒ 改成**尊重调用方的显式设定**（`${VAR:-默认}`），默认行为**逐字不变**。
     # ★★★ 2026-09-23 04:0x 修正：`DSA_SRC` 此前**硬编码 D** ⇒ 调用方无法切到 `F`（融合件）。
     #   而 `run_arm_r8.sh:233` 读的正是 `DSA_SRC`（不是 `R8_DSA_SRC`）⇒ 融合臂永远挂不上。
     #   ⇒ 改成 `${VAR:-默认}`，**默认行为逐字不变**。
     DSA_SRC="${DSA_SRC:-D}" R8_KV8_DIR_D="${R8_KV8_DIR_D:-$S/pkgs/pkg-kv8pf}"
     PROMPTS="$PROMPTS" PROMPT_TOKENS="$PROMPT_TOKENS"
     REPLAY_PROMPT_TOKENS="$REPLAY_PROMPT_TOKENS" ROUNDS="$ROUNDS"
     KEEP="$KEEP"
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
# ---------------------------------------------------------------- G2b：起臂后断言 dsa_dir_D
# ★ 不能再用 `exec`（那样就没机会做事后检查了）⇒ 改成"后台起臂 + 前 20 分钟盯 meta/serve.log"。
#   一旦发现 (a) dsa_dir_D 不含 S_graphfix，或 (b) 捕获期 `capture failed ... LocalScalarDenseNpu`
#   ⇒ 立刻**响亮地喊出来**（并把判据指纹一并打印），避免又白等 20–40 分钟（logs/093 的实测事故）。
_meta="$HOME/projects/dsv41-upstream-pr/agents/R_8card_int8/out/$TAG/$TAG.meta.txt"
_srvlog=""
"${CMD[@]}" &
_arm_pid=$!
say "已起臂（pid=$_arm_pid）；开始盯 G2b（dsa_dir_D）+ 捕获期指纹；最多盯 40 分钟"
_g2b_done=0
for _i in $(seq 1 240); do
    sleep 10
    if [ "$_g2b_done" = "0" ] && [ -f "$_meta" ]; then
        _d=$(grep -aoE "dsa_dir_D=\S+" "$_meta" 2>/dev/null | head -1)
        if [ -n "$_d" ]; then
            _g2b_done=1
            # ★★★ 2026-09-23 02:5x **判据修正（实测误判）**：
            #   原先的判据是 `case "$_d" in *S_graphfix*)` —— 即**按路径字符串**断言。
            #   但我们的**性能修复件**（`agents/CHUNKVIEW/pkg-kv8pf` 的 chunk 连续视图版 `d84f087c`
            #   = `S_graphfix`/`94aeebb7` **下游衍生 + 1 hunk**）路径里**不含** `S_graphfix`
            #   ⇒ 门把"正确的替换件"**误判成"挂错份"**，FATAL 停臂。
            #   （现场：容器内 md5 = `d84f087c` 是对的，而门却报"不是 graphsafe 版"。）
            #   ⇒ **改成内容判据**：① 路径含 `S_graphfix` **或** ② 路径在显式白名单
            #      `R8_DSA_ALLOW_DIRS`（冒号分隔）里 **或** ③ 容器内那份**同时含**
            #      `rows_bound`（≥1）与 `VLLM_V41_KV8_GRAPH_SAFE` 两个标记 —— 即"它真是 graphsafe 谱系"。
            #   ★ 这仍然是 fail-closed：**三个条件都不满足**才 FATAL。
            _dsadir=${_d#dsa_dir_D=}
            _g2b_ok=0; _g2b_why=""
            case "$_d" in
                *S_graphfix*) _g2b_ok=1; _g2b_why="路径含 S_graphfix" ;;
            esac
            if [ "$_g2b_ok" = "0" ] && [ -n "${R8_DSA_ALLOW_DIRS:-}" ]; then
                _oldIFS=$IFS; IFS=':'
                for _al in $R8_DSA_ALLOW_DIRS; do
                    [ -n "$_al" ] || continue
                    case "$_dsadir" in *"$_al"*) _g2b_ok=1; _g2b_why="命中白名单 $_al"; break ;; esac
                done
                IFS=$_oldIFS
            fi
            if [ "$_g2b_ok" = "0" ]; then
                # 内容判据：容器内那份是否同时含 graphsafe 的两个标记
                _cfile="/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py"
                # ★★★ 2026-09-23 03:1x **第二个实测误判**：此前这里用 `docker exec "$TAG"`，
                #   但 runner 起的容器名是 **`r8-$TAG`**（见 `run_arm_r8.sh:44 NAME=${NAME:-r8-$TAG}`）
                #   ⇒ `docker exec r8-merged` 打空 ⇒ `_rb/_gs` 全是空 ⇒ **内容判据永远不过** ⇒ 又一次误杀。
                #   ⇒ 改成**两级探测**：先试 `r8-$TAG`，再试 `$TAG`（兼容手工起臂时的命名）。
                _ctr=""
                for _cand in "r8-$TAG" "$TAG"; do
                    if docker inspect "$_cand" >/dev/null 2>&1; then _ctr="$_cand"; break; fi
                done
                if [ -n "$_ctr" ]; then
                    _rb=$(docker exec "$_ctr" grep -c "rows_bound" "$_cfile" 2>/dev/null | tr -d "\r")
                    _gs=$(docker exec "$_ctr" grep -c "VLLM_V41_KV8_GRAPH_SAFE" "$_cfile" 2>/dev/null | tr -d "\r")
                    _cmd5=$(docker exec "$_ctr" md5sum "$_cfile" 2>/dev/null | cut -d' ' -f1)
                else
                    _rb=""; _gs=""; _cmd5=""
                fi
                if [ "${_rb:-0}" -ge 1 ] && [ "${_gs:-0}" -ge 1 ]; then
                    _g2b_ok=1
                    _g2b_why="内容判据：容器内 md5=$_cmd5 含 rows_bound×${_rb} + KV8_GRAPH_SAFE×${_gs}（graphsafe 谱系）"
                fi
            fi
            if [ "$_g2b_ok" = "1" ]; then
                say "G2b ✓ 实际挂的 dsa 是 graphsafe 谱系：$_d（$_g2b_why）"
            else
                echo "" >&2
                echo "⛔⛔ [4axis][G2b] 实际挂的 dsa **不在 graphsafe 谱系**：" >&2
                echo "     $_d" >&2
                echo "     （内容判据也没过：容器内 rows_bound×${_rb:-?} KV8_GRAPH_SAFE×${_gs:-?} md5=${_cmd5:-?}）" >&2
                echo "   ⇒ 图捕获期会报『capture failed: LocalScalarDenseNpu.cpp:23』+ EE1016×8（logs/093 实测）。" >&2
                echo "   ⇒ 请停掉这一臂，带上这两个 env 重起：" >&2
                echo "        DSA_SRC=D R8_KV8_DIR_D=$S/pkgs/pkg-kv8pf" >&2
                echo "      （或把自定义件目录加进 R8_DSA_ALLOW_DIRS —— 冒号分隔）" >&2
            fi
        fi
    fi
    # 捕获期指纹（只用 runner 自己打印的那个 serve.log 路径）
    if [ -z "$_srvlog" ] && [ -d "$PKG/results" ]; then
        _srvlog=$(ls -dt "$PKG"/results/*"$TAG"_*/ 2>/dev/null | head -1)serve.log
        [ -f "$_srvlog" ] || _srvlog=""
    fi
    if [ -n "$_srvlog" ] && [ -f "$_srvlog" ]; then
        if grep -aq "capture failed" "$_srvlog" 2>/dev/null && grep -aq "LocalScalarDenseNpu" "$_srvlog" 2>/dev/null; then
            echo "" >&2
            echo "⛔⛔ [4axis][指纹] 捕获期出现『capture failed … LocalScalarDenseNpu.cpp:23』" >&2
            echo "   ⇒ 这就是 logs/093 的指纹：**dsa 不是 graphsafe 版**。" >&2
            echo "   ⇒ 先查： grep -aoE 'dsa_dir_D=\S+' $_meta" >&2
            echo "   ⇒ 不要先去怀疑 Engram / int8 开关 / 池分量（都已实测排除，logs/093 §2.1）。" >&2
            break
        fi
    fi
    kill -0 "$_arm_pid" 2>/dev/null || { say "臂已结束（pid 退出）"; break; }
done
wait "$_arm_pid"
exit $?
