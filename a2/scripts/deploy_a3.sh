#!/usr/bin/env bash
# =============================================================================
# deploy_a3.sh —— 在**一台全新的 A3 机器**上，把 GitHub 上这个仓 clone 下来后
#                **一条命令**完成部署（= 前置体检 + 选卡 + 造 shadow-pkg + 干跑，默认不起服务）
#
# 为什么需要它（与已有脚本的分工，别互相替代）：
#   | 脚本 | 管什么 | 新机器上够不够 |
#   |---|---|---|
#   | `scripts/serve_a3.sh`            | 引擎入口（与 A2 共用），要求 `DEVS` 显式给 | 不够：它不造 shadow-pkg |
#   | `a2/scripts/serve_a3_offload.sh` | 薄包装（PLAT=a3 + 卸载/int8/draft 入图默认值） | 不够：**前提是 shadow-pkg 已存在** |
#   | **本脚本 `deploy_a3.sh`**         | ★ 把"新机器上什么都没有"这件事补齐，再交给上面那个 | 从零到干跑通过 |
#
# 新机器上真正会卡住的四件事（本脚本逐个前置拦住，且都**打印确切修法**）：
#   ① **模型不是"一个目录"**：模型目录里的权重是**软链**，指到同级/邻级的其它产物目录
#      （实测：`v41-w4a8-engram-dr-vision-qrot-mtpq` 只占 926 MB，但**闭包 = 它 + 6 个兄弟目录**，
#        含 `engram-int8`(206 GiB)、`v41-w4a8-stage1`(273 GiB) ⇒ **要搬 ≈520 GiB**）。
#      只搬那一个目录 ⇒ 容器里权重**读到断链**，而且报错发生在很晚（worker 加载期）。
#   ② **镜像不在**：A3 用官方镜像 + `PATCH_MODE=mount`（不烘焙），新机器第一次要 `docker pull`。
#   ③ **没有 shadow-pkg**：`serve_*_offload.sh` 的补丁全靠它挂；新机器上必须**现场造**（幂等）。
#   ④ **卡不是你的**：A3 是共用机 ⇒ 默认只从 `npu-smi` 报**空闲**的卡里选，并响亮打印选了哪些；
#      `scripts/serve_a3.sh` 在真起服前还会**再查一次**占用（双保险）。
#
# 用法（新机器上；**默认只干跑，不碰 docker**）：
#   MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash a2/scripts/deploy_a3.sh
#   MODEL=... LAUNCH=1 bash a2/scripts/deploy_a3.sh              # 真起服（前台等待就绪）
#   MODEL=... DEVS="8 9 10 11 12 13 14 15" LAUNCH=1 ...          # 显式指定卡（共用机推荐）
#   MODEL=... OFFLOAD=0 LAUNCH=1 ...                             # ★ 单变量关掉 DRAM 卸载
#   MODEL=... PULL=1 bash a2/scripts/deploy_a3.sh                # 镜像不在时自动 docker pull
#   MODEL=... SHOW_SIZES=1 bash a2/scripts/deploy_a3.sh          # 额外算闭包体积（慢，几百 GB）
#
# 退出码：0 = 干跑通过（或已起服）；2 = 前置/环境不合格；3 = 空闲卡不够；64 = 用法错
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# ---------------------------------------------------------------- 可注入的"外部命令"
# 为什么做成变量：① 自测要靠桩（零真机）；② 有的机器 docker 要写成 `sudo docker`。
DOCKER=${DOCKER:-docker}
NPU_SMI=${NPU_SMI:-npu-smi}
PY=${PY:-python3}

say() { printf '\n\033[1m======== %s ========\033[0m\n' "$*"; }
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn(){ printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; _fail=$((_fail+1)); }
_fail=0

# ---------------------------------------------------------------- 参数
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
    cat >&2 <<'MSG'
⛔ 必须给模型目录：MODEL=<模型目录>（= 含 config.json / *.safetensors 的量化产物目录）

   例：MODEL=$HOME/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
         bash a2/scripts/deploy_a3.sh

   ★ 新机器上还没有模型？看 a2/docs/A3-DEPLOY.md §2（搬运清单 + 闭包体积），
     别只搬那一个目录 —— 它是软链构造的，闭包 ≈520 GiB。
MSG
    exit 64
fi

PLAT=a3                                  # 本脚本只管 A3；A2 用 serve_a2_offload.sh
TP=${TP:-8}
DEVS=${DEVS:-}
LAUNCH=${LAUNCH:-0}                      # ★ 默认干跑
PULL=${PULL:-0}
SHOW_SIZES=${SHOW_SIZES:-0}
SHADOW_DST=${SHADOW_DST:-$HOME/projects/dsv41-upstream-pr/shadow-pkg-a3}
IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}

# 与 A2 生产同口径的默认档位（改一律用**用户接口**名字，见 serve_a2_offload.sh 的文件头）
OFFLOAD=${OFFLOAD:-1}
ENGRAM=${ENGRAM:-1}
ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
KV8_SWA=${KV8_SWA:-1}
KV8_RING_FP16=${KV8_RING_FP16:-1}
# ★ DRAFT_GRAPH 显式给（不靠下游默认）：这是"四轴"里最容易**静默退回 eager** 的那一轴
#   （A2 实测：入图 88.7 vs eager 54.7 tok/s）。显式传 ⇒ 干跑日志里必然能核对到 DRAFT_GRAPH=1。
DRAFT_GRAPH=${DRAFT_GRAPH:-1}
DROPCACHE=${DROPCACHE:-0}                # A3 共用机：默认不清整机 page cache
DRY_LOG=${DRY_LOG:-$HOME/a3_deploy_dryrun_$(date +%Y%m%d_%H%M%S).log}

printf '\033[1m######## A3 新机部署（deploy_a3.sh）########\033[0m\n'
echo "  仓库   : $REPO"
echo "  模型   : $MODEL"
echo "  shadow : $SHADOW_DST（本脚本会现场造/刷新）"
echo "  镜像   : $IMAGE"
echo "  档位   : OFFLOAD=$OFFLOAD ENGRAM=$ENGRAM ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX KV8_SWA=$KV8_SWA KV8_RING_FP16=$KV8_RING_FP16 DRAFT_GRAPH=$DRAFT_GRAPH TP=$TP"
if [ "$LAUNCH" = "1" ]; then echo "  模式   : ★ 真起服（LAUNCH=1）"
else echo "  模式   : 干跑（默认；要真起服加 LAUNCH=1）"; fi

# ================================================================ ① 前置体检
say "① 前置体检（宿主 / 工具 / 仓库 / 镜像）"

# ①a 必须在**宿主机**上跑（容器里没有宿主视角的 npu-smi/docker，挂载路径也全是宿主的）
if [ -d /vllm-workspace ]; then
    bad "看着像在**容器里**（存在 /vllm-workspace）—— 本脚本要在**宿主机**上跑"
fi
for _c in "$DOCKER" "$NPU_SMI" "$PY"; do
    if command -v "$_c" >/dev/null 2>&1; then ok "找到命令：$_c（$(command -v "$_c")）"
    else bad "找不到命令：$_c"; fi
done

# ①b 仓库完整性（从 GitHub clone 的裸仓必须自带这些；缺任何一个都别继续）
for _f in scripts/serve_a3.sh scripts/serve_a2.sh scripts/serve_v2.sh \
          a2/scripts/serve_a2_offload.sh a2/scripts/serve_a3_offload.sh \
          a2/scripts/make_shadow_pkg.sh tools/list_chips.sh tools/check_model_dir.sh \
          a2/patches/0001-offload-scheduler.patch.py \
          a2/patches/kv8-offload-pool/p2_pool.py \
          a2/patches/kv8-graphsafe/dsa_v41.py; do
    if [ -f "$REPO/$_f" ]; then ok "仓库文件：$_f"
    else bad "仓库里缺 $_f（clone 不完整，或不是这个仓）"; fi
done

# ①c docker 守护进程可达（新机器最常见：用户不在 docker 组 ⇒ 所有 docker 命令 permission denied）
if command -v "$DOCKER" >/dev/null 2>&1; then
    if _dout=$("$DOCKER" info 2>&1); then
        ok "docker 可用（server $("$DOCKER" version --format '{{.Server.Version}}' 2>/dev/null || echo '?')）"
    else
        bad "docker 不可用（守护进程没起 / 当前用户不在 docker 组 / 需要 sudo）"
        printf '%s\n' "$_dout" | head -3 | sed 's/^/        /'
        echo "        修法：sudo systemctl start docker；或把用户加进 docker 组后重新登录（newgrp docker）"
    fi
fi

# ①d 镜像在不在（A3 走 mount 模式 ⇒ **官方镜像就够**，不需要自己 build）
if command -v "$DOCKER" >/dev/null 2>&1 && "$DOCKER" info >/dev/null 2>&1; then
    if "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
        ok "镜像已在本地：$IMAGE"
    elif [ "$PULL" = "1" ]; then
        say "①d 镜像不在本地 ⇒ PULL=1，开始 docker pull（几 GB～20 GB，等一会儿）"
        if "$DOCKER" pull "$IMAGE"; then ok "拉取完成"; else bad "docker pull 失败（内网 registry 不通 / 没权限）"; fi
    else
        bad "镜像不在本地：$IMAGE"
        echo "        修法二选一："
        echo "          a) 加 PULL=1 让本脚本拉：MODEL=... PULL=1 bash a2/scripts/deploy_a3.sh"
        echo "          b) 手工拉：$DOCKER pull $IMAGE"
        echo "        （内网拉不动就找运维要镜像；★ 不要在 A3 上重新 build —— mount 模式不需要）"
    fi
fi

if [ "$_fail" != "0" ]; then
    echo
    bad "① 有 $_fail 项不合格 ⇒ **停在这里**（继续下去只会在容器里报更难懂的错）"
    exit 2
fi

# ================================================================ ② 模型（含"闭包"检查）
say "② 模型目录 + 软链闭包（新机器上最容易踩的一条）"
if [ ! -d "$MODEL" ]; then
    bad "模型目录不存在：$MODEL"
    echo "        新机器上要先搬模型；搬运清单见 a2/docs/A3-DEPLOY.md §2（★ 闭包 ≈520 GiB）"
    exit 2
fi
ok "模型目录存在：$MODEL"

# ★ 为什么必须查闭包：模型目录是"软链拼出来的"，只搬它自己 ⇒ 容器里读到断链，
#   而报错发生在加载权重的中途（白等 10+ 分钟）。这里**先**把断链与"必须一起搬的目录"列出来。
_py_out=$("$PY" - "$MODEL" <<'PY' 2>&1
import os, sys, json
root = os.path.realpath(sys.argv[1])
seen, todo, links = set(), [root], []
while todo:
    d = todo.pop()
    if d in seen:
        continue
    seen.add(d)
    for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
        for n in dirnames + filenames:
            p = os.path.join(dirpath, n)
            if os.path.islink(p):
                t = os.path.realpath(p)
                links.append((p, t))
                if not os.path.exists(t):
                    print("BROKEN\t%s\t%s" % (p, t))
                elif os.path.isdir(t) and t not in seen:
                    todo.append(t)
ext = sorted({os.path.dirname(t) for _, t in links
              if os.path.exists(t) and not os.path.realpath(t).startswith(root + os.sep)})
print("EXTERNAL\t%d\t%s" % (len(links), json.dumps(ext)))
PY
)
_broken=$(printf '%s\n' "$_py_out" | grep -c '^BROKEN' || true)
if [ "${_broken:-0}" -gt 0 ]; then
    bad "模型目录里有 $_broken 个**断链**（容器里会读到不存在的权重）"
    printf '%s\n' "$_py_out" | grep '^BROKEN' | head -10 | sed 's/^/        /'
    echo "        ⇒ 这些链接指向的产物也要一起搬（逐条见上）"
    exit 2
fi
ok "软链闭包无断链"
_ext=$(printf '%s\n' "$_py_out" | awk -F'\t' '$1=="EXTERNAL"{print $3}')
if [ -n "${_ext:-}" ] && [ "$_ext" != "[]" ]; then
    _n=$("$PY" -c 'import json,sys;print(len(json.loads(sys.argv[1])))' "$_ext" 2>/dev/null || echo 0)
    echo "  ★ 该模型还引用外部目录：$_n 个 —— 新机器上要**连同它自己一起**搬："
    "$PY" -c 'import json,sys
for d in json.loads(sys.argv[1]): print("        " + d)' "$_ext" 2>/dev/null
    if [ "$SHOW_SIZES" = "1" ]; then
        echo "  ★ 体积（SHOW_SIZES=1；几百 GB 的目录会比较慢）："
        { printf '%s\n' "$MODEL"
          "$PY" -c 'import json,sys
for d in json.loads(sys.argv[1]): print(d)' "$_ext" 2>/dev/null
        } | while read -r _d; do
            [ -n "$_d" ] && printf '        %-8s %s\n' "$(du -sh "$_d" 2>/dev/null | cut -f1)" "$_d"
        done
    else
        echo "        （加 SHOW_SIZES=1 可同时算出各目录体积）"
    fi
fi

# 模型目录自检（复用仓里的检查器；rc=1 致命、rc=2 仅告警）
if [ -f "$REPO/tools/check_model_dir.sh" ]; then
    _mc=/tmp/.a3_model_check.$$
    bash "$REPO/tools/check_model_dir.sh" "$MODEL" >"$_mc" 2>&1
    _rc=$?
    if [ "$_rc" = "0" ]; then ok "模型目录自检通过（tools/check_model_dir.sh）"
    elif [ "$_rc" = "2" ]; then warn "模型目录自检有告警（rc=2，能跑）"
    else bad "模型目录自检**致命**失败（rc=$_rc）"; fi
    [ "$_rc" != "0" ] && tail -12 "$_mc" | sed 's/^/        /'
    rm -f "$_mc"
    [ "$_rc" = "1" ] && exit 2
fi

# ================================================================ ③ 选卡
say "③ 选卡（默认只用 npu-smi 报**空闲**的卡；不对就自己给 DEVS=）"
if [ -n "$DEVS" ]; then
    _n=0
    for _c in $DEVS; do
        case "$_c" in ''|*[!0-9]*) bad "DEVS 里有非法项：'$_c'（只能数字，空格分隔）"; exit 2 ;; esac
        _n=$((_n+1))
    done
    if [ "$_n" != "$TP" ]; then
        bad "DEVS 有 $_n 张，而 TP=$TP ⇒ 数量不匹配"
        echo "        要么给满 $TP 张，要么显式改 TP=$_n（只在你确定要少卡跑时）"
        exit 2
    fi
    ok "用你指定的 DEVS='$DEVS'（$_n 张 = TP=$TP）"
else
    if ! command -v "$NPU_SMI" >/dev/null 2>&1; then
        bad "没有 $NPU_SMI ⇒ 不能自动选卡；请显式给 DEVS=<卡号...>"
        exit 2
    fi
    _free=$(bash "$REPO/tools/list_chips.sh" --free 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' ')
    _cnt=$(printf '%s' "$_free" | tr ' ' '\n' | grep -cE '^[0-9]+$' || true)
    echo "  npu-smi 报的空闲卡：${_free:-（无）}（$_cnt 张，需要 $TP 张）"
    if [ "${_cnt:-0}" -lt "$TP" ]; then
        bad "空闲卡不够 $TP 张 ⇒ 不能自动选"
        echo "        修法：① 先看谁占着：bash tools/list_chips.sh"
        echo "              ② 若里面有**你自己的**残留进程，停掉后重试"
        echo "              ③ 卡够但不想用自动选的：显式 DEVS=\"<8 个卡号>\""
        echo "              ④ 确实要带别人的占用起服务：ALLOW_BUSY=1（危险；本脚本不替你决定）"
        exit 3
    fi
    DEVS=$(printf '%s\n' $_free | head -"$TP" | tr '\n' ' ')
    DEVS=${DEVS% }
    echo
    echo "  ★★ 我替你选了这 $TP 张：DEVS=\"$DEVS\""
    echo "     —— 若这些卡不属于你（A3 是共用机），请 Ctrl-C 后用 DEVS=\"...\" 显式指定；"
    echo "        scripts/serve_a3.sh 在真起服前会再查一次占用，但'别人刚空出来的卡'它拦不住。"
fi
export DEVS

# ================================================================ ④ 造 shadow-pkg（新机器上必须现场造）
say "④ 造 shadow-pkg（补丁全靠它挂进容器；幂等，重复跑安全）"
mkdir -p "$(dirname "$SHADOW_DST")"
_sm=/tmp/.a3_shadow.$$
if PKG="$REPO" DST="$SHADOW_DST" bash "$REPO/a2/scripts/make_shadow_pkg.sh" >"$_sm" 2>&1; then
    ok "shadow-pkg 就绪：$SHADOW_DST"
    grep -E '^✓' "$_sm" | tail -6 | sed 's/^/        /'
else
    bad "make_shadow_pkg.sh 失败（这一步失败 = 后面所有补丁都不会挂）"
    tail -15 "$_sm" | sed 's/^/        /'
    rm -f "$_sm"
    exit 2
fi
rm -f "$_sm"

# ================================================================ ⑤ 干跑（把真正生效的配置打全）
say "⑤ 干跑 DRY=1（不碰 docker；下面这些值是**容器里真会生效**的）"
_run() {   # $1 = DRY(1/0)
    env SHADOW_PKG="$SHADOW_DST" MODEL="$MODEL" DEVS="$DEVS" TP="$TP" IMAGE="$IMAGE" \
        ENGRAM="$ENGRAM" ENGRAM_DEVICE_INDEX="$ENGRAM_DEVICE_INDEX" \
        KV8_SWA="$KV8_SWA" KV8_RING_FP16="$KV8_RING_FP16" \
        DRAFT_GRAPH="$DRAFT_GRAPH" \
        OFFLOAD="$OFFLOAD" DROPCACHE="$DROPCACHE" DRY="$1" \
        bash "$REPO/a2/scripts/serve_a3_offload.sh"
}
if _run 1 >"$DRY_LOG" 2>&1; then
    ok "干跑 rc=0（日志：$DRY_LOG）"
else
    bad "干跑 rc≠0 ⇒ 别起服务，先看日志：$DRY_LOG"
    tail -25 "$DRY_LOG" | sed 's/^/        /'
    exit 2
fi

# 判据绑**内容**（不是"我传了变量"）：把要生效的关键值从干跑日志里真的读出来
_expect() {  # $1 = 必须出现的字面量  $2 = 说明
    if grep -qF -- "$1" "$DRY_LOG"; then ok "$2"
    else bad "$2（干跑日志里找不到：$1）"; fi
}
_expect "DEVS=$DEVS"               "选中的卡真的传下去了"
_expect "PATCH_MODE=mount"         "mount 模式（官方镜像 + 挂载补丁）"
_expect "served_name=deepseek-v41" "服务名 = deepseek-v41（全仓工具默认；填错会 404）"
_expect "max_len=1048576"          "1M 上下文"
_expect "DRAFT_GRAPH=1"            "draft 入图（四轴之一）"
if [ "$OFFLOAD" = "1" ]; then
    _expect "A2-OFFLOAD"           "卸载补丁已挂（DRAM KV 卸载）"
    _expect "kv-transfer-config"   "kv-transfer-config 已带进容器"
else
    echo "  ★ OFFLOAD=0 ⇒ 负判据：下面三样**都不许**出现在干跑日志里"
    for _neg in "offloading/scheduler.py" "kv-transfer-config" "L1-POOL"; do
        if grep -qF -- "$_neg" "$DRY_LOG"; then bad "OFFLOAD=0 但仍出现 $_neg（关得不干净）"
        else ok "OFFLOAD=0：$_neg 已消失"; fi
    done
fi
if [ "$_fail" != "0" ]; then
    echo
    bad "⑤ 干跑内容判据有 $_fail 项不过 ⇒ 停（日志：$DRY_LOG）"
    exit 2
fi

# ================================================================ ⑥ 起服（显式 LAUNCH=1）
if [ "$LAUNCH" != "1" ]; then
    say "⑥ 干跑通过 —— **默认不起服务**"
    echo
    echo "  真要起服，把下面这条原样再跑一次（多了 LAUNCH=1）："
    echo
    echo "      MODEL=$MODEL LAUNCH=1 \\"
    [ -n "${DEVS:-}" ] && echo "      DEVS=\"$DEVS\" \\"
    echo "      OFFLOAD=$OFFLOAD ENGRAM=$ENGRAM ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX \\"
    echo "      KV8_SWA=$KV8_SWA KV8_RING_FP16=$KV8_RING_FP16 DRAFT_GRAPH=$DRAFT_GRAPH \\"
    echo "      bash a2/scripts/deploy_a3.sh"
    echo
    echo "  （干跑日志留着别删：$DRY_LOG）"
    exit 0
fi

say "⑥ 真起服（前台等待就绪；首次冷编译 15–20 min 属正常）"
echo "  起服后的三条自检（**服务活着时**另开终端跑）："
echo "      PLAT=a3 PROMPT_TOKENS=131072 bash a2/scripts/verify_dram_offload.sh   # 卸载：存 + 取回"
echo "      $PY a2/scripts/text_correctness_probe.py --base-url http://127.0.0.1:8020 --model deepseek-v41 --mode all"
echo "      PLAT=a3 bash a2/scripts/collect_evidence.sh                            # 一键收 EVIDENCE.txt"
echo
_run 0
rc=$?
if [ "$rc" != "0" ]; then
    echo
    bad "起服未成功（rc=$rc）—— 常见原因与判读见 a2/docs/A3-DEPLOY.md §5"
fi
exit "$rc"
