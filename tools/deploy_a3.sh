#!/usr/bin/env bash
# =============================================================================
# deploy_a3.sh —— 在**一台全新的 A3 机器**上，从 GitHub clone 本仓后**一条命令**完成部署
#
#   「部署」= 前置体检 → 模型软链闭包体检 → 选卡 → 干跑（默认不起服务）
#   真起服再加一个 `LAUNCH=1`（本脚本仍然会**先把干跑的判据全过一遍**才起）。
#
# 与其它脚本的分工（别互相替代）：
#   | 脚本 | 管什么 |
#   |---|---|
#   | `scripts/serve_a3.sh` | 引擎入口：选卡校验 + 与 A2 共用的全部开关（要求 `DEVS` 显式给） |
#   | `scripts/serve_a2.sh` | 引擎本体（A2/A3 共用） |
#   | **本脚本** | ★ 把"新机器上什么都没有"这件事补齐：镜像在不在、模型搬全没、卡是不是空的、配置到底生不生效 |
#
# ★ 新机器上真正会卡住的四件事（本脚本逐个前置拦住，且都打印**确切修法**）：
#   ① **模型不是"一个目录"**：模型目录里的权重是**软链**，指到同级的其它产物目录。
#      实测（a3-21）：`v41-w4a8-engram-dr-vision-qrot-mtpq` 自己只 **926 MB**，
#      但**闭包 = 它 + 6 个兄弟目录**（含 `engram-int8` **206 GiB**、`v41-w4a8-stage1` **273 GiB**）
#      ⇒ **要搬 ≈520 GiB**。只搬那一个目录 ⇒ 容器里权重**读到断链**，而且要到 worker 加载期才炸。
#   ② **镜像不在**：A3 用**官方镜像 + `PATCH_MODE=mount`**（补丁挂载进去，不烘焙）⇒ 新机器第一次要 `docker pull`。
#   ③ **卡不是你的**：A3 是**共用机** ⇒ 默认只从 `npu-smi` 报**空闲**的卡里选，并把选了哪些**大声打印**；
#      `scripts/serve_a3.sh` 在真起服前还会**再查一次**占用（双保险）。
#   ④ **共享机的 page cache**：`scripts/serve_a2.sh` 的 `DROPCACHE` 默认是 **1 = 清整机 page cache**
#      ⇒ 在共用 A3 上会打到别人的租户。本脚本默认传 `DROPCACHE=0`（要清就显式 `DROPCACHE=1`）。
#
# 用法（**在宿主机上**；默认只干跑）：
#   MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash tools/deploy_a3.sh
#   MODEL=... LAUNCH=1 bash tools/deploy_a3.sh                  # 真起服（前台等待就绪）
#   MODEL=... DEVS="8 9 10 11 12 13 14 15" LAUNCH=1 bash tools/deploy_a3.sh
#   MODEL=... PULL=1 bash tools/deploy_a3.sh                    # 镜像不在时自动 docker pull
#   MODEL=... SHOW_SIZES=1 bash tools/deploy_a3.sh              # 额外算闭包体积（慢）
#   MODEL=... DRAFT_GRAPH=1 bash tools/deploy_a3.sh             # ★ A3 推荐：draft 入图（默认已开）
#
# 退出码：0 = 干跑通过 / 已起服；2 = 前置或环境不合格；3 = 空闲卡不够；64 = 用法错
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/.." && pwd)

# 外部命令做成变量：① 有的机器 docker 要 `sudo docker`；② 自测用桩注入。
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
⛔ 必须给模型目录：MODEL=<模型目录>（含 config.json / *.safetensors 的量化产物目录）

   例：MODEL=$HOME/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
         bash tools/deploy_a3.sh

   ★ 新机器上还没有模型？看 docs/A3-DEPLOY.md §2（搬运清单 + 闭包体积）：
     别只搬那一个目录 —— 它是软链构造的，闭包 ≈520 GiB。
MSG
    exit 64
fi

TP=${TP:-8}
DEVS=${DEVS:-}
LAUNCH=${LAUNCH:-0}          # ★ 默认干跑
PULL=${PULL:-0}
SHOW_SIZES=${SHOW_SIZES:-0}
IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}

# 平台口径默认值（改一律用这些**用户接口**名字；`scripts/serve_a3.sh` 认得它们）
# ★★ 上下文/并发：默认取 **A3 上已验证过的口径**（128K 档）
#   为什么不用 `serve_a2.sh` 的默认 `MAX_LEN=1048576`：那是 **A2 生产**口径，
#   在 **A3 上没有验证过**（A3 历史臂全是 `max_len=133120 max_seqs=32`，见
#   `shadow-pkg/results/r8_*/serve_cmd.txt`）。本脚本的职责是"新机器上第一条命令跑通"，
#   所以默认必须是**验证过的**口径；要 1M 就显式给（见文档 §4）。
MAX_LEN=${MAX_LEN:-133120}
MAX_SEQS=${MAX_SEQS:-32}
# ★★ draft 入图（DRAFT_GRAPH）：**默认 0，别改**
#   为什么（`scripts/serve_a2.sh:217-223` 的实测记录）：A3 上开它 ⇒ **接受长度 A 掉到 1.06–1.08**
#   （draft 完全不产出 = 静默失效形态），而 `ms/step` 反而"更好看"（25.1 vs 27–30），
#   因为每步只出 1.08 个 token 而不是 2.85 个 ⇒ **真实吞吐慢 2.2×**。
#   ⇒ 判据必须是 **(A, tok/s) 这一对**，不是 ms/step 单值。
#   想实验必须用：`bash tools/draft_graph_guard.sh` 确认 **A ≥ 1.3**，否则不要用。
DRAFT_GRAPH=${DRAFT_GRAPH:-0}
DROPCACHE=${DROPCACHE:-0}         # ★ 共用机：不清整机 page cache
SERVED_NAME=${SERVED_NAME:-deepseek-v41}
ENGRAM=${ENGRAM:-1}
PORT=${PORT:-8020}
DRY_LOG=${DRY_LOG:-$HOME/a3_deploy_dryrun_$(date +%Y%m%d_%H%M%S).log}

printf '\033[1m######## A3 新机部署（tools/deploy_a3.sh）########\033[0m\n'
echo "  仓库   : $PKG"
echo "  模型   : $MODEL"
echo "  镜像   : $IMAGE（mount 模式：补丁由本仓 patches/files 挂载，不烘焙）"
echo "  档位   : ENGRAM=$ENGRAM DRAFT_GRAPH=$DRAFT_GRAPH MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS SERVED_NAME=$SERVED_NAME DROPCACHE=$DROPCACHE TP=$TP"
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
          tools/list_chips.sh tools/check_model_dir.sh \
          patches/files/engram_hash.py patches/files/engram_jit_kernel.py \
          patches/files/model.py patches/files/token_dispatcher_moemask.py; do
    if [ -f "$PKG/$_f" ]; then ok "仓库文件：$_f"
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
        say "①d 镜像不在本地 ⇒ PULL=1，开始 docker pull（几 GB～25 GB，等一会儿）"
        if "$DOCKER" pull "$IMAGE"; then ok "拉取完成"; else bad "docker pull 失败（内网 registry 不通 / 没权限）"; fi
    else
        bad "镜像不在本地：$IMAGE"
        echo "        修法二选一："
        echo "          a) 加 PULL=1 让本脚本拉：MODEL=... PULL=1 bash tools/deploy_a3.sh"
        echo "          b) 手工拉：$DOCKER pull $IMAGE"
        echo "        （内网拉不动就找运维要镜像；★ 不要在 A3 上重新 build —— mount 模式不需要）"
    fi
fi

if [ "$_fail" != "0" ]; then
    echo
    bad "① 有 $_fail 项不合格 ⇒ **停在这里**（继续下去只会在容器里报更难懂的错）"
    exit 2
fi

# ================================================================ ② 模型（含软链闭包）
say "② 模型目录 + 软链闭包（新机器上最容易踩的一条）"
if [ ! -d "$MODEL" ]; then
    bad "模型目录不存在：$MODEL"
    echo "        新机器上要先搬模型；搬运清单见 docs/A3-DEPLOY.md §2（★ 闭包 ≈520 GiB）"
    exit 2
fi
ok "模型目录存在：$MODEL"

# ★ 为什么必须查闭包：模型目录是软链拼出来的，只搬它自己 ⇒ 容器里读到断链，
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
# 外部依赖：链接指到 root 之外时，要一起搬的目标。
#   - 链到**目录**（如整块 engram 表目录）⇒ 要搬的就是那个目录本身
#   - 链到**文件**（如某个 .safetensors 分片）⇒ 要搬的是它所在目录
#   （只算 dirname 会漏掉/错位目录型链接 —— 实测踩到）
ext = sorted({(t if os.path.isdir(t) else os.path.dirname(t))
              for _, t in links
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
        _tot=0
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
if [ -f "$PKG/tools/check_model_dir.sh" ]; then
    _mc=/tmp/.a3_model_check.$$
    bash "$PKG/tools/check_model_dir.sh" "$MODEL" >"$_mc" 2>&1
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
    _free=$(bash "$PKG/tools/list_chips.sh" --free 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' ')
    _cnt=$(printf '%s' "$_free" | tr ' ' '\n' | grep -cE '^[0-9]+$' || true)
    echo "  npu-smi 报的空闲卡：${_free:-（无）}（$_cnt 张，需要 $TP 张）"

    # ★★★ 2026-09-23 **"空闲" ≠ "可以拿"** —— 这一条是真机上抓到的：
    #   首版按"空闲就选"跑在 a3-21 上，选中了 **0–7**；而本项目的约定是
    #     **A3 上 0–7 不是我们的**（我们在 A3 用 **Phy-ID 8–15**）。机器当时"看起来全空闲"
    #     （别人的任务刚好退了），于是脚本会把服务开到**不属于我们的卡**上。
    #   ⇒ 现在引入 **优先区间** `PREFER_DEVS`（A3 默认 8–15）：先从优先区间里挑；
    #     不够时才**显式告知**并降级到其余空闲卡（绝不静默）。
    #     为什么不用"避让列表"：避让列表表达不了"这一批是我们的"这层含义，
    #     而"优先区间"能，并且在区间够用时**天然不会碰别人的卡**。
    PREFER_DEVS=${PREFER_DEVS:-"8 9 10 11 12 13 14 15"}
    _pick=""
    for _c in $PREFER_DEVS; do
        case " $_free " in *" $_c "*) _pick="$_pick $_c" ;; esac
        [ "$(printf '%s\n' $_pick | grep -cE '^[0-9]+$')" -ge "$TP" ] && break
    done
    _pick=${_pick# }
    _pn=$(printf '%s' "$_pick" | tr ' ' '\n' | grep -cE '^[0-9]+$' || true)
    if [ "${_pn:-0}" -lt "$TP" ]; then
        # 优先区间不够 ⇒ 用其余空闲卡补齐，但要**说清**补了哪些
        _rest=""
        for _c in $_free; do
            case " $_pick " in *" $_c "*) continue ;; esac
            _rest="$_rest $_c"
            _pick="$_pick $_c"
            _pn=$((_pn+1))
            [ "$_pn" -ge "$TP" ] && break
        done
        if [ "$_pn" -lt "$TP" ]; then
            bad "空闲卡不够 $TP 张（优先区间 $PREFER_DEVS 里只有 $(printf '%s' "${_pick:- }" | wc -w) 张，其余空闲卡也用上后仍不足）⇒ 不能自动选"
            echo "        修法：① 先看谁占着：bash tools/list_chips.sh"
            echo "              ② 若里面有**你自己的**残留进程，停掉后重试"
            echo "              ③ 卡够但不想用自动选的：显式 DEVS=\"<8 个卡号>\""
            echo "              ④ 确实要带别人的占用起服务：ALLOW_BUSY=1（危险；本脚本不替你决定）"
            exit 3
        fi
        echo
        warn "★ 优先区间（PREFER_DEVS=$PREFER_DEVS）里的空闲卡不足 $TP 张 ⇒ 从**其余**空闲卡补了：${_rest# }"
        warn "  A3 是共用机：这些卡可能属于别人（本项目在 A3 的约定是 8–15）。"
        warn "  ⇒ 务必确认你有权使用；否则 Ctrl-C 后显式 DEVS=\"...\"，或等优先区间的卡空出来。"
    fi
    if [ "${_pn:-0}" -lt "$TP" ]; then
        # 双保险：上面任何分支算完仍不足都不许硬凑
        bad "空闲卡不够 $TP 张 ⇒ 不能自动选"
        echo "        修法：① 先看谁占着：bash tools/list_chips.sh"
        echo "              ② 若里面有**你自己的**残留进程，停掉后重试"
        echo "              ③ 卡够但不想用自动选的：显式 DEVS=\"<8 个卡号>\""
        echo "              ④ 确实要带别人的占用起服务：ALLOW_BUSY=1（危险；本脚本不替你决定）"
        exit 3
    fi
    DEVS=$(printf '%s\n' $_pick | head -"$TP" | tr '\n' ' ')
    DEVS=${DEVS% }
    echo
    echo "  ★★ 我替你选了这 $TP 张：DEVS=\"$DEVS\""
    echo "     （优先区间 PREFER_DEVS=\"$PREFER_DEVS\"；改它或直接给 DEVS= 都能覆盖）"
    echo "     —— 若这些卡不属于你（A3 是共用机），请 Ctrl-C 后用 DEVS=\"...\" 显式指定；"
    echo "        scripts/serve_a3.sh 在真起服前会再查一次占用，但'别人刚空出来的卡'它拦不住。"
fi
export DEVS

# ================================================================ ④ 干跑（把真正生效的配置打全）
say "④ 干跑 DRY_RUN=1（不碰 docker、不占卡；下面这些值是**容器里真会生效**的）"
_run() {   # $1 = DRY_RUN(1/0)
    env MODEL="$MODEL" DEVS="$DEVS" TP="$TP" IMAGE="$IMAGE" PORT="$PORT" \
        ENGRAM="$ENGRAM" DRAFT_GRAPH="$DRAFT_GRAPH" DROPCACHE="$DROPCACHE" \
        MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" SERVED_NAME="$SERVED_NAME" \
        DRY_RUN="$1" bash "$PKG/scripts/serve_a3.sh"
}
if _run 1 >"$DRY_LOG" 2>&1; then
    ok "干跑 rc=0（日志：$DRY_LOG）"
else
    bad "干跑 rc≠0 ⇒ 别起服务，先看日志：$DRY_LOG"
    tail -25 "$DRY_LOG" | sed 's/^/        /'
    exit 2
fi

# 判据绑**内容**（不是"我传了这个变量"）：把要生效的关键值从干跑日志里真的读出来
_expect() {  # $1 = 必须出现的字面量  $2 = 说明
    if grep -qF -- "$1" "$DRY_LOG"; then ok "$2"
    else bad "$2（干跑日志里找不到：$1）"; fi
}
_expect "devs='$DEVS'"                        "选中的卡真的传下去了"
_expect "PATCH_MODE=mount"                    "mount 模式（官方镜像 + 挂载补丁）"
_expect "served_name=$SERVED_NAME"            "服务名（API body 的 \"model\" 字段必须填它）"
_expect "max_len=$MAX_LEN"                    "上下文长度"
_expect "DRAFT_GRAPH=$DRAFT_GRAPH"            "draft 入图开关真的传下去了"
if [ "$DRAFT_GRAPH" = "1" ]; then
    echo
    warn "★★ 你显式开了 DRAFT_GRAPH=1。A3 上**实测过它是坏的**：A(接受长度)=1.06–1.08（draft 不产出）、"
    warn "   真实吞吐 −2.2×（ms/step 反而更好看，因为每步只出 1.08 token）。"
    warn "   ⇒ 起服后**必须**跑： bash tools/draft_graph_guard.sh   （判据 A ≥ 1.3；不到就退回 0）"
fi
if [ "$_fail" != "0" ]; then
    echo
    bad "④ 干跑内容判据有 $_fail 项不过 ⇒ 停（日志：$DRY_LOG）"
    exit 2
fi

# ================================================================ ⑤ 起服（LAUNCH=1）
if [ "$LAUNCH" != "1" ]; then
    say "⑤ 干跑通过 —— **默认不起服务**"
    echo
    echo "  真要起服，把下面这条原样再跑一次（多了 LAUNCH=1）："
    echo
    echo "      MODEL=$MODEL LAUNCH=1 \\"
    [ -n "${DEVS:-}" ] && echo "      DEVS=\"$DEVS\" \\"
    echo "      ENGRAM=$ENGRAM DRAFT_GRAPH=$DRAFT_GRAPH MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS \\"
    echo "      DROPCACHE=$DROPCACHE SERVED_NAME=$SERVED_NAME \\"
    echo "      bash tools/deploy_a3.sh"
    echo
    echo "  （干跑日志留着别删：$DRY_LOG）"
    exit 0
fi

say "⑤ 真起服（前台等待就绪；首次**冷编译静态内核** 15–20 min 属正常）"
echo "  起服后的三条自检（**服务活着时**另开终端跑）："
echo "      bash tools/attach_test.sh                                  # 端到端连通 + 文本/工具/图片"
echo "      bash tests/t_quote.sh                                      # 时延（ms/step）"
echo "      python3 tools/bench_concurrency.py --base-url http://127.0.0.1:$PORT --model $SERVED_NAME"
echo
_run 0
rc=$?
if [ "$rc" != "0" ]; then
    echo
    bad "起服未成功（rc=$rc）—— 常见原因与判读见 docs/A3-DEPLOY.md §5"
fi
exit "$rc"
