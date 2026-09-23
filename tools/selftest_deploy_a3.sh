#!/usr/bin/env bash
# =============================================================================
# selftest_deploy_a3.sh —— `tools/deploy_a3.sh` 的**沙箱自测**（零真机、零 NPU、零容器、零网络）
#
# 为什么需要它：`deploy_a3.sh` 是**一台什么都没有的新机器上跑的第一条命令**。
#   它自己的每条门都必须"该拦的拦住、该过的过"；而它的门里有一堆外部依赖
#   （docker / npu-smi / 选卡 / 模型自检 / 起服入口）⇒ 真机验证又贵又不可复现。
#   这里全部用**桩**替换，把每条门都走一遍；判据尽量绑在**下游真收到的值**上
#   （而不是"deploy 自己的自述"）—— 这是本仓反复栽过的那类坑（"我传了" ≠ "生效了"）。
#
# 用法： bash tools/selftest_deploy_a3.sh
# 退出码：0 = 全过；9 = 有失败
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_PKG=$(cd "$HERE/.." && pwd)
SCRIPT=${SCRIPT_SRC:-$SRC_PKG/tools/deploy_a3.sh}
[ -f "$SCRIPT" ] || { echo "⛔ 找不到待测脚本：$SCRIPT" >&2; exit 9; }

V=0
F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }
OUTF=""

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
R=$T/pkg

# ---------------------------------------------------------------- 夹具
mk_pkg() {
    rm -rf "$R"
    mkdir -p "$R"/{scripts,tools,patches/files,bin}
    cp "$SCRIPT" "$R/tools/deploy_a3.sh"

    # 真件（内容判据要检查的这些文件必须存在且非空）
    for f in scripts/serve_a3.sh scripts/serve_a2.sh scripts/serve_v2.sh \
             tools/list_chips.sh tools/check_model_dir.sh \
             patches/files/engram_hash.py patches/files/engram_jit_kernel.py \
             patches/files/model.py patches/files/token_dispatcher_moemask.py; do
        printf '#!/usr/bin/env bash\n:\n' > "$R/$f"
    done

    printf '#!/usr/bin/env bash\necho "stub npu-smi"\n' > "$R/bin/npu-smi"

    cat > "$R/bin/docker" <<'STUB'
#!/usr/bin/env bash
case "$1 $2" in
  "info "*|"info")     exit "${STUB_DOCKER_INFO_RC:-0}" ;;
  "version --format")  echo "0.0.0-stub"; exit 0 ;;
  "image inspect")     [ "${STUB_IMAGE_PRESENT:-1}" = "1" ] && exit 0 || exit 1 ;;
  "pull "*)            echo "[stub docker] pulled $2"; exit "${STUB_PULL_RC:-0}" ;;
esac
exit 0
STUB

    cat > "$R/tools/list_chips.sh" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = "--free" ] && printf '%s\n' ${STUB_FREE_CHIPS:-}
exit 0
STUB

    cat > "$R/tools/check_model_dir.sh" <<'STUB'
#!/usr/bin/env bash
echo "[stub check_model_dir] $1"
exit "${STUB_MODEL_CHECK_RC:-0}"
STUB

    # ★ 起服入口桩：输出必须与真 serve_a2.sh 的 DRY_RUN 出口**同形**
    #   （deploy 的 _expect 判据就绑在这些字面量上；桩不同形 ⇒ 自测变成"测桩"）。
    cat > "$R/scripts/serve_a3.sh" <<'STUB'
#!/usr/bin/env bash
echo "[serve_a3] DEVS='${DEVS:-}'"
echo "$DRY_RUN" >> "${STUB_LOG:-/dev/null}"
echo "${DROPCACHE:-<unset>}" >> "${STUB_LOG_DC:-/dev/null}"
[ "${STUB_SERVE_RC:-0}" = "0" ] || { echo "[stub] 起服包装失败"; exit 2; }
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[a2-dry] OK"
  echo "[a2-dry] image=${IMAGE:-} name=dsv41-a3 port=${PORT:-} served_name=${SERVED_NAME:-} devs='${DEVS:-}' util=0.90 max_len=${MAX_LEN:-}"
  echo "[a2-dry] DROPCACHE=${DROPCACHE:-<unset>}（起服前清 page cache；0 关闭）"
  echo "[a2-dry] MOE_ZERO=0 MOE_NF=0 DRAFT_GRAPH=${DRAFT_GRAPH:-} PYTHON_PGO=0 PATCH_MODE=${PATCH_MODE:-mount}"
fi
exit 0
STUB
    chmod +x "$R/bin/"* "$R/tools/"*.sh "$R/scripts/"*.sh
}

mk_model() {   # 造一个"真"模型目录：普通文件 + 一个指向外部目录的软链
    local m="$T/models/out/child"
    rm -rf "$T/models"; mkdir -p "$m" "$T/models/out/external"
    echo '{"model_type":"deepseek_v41"}' > "$m/config.json"
    echo w > "$m/weights-00001.safetensors"
    ln -s "$T/models/out/external" "$m/linked_dir"
    printf '%s' "$m"
}

run_deploy() {   # <名> <env...>
    local name="$1"; shift
    OUTF="$T/out_$name.txt"
    ( cd "$R" && env PATH="$R/bin:$PATH" STUB_LOG="$T/stub_$name.log" \
        STUB_LOG_DC="$T/dc_$name.log" MODEL="$M" \
        DRY_LOG="$T/dry_$name.log" \
        "$@" bash tools/deploy_a3.sh ) >"$OUTF" 2>&1
    rc=$?
    echo "--- [$name] rc=$rc"
    return $rc
}
check() {   # <名> <期望rc> <实际rc> [必须出现] [禁止出现]
    local name="$1" want="$2" got="$3" need="${4:-}" deny="${5:-}"
    _dbg() { printf '        └── 实际尾部：\n'; tail -6 "$OUTF" 2>/dev/null | sed 's/^/            /'; }
    if [ "$want" != "$got" ]; then bad "$name（rc=$got 期望 $want）"; _dbg; return; fi
    if [ -n "$need" ] && ! grep -qF -- "$need" "$OUTF"; then bad "$name（少了判据：$need）"; _dbg; return; fi
    if [ -n "$deny" ] && grep -qF -- "$deny" "$OUTF"; then bad "$name（出现禁止项：$deny）"; _dbg; return; fi
    ok "$name"
}
check_dry() {   # <名> <干跑日志> <必须出现> —— ★ 绑**下游真收到的值**
    local name="$1" log="$2" need="$3"
    if [ -f "$log" ] && grep -qF -- "$need" "$log"; then ok "$name"
    else bad "$name（下游干跑日志 $log 里没有：$need）"; fi
}

mk_pkg; M=$(mk_model)

# ================================================================ ① 用法
say "① 不给 MODEL ⇒ rc=64，并指出搬运清单在哪"
( cd "$R" && env PATH="$R/bin:$PATH" bash tools/deploy_a3.sh ) >"$T/out_nom.txt" 2>&1; rc=$?
OUTF="$T/out_nom.txt"; check "① 缺 MODEL" 64 "$rc" "A3-DEPLOY.md"

# ================================================================ ② docker 不可用
say "② docker 守护进程不可达 ⇒ rc=2（新机器最常见：用户不在 docker 组）"
run_deploy nodocker STUB_DOCKER_INFO_RC=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "② docker 不可用" 2 "$rc" "docker 不可用" "① 有 0 项"

# ================================================================ ③ 镜像缺失 / pull
say "③ 镜像不在本地且没 PULL=1 ⇒ rc=2，并打印 docker pull 的确切命令"
run_deploy noimage STUB_IMAGE_PRESENT=0 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "③ 镜像缺失拦住" 2 "$rc" "docker pull quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3" "① 有 0 项"

say "③b PULL=1 ⇒ 真的调 docker pull，然后继续往后走"
run_deploy pull STUB_IMAGE_PRESENT=0 PULL=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "③b PULL=1 起效" 0 "$rc" "[stub docker] pulled" "① 有 "

# ================================================================ ④ 模型
say "④ 模型目录有断链 ⇒ rc=2（新机器"只搬一个目录"的典型症状），并打出断链目标"
M2="$T/models/broken"; mkdir -p "$M2"; echo '{}' > "$M2/config.json"
ln -s "$T/models/out/does-not-exist" "$M2/weights-link"
( cd "$R" && env PATH="$R/bin:$PATH" STUB_LOG=/dev/null STUB_LOG_DC=/dev/null \
    MODEL="$M2" STUB_FREE_CHIPS="0 1 2 3 4 5 6 7" \
    bash tools/deploy_a3.sh ) >"$T/out_broken.txt" 2>&1; rc=$?
OUTF="$T/out_broken.txt"; M="$M"      # 复原 M 给后续用例
check "④ 断链拦住" 2 "$rc" "个**断链**" "① 有 "

say "④b 外部依赖目录必须被列出来（新机器要连着一起搬）"
( cd "$R" && env PATH="$R/bin:$PATH" STUB_LOG=/dev/null STUB_LOG_DC=/dev/null \
    MODEL="$M" STUB_FREE_CHIPS="0 1 2 3 4 5 6 7" \
    bash tools/deploy_a3.sh ) >"$T/out_ext.txt" 2>&1; rc=$?
OUTF="$T/out_ext.txt"
check "④b 列出外部依赖目录" 0 "$rc" "$T/models/out/external"

say "④c tools/check_model_dir.sh rc=1 ⇒ 拦住（别等 worker 加载期才炸）"
run_deploy badmodel STUB_MODEL_CHECK_RC=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "④c 模型自检致命" 2 "$rc" "模型目录自检**致命**失败" "① 有 "

# ================================================================ ⑤ 选卡
say "⑤ 空闲卡只有 6 张 ⇒ rc=3（不许硬凑、不许抢别人的卡）"
# 注：优先区间 8–15 全空时才算"够"；这里给 6 张 ⇒ 怎么算都不足
run_deploy fewchips STUB_FREE_CHIPS="0 1 2 3 4 5"; rc=$?
check "⑤ 空闲卡不够" 3 "$rc" "空闲卡不够 8 张"

say "⑤b DEVS 给 4 张而 TP=8 ⇒ rc=2（宁可不跑，不许静默按别的卡数跑）"
run_deploy devsmismatch DEVS="0 1 2 3" STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑤b DEVS/TP 不匹配" 2 "$rc" "数量不匹配"

say "⑤c 空闲 10 张 ⇒ 自动选前 8 张，且**下游真收到**这 8 张"
run_deploy auto STUB_FREE_CHIPS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"; rc=$?
check "⑤c 自动选卡" 0 "$rc" "我替你选了这 8 张" "FAIL"
check_dry "⑤d 下游收到 DEVS"        "$T/dry_auto.log" "devs='8 9 10 11 12 13 14 15'"

say "⑤e ★ 全空闲时**必须优先选 8–15**（'空闲' ≠ '可以拿'：A3 上 0–7 不是我们的）"
check "⑤e 选中优先区间" 0 "$rc" '我替你选了这 8 张：DEVS="8 9 10 11 12 13 14 15"'
if grep -qF "从**其余**空闲卡补了" "$OUTF"; then bad "⑤f 优先区间够用时却去补别的卡"
else ok "⑤f 优先区间够用时没有碰区间外的卡"; fi

say "⑤g 优先区间只剩 3 张空闲 ⇒ 用其余空闲卡补齐，但**必须响亮警告**"
run_deploy prefer STUB_FREE_CHIPS="0 1 2 3 4 8 9 10 11"; rc=$?
check "⑤g 不足时补齐 + 警告" 0 "$rc" "从**其余**空闲卡补了：0 1 2 3"
check_dry "⑤h 补齐后下游收到 8 张" "$T/dry_prefer.log" "devs='8 9 10 11 0 1 2 3'"

# ================================================================ ⑥ 干跑内容判据（关键配置真的生效）
say "⑥ 干跑内容判据：mount / 服务名 / 1M / DRAFT_GRAPH 默认 0 / 共享机不动 page cache"
check_dry "⑥a mount 模式"       "$T/dry_auto.log" "PATCH_MODE=mount"
check_dry "⑥b 服务名"           "$T/dry_auto.log" "served_name=deepseek-v41"
check_dry "⑥c 上下文=A3 已验证口径 133120（不是 A2 的 1M）" "$T/dry_auto.log" "max_len=133120"
check_dry "⑥d DRAFT_GRAPH=0（A3 上 1 是坏的：A≈1.06、吞吐 −2.2×）" "$T/dry_auto.log" "DRAFT_GRAPH=0"
check_dry "⑥e DROPCACHE=0（共用机不清整机 page cache）" "$T/dry_auto.log" "DROPCACHE=0"
if grep -qx "0" "$T/dc_auto.log" 2>/dev/null; then ok "⑥f 下游真收到 DROPCACHE=0（不是只打印）"
else bad "⑥f 下游没收到 DROPCACHE=0（拿到：$(tr '\n' ' ' < "$T/dc_auto.log" 2>/dev/null)）"; fi

say "⑥g 显式 DRAFT_GRAPH=1 ⇒ 必须**响亮警告**并指向 draft_graph_guard.sh"
run_deploy dg1 DRAFT_GRAPH=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑥g DRAFT_GRAPH=1 警告" 0 "$rc" "draft_graph_guard.sh"
check_dry "⑥h 下游真收到 DRAFT_GRAPH=1" "$T/dry_dg1.log" "DRAFT_GRAPH=1"

say "⑥i 显式 MAX_LEN=1048576（A2 生产口径）⇒ 必须照传，不许被 A3 默认值吃掉"
run_deploy ml1m MAX_LEN=1048576 MAX_SEQS=4 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑥i 显式 1M（部署器自述）" 0 "$rc" "MAX_LEN=1048576"
check_dry "⑥j 下游真收到 1M" "$T/dry_ml1m.log" "max_len=1048576"

# ================================================================ ⑦ 起服
say "⑦ LAUNCH=1 ⇒ 真跑（桩记录 DRY_RUN=0），且起服前**先干跑一次**"
run_deploy launch LAUNCH=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑦a LAUNCH=1 rc=0" 0 "$rc" "真起服（前台等待就绪"
if grep -qx "0" "$T/stub_launch.log" 2>/dev/null; then ok "⑦b 下游收到 DRY_RUN=0（真起服）"
else bad "⑦b 下游没收到 DRY_RUN=0（拿到：$(tr '\n' ' ' < "$T/stub_launch.log" 2>/dev/null)）"; fi
if grep -qx "1" "$T/stub_launch.log" 2>/dev/null; then ok "⑦c 起服前先干跑了一次（DRY_RUN=1）"
else bad "⑦c 起服前没干跑"; fi

say "⑦d 起服入口失败（rc≠0）⇒ deploy 必须把 rc 传出来（不许假装成功）"
run_deploy servefail LAUNCH=1 STUB_SERVE_RC=2 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
# 注意：干跑也要 rc≠0 才会走到"起服未成功"；这里 STUB_SERVE_RC=2 让干跑就失败 ⇒ 期望 rc=2
check "⑦d 起服入口失败不被吞" 2 "$rc" ""

# ================================================================ ⑧ 默认不改别人状态
say "⑧ 默认（无 LAUNCH）**不许**调用真起服（DRY_RUN 只出现 1，不出现 0）"
if grep -qx "0" "$T/stub_auto.log" 2>/dev/null; then bad "⑧ 默认干跑却调了真起服"
else ok "⑧ 默认只干跑（下游只收到 DRY_RUN=1）"; fi

echo
echo "=============== 通过 $V 条 / 失败 $F 条 ==============="
if [ "$F" = "0" ] && [ "$V" -ge 20 ]; then echo "✅ 自测全过"; exit 0; fi
echo "❌ 不合格"; exit 9
