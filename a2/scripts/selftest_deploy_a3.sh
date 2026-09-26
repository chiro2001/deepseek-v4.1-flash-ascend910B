#!/usr/bin/env bash
# selftest_deploy_a3.sh —— deploy_a3.sh 的**沙箱自测**（零真机、零 NPU、零容器）
#
# 为什么需要：`deploy_a3.sh` 是**在一台啥都没有的新机器上**跑的第一条命令 ——
#   它自己的每条门都必须"该拦的拦住、该过的过"。而它的门里有一堆"外部命令"
#   （docker / npu-smi / 选卡 / 影子包），真机验证贵且不可复现 ⇒ 用**桩**把每条门都走一遍。
#
# 用法： bash a2/scripts/selftest_deploy_a3.sh
# 退出码：0 = 全过；9 = 有失败
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_REPO=$(cd "$HERE/../.." && pwd)
SCRIPT=${SCRIPT_SRC:-$SRC_REPO/a2/scripts/deploy_a3.sh}
[ -f "$SCRIPT" ] || { echo "⛔ 找不到待测脚本：$SCRIPT" >&2; exit 9; }

V=0
F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
R=$T/repo

# ---------------------------------------------------------------- 沙箱夹具
# 一个"最小的看起来像真仓"的树 + 三个桩（docker / npu-smi / 选卡 / 影子包 / 起服包装）。
mk_repo() {
    rm -rf "$R"; mkdir -p "$R"/{scripts,a2/scripts,a2/patches/kv8-offload-pool,a2/patches/kv8-graphsafe,tools,bin}
    cp "$SCRIPT" "$R/a2/scripts/deploy_a3.sh"
    for f in scripts/serve_a3.sh scripts/serve_a2.sh scripts/serve_v2.sh \
             a2/scripts/serve_a2_offload.sh a2/scripts/serve_a3_offload.sh \
             a2/scripts/make_shadow_pkg.sh tools/list_chips.sh tools/check_model_dir.sh \
             a2/patches/0001-offload-scheduler.patch.py \
             a2/patches/kv8-offload-pool/p2_pool.py \
             a2/patches/kv8-graphsafe/dsa_v41.py; do
        printf '#!/usr/bin/env bash\n:\n' > "$R/$f"
    done

    # ── 桩：npu-smi（只要有这个命令，选卡走 list_chips 桩）
    printf '#!/usr/bin/env bash\necho "stub npu-smi"\n' > "$R/bin/npu-smi"

    # ── 桩：docker（image inspect / info / version / pull 四种用法）
    cat > "$R/bin/docker" <<'STUB'
#!/usr/bin/env bash
case "$1 $2" in
  "info "*|"info")  exit "${STUB_DOCKER_INFO_RC:-0}" ;;
  "version --format") echo "29.0.0-stub"; exit 0 ;;
  "image inspect")
      [ "${STUB_IMAGE_PRESENT:-1}" = "1" ] && exit 0 || exit 1 ;;
  "pull "*)  echo "[stub docker] pulled $2"; exit "${STUB_PULL_RC:-0}" ;;
esac
exit 0
STUB

    # ── 桩：选卡（空闲卡由 STUB_FREE_CHIPS 控制）
    cat > "$R/tools/list_chips.sh" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = "--free" ] && printf '%s\n' ${STUB_FREE_CHIPS:-}
exit 0
STUB

    # ── 桩：模型自检（rc 由 STUB_MODEL_CHECK_RC 控制）
    cat > "$R/tools/check_model_dir.sh" <<'STUB'
#!/usr/bin/env bash
echo "[stub check_model_dir] $1"
exit "${STUB_MODEL_CHECK_RC:-0}"
STUB

    # ── 桩：造影子包（把 PKG/DST 记下来；rc 可控）
    cat > "$R/a2/scripts/make_shadow_pkg.sh" <<'STUB'
#!/usr/bin/env bash
echo "[stub make_shadow] PKG=$PKG DST=$DST" >> "${STUB_LOG:-/dev/null}"
[ "${STUB_SHADOW_RC:-0}" = "0" ] || { echo "[stub] 造影子包失败"; exit 2; }
mkdir -p "$DST/scripts"; echo "✓ ①/②/③/④ 块已插入"
exit 0
STUB

    # ── 桩：起服包装（★ 输出必须与真包装**同形**：干跑日志里的那些字面量是 deploy 的判据）
    cat > "$R/a2/scripts/serve_a3_offload.sh" <<'STUB'
#!/usr/bin/env bash
echo "[stub serve_a3_offload] DRY=${DRY:-<unset>}"
echo "${DRY:-}" >> "${STUB_LOG:-/dev/null}"
[ "${STUB_SERVE_RC:-0}" = "0" ] || { echo "[stub] 起服包装失败"; exit 2; }
if [ "${DRY:-0}" = "1" ]; then
  echo "[a2-dry] image=$IMAGE name=dsv41-a3 port=8020 served_name=deepseek-v41 devs='$DEVS' max_len=1048576"
  echo "[a2-dry] DRAFT_GRAPH=$DRAFT_GRAPH PYTHON_PGO=0 PATCH_MODE=mount"
  echo "[a2-dry] DEVS=$DEVS"
  if [ "${OFFLOAD:-1}" = "1" ]; then
    echo "[serve_a2] [A2-OFFLOAD] scheduler.py <- /x/scheduler.py"
    echo "KV_ARGS_EXTRA='--prefix-match-unit 32 --kv-transfer-config {\"kv_connector\":\"OffloadingConnector\"}'"
    echo "[serve_a2] [L1-POOL] 已挂 6 文件"
  fi
  echo "[a2-dry] OK"
fi
exit 0
STUB
    chmod +x "$R/bin/"* "$R/tools/"*.sh "$R/a2/scripts/"*.sh "$R/scripts/"*.sh
}

mk_model() {   # 造一个"真"模型目录：文件 + 一个指向外部目录的软链
    local m="$T/models/out/child"
    rm -rf "$T/models"; mkdir -p "$m" "$T/models/out/external"
    echo '{"model_type":"deepseek_v41"}' > "$m/config.json"
    echo 'w' > "$m/weights-00001.safetensors"
    ln -s "$T/models/out/external" "$m/linked_dir"
    printf '%s' "$m"
}

# 统一入口：清环境 ⇒ 跑 deploy_a3.sh
run_deploy() {
    local name="$1"; shift
    local m="${MODEL_OVERRIDE:-$M}"
    OUTF="$T/out_$name.txt"
    ( cd "$R" && env PATH="$R/bin:$PATH" STUB_LOG="$T/stub.log" SHADOW_DST="$T/shadow-a3" \
        DRY_LOG="$T/dry_$name.log" MODEL="$m" "$@" \
        bash a2/scripts/deploy_a3.sh ) >"$OUTF" 2>&1
    rc=$?
    echo "--- [$name] rc=$rc"
    return $rc
}
check() {  # <名> <期望rc> <实际rc> [必须出现] [禁止出现]
    local name="$1" want="$2" got="$3" need="${4:-}" deny="${5:-}"
    _dbg() { printf '        └── 实际输出尾部：\n'; tail -6 "$OUTF" 2>/dev/null | sed 's/^/            /'; }
    if [ "$want" != "$got" ]; then bad "$name（rc=$got 期望 $want）"; _dbg; return; fi
    if [ -n "$need" ] && ! grep -qF -- "$need" "$OUTF"; then bad "$name（少了判据：$need）"; _dbg; return; fi
    if [ -n "$deny" ] && grep -qF -- "$deny" "$OUTF"; then bad "$name（出现禁止项：$deny）"; _dbg; return; fi
    ok "$name"
}
check_log() {  # <名> <干跑日志路径> <必须出现> —— 判据绑**下游真收到的值**（不是 deploy 的自述）
    local name="$1" log="$2" need="$3"
    if [ -f "$log" ] && grep -qF -- "$need" "$log"; then ok "$name"
    else bad "$name（干跑日志 $log 里没有：$need）"; fi
}

mk_repo; M=$(mk_model)

# ================================================================ ① 用法
say "① 不给 MODEL ⇒ rc=64，并指出搬运清单在哪"
# 直接跑（不带 MODEL）
( cd "$R" && env PATH="$R/bin:$PATH" bash a2/scripts/deploy_a3.sh ) >"$T/out_nom.txt" 2>&1; rc=$?
OUTF="$T/out_nom.txt"
check "① 缺 MODEL" 64 "$rc" "A3-DEPLOY.md"

# ================================================================ ② 镜像缺失（新机器第一个真坑）
say "② 镜像不在本地且没 PULL=1 ⇒ rc=2，并打印 docker pull 的确切命令"
run_deploy noimage STUB_IMAGE_PRESENT=0 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "② 镜像缺失" 2 "$rc" "docker pull quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3" "① 有 0 项"

# ================================================================ ③ PULL=1 能拉
say "③ PULL=1 ⇒ 真的调 docker pull，然后继续往后走"
run_deploy pull STUB_IMAGE_PRESENT=0 PULL=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "③ PULL=1" 0 "$rc" "[stub docker] pulled" "① 有 "

# ================================================================ ④ 模型断链
say "④ 模型目录里有断链 ⇒ rc=2，且把断链目标打出来（新机器只搬一个目录的典型症状）"
M2="$T/models/broken"; mkdir -p "$M2"; echo '{}' > "$M2/config.json"
ln -s "$T/models/out/does-not-exist" "$M2/weights-link"
run_deploy broken MODEL="$M2" STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "④ 断链拦住" 2 "$rc" "个**断链**" "① 有 0 项"

# ================================================================ ⑤ 模型自检致命
say "⑤ tools/check_model_dir.sh rc=1 ⇒ 拦住（别等 worker 加载期才炸）"
run_deploy badmodel STUB_MODEL_CHECK_RC=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑤ 模型自检致命" 2 "$rc" "模型目录自检**致命**失败" "① 有 0 项"

# ================================================================ ⑥ 空闲卡不够
say "⑥ 空闲卡只有 6 张 ⇒ rc=3（不许硬凑、不许抢别人的卡）"
run_deploy fewchips STUB_FREE_CHIPS="0 1 2 3 4 5"; rc=$?
check "⑥ 空闲卡不够" 3 "$rc" "空闲卡不够 8 张"

# ================================================================ ⑦ 自动选卡 + 干跑全绿
say "⑦ 空闲 10 张 ⇒ 自动选前 8 张，干跑 rc=0，内容判据逐条过"
run_deploy auto STUB_FREE_CHIPS="0 1 2 3 4 5 6 7 8 9"; rc=$?
check "⑦a 自动选卡干跑" 0 "$rc" "我替你选了这 8 张" "FAIL"
check "⑦b deploy 自述：卡传下去了" 0 "$rc" "选中的卡真的传下去了"
check "⑦c deploy 自述：mount 模式" 0 "$rc" "mount 模式（官方镜像 + 挂载补丁）"
check "⑦d deploy 自述：服务名"     0 "$rc" "服务名 = deepseek-v41"
check "⑦e deploy 自述：卸载件已挂" 0 "$rc" "卸载补丁已挂"
check "⑦f 默认不起服务"   0 "$rc" "LAUNCH=1"
# ★ 再加一层：判据绑**下游真收到的值**（deploy 自述通过 ≠ 值真的传下去了）
check_log "⑦g 下游真收到 DEVS"   "$T/dry_auto.log" "DEVS=0 1 2 3 4 5 6 7"
check_log "⑦h 下游真收到卸载参数" "$T/dry_auto.log" "kv-transfer-config"
check_log "⑦i 下游真收到 DRAFT_GRAPH=1" "$T/dry_auto.log" "DRAFT_GRAPH=1"

# ================================================================ ⑧ DEVS 数量不匹配
say "⑧ DEVS 给了 4 张而 TP=8 ⇒ rc=2（宁可不跑，不许静默按别的卡数跑）"
run_deploy devsmismatch DEVS="0 1 2 3" STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑧ DEVS/TP 不匹配" 2 "$rc" "数量不匹配"

# ================================================================ ⑨ OFFLOAD=0 的负判据
say "⑨ OFFLOAD=0 ⇒ 干跑日志里不许出现卸载件（单变量关干净）"
run_deploy off0 OFFLOAD=0 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑨a OFFLOAD=0 干跑通过" 0 "$rc" "OFFLOAD=0：kv-transfer-config 已消失"
check "⑨b 无卸载挂载"         0 "$rc" "OFFLOAD=0：offloading/scheduler.py 已消失"
if grep -qF "A2-OFFLOAD" "$OUTF"; then bad "⑨c OFFLOAD=0 但仍出现 A2-OFFLOAD"; else ok "⑨c 干跑日志里没有 A2-OFFLOAD"; fi

# ================================================================ ⑩ LAUNCH=1 真的走到起服
say "⑩ LAUNCH=1 ⇒ 真跑（桩记录 DRY=0），而不是只干跑"
rm -f "$T/stub.log"
run_deploy launch LAUNCH=1 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑩a LAUNCH=1 rc=0" 0 "$rc" "真起服（前台等待就绪"
if grep -qx "0" "$T/stub.log" 2>/dev/null; then ok "⑩b 起服包装收到 DRY=0（真起服，不是干跑）"
else bad "⑩b 起服包装没收到 DRY=0（拿到的：$(tr '\n' ' ' < "$T/stub.log" 2>/dev/null)）"; fi
if grep -qx "1" "$T/stub.log" 2>/dev/null; then ok "⑩c 起服前先做了一次干跑（DRY=1）"; else bad "⑩c 起服前没干跑"; fi

# ================================================================ ⑪ 影子包失败要拦住
say "⑪ 造影子包失败 ⇒ rc=2（新机器上这一步失败 = 所有补丁都不会挂）"
run_deploy shadowfail STUB_SHADOW_RC=2 STUB_FREE_CHIPS="0 1 2 3 4 5 6 7"; rc=$?
check "⑪ 影子包失败拦住" 2 "$rc" "make_shadow_pkg.sh 失败"

echo
echo "=============== 通过 $V 条 ==============="
[ "$V" -ge 18 ] && { echo "✅ 自测全过"; exit 0; } || { echo "❌ 不合格（通过数 $V < 18）"; exit 9; }
