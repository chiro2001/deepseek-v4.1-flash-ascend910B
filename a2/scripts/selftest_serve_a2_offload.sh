#!/usr/bin/env bash
# selftest_serve_a2_offload.sh —— serve_a2_offload.sh 的**沙箱端到端自测**（不碰真环境）
#
# 为什么需要它（同一天两次、同一类事故）：
#   ① 头部打印引用了 `$OUT`（那只在 shadow 的 serve_a2.sh 里定义）⇒ `set -u` 崩溃；
#   ② 头部打印引用了 `$LAUNCH_DIR`，而它当时定义在**文件后半** ⇒ `line 247: LAUNCH_DIR: unbound variable`。
#   ⇒ 两次都是"变量在定义之前被使用"，而 `bash -n` **查不出来**（它只查语法，不查未定义变量）。
#   本测试把脚本放进**沙箱仓库**里真跑（DRY=1），任何未定义变量都会以非零退出 + "unbound" 暴露。
#
# 用法： bash a2/scripts/selftest_serve_a2_offload.sh
# 退出码：0 = 全过；9 = 有失败
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_REPO=$(cd "$HERE/../.." && pwd)
SCRIPT_REL=a2/scripts/serve_a2_offload.sh
V=0
say() { printf '\n==== %s ====\n' "$*"; }

setup_sandbox() {
    local T="$1"
    mkdir -p "$T/repo/a2/scripts" "$T/repo/a2/patches" \
             "$T/repo/a2/patches/kv8-graphsafe" \
             "$T/repo/a2/patches/kv8-int8-pkg/vllm_ascend/attention" \
             "$T/shadow/scripts" "$T/shadow/patches/files" "$T/bin"
    cp "${SCRIPT_SRC:-$SRC_REPO/$SCRIPT_REL}" "$T/repo/$SCRIPT_REL"   # ★ SCRIPT_SRC 可指向"待测脚本"，用于反向对照
    # ★ 用**真件**（不是桩）：内容判据要检查"有没有包路径回退"，桩里没有那个对象。
    for f in 0001-offload-scheduler.patch.py 0001b-offload-per-group-bpc-manager.patch.py \
             0001c-offload-per-group-bpc-hooks.patch.py 0002-offload-cpu-pool-host-registered.patch.py; do
        cp "$SRC_REPO/a2/patches/$f" "$T/repo/a2/patches/$f"
    done
    mkdir -p "$T/repo/a2/patches/kv8-offload-pool"
    cp "$SRC_REPO/a2/patches/kv8-offload-pool/"*.py "$T/repo/a2/patches/kv8-offload-pool/"
    cp "$SRC_REPO/a2/patches/kv8-graphsafe/dsa_v41.py" "$T/repo/a2/patches/kv8-graphsafe/dsa_v41.py"
    cp "$SRC_REPO/a2/patches/kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py" \
       "$T/repo/a2/patches/kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py"
    printf 'stub-engram-hash\n' > "$T/shadow/patches/files/engram_hash.py"
    printf 'stub-engram-jit\n'  > "$T/shadow/patches/files/engram_jit_kernel.py"
    cat > "$T/shadow/scripts/serve_a2.sh" <<'STUB'
#!/usr/bin/env bash
# ---------- [A2-OFFLOAD] 由 make_shadow_pkg.sh 注入（沙箱桩） ----------
# ★ 下面两个字符串是"影子包认不认 A2_*"自检门要求的内容判据（真实影子包里也有）
: "${A2_KV8_SWA:=0}" "${A2_GRAPH_SAFE:=0}"
# ★ 桩要打印**容器内目标路径**（wrapper 的 DRY 断言现在绑的是目标，不是源文件名）
# ★★ 并且**按 OFFLOAD_SCHED_PATCH / L1_POOL_PATCH 门控** —— 真实 shadow 就是这么做的
#    （`make_shadow_pkg.sh` 注入块里 `if [ "${OFFLOAD_SCHED_PATCH:-0}" = "1" ]`）。
#    桩若无条件打印，就会让"OFFLOAD=0 关得干净"这条断言**假失败**（本仓同族第 N 次：
#    **桩/门必须与真实对象同构**）。
echo "[a2-dry] MOUNTS(...)"
echo "  TP=${TP:-<unset>} DP=${DP:-<unset>} CPU_BIND=${CPU_BIND:-<unset>}"
if [ "${OFFLOAD_SCHED_PATCH:-0}" = "1" ]; then
echo "  -v /x/scheduler.py:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro"
echo "  -v /x/offloading_config.py:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro"
echo "  -v /x/cpu_spec.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/spec.py:ro"
echo "  -v /x/pgp_manager.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py:ro"
echo "  -v /x/p2_pool.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/p2_pool.py:ro"
echo "  -v /x/p2_worker.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/p2_worker.py:ro"
echo "  -v /x/cpu_npu.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro"
fi
if [ "${L1_POOL_PATCH:-0}" = "1" ]; then
echo "  -v /x/p2_worker.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/p2_worker.py:ro"
echo "  DRY_RUN=${DRY_RUN:-<unset>} PROFILE=${PROFILE:-<unset>} V41_PROFILE=${V41_PROFILE:-<unset>}"
echo "  DRAFT_GRAPH=${DRAFT_GRAPH:-<unset>}"
fi
# ★ 这两行是 **mount 模式指纹门**要 grep 的"挂载行"（真实 shadow 里由注入块生成）
echo "  MOUNTS+=(-v \"\$F/engram_hash.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py:rw\")"
echo "  MOUNTS+=(-v \"\$F/engram_jit_kernel.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_jit_kernel.py:ro\")"
STUB
    # ★ A3 的入口桩：只做 A3 特有的校验（DEVS 必填）+ 打印，然后 exec serve_a2.sh（与真实同形）
    cat > "$T/shadow/scripts/serve_a3.sh" <<'STUB3'
#!/usr/bin/env bash
[ -n "${DEVS:-}" ] || { echo "[serve_a3][FAIL] 必须显式指定 DEVS" >&2; exit 2; }
echo "[serve_a3] DEVS='$DEVS' PATCH_MODE=${PATCH_MODE:-<unset>} NAME=${NAME:-<unset>} PORT=${PORT:-<unset>} PYTHON_PGO=${PYTHON_PGO:-<unset>}"
exec bash "$(dirname "$0")/serve_a2.sh" "$@"
STUB3
    cat > "$T/bin/docker" <<STUB
#!/usr/bin/env bash
case "\$1 \$2" in
  "image inspect") exit 0 ;;
  "image ls")      echo "dsv41-a2:v9  deadbeef  1 minute ago"; exit 0 ;;
esac
if [ "\$1" = "run" ]; then
  case " \$* " in
    *engram_hash.py*)       md5sum "$T/shadow/patches/files/engram_hash.py"       | sed 's# .*#  /p#'; exit 0 ;;
    *engram_jit_kernel.py*) md5sum "$T/shadow/patches/files/engram_jit_kernel.py" | sed 's# .*#  /p#'; exit 0 ;;
  esac
  # ★ 挂载件"可导入性"预检：默认吐两行 OK-IMPORT；STUB_IMPORT_FAIL=1 时模拟失败
  case " \$* " in
    *import*and*vllm*|*import*and*pgp*|*OK-IMPORT*|*import*)
      if [ "\${STUB_IMPORT_FAIL:-0}" = "1" ]; then
        echo "ModuleNotFoundError: No module named 'pgp_manager'" >&2; exit 1
      fi
      echo "OK-IMPORT /vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py"
      echo "OK-IMPORT /vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_hooks.py"
      exit 0 ;;
  esac
  exit 0
fi
exit 0
STUB
    chmod +x "$T/bin/docker"
}

# 全局：最近一次运行的输出文件
OUTF=""
# run_case <名> <env...>  —— 用 DRY=1 跑；输出落 $OUTF；返回其 rc
run_case() {
    local name="$1"; shift
    local T; T=$(mktemp -d); setup_sandbox "$T"
    OUTF="$T/out.txt"
    ( cd "$T/repo" && env PATH="$T/bin:$PATH" SHADOW_PKG="$T/shadow" MODEL=/stub/model DRY=1 \
        "$@" bash "$SCRIPT_REL" ) >"$OUTF" 2>&1
    local rc=$?
    echo "--- [$name] rc=$rc（输出 $(wc -l <"$OUTF") 行）"
    return $rc
}

check() {  # <名> <期望rc> <实际rc> [必须出现] [禁止出现]
    local name="$1" want="$2" got="$3" need="${4:-}" deny="${5:-}"
    if [ "$want" != "$got" ]; then printf '⛔ [%s] rc=%s 期望 %s\n' "$name" "$got" "$want"; V=1; return; fi
    if [ -n "$need" ] && ! grep -qE "$need" "$OUTF"; then printf '⛔ [%s] 缺少 %s\n' "$name" "$need"; V=1; return; fi
    if [ -n "$deny" ] && grep -qE "$deny" "$OUTF"; then printf '⛔ [%s] 出现 %s\n' "$name" "$deny"; V=1; return; fi
    printf '✓ [%s]\n' "$name"
}

say "① 最小 DRY（ENGRAM=0）—— 主目标：未定义变量 + ★ DRAFT_GRAPH 默认必须是 1"
run_case base ENGRAM=0; check "① 最小 DRY" 0 $? 'L1_POOL_PATCH=1' 'unbound variable'
say "①b 输出尾部（看有没有 unbound）"; tail -3 "$OUTF"; grep -c "unbound variable" "$OUTF" || true

say "② int8 档 C（走 int8 修复件自检门）"
run_case int8 ENGRAM=0 KV8_SWA=1 KV8_RING_FP16=1; check "② int8 DRY" 0 $? 'int8 修复件' 'unbound variable'

say "③ PROFILE=1 透传"
run_case prof ENGRAM=0 PROFILE=1; check "③ PROFILE=1" 0 $? 'PROFILE=1' 'unbound variable'

say "④ V41_PROFILE=1 透传"
run_case prof2 ENGRAM=0 V41_PROFILE=1; check "④ V41_PROFILE=1" 0 $? 'PROFILE=1' 'unbound variable'

say "⑤ ENGRAM=1 指纹门（stub docker 两侧一致）"
run_case engram ENGRAM=1; check "⑤ ENGRAM=1" 0 $? '两项一致' 'unbound variable'

say "⑥ 起服对象缺 [A2-OFFLOAD] ⇒ 必须拒绝 rc=2"
T6=$(mktemp -d); setup_sandbox "$T6"
printf '#!/usr/bin/env bash\necho "旧影子包（无注入）"\n' > "$T6/shadow/scripts/serve_a2.sh"
OUTF="$T6/out.txt"
( cd "$T6/repo" && env PATH="$T6/bin:$PATH" SHADOW_PKG="$T6/shadow" MODEL=/stub/model DRY=1 ENGRAM=0 \
    bash "$SCRIPT_REL" ) >"$OUTF" 2>&1; rc=$?
tail -3 "$OUTF"
check "⑥ 缺注入拒绝" 2 "$rc" '没有 .A2-OFFLOAD.' 'unbound variable'

say "⑦ 挂载件只有裸 import、无包路径回退 ⇒ 内容判据必须拦 rc=2"
T7=$(mktemp -d); setup_sandbox "$T7"
cat > "$T7/mk_bad.py" <<'PYX'
import sys
T = sys.argv[1]
p = T + "/repo/a2/patches/0001-offload-scheduler.patch.py"
s = open(p, encoding="utf-8").read()
i = s.find("try:  # [A2-OFFLOAD]")
end = s.find(chr(10) + "    )" + chr(10), i) + len(chr(10) + "    )" + chr(10))
assert i > 0 and end > i, "fallback block not found"
bad = "from pgp_manager import BPC_BY_GROUP_KEY, bpc_map_from_extra  # noqa: E402" + chr(10)
open(p, "w", encoding="utf-8").write(s[:i] + bad + s[end:])
print("  已把 scheduler 回退成裸 import 版（反例）")
PYX
python3 "$T7/mk_bad.py" "$T7"
OUTF="$T7/out.txt"
( cd "$T7/repo" && env PATH="$T7/bin:$PATH" SHADOW_PKG="$T7/shadow" MODEL=/stub/model DRY=1 ENGRAM=0 \
    bash "$SCRIPT_REL" ) >"$OUTF" 2>&1; rc=$?
tail -3 "$OUTF"
check "⑦ 裸 import 无回退须拒绝" 2 "$rc" '只有裸 import' 'unbound variable'

say "⑧ DRAFT_GRAPH=0 ⇒ 必须响亮警告（四轴变三轴）"
run_case draft0 ENGRAM=0 DRAFT_GRAPH=0; check "⑧ DRAFT_GRAPH=0 须警告" 0 $? 'DRAFT_GRAPH=0 ⇒' 'unbound variable'

say "⑨ DROPCACHE=0 必须透传（不许静默仍然清 page cache）"
run_case nodrop ENGRAM=0 DROPCACHE=0; check "⑨ DROPCACHE=0 透传" 0 $? 'DROPCACHE=0' 'unbound variable'

say "⑩ P2_COMP_JSON 必须**按档位**推导（档 C=20 张量 / 档 B=16 张量）"
# 为什么单列一条：分量是"哪些组的张量集合完全相同"的等价类，**张量数随档位变**。
# 硬编码过档 B 的那套 ⇒ 档 C 起服会在 p2_pool.worker_rows() 里 fail-closed
# （`P2_COMP_JSON 把共享张量的组拆到了不同分量`，实测）。
run_case compc ENGRAM=0 KV8_SWA=1 KV8_RING_FP16=1
check "⑩a 档 C 分量" 0 $? 'comp=\[\[0,2,3,4,5,6,7,8,9,10,11\],\[1,12\]\]' 'unbound variable'
run_case compb ENGRAM=0
check "⑩b 档 B 分量" 0 $? 'comp=\[\[0\],\[1,2,3,4,5,6,7,8,9,10,11,12\]\]' 'unbound variable'

say "⑪ PLAT=a3 ⇒ 默认值必须整组切换（DEVS/入口/PATCH_MODE/PORT/DROPCACHE）"
# 为什么单列一条：★ 照 A2 的默认抄到 A3 上，`DEVS` 会去抢 **0–7 卡**（本仓红线：那不是我们的）；
# 且 `PATCH_MODE=baked` 在 A3 官方镜像上会**静默零补丁**（服务照常起、优化全不在）。
run_case a3 PLAT=a3 ENGRAM=0
check "⑪a A3 DEVS=8–15"  0 $? 'DEVS=8 9 10 11 12 13 14 15' 'unbound variable'
check "⑪b A3 入口"       0 $? '入口=serve_a3.sh'            'unbound variable'
check "⑪c A3 PATCH_MODE" 0 $? 'PATCH_MODE=mount'            'unbound variable'
run_case a3dp2 PLAT=a3 ENGRAM=0 DP=2
check "⑪c2 A3 DP=2 透传到 shadow" 0 $? 'DP=2' 'unbound variable'
run_case a3cpu0 PLAT=a3 ENGRAM=0 CPU_BIND=0
check "⑪c3 A3 CPU_BIND=0 透传到 shadow" 0 $? 'CPU_BIND=0' 'unbound variable'
run_case a3b PLAT=a3 ENGRAM=0
check "⑪d A3 DROPCACHE=0" 0 $? 'DROPCACHE=0'                'unbound variable'

say "⑫ A3 但 shadow 里缺 serve_a3.sh ⇒ 必须响亮失败（不许静默换入口）"
T12=$(mktemp -d); setup_sandbox "$T12"
mv "$T12/shadow/scripts/serve_a3.sh" "$T12/shadow/scripts/serve_a3.sh.moved"
OUTF="$T12/out.txt"
( cd "$T12/repo" && env PATH="$T12/bin:$PATH" SHADOW_PKG="$T12/shadow" MODEL=/stub/model DRY=1 ENGRAM=0 \
    bash "$SCRIPT_REL" --plat-unused ) >"$OUTF" 2>&1 || true
( cd "$T12/repo" && env PATH="$T12/bin:$PATH" SHADOW_PKG="$T12/shadow" MODEL=/stub/model DRY=1 ENGRAM=0 PLAT=a3 \
    bash "$SCRIPT_REL" ) >"$OUTF" 2>&1; rc=$?
tail -3 "$OUTF"
if [ "$rc" = "0" ]; then printf '⛔ [⑫ 缺 serve_a3.sh] 竟然 rc=0 —— 静默换入口/漏检\n'; V=1; else printf '✓ [⑫ 缺 serve_a3.sh 会失败]\n'; fi

say "⑬ OFFLOAD=0 ⇒ 必须**关干净**（不挂 offload 补丁、不带 kv-transfer-config）"
run_case onlyload0 ENGRAM=0 OFFLOAD=0; check "⑬a OFFLOAD=0 起服" 0 $? '关得干净' 'unbound variable'
if grep -q "offloading/scheduler.py" "$OUTF"; then printf '⛔ [⑬b MOUNTS 仍含 offloading]\n'; V=1; else printf '✓ [⑬b MOUNTS 无 offloading]\n'; fi
# ★ 判据：`KV_ARGS_EXTRA` 里不能再有 kv-transfer-config，也不能再挂池的补丁
run_case onlyload0b ENGRAM=0 OFFLOAD=0
if grep -E "KV_ARGS_EXTRA=.*kv-transfer-config" "$OUTF" >/dev/null; then
    printf '⛔ [⑬c KV_ARGS 仍带 kv-transfer-config]\n'; V=1
else printf '✓ [⑬c KV_ARGS 已去掉 kv-transfer-config]\n'; fi

say "⑬d ★ BAT_TOKENS 默认必须是 8192（= 模板默认；2048 是已知的长上下文退化开关）"
# 为什么单列：`8eb2613` 把**模板**默认改成 8192 修乱码，而本包装脚本是后来写的、
#   第 584 行**显式**传 BAT_TOKENS 下去 ⇒ 包装脚本的默认值必然覆盖模板。
#   曾经这里是 `:-2048` ⇒ 用包装脚本起服 = 退回修复前（1M 下 256 刀 ⇒ 通过率≈0）。
# ★ 判据绑**权威对象**：脚本里那行 `BAT_TOKENS=${BAT_TOKENS:-N}` 的 N。
#   （干跑输出里**没有**这个字符串 —— 第一版拿干跑日志判，误报"不是 8192"。）
# ★ 注意 `-oE '[0-9]+$'` 抓不到 —— 值后面紧跟 `}`，`$` 锚不匹配（实测踩到）。
#   改成在完整匹配 `BAT_TOKENS=${BAT_TOKENS:-NNN}` 上取 `:-` 与 `}` 之间的数字。
_bat_line=$(grep -oE 'BAT_TOKENS=\$\{BAT_TOKENS:-[0-9]+\}' "$SRC_REPO/$SCRIPT_REL" | head -1)
_bat_def=$(printf '%s' "$_bat_line" | sed -n 's/.*:-\{0,1\}\([0-9]\{1,\}\)}.*/\1/p')
if [ "${_bat_def:-}" = "8192" ]; then
    printf '✓ [⑬d-2 默认 BAT_TOKENS=8192（读脚本源码，值=%s）]\n' "$_bat_def"
else
    printf '⛔ [⑬d-2 默认 BAT_TOKENS=%s（期望 8192；2048 是长上下文退化开关）]\n' "${_bat_def:-读不到}"; V=1
fi
run_case batdefault ENGRAM=0; check "⑬d-1 起服 rc=0" 0 $? ''
# ★ 反向对照：默认 8192 时**不许**打印那条警告（否则说明警告条件写错了）
if grep -q "长上下文退化" "$OUTF"; then
    printf '⛔ [⑬d-2b 默认 8192 却打印了退化警告]\n'; V=1
else
    printf '✓ [⑬d-2b 默认 8192 不打印警告]\n'
fi
if grep -aoE "BAT_TOKENS=[0-9]+" "$OUTF" | grep -qv "BAT_TOKENS=8192"; then
    printf '⛔ [⑬d-3 出现了非 8192 的 BAT_TOKENS]\n'; V=1
else
    printf '✓ [⑬d-3 没有非 8192 的默认]\n'
fi
# 反例：显式给 2048 时必须**响亮警告**（不许静默退回）
run_case bat2048 ENGRAM=0 BAT_TOKENS=2048
if grep -q "长上下文退化" "$OUTF"; then printf '✓ [⑬d-4 显式 2048 会警告]\n'
else printf '⛔ [⑬d-4 显式 2048 没警告]\n'; V=1; fi

say "⑭ ★ PLAT=a3 × OFFLOAD=0 × int8 ⇒ A3 默认值全保留、只有卸载被关掉（**A3 测试脚本的核心判据**）"
# 为什么单列一条：A3 是**另一台机器、另一个镜像、另一批卡**，而 `OFFLOAD=0` 的实现是
#   "不导出两个补丁开关"。两者**相乘**的组合此前从未被测过 ⇒ 一旦哪天真在 A3 上用它排查，
#   可能踩到"关卸载把平台默认值也一起带偏"这类错（本仓同族第 N 次）。
run_case a3off0 PLAT=a3 ENGRAM=0 KV8_SWA=1 KV8_RING_FP16=1 OFFLOAD=0
check "⑭a A3+OFFLOAD=0 起服"            0 $? '关得干净'            'unbound variable'
check "⑭b A3 DEVS 未被带偏"             0 $? 'DEVS=8 9 10 11 12 13 14 15' 'unbound variable'
check "⑭c A3 PATCH_MODE=mount 保留"     0 $? 'PATCH_MODE=mount'     'unbound variable'
check "⑭d A3 入口仍是 serve_a3.sh"      0 $? 'serve_a3.sh'          'unbound variable'
check "⑭e int8 档 C 仍自报"             0 $? '档位        : C'      'unbound variable'
check "⑭f draft 入图仍是 1（四轴不许被卸载开关带走）" 0 $? 'draft 入图    : DRAFT_GRAPH=1' 'unbound variable'
check "⑭g 1M 几何保留"                  0 $? '上下文/并发   : 1048576 / 4' 'unbound variable'
check "⑭j A3 服务名=deepseek-v41（与 serve_a3.sh 及全仓工具一致）" 0 $? '服务名        : deepseek-v41' 'unbound variable'
# ★ 判据必须绑**真实对象**：① 挂载行的**容器内目标路径**；② `KV_ARGS_EXTRA=` 的**值**。
#   ★ 不许直接 grep `kv-transfer-config` —— OFFLOAD=0 的**说明文字**里就有这个词
#     （第一版就是这么写的，结果**误报**；同族教训见 logs/124 "判据绑错对象"）。
if grep -qE "/vllm-workspace/vllm/vllm/distributed/kv_transfer" "$OUTF"; then
    printf '⛔ [⑭h A3+OFFLOAD=0 仍挂了卸载件（挂载目标路径命中）]\n'; V=1
elif grep -qE "KV_ARGS_EXTRA=.*kv-transfer-config" "$OUTF"; then
    printf '⛔ [⑭h A3+OFFLOAD=0 仍带 kv-transfer-config（KV_ARGS_EXTRA 命中）]\n'; V=1
else printf '✓ [⑭h A3+OFFLOAD=0 无卸载件/无 kv-transfer-config（判据绑挂载目标与 KV_ARGS 值）]\n'; fi
if grep -q "L1 (P2_POOL_PATCH): 1" "$OUTF"; then
    printf '⛔ [⑭i 卸载关了但 L1 池补丁还开着（没有消费者的开关）]\n'; V=1
else printf '✓ [⑭i L1 池补丁随卸载一起关（不留无消费者的开关）]\n'; fi

say "结果"
if [ "$V" = "0" ]; then echo "✅ 全部通过"; else echo "⛔ 有用例失败"; fi
exit $((V * 9))
