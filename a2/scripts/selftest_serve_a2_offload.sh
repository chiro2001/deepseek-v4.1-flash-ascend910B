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
echo "[a2-dry] MOUNTS(18)"
echo "  -v /x/scheduler.py:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro"
echo "  -v /x/offloading_config.py:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro"
echo "  -v /x/cpu_spec.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/spec.py:ro"
echo "  -v /x/pgp_manager.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py:ro"
echo "  -v /x/p2_pool.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/p2_pool.py:ro"
echo "  -v /x/p2_worker.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/p2_worker.py:ro"
echo "  -v /x/cpu_npu.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro"
echo "  DRY_RUN=${DRY_RUN:-<unset>} PROFILE=${PROFILE:-<unset>} V41_PROFILE=${V41_PROFILE:-<unset>}"
echo "  DRAFT_GRAPH=${DRAFT_GRAPH:-<unset>}"
STUB
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

say "结果"
if [ "$V" = "0" ]; then echo "✅ 全部通过"; else echo "⛔ 有用例失败"; fi
exit $((V * 9))
