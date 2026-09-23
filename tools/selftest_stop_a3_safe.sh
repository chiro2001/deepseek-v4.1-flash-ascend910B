#!/usr/bin/env bash
# =============================================================================
# selftest_stop_a3_safe.sh —— `tools/stop_a3_safe.sh` 的沙箱自测（零真机、零容器、零 sudo）
#
# 为什么需要：停机脚本会在**已经出事的机器**上跑 —— 那时候人最急、最容易敲错。
#   所以它自己的每条台阶都必须被测过：docker/pgrep/pkill/sudo/ps 全用**桩**替换。
#
# 用法： bash tools/selftest_stop_a3_safe.sh
# 退出码：0 = 全过；9 = 有失败
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
SCRIPT=${SCRIPT_SRC:-$PKG/tools/stop_a3_safe.sh}
[ -f "$SCRIPT" ] || { echo "⛔ 找不到待测脚本：$SCRIPT" >&2; exit 9; }

V=0; F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }
OUTF=""

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
B=$T/bin; mkdir -p "$B"

# ---------------------------------------------------------------- 桩
# docker：行为由 STUB_* 控制
cat > "$B/docker" <<'STUB'
#!/usr/bin/env bash
echo "docker $*" >> "${STUB_LOG:-/dev/null}"
case "$1" in
  inspect)
      [ "${STUB_CTR_EXISTS:-1}" = "1" ] || exit 1
      case "$*" in
        *"{{.State.Status}}"*)        echo "${STUB_STATUS:-running}" ;;
        *"{{.HostConfig.CpusetCpus}}"*) echo "" ;;
        *"{{.HostConfig.CpusetMems}}"*) echo "" ;;
      esac
      exit 0 ;;
  stop)
      # ★ 忠实模型：只要 migratepages 还在，stop 就**一直**卡（不是"只卡第一次"）
      _st=$(cat "${STUB_STATE:-/dev/null}" 2>/dev/null || echo "")
      _hang="no"
      [ "${STUB_STOP_HANGS:-1}" = "1" ] && _hang="yes"
      case "$_st" in *killed*) _hang="no" ;; esac
      if [ "$_hang" = "yes" ]; then
          echo "Error response from daemon: cannot stop container: $3: tried to kill container, but did not receive an exit event" >&2
          exit 1
      fi
      echo "$3"; exit "${STUB_STOP_RC:-0}" ;;
  rm)
      # ★ 忠实模型：进程杀不动时，`docker rm -f` 同样拿不到 exit event（除非显式放宽）
      _st=$(cat "${STUB_STATE:-/dev/null}" 2>/dev/null || echo "")
      case "$_st" in
        *killed*) [ "${STUB_RM_OK:-1}" = "1" ] && exit 0 || exit 1 ;;
        *) [ "${STUB_RM_HANGS:-1}" = "1" ] && { echo "Error response from daemon: cannot remove container: tried to kill container, but did not receive an exit event" >&2; exit 1; } || exit 0 ;;
      esac ;;
esac
exit 0
STUB

# pgrep：migratepages / worker / engine 三种查询
cat > "$B/pgrep" <<'STUB'
#!/usr/bin/env bash
echo "$*" >> "${STUB_PG_LOG:-/dev/null}"
case "$*" in
  "-x migratepages")
      [ "${STUB_MP:-0}" = "1" ] && echo 11111 || true ;;
  *"VLLM::Worker_TP"*) [ "${STUB_WK:-0}" = "1" ] && printf '22222\n22223\n' || true ;;
  *"VLLM::EngineCor"*) [ "${STUB_EC:-0}" = "1" ] && echo 33333 || true ;;
esac
exit 0
STUB

# ps：① 计数僵尸 ② 单进程字段 ③ 全表
cat > "$B/ps" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  "-eo stat,comm") printf 'Z VLLM::Worker_TP\nZ python3\nS bash\n' ;;
  "-o ppid= -p "*|"-o user= -p "*|"-o etimes= -p "*|"-o pcpu= -p "*|"-o stat= -p "*)
      for a in "$@"; do case "$a" in -p) shift;; esac; done
      echo "1" ;;
  "-eo pid,ppid,stat,wchan:20,etimes,comm")
      printf '22222 1 D migrate_pages 300 VLLM::Worker_TP\n' ;;
  *) : ;;
esac
exit 0
STUB

cat > "$B/pkill" <<'STUB'
#!/usr/bin/env bash
echo "pkill $*" >> "${STUB_LOG:-/dev/null}"
exit "${STUB_PKILL_RC:-0}"
STUB

cat > "$B/sudo" <<'STUB'
#!/usr/bin/env bash
echo "sudo $*" >> "${STUB_LOG:-/dev/null}"
[ "${1:-}" = "-n" ] && shift
[ "${STUB_SUDO_OK:-1}" = "1" ] || { echo "sudo: a password is required" >&2; exit 1; }
exec "$@"
STUB

# 要让 pgrep 的结果**随场景变化**（杀了 migratepages 之后就没了），用一个"状态文件"
cat > "$B/pgrep" <<'STUB'
#!/usr/bin/env bash
echo "$*" >> "${STUB_PG_LOG:-/dev/null}"
_st=$(cat "${STUB_STATE:-/dev/null}" 2>/dev/null || echo "")
case "$*" in
  "-x migratepages")
      # 状态含 killed ⇒ 已杀掉
      case "$_st" in *killed*) : ;; *) [ "${STUB_MP:-0}" = "1" ] && echo 11111 ;; esac ;;
  *"VLLM::Worker_TP"*) [ "${STUB_WK:-0}" = "1" ] && printf '22222\n22223\n' ;;
  *"VLLM::EngineCor"*) [ "${STUB_EC:-0}" = "1" ] && echo 33333 ;;
esac
exit 0
STUB

# sudo pkill 之后标记状态（模拟"migratepages 被杀掉"这个**关键事实**）
cat > "$B/sudo" <<'STUB'
#!/usr/bin/env bash
echo "sudo $*" >> "${STUB_LOG:-/dev/null}"
[ "${1:-}" = "-n" ] && shift
[ "${STUB_SUDO_OK:-1}" = "1" ] || { echo "sudo: a password is required" >&2; exit 1; }
case "$*" in
  "pkill -9 -x migratepages") echo killed >> "${STUB_STATE:-/dev/null}" ;;
esac
exec "$@"
STUB

chmod +x "$B"/*

run_stop() {   # <名> <env...>
    local name="$1"; shift
    OUTF="$T/out_$name.txt"
    : > "$T/state_$name"
    ( env PATH="$B:$PATH" STUB_LOG="$T/log_$name.txt" STUB_PG_LOG="$T/pg_$name.txt" \
        STUB_STATE="$T/state_$name" STUB_MARK="$T/mark_$name" \
        CTR=dsv41-a3 "$@" bash "$SCRIPT" ) >"$OUTF" 2>&1
    rc=$?
    echo "--- [$name] rc=$rc"
    return $rc
}
check() {   # <名> <期望rc> <实际rc> [必须出现] [禁止出现]
    local name="$1" want="$2" got="$3" need="${4:-}" deny="${5:-}"
    _dbg() { printf '        └── 尾部：\n'; tail -8 "$OUTF" 2>/dev/null | sed 's/^/            /'; }
    if [ "$want" != "$got" ]; then bad "$name（rc=$got 期望 $want）"; _dbg; return; fi
    if [ -n "$need" ] && ! grep -qF -- "$need" "$OUTF"; then bad "$name（少了判据：$need）"; _dbg; return; fi
    if [ -n "$deny" ] && grep -qF -- "$deny" "$OUTF"; then bad "$name（出现禁止项：$deny）"; _dbg; return; fi
    ok "$name"
}
logged() {  # <名> <日志文件> <必须出现>
    if [ -f "$2" ] && grep -qF -- "$3" "$2"; then ok "$1"
    else bad "$1（$2 里没有：$3）"; fi
}

# ================================================================ ① 容器不存在
say "① 容器不存在 ⇒ rc=0 且直接说清（不误报成功停机）"
run_stop gone STUB_CTR_EXISTS=0; rc=$?
check "① 容器不存在" 0 "$rc" "已经清掉了"

# ================================================================ ①b 反引号不许被当命令执行
say "①b 双引号里的反引号会被 shell 当**命令替换**执行 ⇒ 文案必须用单引号（真机踩到过）"
run_stop bq STUB_STOP_HANGS=1 STUB_MP=1; rc=$?
if grep -qF 'CPU_BIND=1 是**进程内**用 set_mempolicy' "$OUTF"; then
    ok "①b 那句 CPU_BIND=1 的文案完整保留（没被命令替换吃掉）"
else
    bad "①b 文案被 shell 吃掉了（说明用了双引号+反引号）"
fi

# ================================================================ ② 正常能停
say "② 容器能正常停 ⇒ rc=0，且**不含**任何 kill（不许乱动手）"
run_stop normal STUB_STOP_HANGS=0; rc=$?
check "②a 正常停" 0 "$rc" "容器已停"
if grep -qE "pkill|rm -f" "$T/log_normal.txt" 2>/dev/null; then bad "②b 正常停时**动了手**（不该）"
else ok "②b 正常停时没有 pkill/rm"; fi

# ================================================================ ③ 卡住（用户现场）+ 只诊断
say "③ 卡住现场 + 默认（只诊断）⇒ 必须识别出那句原文，并把该敲的命令打出来，但**不动手**"
run_stop diag STUB_STOP_HANGS=1 STUB_MP=1 STUB_WK=1; rc=$?
check "③a 识别卡住原文" 3 "$rc" "did not receive an exit event"
check "③b 给出 sudo pkill migratepages" 3 "$rc" "sudo pkill -9 -x migratepages"
check "③c 说明为什么必须 sudo" 3 "$rc" "Operation not permitted"
if grep -q "sudo pkill" "$T/log_diag.txt" 2>/dev/null; then bad "③d 只诊断模式却真的 pkill 了"
else ok "③d 只诊断模式没有动手"; fi

# ================================================================ ④ 卡住 + YES=1（★ 核心：实测有效的那条阶梯）
say "④ 卡住 + YES=1 ⇒ 先 sudo pkill -9 -x migratepages，再 stop（★ 这次必须成功，rc=0）"
run_stop fix YES=1 STUB_MP=1 STUB_WK=1; rc=$?
check "④a 阶梯成功" 0 "$rc" "容器已停"
check "④b 点明根因" 0 "$rc" "CPU_BIND=1"
logged "④c 真的执行了 sudo pkill -9 -x migratepages" "$T/log_fix.txt" "pkill -9 -x migratepages"
logged "④d 而且是在 stop 之前" "$T/log_fix.txt" "sudo -n pkill"

# ================================================================ ⑤ sudo 不可用
say "⑤ sudo 要密码 ⇒ 必须**明确告知**手工执行，不许假装成功"
run_stop nosudo YES=1 STUB_STOP_HANGS=1 STUB_MP=1 STUB_SUDO_OK=0; rc=$?
check "⑤a sudo 不可用仍给出手工命令" 3 "$rc" "sudo pkill -9 -x migratepages"

# ================================================================ ⑥ 没有 migratepages（D 状态）
say '⑥ 没有 migratepages ⇒ 走第 2/3 步，并给出 D 状态判定与"重启"的建议'
run_stop dstate YES=1 STUB_STOP_HANGS=1 STUB_MP=0 STUB_WK=1 STUB_RM_OK=0; rc=$?
check "⑥a 走到 D 状态判定" 3 "$rc" "内核态卡死（D 状态）"
check "⑥b 给出 /proc/<pid>/stack" 3 "$rc" "/proc/<pid>/stack"
check "⑥c 给重启建议" 3 "$rc" "直接重启该节点"
check "⑥d 警告别 restart docker" 3 "$rc" "不要**为了清残留去"

echo
echo "=============== 通过 $V 条 / 失败 $F 条 ==============="
if [ "$F" = "0" ] && [ "$V" -ge 12 ]; then echo "✅ 自测全过"; exit 0; fi
echo "❌ 不合格"; exit 9
