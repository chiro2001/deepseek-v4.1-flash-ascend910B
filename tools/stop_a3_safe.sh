#!/usr/bin/env bash
# =============================================================================
# stop_a3_safe.sh —— 把「vLLM 卡死 ⇒ 容器停不下来」这件事**分步解开**（带诊断，不是硬杀）
#
# ── 症状（用户在新 A3 上实测的原文）─────────────────────────────────────────────
#     $ docker stop dsv41-a3
#     Error response from daemon: cannot stop container: dsv41-a3:
#         tried to kill container, but did not receive an exit event
#   卡在那里不动；`--time 10` 之类的等待也无用。
#
# ── 根因（2026-09-23 在 a3-21 上完整复现并定位）──────────────────────────────
#   A3 起服默认 **`CPU_BIND=1`**（`scripts/serve_a3.sh`）⇒ vllm-ascend 内部 `cpu_binding`
#   会把**每个 rank 的常驻内存（实测 ≈180–196 GB/worker）迁到它那张卡所在的 NUMA 节点**，
#   手法是给每个 worker 起一个 **`migratepages`** 子进程。
#   ★ 而 A3 是**共用机**：本机实测 8 个 NUMA 节点里 **6 个只剩 0.7–9 GB 空闲** ⇒
#     `migratepages` **无处可迁**，于是在 **100% CPU 上无限自旋**（实测连续 >2.5 min 不停），
#     服务永远不就绪（日志里的伴生行是
#       `shm_broadcast.py:802 No available shared memory broadcast block found in 60 seconds`）。
#   ⇒ 停容器时 SIGTERM/SIGKILL 都"发到了但拿不到 exit event" ⇒ docker 报上面那句。
#
# ── 实测有效的解法（a3-21 上验过：`docker stop` 从"失败 16 s"变成"**1 s 成功**"）──
#     1) `sudo pkill -9 -x migratepages`   ← ★ 关键一步（它们是 **root** 拥有的，
#                                             普通用户 kill 会 `Operation not permitted`）
#     2) 再 `docker stop -t 2 <容器>`（或 `docker rm -f`）
#
# 本脚本就是把上面这套**按阶梯**做完，并在每步打印证据；默认**只诊断不动手**。
#
# 用法：
#   CTR=dsv41-a3 bash tools/stop_a3_safe.sh              # 只诊断 + 打印该敲的命令
#   CTR=dsv41-a3 YES=1 bash tools/stop_a3_safe.sh        # 允许它动手（含 sudo pkill / docker rm -f）
#   CTR=dsv41-a3 SOFT_ONLY=1 bash tools/stop_a3_safe.sh  # 与默认同义（显式表达"只看不动"）
#
# 退出码：0 = 容器已停/已清理；3 = 需要 root 或需重启节点（脚本会说明）；2 = 用法错
# =============================================================================
set -uo pipefail

CTR=${CTR:-dsv41-a3}
YES=${YES:-0}
SOFT_ONLY=${SOFT_ONLY:-0}
[ "$SOFT_ONLY" = "1" ] && YES=0
GRACE=${GRACE:-5}                 # docker stop 的宽限秒数
STEP_TIMEOUT=${STEP_TIMEOUT:-30}  # 每一步 docker 命令的墙钟上限
PS_BIN=${PS_BIN:-ps}

say() { printf '\n\033[1m======== %s ========\033[0m\n' "$*"; }
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn(){ printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
run() { if [ "$YES" = "1" ]; then "$@"; else printf '  \033[36m[只诊断]\033[0m 该敲：%s\n' "$*"; return 0; fi; }

printf '\033[1m######## 安全停机（CTR=%s）########\033[0m\n' "$CTR"
echo "  模式：$([ "$YES" = "1" ] && echo '★ 允许动手（YES=1）' || echo '只诊断（要它动手加 YES=1）')"

# ================================================================ ① 现状
say "① 现状（容器 + 宿主侧的卡住痕迹）"
if ! docker inspect "$CTR" >/dev/null 2>&1; then
    ok "容器 $CTR 不存在（已经清掉了）"
    exit 0
fi
_st=$(docker inspect -f '{{.State.Status}}' "$CTR" 2>/dev/null)
echo "  容器状态：$_st"
echo "  cpuset：cpuset-cpus=[$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$CTR" 2>/dev/null)] cpuset-mems=[$(docker inspect -f '{{.HostConfig.CpusetMems}}' "$CTR" 2>/dev/null)]"
echo "  （cpuset-mems 为空 ≠ 没绑节点：`CPU_BIND=1` 是**进程内**用 set_mempolicy 绑的，docker 里看不到）"

_mp=$(pgrep -x migratepages 2>/dev/null | tr '\n' ' ')
_wk=$(pgrep -f 'VLLM::Worker_TP' 2>/dev/null | wc -l)
_ec=$(pgrep -f 'VLLM::EngineCor' 2>/dev/null | wc -l)
_zb=$(ps -eo stat,comm 2>/dev/null | grep -c '^Z' || true)
printf '  migratepages=%s  VLLM::Worker_TP=%s  VLLM::EngineCor=%s  僵尸(Z)总数=%s\n' \
       "$(printf '%s' "$_mp" | wc -w)" "$_wk" "$_ec" "${_zb:-0}"
if [ -n "${_mp// /}" ]; then
    echo "  ★ 正在迁移的进程（就是它们把容器钉住的）："
    for _p in $_mp; do
        _pp=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
        _u=$(ps -o user= -p "$_p" 2>/dev/null | tr -d ' ')
        _e=$(ps -o etimes= -p "$_p" 2>/dev/null | tr -d ' ')
        _pc=$(ps -o pcpu= -p "$_p" 2>/dev/null | tr -d ' ')
        printf '        pid=%-8s owner=%-8s parent=%-8s 已跑=%-6ss cpu=%s%%\n' \
               "$_p" "${_u:-?}" "${_pp:-?}" "${_e:-?}" "${_pc:-?}"
    done
    echo "        ★ 若 owner=root（容器以 root 跑）：**必须 sudo 才能杀**，普通 kill 会 Operation not permitted"
else
    echo "  （没有 migratepages 在跑 —— 那么卡住的可能是别的内核态调用，见第 ④ 步）"
fi

# ================================================================ ② 先试一次有界 stop
say "② 先试一次**有界**的 docker stop（宽限 ${GRACE}s，墙钟上限 ${STEP_TIMEOUT}s）"
_t0=$(date +%s)
_out=$(timeout "$STEP_TIMEOUT" docker stop -t "$GRACE" "$CTR" 2>&1); _rc=$?
_dt=$(( $(date +%s) - _t0 ))
echo "  rc=$_rc  耗时=${_dt}s"
[ -n "$_out" ] && printf '%s\n' "$_out" | sed 's/^/        /'
if [ "$_rc" = "0" ]; then
    ok "容器已停（${_dt}s）"
    say "收尾"
    echo "  ★ 下次避免：起服时用 **CPU_BIND=0**（不做内部绑核/迁移），见 docs/A3-DEPLOY.md §5.2"
    exit 0
fi
case "$_out" in
  *"did not receive an exit event"*|*"cannot stop container"*)
      bad 'docker 拿不到 exit event —— 与 vLLM 卡死完全同型（本脚本 §③ 就是解这个）' ;;
  *)
      warn "stop 没成功（rc=$_rc）；继续走阶梯" ;;
esac

# ================================================================ ③ 阶梯：先解 migratepages
say "③ 阶梯第 1 步：杀掉 NUMA 迁移进程（★ 实测这一步就能让 stop 从失败变成 1 s 成功）"
if [ -n "${_mp// /}" ]; then
    echo "  目标：sudo pkill -9 -x migratepages"
    if [ "$YES" != "1" ]; then
        printf '  \033[36m[只诊断]\033[0m 该敲：sudo pkill -9 -x migratepages\n'
        printf '  \033[36m[只诊断]\033[0m 然后再：docker stop -t 2 %s\n' "$CTR"
    else
        if timeout "$STEP_TIMEOUT" sudo -n pkill -9 -x migratepages 2>&1; then
            ok "已请求杀掉 migratepages"
        else
            bad "sudo pkill 失败（没有免密 sudo？或进程已不在）—— 请手工执行：sudo pkill -9 -x migratepages"
        fi
    fi
    sleep 2
    echo "  现在还剩 migratepages：$(pgrep -x migratepages 2>/dev/null | wc -l) 个"
    # ★ 重试 stop **是一次动作**（它会去杀容器）⇒ 只在 YES=1 时做。
    #   （第一版没加这个门：只诊断模式也会重试，等于"说好不动手却动了手" —— 自测 ③ 抓到。）
    if [ "$YES" = "1" ]; then
        _t0=$(date +%s)
        _out2=$(timeout "$STEP_TIMEOUT" docker stop -t "$GRACE" "$CTR" 2>&1); _rc2=$?
        _dt2=$(( $(date +%s) - _t0 ))
        echo "  重试 stop：rc=$_rc2 耗时=${_dt2}s"
        [ -n "$_out2" ] && printf '%s\n' "$_out2" | sed 's/^/        /'
        if [ "$_rc2" = "0" ]; then
            ok "容器已停（${_dt2}s）★ 根因 = NUMA 迁移进程把 worker 钉住"
            say "收尾"
            echo "  ★ 根因已确认：**CPU_BIND=1** 触发 migratepages，而目标 NUMA 节点没有空间。"
            echo "    下次起服请用：DEVS=\"...\" CPU_BIND=0 LAUNCH=1 bash tools/deploy_a3.sh"
            exit 0
        fi
    else
        echo "  （只诊断模式：不替你重试 stop —— 上面两条命令敲完，容器通常 1 s 内就停了）"
    fi
else
    echo "  没有 migratepages ⇒ 跳过这一步（直接进第 ④ 步）"
fi

# ================================================================ ④ 阶梯：直接杀 worker / engine
say "④ 阶梯第 2 步：直接杀 worker 与 engine（容器以 root 跑 ⇒ 需 sudo）"
echo "  sudo pkill -9 -f 'VLLM::Worker_TP'"
echo "  sudo pkill -9 -f 'VLLM::EngineCor'"
echo "  docker rm -f $CTR"
if [ "$YES" = "1" ]; then
    timeout "$STEP_TIMEOUT" sudo -n pkill -9 -f 'VLLM::Worker_TP' 2>&1 || true
    timeout "$STEP_TIMEOUT" sudo -n pkill -9 -f 'VLLM::EngineCor' 2>&1 || true
    sleep 2
    _out3=$(timeout "$STEP_TIMEOUT" docker rm -f "$CTR" 2>&1); _rc3=$?
    echo "  docker rm -f：rc=$_rc3"
    [ -n "$_out3" ] && printf '%s\n' "$_out3" | sed 's/^/        /'
    if [ "$_rc3" = "0" ] || ! docker inspect "$CTR" >/dev/null 2>&1; then
        ok "容器已清理"
        exit 0
    fi
fi

# ================================================================ ⑤ 还是不行 ⇒ D 状态判定
say "⑤ 阶梯第 3 步：判定是不是**内核态卡死（D 状态）**"
echo "  命令（逐条看）："
echo "      ps -eo pid,ppid,stat,wchan:24,etimes,comm | grep -E 'VLLM|vllm'"
echo "      sudo cat /proc/<pid>/stack        # 卡在哪个内核函数（npu 驱动/HCCL/迁移）"
echo "      sudo dmesg -T | tail -30          # 找 'task hung' / 'oom-kill' / npu 报错"
if command -v "$PS_BIN" >/dev/null 2>&1; then
    echo
    echo "  现场快照（只列非 0 状态的 VLLM 进程）："
    "$PS_BIN" -eo pid,ppid,stat,wchan:20,etimes,comm 2>/dev/null \
      | grep -E 'VLLM|vllm' | head -20 | sed 's/^/        /'
    echo '        ★ stat 里含 **D** = 不可中断睡眠：SIGKILL 对它无效（这就是"杀不动"的最终原因）'
fi

say "结论与建议"
cat <<'MSG'
  · 若上面出现 **D** 状态、或 `dmesg` 有 'task hung' / npu 相关报错
      ⇒ 那一批进程**只能等内核调用返回**，用户态没有任何办法杀掉它。
        在**新机器/可重启**的场合：**直接重启该节点**是最省事的正解。
  · 重启前把证据留下来（重启后现场就没了）：
        sudo dmesg -T > /tmp/dmesg_$(date +%s).txt
        sudo cat /proc/<D状态pid>/stack > /tmp/stack_<pid>.txt 2>&1
  · 重启后**务必**用 CPU_BIND=0 起一次，确认问题与绑核/迁移相关：
        DEVS="8 9 10 11 12 13 14 15" CPU_BIND=0 LAUNCH=1 bash tools/deploy_a3.sh
  · ⚠️ 共享机上 **不要**为了清残留去 `systemctl restart docker` ——
        那会连带影响别人的容器。僵尸进程（Z）本身不占资源，留着即可。
MSG
exit 3
