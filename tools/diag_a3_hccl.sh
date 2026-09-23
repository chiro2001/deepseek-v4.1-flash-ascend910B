#!/usr/bin/env bash
# =============================================================================
# diag_a3_hccl.sh —— 新机上「HCCL 建链失败」的诊断助手
#
# 什么时候用它：起服在 **模型初始化阶段**就崩，栈底是
#     .../quantization/methods/w4a8/w4a8.py: self.moe_all_to_all_group_name =
#         backend.get_hccl_comm_name(local_rank)
#     RuntimeError: ... hcclCommInitRootInfoConfig(...), error code is 1
#     ERR02200 DIST call hccl api failed.
#     Communication_Error_Ranktable_Detect(EI0015): ... No rank in the communicator can
#     connect to the root node within the timeout period. List of unconnected ranks: "[3,]"
#
# ★ 这条错误的**判读要点**：请注意那句 `unconnected ranks: "[3,]"` ——
#   它说的是**某一个 rank**掉队了（不是全部）。⇒ 两类根因：
#     (a) **那张卡当时不可用**：别人的进程占着、或进程被 OOM 杀了
#         （一个 rank 死掉 ⇒ 其余 rank 永远等不到它 ⇒ 就是这个签名）
#     (b) **HCCL 选错网卡**：多网卡机器上自动挑到一张走不通的（错误信息里"第 3 条建议"就是这个）
#   本脚本把这两类的证据**一次性收齐**，省得来回问。
#
# 用法（**在宿主机上**，只读；不碰 docker 除非你显式开 RUN_HCCL_TEST=1）：
#   DEVS="8 9 10 11 12 13 14 15" bash tools/diag_a3_hccl.sh
#   DEVS="..." RUN_HCCL_TEST=1 bash tools/diag_a3_hccl.sh   # ★ 额外真跑一次 8 卡 HCCL（会短暂占卡）
#   DEVS="..." SERVE_LOG=/path/to/serve.log bash tools/diag_a3_hccl.sh
#
# 退出码：0 = 证据已收集（**不表示问题已修**）；2 = 前置不满足
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/.." && pwd)

DOCKER=${DOCKER:-docker}
NPU_SMI=${NPU_SMI:-npu-smi}
DEVS=${DEVS:-""}
IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
RUN_HCCL_TEST=${RUN_HCCL_TEST:-0}
SERVE_LOG=${SERVE_LOG:-}

say() { printf '\n\033[1m======== %s ========\033[0m\n' "$*"; }
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn(){ printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

if [ -z "$DEVS" ]; then
    echo "⛔ 必须给 DEVS=<你这次实际用的卡号>（从起服日志的 devs='...' 抄）" >&2
    echo "   例：DEVS=\"8 9 10 11 12 13 14 15\" bash tools/diag_a3_hccl.sh" >&2
    exit 2
fi

printf '\033[1m######## A3 HCCL 建链失败诊断（DEVS="%s"）########\033[0m\n' "$DEVS"
echo "  宿主   : $(hostname)"
echo "  镜像   : $IMAGE（仅 RUN_HCCL_TEST=1 时用）"

# ================================================================ ① 卡占用
say "① 你选的这些卡现在有没有**别人的**进程（这是"某个 rank 掉队"最直接的来源）"
if ! command -v "$NPU_SMI" >/dev/null 2>&1; then
    warn "没有 $NPU_SMI ⇒ 跳过（请在有 npu-smi 的宿主机上跑）"
else
    _raw=$("$NPU_SMI" info 2>/dev/null)
    # HBM 用量（枚举序号 = device 号）
    _hbm=$(printf '%s\n' "$_raw" | awk -F'|' '
      NF>=5 && $3 ~ /[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]/ {
        s=$4; gsub(/[^0-9\/]/,"",s); n=split(s,p,"/");
        if (n>=2) printf "%d\t%d\n", seq, p[n-1]+0; seq++ }')
    # 进程表：(npu,chip) -> device = npu*2+chip
    _proc=$(printf '%s\n' "$_raw" | awk -F'|' '
      NF>=6 && $3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
        split($2,a,/[[:space:]]+/); d=(a[2]+0)*2+(a[3]+0);
        pid=$3; gsub(/[^0-9]/,"",pid); name=$4; gsub(/^[[:space:]]+| *$/,"",name);
        printf "%d\t%s\t%s\n", d, pid, name }')
    _busy=""
    for _c in $DEVS; do
        _p=$(printf '%s\n' "$_proc" | awk -F'\t' -v d="$_c" '$1==d{print $2"/"$3}' | paste -sd, -)
        _u=$(printf '%s\n' "$_hbm" | awk -F'\t' -v d="$_c" '$1==d{print $2}')
        if [ -n "$_p" ]; then
            printf '  device %-3s 占用   进程=%s\n' "$_c" "$_p"
            _busy="$_busy $_c"
        elif [ "${_u:-0}" -ge 4096 ] 2>/dev/null; then
            printf '  device %-3s 无进程但 HBM=%sMB（残留/驱动预留 —— ★ 可疑）\n' "$_c" "$_u"
            _busy="$_busy $_c"
        else
            printf '  device %-3s 空闲\n' "$_c"
        fi
    done
    if [ -n "$_busy" ]; then
        bad "有可疑的卡：$_busy"
        echo "        ⇒ 若这些进程**不是你的**：换卡（DEVS=...）；若是你的残留：先停掉再起服。"
        echo '        ⇒ 这正是 unconnected ranks: [N,] 的典型来源（一个 rank 进不去，其余全等它）。'
    else
        ok "选中的卡当前都干净"
    fi
fi

# ================================================================ ② 网卡（HCCL 选错网卡）
say "② 宿主网卡（HCCL 自动挑；**多张真实网卡**时可能挑错 ⇒ 需要 HCCL_SOCKET_IFNAME）"
if command -v ip >/dev/null 2>&1; then
    _all=$(ip -o -4 addr show 2>/dev/null | awk '{print $2, $4}')
    printf '%s\n' "$_all" | sed 's/^/        /'
    _real=$(printf '%s\n' "$_all" | awk '{print $1}' \
            | grep -vE '^(lo|docker[0-9]*|br-|veth|virbr)' | wc -l)
    if [ "${_real:-0}" -gt 1 ]; then
        warn "真实网卡有 **$_real 张** ⇒ HCCL 自动挑选有风险（对照：工作机 a3-21 只有 1 张 enp196s0f0/192.168.45.21）"
        echo "        ⇒ 起服时显式指定同一张网卡（换成你机器上那张真实网卡名）："
        echo "             HCCL_SOCKET_IFNAME=<网卡名> ... bash tools/deploy_a3.sh"
        echo "          若部署脚本不认这个变量，就先 export 再跑（它是靠环境继承进容器的）："
        echo "             export HCCL_SOCKET_IFNAME=<网卡名>"
    else
        ok "只有 ${_real:-0} 张真实网卡（与工作机同形）⇒ 网卡这条嫌疑低"
    fi
else
    warn "没有 ip 命令 ⇒ 跳过"
fi

# ================================================================ ③ A3 特性 + ④ OOM
say "③ A3 特性位（应该是 1）"
if [ -r /proc/svm/dev0/feature/host_mem_pool ]; then
    _v=$(cat /proc/svm/dev0/feature/host_mem_pool 2>/dev/null)
    [ "$_v" = "1" ] && ok "host_mem_pool=$_v（A3 口径）" || warn "host_mem_pool=$_v（工作机是 1；不是 1 要查机型/驱动）"
else
    warn "读不到 /proc/svm/dev0/feature/host_mem_pool（可能不是 A3 机型，或权限不足）"
fi

say "④ 内存与 OOM（**一个 rank 被 OOM 杀 ⇒ 其余 rank 全报它 unconnected**）"
free -g 2>/dev/null | head -2 | sed 's/^/        /'
if command -v dmesg >/dev/null 2>&1; then
    _oom=$( { dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null; } \
            | grep -iE "out of memory|oom-kill|killed process" | tail -5 )
    if [ -n "$_oom" ]; then
        bad "dmesg 里有 OOM 记录："; printf '%s\n' "$_oom" | sed 's/^/        /'
        echo "        ⇒ 起服峰值需要 ≈1 TB 可用（8 rank × 权重页 + Engram 表）；"
        echo "          内存不足时先腾内存，或降低并发/上下文（MAX_SEQS/MAX_LEN）。"
    else
        ok "dmesg 里没看到 OOM 记录（可能需 root 才能读到全部）"
    fi
fi

# ================================================================ ⑤ 起服日志里的 rank 证据
if [ -n "$SERVE_LOG" ] && [ -f "$SERVE_LOG" ]; then
    say "⑤ 起服日志：**掉队的那个 rank 自己的行**（它比其他 rank 少 ⇒ 它是被卡住/被杀的那个）"
    for _r in 0 1 2 3 4 5 6 7; do
        _n=$(grep -ac "Worker_TP${_r}_EP${_r}" "$SERVE_LOG" 2>/dev/null || true)
        printf '        Worker_TP%s_EP%s 行数=%s\n' "$_r" "$_r" "${_n:-0}"
    done
    echo "        （行数明显少的那个 = 掉队的 rank；再看它最后一行停在哪一步）"
    echo '        ★ 关键判读：停在 device 初始化之前=卡被占；停在权重加载=内存；停在 HCCL=网卡/连接'
    grep -a "unconnected ranks" "$SERVE_LOG" 2>/dev/null | tail -2 | sed 's/^/        /'
fi

# ================================================================ ⑥ 可选：真跑 8 卡 HCCL
if [ "$RUN_HCCL_TEST" = "1" ]; then
    say "⑥ ★ 真跑一次 8 卡 HCCL（直接用 torch_npu，绕开 vLLM）—— 决策性证据"
    _devargs=""; _artv=""
    for _c in $DEVS; do _devargs="$_devargs --device /dev/davinci$_c"; _artv="$_artv,$_c"; done
    _artv=${_artv#,}
    T=$(mktemp -d)
    cat > "$T/hccl_probe.py" <<'PY'
import os, sys, torch, torch_npu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("hccl", rank=rank, world_size=world)
t = torch.ones(1024, dtype=torch.float16).npu() * (rank + 1)
dist.all_reduce(t)
exp = world * (world + 1) / 2
got = float(t[0].item())
ok = abs(got - exp) < 1e-3
print(f"[hccl_probe] rank={rank} all_reduce={got} expect={exp} {'PASS' if ok else 'FAIL'}", flush=True)
dist.destroy_process_group()
sys.exit(0 if ok else 1)
PY
    echo "  跑：torchrun --nproc_per_node=$(printf '%s\n' $DEVS | wc -l)（容器 $IMAGE）"
    if "$DOCKER" run --rm --privileged --network host --ipc host \
         $_devargs -e ASCEND_RT_VISIBLE_DEVICES="$_artv" \
         -v "$T:/probe:ro" "$IMAGE" \
         bash -lc "cd /probe && torchrun --standalone --nproc_per_node=$(printf '%s\n' $DEVS | wc -l) hccl_probe.py" 2>&1 | tail -20 | sed 's/^/        /'
    then
        ok '8 卡 HCCL all-reduce 通过 ⇒ 集合通信本身没问题（回到"卡被占/内存"两条）'
    else
        bad "8 卡 HCCL all-reduce **失败** ⇒ 问题在环境层（网卡 / 某张卡 / 驱动），不在 vLLM 配置"
    fi
    rm -rf "$T"
else
    say "⑥ 真跑 HCCL（默认跳过）"
    echo "  要跑就加 RUN_HCCL_TEST=1（会短暂占用你选的这 8 张卡）："
    echo "      DEVS=\"$DEVS\" RUN_HCCL_TEST=1 bash tools/diag_a3_hccl.sh"
fi

say "收尾：把上面整段贴回来即可判读"
echo "  ★ 判读速查："
echo '    · ① 有别人的进程/无进程但 HBM 高   ⇒ 换卡或停自己的残留（最常见）'
echo "    · ② 真实网卡 >1                    ⇒ 显式 HCCL_SOCKET_IFNAME=<网卡名>"
echo "    · ④ 有 OOM                          ⇒ 内存不足（起服峰值 ≈1 TB 可用）"
echo '    · ⑤ 某 rank 行数明显少              ⇒ 它就是掉队的那个，看它停在哪一步'
echo "    · ⑥ HCCL 自体失败                   ⇒ 环境/硬件层，不是部署脚本的问题"
