#!/usr/bin/env bash
# =============================================================================
# A3 一键起服（默认 TP8×DP1；DP2×TP8 时须指定 16 张卡）
#
# 与 A2 共用同一个引擎（serve_a2.sh），本文件只覆盖**平台相关默认值**
# （IMAGE / NAME / PORT / PGO）并做 A3 特有的**选卡校验**。
# 其余（优化开关、门控 env、模型挂载、静态内核缓存、admission gate …）完全一致。
#
# 用法：
#   # ① 先看哪些卡空着（只读，打印每张卡的占用与进程属主）
#   bash tools/list_chips.sh
#
#   # ② 自己指定要用的 8 张卡（DEVS 必填，不给就报错退出）
#   DEVS="8 9 10 11 12 13 14 15" \
#     MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
#     bash scripts/serve_a3.sh
#
#   # 先干跑核对（不碰 docker）：加 DRY_RUN=1
#
# ★ DEVS 是**用户输入**，脚本不替你选卡：一台机器上哪 8 张能用取决于当前谁在跑什么，
#   脚本无法替你判断。CPU/NUMA 绑定默认按选中卡的 PCI numa_node **自动推导**
#   （CPUSET=auto MEMS=auto），也可显式覆盖。
# ★ 默认拒绝**已被占用的卡**（防误伤他人任务）：若确实要用自己的残留进程占着的卡，
#   显式加 ALLOW_BUSY=1。
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# [NO_PROXY] 企业代理会**拦截 127.0.0.1**，把"服务已就绪"判成"起服挂死"。
#
# 实测（issue #2 报告者，2026-09-21）：他们的 Squid 代理对 `127.0.0.1` 的请求
# 直接返回 **503 错误页**。表现是模型已经启动完成、直连 `/v1/models` 也正常，
# 但走代理的 `curl http://127.0.0.1:<port>/health` **永远拿 503**
# ⇒ 所有就绪轮询/健康检查超时 ⇒ 看起来像"起服挂死"，把后面的判断全带偏。
#
# 一眼识别：返回的是 **HTML** 而不是 JSON 就是被劫持了：
#     curl -s http://127.0.0.1:8100/health | head -3
#     curl -s --noproxy '*' http://127.0.0.1:8100/health | head -3   # 立即 200
#
# 只在**用户没设过**时补默认值 ⇒ **不覆盖**已有的 no_proxy 配置。
# 要显式关掉：`KEEP_PROXY_FOR_LOCALHOST=1`。
# ---------------------------------------------------------------------------
if [ "${KEEP_PROXY_FOR_LOCALHOST:-0}" != "1" ]; then
  _v41_np_default='127.0.0.1,localhost,::1'
  if [ -z "${no_proxy:-}" ]; then
    export no_proxy="$_v41_np_default"
  elif ! printf '%s' "$no_proxy" | grep -q '127\.0\.0\.1'; then
    export no_proxy="${no_proxy},${_v41_np_default}"
  fi
  export NO_PROXY="${no_proxy}"
fi
unset _v41_np_default

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=${DRY_RUN:-0}

export IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
export NAME=${NAME:-dsv41-a3}
export PORT=${PORT:-8020}
# [SERVED_NAME] 与 PORT 同口径：env 优先，默认 `deepseek-v41`。
# 它同时决定 API 请求 body 里必须填的 `"model"` 字段；测试脚本请用同一个名字。
export SERVED_NAME=${SERVED_NAME:-deepseek-v41}
export TOOL_CALLING=${TOOL_CALLING:-1}
export MAX_SEQS=${MAX_SEQS:-32}
export DP=${DP:-1}
# [绑核] 外部**不做** CPU/NUMA 绑定：容器不设 --cpuset-cpus/--cpuset-mems，
#   由 vllm-ascend 内部的 cpu_binding 按 NPU 拓扑给每个 rank 自己绑
#   （additional-config 的 enable_cpu_binding=true，由 CPU_BIND=1 控制）。
#   起服日志证据：[cpu_binding.py] mode=topo_affinity rank=N / [migrate] NPU:N -> NUMA [M]
#   需要复现历史口径或做 AB 时才显式给 CPUSET=<核列表> MEMS=<节点列表>。
export CPUSET=${CPUSET:--1}
export MEMS=${MEMS:--1}
export CPU_BIND=${CPU_BIND:-1}
# [PGO] A2 的 PGO 产物是针对 **A2 镜像** 的 libpython 编译的：
#   A2 镜像 md5(libpython3.12.so.1.0) = f1ebbee1405d0e31136aa4480b57b3dc
#   A3 镜像 md5(libpython3.12.so.1.0) = eaea156ea8ddf85b0b2d71f77872991e   ← 不同
#   ⇒ A3 默认不挂 PGO（要试可显式 PYTHON_PGO=1，但属于未验证改动）。
export PYTHON_PGO=${PYTHON_PGO:-0}

# [PATCH_MODE] A3 用的是**官方镜像**（quay.nju.edu.cn/ascend/vllm-ascend:
# deepseek-v4.1-flash-a3），里面**没有**本包的补丁 —— 所以 A3 必须走 mount
# 模式把 patches/files/* 挂进去。否则跑的是未优化版本，而且**不会有任何报错**，
# 只是所有性能补丁（含 Engram device-index）都静默没生效。
# 与 A2 相反：A2 用 build_image.sh 烘焙出 dsv41-a2:v8，默认 baked 是对的。
# 想显式覆盖就设 PATCH_MODE=baked（例如你自己烘焙了一个 A3 镜像）。
export PATCH_MODE=${PATCH_MODE:-mount}

# [DROPCACHE] 起服前清 page cache —— ★ **A3 默认关闭**（与 A2 相反）。
#   为什么两者不同（这条是 2026-09-23 补的，之前 A3 会**静默继承 A2 的 1**）：
#     * 机制：`serve_a2.sh:168` 的默认是 `1`，而本脚本此前**没有**给 `DROPCACHE` 任何默认值
#       ⇒ A3 上实际生效的是 **1 = 起服前 `echo 1 > /proc/sys/vm/drop_caches`**。
#     * 危害：那个写操作是**整机**的 —— A3 是**共用机**（多租户），清掉的会连带包括
#       **别人的** page cache，别人的任务下一次读文件全部回盘变慢。
#       A2 是独占机，所以 A2 默认 1 是划算的（省下 564 GiB，见 serve_a2.sh 里的实测）。
#     * 判据：A2 独占 ⇒ 1；A3 共用 ⇒ 0。**显式给的一律优先**（`DROPCACHE=1` 仍可用，
#       但请先确认此刻机器上没有别人的任务）。
export DROPCACHE=${DROPCACHE:-0}

# ---------- 1) DEVS 必填 ----------
if [ -z "${DEVS:-}" ]; then
  cat >&2 <<'MSG'

[serve_a3][FAIL] 必须显式指定 DEVS=<要用的 chip 列表>（脚本不替你选卡）。

  这台机器上哪 8 张卡能用，取决于当前谁在跑什么 —— 先看一眼：

      bash tools/list_chips.sh

  然后按空闲情况指定，例如：

      DEVS="8 9 10 11 12 13 14 15" \
        MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
        bash scripts/serve_a3.sh

  说明：
    * DEVS 的个数应与 TP×DP（默认 8×1）一致；
    * CPU/NUMA 绑定默认按选中卡自动推导（CPUSET=auto MEMS=auto），也可显式覆盖；
    * 默认拒绝已被占用的卡；确实要用自己的残留进程占着的卡时加 ALLOW_BUSY=1。

MSG
  exit 2
fi

# ---------- 2) DEVS 格式与数量 ----------
_n=0
for _c in $DEVS; do
  case "$_c" in
    ''|*[!0-9]*) echo "[serve_a3][FAIL] DEVS 里有非法项：'$_c'（只能是数字，空格分隔）" >&2; exit 2 ;;
  esac
  _n=$((_n + 1))
done
export DEVS
echo "[serve_a3] DEVS='$DEVS'（$_n 张）CPUSET=$CPUSET MEMS=$MEMS"
_tp=${TP:-8}
case "$_tp:$_n:$DP" in
  *[!0-9:]*|0:*|*:0|*:0:*) echo "[serve_a3][FAIL] TP/DP/DEVS 数量必须是正整数" >&2; exit 2 ;;
esac
_want=$((_tp * DP))
if [ "$_n" != "$_want" ]; then
  if [ "${I_KNOW:-0}" != "1" ]; then
    echo "[serve_a3][FAIL] DEVS 有 $_n 张，而 TP=$_tp × DP=$DP 需要 $_want 张 ⇒ 数量不匹配。" >&2
    echo "  请指定 $_want 张卡（或 I_KNOW=1 自行承担不匹配风险）。" >&2
    exit 2
  fi
  echo "[serve_a3] NOTE: DEVS 有 $_n 张 ≠ TP=$_tp × DP=$DP 所需 $_want，已按 I_KNOW=1 继续。"
fi

# ---------- 3) 占用检测（只读；默认拒绝） ----------
# 解析 npu-smi info 的进程表：| <npu> <chip> | <pid> | <name> | ... |
#   device 号统一按 npu*2+chip 折算（信息表两列的含义随机型而异，进程表两列在这两台机器上一致）。
# 解析不到任何行时**不阻塞**（可能只是输出格式变了），只提示。
if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_BUSY:-0}" != "1" ]; then
  if command -v npu-smi >/dev/null 2>&1; then
    _busy=$(
      npu-smi info 2>/dev/null | awk -F'|' '
        NF>=6 && $3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
          split($2, a, /[[:space:]]+/); print (a[2]+0)*2 + (a[3]+0)
        }' | sort -n -u | tr '\n' ' '
    )
    _hit=""
    for _c in $DEVS; do
      case " $_busy " in *" $_c "*) _hit="$_hit $_c" ;; esac
    done
    if [ -n "$_hit" ]; then
      {
        echo
        echo "[serve_a3][FAIL] 选中的卡里有正在被占用的：$_hit"
        echo "  （npu-smi 报出的占用卡：${_busy:-无}）"
        echo
        echo "  逐卡详情："
        npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | head -30
        echo
        echo "  处置："
        echo "    * 换成空闲的卡：bash tools/list_chips.sh 看全貌，再改 DEVS=..."
        echo "    * 若那些进程确实是你自己的残留，先停掉它（不要 blind kill 别人的）"
        echo "    * 确实要带占用起服务：加 ALLOW_BUSY=1（危险，会与他人任务抢卡）"
      } >&2
      exit 3
    fi
    echo "[serve_a3] 占用检查：DEVS 全部空闲（npu-smi 报出占用卡：${_busy:-无}）"
  else
    echo "[serve_a3] WARN: 找不到 npu-smi，跳过占用检查" >&2
  fi
else
  [ "$DRY_RUN" = "1" ] && echo "[serve_a3] dry-run：跳过占用检查"
  [ "${ALLOW_BUSY:-0}" = "1" ] && echo "[serve_a3] ALLOW_BUSY=1：跳过占用检查"
fi

# ---------- 4) 只校验不起服 ----------
# 想"先确认选卡正确、再决定何时起服务"时用：CHECK_ONLY=1
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "[serve_a3] CHECK_ONLY=1 ⇒ 选卡校验通过，不启动服务。"
  exit 0
fi

exec bash "$HERE/serve_a2.sh" "$@"
