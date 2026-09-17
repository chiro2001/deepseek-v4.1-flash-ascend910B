#!/usr/bin/env bash
# =============================================================================
# A2 起服脚本（serve_a21.sh 的 A2 适配版）
#
#   MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/serve_a2.sh
#
# 与 A3-node1 的差异（逐条）：
#   * DEVS 固定 0..7（A2 单机 8 卡；A3-node1 用 back8 = 8..15）
#   * 补丁来自**镜像内已烘焙**的版本（build_image.sh 产出）；PATCH_MODE=mount 时才用 -v 挂
#   * 默认关掉 A3-node1 的实验设施：HOTSPIKE=0、ROUTE_PROBE=0、探针不挂
#   * 默认关掉未验证开关：MOE_ZERO=0、DRAFT_GRAPH=0
#   * 缓存目录默认落在包目录 ./cache（A2 无外网，缓存持久化很重要）
#
# 常用变量（都有默认值，绝大多数不用改）：
#   MODEL      必填，模型目录
#   IMAGE      默认 dsv41-a2:v6
#   NAME       容器名，默认 dsv41-a2
#   PORT       默认 8100
#   GPU_UTIL   默认 0.94（容量不够就抬到 0.95/0.96）
#   MAX_SEQS   默认 4（**性能口径**；A2 生产是 32，见 MODE=prod）
#   PREFIX     默认 0 = 不启用 prefix caching（**历史性能口径**，与 v3 逐字节一致）
#              **A2 生产用 1**；PREFIX=1 时才会出现"decode 队列里插入新请求"这一生产形态
#   SP_TOKENS  默认 5（DSpark 原生 block size；CAPTURE_SIZES 按 S+1 自动推导）
#   PYTHON_PGO 默认 1（挂 optim/pgo 里编译好的 libpython；文件不存在则自动降级为 0）
#   LOAD_FORMAT 空 = 读真权重；dummy = 只按 shape 建模型（**只测时延，A 恒为 1.0**）
#   MOE_ZERO / DRAFT_GRAPH 默认 0（未验证，见 CHANGELOG 标红项）
#   MOE_NF     默认 0（负结果，不采纳；见 README「别踩坑」表）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"

MODEL=${MODEL:-}
IMAGE=${IMAGE:-dsv41-a2:v6}
NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8100}
TP=${TP:-8}
DEVS=${DEVS:-"0 1 2 3 4 5 6 7"}
GPU_UTIL=${GPU_UTIL:-0.94}
MAX_LEN=${MAX_LEN:-1048576}
# [保留] MAX_SEQS 影响 CAPTURE_SIZES 的桶数（32 → 最大桶 192，比 4 多约 5 个桶，
# 首次捕获多花 1~2 min）。越小越省启动时间，越大越能扛并发。默认取生产值。
MAX_SEQS=${MAX_SEQS:-32}
# [PREFIX-CACHE] 面向用户的默认 = **开**（生产形态：prefix caching ON）。
#   * 关掉它（做"无前缀缓存"的性能对照）用： NO_PREFIX=1
#   * 也可以直接 PREFIX=0（等价写法，两者给一个就行）
#   * 两个都显式给且冲突时以 PREFIX 为准，并打印提醒
#   注意：prefix caching ON/OFF 的 ms/step 与接受长度**不可直接混比**，
#   报数字时必须写明用的是哪种口径。
PREFIX=${PREFIX:-}
if [ -n "${NO_PREFIX:-}" ] && [ "${NO_PREFIX}" != "0" ]; then
  if [ -n "$PREFIX" ] && [ "$PREFIX" != "0" ]; then
    echo "[serve_a2] WARNING: NO_PREFIX=$NO_PREFIX 与 PREFIX=$PREFIX 冲突；以 PREFIX=$PREFIX 为准"
  else
    PREFIX=0
  fi
fi
PREFIX=${PREFIX:-1}
BAT_TOKENS=${BAT_TOKENS:-2048}
BLOCK=${BLOCK:-128}
KV_DTYPE=${KV_DTYPE:-bfloat16}
STATIC_KERNEL=${STATIC_KERNEL:-1}
NPUGRAPH_EX=${NPUGRAPH_EX:-1}
SP_TOKENS=${SP_TOKENS:-5}
SPEC=${SPEC:-1}
ENGRAM=${ENGRAM:-1}
VISION=${VISION:-1}
HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-8}
HOTSPIKE=${HOTSPIKE:-0}
ROUTE_PROBE=${ROUTE_PROBE:-0}
LOAD_FORMAT=${LOAD_FORMAT:-}
# 已验证开关（默认全开）—— 关掉任何一个都会变慢，除非在排查
MOE_AG=${MOE_AG:-1}            # MoE AllGather            −4.25 ms @128K
O_PROJ_2D=${O_PROJ_2D:-1}      # F3 wo_a 2D matmul        −0.31~0.76 ms
MOE_MASK=${MOE_MASK:-1}        # moe-mask-range           −0.51 ms
ROPE_IDXSEL=${ROPE_IDXSEL:-1}  # rope-idxsel              −0.45~0.62 ms
ENGRAM_JIT=${ENGRAM_JIT:-1}    # hash/plan numba JIT      −0.54 / −0.11 ms
QLI_NOCAND=${QLI_NOCAND:-1}    # QLI no-candidate         −0.49 ms
LOCAL_OWNER=${LOCAL_OWNER:-fast}
GATE_CHUNK=${GATE_CHUNK:-0}
GATE_MAX_TOKENS=${GATE_MAX_TOKENS:-2048}
# [TOOL_CALLING] 默认 1：用官方 deepseek_v41 前端（tokenizer + reasoning/tool parser）。
#   旧前端 `--tokenizer-mode=deepseek_v4` 的 chat template 不能正确处理 agent 工具调用
#   （DSML 标签形态不同、默认 thinking 开关不同），A2 首轮实测暴露的问题之一。
#   置 0 可回到旧前端，仅用于与历史数据对齐。
TOOL_CALLING=${TOOL_CALLING:-1}
# [PGO-AUTODETECT] 若 TARGET_PATH.txt 缺失（v5 的常见故障：它是 build_image.sh 生成的，
# 不在包里），这里自动探测一次并落盘，避免"静默降级 PYTHON_PGO=0"。
PYTHON_PGO=${PYTHON_PGO:-1}
# ❌ 未验证（默认关；不要在生产/正式测试里打开）
MOE_ZERO=${MOE_ZERO:-0}
DRAFT_GRAPH=${DRAFT_GRAPH:-0}
# ❌ 负结果（默认关）：只零化 MoE 无效行中的非有限元素。同会话交错 A/B N=24/臂：
# clean 2/24 vs 2/24 ⇒ **无差异** ⇒ `0 权重 × Inf = NaN` 通道被排除。
MOE_NF=${MOE_NF:-0}
# ⚠️ 仅诊断（默认不传 = 与 v3/历史口径一致）：HCCL 确定性归约。
#   合法值 **必须**是 true/false/strict —— 写 1 会 Config_Error_Invalid_Environment_Variable(EI0001)。
#   实测代价（正确性线，真权重）：true 把 GSM8K 打到 **91/100** ⇒ **不能进交付**；
#   strict 保精度（100/100）但 ≤18432 的"确定性"只是每批抽签（同一 ctx 三次重复 0.913/0.000/1.435）。
#   所以 v4 只把它当工具，默认不传。详见 CORRECTNESS_STATUS.md。
HCCL_DET=${HCCL_DET:-}
# 挂载模式 / 其他
PATCH_MODE=${PATCH_MODE:-baked}     # baked（用镜像内烘焙版本）| mount（-v 挂 patches/files/*）
CAND_MODE=${CAND_MODE:-0}           # 诊断用，0=不挂 indexer.py
# [CPU_BIND] 1 = 打开 vllm-ascend 内部绑核（additional-config 的 enable_cpu_binding）。
# 外部不再设 cpuset/mems（CPUSET/MEMS 默认 -1），绑核位置完全由内部按 NPU 拓扑决定。
CPU_BIND=${CPU_BIND:-1}
MULTISTREAM=${MULTISTREAM:-1}
# [CPUS-ALIAS] v5 的变量名是 CPUSET（与 MEMS 不对称），用户传 CPUS=-1 会被静默忽略。
# 这里同时接受两个名字，并在两者都给了时以 CPUSET 为准（同时打印提醒）。
# 默认 **-1 = 外部完全不管 CPU**（绑核交给 vLLM 内部的 enable_cpu_binding，见下面 [CPU/NUMA]）。
if [ -n "${CPUS:-}" ] && [ -z "${CPUSET:-}" ]; then
  CPUSET="$CPUS"
  echo "[serve_a2] NOTE: 收到 CPUS=$CPUS（v5 里这个名字不生效）；已按 CPUSET 处理。"
elif [ -n "${CPUS:-}" ] && [ -n "${CPUSET:-}" ] && [ "$CPUS" != "$CPUSET" ]; then
  echo "[serve_a2] WARNING: CPUS=$CPUS 与 CPUSET=$CPUSET 冲突；以 CPUSET 为准。"
fi
CPUSET=${CPUSET:--1}
MEMS=${MEMS:--1}
RUN_ID=${RUN_ID:-a2_$(date +%Y%m%d_%H%M%S)}
CACHE=${CACHE:-$PKG/cache}
OUT=${OUT:-$PKG/results/$RUN_ID}
LOG=${LOG:-$OUT/serve.log}
READY_TIMEOUT=${READY_TIMEOUT:-2100}
WAIT_READY=${WAIT_READY:-1}
# [DRY_RUN] 1 = 只走"开关解析 + MOUNTS 组装"并打印结果，**不碰 docker**。
# 用途：`set -u` 下的变量顺序 bug（`bash -n` 抓不到，只有真正展开变量才暴露）
# 由 tests/multibatch/verify_serve_flags.sh 的 12 组合矩阵调用。
DRY_RUN=${DRY_RUN:-0}

say() { printf '\n\033[1m[serve_a2]\033[0m %s\n' "$*" | tee -a "$OUT/driver.log"; }
# [FAIL-CLEANUP] v5 的坑：容器入口是 `bash -lc "sleep infinity"`，vLLM 崩了容器**还活着**，
# 一直占着 ~313 GB（Engram 206 GB 常驻 + 权重）。这直接导致"第二次起服叠加失败"。
# 这里在 die() 里清理：只要容器已经创建过，失败时就删掉它（日志已落盘，不会丢证据）。
_CONTAINER_STARTED=0
die() {
  printf '\n\033[31m[serve_a2][FAIL]\033[0m %s\n' "$*" >&2
  if [ "${_CONTAINER_STARTED:-0}" = "1" ] && [ -n "${NAME:-}" ] && [ -n "${DOCKER:-}" ]; then
    printf '\033[33m[serve_a2]\033[0m 清理容器 %s（避免悬挂占用内存；日志保留在 %s）\n' \
      "$NAME" "${LOG:-<无>}" >&2
    $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  exit 1
}

if [ "$DRY_RUN" = "1" ]; then
  MODEL=${MODEL:-/nonexistent/DRY_RUN_MODEL}
  DOCKER=docker
  # dry-run 不落地任何东西到包里（避免污染 results/ / cache/）
  OUT=${OUT_DRYRUN_DIR:-/tmp/serve_a2_dryrun}
  mkdir -p "$OUT" || die "dry-run 目录建不出来：$OUT"
  LOG=$OUT/serve.log
  CACHE=$OUT/cache
else
  [ -n "$MODEL" ] || die "必须设置 MODEL=<模型目录>"
  [ -d "$MODEL" ] || die "模型目录不存在：$MODEL"
  DOCKER="docker"
  $DOCKER info >/dev/null 2>&1 || DOCKER="sudo -n docker"
  $DOCKER info >/dev/null 2>&1 || die "无法访问 docker（试过 docker 与 sudo -n docker）"
  $DOCKER image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "镜像 $IMAGE 不存在。先执行：bash scripts/build_image.sh"
fi

# =============================================================================
# [SYMLINK-MODEL-MOUNT] 量化流水线产出的模型目录是**软链构造**的（零拷贝），
# 且软链写的是**绝对路径**，链条可达 5 层（L5→L4→L3→L2→L1）。
# 只挂 `-v "$MODEL:$MODEL"` 会让容器里所有软链**悬空** -> 起服报
# `No such file or directory`（通常最先在 config.json / tokenizer 上炸）。
# 实测复现（宿主机一切正常，容器内 cat 失败）：
#   docker run --rm -v <leaf>:/m:ro alpine cat /m/config.json
#   -> cat: can't open '/m/config.json': No such file or directory
#
# 这里用 tools/model_mount_args.sh 逐跳解析**字面软链**，把每个目标目录都挂上。
# 覆盖：MODEL_MOUNT_MODE=auto（默认，逐目录）| ancestor（只挂公共祖先，更宽但更省）；
#       MODEL_MOUNT_MODE=none（旧行为，只挂 MODEL —— 仅用于复现该故障）。
# 另外可用 EXTRA_MODEL_MOUNTS 追加（分号分隔的宿主路径）。
# =============================================================================
MODEL_MOUNT_MODE=${MODEL_MOUNT_MODE:-auto}
MODEL_MOUNTS=()
# 注意：这里用 -f 而不是 -x —— 交付包里脚本的执行位可能在打包/解包过程中丢失，
# 我们本来就用 `bash <script>` 调用，不需要执行位。（曾因 -x 导致静默走 fallback，
#  只挂 MODEL 一层 —— 正是本次要修的 bug 又复现了一遍。）
if [ "$MODEL_MOUNT_MODE" != "none" ] && [ -f "$PKG/tools/model_mount_args.sh" ]; then
  _mma_err=$(mktemp)
  _links=$(bash "$PKG/tools/model_mount_args.sh" "$MODEL" 2>"$_mma_err")
  _mma_rc=$?
  if [ "$_mma_rc" -ne 0 ]; then
    say "⚠️  软链解析发现问题（见下）"
    cat "$_mma_err" >&2
    die "模型目录存在悬空软链 —— 起服必然失败。请先修复软链（装配脚本应写绝对路径且源目录不能移动）。"
  fi
  # ---------------------------------------------------------------------------
  # 组参数：`-v` 与 `路径:路径:ro` 必须是**两个独立的数组元素**。
  #
  # ⚠️ 踩过的坑（A2 实测起服失败）：`mapfile` 每行只给**一个**元素，如果上游
  #    输出的是 "-v /path:/path:ro"，那整个串会变成一个参数，Go 的 pflag 把
  #    `-v` 后的空格也算进值里 ⇒ docker 报
  #      create  /path: " /path" includes invalid characters for a local volume name
  #    （错误信息里路径前面那个空格就是指纹）。
  #    所以上游改成只吐裸路径，这里显式拼成 2 个元素。
  # ---------------------------------------------------------------------------
  mapfile -t _mdirs < <(printf '%s\n' "$_links" | sed '/^[[:space:]]*$/d')
  if [ "$MODEL_MOUNT_MODE" = "ancestor" ] && [ "${#_mdirs[@]}" -gt 1 ]; then
    _anc=$(printf '%s\n' "${_mdirs[@]}" | xargs -r -n1 dirname | sort -u | head -1)
    if [ -n "${_anc:-}" ] && [ -d "$_anc" ]; then
      MODEL_MOUNTS=(-v "$_anc:$_anc:ro")
      say "模型挂载（ancestor 模式，1 个目录）：$_anc"
    else
      MODEL_MOUNT_MODE=auto       # 取不到公共祖先就退回 auto
    fi
  fi
  if [ "${#MODEL_MOUNTS[@]}" -eq 0 ]; then
    for _d in "${_mdirs[@]}"; do
      MODEL_MOUNTS+=(-v "$_d:$_d:ro")
    done
    say "模型挂载（auto 模式，${#_mdirs[@]} 个目录，含软链链条）"
    for _d in "${_mdirs[@]}"; do say "   -v $_d:$_d:ro"; done
  fi
  rm -f "$_mma_err"
else
  MODEL_MOUNTS=(-v "$MODEL:$MODEL:ro")
  if [ "$MODEL_MOUNT_MODE" = "none" ]; then
    say "模型挂载（none 模式 —— 软链会悬空，仅用于复现故障）"
  else
    say "⚠️  找不到 tools/model_mount_args.sh，退回只挂 MODEL 一层（软链会悬空）"
  fi
fi

if [ -n "${EXTRA_MODEL_MOUNTS:-}" ]; then
  IFS=';' read -r -a _extra <<<"$EXTRA_MODEL_MOUNTS"
  for _p in "${_extra[@]}"; do
    [ -n "$_p" ] && MODEL_MOUNTS+=(-v "$_p:$_p:ro")
  done
fi

mkdir -p "$OUT" "$CACHE/vllm" "$CACHE/npugraph" "$CACHE/skcache/compile_outputs" "$CACHE/skcache/install" "$CACHE/numba"

# =============================================================================
# [SKCACHE-GC] static kernel 编译会在 compile_outputs/ 下留一堆 ts<时间戳>_pid<N>_outputs/
# 临时目录，**从不自动清理**：A3 上攒到 1491 个 / 847 MB。真正要保留并复用的是
# static_kernel_cache/（缓存键 = CANN-<版本>_<SoC>，见 static_kernel.py）。
# 这里在每次起服前清掉临时目录，但**绝不碰 static_kernel_cache/**。
# 可关闭：SKCACHE_GC=0。
# =============================================================================
SKCACHE_GC=${SKCACHE_GC:-1}
_skc="$CACHE/skcache/compile_outputs"
if [ "$SKCACHE_GC" = "1" ] && [ -d "$_skc" ]; then
  _n=$(find "$_skc" -maxdepth 1 -type d -name 'ts*_outputs' 2>/dev/null | wc -l)
  if [ "${_n:-0}" -gt 0 ]; then
    find "$_skc" -maxdepth 1 -type d -name 'ts*_outputs' -exec rm -rf {} + 2>/dev/null || true
    echo "[serve_a2] skcache GC: 清掉 $_n 个 ts*_outputs 临时目录（保留 static_kernel_cache/）"
  fi
fi
if [ -d "$_skc/static_kernel_cache" ]; then
  echo "[serve_a2] skcache: static_kernel_cache/ 命中（$(ls -1 "$_skc/static_kernel_cache" 2>/dev/null | grep -c '\.json$') 个缓存文件）"
else
  echo "[serve_a2] skcache: 无 static_kernel_cache/ ⇒ 本次要冷编译（每 SoC 一次性，A2 首次 15–20 min）"
fi

# ---------- [CAPTURE_SIZES] 每步 token 数 = 并发请求数 × (1 + SP_TOKENS) ----------
# `1 + SP_TOKENS` 若不在 cudagraph_capture_sizes 里，aclgraph 会 padding 到更大的桶
# （实测 +5.9 ms/step）。MAX_SEQS>1 时还必须覆盖到 MAX_SEQS × (1+SP_TOKENS)，否则大
# batch 会被 padding 到"没有图"→ 报错或退回 eager（A2 生产 MAX_SEQS=32 ⇒ 需要 192）。
if [ -z "${CAPTURE_SIZES:-}" ]; then
  _step_tokens=$(( SP_TOKENS + 1 ))
  CAPTURE_SIZES="1,2,3,4"
  # [MULTI-SEQ-CAPTURE] MAX_SEQS=1（历史性能口径）时下面这行算出 32，桶列表与 v3
  # **逐字节相同**（1,2,3,4,6,8,12,16,20,24,32）；MAX_SEQS=32 时扩到 192。
  # 桶列故意稀疏（几何级数）：每个桶要多花 ~10-30 s 捕获，32 个桶不现实。
  _cap_max=$(( MAX_SEQS * _step_tokens ))
  [ "$_cap_max" -lt 32 ] && _cap_max=32
  for _c in 6 8 12 16 20 24 32 40 48; do
    if [ "$_c" -ge "$_step_tokens" ] && [ "$_c" -le "$_cap_max" ]; then CAPTURE_SIZES="$CAPTURE_SIZES,$_c"; fi
  done
  # 48 以上按 2 倍增长（桶越少捕获越快；padding 只浪费算力，不影响正确性）。
  _b=96
  while [ "$_b" -le "$_cap_max" ]; do CAPTURE_SIZES="$CAPTURE_SIZES,$_b"; _b=$(( _b * 2 )); done
  case ",$CAPTURE_SIZES," in
    *",$_step_tokens,"*) : ;;
    *) CAPTURE_SIZES="$CAPTURE_SIZES,$_step_tokens" ;;
  esac
  # 保证最大桶 >= _cap_max（vLLM 会把超过最大桶的 batch 直接判为不可图）
  case ",$CAPTURE_SIZES," in
    *",$_cap_max,"*) : ;;
    *) CAPTURE_SIZES="$CAPTURE_SIZES,$_cap_max" ;;
  esac
fi

# ---------- [PYTHON_PGO] 只在产物存在且版本匹配时启用 ----------
PGO_LIB=""
if [ "$PYTHON_PGO" = "1" ]; then
  # [PGO-AUTODETECT] v5 的行为：TARGET_PATH.txt 缺失 → 只打一行 WARNING 就静默降级，
  # 很容易被漏掉（这正是用户遇到的情况）。v6 改为**自动探测并落盘**。
  if [ "$DRY_RUN" != "1" ] \
     && [ ! -f "$PKG/optim/pgo/TARGET_PATH.txt" ] \
     && [ -f "$PKG/optim/pgo/libpython3.12.so.1.0" ]; then
    echo "[serve_a2] optim/pgo/TARGET_PATH.txt 缺失 → 自动探测容器内 libpython 落点…"
    _detected=$($DOCKER run --rm --entrypoint python3 "$IMAGE" - <<'PYEOF' 2>/dev/null | tail -1
import os, sysconfig
try:
    name = sysconfig.get_config_var('INSTSONAME') or 'libpython%s.so.1.0' % sysconfig.get_config_var('VERSION')
    dirs = [sysconfig.get_config_var('LIBDIR'), '/usr/local/python3.12.13/lib',
            '/usr/lib', '/usr/local/lib']
    for d in dirs:
        if d and os.path.exists(os.path.join(d, name)):
            print(os.path.join(d, name)); break
except Exception:
    pass
PYEOF
)
    case "${_detected:-}" in
      /*.so*) printf '%s' "$_detected" > "$PKG/optim/pgo/TARGET_PATH.txt"
              echo "[serve_a2]   探测到: $_detected（已写入 TARGET_PATH.txt）" ;;
      *)      echo "[serve_a2]   ⚠️ 探测失败：容器内找不到 libpython。PGO 降级为 0。" ;;
    esac
  fi
  if [ -f "$PKG/optim/pgo/libpython3.12.so.1.0" ] && [ -f "$PKG/optim/pgo/TARGET_PATH.txt" ]; then
    PGO_LIB=$(cat "$PKG/optim/pgo/TARGET_PATH.txt")
    case "$PGO_LIB" in *.so*) : ;; *) PGO_LIB="" ;; esac
  fi
  if [ -z "$PGO_LIB" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "[serve_a2] NOTE: dry-run 不校验/不探测 PGO 落点（TARGET_PATH.txt 由 build_image.sh 生成）"
    else
      echo "[serve_a2] WARNING: PYTHON_PGO=1 但没有可用的 PGO 产物 → 降级为不挂"
    fi
    PYTHON_PGO=0
  fi
fi

# ---------- [CPU/NUMA] 交给 vLLM 内部绑核，外部不做 ----------
# 原则（2026-09-17 定）：**不在外部计算绑核位置**。
#   * 容器默认**不设** --cpuset-cpus / --cpuset-mems（= 看得到全部 CPU/NUMA）；
#   * 由 vllm-ascend 内部的 cpu_binding 自己按 NPU 拓扑给每个 rank 绑核：
#       additional-config 的 "enable_cpu_binding": true   ← 由 CPU_BIND=1 控制
#     起服后在日志里能看到它的决策：
#       [cpu_binding.py] [cpu_bind_mode] mode=topo_affinity rank=N visible_npus=[...]
#       [cpu_binding.py] NPUx: main=[...] acl=[...] release=[[...]]
#       [cpu_binding.py] [migrate] NPU:N -> NUMA [M]
#
# 为什么不用外部 cpuset：外部先把容器圈到某几个 NUMA 上，等于**替内部绑核做了决定**，
#   一旦选卡组合跟外部区间对不上（多机、多租户、混合选卡），内部再绑也绑不回正确的节点。
#   外部不设限 + 内部按拓扑绑，才是"位置由知道拓扑的那一方决定"。
#
# CPUSET/MEMS 仍保留为**高级逃生口**（做 AB 或复现历史口径时才用）：
#   CPUSET=<核列表> MEMS=<节点列表>  → 显式透传给 docker
#   CPUSET=-1 MEMS=-1（默认）        → 不加任何 cpuset 参数
CGROUP_ARGS=()
# 兼容老写法的 "auto"：语义 = 外部不管（绑核交给 vLLM 内部），不再做任何推导。
case "${CPUSET:--1}" in auto|AUTO) CPUSET=-1 ;; esac
case "${MEMS:--1}"   in auto|AUTO) MEMS=-1 ;; esac
if [ "${CPUSET:--1}" != "-1" ] && [ "${CPUSET:--1}" != "none" ]; then
  CGROUP_ARGS+=(--cpuset-cpus "$CPUSET")
fi
if [ "${MEMS:--1}" != "-1" ] && [ "${MEMS:--1}" != "none" ]; then
  CGROUP_ARGS+=(--cpuset-mems "$MEMS")
fi
if [ "${#CGROUP_ARGS[@]}" -gt 0 ]; then
  echo "[serve_a2] NOTE: 外部显式绑核 CPUSET=$CPUSET MEMS=$MEMS ⇒ 会覆盖 vLLM 内部绑核的结果（默认不这么做）"
fi

# ---------- [MOUNTS] ----------
MOUNTS=()
if [ "$PATCH_MODE" = "mount" ]; then
  F=$PKG/patches/files
  MOUNTS+=(-v "$F/engram_hbm.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hbm.py:rw")
  MOUNTS+=(-v "$F/engram_hash.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py:rw")
  MOUNTS+=(-v "$F/engram_jit_kernel.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_jit_kernel.py:ro")
  MOUNTS+=(-v "$F/engram_plan_kernel.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_plan_kernel.py:ro")
  MOUNTS+=(-v "$F/engram_gate.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_gate.py:ro")
  MOUNTS+=(-v "$F/model.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:rw")
  MOUNTS+=(-v "$F/ascend_forward_context.py:/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py:ro")
  MOUNTS+=(-v "$F/rope_dsv4.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/rope_dsv4.py:ro")
  if [ "$DRAFT_GRAPH" = "1" ]; then
    # draft 版 dsa_v1.py 是「stock + 0004 + F3」合并版 ⇒ 此时不要再挂 F3 版（同一目标路径会 duplicate mount）
    MOUNTS+=(-v "$F/draft/dsa_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py:ro")
    MOUNTS+=(-v "$F/draft/dspark_proposer.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:ro")
    MOUNTS+=(-v "$F/draft/llm_base_proposer.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:ro")
  else
    MOUNTS+=(-v "$F/dsa_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py:rw")
  fi
  if [ "$MOE_NF" != "0" ]; then
    # 负结果臂（不采纳）：只零化非有限元素；文件缺失时退回 moemask 版并告警
    if [ -f "$F/token_dispatcher_moennf.py" ]; then
      MOUNTS+=(-v "$F/token_dispatcher_moennf.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:ro")
    else
      echo "[serve_a2] WARNING: MOE_NF=$MOE_NF 但缺 patches/files/token_dispatcher_moennf.py → 退回 moemask 版"
      MOUNTS+=(-v "$F/token_dispatcher_moemask.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:ro")
    fi
  elif [ "$MOE_ZERO" != "0" ]; then
    MOUNTS+=(-v "$F/token_dispatcher_moezero.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:ro")
  else
    MOUNTS+=(-v "$F/token_dispatcher_moemask.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:ro")
  fi
  if [ "$CAND_MODE" != "0" ]; then
    MOUNTS+=(-v "$F/indexer.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:rw")
  fi
fi
[ -n "$PGO_LIB" ] && MOUNTS+=(-v "$PKG/optim/pgo/libpython3.12.so.1.0:$PGO_LIB:ro")
# [HCCL-DET] 只在非空时传（空值会让 HCCL 报 EI0001）
HCCL_ENV_ARGS=()
if [ -n "$HCCL_DET" ]; then HCCL_ENV_ARGS+=(-e "HCCL_DETERMINISTIC=$HCCL_DET"); fi

DEV_ARGS=(); ARTV=""
for d in $DEVS; do DEV_ARGS+=(--device "/dev/davinci$d"); ARTV="$ARTV,$d"; done
ARTV=${ARTV#,}

# ---------- [DRY_RUN] 开关矩阵烟测出口（不碰 docker） ----------
if [ "$DRY_RUN" = "1" ]; then
  echo "[a2-dry] OK"
  echo "[a2-dry] image=$IMAGE name=$NAME port=$PORT devs='$DEVS' util=$GPU_UTIL max_len=$MAX_LEN"
  echo "[a2-dry] MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX SP_TOKENS=$SP_TOKENS BAT_TOKENS=$BAT_TOKENS"
  echo "[a2-dry] CAPTURE_SIZES=$CAPTURE_SIZES"
  echo "[a2-dry] MOE_AG=$MOE_AG O_PROJ_2D=$O_PROJ_2D MOE_MASK=$MOE_MASK ROPE_IDXSEL=$ROPE_IDXSEL ENGRAM_JIT=$ENGRAM_JIT QLI_NOCAND=$QLI_NOCAND LOCAL_OWNER=$LOCAL_OWNER"
  echo "[a2-dry] MOE_ZERO=$MOE_ZERO MOE_NF=$MOE_NF DRAFT_GRAPH=$DRAFT_GRAPH PYTHON_PGO=$PYTHON_PGO pgo_target=${PGO_LIB:-none} LOAD_FORMAT=${LOAD_FORMAT:-<real>} CAND_MODE=$CAND_MODE PATCH_MODE=$PATCH_MODE"
  echo "[a2-dry] CPUSET=$CPUSET${CPUSET_SRC:+ ($CPUSET_SRC)} MEMS=$MEMS${MEMS_SRC:+ ($MEMS_SRC)} STATIC_KERNEL=$STATIC_KERNEL NPUGRAPH_EX=$NPUGRAPH_EX MULTISTREAM=$MULTISTREAM HCCL_DET=${HCCL_DET:-none}"
  echo "[a2-dry] MOUNTS(${#MOUNTS[@]}): ${MOUNTS[*]:-<none>}"
  echo "[a2-dry] MODEL_MOUNT_MODE=$MODEL_MOUNT_MODE MODEL_MOUNTS(${#MODEL_MOUNTS[@]}):"
  for _m in "${MODEL_MOUNTS[@]}"; do echo "[a2-dry]    $_m"; done
  exit 0
fi

say "起容器 $NAME（image=$IMAGE port=$PORT devs='$DEVS' util=$GPU_UTIL pgo=$PYTHON_PGO patch_mode=$PATCH_MODE mseqs=$MAX_SEQS prefix=$PREFIX）"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
: > "$LOG"
# [MEMLOCK] A2 实测：pin_memory 报 207001（`aclrtMallocHostWithCfg` 失败）而 free -g
# 仍有 667 GiB 空闲 —— 不是容量问题。解法是更新 driver + 解除 memlock 限制。
# 这里默认 `--ulimit memlock=-1`（无限），A3 上无副作用。
$DOCKER run -d --name "$NAME" --net=host --shm-size=512g --privileged=true \
  --ulimit memlock=-1 \
  "${CGROUP_ARGS[@]}" \
  "${DEV_ARGS[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  "${MODEL_MOUNTS[@]}" \
  -v "$CACHE/vllm:/root/.cache/vllm" \
  -v "$CACHE/npugraph:/root/npugraph_ex_cache" \
  -v "$CACHE/numba:/numba_cache" \
  -v "$CACHE/skcache/compile_outputs:/vllm-workspace/vllm/static_kernel_compile_outputs" \
  -v "$CACHE/skcache/compile_outputs:/vllm-workspace/vllm-ascend/static_kernel_compile_outputs" \
  -v "$CACHE/skcache/install:/vllm-workspace/vllm-ascend/static_kernel_install" \
  -v "$OUT:/opt/dsv41/results/$RUN_ID" \
  "${MOUNTS[@]}" \
  -e ASCEND_RT_VISIBLE_DEVICES="$ARTV" \
  -e LOCAL_WORLD_SIZE="$LOCAL_WORLD_SIZE" \
  -e HCCL_BUFFSIZE="$HCCL_BUFFSIZE" \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e TASK_QUEUE_ENABLE=1 \
  -e HCCL_OP_EXPANSION_MODE=AIV \
  -e ASCEND_MAX_OP_CACHE_SIZE=-1 \
  -e CAPTURE_SIZES="$CAPTURE_SIZES" \
  -e NUMBA_CACHE_DIR=/numba_cache \
  -e VLLM_ADMISSION_GATE=1 \
  -e V41_ENGRAM_HOST_RESIDENT=1 -e V41_ENGRAM_REUSE_EP_GROUP=1 \
  -e V41_ENGRAM_GATE_CHUNK="$GATE_CHUNK" -e V41_ENGRAM_GATE_MAX_TOKENS="$GATE_MAX_TOKENS" -e MAX_TOKENS="$GATE_MAX_TOKENS" \
  -e V41_ENGRAM_GATE_HOIST=0 \
  -e V41_ENGRAM_JIT="$ENGRAM_JIT" \
  -e V41_QLI_NO_CANDIDATE="$QLI_NOCAND" \
  -e V41_MOE_COMM_ALLGATHER="$MOE_AG" \
  -e V41_MOE_MASK_RANGE="$MOE_MASK" \
  -e V41_ROPE_IDXSEL="$ROPE_IDXSEL" \
  -e V41_O_PROJ_2D="$O_PROJ_2D" \
  -e V41_ENGRAM_ROUTE_PROBE="$ROUTE_PROBE" \
  -e V41_MOE_ZERO_INVALID="$MOE_ZERO" -e V41_MOE_ZERO_INVALID_FILE=/tmp/v41_moe_zero_file \
  -e V41_MOE_ZERO_NONFINITE="$MOE_NF" -e V41_MOE_ZERO_NONFINITE_FILE=/tmp/v41_moe_nf \
  -e V41_FORCE_CAND_MODE="$CAND_MODE" \
  -e LOAD_FORMAT="$LOAD_FORMAT" \
  ${HCCL_ENV_ARGS[@]+"${HCCL_ENV_ARGS[@]}"} \
  -w /workspace "$IMAGE" \
  bash -lc "sleep infinity" >/dev/null || die "docker run 失败"
_CONTAINER_STARTED=1     # [FAIL-CLEANUP] 之后任何 die() 都会删掉这个容器

# ---------- 容器内环境 + 起服 ----------
# engram local-owner 用 /tmp 文件热切换（与 A3-node1 完全一致）
$DOCKER exec "$NAME" bash -lc "printf '%s' '$LOCAL_OWNER' > /tmp/v41_engram_localowner; printf '%s' 'fast' > /tmp/v41_hash_mode" || true

if [ "$DRAFT_GRAPH" = "1" ]; then
  # 用 probe_draft 的三个整文件覆盖（含 0002/0004/0005/0006 + F3），并打开 draft 图捕获
  say "DRAFT_GRAPH=1（实验，未验证）：用 patch_mode=mount 挂 draft 版文件"
  $DOCKER exec "$NAME" bash -lc "ls -la /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py" >/dev/null
fi

mkdir -p "$OUT"
{
  echo "[serve_a2] run_id=$RUN_ID image=$IMAGE model=$MODEL"
  echo "[serve_a2] port=$PORT tp=$TP util=$GPU_UTIL max_len=$MAX_LEN max_seqs=$MAX_SEQS bat=$BAT_TOKENS"
  echo "[serve_a2] sptok=$SP_TOKENS capture_sizes=$CAPTURE_SIZES"
  echo "[serve_a2] MOE_AG=$MOE_AG O_PROJ_2D=$O_PROJ_2D MOE_MASK=$MOE_MASK ROPE_IDXSEL=$ROPE_IDXSEL"
  echo "[serve_a2] ENGRAM_JIT=$ENGRAM_JIT QLI_NOCAND=$QLI_NOCAND LOCAL_OWNER=$LOCAL_OWNER GATE_CHUNK=$GATE_CHUNK"
  echo "[serve_a2] PYTHON_PGO=$PYTHON_PGO pgo_target=${PGO_LIB:-none} STATIC_KERNEL=$STATIC_KERNEL NPUGRAPH_EX=$NPUGRAPH_EX"
  echo "[serve_a2] LOAD_FORMAT=${LOAD_FORMAT:-<real weights>} MOE_ZERO=$MOE_ZERO(unverified) MOE_NF=$MOE_NF(negative-result) DRAFT_GRAPH=$DRAFT_GRAPH(unverified)"
  echo "[serve_a2] 口径：MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX(0=性能口径/1=生产口径) capture_max=${CAPTURE_SIZES##*,}"
  echo "[serve_a2] 诊断项：HCCL_DET=${HCCL_DET:-none}（true 会掉 GSM8K 到 91/100，仅诊断）"
  echo "[serve_a2] cpuset=$CPUSET mems=$MEMS"
} | tee "$OUT/serve_cmd.txt"

INNER=/opt/dsv41/results/$RUN_ID/inner.sh
cat > "$OUT/inner.sh" <<INNER_EOF
#!/usr/bin/env bash
set -uo pipefail
cd /workspace
export MODEL="$MODEL" TP=$TP DP=1 PORT=$PORT SERVED_NAME=deepseek-v41
export MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS BAT_TOKENS=$BAT_TOKENS GPU_UTIL=$GPU_UTIL BLOCK=$BLOCK
export KV_DTYPE=$KV_DTYPE GRAPH=1 EAGER=0 PREFIX=$PREFIX SPEC=$SPEC SP_TOKENS=$SP_TOKENS
if [ "$DRAFT_GRAPH" = "1" ]; then export SPEC_EAGER=0; else export SPEC_EAGER=1; fi
export ENGRAM=$ENGRAM ENGRAM_STORAGE=int8 VISION=$VISION
export NPUGRAPH_EX=$NPUGRAPH_EX STATIC_KERNEL=$STATIC_KERNEL CPU_BIND=$CPU_BIND
export MULTISTREAM=$MULTISTREAM DSA_OVERLAP=1 FUSED_MC2=0 MC2=0 MC2_HIER=0 REDUCE_SAMPLE=0
export LOADER_MT=1 LAZY=1
export V41_KV_TIER=off
export V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner
export CAPTURE_SIZES="$CAPTURE_SIZES"
export ASCEND_MAX_OP_CACHE_SIZE=-1
if [ "$TOOL_CALLING" = "1" ]; then
  export EXTRA='--tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --tool-call-parser=deepseek_v41 --enable-auto-tool-choice --default-chat-template-kwargs={"enable_thinking":false}'
else
  export EXTRA='--tokenizer-mode=deepseek_v4 --default-chat-template-kwargs={"enable_thinking":false}'
fi
if [ -n "$LOAD_FORMAT" ]; then
  export V41_ENGRAM_WITH_DUMMY=1 V41_DUMMY_WO_A_FIX=1
fi
if [ "$DRAFT_GRAPH" = "1" ]; then export DSPARK_DRAFT_METADATA_MODE=sync; fi
echo "[a2] run_id=$RUN_ID port=$PORT static=$STATIC_KERNEL sptok=$SP_TOKENS capture_sizes=$CAPTURE_SIZES mseqs=$MAX_SEQS prefix=$PREFIX moeag=$MOE_AG local_owner=$LOCAL_OWNER pgo=$PYTHON_PGO load_format=\${LOAD_FORMAT:-real} draft_graph=$DRAFT_GRAPH moe_zero=$MOE_ZERO moe_nf=$MOE_NF"
md5sum /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hbm.py \\
       /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py 2>/dev/null
exec bash /opt/dsv41/scripts/serve_v2.sh
INNER_EOF
chmod +x "$OUT/inner.sh"

$DOCKER exec -d "$NAME" bash -lc "bash $INNER > /opt/dsv41/results/$RUN_ID/serve.log 2>&1"
say "服务已提交启动，日志：$LOG"

if [ "$WAIT_READY" != "1" ]; then
  echo "$RUN_ID" > "$PKG/.last_run_id"
  exit 0
fi

say "等待就绪（每 15 s 报一次，最多 $((READY_TIMEOUT/60)) 分钟；首次要编译 static kernel）"
t0=$(date +%s)
while :; do
  el=$(( $(date +%s) - t0 ))
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { say "就绪（用时 ${el}s）"; break; }
  if ! $DOCKER inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
    tail -30 "$LOG" 2>/dev/null; die "容器退出"
  fi
  if grep -qE "EngineCore.*(failed|Error)|NPUModelRunner failed|Traceback" "$LOG" 2>/dev/null; then
    grep -nE "Error|failed|Traceback" "$LOG" | tail -15; die "启动报错（见上）"
  fi
  [ "$el" -ge "$READY_TIMEOUT" ] && { tail -40 "$LOG"; die "等待超时"; }
  printf '  … %ss health=%s log=%sKB\n' "$el" "$code" "$(du -k "$LOG" 2>/dev/null | cut -f1)"
  sleep 15
done

# ---------- 起服必查 ①：static_kernel 是否被静默降级 ----------
SK_BAD=$(grep -ac "static_kernel.py:650" "$LOG" || true)
if [ "${SK_BAD:-0}" != "0" ]; then
  echo
  echo "  \033[31m✗ static_kernel 被静默降级（static_kernel.py:650 命中 $SK_BAD 次）\033[0m"
  echo "    原因通常是 LOCAL_WORLD_SIZE 没进 os.environ。请把本行日志 + serve_cmd.txt 发回。"
  echo "    临时处置：STATIC_KERNEL=0 重跑（慢 ~1 ms/step，但数值正确）"
else
  echo "  ✓ static_kernel 无降级（static_kernel.py:650 命中 0 次）"
fi
grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed 's/^/  /' || true
# 镜像指纹落到结果目录（make_report.sh 会读它）
$DOCKER exec "$NAME" bash -lc 'cat /opt/dsv41/BUILD_INFO.txt 2>/dev/null' > "$OUT/BUILD_INFO.txt" 2>/dev/null || true
$DOCKER exec "$NAME" bash -lc 'cat /opt/dsv41/BUILD_INFO.txt 2>/dev/null' | sed 's/^/  /' || true
echo "$RUN_ID" > "$PKG/.last_run_id"
echo "RUN_ID=$RUN_ID OUT=$OUT LOG=$LOG"
