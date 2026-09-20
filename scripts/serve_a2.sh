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
#   * 默认开 DRAFT_GRAPH=1（draft 入图，含 DSPARK_GRAPH_CAPTURE_METADATA 绑定与校验）
#   * 默认关未验证/负结果开关：MOE_ZERO=0、MOE_NF=0
#   * 缓存目录默认落在包目录 ./cache（A2 无外网，缓存持久化很重要）
#
# 常用变量（都有默认值，绝大多数不用改）：
#   MODEL      必填，模型目录
#   IMAGE      默认 dsv41-a2:v8
#   NAME       容器名，默认 dsv41-a2
#   PORT       默认 8100
#   GPU_UTIL   默认 0.92（**长 prompt 首 token 延迟的关键**；
#              机制与实测见 docs/prefill-memory-headroom.md）
#   MAX_SEQS   默认 4（**性能口径**；A2 生产是 32，见 MODE=prod）
#   PREFIX     默认 0 = 不启用 prefix caching（**历史性能口径**，与 v3 逐字节一致）
#              **A2 生产用 1**；PREFIX=1 时才会出现"decode 队列里插入新请求"这一生产形态
#   SP_TOKENS  默认 5（DSpark 原生 block size；CAPTURE_SIZES 按 S+1 自动推导）
#   PYTHON_PGO 默认 1（挂 optim/pgo 里编译好的 libpython；文件不存在则自动降级为 0）
#   LOAD_FORMAT 空 = 读真权重；dummy = 只按 shape 建模型（**只测时延，A 恒为 1.0**）
#   MOE_ZERO / MOE_NF 默认 0（未验证 / 负结果）；DRAFT_GRAPH 默认 1（见 CHANGELOG v8 §3）
#   MOE_NF     默认 0（负结果，不采纳；见 README「别踩坑」表）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"

MODEL=${MODEL:-}
IMAGE=${IMAGE:-dsv41-a2:v8}
NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8100}
TP=${TP:-8}
DEVS=${DEVS:-"0 1 2 3 4 5 6 7"}
# [MEM-HEADROOM] 默认 0.92，**不是贪图显存，而是留出 activation 余量**。
#
# 实测（8×910C）：真实 prefill 的 activation 峰值约 6 GiB，而 vLLM 在
# startup profiling 阶段只量到 3.21 GiB（那时 KV cache 还没分配）。
# 这个差额靠"显存余量"兜；余量不足时分配器要反复向驱动申请/归还，
# **模型 forward 会慢 2.1×（稳态）**，第一个请求还要额外付一次
# "把池子撑大"的约 6 秒。实测（同一台机、同一模型、同一 8192-token 请求）：
#
#   GPU_UTIL=0.94  余量 6.11 GiB   forward 慢  →  8K prefill  8.0–8.6 s
#   GPU_UTIL=0.92  余量 7.36 GiB   正常        →  8K prefill  1.14 s   ← 默认
#   GPU_UTIL=0.88  余量 9.80 GiB   正常        →  8K prefill  1.28 s
#
# ⇒ 0.92 与 0.88 等效，但保留更多 KV；0.94 是断崖。
#   代价：KV 池从 3.09M 降到约 2.82M tokens（−8.6%）。
#   详细机制与全部证据：docs/prefill-memory-headroom.md
#
# 适用范围：以上数字**全部在 A3（8×910C）上实测**。机制（profiling 量的
# activation 偏低 → 真实 prefill 峰值超出余量）与平台无关，A2 预期同样适用，
# 但 **A2 上没有复测**；若你在 A2 上遇到异常，先用
# docs/prefill-memory-headroom.md §6 的方法量一次首 token 延迟。
GPU_UTIL=${GPU_UTIL:-0.92}
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
# [BAT-TOKENS] ★ 正确性关键参数，不要随手调小。
#
# chunked prefill 会把长 prompt 切成 ceil(prompt / BAT) 段依次前向。每段都有一次
# 独立的"偏离"机会，且误差沿后续 chunk 累积 —— 实测偏离率约 2%/chunk。
# 因此 **chunk 数越多，长上下文正确率越低**，且是平滑下滑（不是阈值效应）。
#
# 实测（同一 needle 检索探针、每档 10 个不同 nonce、"埋事实在中段"）：
#     prompt_tokens   BAT=2048(chunk)   BAT=2048   BAT=8192(chunk)   BAT=8192
#        10,394            5              10/10          2             10/10
#        20,318           10               8/10          3             10/10
#        40,163           20               7/10          5             10/10
#        60,012           30               5/10          8             10/10
#        79,855           40               3/10         10             10/10
#       149,986           74              ~0%           19              6/6
#       259,985          127              ~0%           32              6/6
#   ⇒ BAT=8192 把这些档位全部拉到 100%，且同一 prompt 重复 10 次输出逐字节一致。
#
# 代价：activation 峰值更高，KV cache 从 4,145,957 → 3,088,738 tokens（本机 8×910C，
#       GPU_UTIL=0.94 时；默认已改为 0.92，KV 约 2.82M，见 §[MEM-HEADROOM]）。
#       若你的场景更看重 KV 容量、且上下文主要在 <20K，
#       可以显式设 BAT_TOKENS=2048。
BAT_TOKENS=${BAT_TOKENS:-8192}
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

# [DEVICE-INDEX] Engram 端到端设备化（表仍常驻 host DRAM，但由 device 算子直索）。
#
# **auto（默认）**：起服时探测本机能否 `aclrtHostRegister` 一个可写映射并让设备
#   算子直读它；支持就启用，不支持就静默回退到 host 路径（功能完全不变）。
#   为什么默认不是 1：该能力只在 A3（910C）实测过，**A2（910B3）从未验证**，
#   而且本项目的 `docs/A2_VS_A3_DIFF.md` §5 明确记着"任何『host 地址可以被
#   device kernel 直接读』的假设在 A2 上都是未验证"（A3 上 pinned 内存做同样
#   的事会报 507035 MTE invalid GM address）。默认 1 会让 A2 起不来或出错。
# 1 / 0：强制开 / 强制关。A3 验收建议用 1，避免"以为开了其实回退成 host 了"。
#
# 启用时会把 engram_int8 目录挂成 **:rw** —— `aclrtHostRegister` 拒绝只读 VMA
# （ret=507899）。代码本身只读这些文件。
ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-auto}
# [DEVICE-INDEX] 只有显式 0 才算关。auto / 1 都要按"可能启用"准备挂载：探测发生在
# 容器内，脚本此刻还不知道结果，而 aclrtHostRegister 只接受可写映射。
_engram_need_rw() {
  case "${ENGRAM_DEVICE_INDEX:-auto}" in
    0|false|off|no|"") return 1 ;;
    *) return 0 ;;
  esac
}
# 每步失败时是否回退 host 路径。默认 0（不回退）：回退要求 host 分片仍然有效，
# 而默认构建为省 25.75 GB/rank/层 的 DRAM 把分片裁掉了，此时回退会**静默读错**。
# 需要回退就两个都设 1（保留完整分片）。
ENGRAM_DEVICE_FALLBACK=${ENGRAM_DEVICE_FALLBACK:-0}

# [DROPCACHE] 起服前清 page cache（默认开）。详见起容器前那段注释。
DROPCACHE=${DROPCACHE:-1}

# [PROFILE] V41_PROFILE=1 给 vllm 传 --profiler-config，从而在 API 上挂出
# /start_profile 与 /stop_profile（vllm 只在 profiler_config.profiler 非空时才
# 注册这两个端点，见 entrypoints/serve/profile/api_router.py:attach_router）。
#   POST /start_profile  → 开始采集（8 卡同步）
#   POST /stop_profile   → 停止并落盘
# 采集文件写到容器内 /opt/dsv41/results/<run_id>/prof（= 宿主 results/<run_id>/prof）。
# 起服时打开它，才能在**起服之后**按需测任意时刻的性能，而不用重启。
# 代价：torch profiler 常驻会有一点开销，且必须记得 stop（否则文件一直涨）。
# 命名用 V41_PROFILE 而不是 PROFILE：PROFILE 在 shell 环境里是个常见名，
# 旧实验脚本（serve_v2.sh / p36 系列）用的是 PROFILE，这里保留兼容：
# 两者任一为 1 都开。
V41_PROFILE=${V41_PROFILE:-${PROFILE:-0}}
PROFILE_DIR=${PROFILE_DIR:-}
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
# [DRAFT_GRAPH] 把 DSpark draft 也放进 ACLGraph。
#
# 默认 **1**（发布口径）。历史上它是 0，原因是当时有一个**静默失效**：draft 图
# 捕获时如果没有 `DSPARK_GRAPH_CAPTURE_METADATA=1`，`AscendDSAImpl.forward()` 会走
# "no metadata" 回退分支，于是**重放的图里根本没有 attention** —— 表现为
# `A 恒 1.0`，而 ms/step 看着正常（`reports/draft-graph-negative-control.md` 的负控）。
# 所以这里把两个开关**绑在一起**设，并且在起服后做一次 A 校验（见下方 DRAFT-GUARD）。
#
# 收益：A3 上约 −0.41 ms/step；A2 上 draft 每轮 ~24 ms，是主矛盾，杠杆大得多
# （EXPECTED_PERF.md §A2GAP 把它列为 A2 唯一的大杠杆）。
# 关掉：DRAFT_GRAPH=0（此时回到 SPEC_EAGER=1 的 eager draft）。
#
# ★★ 2026-09-20 实测：**默认必须是 0**。曾按要求把默认改成 1，并补齐了
#    DSPARK_GRAPH_CAPTURE_METADATA=1、验证了 draft 版文件已装、起服命令行也确实是
#    `enforce_eager:false`（开关全部到位），但**效果是坏的**：
#
#      配置              A(接受长度)   单流 tok/s   ms/step
#      DRAFT_GRAPH=0      2.7–3.0       90–111      27–30
#      DRAFT_GRAPH=1      1.06–1.08     43.0        25.1
#
#    A≈1.0 说明 draft 完全没产出 —— 正是 `reports/draft-graph-negative-control.md`
#    记录的那种**静默失效**。而 ms/step 反而"更好看"，因为每步只出 1.08 个 token
#    而不是 2.85 个 ⇒ **真实吞吐慢 2.2×**。
#    ⇒ 只看 ms/step 会得出完全相反的结论；判据必须是 (A, tok/s) 这一对。
#    在查清根因之前维持 0。想实验：DRAFT_GRAPH=1，但**必须**用
#    `bash tools/draft_graph_guard.sh` 确认 A ≥ 1.3，否则不要用。
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
# [MC2-PARAM] 这几个原本在 inner.sh 里硬编码为 0；改成可参数化，
# 以便复现 2026-09-16 的"已知good"配置（FUSED_MC2=1 MULTISTREAM=0 SP_TOKENS=7）。
FUSED_MC2=${FUSED_MC2:-0}
MC2=${MC2:-0}
MC2_HIER=${MC2_HIER:-0}
REDUCE_SAMPLE=${REDUCE_SAMPLE:-0}
DSA_OVERLAP=${DSA_OVERLAP:-1}
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
      # [DEVICE-INDEX] aclrtHostRegister 拒绝只读 VMA（ret=507899），所以
      # Engram 表所在目录必须可写挂载 —— 代码只读它，但驱动要在上面取引用。
      # 只放开这两个目录，其余模型目录保持 :ro。
      case "${_d##*/}" in
        engram_int8|engram-int8)
          if _engram_need_rw; then
            MODEL_MOUNTS+=(-v "$_d:$_d:rw")
          else
            MODEL_MOUNTS+=(-v "$_d:$_d:ro")
          fi
          ;;
        *) MODEL_MOUNTS+=(-v "$_d:$_d:ro") ;;
      esac
    done
    say "模型挂载（auto 模式，${#_mdirs[@]} 个目录，含软链链条；ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX）"
    for _d in "${_mdirs[@]}"; do
      case "${_d##*/}" in
        engram_int8|engram-int8)
          if _engram_need_rw; then say "   -v $_d:$_d:rw"; else say "   -v $_d:$_d:ro"; fi ;;
        *) say "   -v $_d:$_d:ro" ;;
      esac
    done
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
  _njson=$(ls -1 "$_skc/static_kernel_cache" 2>/dev/null | grep -c '\.json$')
  echo "[serve_a2] skcache: static_kernel_cache/ 命中（${_njson} 个缓存文件）"
  # [SKCACHE-PROOF] 只看"static_kernel_cache/ 目录在不在"会误报：旧版本把产物写在
  # 容器里没挂出来，宿主目录照样能有个空壳。这里用**缓存清单的大小**当判据
  # —— 清单是 `hash -> /workspace/.../*.run` 的映射，一份真实缓存至少几 KB；
  # 空壳是 0 或几十字节。
  # 不用 `du`：产物目录是 root 私有的，非 root 跑 du 会刷一屏 permission denied
  # （实测），而 `stat` 只看文件本身，不需要 root。
  _json=$(ls -1 "$_skc/static_kernel_cache"/*.json 2>/dev/null | head -1)
  _sz=$(stat -c %s "$_json" 2>/dev/null || echo 0)
  if [ "${_sz:-0}" -lt 512 ]; then
    echo "[serve_a2] ⚠️  skcache 清单只有 ${_sz} 字节（$_json）—— 不像是有效缓存，"
    echo "[serve_a2]    本次很可能仍要冷编译。检查挂载点是否与容器 cwd 一致（应为 /workspace）。"
  else
    echo "[serve_a2] skcache: 清单 $(basename "$_json") = ${_sz} 字节（有效）"
  fi
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
  # [PGO-AUTODETECT] 旧版行为：TARGET_PATH.txt 缺失 → 只打一行 WARNING 就静默降级，
  # 很容易被漏掉（这正是用户遇到的情况）。本包改为**自动探测并落盘**。
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
# [SCRIPTS-MOUNT] 把本包的 scripts/ 只读挂进容器。
# 原实现依赖镜像里烘焙好的 /opt/dsv41/scripts/serve_v2.sh ——
# 换用未打我们补丁的基础镜像（用于 A/B 对照）时那个路径不存在，会直接起不来。
MOUNTS+=(-v "$PKG/scripts:/opt/dsv41/scripts:ro")
if [ "$PATCH_MODE" = "mount" ]; then
  F=$PKG/patches/files
  # [ADMISSION-GATE] vLLM core 的 admission gate 是**补丁**（不是整文件），
  # Dockerfile 只在 build_image.sh 烘焙路径里 `git apply`。mount 模式（A3 默认：
  # 官方镜像 + 挂补丁）**不经过那一步** ⇒ 必须把 patch 也挂进来，起容器后现场打。
  # 不补的话 `VLLM_ADMISSION_GATE=1` 就是一个没人消费的 env，保护**静默失效**。
  MOUNTS+=(-v "$PKG/patches/admission_gate.patch:/opt/dsv41/admission_gate.patch:ro")
  MOUNTS+=(-v "$F/engram_hbm.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hbm.py:rw")
  MOUNTS+=(-v "$F/engram_hash.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py:rw")
  # [DEVICE-INDEX] host-mapped 表 + 设备侧哈希（新模块；关掉开关时不会被 import）
  if [ -f "$F/engram_device_index.py" ]; then
    MOUNTS+=(-v "$F/engram_device_index.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_device_index.py:ro")
  elif _engram_need_rw; then
    die "ENGRAM_DEVICE_INDEX=1 但缺 patches/files/engram_device_index.py"
  fi
  if [ -f "$F/engram_graph.py" ]; then
    MOUNTS+=(-v "$F/engram_graph.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_graph.py:ro")
  elif _engram_need_rw; then
    die "ENGRAM_DEVICE_INDEX=1 但缺 patches/files/engram_graph.py"
  fi
  MOUNTS+=(-v "$F/engram_jit_kernel.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_jit_kernel.py:ro")
  MOUNTS+=(-v "$F/engram_plan_kernel.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_plan_kernel.py:ro")
  MOUNTS+=(-v "$F/engram_gate.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_gate.py:ro")
  MOUNTS+=(-v "$F/model.py:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:rw")
  # [DMQ-GUARD-REMOVED] 曾在这里挂 patches/files/model_runner_v1.py（device_metadata
  # 的 submit/release 自愈护栏）。**已撤销**：该护栏在正常运行中也会误触发，
  # 提前释放 device metadata ⇒ 64 并发实测 57/64 + 服务挂（ScatterElements 0x91 →
  # ERR00100 → HCCL watchdog），撤销后同一扫描 64/64 全过。
  # 不要恢复这个挂载 —— 整文件覆盖 model_runner_v1.py 的风险远大于收益。
  MOUNTS+=(-v "$F/ascend_forward_context.py:/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py:ro")
  MOUNTS+=(-v "$F/rope_dsv4.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/rope_dsv4.py:ro")
  # [V41-SLOT-MAP-FUSED] block_table.py：12 次 slot-mapping 启动 → 1 次。
  # 由 env `V41_SLOT_MAP_FUSED` 门控（默认 0/关 = 与 stock 完全一致）。
  # 缺文件不致命（回落 stock），但要**响亮地**告诉用户门控会静默失效。
  if [ -f "$F/block_table.py" ]; then
    MOUNTS+=(-v "$F/block_table.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/block_table.py:rw")
  elif [ "${V41_SLOT_MAP_FUSED:-0}" != "0" ]; then
    echo "[serve_a2] WARNING: V41_SLOT_MAP_FUSED=${V41_SLOT_MAP_FUSED} 但缺 patches/files/block_table.py" >&2
    echo "[serve_a2]           → 门控会静默失效（跑的是 stock 逐组路径）" >&2
  fi
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
# ---------- [PROBE] 稀疏状态插针（事后取证；独立于 PATCH_MODE） ----------
# PROBE=1 时用只读挂载覆盖 dsa_v41.py 并注入 sparse_capture.py。
# L1 元数据常开（~200 B/step/层）；L2 张量快照由 <probe_capture>/ENABLE 开关文件控制。
#
# **默认 0**：插针需要一份"在 dsa_v41.py 里插了 14 行调用"的派生文件
# （`reports/probe/dsa_v41.probe.py`），那是上游代码的 fork，本包不附带以免漂移。
# 需要时见 `reports/probe/README.md` 的生成方法。
PROBE=${PROBE:-0}
if [ "$PROBE" = "1" ]; then
  PB=$PKG/reports/probe
  if [ -f "$PB/dsa_v41.probe.py" ] && [ -f "$PB/sparse_capture.py" ]; then
    MOUNTS+=(-v "$PB/sparse_capture.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/sparse_capture.py:ro")
    MOUNTS+=(-v "$PB/dsa_v41.probe.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:ro")
    mkdir -p "$PKG/probe_capture"
    MOUNTS+=(-v "$PKG/probe_capture:/opt/dsv41/probe")
    echo "[serve_a2] PROBE=1：稀疏状态插针已挂载（输出 $PKG/probe_capture）"
  else
    echo "[serve_a2] WARNING: PROBE=1 但缺 $PB/{dsa_v41.probe.py,sparse_capture.py} → 跳过插针"
  fi
fi

# ---------- [MEM-GUARD] 大 BAT × 大 MAX_SEQS 会 OOM ----------
# 实测（8×910C, 60.96 GiB/卡）：`BAT_TOKENS=8192` 时 peak activation 从
# 0.79 GiB 涨到 3.21 GiB（+2.4 GiB）。若同时把 MAX_SEQS 开到 64（capture
# 桶要覆盖到 384、graph memory 1.15 GiB），`GPU_UTIL=0.94` 会把显存吃干，
# ACL graph **重放时会 OOM**：
#     torch.OutOfMemoryError: NPUGraph.cpp:281
#     Resource_Error_Insufficient_Device_Memory(EL0019)
#     Failed to allocate 2097152 bytes ... halStreamTaskFill failed
#
# 默认 GPU_UTIL 已从 0.94 降到 0.92（见 §[MEM-HEADROOM]），余量从 6.11 涨到
# 7.36 GiB，**上述组合在 0.92 下未复测**；仍按"未验证"处理，所以提示保留。
# 这里只做**提示**（不擅自改用户配置），避免静默行为变化。
if [ "$BAT_TOKENS" -ge 8192 ] && [ "$MAX_SEQS" -ge 64 ]; then
  echo "[serve_a2] WARNING: BAT_TOKENS=$BAT_TOKENS 且 MAX_SEQS=$MAX_SEQS（GPU_UTIL=$GPU_UTIL）。"
  echo "                    该组合在 GPU_UTIL=0.94 下实测 OOM（ACL graph 重放失败）；"
  echo "                    默认的 0.92 尚未复测。建议二选一："
  echo "                      MAX_SEQS=32  （发布默认，已验证）"
  echo "                      GPU_UTIL=0.90（余量更大，prefill 速度不受影响）"
fi

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
  echo "[a2-dry] PROFILE=$V41_PROFILE ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX ENGRAM_DEVICE_FALLBACK=$ENGRAM_DEVICE_FALLBACK"
  echo "[a2-dry] DROPCACHE=$DROPCACHE（起服前清 page cache；0 关闭）"
  echo "[a2-dry] MOE_ZERO=$MOE_ZERO MOE_NF=$MOE_NF DRAFT_GRAPH=$DRAFT_GRAPH PYTHON_PGO=$PYTHON_PGO pgo_target=${PGO_LIB:-none} LOAD_FORMAT=${LOAD_FORMAT:-<real>} CAND_MODE=$CAND_MODE PATCH_MODE=$PATCH_MODE"
  echo "[a2-dry] CPUSET=$CPUSET${CPUSET_SRC:+ ($CPUSET_SRC)} MEMS=$MEMS${MEMS_SRC:+ ($MEMS_SRC)} STATIC_KERNEL=$STATIC_KERNEL NPUGRAPH_EX=$NPUGRAPH_EX MULTISTREAM=$MULTISTREAM HCCL_DET=${HCCL_DET:-none}"
  echo "[a2-dry] MOUNTS(${#MOUNTS[@]}): ${MOUNTS[*]:-<none>}"
  echo "[a2-dry] MODEL_MOUNT_MODE=$MODEL_MOUNT_MODE MODEL_MOUNTS(${#MODEL_MOUNTS[@]}):"
  for _m in "${MODEL_MOUNTS[@]}"; do echo "[a2-dry]    $_m"; done
  exit 0
fi

say "起容器 $NAME（image=$IMAGE port=$PORT devs='$DEVS' util=$GPU_UTIL pgo=$PYTHON_PGO patch_mode=$PATCH_MODE mseqs=$MAX_SEQS prefix=$PREFIX）"

# ---------- [DROPCACHE] 起服前清 page cache（默认开） ----------
# 为什么放在这里：起服需要一段**连续**的宿主内存（权重 206 GB 的 page cache +
# 8 个 worker 的 torch/静态内核缓冲 + numba/torch.compile 缓存），而 page cache
# 是可回收的常驻内存。实测一次 `echo 1 > drop_caches` 能释放 564 GiB
# （Cached 811 -> 247 GiB，MemFree 454 -> 1021 GiB），代价是下一次读文件要重新
# 走盘 —— 起服本来就要读一遍 206 GB 的表，所以这个代价是划算的。
#
# 注意它**不能**清理 tmpfs：`/tmp` 与 `/dev/shm` 里的东西算 Shmem（不可回收），
# drop_caches 对它们完全无效（实测 Shmem 246.9 GiB 一动不动）。所以若宿主内存
# 被 tmpfs 占满，需要单独清理那些目录，这个开关帮不上忙。
#
# 影响面：drop_caches 是**整机**的，会连带清掉同机其它租户的 page cache
# （它们之后首次读文件会变慢）。所以留了开关：
#   DROPCACHE=0 关闭；DROPCACHE=1（默认）打开。
# 需要 root，没有免密 sudo 时只告警不失败。
if [ "$DROPCACHE" = "1" ]; then
  if [ "$(id -u)" = "0" ]; then
    _sync_then_drop() { sync; echo 1 > /proc/sys/vm/drop_caches; }
  elif sudo -n true 2>/dev/null; then
    _sync_then_drop() { sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'; }
  else
    _sync_then_drop() { return 1; }
  fi
  _before_mb=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
  if _sync_then_drop; then
    sleep 2
    _after_mb=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
    say "page cache 已清理：MemFree ${_before_mb} MiB -> ${_after_mb} MiB (+$((_after_mb - _before_mb)) MiB)"
  else
    say "⚠️  DROPCACHE=$DROPCACHE 但当前用户无 root/免密 sudo，跳过（不影响起服）"
  fi
fi

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
: > "$LOG"
# [MEMLOCK] A2 实测：pin_memory 报 207001（`aclrtMallocHostWithCfg` 失败）而 free -g
# 仍有 667 GiB 空闲 —— 不是容量问题。解法是更新 driver + 解除 memlock 限制。
# 这里默认 `--ulimit memlock=-1`（无限），A3 上无副作用。
# [SKCACHE-PATH] static kernel 产物的挂载点必须与**实际写入路径**一致。
# torch_npu 的 `npugraph_ex/.../_acl_concrete_graph/static_kernel.py` 用的是
# `Path.cwd()`：
#     line 888:  base_dir = Path.cwd().resolve()
#     line 912:  base_output_dir = script_dir / "static_kernel_compile_outputs"
#     line 672:  static_kernel_install 也在 <cwd> 下
# 而容器是 `-w /workspace` 起的（本文件下方 docker run 的 -w）⇒ 产物落在
# **/workspace/static_kernel_compile_outputs**。
# 旧写法只挂 /vllm-workspace/... —— 那一层永远收不到东西，于是：
#   ① 每次重启都冷编译（A3 实测多花 ~5 min，A2 更久）；
#   ② 脚本自己的 "skcache 命中" 检查看的是宿主目录，因此还会**误报命中**；
#   ③ 宿主目录只剩 4 KB 旧空壳（实测），而容器内 /workspace 下积了 182 MB。
# 现在两处都挂：/workspace 是真实位置，/vllm-workspace 兼容 workdir 不同的镜像。
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
  -v "$CACHE/skcache/compile_outputs:/workspace/static_kernel_compile_outputs" \
  -v "$CACHE/skcache/compile_outputs:/vllm-workspace/vllm/static_kernel_compile_outputs" \
  -v "$CACHE/skcache/compile_outputs:/vllm-workspace/vllm-ascend/static_kernel_compile_outputs" \
  -v "$CACHE/skcache/install:/workspace/static_kernel_install" \
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
  -e V41_ENGRAM_DEVICE_INDEX="$ENGRAM_DEVICE_INDEX" \
  -e DSPARK_GRAPH_CAPTURE_METADATA="$([ "$DRAFT_GRAPH" = "1" ] && echo 1 || echo 0)" \
  -e DSPARK_DRAFT_USE_CUDAGRAPH="${DSPARK_DRAFT_USE_CUDAGRAPH:-1}" \
  -e DSPARK_DRAFT_METADATA_MODE="${DSPARK_DRAFT_METADATA_MODE:-sync}" \
  -e DSPARK_GRAPH_SHADOW_EAGER="${DSPARK_GRAPH_SHADOW_EAGER:-0}" \
  -e DSPARK_GRAPH_SHADOW_STEPS="${DSPARK_GRAPH_SHADOW_STEPS:-2}" \
  -e DSPARK_GRAPH_DEVICE_METADATA="${DSPARK_GRAPH_DEVICE_METADATA:-0}" \
  -e DSPARK_GRAPH_DEBUG="${DSPARK_GRAPH_DEBUG:-0}" \
  -e DSPARK_GRAPH_PTR_PROBE="${DSPARK_GRAPH_PTR_PROBE:-0}" \
  -e DSPARK_DSA_PROBE="${DSPARK_DSA_PROBE:-0}" \
  -e DSPARK_DSA_PROBE_CAPTURE="${DSPARK_DSA_PROBE_CAPTURE:-0}" \
  -e DSPARK_DSA_WRITE_PROBE="${DSPARK_DSA_WRITE_PROBE:-0}" \
  -e DSPARK_DSA_WRITE_PROBE_STEPS="${DSPARK_DSA_WRITE_PROBE_STEPS:-12}" \
  -e DSPARK_CAPTURE_PAD_SLOTS="${DSPARK_CAPTURE_PAD_SLOTS:-0}" \
  -e DSPARK_DRAFT_SERIAL="${DSPARK_DRAFT_SERIAL:-0}" \
  -e DSPARK_ROW_DUMP="${DSPARK_ROW_DUMP:-0}" \
  -e DSPARK_CAPTURE_MAXSEQLEN="${DSPARK_CAPTURE_MAXSEQLEN:-0}" \
  -e DSPARK_CAPTURE_DISPATCH="${DSPARK_CAPTURE_DISPATCH:-0}" \
  -e DSPARK_NO_TOPK_SHARE="${DSPARK_NO_TOPK_SHARE:-0}" \
  -e DSPARK_DRAFT_NO_ATTN="${DSPARK_DRAFT_NO_ATTN:-0}" \
  -e DSPARK_DRAFT_SYNC_AFTER="${DSPARK_DRAFT_SYNC_AFTER:-0}" \
  -e DSPARK_DRAFT_SYNC_BEFORE="${DSPARK_DRAFT_SYNC_BEFORE:-0}" \
  -e DSPARK_RT_FLAGS="${DSPARK_RT_FLAGS:-0}" \
  -e DSPARK_HOIST_CONTEXT_KV="${DSPARK_HOIST_CONTEXT_KV:-0}" \
  -e DSPARK_TOKEN_DUMP="${DSPARK_TOKEN_DUMP:-0}" \
  -e DSPARK_TOKEN_DUMP_STEPS="${DSPARK_TOKEN_DUMP_STEPS:-12}" \
  -e DSPARK_STEP_PROBE="${DSPARK_STEP_PROBE:-0}" \
  -e DSPARK_STEP_PROBE_STEPS="${DSPARK_STEP_PROBE_STEPS:-40}" \
  -e DSPARK_DSA_PROBE_STEPS="${DSPARK_DSA_PROBE_STEPS:-60}" \
  -e DSPARK_GRAPH_PTR_PROBE_STEPS="${DSPARK_GRAPH_PTR_PROBE_STEPS:-5}" \
  -e V41_ENGRAM_DEVICE_FALLBACK="$ENGRAM_DEVICE_FALLBACK" \
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

# ---------- [ADMISSION-GATE] mount 模式下现场打 vLLM core 补丁 ----------
if [ "$PATCH_MODE" = "mount" ]; then
  say "[ADMISSION-GATE] mount 模式：在容器内现场应用 admission_gate.patch"
  _gate=$($DOCKER exec "$NAME" bash -lc '
    P=/opt/dsv41/admission_gate.patch
    [ -f "$P" ] || { echo MISSING_PATCH; exit 0; }
    cd /vllm-workspace/vllm 2>/dev/null || { echo MISSING_VLLM_ROOT; exit 0; }
    if git apply --check "$P" 2>/dev/null; then
      git apply "$P" 2>/dev/null && echo APPLIED || echo FAILED
    elif git apply --reverse --check "$P" 2>/dev/null; then
      echo ALREADY
    else
      echo FAILED
    fi' 2>/dev/null | tail -1)
  case "${_gate:-}" in
    APPLIED) say "[ADMISSION-GATE] 已应用 ✓" ;;
    ALREADY) say "[ADMISSION-GATE] 基础镜像里已包含 ✓" ;;
    *)       echo "[serve_a2] WARNING: admission gate 未能应用（${_gate:-unknown}）—— 基础镜像的 vLLM 版本可能不同。" >&2
             echo "[serve_a2]         服务仍能起，但 prefill 饿死 decode 的保护不生效；请把这条日志当交付前必须解释的差异。" >&2 ;;
  esac
  # 效果断言：不看有没有打，看 live tree 里有没有
  _gh=$($DOCKER exec "$NAME" bash -lc 'grep -c admission_gate /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py 2>/dev/null || true')
  if [ "${_gh:-0}" -ge 1 ]; then
    say "[ADMISSION-GATE] live tree 命中 ${_gh} 处 ✓"
  else
    echo "[serve_a2] WARNING: live tree 里找不到 admission gate ⇒ 该补丁未生效" >&2
  fi
fi

if [ "$DRAFT_GRAPH" = "1" ]; then
  # draft 版三个整文件（含 0002/0004/0005/0006 + F3）必须真的**装到实际位置**，
  # 否则 DSPARK_GRAPH_CAPTURE_METADATA=1 设了也没人消费 —— 就是那个静默失效。
  #
  # 两条路径：
  #   PATCH_MODE=mount（A3 默认）—— 上面的 MOUNTS 已经把 draft 版挂到目标路径
  #   PATCH_MODE=baked（A2 默认）—— 镜像里只是把 draft 文件放在
  #       /opt/dsv41/patches/draft/，**没有人把它拷到 live tree**。所以这里显式装。
  say "DRAFT_GRAPH=1：安装 draft 版文件 + 打开图捕获元数据（PATCH_MODE=$PATCH_MODE）"
  if [ "$PATCH_MODE" != "mount" ]; then
    if [ -f "$PKG/tools/enable_draft_graph.sh" ]; then
      $DOCKER exec "$NAME" bash -lc "bash /opt/dsv41/tools/enable_draft_graph.sh on" >/dev/null 2>&1 || true
    fi
    # 兜底：直接从镜像内已 COPY 的 draft 目录装（不依赖 tools/ 是否被挂进去）
    $DOCKER exec "$NAME" bash -lc '
      A=/vllm-workspace/vllm-ascend/vllm_ascend; D=/opt/dsv41/patches/draft
      for pair in "dsa_v1.py:$A/attention/dsa_v1.py" \
                  "dspark_proposer.py:$A/spec_decode/dspark_proposer.py" \
                  "llm_base_proposer.py:$A/spec_decode/llm_base_proposer.py"; do
        src=${pair%%:*}; tgt=${pair##*:}
        [ -f "$D/$src" ] || { echo "MISSING $D/$src"; exit 1; }
        cp -f "$D/$src" "$tgt" && echo "installed $src"
      done' || die "DRAFT_GRAPH=1 但 draft 版文件安装失败（见上）"
  fi
  $DOCKER exec "$NAME" bash -lc "ls -la /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py" >/dev/null
  # 断言：装好的文件里必须真的有图捕获的实现（不是 stock 版）
  _has=$($DOCKER exec "$NAME" bash -lc 'grep -c "DSPARK_GRAPH_CAPTURE_METADATA" /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py || true')
  if [ "${_has:-0}" -lt 1 ]; then
    die "DRAFT_GRAPH=1 但 dspark_proposer.py 里找不到 DSPARK_GRAPH_CAPTURE_METADATA —— 
      实际装的是 stock 版，draft 图会静默无 attention。检查 patches/files/draft/ 是否随包。"
  fi
  say "DRAFT-GUARD: dspark_proposer.py 含图捕获实现（命中 $_has 处）✓"
  _cap=$($DOCKER exec "$NAME" bash -lc 'printf %s "${DSPARK_GRAPH_CAPTURE_METADATA:-unset}"')
  if [ "$_cap" != "1" ]; then
    die "DRAFT_GRAPH=1 但容器内 DSPARK_GRAPH_CAPTURE_METADATA=$_cap（应为 1）——
      缺它 draft 图会静默无 attention（A 恒 1.0，ms 却看着正常）。这是构建/挂载问题，不是运行时问题。"
  fi
  say "DRAFT-GUARD: 容器内 DSPARK_GRAPH_CAPTURE_METADATA=$_cap ✓"
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
  echo "[serve_a2] PROFILE=$V41_PROFILE（1 => /start_profile 与 /stop_profile 可用，落到 $OUT/prof）"
  echo "[serve_a2] ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX ENGRAM_DEVICE_FALLBACK=$ENGRAM_DEVICE_FALLBACK"
  echo "[serve_a2] PATCH_MODE=$PATCH_MODE ADMISSION_GATE=${_gate:-n/a}(live_hits=${_gh:-0})"
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
export MULTISTREAM=$MULTISTREAM DSA_OVERLAP=$DSA_OVERLAP FUSED_MC2=$FUSED_MC2 MC2=$MC2 MC2_HIER=$MC2_HIER REDUCE_SAMPLE=$REDUCE_SAMPLE
export LOADER_MT=1 LAZY=1
export V41_KV_TIER=off
export V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner
export CAPTURE_SIZES="$CAPTURE_SIZES"
export ASCEND_MAX_OP_CACHE_SIZE=-1
# [PROFILE] 透传 profiler 开关；PROFILE_DIR 指向本次 run 的结果目录（宿主可见），
# 这样 /stop_profile 一落盘就能直接分析，不用再 docker cp。
export PROFILE=$V41_PROFILE
export PROFILE_DIR=/opt/dsv41/results/$RUN_ID/prof
# [OPS-SWITCHES] 三个排障开关，**默认全关**（发布口径）。
# 排查长上下文/精度问题时把它们打开很有用：
#   VLLM_SERVER_DEV_MODE=1  → 额外挂出 12 个运维端点（/reset_prefix_cache /pause
#                             /resume /sleep /wake_up /collective_rpc /server_info …），
#                             vLLM 自己会打一条 "Development endpoints are enabled!" 安全告警。
#   LOG_REQUESTS=1          → 把请求级 I/O 写进 serve.log（长度上限 MAX_LOG_LEN）。
#   PROBE=1                 → 稀疏状态插针，见上。
export VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-0}
export V41_PROBE_DIR=${V41_PROBE_DIR:-/opt/dsv41/probe}
export LOG_REQUESTS=${LOG_REQUESTS:-0}
export MAX_LOG_LEN=${MAX_LOG_LEN:-4096}

if [ "$TOOL_CALLING" = "1" ]; then
  export EXTRA='--tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --tool-call-parser=deepseek_v41 --enable-auto-tool-choice --default-chat-template-kwargs={"enable_thinking":false}'
else
  export EXTRA='--tokenizer-mode=deepseek_v4 --default-chat-template-kwargs={"enable_thinking":false}'
fi
if [ -n "$LOAD_FORMAT" ]; then
  export V41_ENGRAM_WITH_DUMMY=1 V41_DUMMY_WO_A_FIX=1
fi
if [ "$DRAFT_GRAPH" = "1" ]; then
  export DSPARK_DRAFT_METADATA_MODE=sync
  # ★ 必须同时设这个，否则 draft 图静默无 attention（见上面 DRAFT_GRAPH 注释）。
  export DSPARK_GRAPH_CAPTURE_METADATA=1
fi
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
