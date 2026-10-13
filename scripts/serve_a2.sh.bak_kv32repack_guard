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
#   * **默认关** DRAFT_GRAPH=0（draft 入图；需显式 `DRAFT_GRAPH=1`）
#     ⚠️ 这里原先写的是"默认开"，与代码 `DRAFT_GRAPH=${DRAFT_GRAPH:-0}` 自相矛盾。
#     而"默认关"是**刻意决定**：默认开会让用户拿到 `A≈1.07` 的坏配置且**无任何报错**
#     （见 CHANGELOG §6）。文案与代码不一致会让人以为"我没设应该是开着的"，
#     正好踩中那个静默失效。
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
PKG="$(cd "$HERE/.." && pwd)"

MODEL=${MODEL:-}
# ★★★ 2026-09-22 21:4x：默认镜像升到 **v9** —— v9 起烘焙了 ENGRAM×卸载 的 P0 修复
#   （`patches/files/{engram_hash.py,engram_jit_kernel.py}`：镜像缺页从 `KeyError`
#    降级为 barrier，见 `a2/logs/075`）。
#   ⇒ 若沿用 v8（不含修复），`ENGRAM=1 + 卸载` 会在 replay 轮 `KeyError(2486)` 引擎死。
#   ★ 这条"默认值必须跟着修复走"的纪律，正是本日反复出现的失败模式（`065` §3/`074`：
#     开关/文件送不到 = 静默降级）。`a2/scripts/serve_a2_offload.sh` 里还有一道
#     **起服前指纹门**会在 5 秒内把"镜像里没有修复"这件事抓出来并拒绝起服。
IMAGE=${IMAGE:-dsv41-a2:v9}
NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8100}
# [SERVED_NAME] `--served-model-name`（API 请求 body 里的 `"model"` 字段）。
# ⚠️ 2026-09-20 修：此前这里是**硬编码** `SERVED_NAME=deepseek-v41`（见下方 inner.sh），
#    用户传 `SERVED_NAME=xxx` 会被静默覆盖 —— 多实例部署/避让命名冲突时无法自定义。
#    现在与 `PORT` 同口径：env 优先，默认仍是 `deepseek-v41`（保持向后兼容）。
#    注意这个值同时决定 **API 请求里必须填的 model 名**；测试脚本要用同一个名字
#    （`tools/*.sh` 与 `tests/*.py` 会读 `SERVED_NAME`，默认同样回落 deepseek-v41）。
SERVED_NAME=${SERVED_NAME:-deepseek-v41}
TP=${TP:-8}
DP=${DP:-1}
case "$DP" in
  ''|*[!0-9]*|0) echo "[serve_a2][FAIL] DP 必须是正整数，得到 '$DP'" >&2; exit 2 ;;
esac
DEVS=${DEVS:-"0 1 2 3 4 5 6 7"}
# [SCRIPT-VER] 起服时打印脚本版本 + 指纹。为什么需要：镜像里也有一份烘焙的
# `/opt/dsv41/scripts/serve_a2.sh`（Dockerfile COPY），而镜像可能是**旧脚本**构建的
# —— 用户报障时先看这一行就能判断"修复到底有没有生效"（v8 之后 A2 报障的第一件事）。
# 纪律：README 要求**用发布包里的** scripts/serve_a2.sh；镜像里那份只作兜底。
# ⚠️ 临时诊断版（相对已发布的 12caf9b）：
#   * MODEL_MOUNT_ALL_RW=1 —— 模型树**全部**挂 :rw（排查用，默认仍是最小放开）
#   * 透传 DSPARK_DISPATCH_DIAG_STEPS / CAPTURE_NCTX_FIX / SWA_INDICES_RESIDENT
#     （draft 四件套的实验开关，便于 A/B 关掉对照）
# 正式合并回发布包时会重新定版本号。
SERVE_A2_VER="v8-engram-rw-mount-20260920+allrw"
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
GRAPH=${GRAPH:-1}
EAGER=${EAGER:-0}
STATIC_KERNEL=${STATIC_KERNEL:-1}
NPUGRAPH_EX=${NPUGRAPH_EX:-1}
SP_TOKENS=${SP_TOKENS:-5}
# [DYNAMIC-SPEC] 按"当时并发数"切推测解码的 K。格式 `a,b,K;c,d,K`
# （分号分隔的 range_start,range_end,K，闭区间，按**请求数**查表）。
# 例：`1,1,7;2,8,0` = 单请求走 K=7、≥2 并发直接关推测。
# 空 = 固定 K（历史行为，逐字节相同）。需要 GRAPH=1 EAGER=0 且
# V41_CED_GRAPH_PROMPT_TAIL_EAGER=1（见下方 DYNAMIC-SPEC 段）。
SP_SCHEDULE=${SP_SCHEDULE:-}
SPEC=${SPEC:-1}
ENGRAM=${ENGRAM:-1}
VISION=${VISION:-1}
# [MM-LIMIT] 一个请求允许几张图（编译进 vLLM 的 Rust 校验，改它必须重启）。
#   模型与处理器本身支持多图（deepseek_v41 的 vl_model._process_image_input
#   是逐图循环，每张图有自己的 vit_grid/llm_grid/占位区间），限制纯粹来自
#   这个启动参数。**默认 4**：dsh/Codex 一个回合读两张图是常见形态，设 1 会
#   让整个会话历史永久 400（图片留在历史里，每一轮都超限）。
#   ⚠️ P/D 两个角色必须取同一个值：P 先校验、D 再校验一次，D 更小的话请求
#   会在 D 上被 400。serve_a3_pd.sh 对两个角色都显式导出同一个默认值。
#   ⚠️ 代价：--max-num-seqs 4 时最坏 4×4 张图同时在编，视觉侧要留 HBM 余量；
#   每张图 ≤1024 视觉 token，而 BAT_TOKENS=8192，文本侧一批放得下。
MM_LIMIT_IMAGES=${MM_LIMIT_IMAGES:-4}
# [BIND-HOST] 监听地址。P/D 半边默认只监听回环（角色脚本给 127.0.0.1）；
#   这里给 0.0.0.0 是为了不改变单实例/非 PD 部署的既有行为。
HOST=${HOST:-0.0.0.0}
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
IDS64_HOIST=${IDS64_HOIST:-0}  # [IDS64-HOIST] ★ 2026-10-05 实测：本配置下**无可优化对象**
#   —— 它 targeting 的 `input_ids.to(int64)`（fused_topk_router.py:164）只在
#   `tid2eid`/`bias_vl` 非空（视觉/哈希路由）时执行；纯文本交付实例两者都是 None。
#   逐算子族计数两臂逐项相同（Cast 253.1 vs 252.4/步）⇒ 保持默认 0，不必再验。
PAD_SKIP=${PAD_SKIP:-1}        # [ENGRAM-PAD-SKIP] ★ 已转默认：只清零会被读到的 padded 行
ENGRAM_WKV_TP=${ENGRAM_WKV_TP:-1}  # [ENGRAM-WKV-TP] ★ 已转默认：engram gate 的 wkv 按输出维分片
#   机制实测：全量 matmul 0.709ms/步 → 分片 0.071 + all_gather ~0.07 ⇒ 净省 ~0.57ms/步（2.2%）。
#   受控 A/B（同 bench 参数、各 1 run）：n=6/12/24/48 全部更快 −0.40~−0.66 ms（p10 同向）；144K 11/11 PASS。
#   实测（零噪声判据）：`ZerosLike "8192,6144"` 1.86→0 /步 ⇒ 省 ~58 µs/步（0.22%）；
#   模型只读 lookups[:n]，而清零范围 ≥ n ⇒ 可读区仍被清零；144K 验收 11/11 PASS。
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
# ★★★ 2026-09-28 **修一个配 `GATE_CHUNK` 就必崩的地雷**：
#   分块 gate（`GATE_CHUNK > 0`）把 token 维**静态 pad 到 `GATE_MAX_TOKENS`**，
#   而 `_engram_gate_chunked` 末尾是 `padded[:n]` + `torch.where(mask[n], padded[:n])`：
#       n > MAX  ⇒ 切片只有 MAX 行、与 [n,...] 的 mask 广播不上 ⇒ **直接抛错**
#                 （实际报错形态见 `upstream/logs/41`：「n=8192 那一行里 MAX≤4096
#                  的臂全部 RAISES」）。
#   而本脚本此前默认 **2048**，`BAT_TOKENS` 默认 **8192** ⇒
#   **只要有人设 GATE_CHUNK=512 而忘了同时给 GATE_MAX_TOKENS，第一次长 prefill 就崩**
#   （典型 prompt 8K ⇒ n≈8176 > 2048）。这正是本仓反复栽的"默认值两处不一致"。
#   ⇒ 上界按 **本图最大 token 数**（= `BAT_TOKENS`，prefill chunk 的上限）派生。
#   ★ 代价（必须知道，见 `upstream/logs/41` 的实测曲线）：分块路径的时间只跟 ceiling
#     走、不跟 n 走（每 512 行 ≈ +0.7 ms）⇒ ceiling=8192 时**每次调用 ≈ 12.9 ms**，
#     而 ceiling=2048 时 ≈ 2.8 ms。decode 步的 n 很小（≤ MAX_SEQS×query_len），
#     但 ceiling 一样要垫满 ⇒ **开 GATE_CHUNK 会按 ceiling 给每一步加固定开销**。
#     缺 HBM 才开它；不缺就用默认 0（stock，不 pad，无此项开销）。
#     stock 的代价是峰值激活大方差（BAT=8192 时 ≈2.5–3.2 GiB，实测会撞 207001）。
_GATE_MAX_DEFAULT=$(( BAT_TOKENS > 512 ? BAT_TOKENS : 512 ))
GATE_MAX_TOKENS=${GATE_MAX_TOKENS:-$_GATE_MAX_DEFAULT}
# 一致性守卫：显式给的 GATE_MAX_TOKENS 若小于 BAT_TOKENS，起服前响亮拒绝
#   （否则错误要等到第一次长 prefill 才以"广播失败"的形式出现，排查成本高得多）
if [ "$GATE_CHUNK" -gt 0 ] && [ "$GATE_MAX_TOKENS" -lt "$BAT_TOKENS" ]; then
  echo "⛔ GATE_CHUNK=$GATE_CHUNK 但 GATE_MAX_TOKENS=$GATE_MAX_TOKENS < BAT_TOKENS=$BAT_TOKENS" >&2
  echo "   分块 gate 把 token 维静态 pad 到 GATE_MAX_TOKENS；n > MAX 会**直接抛错**。" >&2
  echo "   ⇒ 要么去掉 GATE_MAX_TOKENS 用派生默认（$_GATE_MAX_DEFAULT），" >&2
  echo "     要么显式给 GATE_MAX_TOKENS=$BAT_TOKENS（时间开销见本行上方注释）。" >&2
  exit 64
fi
# [SELFTEST-HOOK] 只解析并打印 GATE 三元组后退出 —— 供 `tools/selfcheck_pkg.sh` 的
#   9m 项做**正控/负控**（生产环境不会设这个变量）。
#   为什么要这个钩子：守卫本身是纯逻辑，但它在脚本第 250~275 行、后面还有几百行起服
#   动作 ⇒ 想单测它只能"抽片段 eval"（脆弱）或"跑整个脚本"（要 docker/镜像，且会被
#   无关失败干扰）。有了钩子，判据就能**绑在守卫自身的返回值**上，不绑抽片段的写法。
if [ "${V41_GATE_GUARD_CHECK_ONLY:-0}" = "1" ]; then
  echo "GATE_RESOLVED chunk=$GATE_CHUNK max_tokens=$GATE_MAX_TOKENS bat=$BAT_TOKENS"
  exit 0
fi
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
# [GATE-MULTIPREFILL 2026-10-04] admission gate 的一个 prefill-only 步里最多放几个
# prefill 请求。1 = 历史行为（每步只放一个）；8 = 实测 N=8 交付吞吐 +11.0%、N=4 +6.6%、
# N=1 不变、TTFT −29%，且 144K/1M 四针 + 混布 [C] + 多轮全部通过。
# 硬不变量未变：prefill-only 步仍然不含任何 decode token。
# 回退：V41_GATE_MAX_PREFILL=1。CED-PD/P-D 分离场景**未做端到端验证**，如需保守可在
# deploy/a3-ced-pd 的 launcher 里显式设 1。
V41_GATE_MAX_PREFILL=${V41_GATE_MAX_PREFILL:-8}
MULTISTREAM=${MULTISTREAM:-1}
# [MC2-PARAM] 这几个原本在 inner.sh 里硬编码为 0；改成可参数化，
# 以便复现 2026-09-16 的"已知good"配置（FUSED_MC2=1 MULTISTREAM=0 SP_TOKENS=7）。
FUSED_MC2=${FUSED_MC2:-0}
# [EPLB 2026-10-05] 动态专家负载均衡（透传给 serve_v2.sh；与 FORCE_EPLB 互斥）
EPLB_DYNAMIC=${EPLB_DYNAMIC:-0}; EPLB_REDUNDANT=${EPLB_REDUNDANT:-0}
EPLB_STAGE=${EPLB_STAGE:-all}; EPLB_INTERVAL=${EPLB_INTERVAL:-50}; EPLB_HEAT=${EPLB_HEAT:-600}
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
KV_ARGS_EXTRA=${KV_ARGS_EXTRA:-}
OUT=${OUT:-$PKG/results/$RUN_ID}
LOG=${LOG:-$OUT/serve.log}
READY_TIMEOUT=${READY_TIMEOUT:-2100}
WAIT_READY=${WAIT_READY:-1}
# [DRY_RUN] 1 = 只走"开关解析 + MOUNTS 组装"并打印结果，**不碰 docker**。
# 用途：`set -u` 下的变量顺序 bug（`bash -n` 抓不到，只有真正展开变量才暴露）
# 由 tests/multibatch/verify_serve_flags.sh 的 12 组合矩阵调用。
DRY_RUN=${DRY_RUN:-0}

# [SAY-BEFORE-MKDIR] ★ 实测踩过的坑：`say()` 会 `tee -a "$OUT/driver.log"`，
# 而 `OUT` 原先直到 **:664** 才 `mkdir -p`。于是起服时**头 ~100 行日志全部丢失**，
# 终端只留一行红字（issue #2 报告者先发现的）：
#
#     tee: /.../results/<run>/driver.log: No such file or directory
#
# 而 `serve_a3.sh` 结尾是 `exec bash "$HERE/serve_a2.sh"`、**自身不建 OUT**
# ⇒ 官方路径**每次起服必踩**。我们自己的 launcher 里有 `mkdir -p` 才掩盖了它。
#
# 修法：把目录创建提到 `say()` 定义之前。这里**只能建 `$OUT`**（`$CACHE` 那几个
# 目录依赖后面才解析的变量），所以 :664 的 `mkdir -p` 保留不动。
mkdir -p "$OUT" 2>/dev/null || true
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

# ---------- [KV32-POOL-GUARD] 4 GiB 页步长上界（只在会撞它的形态下默认使能） ----------
# 背景（docs/CED-PD-BLOCK-BOUND-20260925.md §5.1.2/§5.1.4）：KV cache 的打包布局
# 里槽位 3 的页步长是 147712 B（layer-20 C1 KV 131072 + INT8 index K 16384
# + FP16 scales 256）。一旦 `num_blocks × 页步长` 越过 2³²，算子按 32 位算出的
# 块地址会回绕到别的块，长上下文请求静默变成"HTTP 200 + 1 token（EOS）"。
#
# 这里放在**公共底层**（serve_a2.sh）而不是某个角色脚本里，因为
# serve_a3_pd.sh / serve_a3_ced_pd.sh / serve_a3_ced_single.sh 最终都汇到这里，
# 放在上层会被实验用的旁路启动器绕过（已踩过一次）。
# 判据同样用**页尾**：num_blocks ≤ ⌊2³² / 147712⌋ = 29076。
# 与它配套的**强制**校验在连接器里（[CED-32BIT-GUARD]，按实测 stride 抛错）。
#
# ★ 2026-09-29 作用域修正（docs/KV32-POOL-GUARD-SCOPE-20260929.md）：
#   本守卫原先在**所有形态**下生效，且"未设置 KV_CACHE_MEMORY_BYTES ⇒ 直接 pin"
#   ⇒ 非 CED 部署（A2 单实例、OffloadingConnector、验证入口）也被强制 pin 到
#   29076 块。后果是 vLLM 走
#     “reserved X GiB for KV Cache as specified by kv_cache_memory_bytes,
#       skipping memory profiling.”（v1/worker/gpu_worker.py:474）
#   即**跳过自动显存 profiling**、`GPU_UTIL` 对 KV 池不再生效。
#   现在：只有真正会撞回绕的形态（CED 角色 / Mooncake PD）默认 pin；
#   其它形态交回 vLLM 自动 profiling，并在起服后**复核**池大小（见下方
#   "起服必查 ②"），把"profiling 恰好算出越界值"这条路也堵上。
#
# 三态开关 V41_KV32_POOL_GUARD：
#   auto（默认）：CED 角色（V41_CED_ROLE 非空）或 Mooncake PD
#                 （KV_ARGS_EXTRA 含连接器名）⇒ pin；其它 ⇒ 不 pin。
#   on          ：无条件 pin（旧行为，逃生用）。
#   off         ：完全不干预（不 pin、不 clamp、不起服后复核）。
_kv32_scope=${V41_KV32_POOL_GUARD:-auto}
case "$_kv32_scope" in
  auto|on|off) ;;
  *)
    echo "[serve_a2] [KV32] WARNING: V41_KV32_POOL_GUARD='$_kv32_scope' 非法（只能是 auto|on|off）⇒ 按 auto 处理" >&2
    _kv32_scope=auto
    ;;
esac
_kv32_pin=0
case "$_kv32_scope" in
  off) _kv32_pin=0 ;;
  on)  _kv32_pin=1 ;;
  auto)
    [ -n "${V41_CED_ROLE:-}" ] && _kv32_pin=1
    case "${KV_ARGS_EXTRA:-}" in *MooncakeHybridConnector*) _kv32_pin=1 ;; esac
    ;;
esac
# 兼容旧逃生口（语义是"已确认接受越界风险"，不是场景开关）：
#   ⇒ 同时关掉 pin / clamp / 起服后复核。仅作用于宿主侧；
#     连接器里的 [CED-32BIT-GUARD] 硬门**不受它影响**（该变量没有进 `docker run -e`）。
_kv32_enforce=1
_kv32_off_reason=""
if [ "$_kv32_scope" = "off" ]; then
  _kv32_pin=0
  _kv32_enforce=0
  _kv32_off_reason="scope=off"
fi
if [ "${V41_CED_ALLOW_32BIT_OVERFLOW:-0}" = "1" ]; then
  _kv32_pin=0
  _kv32_enforce=0
  _kv32_off_reason="V41_CED_ALLOW_32BIT_OVERFLOW=1"
fi
# 兼容 CED 角色脚本历史上用的 D 前缀变量名（两处默认值相同，收敛后仍认它，
#   避免外部按旧名字调参时静默失效）。
_ced_max_blocks=${CED_MAX_NUM_BLOCKS:-29076}
_ced_bytes_per_block=${CED_BYTES_PER_BLOCK:-${CED_D_BYTES_PER_BLOCK:-540928}}
_ced_cap=$(( _ced_max_blocks * _ced_bytes_per_block ))
_kv32_pinned=0
# 记录"调用方**显式**给过值" —— 显式给值说明调用方自己管池大小，
# 起服后就不必再复核（也就避开了非 A2 布局下除数不适用导致的误报）。
_kv32_user_set=0
[ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && _kv32_user_set=1
if [ "$_kv32_enforce" = "1" ]; then
  if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && [ "$KV_CACHE_MEMORY_BYTES" -gt "$_ced_cap" ]; then
    # clamp 与形态无关：回绕是**模型级**风险，显式给大值不改变物理事实。
    echo "[serve_a2] [KV32] WARNING: KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES 会让池超过 4 GiB 寻址上界（$_ced_max_blocks 块）"
    echo "[serve_a2] [KV32] WARNING: 钳到 $_ced_cap B。要完全绕过设 V41_KV32_POOL_GUARD=off"
    KV_CACHE_MEMORY_BYTES=$_ced_cap
  elif [ "$_kv32_pin" = "1" ] && [ -z "${KV_CACHE_MEMORY_BYTES:-}" ]; then
    # 这些形态**不能**靠 GPU_UTIL 自动 profiling —— 它会按"显存能装多少"算出
    # 30080 块（P 侧实测），越界后静默空答。直接给一个安全值。
    KV_CACHE_MEMORY_BYTES=$_ced_cap
    _kv32_pinned=1
    echo "[serve_a2] [KV32] scope=$_kv32_scope pin=1 ⇒ 池按 4 GiB 上界 pin：$KV_CACHE_MEMORY_BYTES B（num_blocks=$_ced_max_blocks）"
  fi
else
  echo "[serve_a2] [KV32] $_kv32_off_reason ⇒ 完全不干预：不 pin / 不 clamp / 起服后不复核"
fi
if [ "$_kv32_enforce" = "1" ] && [ "$_kv32_pinned" != "1" ] && [ -z "${KV_CACHE_MEMORY_BYTES:-}" ]; then
  echo "[serve_a2] [KV32] scope=$_kv32_scope pin=0 ⇒ 不设置 KV_CACHE_MEMORY_BYTES（交回 vLLM 自动 profiling；GPU_UTIL 生效）；起服后将复核池大小"
fi
# [SELFTEST-HOOK] 只解析并打印 KV32 作用域三元组后退出 —— 供 tools/selftest_kv32_scope.sh
#   做正控/负控（生产环境不会设这个变量）。放在守卫尾、任何 docker 动作之前，
#   所以它不依赖镜像、不占卡、不起容器。
if [ "${V41_KV32_GUARD_CHECK_ONLY:-0}" = "1" ]; then
  echo "KV32_RESOLVED scope=$_kv32_scope pin=$_kv32_pin enforce=$_kv32_enforce pinned=$_kv32_pinned cap=$_ced_cap bytes=${KV_CACHE_MEMORY_BYTES:-<unset>}"
  exit 0
fi

# [SCRIPT-VER] 让"跑的是哪一份脚本"在日志里可查（用户报障的第一件事）
_script_self="${BASH_SOURCE[0]}"
_script_md5=$(md5sum "$_script_self" 2>/dev/null | cut -c1-12 || echo "?")
echo "[serve_a2] script=$_script_self ver=$SERVE_A2_VER md5=$_script_md5"
case "$HERE" in
  /opt/dsv41/scripts)
    echo "[serve_a2] ⚠️  你现在跑的是**镜像里烘焙**的那份脚本（$HERE）。
          镜像内的副本是 build_image.sh 构建时 COPY 进去的，可能比发布包旧；
          若本次要修的挂载逻辑看起来没生效，请改用**发布包里**的：
            bash <发布包>/scripts/serve_a2.sh" ;;
esac

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

# [MODEL-MOUNT-ALL-RW] ⚠️ 仅诊断用开关（默认 0）：把**整棵模型树**都挂成 :rw。
#
# 为什么会有这个开关：v8 起 `engram_int8/` 必须可写（aclrtHostRegister 只接受可写
# 映射，只读 VMA → ret=507899）。默认逻辑（上面 [ENGRAM-RW]）只放开 engram 表目录，
# 其余模型目录保持 :ro —— 这是**最小暴露面**的生产口径。
# 但排查阶段（例如怀疑还有别的目录被驱动/代码要求可写、或想快速排除 :ro 因素）需要
# 一个"一把全开"的手段，免得逐个目录试。
#
# 用法：MODEL_MOUNT_ALL_RW=1 bash scripts/serve_a2.sh ...
# 效果：所有 MODEL_MOUNTS 条目（含 ancestor / auto 逐目录 / 单层 fallback /
#       EXTRA_MODEL_MOUNTS）的 `:ro` 一律变 `:rw`；起服日志会打印 ALL-RW 告警。
# 注意：**不要用于生产**。放宽 :ro 意味着容器内进程（以 root 跑）可以改写权重与
#       配置；调试完请去掉该 env，或改回 ENGRAM_DEVICE_INDEX=0。
MODEL_MOUNT_ALL_RW=${MODEL_MOUNT_ALL_RW:-0}
_ROMODE=ro
if [ "$MODEL_MOUNT_ALL_RW" = "1" ]; then _ROMODE=rw; fi

# ---------------------------------------------------------------------------
# [ENGRAM-RW] engram 表目录必须**可写**挂载 —— 为什么，以及怎么判定
#
# 代码级事实（patches/files/engram_device_index.py::_map_and_register）：
#     fd   = os.open(self.path, os.O_RDWR)                      # ← 要求可写
#     addr = libc.mmap(None, maplen, PROT_READ|PROT_WRITE, MAP_SHARED, fd, aligned)
#     dev, ret = acl.rt.host_register(addr, maplen, ACL_HOST_REGISTER_MAPPED)
# ⇒ 只读 VMA 会被 aclrtHostRegister 拒绝（**实测 ret=507899**），而且 os.open 在
#   read-only 挂载上先就报 EROFS。所以 engram 表的**每一个落盘目录**都必须 :rw。
#   注意：代码只**读**这些文件，要写权限纯粹是驱动注册的要求。
#
# 用户报障（v8，A2 真机）：默认路径之外的另外两条路径
#   （MODEL_MOUNT_MODE=ancestor / 单层 fallback）本来**完全没有 engram 特判**，
#   整棵模型树都是 :ro ⇒ 报错发生在**容器内、起服中途**
#   `aclrtHostRegister failed: ret=507899`，极难定位。本段把这件事前移到脚本里。
#
# 判定口径（三个来源都要，少一个就会漏）：
#   ① MODEL 自己：$MODEL/engram_int8（**是真实目录时**它根本不在
#      model_mount_args.sh 的输出里 ⇒ 旧代码的 `case ${_d##*/}` 永远不命中）；
#   ② 软链链条上的 engram 目录（名字含 engram 且含 int8：engram_int8 /
#      engram-int8 / engram_int8_data …；不能只匹配两个固定名字）；
#   ③ **真正落盘的那个目录**：engram_int8/ 里的条目本身还是软链
#      （quant/scripts/engram_dr_build.py 就是这么造的），而 O_RDWR 是按最终
#      inode 所在目录判定的 ⇒ 必须把 `readlink -f` 之后的目录也挂成 :rw。
#      A3 真机实测布局（同一份交付包）：
#        $MODEL/engram_int8 -> L4/engram_int8 -> L3/engram_int8（实体目录）
#          -> 4 个软链 -> /home/…/projects/dsv41/models/out/engram-int8/…
#      最后一层跟模型树**不在同一棵目录树里**（一个在 models/out、一个在
#      projects/dsv41/models/out）—— 只挂模型树那几层是不够的。
# ---------------------------------------------------------------------------
_ENGRAM_RW_DIRS=()

_engram_rw_add() {   # 登记一个"必须 :rw"的宿主目录（去重）
  local _x="$1" _i
  [ -n "$_x" ] || return 0
  for _i in ${_ENGRAM_RW_DIRS[@]+"${_ENGRAM_RW_DIRS[@]}"}; do
    [ "$_i" = "$_x" ] && return 0
  done
  _ENGRAM_RW_DIRS+=("$_x")
}

_engram_collect_rw() {   # $1 = 模型根目录（宿主路径 = 容器路径）
  local _base="$1" _c _real _f
  for _c in "$_base/engram_int8" "$_base/engram-int8"; do
    [ -e "$_c" ] || continue
    _engram_rw_add "$_c"                       # ← 容器里被 open() 的那个路径
    _real=$(readlink -f "$_c" 2>/dev/null || true)
    _engram_rw_add "$_real"                    # ← 软链真正指向的目录
    # 目录内的条目：glob **会穿过软链**列出真实内容，逐个解析到最终落盘目录
    for _f in "$_c"/*; do
      [ -e "$_f" ] || continue
      [ -L "$_f" ] || continue
      _real=$(readlink -f "$_f" 2>/dev/null || true)
      [ -n "$_real" ] && _engram_rw_add "$(dirname "$_real")"
    done
  done
}

_engram_is_rw_dir() {   # $1 = 目录；0 = 它（或它下面的东西）必须 :rw
  local _d="$1" _r
  for _r in ${_ENGRAM_RW_DIRS[@]+"${_ENGRAM_RW_DIRS[@]}"}; do
    case "$_d/" in "$_r"/*) return 0 ;; esac
  done
  return 1
}

# [MODEL-MOUNT-ALL-RW] 统一判定"这条模型目录要不要挂成 :rw"。
# 默认口径 = 只有 engram 表目录要 rw；MODEL_MOUNT_ALL_RW=1 时**全部**要 rw。
_model_dir_needs_rw() {   # $1 = 宿主目录；0 = 需要 :rw
  [ "$_ROMODE" = "rw" ] && return 0
  _engram_need_rw && _engram_is_rw_dir "$1"
}

_engram_mount_mode_for() {   # $1 = 宿主目录 → 打印 "<最深覆盖它的 mode> <容器路径>"
  local _t="$1" _i _entry _dst _mode _best_dst="" _best_mode=""
  for ((_i=0; _i<${#MODEL_MOUNTS[@]}; _i++)); do
    [ "${MODEL_MOUNTS[$_i]}" = "-v" ] || continue
    _entry="${MODEL_MOUNTS[$((_i+1))]:-}"
    [ -n "$_entry" ] || continue
    _dst="${_entry#*:}"; _dst="${_dst%:*}"
    _mode="${_entry##*:}"
    # 覆盖判定：相等，或 _t 在 _dst 之下（嵌套挂载里**最深的那条**说了算）
    case "$_t/" in "$_dst"/*) : ;; *) continue ;; esac
    if [ "${#_dst}" -ge "${#_best_dst}" ]; then _best_dst="$_dst"; _best_mode="$_mode"; fi
  done
  printf '%s %s' "${_best_mode:--}" "${_best_dst:--}"
}

_engram_required_by_config() {   # 0 = 模型 config 声明了 engram_layer_ids
  local _cfg="$MODEL/config.json"
  [ -f "$_cfg" ] || return 1
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$_cfg" <<'PYENGRAM' 2>/dev/null
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ids = (cfg.get("text_config") or {}).get("engram_layer_ids") \
      or cfg.get("engram_layer_ids") or []
sys.exit(0 if ids else 1)
PYENGRAM
  else
    grep -q '"engram_layer_ids"[[:space:]]*:[[:space:]]*\[[[:space:]]*"' "$_cfg"
  fi
}

# ⚠️ 关于"宿主上可写（[ -w ]）"这条：**只告警，不致命**。为什么（A3 真机实测）：
#   交付模型里 `…/projects/dsv41/models/out/engram-int8/*.safetensors` 是
#   **root:root 0600**，跑脚本的普通用户 `[ -w ]` 判为假，而**容器是以 root 运行的**
#   （Dockerfile `USER root` + docker run 不带 --user）⇒ root 打开 rw 挂载里的这些
#   文件完全没问题。拿 [ -w ] 当硬判据会在这种**完全正常**的部署上直接拦死起服
#   （本修复的第一版就在 A3 上踩了这个假阳性）。
#   ⇒ 硬判据只有两条：**目录存在** + **最深覆盖它的挂载是 :rw**。
#   宿主不可写只作为提示（说明容器必须是以 root 起；换非 root 才需要 chmod/换属主）。
_engram_warn_host_ro() {   # $@ = 宿主上不可写的路径
  [ "$#" -gt 0 ] || return 0
  echo "[serve_a2] WARNING: 宿主上 $(id -un) 对下面这些 engram 路径不可写（[ -w ] 为假）："
  local _x
  for _x in "$@"; do echo "[serve_a2]          $_x"; done
  echo "[serve_a2]          容器默认以 **root** 运行（本脚本不带 --user，镜像也是 USER root）"
  echo "[serve_a2]          ⇒ 只要挂载是 :rw，root 打开它们没问题（A3 真机实测就是这个形态："
  echo "[serve_a2]          engram-int8/*.safetensors 是 root:root 0600，服务正常）。"
  echo "[serve_a2]          只有当你改用**非 root** 起容器时才会 os.open EACCES："
  echo "[serve_a2]          那时修法 = chmod u+w / chown 这些文件，或 ENGRAM_DEVICE_INDEX=0（走 host 路径）。"
}

# [ENGRAM-RW-PREFLIGHT] 起服前自检：存在 + 在 MODEL_MOUNTS 里**没有被一条 :ro 覆盖**
# （即最深覆盖它的那条必须是 :rw）。不满足就 die 并给出可直接照做的修法。
_engram_preflight_check() {
  local _d _info _mode _dst _bad="" _why="" _f
  local -a _ro=()
  for _d in ${_ENGRAM_RW_DIRS[@]+"${_ENGRAM_RW_DIRS[@]}"}; do
    if [ ! -d "$_d" ]; then _bad="$_d"; _why="目录不存在（软链悬空？）"; break; fi
    _info=$(_engram_mount_mode_for "$_d"); _mode="${_info%% *}"; _dst="${_info##* }"
    if [ "$_mode" != "rw" ]; then
      _bad="$_d"
      _why="当前 MODEL_MOUNTS 里它被 '$_dst:$_mode' 覆盖（只读），或根本没有被挂载"
      break
    fi
    if [ ! -w "$_d" ]; then _ro+=("$_d（目录）"); continue; fi
    for _f in "$_d"/*.safetensors; do
      [ -e "$_f" ] || continue
      [ -w "$_f" ] || { _ro+=("$_f"); break; }
    done
  done
  _engram_warn_host_ro ${_ro[@]+"${_ro[@]}"}
  [ -n "$_bad" ] || return 0
  die "engram 表目录需要**可写**挂载（aclrtHostRegister 要求 O_RDWR 映射；只读 VMA 会
       ret=507899 / os.open 先报 EROFS），但起服前自检没过：$_why
       受影响目录：$_bad
       修法二选一：
         * 用 MODEL_MOUNT_MODE=auto（默认，逐目录挂载 ⇒ 脚本会自动把 engram 表目录叠加成 :rw）
         * 或显式 ENGRAM_DEVICE_INDEX=0（走 host 路径，不要求可写；代价是关掉 v8 的
           device-index 加速）"
}

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

  # [ENGRAM-RW] 登记必须 :rw 的 engram 表目录（只在本机可能启用 device-index 时才做）
  if _engram_need_rw; then
    for _d in ${_mdirs[@]+"${_mdirs[@]}"}; do
      _bn=$(printf '%s' "${_d##*/}" | tr 'A-Z' 'a-z')
      case "$_bn" in *engram*int8*) _engram_rw_add "$_d" ;; esac
    done
    _engram_collect_rw "$MODEL"
  fi

  if [ "$MODEL_MOUNT_MODE" = "ancestor" ] && [ "${#_mdirs[@]}" -gt 1 ]; then
    _anc=$(printf '%s\n' "${_mdirs[@]}" | xargs -r -n1 dirname | sort -u | head -1)
    # [ANCESTOR-COVER] `dirname | sort -u | head -1` **不是真的公共祖先**，它只取
    # 字典序最小的那个父目录。A3 真机布局里模型树横跨 models/out 与
    # projects/dsv41/models/out **两棵树** ⇒ 挑出来的 models/out 覆盖不到
    # projects/... 下的目录，挂进容器后那些软链全是悬空（半坏，且不报错）。
    # 这里加覆盖性检查：覆盖不全就退回 auto（与"取不到公共祖先就退回 auto"同一策略）。
    _miss=""
    if [ -n "${_anc:-}" ] && [ -d "$_anc" ]; then
      for _d in ${_mdirs[@]+"${_mdirs[@]}"}; do
        case "$_d/" in "$_anc"/*) : ;; *) _miss="$_d"; break ;; esac
      done
    else
      _miss="（取不到公共祖先）"
    fi
    if [ -n "$_miss" ]; then
      say "⚠️  MODEL_MOUNT_MODE=ancestor 选出的祖先 $_anc 覆盖不到 $_miss ⇒ 退回 auto（逐目录挂载）"
      MODEL_MOUNT_MODE=auto
    else
      MODEL_MOUNTS=(-v "$_anc:$_anc:$_ROMODE")
      if [ "$_ROMODE" = "rw" ]; then
        say "模型挂载（ancestor 模式，1 个目录）：$_anc（:rw ← MODEL_MOUNT_ALL_RW=1 强制）"
      else
        say "模型挂载（ancestor 模式，1 个目录）：$_anc（:ro；engram 表目录会单独叠加 :rw）"
      fi
    fi
  fi
  if [ "${#MODEL_MOUNTS[@]}" -eq 0 ]; then
    for _d in ${_mdirs[@]+"${_mdirs[@]}"}; do
      # [DEVICE-INDEX] aclrtHostRegister 拒绝只读 VMA（ret=507899），所以
      # Engram 表所在目录必须可写挂载 —— 代码只读它，但驱动要在上面取引用。
      # 只放开 engram 表目录（判定见上面 [ENGRAM-RW]），其余模型目录保持 :ro。
      # [MODEL-MOUNT-ALL-RW] 开该开关时全部走 :rw（见 _model_dir_needs_rw 注释）。
      if _model_dir_needs_rw "$_d"; then
        MODEL_MOUNTS+=(-v "$_d:$_d:rw")
      else
        MODEL_MOUNTS+=(-v "$_d:$_d:$_ROMODE")
      fi
    done
    say "模型挂载（auto 模式，${#_mdirs[@]} 个目录，含软链链条；ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX MODEL_MOUNT_ALL_RW=$MODEL_MOUNT_ALL_RW）"
    for _d in ${_mdirs[@]+"${_mdirs[@]}"}; do
      if _model_dir_needs_rw "$_d"; then
        say "   -v $_d:$_d:rw"
      else
        say "   -v $_d:$_d:$_ROMODE"
      fi
    done
  fi
  rm -f "$_mma_err"
else
  MODEL_MOUNTS=(-v "$MODEL:$MODEL:$_ROMODE")
  if [ "$MODEL_MOUNT_MODE" = "none" ]; then
    say "模型挂载（none 模式 —— 软链会悬空，仅用于复现故障）"
  else
    say "⚠️  找不到 tools/model_mount_args.sh，退回只挂 MODEL 一层（软链会悬空）"
  fi
  if [ "$_ROMODE" = "rw" ]; then
    say "   ← MODEL_MOUNT_ALL_RW=1：这一层也是 :rw"
  fi
  # [ENGRAM-RW] 单层 fallback（用户报障的第 3 条路径）同样要叠加 engram 表目录 :rw
  if _engram_need_rw; then
    _engram_collect_rw "$MODEL"
  fi
fi

# [ENGRAM-RW-OVERLAY] 把还没作为容器路径出现过的 engram 表目录**叠加**挂成 :rw。
# 祖先/单层挂载都是 :ro 的宽挂载，这里用嵌套挂载只放开 engram 那几层目录
# （docker 允许嵌套覆盖，且深的那条赢），避免"把整棵模型目录开成可写"。
if [ "${#_ENGRAM_RW_DIRS[@]}" -gt 0 ]; then
  for _r in ${_ENGRAM_RW_DIRS[@]+"${_ENGRAM_RW_DIRS[@]}"}; do
    _seen=0
    for ((_i=0; _i<${#MODEL_MOUNTS[@]}; _i++)); do
      [ "${MODEL_MOUNTS[$_i]}" = "-v" ] || continue
      _entry="${MODEL_MOUNTS[$((_i+1))]:-}"; _dst="${_entry#*:}"; _dst="${_dst%:*}"
      [ "$_dst" = "$_r" ] && { _seen=1; break; }
    done
    [ "$_seen" = "1" ] || MODEL_MOUNTS+=(-v "$_r:$_r:rw")
  done
fi

# [ENGRAM-RW-PREFLIGHT] 起服前自检（在 docker run 之前失败，且给出修法）
if _engram_need_rw; then
  if [ "${#_ENGRAM_RW_DIRS[@]}" -eq 0 ] && _engram_required_by_config; then
    die "$MODEL/config.json 声明了 engram_layer_ids，但模型目录里找不到 engram 表目录
         （$MODEL/engram_int8 或 $MODEL/engram-int8）。
         这不是挂载问题，而是模型目录本身不完整（Engram 表 ≈206 GiB 没就位）——
         起服会在容器里以另一种面目失败（读不到表文件），所以在这里先拦。
         修法：把 engram_int8/ 放回模型目录（quant/scripts/engram_dr_build.py 的产物），
         或换一个 config.json 里 engram_layer_ids 为空的模型目录。"
  fi
  _engram_preflight_check
  if [ "${#_ENGRAM_RW_DIRS[@]}" -gt 0 ]; then
    say "engram 表目录（${#_ENGRAM_RW_DIRS[@]} 个）挂成 :rw（只读 VMA 会被 aclrtHostRegister 拒绝：ret=507899）："
    for _r in ${_ENGRAM_RW_DIRS[@]+"${_ENGRAM_RW_DIRS[@]}"}; do say "   -v $_r:$_r:rw"; done
  fi
fi

if [ -n "${EXTRA_MODEL_MOUNTS:-}" ]; then
  IFS=';' read -r -a _extra <<<"$EXTRA_MODEL_MOUNTS"
  for _p in "${_extra[@]}"; do
    [ -n "$_p" ] && MODEL_MOUNTS+=(-v "$_p:$_p:$_ROMODE")
  done
fi

# [MODEL-MOUNT-ALL-RW] 生效时打一条**显眼**告警：这是诊断口径，不是生产口径。
if [ "$MODEL_MOUNT_ALL_RW" = "1" ]; then
  say "⚠️⚠️  MODEL_MOUNT_ALL_RW=1：模型树**全部**挂成 :rw（诊断口径，勿用于生产）"
  say "       放宽范围：MODEL_MOUNTS 里所有条目（含 ancestor / auto / fallback / EXTRA）"
  say "       风险：容器内以 root 运行的进程可改写权重与 config.json"
  say "       调试完请去掉该 env（回到默认只放开 engram 表目录），或改用 ENGRAM_DEVICE_INDEX=0"
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
  # [CAPTURE-BUCKET-6N 2026-10-04] ★ 用 `N × (SP_TOKENS+1)` 对齐的桶，替换原来的
  # 几何级数（2 的幂/整十）。为什么：静态 K 下每步行数恒为 `T = N × (1+K)`，
  # 几何表里没有 T=18/30/36/42（N=3/5/6/7），会被 padding 到 20/32/40/48 ——
  # 白算 6.7%~14% 的行。而 8 路并发**不是同时进 decode 的**（admission gate 把 prefill
  # 串行化 ⇒ 每个窗口都有 1→N 爬升与 N→1 回落），所以这些档真的会被走到。
  # 受控 A/B（同脚本同 4 rep、两轮相隔 33 分钟、冷启值逐位相同）：**N=8 +4.3%、N=16 +0.7%**。
  # 见 docs/CAPTURE-BUCKET-6N-20261004.md。
  # 覆盖的并发档是**稀疏但完整覆盖关键档**的集合（桶数 ~16，与几何表相当，
  # 起服捕获时间几乎不变；实测 9:56）。其余档 padding 到最近的上方桶。
  for _n in 1 2 3 4 5 6 7 8 10 12 16; do
    _c=$(( _n * _step_tokens ))
    if [ "$_c" -ge "$_step_tokens" ] && [ "$_c" -le "$_cap_max" ]; then CAPTURE_SIZES="$CAPTURE_SIZES,$_c"; fi
  done
  # 覆盖到 MAX_SEQS 档（大 batch 若没有桶会被判为不可图 ⇒ 退 eager）
  if [ "$_cap_max" -gt $(( 16 * _step_tokens )) ]; then CAPTURE_SIZES="$CAPTURE_SIZES,$_cap_max"; fi
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
# [DECODE-API-GUARD] decode 半边的请求边界护栏（事故 2026-09-27 00:01：
# 一条直连 18991 的普通请求让 EngineCore 退出、整个 D 实例死掉）。
# 护栏本体是 patches/files/v41_decode_guard.py，挂到 /opt/dsv41/guards/
# 让 vLLM 的 `--middleware v41_decode_guard.decode_guard` 能 import 到。
# 这里**无条件挂**（与 PATCH_MODE 无关）：判据是"起服日志里那行 middleware loaded"，
# 缺文件时 serve_v2.sh 会响亮告警，不会静默退化。
if [ -f "$PKG/patches/files/v41_decode_guard.py" ]; then
  MOUNTS+=(-v "$PKG/patches/files/v41_decode_guard.py:/opt/dsv41/guards/v41_decode_guard.py:ro")
fi
# [DCP-DEV] V4.1 DCP 开发用的**整文件覆盖挂载**（2026-09-29）。
#   §动机：DCP 需要替换 `vllm_ascend/core/deepseek_v41.py`（cache spec 与分配）、
#   `attention/dsa_v41.py`（attention 执行）、以及新增 `attention/context_parallel/*_dcp.py`。
#   每改一行就重打镜像不现实；这个开关把一棵**镜像容器路径布局**的目录整棵挂进去，
#   让「改文件 → 重启服务」闭环，而不用动生产路径（默认不设 = 完全不生效）。
#
#   用法：
#     mkdir -p ~/dcpw/vllm_ascend/core && cp <image>/.../deepseek_v41.py ~/dcpw/vllm_ascend/core/
#     V41_DCP_MOUNT=$HOME/dcpw bash scripts/serve_a3.sh ...
#   目录里的相对路径 = 相对 `/vllm-workspace/vllm-ascend/` 的路径。
#   ★ 判据落在「实际生效」：起服日志会打印每个被挂的文件；挂载了不存在的容器路径
#     docker 会自己建目录（静默），所以这里**逐个校验**相对路径在镜像里存在。
if [ -n "${V41_DCP_MOUNT:-}" ]; then
  [ -d "$V41_DCP_MOUNT" ] || die "V41_DCP_MOUNT=$V41_DCP_MOUNT 不是目录"
  _dcp_n=0
  # ★ 先收进**独立数组**，不要直接进 MOUNTS：下面 mount 模式还可能挂同一路径，
  #   而去重时若不分家，就会把 overlay 自己删掉（2026-09-29 实踩）。
  DCP_MOUNTS=()
  while IFS= read -r _rel; do
    _rel=${_rel#./}
    case "$_rel" in
      *.py) ;;
      # ★★★ [V41-DCP-SO-MOUNT 2026-10-01] 同时挂 `.so`。
      #   为什么需要：AscendC 融合算子（`v41_merge_kernel.so`）与 `.py` 同目录，
      #   原来这里只放行 `*.py` ⇒ 容器里看不到 `.so` ⇒
      #   `v41_merge_kernel.available()` 恒为 False，融合算子**静默退回
      #   Python 路径**（不报错、只是没有加速，极难发现）。
      *.so) ;;
      *) continue ;;
    esac
    _dst="/vllm-workspace/vllm-ascend/$_rel"
    DCP_MOUNTS+=(-v "$V41_DCP_MOUNT/$_rel:$_dst:rw")
    _dcp_n=$((_dcp_n + 1))
    echo "[serve_a2][DCP] mount $_rel → $_dst"
  done < <(cd "$V41_DCP_MOUNT" && find . -type f \( -name '*.py' -o -name '*.so' \) | sort)
  [ "$_dcp_n" -gt 0 ] || die "V41_DCP_MOUNT=$V41_DCP_MOUNT 下没有 .py/.so 文件"
  echo "[serve_a2][DCP] 共挂 $_dcp_n 个文件（覆盖镜像内对应模块）"
  # ★ 挂载本身不构成证据：`-v SRC:DST` 在 DST 是文件、SRC 是文件时才有意义；
  #   写错路径 docker 会在宿主机建目录，容器里静默变成目录。所以把清单**写进
  #   serve_cmd.txt**（那个文件不会被 `: > $LOG` 截断），起服后据此核对。
  DCP_MOUNT_LIST=$(cd "$V41_DCP_MOUNT" && find . -type f \( -name '*.py' -o -name '*.so' \) | sed 's|^\./||' | sort | tr '\n' ' ')
  # 后面 mount 模式还会挂 patches/files/*，二者可能指向同一容器路径
  # （实测：engram_hbm.py）。docker 对重复的目的地直接报
  # `Duplicate mount point` 并拒绝起容器，所以这里记下 DCP 的目的地集合，
  # 在 mount 模式那一段结束后**移除冲突项**，让开发期 overlay 优先。
  DCP_MOUNT_DSTS=" $(cd "$V41_DCP_MOUNT" && find . -type f \( -name '*.py' -o -name '*.so' \) | sed 's|^\./||' | sed 's|^|/vllm-workspace/vllm-ascend/|' | sort | tr '\n' ' ')"
  echo "[serve_a2][DCP] MOUNT_LIST=$DCP_MOUNT_LIST"
  # 开发期需要透传给容器的 env（DCP 各阶段开关）：DCP_EXTRA_ENV="A=1 B=2"
  for _kv in ${DCP_EXTRA_ENV:-}; do
    MOUNTS+=(-e "$_kv")
    echo "[serve_a2][DCP] -e $_kv"
  done
fi
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
  # [V41-KV32-REPACK] core/deepseek_v41.py：把 ratio-1 源的 index 平面挪到另一个 slot 的
  #   空闲区 ⇒ 四个 slot 页步长全为 131072 ⇒ 块上限 29076 → 32768（+12.7% 容量，零精度风险）。
  #   门控默认关：只有 V41_KV32_REPACK=1 且 patches/files/deepseek_v41.repack.py 存在才挂。
  if [ "${V41_KV32_REPACK:-0}" = "1" ]; then
    if [ -f "$F/deepseek_v41.repack.py" ]; then
      MOUNTS+=(-v "$F/deepseek_v41.repack.py:/vllm-workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py:ro")
      say "[V41-KV32-REPACK] 已挂载 slot 重排版 deepseek_v41.py（四 slot 全 131072）"
    else
      die "V41_KV32_REPACK=1 但缺 patches/files/deepseek_v41.repack.py"
    fi
  fi
  # [META-HOST-SLEEP] 2026-10-04：device_metadata.py 的 host 侧 sleep 注入（**诊断用**）。
  #   背景：api_statistic 显示三件套 GetWorkspaceSize 合计 919.7ms/4396 次
  #         （平均 209µs，是算子入队 21µs 的 10 倍），但它在 host CPU 上跑，
  #         **可能被其它线程盖住** ⇒ 必须实测才能判定是否在关键路径上。
  #   用法：echo 300 > /tmp/v41_meta_host_sleep_us （单位 µs；0.25s 热切，无需重启）
  #         缺文件 / 非法值 / 未设 ⇒ 恒 0 ⇒ 行为与 stock 相同。
  #   来源：metadata 线子代理；见 docs/METADATA-AICPU-AUDIT-20261004.md。
  if [ -f "$F/device_metadata.py" ]; then
    MOUNTS+=(-v "$F/device_metadata.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py:ro")
    say "[META-HOST-SLEEP] device_metadata.py 已挂载（host 侧 sleep 注入，默认 0 = 与 stock 相同）"
  fi
  # [V41-SLOT-MAP-FUSED] block_table.py：12 次 slot-mapping 启动 → 1 次。
  # 由 env `V41_SLOT_MAP_FUSED` 门控。
  #
  # ★ 2026-10-02：**DCP>1 时默认改成 on**（见 docs/V41-DCP8-DECODE-PERF-20261002.md）。
  #   只对 DCP>1 改默认，不动 DCP=1（A2 生产）的既有行为；显式传值优先。
  case "${KV_ARGS_EXTRA:-}" in
    *--decode-context-parallel-size\ 1|*--decode-context-parallel-size\ 0|"") : ;;
    *--decode-context-parallel-size\ *) : "${V41_SLOT_MAP_FUSED:=on}" ;;
  esac
  export V41_SLOT_MAP_FUSED="${V41_SLOT_MAP_FUSED:-0}"
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

  # [V41-HC-FUSE] 自定义融合算子包：hc_pre_norm 需要 ASCEND_CUSTOM_OPP_PATH 指向它。
  # 用法：V41_HC_OPP_PKG=<宿主目录，内含 vendors/custom_transformer>。
  if [ -n "${V41_HC_OPP_PKG:-}" ]; then
    if [ -d "$V41_HC_OPP_PKG/vendors/custom_transformer" ]; then
      MOUNTS+=(-v "$V41_HC_OPP_PKG:/opt/dsv41/hcfuse_opp:ro")
      export ASCEND_CUSTOM_OPP_PATH=/opt/dsv41/hcfuse_opp/vendors/custom_transformer
      say "[V41-HC-FUSE] opp pkg: $V41_HC_OPP_PKG -> /opt/dsv41/hcfuse_opp（ASCEND_CUSTOM_OPP_PATH 已设）"
    else
      echo "[serve_a2] WARNING: V41_HC_OPP_PKG=$V41_HC_OPP_PKG 缺少 vendors/custom_transformer；跳过" >&2
    fi
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
# [DCP-DEV] 去重：开发期 overlay 与 patches/files 可能覆盖**同一个容器路径**，
# 而 docker 对重复目的地直接报 `Duplicate mount point` 并拒绝起容器。
# 语义定为 **overlay 优先**（开发期改的就是它），在这里把冲突的旧挂载项整对剔除。
if [ -n "${DCP_MOUNT_DSTS:-}" ]; then
  _new_mounts=()
  _mi=0
  _dropped=0
  while [ "$_mi" -lt "${#MOUNTS[@]}" ]; do
    if [ "${MOUNTS[$_mi]}" = "-v" ] && [ $((_mi + 1)) -lt "${#MOUNTS[@]}" ]; then
      _spec="${MOUNTS[$((_mi + 1))]}"
      _dst="${_spec#*:}"
      _dst="${_dst%%:*}"
      case "$DCP_MOUNT_DSTS" in
        *" $_dst "*)
          say "[DCP-DEV] 移除与 overlay 冲突的挂载：$_dst"
          _dropped=$((_dropped + 1))
          _mi=$((_mi + 2))
          continue
          ;;
      esac
      _new_mounts+=("${MOUNTS[$_mi]}" "${MOUNTS[$((_mi + 1))]}")
      _mi=$((_mi + 2))
      continue
    fi
    _new_mounts+=("${MOUNTS[$_mi]}")
    _mi=$((_mi + 1))
  done
  MOUNTS=("${_new_mounts[@]}")
  [ "$_dropped" = "0" ] || say "[DCP-DEV] 共移除 $_dropped 项冲突挂载（overlay 优先）"
  MOUNTS+=("${DCP_MOUNTS[@]}")
  say "[DCP-DEV] overlay 挂载已追加（${#DCP_MOUNTS[@]} 项，最后生效）"
fi
if [ -n "${V41_CED_ROLE:-}" ]; then
  # [PATCH_MODE] 两条路都支持：
  #   mount —— 官方基础镜像 + `-v` 挂本仓的 CED 文件（默认，开发时用）
  #   baked —— `local/dsv41-a3-ced-pd:*` 工作镜像，文件已在真实路径（部署时用）
  # 两者装的是**同一批文件**（清单见 deploy/a3-ced-pd/PAYLOAD.md）。
  case "$PATCH_MODE" in
    mount)
      _ced_connector="$PKG/experimental/ced/mooncake_hybrid_connector.py"
      [ -f "$_ced_connector" ] || die "V41_CED_ROLE 缺少 $_ced_connector"
      MOUNTS+=(-v "$_ced_connector:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:ro")
      ;;
    baked)
      # baked 不挂载；但要**立刻**证明镜像里那份确实是 CED 版，
      # 否则会静默跑 stock 连接器（起服成功、行为完全不同）。
      # 真正的判据放在容器起来之后（见下方 [CED-BAKED-GUARD]）。
      ;;
    *)
      die "V41_CED_ROLE 要求 PATCH_MODE=mount 或 baked，当前 $PATCH_MODE"
      ;;
  esac
  if [ "${PATCH_MODE}" = "mount" ] && [ "${V41_CED_ROLE:-}" = "decode" ]; then
    [ "${PROBE:-0}" != "1" ] || die "CED decode 实验不能与 PROBE=1 同时覆盖 dsa_v41.py"
    _ced_dsa="$PKG/experimental/ced/dsa_v41.py"
    _ced_scheduler="$PKG/experimental/ced/core_scheduler_replay.patch"
    [ -f "$_ced_dsa" ] && [ -f "$_ced_scheduler" ] || die "CED decode 缺少注意力或调度补丁"
    MOUNTS+=(-v "$_ced_dsa:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:ro")
    MOUNTS+=(-v "$_ced_scheduler:/opt/dsv41/ced_scheduler_replay.patch:ro")
  elif [ "${PATCH_MODE}" = "mount" ] && [ "${V41_CED_P_HIT_DIAG:-1}" = "1" ]; then
    # [CED-P-HIT] 2026-09-26：P（prefill 角色）开 PREFIX=1 时会在 stock 的
    # `assert num_new_tokens > 0` 上崩。P 不装 decode 的 replay 补丁，所以这里单独
    # 挂一份**只读诊断**补丁，把 num_tokens / num_computed_tokens / local / external
    # 四个量摊开，用来定位是"整段本地命中"还是"截断导致的口径不一致"。
    # 纯观测，不改任何分支。设 V41_CED_P_HIT_DIAG=0 可关。
    _ced_p_hit_patch="$PKG/experimental/ced/core_scheduler_prefill_hit.patch"
    [ -f "$_ced_p_hit_patch" ] || die "CED prefill 命中诊断缺少 $_ced_p_hit_patch"
    MOUNTS+=(-v "$_ced_p_hit_patch:/opt/dsv41/ced_scheduler_prefill_hit.patch:ro")
  fi
fi
if [ "${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-0}" = "1" ]; then
  [ "${V41_CED_ROLE:-}" = "decode" ] || die "CED prompt-tail 图补丁只允许 decode 角色"
  [ "$GRAPH" = "1" ] && [ "$EAGER" = "0" ] || die "CED prompt-tail 图补丁要求 GRAPH=1 EAGER=0"
  if [ "$PATCH_MODE" = "mount" ]; then
    _ced_runner_patch="$PKG/experimental/ced/core_model_runner_prompt_tail.patch"
    [ -f "$_ced_runner_patch" ] || die "CED prompt-tail 图补丁缺少 $_ced_runner_patch"
    MOUNTS+=(-v "$_ced_runner_patch:/opt/dsv41/ced_runner_prompt_tail.patch:ro")
  fi
fi
# [DYNAMIC-SPEC] 按并发切 K 所需要的两处 Ascend 侧改动：
#   ① patch_cudagraph.py —— 整文件替换 base 镜像的
#      `vllm_ascend/patch/worker/patch_cudagraph.py`：让 dispatcher 认得"本步
#      query_len"，并为每个 query_len 各建一组 decode 图；
#   ② core_model_runner_dynamic_spec.patch —— 运行期补丁，把"本步 query_len"
#      从 runner 逐帧传给 dispatcher，并修掉两处按静态 query_len 判分支的地方
#      （其中 `_pad_query_start_loc_for_fia` 会 `assert num_reqs == num_reqs_padded`，
#      不改就是"K=0 且 batch≠8 时直接打死引擎"）。
#   两者都只在 SP_SCHEDULE 非空时才有意义；此处**无条件挂载/应用**，
#   因为 mount 与否必须与"是否开了 dynamic SD"解耦（否则开关一开就缺文件）。
_dynspec_pc="$PKG/patches/files/patch_cudagraph.py"
if [ -f "$_dynspec_pc" ]; then
  MOUNTS+=(-v "$_dynspec_pc:/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_cudagraph.py:ro")
else
  [ -z "${SP_SCHEDULE:-}" ] || die "SP_SCHEDULE 需要 $_dynspec_pc（缺它则 query_len 只有单值 ⇒ K=0 的步会错配/崩溃）"
fi
if [ "$PATCH_MODE" = "mount" ] && [ -n "${SP_SCHEDULE:-}" ]; then
  _dynspec_runner="$PKG/experimental/ced/core_model_runner_dynamic_spec.patch"
  [ -f "$_dynspec_runner" ] || die "SP_SCHEDULE 需要 $_dynspec_runner"
  MOUNTS+=(-v "$_dynspec_runner:/opt/dsv41/ced_runner_dynamic_spec.patch:ro")
  # [DYNAMIC-SPEC] 上游那道"MRV1 + dynamic SD ⇒ 无条件降级 PIECEWISE"的门
  # 在 `VllmConfig.__post_init__` 里，**早于** runner 的任何代码。
  # 不在这里关掉它，V4.1 的 cache 初始化会在模型构造期直接 raise（约 2 分钟后），
  # 而且报错完全指不到这里。
  _dynspec_gate="$PKG/experimental/ced/core_config_dynamic_sd_gate.patch"
  [ -f "$_dynspec_gate" ] || die "SP_SCHEDULE 需要 $_dynspec_gate"
  MOUNTS+=(-v "$_dynspec_gate:/opt/dsv41/ced_config_dynamic_sd_gate.patch:ro")
fi
if [ -n "${V41_CED_SNAPSHOT_POS:-}" ] && [ -z "${V41_CED_ROLE:-}" ]; then
  [ "${PROBE:-0}" != "1" ] || die "CED cache snapshot 不能与 PROBE=1 同时覆盖 dsa_v41.py"
  _ced_dsa="$PKG/experimental/ced/dsa_v41.py"
  [ -f "$_ced_dsa" ] || die "CED cache snapshot 缺少 $_ced_dsa"
  MOUNTS+=(-v "$_ced_dsa:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:ro")
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
  echo "[a2-dry] ver=$SERVE_A2_VER md5=$_script_md5 script=$_script_self"
  echo "[a2-dry] image=$IMAGE name=$NAME port=$PORT served_name=$SERVED_NAME devs='$DEVS' util=$GPU_UTIL max_len=$MAX_LEN"
  echo "[a2-dry] TP=$TP DP=$DP MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX SP_TOKENS=$SP_TOKENS BAT_TOKENS=$BAT_TOKENS"
  echo "[a2-dry] GRAPH=$GRAPH EAGER=$EAGER"
  echo "[a2-dry] CAPTURE_SIZES=$CAPTURE_SIZES"
  echo "[a2-dry] MOE_AG=$MOE_AG O_PROJ_2D=$O_PROJ_2D MOE_MASK=$MOE_MASK ROPE_IDXSEL=$ROPE_IDXSEL ENGRAM_JIT=$ENGRAM_JIT QLI_NOCAND=$QLI_NOCAND LOCAL_OWNER=$LOCAL_OWNER IDS64_HOIST=$IDS64_HOIST PAD_SKIP=$PAD_SKIP ENGRAM_WKV_TP=$ENGRAM_WKV_TP"
  echo "[a2-dry] PROFILE=$V41_PROFILE ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX ENGRAM_DEVICE_FALLBACK=$ENGRAM_DEVICE_FALLBACK"
  echo "[a2-dry] KV_ARGS_EXTRA=${KV_ARGS_EXTRA:-<none>}（inner.sh 会原样透传）"
  echo "[a2-dry] DROPCACHE=$DROPCACHE（起服前清 page cache；0 关闭）"
  echo "[a2-dry] MOE_ZERO=$MOE_ZERO MOE_NF=$MOE_NF DRAFT_GRAPH=$DRAFT_GRAPH PYTHON_PGO=$PYTHON_PGO pgo_target=${PGO_LIB:-none} LOAD_FORMAT=${LOAD_FORMAT:-<real>} CAND_MODE=$CAND_MODE PATCH_MODE=$PATCH_MODE"
  echo "[a2-dry] CPUSET=$CPUSET${CPUSET_SRC:+ ($CPUSET_SRC)} MEMS=$MEMS${MEMS_SRC:+ ($MEMS_SRC)} STATIC_KERNEL=$STATIC_KERNEL NPUGRAPH_EX=$NPUGRAPH_EX MULTISTREAM=$MULTISTREAM HCCL_DET=${HCCL_DET:-none}"
  echo "[a2-dry] MOUNTS(${#MOUNTS[@]}): ${MOUNTS[*]:-<none>}"
  echo "[a2-dry] MODEL_MOUNT_MODE=$MODEL_MOUNT_MODE MODEL_MOUNTS(${#MODEL_MOUNTS[@]}):"
  # 打印成 "一行一条挂载"（`-v` 与 `路径:路径:mode` 拼在一行）—— 方便人工核对，
  # 也方便测试直接断言整条（tests/engram_rw_mount_test.sh）。
  _mi=0
  while [ "$_mi" -lt "${#MODEL_MOUNTS[@]}" ]; do
    echo "[a2-dry]    ${MODEL_MOUNTS[$_mi]} ${MODEL_MOUNTS[$((_mi+1))]:-}"
    _mi=$((_mi+2))
  done
  echo "[a2-dry] ENGRAM_RW_DIRS(${#_ENGRAM_RW_DIRS[@]}): ${_ENGRAM_RW_DIRS[*]:-<none>}"
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
#
# [DRAFT-FOUR-PIECE] `DSPARK_CAPTURE_VALUE_FIX` 默认 **1**（2026-09-20 端到端验证后定稿）。
#
# 四件套是 `DRAFT_GRAPH=1` 能正常工作的**最小集合**，缺一件就会静默退化：
#   * `DSPARK_CAPTURE_VALUE_FIX=1` —— 捕获期填代表值 + **恢复图内 context KV 写入**。
#     这一件必须显式传 1：缺它时 `_context_slot_mapping_buffers` 仍是 None，
#     `precompute_and_store_context_kv` 在捕获时提前 return ⇒ **图里根本没有"写 KV"
#     那串算子** ⇒ A 从 ~2.6 掉到 **1.07**、单流从 ~100 掉到 **40 tok/s**（实测）。
#   * 另外三件在代码/脚本里的默认值已经是 1：
#     `DSPARK_SWA_INDICES_RESIDENT`（`dsa_v1.py`，常驻索引缓冲）、
#     `DSPARK_CAPTURE_NCTX_FIX`（`dspark_proposer.py`，`num_reqs×(1+SP)`）、
#     `DSPARK_DISPATCH_QUERY_LEN_FIX`（`llm_base_proposer.py`，P0-B 高并发崩溃修复）。
#
# 为什么默认必须改成 1：只写 `DRAFT_GRAPH=1` 是最自然的用法，而旧默认 0 会让用户
# **拿到一个能起服、但 A≈1.07 的坏配置**，且没有任何报错 —— 只能靠 A/单流数字发现。
# 传 `DSPARK_CAPTURE_VALUE_FIX=0` 仍可复现旧行为（用于对照实验）。
# 注：draft 版文件只在 `DRAFT_GRAPH=1` 时才挂载，所以本默认对 `DRAFT_GRAPH=0` 无影响。
#
# ⚠️ 上面所有注释都必须在 `$DOCKER run` **之前** —— 曾把这段插进 `docker run` 的
#    续行链中间（`-e ... \` 之后），注释会**中断续行**，导致命令被截断成
#    `docker run ... -e DSPARK_HOIST_CONTEXT_KV=0` 而丢掉 IMAGE 参数，报
#    `"docker run" requires at least 1 argument` + `-e: command not found`。
#    `bash -n` **抓不到**这种错（语法合法），所以已加 `tools/check_serve_run_chain.py` 回归。
  # [V41-HC-FUSE] HcPre+RMSNorm 融合（Track B，env 门控，默认关）：V41_HC_FUSE_NORM / V41_HC_NORM_LIB / ASCEND_CUSTOM_OPP_PATH
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
  -e HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}" \
  -e ASCEND_MAX_OP_CACHE_SIZE=-1 \
  -e CAPTURE_SIZES="$CAPTURE_SIZES" \
  -e NUMBA_CACHE_DIR=/numba_cache \
  -e VLLM_ADMISSION_GATE="${VLLM_ADMISSION_GATE:-1}" \
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
  -e DSPARK_CAPTURE_VALUE_FIX="${DSPARK_CAPTURE_VALUE_FIX:-1}" \
  -e DSPARK_CAPTURE_SEQ_LEN="${DSPARK_CAPTURE_SEQ_LEN:-0}" \
  -e DSPARK_CAPTURE_NCTX_FIX="${DSPARK_CAPTURE_NCTX_FIX:-1}" \
  -e DSPARK_SWA_INDICES_RESIDENT="${DSPARK_SWA_INDICES_RESIDENT:-1}" \
  -e DSPARK_TOKEN_DUMP="${DSPARK_TOKEN_DUMP:-0}" \
  -e DSPARK_TOKEN_DUMP_STEPS="${DSPARK_TOKEN_DUMP_STEPS:-12}" \
  -e DSPARK_STEP_PROBE="${DSPARK_STEP_PROBE:-0}" \
  -e DSPARK_STEP_PROBE_STEPS="${DSPARK_STEP_PROBE_STEPS:-40}" \
  -e DSPARK_DISPATCH_DIAG_STEPS="${DSPARK_DISPATCH_DIAG_STEPS:-0}" \
  -e DSPARK_DSA_PROBE_STEPS="${DSPARK_DSA_PROBE_STEPS:-60}" \
  -e DSPARK_GRAPH_PTR_PROBE_STEPS="${DSPARK_GRAPH_PTR_PROBE_STEPS:-5}" \
  -e V41_ENGRAM_DEVICE_FALLBACK="$ENGRAM_DEVICE_FALLBACK" \
  -e V41_QLI_NO_CANDIDATE="$QLI_NOCAND" \
  -e V41_MOE_COMM_ALLGATHER="$MOE_AG" \
  -e V41_MOE_MASK_RANGE="$MOE_MASK" \
  -e V41_ROPE_IDXSEL="$ROPE_IDXSEL" \
  -e V41_IDS64_HOIST="$IDS64_HOIST" \
  -e V41_ENGRAM_PAD_SKIP="$PAD_SKIP" \
  -e V41_ENGRAM_WKV_TP="$ENGRAM_WKV_TP" \
  -e V41_O_PROJ_2D="$O_PROJ_2D" \
  -e V41_ENGRAM_ROUTE_PROBE="$ROUTE_PROBE" \
  -e V41_MOE_ZERO_INVALID="$MOE_ZERO" -e V41_MOE_ZERO_INVALID_FILE=/tmp/v41_moe_zero_file \
  -e V41_MOE_ZERO_NONFINITE="$MOE_NF" -e V41_MOE_ZERO_NONFINITE_FILE=/tmp/v41_moe_nf \
  -e V41_FORCE_CAND_MODE="$CAND_MODE" \
  -e V41_CED_SOURCE_COMPARE="${V41_CED_SOURCE_COMPARE:-0}" \
  -e V41_CED_SOURCE_COMPARE_CHUNKS="${V41_CED_SOURCE_COMPARE_CHUNKS:-1}" \
  -e V41_CED_ROLE="${V41_CED_ROLE:-}" \
  -e V41_CED_ALLOW_DSPARK="${V41_CED_ALLOW_DSPARK:-0}" \
  -e V41_SLOT_MAP_FUSED="${V41_SLOT_MAP_FUSED:-0}" \
  -e V41_CED_GRAPH_PROMPT_TAIL_EAGER="${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-0}" \
  -e V41_CED_SWA_CLIP="${V41_CED_SWA_CLIP:-1}" \
  -e V41_CED_SWA_TRACE="${V41_CED_SWA_TRACE:-0}" \
  -e V41_CED_BLOCK_TRACE="${V41_CED_BLOCK_TRACE:-0}" \
  -e V41_CED_BLOCK_DUMP_DIR="${V41_CED_BLOCK_DUMP_DIR:-}" \
  -e V41_CED_KVGEOM="${V41_CED_KVGEOM:-0}" \
  -e V41_ENGRAM_HIST_TRACE_POS="${V41_ENGRAM_HIST_TRACE_POS:-}" \
  -e V41_CED_SNAPSHOT_POS="${V41_CED_SNAPSHOT_POS:-}" \
  -e V41_CED_SNAPSHOT_DIR="${V41_CED_SNAPSHOT_DIR:-}" \
  -e V41_CED_H20_SNAPSHOT_POS="${V41_CED_H20_SNAPSHOT_POS:-}" \
  -e V41_CED_H20_SNAPSHOT_DIR="${V41_CED_H20_SNAPSHOT_DIR:-}" \
  -e V41_CED_LAYER_SNAPSHOT_POS="${V41_CED_LAYER_SNAPSHOT_POS:-}" \
  -e V41_CED_LAYER_SNAPSHOT_DIR="${V41_CED_LAYER_SNAPSHOT_DIR:-}" \
  -e V41_CED_LAYER_SNAPSHOT_LAYERS="${V41_CED_LAYER_SNAPSHOT_LAYERS:-0,1,2,13,14,15,19,20}" \
  -e V41_CED_CAPTURE_DECODE="${V41_CED_CAPTURE_DECODE:-0}" \
  -e V41_DECODE_API_GUARD="${V41_DECODE_API_GUARD:-1}" \
  -e SP_SCHEDULE="${SP_SCHEDULE:-}" \
  -e V41_CED_DYNAMIC_SPEC_FULL_GRAPHS="${V41_CED_DYNAMIC_SPEC_FULL_GRAPHS:-0}" \
  ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S"} \
  ${V41_DYNSPEC_BT_PERSIST:+-e V41_DYNSPEC_BT_PERSIST="$V41_DYNSPEC_BT_PERSIST"} \
  ${V41_DYNSPEC_SKIP_K0_DRAFT_COPY:+-e V41_DYNSPEC_SKIP_K0_DRAFT_COPY="$V41_DYNSPEC_SKIP_K0_DRAFT_COPY"} \
  ${V41_HC_FUSE_NORM:+-e V41_HC_FUSE_NORM="$V41_HC_FUSE_NORM"} \
  ${V41_GATE_MAX_PREFILL:+-e V41_GATE_MAX_PREFILL="$V41_GATE_MAX_PREFILL"} \
  ${V41_HC_FUSE_ATTN_FP32:+-e V41_HC_FUSE_ATTN_FP32="$V41_HC_FUSE_ATTN_FP32"} \
  ${V41_HC_NORM_LIB:+-e V41_HC_NORM_LIB="$V41_HC_NORM_LIB"} \
  ${ASCEND_CUSTOM_OPP_PATH:+-e ASCEND_CUSTOM_OPP_PATH="$ASCEND_CUSTOM_OPP_PATH"} \
  -e LOAD_FORMAT="$LOAD_FORMAT" \
  -e KV_ARGS_EXTRA="$KV_ARGS_EXTRA" \
  ${HCCL_ENV_ARGS[@]+"${HCCL_ENV_ARGS[@]}"} \
  -w /workspace "$IMAGE" \
  bash -lc "sleep infinity" >/dev/null || die "docker run 失败"
_CONTAINER_STARTED=1     # [FAIL-CLEANUP] 之后任何 die() 都会删掉这个容器

# ---------- 容器内环境 + 起服 ----------
# engram local-owner 用 /tmp 文件热切换（与 A3-node1 完全一致）
$DOCKER exec "$NAME" bash -lc "printf '%s' '$LOCAL_OWNER' > /tmp/v41_engram_localowner; printf '%s' 'fast' > /tmp/v41_hash_mode" || true

# ---------- [ADMISSION-GATE] mount 模式下现场打 vLLM core 补丁 ----------
# ---------- [DCP-MOUNT-GUARD] DCP 覆盖挂载必须**逐个证明**真的换掉了文件 ----------
# 判据不能落在"我传了 V41_DCP_MOUNT"：`-v SRC:DST` 在 DST 不存在时 docker 会
# 创建目录；而 DST 写错一层（例如少了 `vllm_ascend/`）时容器里那份代码根本没变，
# 起服照样成功、行为却完全不同（2026-09-29 容量探针已因此白跑一轮）。
# 这里比对**容器内 md5 vs 宿主 md5**，不一致就直接 die。
if [ -n "${V41_DCP_MOUNT:-}" ] && [ "${DCP_MOUNT_SKIP_VERIFY:-0}" != "1" ]; then
  _dfail=0
  while IFS= read -r _rel; do
    _dst="/vllm-workspace/vllm-ascend/$_rel"
    _want=$(md5sum "$V41_DCP_MOUNT/$_rel" | awk '{print $1}')
    _got=$($DOCKER exec "$NAME" bash -lc "test -f '$_dst' && md5sum '$_dst' | awk '{print \$1}' || echo NOT_A_FILE" 2>/dev/null | tail -1)
    if [ "$_got" != "$_want" ]; then
      echo "[serve_a2][DCP][FAIL] $_rel：容器内=$_got 宿主=$_want（挂载未生效）" >&2
      _dfail=1
    else
      say "[DCP-MOUNT-GUARD] $_rel ✓ md5=${_want:0:8}"
    fi
  done < <(cd "$V41_DCP_MOUNT" && find . -type f \( -name '*.py' -o -name '*.so' \) | sed 's|^\./||' | sort)
  [ "$_dfail" = "0" ] || die "DCP 覆盖挂载未生效（见上）；不要在该状态下做任何结论"
fi

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

# ---------- [CED-BAKED-GUARD] baked 模式必须**证明**镜像里的 CED 件真的在 ----------
# 判据落在"实际生效后的可观测痕迹"上，不能落在"我传了 PATCH_MODE=baked"：
# 烘错一层会**静默跑 stock 连接器**（起服成功、health 200、行为完全不同，
# 而且往往是长上下文才发作）。
if [ "${V41_CED_ROLE:-}" != "" ] && [ "${PATCH_MODE:-}" = "baked" ]; then
  say "[CED-BAKED-GUARD] 校验镜像内 CED 件（baked 模式不挂载）"
  _cedchk=$($DOCKER exec "$NAME" bash -lc '
    A=/vllm-workspace/vllm-ascend/vllm_ascend
    C=$A/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py
    D=$A/attention/dsa_v41.py
    [ -f "$C" ] || { echo "MISSING_CONNECTOR"; exit 0; }
    [ -f "$D" ] || { echo "MISSING_DSA"; exit 0; }
    c1=$(grep -c "CED-32BIT-GUARD" "$C" || true)
    c2=$(grep -c "ced_missing_swa_groups" "$C" || true)
    d1=$(grep -c "CED-SWA-CLIP" "$D" || true)
    echo "conn=${c1:-0}/${c2:-0} dsa=${d1:-0}"' 2>/dev/null | tail -1)
  case "${_cedchk:-}" in
    MISSING_*) die "[CED-BAKED-GUARD] 镜像里缺 CED 件（$_cedchk）——装的不是 dsv41-a3-ced-pd 工作镜像" ;;
    "")        die "[CED-BAKED-GUARD] 无法校验镜像内 CED 件（docker exec 失败）" ;;
    *)         say "[CED-BAKED-GUARD] 标记命中 = $_cedchk" ;;
  esac
  _ok=$($DOCKER exec "$NAME" bash -lc '
    A=/vllm-workspace/vllm-ascend/vllm_ascend
    c=$(grep -c "CED-32BIT-GUARD" $A/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py || true)
    d=$(grep -c "CED-SWA-CLIP" $A/attention/dsa_v41.py || true)
    [ "${c:-0}" -ge 1 ] && [ "${d:-0}" -ge 1 ] && echo OK || echo BAD' 2>/dev/null | tail -1)
  [ "${_ok:-}" = "OK" ] || die "[CED-BAKED-GUARD] CED 件未生效（connector/dsa_v41 里找不到 CED 标记）——结果不可当正确性证据"
  say "[CED-BAKED-GUARD] CED 连接器 + dsa_v41 均已生效 ✓"
fi

if [ "${V41_CED_ROLE:-}" = "decode" ]; then
  say "[CED-D] 应用固定 Core 版本的 128-token replay 调度补丁"
  _ced_patch=$($DOCKER exec "$NAME" bash -lc '
    cd /vllm-workspace/vllm || exit 1
    _base=$(sha256sum vllm/v1/core/sched/scheduler.py | cut -d " " -f1)
    [ "$_base" = 533eed493cb307e6d4423ff550910278f6434d71f00581737ce420d60298e8bc ] || exit 1
    git apply --unidiff-zero --check /opt/dsv41/ced_scheduler_replay.patch || exit 1
    git apply --unidiff-zero /opt/dsv41/ced_scheduler_replay.patch || exit 1
    grep -Fq "[CED-D] replay request=" vllm/v1/core/sched/scheduler.py || exit 1
    grep -Fq "[CED-KVRECV]" vllm/v1/core/sched/scheduler.py || exit 1
    echo APPLIED' 2>/dev/null | tail -1)
  [ "${_ced_patch:-}" = "APPLIED" ] || die "CED decode replay 调度补丁未应用"
fi
if [ "${V41_CED_ROLE:-}" = "prefill" ] && [ "${V41_CED_P_HIT_DIAG:-1}" = "1" ]; then
  say "[CED-P] 应用只读命中诊断补丁（不改分支）"
  _ced_phhit=$($DOCKER exec "$NAME" bash -lc '
    cd /vllm-workspace/vllm || exit 1
    _base=$(sha256sum vllm/v1/core/sched/scheduler.py | cut -d " " -f1)
    [ "$_base" = 533eed493cb307e6d4423ff550910278f6434d71f00581737ce420d60298e8bc ] || exit 1
    git apply --unidiff-zero --check /opt/dsv41/ced_scheduler_prefill_hit.patch || exit 1
    git apply --unidiff-zero /opt/dsv41/ced_scheduler_prefill_hit.patch || exit 1
    grep -Fq "[CED-P-HIT]" vllm/v1/core/sched/scheduler.py || exit 1
    python3 -m py_compile vllm/v1/core/sched/scheduler.py || exit 1
    echo APPLIED' 2>/dev/null | tail -1)
  [ "${_ced_phhit:-}" = "APPLIED" ] || die "CED prefill 命中诊断补丁未应用"
fi
if [ "${V41_CED_GRAPH_PROMPT_TAIL_EAGER:-0}" = "1" ]; then
  say "[CED-GRAPH] 应用固定 runner 版本的单 token prompt 尾部 eager 补丁"
  _ced_runner=$($DOCKER exec "$NAME" bash -lc '
    cd /vllm-workspace/vllm-ascend || exit 1
    _base=$(sha256sum vllm_ascend/worker/model_runner_v1.py | cut -d " " -f1)
    [ "$_base" = 67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e ] || exit 1
    git apply --check /opt/dsv41/ced_runner_prompt_tail.patch || exit 1
    git apply /opt/dsv41/ced_runner_prompt_tail.patch || exit 1
    grep -Fq "[CED-GRAPH] one-token prompt tail forced eager" vllm_ascend/worker/model_runner_v1.py || exit 1
    python3 -m py_compile vllm_ascend/worker/model_runner_v1.py || exit 1
    echo APPLIED' 2>/dev/null | tail -1)
  [ "${_ced_runner:-}" = "APPLIED" ] || die "CED prompt-tail runner 补丁未应用"
fi

# ---------- [DYNAMIC-SPEC] 运行期给 runner 打"按并发切 K"的补丁 ----------
# 顺序要求：必须在 prompt-tail 补丁**之后**打，本补丁的 sha 门对应的是
# "base 文件 + prompt-tail 补丁"之后的内容（bd250a59…），不是 base 文件本身
# （base 是 67035d97…）。这与仓库里踩过的 durian 坑同源：拿错基线会让
# `git apply --check` 失败，或者在容器里 `git checkout` 把 prompt-tail 抹掉。
if [ -n "${SP_SCHEDULE:-}" ]; then
  # [DYNSPEC-ROLE-GATE 2026-10-03] 原来只允许 CED 的 decode 角色。现放宽到
  # **空角色（独立 TP8/非 PD 形态）** 也允许 —— 动态 K 在单实例上同样成立
  # （见 docs/PLAN-DYNAMIC-SPEC-AND-FUSION-20261003.md Track A）。
  # **CED 的 prefill 角色仍然被拒**（P 侧恒关推测是架构性的，不是配置问题）。
  # 这个放宽**不放宽安全门**：下面那条 `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`
  # 的显式承担依旧强制（否则模型构造期就会炸）。
  case "${V41_CED_ROLE:-}" in
    ""|decode) : ;;
    *) die "SP_SCHEDULE（按并发切 K）只对 decode 角色或非 PD（空角色）有意义，当前 V41_CED_ROLE=$V41_CED_ROLE" ;;
  esac
  [ "$PATCH_MODE" = "mount" ] || die "SP_SCHEDULE 目前只支持 PATCH_MODE=mount（baked 镜像不含该补丁）"
  # [UPSTREAM-GUARD] 上游在 MRV1 上会**无条件**把 dynamic SD 的 cudagraph_mode
  # 降级成 PIECEWISE（vllm/config/vllm.py::_maybe_override_dynamic_sd_cudagraph_mode，
  # 理由写着"dynamic SD 会在运行时改变 target 的验证长度，为可靠性起见降级"）。
  # 而 V4.1 的 cache 只支持 eager / FULL_DECODE_ONLY
  # （core/deepseek_v41.py::validate_cache_runtime）⇒ 不处理就会在**模型构造期**
  # 炸，且报错指不到这里。
  #
  # 降级的理由是 MRV1 只有单一 query_len 的概念；本 build 的 Ascend 补丁已把
  # MRV1 补成"按本步 query_len 各建一组图"（等价 MRV2 的
  # cudagraph_utils.decode_query_lens 做法）⇒ 前提不再成立，故显式关掉它。
  # 代价 = 主动放弃一道上游保护，所以必须显式承担 + 用正确性探针验收。
  if [ "${V41_CED_DYNAMIC_SPEC_FULL_GRAPHS:-0}" != "1" ]; then
    die "SP_SCHEDULE 在当前镜像上**起不来**：上游 MRV1 会把 dynamic SD 的
      cudagraph_mode 降级为 PIECEWISE，而 V4.1 的 cache 初始化只支持
      eager / FULL_DECODE_ONLY ⇒ 约 2 分钟后在模型构造期报
      『V4.1 currently supports only eager or FULL_DECODE_ONLY』。
      出路只有一条（上游建议的 MRV2 被 V4.1 明确拒绝，见
      core/deepseek_v41.py::validate_cache_runtime）：
        显式承担风险 → V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1
      本包会用 core_config_dynamic_sd_gate.patch 关掉那道降级，其依据是
      我们已把 MRV1 补成按 query_len 建键/派发（等价 MRV2）。主要失效模式是
      **静默算错**，必须用 144K/1M 正确性探针验收。详见
      docs/CED-PD-DYNAMIC-SPEC-20260926.md §9。"
  fi
  say "[DYNAMIC-SPEC] 应用 config 侧 gate 补丁（关掉上游的 PIECEWISE 降级）"
  _dynspec_gate_applied=$($DOCKER exec "$NAME" bash -lc '
    cd /vllm-workspace/vllm || exit 1
    _base=$(sha256sum vllm/config/vllm.py | cut -d " " -f1)
    [ "$_base" = 66e82e95c5cdbb88715e25feb65f4cc2be83cd67bff5b5f703d52dea43a0815e ] || exit 1
    if grep -Fq "V41_CED_DYNAMIC_SPEC_FULL_GRAPHS" vllm/config/vllm.py; then
      echo ALREADY; exit 0
    fi
    git apply --check /opt/dsv41/ced_config_dynamic_sd_gate.patch || exit 1
    git apply /opt/dsv41/ced_config_dynamic_sd_gate.patch || exit 1
    grep -Fq "V41_CED_DYNAMIC_SPEC_FULL_GRAPHS" vllm/config/vllm.py || exit 1
    python3 -m py_compile vllm/config/vllm.py || exit 1
    echo APPLIED' 2>/dev/null | tail -1)
  case "${_dynspec_gate_applied:-}" in
    APPLIED) say "[DYNAMIC-SPEC] config gate 补丁已应用 ✓" ;;
    ALREADY) say "[DYNAMIC-SPEC] config gate 补丁已存在 ✓" ;;
    *) die "DYNAMIC-SPEC config gate 补丁未应用（$_dynspec_gate_applied）——
      sha 门不匹配或补丁冲突。**不要**带着它起服：那样会在模型构造期失败。" ;;
  esac
  say "[DYNAMIC-SPEC] 应用 runner 侧 dynamic-spec 补丁（schedule=$SP_SCHEDULE）"
  _dynspec=$($DOCKER exec "$NAME" bash -lc '
    cd /vllm-workspace/vllm-ascend || exit 1
    _base=$(sha256sum vllm_ascend/worker/model_runner_v1.py | cut -d " " -f1)
    # [DYNSPEC-BASE 2026-10-03] 原来只接受 **CED prompt-tail 之后**的 bd250a59…，
    # 于是独立 TP8（空角色、不打 prompt-tail）永远过不了这道门 —— 尽管补丁本身
    # 在两条基线上都能干净应用。已用 A/B 双路验证（见
    # docs/PLAN-DYNAMIC-SPEC-AND-FUSION-20261003.md Track A）：
    #   A 路 = prompt-tail + dynamic-spec，B 路 = 只 dynamic-spec；
    #   两者 diff 只有 25 行，且**全部是 prompt-tail 自己的改动**
    #   （import os / ced_prompt_tail_eager 18 行 / 一处 force_eager），
    #   ⇒ dynamic-spec 的 8 个 hunk 与 prompt-tail 无耦合。
    # 所以这里接受**两个**基线；仍然要求 sha 命中（镜像换版照样 fail-closed），
    # 并且应用后照旧断言可观测痕迹（[DYNAMIC-SPEC] + _v41_effective_udql + py_compile）。
    case "$_base" in
      bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff) : ;;   # CED prompt-tail 之后
      67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e) : ;;   # 原始基线（独立 TP8）
      *) echo "BASE_SHA_UNKNOWN:$_base"; exit 1 ;;
    esac
    if grep -Fq "[DYNAMIC-SPEC]" vllm_ascend/worker/model_runner_v1.py; then
      echo ALREADY; exit 0
    fi
    git apply --check /opt/dsv41/ced_runner_dynamic_spec.patch || exit 1
    git apply /opt/dsv41/ced_runner_dynamic_spec.patch || exit 1
    grep -Fq "_v41_effective_udql" vllm_ascend/worker/model_runner_v1.py || exit 1
    python3 -m py_compile vllm_ascend/worker/model_runner_v1.py || exit 1
    echo APPLIED' 2>/dev/null | tail -1)
  case "${_dynspec:-}" in
    APPLIED) say "[DYNAMIC-SPEC] runner 补丁已应用 ✓" ;;
    ALREADY) say "[DYNAMIC-SPEC] runner 补丁已存在 ✓" ;;
    BASE_SHA_UNKNOWN:*) die "DYNAMIC-SPEC runner 补丁未应用：model_runner_v1.py 的 sha 不在允许列表里（${_dynspec}）——
      镜像换版了。不要在未知基线上带着它起服；确认新基线后再扩列表。" ;;
    *) die "DYNAMIC-SPEC runner 补丁未应用（$_dynspec）。常见原因：
      ① sha 门通过但 \`git apply --check\` 失败 ⇒ 基线被别的补丁改过；
      ② patch_cudagraph.py 或 runner 补丁文件缺失。
      两种情况都不要带着它起服。" ;;
  esac
  # 效果断言：patch_cudagraph.py 也必须在位（否则 runner 传的 query_len 没人消费）
  _dynspec_pc_hit=$($DOCKER exec "$NAME" bash -lc '
    grep -c "dynamic_decode_query_lens" \
      /vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_cudagraph.py 2>/dev/null || true')
  [ "${_dynspec_pc_hit:-0}" -ge 1 ] \
    || die "patch_cudagraph.py 不是我们的 dynamic-spec 版（命中 ${_dynspec_pc_hit:-0}）⇒ query_len 只有单值，K=0 的步会错配"
  say "[DYNAMIC-SPEC] patch_cudagraph.py 在位（命中 $_dynspec_pc_hit 处）✓"
fi

# ---------- [DECODE-API-GUARD] decode 侧请求边界护栏 ----------
# 事故（2026-09-27 00:01）：18991 是 P/D 分离的 decode 半边，被一条普通请求
# 直连后自己去 prefill，撞上固定 128-token replay 的守卫并在 worker 里 raise
# ⇒ EngineCore 退出（EngineDeadError），整个 D 实例死掉、要重载 ~20 分钟。
# 护栏在 HTTP 层就把"没有 kv_transfer_params 的生成请求"判成 400，进不了引擎。
#
# 这里断言的是**可加载**（而不是"文件在"）：真正生效的判据是起服日志里
# `[V41-DECODE-GUARD] middleware loaded`，由 serve_v2.sh 打印。
if [ "${V41_CED_ROLE:-}" = "decode" ]; then
  say "[DECODE-API-GUARD] 校验 decode 侧请求边界护栏可加载"
  _dg=$($DOCKER exec "$NAME" bash -lc '
    G=/opt/dsv41/guards/v41_decode_guard.py
    [ -f "$G" ] || { echo MISSING_FILE; exit 0; }
    cd /tmp || exit 0
    PYTHONPATH=/opt/dsv41/guards python3 -c "import sys, v41_decode_guard as m; sys.exit(0 if callable(m.decode_guard) else 3)" 2>/dev/null \
      || { echo IMPORT_FAILED; exit 0; }
    echo OK' 2>/dev/null | tail -1)
  case "${_dg:-}" in
    OK) say "[DECODE-API-GUARD] 可加载（decode_guard 是 callable）✓" ;;
    MISSING_FILE) die "V41_CED_ROLE=decode 但容器里没有 /opt/dsv41/guards/v41_decode_guard.py ——
      护栏缺失意味着任何直连 decode 的请求仍能打死实例。请确认包内有 patches/files/v41_decode_guard.py。" ;;
    *) die "V41_CED_ROLE=decode 但护栏模块无法加载（$_dg）——不要带着未加固的 D 起服。" ;;
  esac
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
    # ⚠️ 这里**不能带 `|| true`**（原先有）：吞掉失败会让"幂等入口没跑起来"
    # 和"跑起来了"长得一模一样，只能靠后面的兜底分支兜住；而兜底一旦也失败，
    # 报出来的原因会指向兜底而不是真正失效的那一步（issue #2 报告者踩的正是这类）。
    # 现在：入口失败就明确打印，接着走兜底；**兜底才是最终判据**。
    if [ -f "$PKG/tools/enable_draft_graph.sh" ]; then
      if ! $DOCKER exec "$NAME" bash -lc "bash /opt/dsv41/tools/enable_draft_graph.sh on" 2>&1 | tail -3; then
        echo "[serve_a2] WARNING: enable_draft_graph.sh on 失败 ⇒ 改走下面的直接安装兜底" >&2
      fi
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
  echo "[serve_a2] port=$PORT served_name=$SERVED_NAME tp=$TP dp=$DP util=$GPU_UTIL max_len=$MAX_LEN max_seqs=$MAX_SEQS bat=$BAT_TOKENS"
  echo "[serve_a2] sptok=$SP_TOKENS capture_sizes=$CAPTURE_SIZES"
  echo "[serve_a2] MOE_AG=$MOE_AG O_PROJ_2D=$O_PROJ_2D MOE_MASK=$MOE_MASK ROPE_IDXSEL=$ROPE_IDXSEL IDS64_HOIST=$IDS64_HOIST PAD_SKIP=$PAD_SKIP ENGRAM_WKV_TP=$ENGRAM_WKV_TP"
  echo "[serve_a2] ENGRAM_JIT=$ENGRAM_JIT QLI_NOCAND=$QLI_NOCAND LOCAL_OWNER=$LOCAL_OWNER GATE_CHUNK=$GATE_CHUNK"
  echo "[serve_a2] PYTHON_PGO=$PYTHON_PGO pgo_target=${PGO_LIB:-none} STATIC_KERNEL=$STATIC_KERNEL NPUGRAPH_EX=$NPUGRAPH_EX"
  echo "[serve_a2] LOAD_FORMAT=${LOAD_FORMAT:-<real weights>} MOE_ZERO=$MOE_ZERO(unverified) MOE_NF=$MOE_NF(negative-result) DRAFT_GRAPH=$DRAFT_GRAPH(unverified)"
  echo "[serve_a2] 口径：MAX_SEQS=$MAX_SEQS PREFIX=$PREFIX(0=性能口径/1=生产口径) capture_max=${CAPTURE_SIZES##*,}"
  echo "[serve_a2] 诊断项：HCCL_DET=${HCCL_DET:-none}（true 会掉 GSM8K 到 91/100，仅诊断）"
  echo "[serve_a2] cpuset=$CPUSET mems=$MEMS"
  echo "[serve_a2] PROFILE=$V41_PROFILE（1 => /start_profile 与 /stop_profile 可用，落到 $OUT/prof）"
  echo "[serve_a2] ENGRAM_DEVICE_INDEX=$ENGRAM_DEVICE_INDEX ENGRAM_DEVICE_FALLBACK=$ENGRAM_DEVICE_FALLBACK"
  echo "[serve_a2] KV_ARGS_EXTRA=${KV_ARGS_EXTRA:-<none>}"
  echo "[serve_a2] SP_SCHEDULE=${SP_SCHEDULE:-<fixed K=$SP_TOKENS>}（按并发切 K；空=固定）"
  echo "[serve_a2] PATCH_MODE=$PATCH_MODE ADMISSION_GATE=${_gate:-n/a}(live_hits=${_gh:-0})"
} | tee "$OUT/serve_cmd.txt"

INNER=/opt/dsv41/results/$RUN_ID/inner.sh
cat > "$OUT/inner.sh" <<INNER_EOF
#!/usr/bin/env bash
set -uo pipefail
cd /workspace
export MODEL="$MODEL" TP=$TP DP=$DP PORT=$PORT SERVED_NAME="$SERVED_NAME"
export MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS BAT_TOKENS=$BAT_TOKENS GPU_UTIL=$GPU_UTIL BLOCK=$BLOCK
export QUANTIZATION=${QUANTIZATION:-ascend} KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-} SEED=${SEED:-}
export KV_DTYPE=$KV_DTYPE GRAPH=$GRAPH EAGER=$EAGER PREFIX=$PREFIX SPEC=$SPEC SP_TOKENS=$SP_TOKENS
if [ "$DRAFT_GRAPH" = "1" ]; then export SPEC_EAGER=0; else export SPEC_EAGER=1; fi
export ENGRAM=$ENGRAM ENGRAM_STORAGE=int8 VISION=$VISION
export MM_LIMIT_IMAGES=$MM_LIMIT_IMAGES HOST=$HOST
export NPUGRAPH_EX=$NPUGRAPH_EX STATIC_KERNEL=$STATIC_KERNEL CPU_BIND=$CPU_BIND
export MULTISTREAM=$MULTISTREAM DSA_OVERLAP=$DSA_OVERLAP FUSED_MC2=$FUSED_MC2 MC2=$MC2 MC2_HIER=$MC2_HIER REDUCE_SAMPLE=$REDUCE_SAMPLE
# ★ [DUMB-KNOBS 2026-10-05] 这 5 个开关 serve_v2.sh 会读、但本脚本从不设置 ⇒ 从启动器设它们
#   是静默无效的（实测：FORCE_EPLB=1 起的臂，additional-config 里根本没有 enable_force_eplb，
#   于是那次 A/B 其实是同配置对同配置）。这里补齐透传；默认值与 serve_v2.sh 的内建默认一致。
#   ⚠️ 本 heredoc 不加引号：**禁止在此块内使用反引号或 $() 形式**（会被当命令替换执行，set -e 下会毁掉 inner.sh）。
export FORCE_EPLB=${FORCE_EPLB:-0} DSA_CP=${DSA_CP:-0} ENGRAM_HOST_RESTORE=${ENGRAM_HOST_RESTORE:-0}
export MC2_ALG=${MC2_ALG:-} WEIGHT_NZ=${WEIGHT_NZ:-}
export LOADER_MT=1 LAZY=1
export V41_KV_TIER=off
export V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner
export CAPTURE_SIZES="$CAPTURE_SIZES"
export ASCEND_MAX_OP_CACHE_SIZE=-1
# [PROFILE] 透传 profiler 开关；PROFILE_DIR 指向本次 run 的结果目录（宿主可见），
# 这样 /stop_profile 一落盘就能直接分析，不用再 docker cp。
export PROFILE=$V41_PROFILE
export PROFILE_DIR=/opt/dsv41/results/$RUN_ID/prof
export KV_ARGS_EXTRA="\${KV_ARGS_EXTRA:-}"
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
# [OPP-OVERRIDE 2026-10-04] 把自定义 vendor 复制覆盖到**镜像 vendor 路径**，再起服务。
# 必须这样做：vllm_ascend.utils.bootstrap_custom_op_env() 会把镜像自带 vendor 路径**前插**
# 到 ASCEND_CUSTOM_OPP_PATH（utils.py:323-332）⇒ 只设 env 时镜像内同名 kernel 优先，
# 改 kernel 不生效（实测 profile 逐字段相同）；而 -v 覆盖挂载（:ro）会让内核启动期报
# aicore exception/IndexCheck 507015。所以用"起服前复制"。
# ★ 注入点刻意放在生成之后（不在 heredoc 内）：往该 heredoc 里插任何块都会让生成的
#   inner.sh 变成 0 字节（实测复现 3 次，原因未查明），放在这里完全避开。
if [ -n "${V41_HC_OPP_PKG:-}" ] && [ -f "$PKG/patches/opp_override_block.sh" ]; then
  _blkf=$(mktemp)
  cat "$PKG/patches/opp_override_block.sh" > "$_blkf"
  awk -v blk="$(cat "$_blkf")" '
    /^exec bash \/opt\/dsv41\/scripts\/serve_v2\.sh/ && !done {print blk; done=1}
    {print}' "$OUT/inner.sh" > "$OUT/inner.sh.tmp" \
    && mv "$OUT/inner.sh.tmp" "$OUT/inner.sh"
  mv "$_blkf" "$_blkf.used" 2>/dev/null || true
  say "[OPP-OVERRIDE] 已注入 inner.sh 覆盖块（$V41_HC_OPP_PKG/vendors/custom_transformer）"

  # ★★ [SKCACHE-STALE 2026-10-04] 换 kernel 时**必须**同时让 static kernel 缓存失效，
  # 否则运行时可能复用旧内核 ⇒ 改动静默不生效（实测：HcPre A1 的 .o 已在容器里、
  # md5 已校验，但服务执行的仍是旧内核 —— 用 `aic_mac_time` 指纹 1.282 vs 1.282 判定）。
  # 为什么单靠 .o 不够：static kernel 的缓存 key **不含被替换 .o 的内容**
  # （文件哈希只由 op 定义决定），所以换 vendor 里的 .o 不会让它失效。
  # 自动清 13GB 缓存会让起服多花 ~18 分钟（重编译），所以默认只**响亮警告**；
  # 做 kernel A/B 时请显式 `V41_OPP_CLEAR_SKCACHE=1`。
  if [ "${V41_OPP_CLEAR_SKCACHE:-0}" = "1" ]; then
    for _d in "$CACHE/skcache/compile_outputs" "$CACHE/skcache/install"; do
      if [ -d "$_d" ]; then
        _ts=$(date +%Y%m%d_%H%M%S)
        mv "$_d" "${_d}.stale_$_ts" 2>/dev/null && say "[OPP-OVERRIDE] 已让缓存失效：$_d → ${_d}.stale_$_ts"
      fi
    done
    say "[OPP-OVERRIDE] static kernel 缓存已失效 ⇒ 本次会重编译（起服约 +18 min）"
  else
    echo "[serve_a2][OPP-OVERRIDE] ⚠️  static kernel 缓存**未清**（cache/skcache/compile_outputs）"
    echo "[serve_a2][OPP-OVERRIDE]     若本次改了 kernel，运行时可能**复用旧内核**导致改动静默不生效。"
    echo "[serve_a2][OPP-OVERRIDE]     kernel A/B 请改用 V41_OPP_CLEAR_SKCACHE=1。"
    echo "[serve_a2][OPP-OVERRIDE]     起服后必须验执行：用资源计数指纹（aic_mac_time / cycles /"
    echo "[serve_a2][OPP-OVERRIDE]     目标算子 Duration）与基线比对，看不到预期变化即判"内核没换"。"
  fi
fi
# [OPP-OVERRIDE-SAFETY 2026-10-04] 自定义 kernel 若在**图捕获/重放**下不兼容，症状是
# 服务"看起来在启动"但永远不 ready（EngineCore 每 60s 报 shm broadcast 超时），
# 实测一次 35 分钟才被默认 READY_TIMEOUT 收掉，且现场没有指向 kernel 的线索。
# 这里：只要启用了 OPP-OVERRIDE 就把 ready 上限收到 10 分钟，并在失败信息里点名。
if [ -n "${V41_HC_OPP_PKG:-}" ]; then
  # ★ 2026-10-04 修正：原先这里把上限收到 600s，**把一次合法的启动杀掉了** ——
  # 实测"静态内核编译(≈8min) + 121 桶图捕获(≈10min)"合法地超过 600s，
  # 于是服务在 100% 捕获完成后被 die() 清掉，日志里只有一句"等待超时"。
  # 教训：**安全闸不能短于合法的最慢路径**。改为 1800s（仍短于默认 2100s，
  # 真·挂死的 kernel 会更快失败），并在失败信息里点名 kernel 是第一嫌疑。
  : "${V41_OPP_READY_TIMEOUT:=1800}"
  READY_TIMEOUT="$V41_OPP_READY_TIMEOUT"
  echo "[serve_a2][OPP-OVERRIDE] ⚠️ 已启用自定义 kernel（$V41_HC_OPP_PKG）"
  echo "[serve_a2][OPP-OVERRIDE]    ready 上限 ${READY_TIMEOUT}s（覆盖静态编译+图捕获）；"
  echo "[serve_a2][OPP-OVERRIDE]    若仍起不来，第一嫌疑是该 kernel（回退：不设 V41_HC_OPP_PKG）"
fi
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
  # ⚠️ 不能写 `|| echo 000`：curl 失败时 -w **已经**打印了 000，再追加一个
  # 就变成 "000000"，后续所有 `= "200"` 比较都失效（AGENTS.md §3.2 同族坑）。
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  code=${code:-000}
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
# ---------- 起服必查 ②：KV32 池上界复核（只在"交回 vLLM 自动 profiling"时） ----------
# 为什么需要：放开 pin 之后，非 CED 形态的池大小由 vLLM 的显存 profiling 决定。
# 若它恰好算出 > 29076 块，长上下文会静默变成"HTTP 200 + 1 token（EOS）"
# （见 docs/CED-PD-BLOCK-BOUND-20260925.md §0/§5.1.2）。这一步把那条路也堵上。
#
# 判据来源（**已用仓库内证据校准**）：profiling 路径会打印
#   v1/worker/gpu_worker.py: "Available KV cache memory: %s GiB"（format_gib = round(b/GiB,2)）
# 取**所有 rank 的最小值** —— vLLM 的最终 num_blocks 正是各 rank 取 min
# （v1/core/kv_cache_utils.py: "Change the num_blocks of each rank to the smallest"）。
# 例：A2 历史 profiling 14.40 GiB ⇒ ⌊14.40×2³⁰/540928⌋ = 28583 块，
# 与文档记载的 28,577 块相差 7（日志只保留 2 位小数 ⇒ 估算误差 ≤ ±10 块），
# 离上界 29076 还有 492 块余量，判据可用（越界现场是 30080 vs 29076，差 1004）。
#
# ★ 抽成函数：它是**纯文本逻辑**，抽出来才能离线自检
#   （tools/selftest_kv32_scope.sh 用合成日志正/负控），不必占 8 张卡起服。
kv32_pool_blocks_from_log() {   # <logfile> <bytes_per_block> → 打印最小 rank 的块数；无数据返回 1
  local _log=$1 _bpb=$2 _g
  _g=$(grep -aoE 'Available KV cache memory: [0-9.]+ GiB' "$_log" 2>/dev/null \
       | awk '{print $(NF-1)}' | sort -g | head -1)
  [ -n "$_g" ] || return 1
  awk -v g="$_g" -v bpb="$_bpb" 'BEGIN{printf "%d", (g*1073741824)/bpb}'
  return 0
}
# --- [KV32] 辅助函数结束（selftest 按这两行标记抽取本函数）---

if [ "$_kv32_enforce" = "1" ] && [ "$_kv32_pinned" != "1" ] && [ "$_kv32_user_set" != "1" ]; then
  _kv32_avail=$(grep -aoE 'Available KV cache memory: [0-9.]+ GiB' "$LOG" 2>/dev/null | awk '{print $(NF-1)}' | sort -g | head -1 || true)
  # ★★ [KV32-DCP 2026-10-05] DCP 形态下**不要**用固定 bytes_per_block 估算：
  # DCP 把每个 KV group 按 decode_context_parallel_size 路分片 ⇒ 每块字节数变小，
  # 用非 DCP 的常数会**假通过**。实测（DCP8，GPU_UTIL=0.85）：真值 92,363 B/块，
  # 而常数 540,928 ⇒ 估出 22,863 块（<= 上界 29,076，放行），实际 133,924 块（5.9×）。
  # 修法：**优先用 vLLM 自己打印的 `GPU KV cache size: <N> tokens` 反算块数**（精确，
  # 不依赖任何常数）；只有拿不到那行时才退回常数估算（并明确标注是估算）。
  _kv32_tokens=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9,]+' | tr -d ',' | head -1)
  if [ -n "${_kv32_tokens:-}" ] && [ "${BLOCK:-0}" -gt 0 ] 2>/dev/null; then
    _kv32_blocks=$(( _kv32_tokens / BLOCK ))
    # ⚠️ [KV32-DCP 2026-10-05] 这里**只报告精确块数，不改上界**。
    # 为什么不去推"本配置自己的页步长"：`Available KV cache memory` 是**所有 cache group
    # 的总预算**，而回绕判据用的是**单个绑定组**的页步长（CED 文档：147712 B）——
    # 两者不同源，用前者除总块数会得到错误的页步长（实测推导值与已知常数 540,928/147712
    # 都不吻合）⇒ 那样改会引入新的误判。**上界仍沿用 _ced_max_blocks（已由 CED 文档校准）**，
    # 但块数现在是**精确值**（以前是估算），所以 DCP 形态下的偏差会被如实报出来（供人工判断）。
    _kv32_src="精确块数（GPU KV cache size ${_kv32_tokens} tokens ÷ block ${BLOCK}）"
  elif [ -z "$_kv32_avail" ]; then
    echo "  [KV32] 复核跳过：日志里既没有 'GPU KV cache size' 也没有 'Available KV cache memory'"
    _kv32_blocks=0
    _kv32_src=""
  else
    _kv32_blocks=$(kv32_pool_blocks_from_log "$LOG" "$_ced_bytes_per_block" || echo 0)
    _kv32_src="估算（${_kv32_avail} GiB ÷ ${_ced_bytes_per_block} B/块）⚠️ DCP 形态下此估算偏小"
  fi
  if [ -n "${_kv32_src:-}" ]; then
    echo "  [KV32] 池大小来源：$_kv32_src"
  fi
    # ★★ [KV32-DCP-WARN 2026-10-05] DCP 形态下**只警告不拦截**：
    # 上界 29076 是用 **CED 的单组页步长 147712 B** 校准的，对 DCP 不适用 ——
    # **实测**（DCP8，4 条并发 × 960K = 3,840,166 tokens ≈ 30,001 块）**超过 29,076 仍全部正确**，
    # 说明 DCP 的页步长更小、真实上界更高（按 92,363 B/块 推 ≈46,498 块）。
    # 但 DCP 的真实上界**尚未实测确定** ⇒ 不静默放行，也不误拦已证可用的配置，
    # 而是**响亮警告**并打印精确块数与已验证的观测上限。
    _kv32_is_dcp=0
    case "${KV_ARGS_EXTRA:-}" in *--decode-context-parallel-size\ *) _kv32_is_dcp=1 ;; esac
    if [ "${_kv32_blocks:-0}" -gt "$_ced_max_blocks" ] && [ "$_kv32_is_dcp" = "1" ]; then
      echo
      echo -e "  \033[33m⚠️  [KV32] DCP 形态：池 ${_kv32_blocks} 块 > 非 DCP 上界 ${_ced_max_blocks}（$_kv32_src）\033[0m"
      echo "    该上界由 **CED 单组页步长 147712 B** 校准，**对 DCP 不适用**：实测 4 条并发 960K"
      echo "    （≈30,001 块）**超过 29,076 仍 4/4 正确** ⇒ DCP 的页步长更小、真实上界更高"
      echo "    （按 92,363 B/块 推 ≈46,498 块）。**但 DCP 真实上界尚未实测确定。**"
      echo "    ⇒ 若要把池用满到 >30,001 块，请先跑长上下文边界测试；否则按实测上限使用。"
      echo "    已知风险症状：块号回绕 ⇒ HTTP 200 + 1 token（EOS）的**静默**空答。"
      echo "    关闭本警告：V41_KV32_POOL_GUARD=off（同时关掉 pin/clamp/本复核）"
    elif [ "${_kv32_blocks:-0}" -gt "$_ced_max_blocks" ]; then
      echo
      echo -e "  \033[31m✗ [KV32] KV 池超出 4 GiB 寻址上界：${_kv32_blocks} 块 > 上界 ${_ced_max_blocks}（$_kv32_src）\033[0m"
      echo "    该配置下块号 ≥ ${_ced_max_blocks} 的访问会 32 位回绕，长上下文请求会**静默**变成 1 token（EOS）。"
      echo "    处置（任选其一）："
      echo "      1) 显式压池：KV_CACHE_MEMORY_BYTES=$(( _ced_max_blocks * _ced_bytes_per_block ))"
      echo "      2) 降低显存利用率：GPU_UTIL 调小后重跑"
      echo "      3) 确知风险仍要跑：V41_KV32_POOL_GUARD=off（会同时关掉 pin/clamp/本复核）"
      die "[KV32] 拒绝以越界池起服（避免长上下文静默空答）"
    fi
    if [ -n "${_kv32_src:-}" ] && [ "${_kv32_blocks:-0}" -le "$_ced_max_blocks" ]; then
      echo "  ✓ [KV32] 池上界复核：${_kv32_blocks} 块 ≤ 上界 ${_ced_max_blocks}（用到 $(( _kv32_blocks * 100 / _ced_max_blocks ))%；来源见上）"
    elif [ "${_kv32_blocks:-0}" -gt "$_ced_max_blocks" ]; then
      echo "  ⚠️  [KV32] 池上界复核：${_kv32_blocks} 块 **超过** 上界 ${_ced_max_blocks}（用到 $(( _kv32_blocks * 100 / _ced_max_blocks ))%）—— 见上方警告/拒绝说明"
    fi
fi
# 镜像指纹落到结果目录（make_report.sh 会读它）
$DOCKER exec "$NAME" bash -lc 'cat /opt/dsv41/BUILD_INFO.txt 2>/dev/null' > "$OUT/BUILD_INFO.txt" 2>/dev/null || true
$DOCKER exec "$NAME" bash -lc 'cat /opt/dsv41/BUILD_INFO.txt 2>/dev/null' | sed 's/^/  /' || true
echo "$RUN_ID" > "$PKG/.last_run_id"
echo "RUN_ID=$RUN_ID OUT=$OUT LOG=$LOG"
