#!/usr/bin/env bash
# =============================================================================
# serve_a2_offload.sh —— A2（8×910B3）DRAM KV 卸载一键起服
#
# 设计原则：**不改** dsv41-release/scripts/serve_a2.sh（那是生产脚本）。
#
# 依赖 **shadow-pkg**（一份把 serve_a2.sh 打了两处注入补丁的副本）：
#   * 它认 `KV_ARGS_EXTRA`（把卸载参数带进容器）
#   * 它认 `OFFLOAD_SCHED_PATCH` / `OFFLOAD_NPU_WORKER_PATCH`（挂两个补丁）
# 本脚本把 `a2/patches/` 的四个文件复制进 shadow-pkg 的补丁目录，再调它的 serve_a2.sh。
#
# 用法（默认 = A2 生产口径：OFFLOAD_GB=85 MAX_LEN=1048576 MAX_SEQS=4）：
#   bash a2/scripts/serve_a2_offload.sh
#   MAX_LEN=131072 MAX_SEQS=16 OFFLOAD_GB=56 bash a2/scripts/serve_a2_offload.sh   # 回到 128K 场景
#
# ★★ profiler（**透传，已实测**）：`PROFILE=1` 或 `V41_PROFILE=1` 都可以 ——
#   模板读的是 `V41_PROFILE=${V41_PROFILE:-${PROFILE:-0}}`（`serve_a2.sh:188`），
#   本脚本的 `_launch_serve()` 会把这一个值**同时显式传给两个名字**（不再靠环境继承，见 logs/112）。
#   开了之后 `/start_profile` 与 `/stop_profile` 可用，产物落 **宿主可见** 的
#     $LAUNCH_DIR/results/<RUN_ID>/prof        （LAUNCH_DIR 默认 = shadow-pkg）
#   例： PROFILE=1 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 KV8_SWA=1 KV8_RING_FP16=1 \
#          MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
#   ★ 采完必须在**容器内** msprof --export=on，产物属主 root ⇒ 宿主侧分析前先 sudo chown -R
#     （见 a2/logs/128 的两条踩坑记录）
#
# ★ 起服后**必须先跑自检**（脚本末尾会打印命令），否则可能白等 20 分钟。
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2DIR=$(cd "$HERE/.." && pwd)            # a2/ 自己（补丁可能在 a2/patches 或 a2/publish）
REPO=$(cd "$HERE/../.." && pwd)          # dsv41-release/

# ---------------------------------------------------------------- shadow / 起服对象
# ★★★ 2026-09-23 09:4x 修一个 `set -u` 崩溃（用户实测）：
#   这三个定义**原本在文件后半**（约 355 行），但头部打印（约 247 行）已经引用 `$LAUNCH_DIR`
#   ⇒ `set -u` 下直接 `line 247: LAUNCH_DIR: unbound variable`。
#   ★ 这是**同一天第二次**犯同一个错（上一次是同一行里的 `$OUT`，那次我改成了字面路径就以为好了
#     —— **没有从根上修**）。⇒ 现在把定义**统一提到参数区**，并用 §6 的沙箱测试兜住这类错。
SHADOW=${SHADOW_PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
PATCHDIR="$SHADOW/patches/files/offload_dsv41"
LAUNCH_DIR=${LAUNCH_DIR:-$SHADOW}       # 起服对象（默认 shadow-pkg；改它可做 A/B 对照）
_SV="$LAUNCH_DIR/scripts/serve_a2.sh"

# ---------------------------------------------------------------- 参数
# ★ 模型路径不硬编码（发布包不留真实账号名）；必须由调用方给
MODEL=${MODEL:?请设 MODEL=<模型目录>}
# ★★★ 2026-09-23 09:1x 修一个真 bug：这里原来是 `IMAGE=${IMAGE:-}`（**空**）
#   ⇒ ①「起服前指纹门」拿**空字符串**去 `docker image inspect ""` ⇒ 必然报"本地没有镜像 "
#         而且报错里镜像名是空的（用户实测就是这个形态）；
#      ② 更隐蔽：空值传给 shadow 的 `serve_a2.sh` 时，`${IMAGE:-dsv41-a2:v9}` 会**用默认值**
#         ⇒ 门拦住的理由是"名字空"而不是"镜像旧" —— 哪天门被跳过，就会**静默用默认镜像**。
#   ⇒ 与 `scripts/serve_a2.sh:44` 的默认**对齐**（那里是唯一权威默认，见 commit 550d29c）。
IMAGE=${IMAGE:-dsv41-a2:v9}
GPU_UTIL=${GPU_UTIL:-0.90}
PORT=${PORT:-8077}
SERVED_NAME=${SERVED_NAME:-deepseek-v4-flash}

# ---------------------------------------------------------------- ★ 档位
# 档 A（默认，8 卡真权重已验，logs/027）：
#   128K x 16 并发 -> OFFLOAD_GB=56（57,344 unit = 实测需求 48,064 的 1.193x）
#                     宿主实占 392.4 GiB（A2 余量 442 GiB 的 88%，⚠️ 紧）
# 档 B（+L1，内存砍半）：加 P2_POOL_PATCH=1 + P2_COMP_JSON=<见 §2.2>
#                     同样 OFFLOAD_GB=56 -> 宿主约 203.6 GiB（45%）
#   32K x 16 并发  -> OFFLOAD_GB=10
# ★★★ 2026-09-23 09:2x 默认值对齐 **A2 生产**（用户要求 maxlen 改 1M）
#   生产配置（`a2/docs/A2-ENGRAM-PATHS.md:172` 逐字，实测 3,498,354 tokens）：
#       OFFLOAD_GB=85 MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=2048
#   两个独立理由支持 1M 是**默认**（不是"为了测大上下文才开"）：
#     ① `scripts/serve_a2.sh:87` 的默认本来就是 `MAX_LEN=1048576` —— 本包装脚本此前是 131072，
#        属于"两处默认不一致"（与刚修的 IMAGE 空默认值同一类）；空/短默认会让门拦错的理由。
#     ② ★ **1M 反而多买一倍 KV**：`GPU KV cache size = num_blocks / BPR × max_len`，而 BPR 含
#        **每请求固定开销**（10 个 SWA 窗口块 + draft 块）：
#          max_len=133120 : BPR=2471  vs 真实 token 块 1040  ⇒ 开销系数 2.376
#          max_len=1048576: BPR=9623  vs 真实 token 块 8192  ⇒ 开销系数 1.175
#        ⇒ 同样 4 GiB 预算，1M 下能买的 token 数 ≈ 128K 下的 **2.02×**（A2 文档实测）。
OFFLOAD_GB=${OFFLOAD_GB:-85}
MAX_LEN=${MAX_LEN:-1048576}
MAX_SEQS=${MAX_SEQS:-4}
BAT_TOKENS=${BAT_TOKENS:-2048}

# ★★★ 2026-09-23 10:1x 补一个**会静默关掉四轴之一的默认值**：`DRAFT_GRAPH`
#   事实（三条，都能复算）：
#     ① 模板 `scripts/serve_a2.sh:230` 的默认是 **0**（`DRAFT_GRAPH=${DRAFT_GRAPH:-0}`）；
#     ② ★ 而**同一份模板的第 11 行**写的是「* 默认开 DRAFT_GRAPH=1（draft 入图，含
#        DSPARK_GRAPH_CAPTURE_METADATA 绑定与校验）」⇒ **模板内部自相矛盾**；
#     ③ A2 **生产**用的是 **1**（`a2/docs/A2-ENGRAM-PATHS.md:172` 逐字），且 A2 上实测
#        **draft 入图 = 唯一的大杠杆**：单流 **54.7 → 88.7 tok/s（+62%）**、`[bneck] hp`
#        **64.6–65.3 → 34.0–34.8（−47%）**、稳态 **A=3.03**（健康区间 2.8–3.1）
#        —— 见 `reports/a2-draft-graph-20260920.md`（A2 真机、`ENGRAM_DEVICE_INDEX=0` 同口径）。
#   ⇒ 本包装脚本此前**完全没提 DRAFT_GRAPH**（grep 命中 0）⇒ 不显式传就**静默退回 eager**：
#     四轴变三轴，而 `ms/step` 会从 ~34 回到 ~65 —— 却没有任何报错。
#     A2-DEPLOY-NOW.md §「起服前必读」早就点名过这件事（"照抄会把 draft 从入图退回 eager"）。
#   ⇒ 默认取 **1**（= A2 生产 = 四轴目标）；要退回 eager 必须**显式** `DRAFT_GRAPH=0`（会响亮警告）。
DRAFT_GRAPH=${DRAFT_GRAPH:-1}

# ★ L1（池张量按需分配行数）—— ★★ 8 卡真权重实测 1.9895x（392.35 -> 197.21 GiB）
#   defaults 到这里 = 档 B（推荐）；置 0 即回档 A
#   P2_POOL_PATCH=1 时必须同时给对的 P2_COMP_JSON（与"张量数"匹配，给错会 fail-closed）
P2_POOL_PATCH=${P2_POOL_PATCH:-1}
# 16 张量几何的【实测·worker 侧真值】分量（8 卡真权重，8 rank x 9 行一致）
P2_COMP_JSON=${P2_COMP_JSON:-'[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]'}

# ★★★ 2026-09-23 10:2x **per-group `blocks_per_chunk`（dict 形式）真正生效的那条路**
#   实测事故：A2 起服在**模型加载完之后**崩，栈底是
#       File ".../kv_connector/v1/offloading/config.py", line 78, in build_offloading_config
#           blocks_per_chunk = int(blocks_per_chunk_config)
#       TypeError: int() argument must be a string, a bytes-like object or a real number, not 'dict'
#   根因：我们挂了 `pgp_hooks.py`（一个**运行期 monkeypatch**），但**全仓没有任何人 import 它**
#     ⇒ `build_offloading_config` 保持镜像内原样 ⇒ 见到 dict 就 `int(dict)` 崩。
#     （`grep -rn "import pgp_hooks"` 全仓 0 命中 —— 这个洞一直都在。）
#   修法：改用 **A3 8 卡臂一直在用、有实测** 的那条路 —— **整份替换 6 个文件**
#     （`L1_POOL_PATCH=1`）：除 per-group bpc 外，**顺带给出 L1（池按需行数）**。
#     那 6 份已随包（`a2/patches/kv8-offload-pool/`，与 A3 的 `manifest.md5` 逐字一致）。
#   ⇒ 默认开；要退回旧的 monkeypatch 路线（**已知会在 dict 下崩**）才显式置 0。
L1_POOL_PATCH=${L1_POOL_PATCH:-1}
L1_POOL_DIR=${L1_POOL_DIR:-$A2DIR/patches/kv8-offload-pool}

# ★ [DROPCACHE] 起服前清 page cache（模板默认 **1**）。**整机**生效，会连带清掉同机其它租户的
#   page cache（`refresh pattern`）。大内存机器上它通常值（本机实测一次能放 564 GiB），
#   但如果这台机器不是你独占、或你不想影响别人 ⇒ `DROPCACHE=0` 关掉。
DROPCACHE=${DROPCACHE:-1}

# ★ 池分配器加固（logs/041）：默认关；=1 把三种静默失败变成响亮 raise
PGP_MGR_HARDEN=${PGP_MGR_HARDEN:-0}
PGP_MGR_STATS=${PGP_MGR_STATS:-0}

# ---------------------------------------------------------------- ★ int8 档（logs/047）
# 档 C（容量 ×1.4655）：SWA 页 INT8 + ring16 + APC 对齐
# 档 D（容量 ×1.9133）：再加 KV8 双平面 + prefill 融合
# ★ 前置：VLLM_V41_APC_ALIGN=3 是它们能正确工作的前提（否则 D/F 几何会翻 token）
#
# ★★★ 2026-09-22 16:5x **防"用内部名当外部接口"**（这是本日第 4 次同类静默失败，见 logs/065 §3/§3b.0）
#   本脚本有两套名字，方向是**单向**的：
#       用户接口（本文件读）  : KV8_SWA / KV8_RING_FP16 / KV8_FULL / KV8_PREFILL / APC_ALIGN / GRAPH_SAFE
#       内部名（本文件写出去）: A2_KV8_SWA / A2_RING_FP16 / A2_KV8 / A2_KV8_PREFILL / A2_APC_ALIGN / A2_GRAPH_SAFE
#                              ↑ 由本文件末尾 export 给 shadow 包的挂载块
#   ⇒ 若用户**在外面传 `A2_*`**，本文件会读不到它（读到默认 0），随后**用 0 覆盖它**：
#       现象【实测】：档位自报 **B**（不是 C）、`APC_ALIGN=0`（会翻 token）、`GRAPH_SAFE=0`（图模式捕获期炸 EE1016）；
#         而 shadow 的挂载块**可能照样挂上 7 件**（若同时传了 `A2_RING_FP16`）⇒ 看起来"int8 开了"，
#         实际两个致命开关都是 0。**比不挂更危险。**
#   ⇒ 因此：**只要检测到用户传了任一 `A2_*`，直接 fail-closed（exit 64）**，并要求改用 `KV*` 那套。
for _v in A2_KV8 A2_KV8_SWA A2_RING_FP16 A2_KV8_PREFILL A2_APC_ALIGN A2_GRAPH_SAFE A2_KV8_GRAPHSAFE; do
    if [ -n "${!_v:-}" ]; then
        echo "⛔ 检测到你在外面设了内部变量 ${_v}=${!_v} —— 这是本脚本**向下**翻译给 shadow 用的名字，不是用户接口。" >&2
        echo "   ⇒ 本脚本会读不到它、并用默认 0 覆盖 ⇒ 档位会被误判成 B、APC_ALIGN/GRAPH_SAFE 都不生效" >&2
        echo "      （现象：int8 文件挂上了，但会翻 token / 图模式捕获期炸）。" >&2
        echo "   ⇒ 请改用用户接口：KV8_SWA / KV8_RING_FP16 / KV8_FULL / KV8_PREFILL / APC_ALIGN / GRAPH_SAFE" >&2
        echo "      例：KV8_SWA=1 KV8_RING_FP16=1 bash a2/scripts/serve_a2_offload.sh" >&2
        exit 64
    fi
done

KV8_SWA=${KV8_SWA:-0}        # 1 = SWA 页 INT8（档 C 起）
KV8_RING_FP16=${KV8_RING_FP16:-0}  # 1 = state ring FP32→FP16（档 C 的必需前置）
KV8_FULL=${KV8_FULL:-0}      # 1 = long-KV 也 INT8（档 D）
KV8_PREFILL=${KV8_PREFILL:-0}  # 1 = prefill 融合 kernel（档 D）
# ★ APC 对齐：0 = 旧行为；3 = 段栅格（推荐，档 C/D 必开）
APC_ALIGN=${APC_ALIGN:-0}

# ★★ int8 的图安全补丁（logs/049 / patches/kv8-graphsafe/）
#   不打开 ⇒ 档 C/D 在 FULL_DECODE_ONLY 下【捕获期直接炸】（EE1016）
#   ★ 前提：挂上 patches/kv8-graphsafe/dsa_v41.py（md5 **7867da2a**… = chunkview 修复版；
#     ★ **94aeebb7 是未修版** —— 在 A2 的 85 GiB 池上会付 ≈125 ms/step，见 logs/106/121）
GRAPH_SAFE=${GRAPH_SAFE:-0}

# ★ 开了 int8 就必须同时开 APC 对齐（否则 D/F 几何会翻 token，logs/047）
if { [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; } \
   && [ "$APC_ALIGN" = "0" ]; then
    echo "⚠⚠ 你开了 int8 但 APC_ALIGN=0 ⇒ 自动置 3（D/F 几何否则会翻 token，logs/047）" >&2
    APC_ALIGN=3
fi

# ★ 开了 int8 + 图模式（GRAPH=1）就必须同时开 GRAPH_SAFE
if { [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; } \
   && [ "$GRAPH_SAFE" = "0" ] && [ "${GRAPH:-1}" != "0" ]; then
    echo "⚠⚠ 你开了 int8 + 图模式但 GRAPH_SAFE=0 ⇒ 自动置 1" >&2
    echo "   （否则档 C/D 在 FULL_DECODE_ONLY 下捕获期会炸 EE1016，logs/049）" >&2
    echo "   前提：已挂 patches/kv8-graphsafe/dsa_v41.py（md5 **7867da2a**… = 修复版）" >&2
    GRAPH_SAFE=1
fi

# ★★★ 2026-09-22 16:4x：**档位判定 + 未验证组合 fail-closed**
#   起因：T_draftceiling 的 8 卡「档 D」臂因为 runner **少挂了一个 model.py**，
#         实际跑的是档 C —— 而**没有任何报错**。它的容量读数 427,643（档 C 的数）
#         与它自称的档 D（应 485,610）矛盾，是靠这条才对出来的。
#   => 本脚本现在：① **自报档位**；② 拒绝**未验证的组合**（宁可拒绝，不要静默跑成别的档）。
#
#   已验证的档位（8 卡真权重实测的 HBM 容量指纹）：
#     档 B : SWA=0 RING=0 FULL=0            => 427,643
#     档 C : SWA=1 RING=1 FULL=0            => 427,643（★ 与档 B **相同**，容量区分不了 B/C）
#     档 D : SWA=1 RING=1 FULL=1 PREFILL=1  => 485,610
#   ★ 所以：**容量只能区分「档 D」与「非档 D」** —— 若你预期 485,610 却拿到 427,643，
#     那就是**静默降档**（零件没挂上、或某个开关没生效）。
_tier=B
if [ "$KV8_SWA" = "1" ] && [ "$KV8_RING_FP16" = "1" ] && [ "$KV8_FULL" = "0" ]; then _tier=C; fi
if [ "$KV8_SWA" = "1" ] && [ "$KV8_RING_FP16" = "1" ] && [ "$KV8_FULL" = "1" ]; then _tier=D; fi
_tier_ok=1
if [ "$KV8_SWA" = "1" ] && [ "$KV8_RING_FP16" = "0" ]; then _tier=UNVERIFIED-swa-without-ring; _tier_ok=0; fi
if [ "$KV8_FULL" = "1" ] && [ "$KV8_SWA" = "0" ]; then _tier=UNVERIFIED-full-without-swa; _tier_ok=0; fi
if [ "$KV8_PREFILL" = "1" ] && [ "$KV8_FULL" = "0" ]; then _tier=UNVERIFIED-prefill-without-full; _tier_ok=0; fi
if [ "$_tier_ok" = "0" ]; then
    echo " 这是**未经实测的组合**：$_tier" >&2
    echo "   已验证的只有三条（见 logs/048 / 050）：" >&2
    echo "     档 B：SWA=0 RING=0 FULL=0            => 427,643" >&2
    echo "     档 C：SWA=1 RING=1 FULL=0            => 427,643（与 B 相同）" >&2
    echo "     档 D：SWA=1 RING=1 FULL=1 PREFILL=1  => 485,610" >&2
    echo "   => 拒绝起服（宁可拒绝，也不要静默跑成另一个档）。" >&2
    exit 2
fi

# 池后端：registered（aclrtHostRegister，推荐）/ pageable / pinned
NPU_OFFLOAD_HOST_MEM=${NPU_OFFLOAD_HOST_MEM:-registered}

# per-group blocks_per_chunk：SWA=1（细粒度）、full=8
#   ★ 这一项让「SWA 只保留每 1024 token 段的尾块」生效，池子需求降 4.89x
BLOCKS_PER_CHUNK=${BLOCKS_PER_CHUNK:-'{"default":8,"swa":1}'}

# ★ 必须：否则撞 tokens_per_block=32 % tokens_per_hash=128
PREFIX_MATCH_UNIT=${PREFIX_MATCH_UNIT:-32}

# ★★★ 2026-09-22 17:5x **默认值改正（本日第 6 次"静默降级"，而且这次是我自己引进的）**
#
#   背景：
#     * A2 **生产默认是 `ENGRAM=1`**（`shadow-pkg/scripts/serve_a2.sh:127` 的
#       `ENGRAM=${ENGRAM:-1}`）；用户的 `run_test.sh` **不传 ENGRAM** ⇒ **生产 Engram 是开的**。
#     * 而本脚本此前默认 `ENGRAM=0`，注释写的是"首版建议 0（Engram + 卸载池曾撞 207001）"。
#     ⇒ ★ **照本脚本的默认值上线，会把 Engram 静默关掉** —— 质量下降但**没有任何报错**，
#       而这正是本日反复出现的失败模式（`logs/065` §3 / §3b.0：开关送不到、程序安静地跑默认值）。
#
#   那条 207001 的旧证据（`logs/001` §4.2）：`ENGRAM=1` 时**连 32 MiB** 的
#   `aclrtMallocHostWithCfg` 都失败、**8/8 worker 全部命中** ⇒ 不是"容量不够"，
#   是**驱动侧 pinned/注册资源争用**。★ 但那条测的是**旧池后端（`pin_memory`）**；
#   本脚本现在用的是 `aclrtHostRegister(MAPPED)`（`NPU_OFFLOAD_HOST_MEM=registered`），
#   而 Engram 的 206 GiB 表**用的正是同一个 API** ⇒ **旧结论既不能证明现在会挂、
#   也不能证明现在不会挂【未确认】**。
#
#   ⇒ 因此：**默认与生产一致（1）**；要降级必须**显式** `ENGRAM=0`，并会**打印响亮警告**。
#     宁可响亮失败，也不要静默降级（响亮失败可回滚，静默降级会带着错误假设跑下去）。
ENGRAM=${ENGRAM:-1}
if [ "${ENGRAM:-1}" = "0" ]; then
    echo "⚠⚠⚠ 你显式设了 ENGRAM=0 —— 这是**质量降级**，不是默认行为。" >&2
    echo "    A2 生产默认是 ENGRAM=1（Engram 2 层 + 206 GiB 表是模型的一部分）；" >&2
    echo "    关掉它是为了规避旧的 207001 争用（logs/001 §4.2，**旧池后端下**的现象）。" >&2
    echo "    若你只是想让首版跑通，请确认你接受这个降级，并把它记在变更单里。" >&2
fi

# ★★★ 2026-09-22 19:4x **默认值第二次修正（后果比 ENGRAM 那条更严重）**
#
#   实测链路：
#     * `shadow-pkg/scripts/serve_a2.sh:153`   `ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-auto}`
#       ⇒ **不显式给，就是 `auto`**。
#     * ★ 而 A2 **生产是显式 `ENGRAM_DEVICE_INDEX=0`**（用户 09-20/09-21 的启动命令，
#       以及 `reports/a2-draft-graph-20260920.md` 第 106 行："`ENGRAM_DEVICE_INDEX=0`，
#       因 `ret=207001` 在 A2 上不可用"）⇒ **生产把这条路关着**。
#     * `auto` 的探针（`engram_device_index.py:probe_host_mapping_capability`）**只测
#       `aclrtHostRegister` 一个 4 KiB 页能否被接受** —— 而 **A2 的实测是
#       1/8/32/64 GiB 全过**（`logs/065` §3c）⇒ ★★ **A2 上探针会通过 ⇒ `auto` 会把
#       device-index 打开**。
#     * ★★★ 而 **A3 上正是这条路崩的**（`logs/069`，实测原文）：
#         ```
#         [DEVICE-INDEX] 能力探测通过：host mapping registered
#         [DEVICE-INDEX] Engram 表已映射为设备可寻址：L1=384006168行, L14=384016682行
#         ⇒ 注册 183 GiB ⇒ EH0012 + 池拿不到注册预算 ⇒ 起服失败 / 推理崩
#         ```
#       A3 上 `ENGRAM=1 + pageable`（**一个字节都不注册**）**照样出 `EH0012`×9** ⇒
#       证明那个失败**与池的 host 内存后端无关，是 device-index 路径本身**。
#   ⇒ ★★ 结论：**A2 生产之所以"一直没问题"，就是因为它显式关掉了 device-index。**
#      而本脚本此前**完全不设**这个变量 ⇒ 继承 shadow 的 `auto` ⇒
#      **在 A2 上会把它打开** ⇒ **正好走进 A3 那条崩掉的路**。
#   ⇒ 因此：**默认与生产一致（0）**；显式给 `auto`/`1` 会**打印响亮警告**并说明依据。
ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-0}
case "${ENGRAM_DEVICE_INDEX}" in
    0|"off"|"false"|"no") : ;;   # 与生产一致 ⇒ 静默通过
    *)
        echo "⚠⚠⚠ 你把 ENGRAM_DEVICE_INDEX 设成了 ${ENGRAM_DEVICE_INDEX} —— 而 A2 生产用的是 0。" >&2
        echo "    依据（实测）：" >&2
        echo "      * auto 的探针只测 aclrtHostRegister(4 KiB) 能否被接受；" >&2
        echo "      * A2 实测 1/8/32/64 GiB 注册全过（logs/065 §3c）⇒ auto 会【打开】device-index；" >&2
        echo "      * 而 A3 上正是这条路崩的：Engram 表注册 183 GiB ⇒ EH0012 + 池拿不到预算" >&2
        echo "        （logs/069；A3 上 ENGRAM=1+pageable 一个字节都不注册，照样 EH0012×9）。" >&2
        echo "    ⇒ 除非你有明确的验收目的，否则请用 ENGRAM_DEVICE_INDEX=0。" >&2
        ;;
esac
export ENGRAM_DEVICE_INDEX
DRY=${DRY:-0}

echo "=============================================================="
echo "A2 DRAM KV 卸载起服"
echo "=============================================================="
echo "  模型          : $MODEL"
echo "  镜像          : $IMAGE（★ 必须带 ENGRAM×卸载 的 P0 修复；指纹门会核对）"
echo "  profiler      : PROFILE=${V41_PROFILE:-${PROFILE:-0}}（1 ⇒ /start_profile 与 /stop_profile 可用；产物落 $LAUNCH_DIR/results/<RUN_ID>/prof）"
echo "  draft 入图    : DRAFT_GRAPH=$DRAFT_GRAPH（1 = 入图，与 A2 生产一致；A2 实测 +62% tok/s / hp −47%）"
if [ "$DRAFT_GRAPH" = "0" ]; then
    echo "  ⚠️⚠️ DRAFT_GRAPH=0 ⇒ **draft 退回 eager**：A2 上单流 88.7 → 54.7 tok/s（−38%）。"
    echo "        这是**四轴变三轴**；若非刻意对照，请去掉 DRAFT_GRAPH=0。"
fi
if [ "$P2_POOL_PATCH" = "1" ]; then
    echo "  池子          : ${OFFLOAD_GB} GiB（★ 档 B 宿主实占 ≈197 GiB，8 卡实测 1.9895x）"
else
    echo "  池子          : ${OFFLOAD_GB} GiB（档 A 宿主实占 ≈392 GiB，8 卡实测）"
fi
echo "  上下文/并发   : ${MAX_LEN} / ${MAX_SEQS}"
# ★★★ 1M 几何的两个必读量（2026-09-23 加；两条都是实测/文档口径，不是估算）
#   ① 池能装几个 1M 会话：1 GiB = 1024 unit，**1 个 1M 会话 = 24,064 unit**（A2 实测，含 1.2× 余量）
#   ② spec-decode 边界：vLLM 用 `num_sampled_tokens_per_step`(=1) 而不是 `num_lookahead_tokens`(=5)
#      裁剪 ⇒ 请求走到 `max_model_len − 6` 以内会 **8 rank 全崩**（Index out of range + ERR02005，logs/109）
#      ⇒ **可用规避**：`prompt + max_tokens ≤ max_model_len − 32`
if [ "$MAX_LEN" -ge 524288 ]; then
    _u=$(( OFFLOAD_GB * 1024 ))
    _one=24064
    echo "  ★ 1M 几何①    : 池 ${_u} unit ÷ ${_one} unit/会话 ⇒ 约 $(( _u / _one )) 个 1M 会话（含 1.2× 余量）"
    if [ "$MAX_SEQS" -gt 4 ] && [ "$MAX_LEN" -ge 1048576 ]; then
        echo "  ⚠️ 1M × MAX_SEQS=$MAX_SEQS ⇒ 需求 $(( MAX_SEQS ))M token，远超 HBM（A2 生产口径 3,498,354）" >&2
        echo "     生产是 MAX_SEQS=4；更多并发只会排队（waiting_by_reason=capacity），不会更快。" >&2
    fi
    _lim=$(( MAX_LEN - 128 - 32 ))
    echo "  ★ 1M 几何②    : 单请求上限为 prompt+max_tokens ≤ ${_lim}（= max_len − 128 − 32）"
    echo "                   超过会撞 spec-decode 边界 ⇒ 8 rank 全崩（logs/109）；${MAX_LEN} 会崩。"
fi
echo "  池后端        : $NPU_OFFLOAD_HOST_MEM"
echo "  ★★ 档位        : $_tier（**容量指纹**：B/C=427,643，D=485,610 —— 起服后核对）"
echo "  blocks_per_chunk: $BLOCKS_PER_CHUNK"
echo "  prefix_match_unit: $PREFIX_MATCH_UNIT"
echo "  ENGRAM        : $ENGRAM"
echo "  L1 (P2_POOL_PATCH): $P2_POOL_PATCH${P2_COMP_JSON:+  comp=$P2_COMP_JSON}"
echo "  per-group bpc : L1_POOL_PATCH=$L1_POOL_PATCH（1 = 整份替换 6 文件，含 config.py 的 dict 解析 + 按需行数）"
echo "  drop cache    : DROPCACHE=$DROPCACHE（1 = 起服前清整机 page cache；0 = 不动）"
echo "  加固 PGP_MGR_HARDEN: $PGP_MGR_HARDEN（stats=$PGP_MGR_STATS）"
if [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; then
    if [ "$KV8_FULL" = "1" ]; then
        echo "  ★ int8 档 D: SWA=$KV8_SWA ring16=$KV8_RING_FP16 full=$KV8_FULL prefill=$KV8_PREFILL"
        echo "     容量（A2 真权重，logs/048）: HBM ×1.1356（485,610 token，档 B 427,643）; 宿主待测"
    else
        echo "  ★ int8 档 C: SWA=$KV8_SWA ring16=$KV8_RING_FP16"
        echo "     容量（A2 真权重，logs/048）: HBM ×1.0000（427,643，与档 B 逐字相同）; ★ 宿主 197.21→150.01 GiB（×1.3146）"
    fi
    echo "  ★ APC_ALIGN=$APC_ALIGN（3 = 段栅格；档 C/D 的必需前提，logs/047）"
    echo "  ★ GRAPH_SAFE=$GRAPH_SAFE（1 = 图安全补丁；档 C/D 图模式必需，logs/049）"
    echo "     ⚠️ 注意：tiny 上的 ×1.4655/×1.9133【不适用于 A2】——A2 多一个 draft 组（logs/050）"
else
    echo "  int8            : 关（档 B，HBM ×1.000 + L1）"
fi
echo "-------------------------------------------------------------"

# ---------------------------------------------------------------- 补丁
# ★ 四种布局都自动认（否则 A2 上第一条命令就会卡在"缺补丁"）：
#     $REPO/a2/patches          （发布包 dsv41-release 的布局）
#     $REPO/a2/publish          （开发工作区的布局）
#     $A2/patches / $A2/publish （脚本被单拷出去用的情形）
PDIR=""
for _c in "$REPO/a2/patches" "$REPO/a2/publish" "$A2DIR/patches" "$A2DIR/publish"; do
    if [ -f "$_c/0001-offload-scheduler.patch.py" ]; then PDIR=$_c; break; fi
done
if [ -z "$PDIR" ]; then
    echo "✗ 找不到补丁目录（试过 \$REPO/a2/{patches,publish} 与 \$A2/{patches,publish}）" >&2
    echo "  \$REPO=$REPO  \$A2DIR=$A2DIR" >&2
    echo "  可用 PDIR=<含 0001-offload-scheduler.patch.py 的目录> 直接指定。" >&2
    exit 2
fi
echo "✓ 补丁目录：$PDIR"
for f in 0001-offload-scheduler.patch.py 0001b-offload-per-group-bpc-manager.patch.py \
         0001c-offload-per-group-bpc-hooks.patch.py 0002-offload-cpu-pool-host-registered.patch.py; do
    if [ ! -f "$PDIR/$f" ]; then
        echo "✗ 缺补丁 $PDIR/$f" >&2
        exit 2
    fi
done
echo "✓ 四个补丁文件已就位（md5 见 a2/patches/README.md）"

# ---------------------------------------------------------------- ★★ int8 修复件自检（2026-09-23 新增）
# 为什么需要：`dsa_v41.py` 有两个版本，**旧的（94aeebb7）在 A2 的 85 GiB 池上会付 ≈125 ms/step**
#   （SWA int8 取页成本正比于池总大小；修复版改成连续 chunk 视图后对池大小变平）。
#   而这两件是**挂载**的（锚在 PGO_LIB 之后，不受 PATCH_MODE 管）⇒ 可以直接在这里核。
if [ "$KV8_SWA" = "1" ] || [ "$KV8_RING_FP16" = "1" ] || [ "$KV8_FULL" = "1" ]; then
    _DSA="" ; _K="" ; _I8=""
    for _c in "$REPO/a2/patches" "$A2DIR/patches"; do
        [ -z "$_DSA" ] && [ -f "$_c/kv8-graphsafe/dsa_v41.py" ] && _DSA="$_c/kv8-graphsafe/dsa_v41.py"
        [ -z "$_K" ]   && [ -f "$_c/kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py" ] \
                       && _K="$_c/kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py"
    done
    _m() { [ -f "$1" ] && md5sum "$1" | cut -d' ' -f1 || echo "(缺失)"; }
    _dmd5=$(_m "$_DSA") ; _kmd5=$(_m "$_K")
    echo "--- int8 修复件（**挂载**，不随镜像；起服后可在容器内反查）---"
    echo "  dsa_v41.py         md5=$_dmd5   $_DSA"
    echo "  kv8_fuse_triton.py md5=$_kmd5   $_K"
    if [ "$_DSA" = "" ] || [ "$_dmd5" = "(缺失)" ]; then
        echo "⛔ 缺 kv8-graphsafe/dsa_v41.py ⇒ 档 C 起不来（挂载块会 die）" >&2; exit 2
    fi
    if [ "$_dmd5" = "94aeebb757d6d5708268754481a05e0a" ]; then
        echo "⛔ 你挂的是 **未修复版** dsa_v41.py（94aeebb7）⇒ A2 的 85 GiB 池会让 decode 每步多 ≈125 ms。" >&2
        echo "   修法： cd \$REPO && git pull --ff-only   （修复版 md5 = 7867da2a345d7135ddbc6919eec144f9）" >&2
        exit 2
    fi
    if [ "$_kmd5" = "(缺失)" ]; then
        echo "⛔ 缺 kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py" >&2
        echo "   ⇒ dsa_v41.py 顶部的 \`from vllm_ascend.attention import kv8_fuse_triton\` 会 **ImportError**（不是静默降级）。" >&2
        echo "   修法： git pull --ff-only （该件在 commit d32be29 里新增）" >&2
        exit 2
    fi
    if [ "$_dmd5" != "7867da2a345d7135ddbc6919eec144f9" ]; then
        echo "⚠️  dsa_v41.py 不是已知的修复版 md5（7867da2a…）也不是已知旧版 ⇒ 请自行确认。" >&2
    fi
fi

# shadow-pkg 的补丁目录（PATCHDIR）与起服对象（LAUNCH_DIR）在**参数区**已定义

# ============================================================================================
# ★★★ 2026-09-23 09:3x **P0 修复：DRY 与真实起服调的不是同一个对象**
#   现象（静态定位，stub 实测复现）：DRY 走 `cd "$SHADOW"`（shadow 的 serve_a2.sh，含注入块），
#     而**真实起服**走 `cd "$REPO"`（= 发布仓里的**模板**，`grep A2-OFFLOAD` 命中 **0**）
#     ⇒ 真正起服的那份 **4 个卸载补丁一个都不挂** = **静默零卸载**
#       （服务照常 READY、serve.log 无任何报错、`--kv-transfer-config` 还在，但卸载不生效）。
#   而 `logs/074` 那次"档 B 一个补丁都没挂"是**同一个坑的另一面**，当时只修了 export 段没修这里。
#   ⇒ 两条处置：
#     ① 起服入口**收敛成一个函数**（DRY 与真实起服共用）⇒ 天然同对象；
#     ② 加一条**内容断言**（不是路径断言）：要起服的那份 serve_a2.sh **必须**含 `[A2-OFFLOAD]`，
#        否则拒绝（宁可响亮失败）。★ 用 `LAUNCH_DIR=` 可显式改对象（A/B 对照时用）。
#   另外本函数**显式传 PROFILE/V41_PROFILE**（不再依赖环境继承 —— `logs/112` 就是继承坑）。
# ============================================================================================
#   （LAUNCH_DIR / _SV 已在**参数区**定义 —— 见那里的崩溃说明）
if [ ! -f "$_SV" ]; then
    echo "⛔ 找不到要起服的脚本：$_SV" >&2
    echo "   （shadow 还没造？ PKG=$REPO DST=$SHADOW bash $A2DIR/scripts/make_shadow_pkg.sh）" >&2
    exit 2
fi
if ! grep -q '\[A2-OFFLOAD\]' "$_SV"; then
    echo "⛔⛔ 要起服的 $_SV 里**没有 [A2-OFFLOAD] 注入块** ⇒ 4 个卸载补丁一个都不会挂（静默零卸载）。" >&2
    echo "     判据是**内容**不是路径：这份 shadow 是旧生成器造的，或 LAUNCH_DIR 指错了。" >&2
    echo "     修法： PKG=$REPO DST=$LAUNCH_DIR bash $A2DIR/scripts/make_shadow_pkg.sh" >&2
    exit 2
fi
echo "✓ 起服对象：$_SV（含 [A2-OFFLOAD] 注入块）"

_launch_serve() {   # $1 = DRY_RUN（0 真起 / 1 干跑）
    cd "$LAUNCH_DIR" || return 2
    # ★ 先把值算成**一个**变量，再赋给两个名字。
    #   否则 `A=x B=...$A...` 这种同前缀多处赋值在 bash 里的求值顺序有歧义
    #   （实测：`PROFILE=0 V41_PROFILE=1` 会得到 PROFILE=0/V41_PROFILE=1 —— 见 stub 自测 ⑤）。
    local _pf="${V41_PROFILE:-${PROFILE:-0}}"
    DRY_RUN="$1" \
    PROFILE="$_pf" \
    V41_PROFILE="$_pf" \
    MODEL="$MODEL" IMAGE="$IMAGE" GPU_UTIL="$GPU_UTIL" PORT="$PORT" \
    SERVED_NAME="$SERVED_NAME" MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" \
    BAT_TOKENS="$BAT_TOKENS" DRAFT_GRAPH="$DRAFT_GRAPH" DROPCACHE="$DROPCACHE" \
    KV_ARGS_EXTRA="$KV_ARGS" \
    bash scripts/serve_a2.sh
}

# ★★ 自检门：开了 int8 但 shadow 不认 `A2_*` ⇒ **拒绝起服**（宁可响亮失败，不要静默跑成档 B）
if [ "$KV8_SWA" = "1" ] || [ "$KV8_FULL" = "1" ] || [ "$KV8_RING_FP16" = "1" ]; then
    _shadow_ok=0
    for _f in "$SHADOW/scripts/serve_a2.sh"; do
        [ -f "$_f" ] || continue
        grep -q "A2_KV8_SWA" "$_f" && _shadow_ok=1
        grep -q "A2_GRAPH_SAFE" "$_f" || _shadow_ok=0
    done
    if [ "$_shadow_ok" != 1 ]; then
        echo "⛔ 你开了 int8（KV8_SWA=$KV8_SWA KV8_FULL=$KV8_FULL RING_FP16=$KV8_RING_FP16），" >&2
        echo "   但这个 shadow-pkg **不认 A2_* 环境变量** ⇒ int8 会**静默失效**（跑起来是档 B）。" >&2
        echo "   修法：用当前版本的生成器重造 shadow：" >&2
        echo "     PKG=$REPO DST=$SHADOW bash $A2DIR/scripts/make_shadow_pkg.sh" >&2
        echo "   （或去掉 KV8_* 只跑档 B）" >&2
        exit 2
    fi
fi
#
# ★★★ 2026-09-22 20:5x —— **修掉一个我自己引入的 P0 静默降级（本日第 7 次同类）**
#
#   现象【实测·本机 DRY=1 对照，唯一变量 = 有没有 KV8_*】：
#     * 档 C（`KV8_SWA=1 KV8_RING_FP16=1`）⇒ `[a2-dry] MOUNTS(24)`，4 个卸载补丁都在；
#     * 档 B（不带 KV8_*）            ⇒ `[a2-dry] MOUNTS(2)`，**4 个卸载补丁一个都没挂**。
#   静态定位：上面那个 `if [ int8 ]; then` 的 `fi` 原本落在**本段末尾**，把
#     「shadow 存在性检查 + 拷补丁 + `export OFFLOAD_*_PATCH=1` + `export P2_*`」
#     **整段吞进了 int8 分支** ⇒ 只要不开 int8，这些 export 一条都不执行：
#       - `OFFLOAD_SCHED_PATCH` / `OFFLOAD_NPU_WORKER_PATCH` 不导出 ⇒ 影子包按 `:-0` 读
#         ⇒ **卸载调度器与 registered 池补丁都不挂** ⇒ 起服能成、但**没有 DRAM 卸载**；
#       - `P2_POOL_PATCH` / `P2_COMP_JSON` 不导出 ⇒ 影子包按 `P2_POOL_PATCH=0` 读
#         ⇒ **L1（池张量按需分配行数）失效** ⇒ 宿主实占回到 ≈392 GiB（档 B 应为 ≈197 GiB）。
#   为什么危险：两条都不会报错，服务照常 READY —— 正是本项目一直在防的"静默降级"。
#   ⇒ 现在：① shadow 存在性/拷补丁/公共 export **移出** int8 分支（只有 `A2_*` 名字在里面）；
#           ② 起服前把这两个补丁开关**断言成 1**（拿不到就 exit 2，宁可响亮失败）；
#           ③ DRY=1 时**断言真实 MOUNTS 里必须出现那 4 个卸载补丁文件**（本 bug 的回归门）。

if [ ! -d "$SHADOW" ]; then
    echo "⚠ 找不到 shadow-pkg（$SHADOW）—— 请设 SHADOW_PKG=<路径>" >&2
    echo "  （补丁必须经 shadow-pkg 挂载；不要直接改 dsv41-release/scripts/serve_a2.sh）" >&2
    exit 2
fi
mkdir -p "$PATCHDIR"
cp "$PDIR/0001-offload-scheduler.patch.py"                  "$PATCHDIR/scheduler.py"
cp "$PDIR/0001b-offload-per-group-bpc-manager.patch.py"     "$PATCHDIR/pgp_manager.py"
cp "$PDIR/0001c-offload-per-group-bpc-hooks.patch.py"       "$PATCHDIR/pgp_hooks.py"
cp "$PDIR/0002-offload-cpu-pool-host-registered.patch.py"   "$PATCHDIR/cpu_npu.py"
echo "✓ 补丁已复制到 $PATCHDIR"

# ★★★ 2026-09-23 09:5x **显式**给出 4 个挂载源 = 本脚本刚刚拷好的那份。
#   为什么：注入块里的默认值是 `$_A2F/...`（`_A2F=${A2_OFFLOAD_FILES:-$PKG/a2/patches}`），
#   即"影子包自己的那份副本" —— 与本脚本刚拷的 `$PATCHDIR/*` 是**两个不同对象**。
#   起服时到底挂哪一份取决于影子包里有什么 ⇒ 不可控。现在钉死成本脚本刚准备的这份。
#   （同族纪律：不依赖继承/内部默认，把要用的东西显式给全 —— 见 logs/112。）
export OFFLOAD_SCHED_FILE="$PATCHDIR/scheduler.py"
export OFFLOAD_PGP_MANAGER="$PATCHDIR/pgp_manager.py"
export OFFLOAD_PGP_HOOKS="$PATCHDIR/pgp_hooks.py"
export OFFLOAD_CPU_NPU_FILE="$PATCHDIR/cpu_npu.py"

# ---------------------------------------------------------------- 起服
export OFFLOAD_SCHED_PATCH=1
export OFFLOAD_NPU_WORKER_PATCH=1
export NPU_OFFLOAD_HOST_MEM
export PREFIX_MATCH_UNIT
export ENGRAM
# ★ 可选项（默认关；不改默认行为）
[ "$P2_POOL_PATCH" = "1" ] && export P2_POOL_PATCH=1 && export P2_WORKER_ROWS=1
export L1_POOL_PATCH L1_POOL_DIR
case "$P2_POOL_PATCH" in
  1) : "${P2_COMP_JSON:?★ 开 L1 时必须给 P2_COMP_JSON（与张量数匹配，给错会 fail-closed）}"; export P2_COMP_JSON ;;
esac
export PGP_MGR_HARDEN PGP_MGR_STATS

# ★ 起服前断言：这两个开关必须真的是 1（上面那段曾被 int8 分支吞掉过 ⇒ 加硬门）
for _req in OFFLOAD_SCHED_PATCH OFFLOAD_NPU_WORKER_PATCH; do
    if [ "${!_req:-0}" != "1" ]; then
        echo "⛔ ${_req}='${!_req:-<unset>}' —— 卸载补丁不会挂进容器。" >&2
        echo "   这会让服务**照常起来但完全没有 DRAM 卸载**（静默降级）。" >&2
        echo "   ⇒ 拒绝起服。请检查本脚本的 export 段是否被条件分支吞掉。" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------- ★★★ 挂载件的"可导入性"预检
# 为什么（2026-09-23 实测事故）：A2 首次起服报
#     ModuleNotFoundError: No module named 'pgp_manager'
# 机制（已用最小包树逐条实测）：
#   `scheduler.py` / `pgp_hooks.py` 里那条**裸 import** `from pgp_manager import …`
#   只有在 `PYTHONPATH` 含该目录时才成立；而容器里 `pgp_manager.py` 是作为
#   **vllm 子模块**挂的（`vllm/v1/kv_offload/cpu/pgp_manager.py`）⇒ 裸 import **必然失败**。
#   ★ 实测三态：裸 import ❌ ｜ 包路径 ✅ ｜ 哪怕同目录再放一份也 ❌（包内绝对导入不查同级目录）。
#   ★ 对照：A3 8 卡链用的是**预先构建的合并版 scheduler**（带 try/except 回退）⇒ 没撞上这个坑。
# 处置（2026-09-23 10:0x **按"零风险 + 不作假"重写**）：
#   ★ 第一版我写成"起一次性容器真 import" ⇒ **在 A2 上误报了**：
#     那个容器没带 NPU 设备，`import vllm` 的链条走到 `torch_npu → libascend_hal.so`
#     必然失败（真容器是 `--privileged=true` + `--device /dev/davinci*` 起的）。
#     ⇒ **判据绑错了对象**：用"没有 NPU 的容器"去判"有 NPU 的容器里能不能 import。
#   ⇒ 现在分两层：
#     ① **默认（零容器、零风险）= 内容判据**：两个挂载件里**必须出现**包路径回退那条 import。
#        判据绑**内容**不绑路径（本仓纪律），且它精确覆盖本次的故障类。
#     ② **可选真 import**（`IMPORT_GATE=1`）：镜像照真起服的设备参数来（privileged + davinci 设备），
#        才去真 import。默认关 —— 因为真起服本身就会在 import 期**响亮报错**，不需要靠预检兜。
if [ "${SKIP_IMPORT_GATE:-0}" = "1" ]; then
    echo "⚠️ 已按 SKIP_IMPORT_GATE=1 跳过「挂载件健全性检查」（★ 不建议）"
else
    echo "-------------------------------------------------------------"
    echo "★ 挂载件健全性检查（纯内容判据，零容器、零风险）"
    _gate_v=0
    for _pair in "scheduler.py:pgp_manager" "pgp_hooks.py:pgp_manager"; do
        _f="${_pair%%:*}"; _dep="${_pair##*:}"
        if [ ! -f "$PATCHDIR/$_f" ]; then
            echo "  ⛔ $PATCHDIR/$_f **不存在**（挂载块会 die）"; _gate_v=1; continue
        fi
        # 内容判据：既要**能**走包路径（容器里唯一的可行路径），也要保留裸 import（PYTHONPATH 场景）
        _has_pkg=$(grep -c "from vllm\.v1\.kv_offload\.cpu\.$_dep import" "$PATCHDIR/$_f" || true)
        _has_bare=$(grep -c "^from $_dep import\|^    from $_dep import" "$PATCHDIR/$_f" || true)
        _has_try=$(grep -c "^try:" "$PATCHDIR/$_f" || true)
        if [ "${_has_pkg:-0}" -ge 1 ] && [ "${_has_try:-0}" -ge 1 ]; then
            echo "  ✓ $_f：含包路径回退（try/except）＋裸 import 兼容路径"
        elif [ "${_has_bare:-0}" -ge 1 ]; then
            echo "  ⛔ $_f：**只有裸 import、没有包路径回退** ⇒ 容器里必然 ModuleNotFoundError: No module named '$_dep'"
            _gate_v=1
        else
            echo "  ⚠ $_f：没找到 \`$_dep\` 的 import（版本可能已变，请人工确认）"
        fi
    done
    if [ "$_gate_v" != "0" ]; then
        echo "" >&2
        echo "⛔⛔⛔ 挂载件健全性检查失败 ⇒ 起服会在 import 期崩（ModuleNotFoundError: No module named 'pgp_manager'）。" >&2
        echo "   修法：\`git pull --ff-only\` 拿修复版（两份补丁应含 try/except 回退）" >&2
        exit 2
    fi
fi

# ★★ 可选：真 import 预检（默认关）。要开就 `IMPORT_GATE=1`。
#   为什么默认关：它必须带 `--privileged` + NS 设备才可能成功，而那与真起服**抢同一批设备**；
#   而"import 不起来"这件事，真起服本身就会**响亮报错**（不会静默）⇒ 预检的边际价值低、风险高。
if [ "${IMPORT_GATE:-0}" = "1" ]; then
    echo "-------------------------------------------------------------"
    echo "★ 真 import 预检（IMPORT_GATE=1；★ 会按真起服的设备参数起一次性容器）"
    _dev=(); for d in $DEVS; do _dev+=(--device "/dev/davinci$d"); done
    _imp=$(docker run --rm --privileged=true \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc "${_dev[@]}" \
        -v "$PATCHDIR/scheduler.py:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro" \
        -v "$PATCHDIR/pgp_manager.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py:ro" \
        -v "$PATCHDIR/pgp_hooks.py:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_hooks.py:ro" \
        --entrypoint python3 "$IMAGE" -c '
import importlib
for m in ("vllm.v1.kv_offload.cpu.pgp_manager", "vllm.v1.kv_offload.cpu.pgp_hooks"):
    mod = importlib.import_module(m)
    print("OK-IMPORT", mod.__file__)
' 2>&1)
    _imp_rc=$?
    printf '%s\n' "$_imp" | tail -5 | sed 's/^/    /'
    if [ "$_imp_rc" != "0" ] || [ "$(printf '%s' "$_imp" | grep -c 'OK-IMPORT')" -lt 2 ]; then
        echo "  ⛔ 真 import 预检失败（详见上；★ 若报 libascend_hal.so / 设备相关 ⇒ 多是设备未就位，不一定是代码问题）" >&2
        exit 2
    fi
    echo "  ✓ 两个挂载件在带 NPU 设备的容器里都能 import"
fi

# ★★ 名字对齐（**这是一个静默 no-op 的坑**，见 logs/055 §7）：
#   `VLLM_V41_*` 只在**容器内**有意义；宿主上导出它们**一个字节都到不了容器**
#   （容器环境由 shadow-pkg 的 inner.sh 建立）。
#   所以这里导出的是 **shadow 认的那套宿主名 `A2_*`**，由 inner.sh 在**容器内**转成 `VLLM_V41_*`。
#   ⇒ 若哪天 shadow 换了一套名字，这里就会**静默退回档 B**（跑得起来、但没有 int8 效果）。
#     为此下面加了一道**起服前**的自检门（拒绝静默失效），起服后还有一条**回读校验**（见脚本末尾的自检清单）。
#   ★ 注意：这一块**只在开了 int8 时**才导出（A2_* 名字对档 B 没有意义）。
if [ "$KV8_SWA" = "1" ] || [ "$KV8_FULL" = "1" ] || [ "$KV8_RING_FP16" = "1" ]; then
export A2_KV8_SWA="$KV8_SWA" \
       A2_RING_FP16="$KV8_RING_FP16" \
       A2_KV8="$KV8_FULL" \
       A2_KV8_PREFILL="$KV8_PREFILL" \
       A2_APC_ALIGN="$APC_ALIGN" \
       A2_GRAPH_SAFE="$GRAPH_SAFE"
# 挂载 dsa_v41.py 的开关：**开了 GRAPH_SAFE 就必须挂**，否则开关是空的（同款静默 no-op）
export A2_KV8_GRAPHSAFE="$GRAPH_SAFE"
# 生成 shadow 时若用的是别处的补丁目录，这里可覆盖
[ -n "${A2_KV8_DSA:-}" ] && export A2_KV8_DSA

fi

KV_ARGS="--prefix-match-unit $PREFIX_MATCH_UNIT \
--kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((OFFLOAD_GB * 1073741824)),\"blocks_per_chunk\":$BLOCKS_PER_CHUNK,\"spec_name\":\"NPUOffloadingSpec\",\"spec_module_path\":\"vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu\"}}"

# ---------------------------------------------------------------- ★★★ 起服前指纹门（2026-09-22 21:2x）
# 为什么必须有这道门：`ENGRAM=1` + 卸载的那条 P0（镜像缺页 ⇒ KeyError ⇒ 引擎死，`logs/073`）
# 的修复是**改在 `patches/files/engram_hash.py` + `engram_jit_kernel.py`** 上的。而 A2 默认
# `PATCH_MODE=baked` ⇒ 容器里读的是**镜像里烘焙的那份**。⇒ 若镜像还是旧的（`dsv41-a2:v8`），
# **修复一个字节都到不了容器**，而症状要等 30 分钟起服 + 一轮 replay 压测才出现
# （= 又一个静默降级）。所以这里在起服前 5 秒把两件事对齐：
#     host 侧权威副本（$SHADOW/patches/files/*.py） vs 镜像里实际那份
# 不一致 ⇒ **拒绝起服**（宁可响亮失败，也不要白等 30 分钟）。
_pf_check() {
    local _f="$1" _cp="$2"
    local _want _tgt
    _want=$(md5sum "$_f" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$_want" ]; then
        echo "⚠ 指纹门：找不到 $_f —— 跳过这一项（无法核对镜像里的 $_cp）" >&2
        return 0
    fi
    # ★ 先确认镜像**本地存在**：否则 `docker run` 会去 registry **拉取**（A2 上可能挂几分钟，
    #   甚至真的把几十 GB 拉下来）。这一步只读本地镜像表，零网络。
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "" >&2
        echo "⛔⛔⛔ 指纹门不通过：**本地没有镜像 $IMAGE**" >&2
        echo "   ⇒ 它要么还没 build，要么名字写错了。请先：" >&2
        echo "       IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh" >&2
        echo "     然后用同样的 IMAGE 起服（见 a2/logs/075）。" >&2
        echo "" >&2
        echo "   · 本机现有的 dsv41-a2 镜像（供对照；用 IMAGE=<名字> 覆盖默认）：" >&2
        docker image ls --format '       {{.Repository}}:{{.Tag}}  {{.ID}}  {{.CreatedSince}}' 2>/dev/null \
          | grep -E 'dsv41-a2' | head -8 >&2 || true
        exit 2
    fi
    # ★ 注意：下面只用 **单个 -c**，且 md5sum 的路径是镜像内绝对路径；
    #   `docker run --rm --entrypoint md5sum <img> <path>` 在路径不存在时**非零退出**，
    #   这里靠空值区分"读不到"与"值不同"。
    _tgt=$(docker run --rm --entrypoint md5sum "$IMAGE" "$_cp" 2>/dev/null | cut -d' ' -f1)
    if [ "$_want" = "$_tgt" ]; then
        echo "  ✓ 指纹门 $_cp  $_tgt"
        return 0
    fi
    echo "" >&2
    echo "⛔⛔⛔ 指纹门不通过：镜像 $IMAGE 里的 $_cp" >&2
    echo "     镜像里 = ${_tgt:-<读不到：镜像不存在或路径不同>}" >&2
    echo "     期望值 = $_want   （来自 $_f）" >&2
    echo "" >&2
    # ★★★ 这里**绝对不能用反引号**（`...`）：shell 会把提示文字里那两条命令**真的执行掉**，
    #   而其中一条正是本脚本自己 ⇒ **自我递归**（2026-09-22 21:0x 实测：进程树自己套了 10+ 层，
    #   跑飞的形态是"输出里混进 build_image 的日志"）。用单引号 + 纯文本。
    echo '   ★ 这意味着 **ENGRAM × 卸载 的 P0 修复不在这个镜像里**（见 a2/logs/075 / 073）：' >&2
    echo '     ENGRAM=1 + 卸载时，一旦发生取回就会 KeyError(2486) ⇒ 引擎死。' >&2
    echo '   ⇒ 二选一：' >&2
    echo '     (a) 用带修复的镜像起服（推荐）：' >&2
    echo '           IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh' >&2
    echo '           IMAGE=dsv41-a2:v9 OFFLOAD_GB=85 ENGRAM=1 bash a2/scripts/serve_a2_offload.sh' >&2
    echo '     (b) 或者显式 ENGRAM=0 起服（**质量降级**，只用于排查，见 A2-DEPLOY-NOW 第 6 条）。' >&2
    exit 2
}

if [ "${ENGRAM:-1}" = "1" ]; then
    echo "-------------------------------------------------------------"
    echo "★ 起服前指纹门（ENGRAM=1）：核对镜像里是否带 ENGRAM×卸载 的 P0 修复"
    _pf_check "$SHADOW/patches/files/engram_hash.py" \
              "/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py"
    _pf_check "$SHADOW/patches/files/engram_jit_kernel.py" \
              "/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_jit_kernel.py"
    echo "  ⇒ 两项一致：镜像里确实带着修复（缺页会被降级成 barrier，而不是 KeyError）"
    echo "-------------------------------------------------------------"
fi

echo "-------------------------------------------------------------"
echo "★ 起服后必须跑这三条自检（任一为 0 就停，别压测）："
echo "    grep -c 'P1_pinned.*ret=0'            <serve.log>   # 期望 8"
echo "    grep -c 'D2_offload'                  <serve.log>   # 期望 >0"
echo "    grep -c 'alignment_chunk_count.*8'    <serve.log>   # 期望 >0（per-group 生效）"
echo "  ★★ 开了 int8 时**必查这条**（否则你会以为在跑档 C，其实在跑档 B）：
    INNER=\$(dirname <serve.log>)/inner.sh
    grep -m1 -a 'VLLM_V41_KV8_SWA=' "\$INNER"    # 期望它不是 0（= env 真的进了容器）
  ★ 跑 L1 时再加两条："
echo "    grep -c 'P2_poolsizing'               <serve.log>   # 期望 >0（L1 生效）"
echo "    grep -a 'P2_WORKER_HOST_BYTES'        <serve.log>   # ★ 宿主实占（档 B 应 ≈1.927x 更省）"
echo "  ★ 上线后监测（logs/027 判据 1 + logs/040 判据 2）："
echo "    kv_offload_block_removed_total{medium=\"CPU\"} == 0"
echo "    units_ratio >= 1.05（工作集 = cpu_cache_usage_perc x num_units）"
echo "-------------------------------------------------------------"

if [ "$DRY" = "1" ]; then
    echo "[DRY] 将要执行："
    echo "  OFFLOAD_GB=$OFFLOAD_GB MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS \\"
    echo "  KV_ARGS_EXTRA='$KV_ARGS' \\"
    echo "  PROFILE=${V41_PROFILE:-${PROFILE:-0}} \\"
    echo "  DRAFT_GRAPH=$DRAFT_GRAPH \\"
    echo "  DROPCACHE=$DROPCACHE L1_POOL_PATCH=$L1_POOL_PATCH \\"
    echo "  bash scripts/serve_a2.sh        # ← 在 $LAUNCH_DIR 下（含 [A2-OFFLOAD]）"
    # ★★★ 2026-09-22 15:3x 补一道**验证盲区**：
    #   此前 DRY=1 在这里就 exit 0 ⇒ **shadow 的 MOUNTS 组装一次都没跑过**
    #   ⇒ "挂载块是否真的生效"在 dry-run 里**完全没被验证**（2026-09-22 实测踩到）。
    #   现在把 shadow 自己也用 DRY_RUN=1 跑一遍，把**真实挂载清单**打出来。
    echo
    echo "[DRY] ↓↓↓ 转调 shadow-pkg 的 DRY_RUN（这一步会打印**真实 MOUNTS**）↓↓↓"
    #   ★ 必须用 **$SHADOW** 自己的 scripts/ —— 不能用 $REPO（那是**本脚本所在仓**，
    #     与 shadow 不是同一个目录；2026-09-22 实测在这里踩过 rc=127）。
    #   ★★★ 2026-09-22 20:5x：把子进程输出**收进变量**，除了打印，还要**断言挂载清单**。
    #     起因：档 B 下 4 个卸载补丁**一个都没挂**而 dry-run 照样 rc=0 打印 "OK"（见上方 P0 注释）。
    #     判据就是 MOUNTS 里那 4 个绝对路径 —— 缺任一 ⇒ 拒绝（这是本 bug 的回归门）。
    _dry_out=$(_launch_serve 1 2>&1)
    _dry_rc=$?
    printf '%s\n' "$_dry_out"
    [ "$_dry_rc" = "0" ] || { echo "⛔ shadow 的 DRY_RUN 失败（rc=$_dry_rc）—— 上面就是原因" >&2; exit 2; }
    # ★★★ 判据改绑**容器内目标路径**（而不是源文件名）—— 2026-09-23 修：
    #   原来 grep 的是 `0001-offload-scheduler.patch.py` 这类**源文件名**，但挂载时源文件被
    #   **改名**成 `scheduler.py` / `pgp_manager.py` … ⇒ 断言恒 FAIL（而"挂载其实是对的"）。
    #   ⇒ 同族纪律：**判据绑实际生效的那个对象（容器内目标路径），不绑中转文件的名字。**
    #   ★ L1 路线（默认）会整份替换 5 个文件；旧路线（L1_POOL_PATCH=0）少 config.py/spec.py。
    _miss=0
    _need="/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro"
    if [ "$L1_POOL_PATCH" = "1" ]; then
        _need="$_need /vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro"
        _need="$_need /vllm-workspace/vllm/vllm/v1/kv_offload/cpu/spec.py:ro"
        _need="$_need /vllm-workspace/vllm/vllm/v1/kv_offload/cpu/p2_pool.py:ro"
        _need="$_need /vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/p2_worker.py:ro"
    fi
    _need="$_need /vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py:ro"
    _need="$_need /vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro"
    for _t in $_need; do
        printf '%s\n' "$_dry_out" | grep -q -- "$_t" || { echo "⛔ MOUNTS 里缺目标：$_t" >&2; _miss=1; }
    done
    if [ "$_miss" != "0" ]; then
        echo "   ⇒ 起服会**跑起来但没有 DRAM 卸载**（静默降级）⇒ 拒绝放行。" >&2
        echo "     查：本脚本的 export 段是否被条件分支吞掉；或影子包是否认 OFFLOAD_SCHED_PATCH/L1_POOL_PATCH。" >&2
        exit 2
    fi
    echo "[DRY] ✓ 卸载/池相关挂载目标齐备（$(echo $_need | wc -w) 个；L1_POOL_PATCH=$L1_POOL_PATCH）"
    echo "[DRY] ↑↑↑ 以上是真实挂载清单 ↑↑↑"
    exit 0
fi

# ★★★ 用**统一入口**（与 DRY 同一对象）：2026-09-23 前这里写死 `cd "$REPO"`，
#   跑的是发布仓的**模板**（无 [A2-OFFLOAD] 注入）⇒ 静默零卸载。见本文件上方 P0 注释。
_launch_serve 0
