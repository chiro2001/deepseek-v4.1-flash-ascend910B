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
# 用法：
#   bash a2/scripts/serve_a2_offload.sh              # 32K 场景（OFFLOAD_GB=16）
#   OFFLOAD_GB=48 MAX_LEN=131072 bash a2/scripts/serve_a2_offload.sh   # 128K 场景
#
# ★ 起服后**必须先跑自检**（脚本末尾会打印命令），否则可能白等 20 分钟。
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2DIR=$(cd "$HERE/.." && pwd)            # a2/ 自己（补丁可能在 a2/patches 或 a2/publish）
REPO=$(cd "$HERE/../.." && pwd)          # dsv41-release/

# ---------------------------------------------------------------- 参数
# ★ 模型路径不硬编码（发布包不留真实账号名）；必须由调用方给
MODEL=${MODEL:?请设 MODEL=<模型目录>}
IMAGE=${IMAGE:-}
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
OFFLOAD_GB=${OFFLOAD_GB:-56}
MAX_LEN=${MAX_LEN:-131072}
MAX_SEQS=${MAX_SEQS:-16}
BAT_TOKENS=${BAT_TOKENS:-2048}

# ★ L1（池张量按需分配行数）—— ★★ 8 卡真权重实测 1.9895x（392.35 -> 197.21 GiB）
#   defaults 到这里 = 档 B（推荐）；置 0 即回档 A
#   P2_POOL_PATCH=1 时必须同时给对的 P2_COMP_JSON（与"张量数"匹配，给错会 fail-closed）
P2_POOL_PATCH=${P2_POOL_PATCH:-1}
# 16 张量几何的【实测·worker 侧真值】分量（8 卡真权重，8 rank x 9 行一致）
P2_COMP_JSON=${P2_COMP_JSON:-'[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]'}

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
#   ★ 前提：挂上 patches/kv8-graphsafe/dsa_v41.py（md5 94aeebb7…）
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
    echo "   前提：已挂 patches/kv8-graphsafe/dsa_v41.py（md5 94aeebb7…）" >&2
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
DRY=${DRY:-0}

echo "=============================================================="
echo "A2 DRAM KV 卸载起服"
echo "=============================================================="
echo "  模型          : $MODEL"
if [ "$P2_POOL_PATCH" = "1" ]; then
    echo "  池子          : ${OFFLOAD_GB} GiB（★ 档 B 宿主实占 ≈197 GiB，8 卡实测 1.9895x）"
else
    echo "  池子          : ${OFFLOAD_GB} GiB（档 A 宿主实占 ≈392 GiB，8 卡实测）"
fi
echo "  上下文/并发   : ${MAX_LEN} / ${MAX_SEQS}"
echo "  池后端        : $NPU_OFFLOAD_HOST_MEM"
echo "  ★★ 档位        : $_tier（**容量指纹**：B/C=427,643，D=485,610 —— 起服后核对）"
echo "  blocks_per_chunk: $BLOCKS_PER_CHUNK"
echo "  prefix_match_unit: $PREFIX_MATCH_UNIT"
echo "  ENGRAM        : $ENGRAM"
echo "  L1 (P2_POOL_PATCH): $P2_POOL_PATCH${P2_COMP_JSON:+  comp=$P2_COMP_JSON}"
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

# shadow-pkg 的补丁目录（serve_a2.sh 的 PATCH_MODE=mount 从这里挂）
SHADOW=${SHADOW_PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
PATCHDIR="$SHADOW/patches/files/offload_dsv41"

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

# ---------------------------------------------------------------- 起服
export OFFLOAD_SCHED_PATCH=1
export OFFLOAD_NPU_WORKER_PATCH=1
export NPU_OFFLOAD_HOST_MEM
export PREFIX_MATCH_UNIT
export ENGRAM
# ★ 可选项（默认关；不改默认行为）
[ "$P2_POOL_PATCH" = "1" ] && export P2_POOL_PATCH=1 && export P2_WORKER_ROWS=1
case "$P2_POOL_PATCH" in
  1) : "${P2_COMP_JSON:?★ 开 L1 时必须给 P2_COMP_JSON（与张量数匹配，给错会 fail-closed）}"; export P2_COMP_JSON ;;
esac
export PGP_MGR_HARDEN PGP_MGR_STATS

# ★★ 名字对齐（**这是一个静默 no-op 的坑**，见 logs/055 §7）：
#   `VLLM_V41_*` 只在**容器内**有意义；宿主上导出它们**一个字节都到不了容器**
#   （容器环境由 shadow-pkg 的 inner.sh 建立）。
#   所以这里导出的是 **shadow 认的那套宿主名 `A2_*`**，由 inner.sh 在**容器内**转成 `VLLM_V41_*`。
#   ⇒ 若哪天 shadow 换了一套名字，这里就会**静默退回档 B**（跑得起来、但没有 int8 效果）。
#     为此下面加了一道**起服前**的自检门（拒绝静默失效），起服后还有一条**回读校验**（见脚本末尾的自检清单）。
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
    echo "  bash scripts/serve_a2.sh"
    # ★★★ 2026-09-22 15:3x 补一道**验证盲区**：
    #   此前 DRY=1 在这里就 exit 0 ⇒ **shadow 的 MOUNTS 组装一次都没跑过**
    #   ⇒ "挂载块是否真的生效"在 dry-run 里**完全没被验证**（2026-09-22 实测踩到）。
    #   现在把 shadow 自己也用 DRY_RUN=1 跑一遍，把**真实挂载清单**打出来。
    echo
    echo "[DRY] ↓↓↓ 转调 shadow-pkg 的 DRY_RUN（这一步会打印**真实 MOUNTS**）↓↓↓"
    #   ★ 必须用 **$SHADOW** 自己的 scripts/ —— 不能用 $REPO（那是**本脚本所在仓**，
    #     与 shadow 不是同一个目录；2026-09-22 实测在这里踩过 rc=127）。
    ( cd "$SHADOW" && DRY_RUN=1 MODEL="$MODEL" IMAGE="$IMAGE" GPU_UTIL="$GPU_UTIL" PORT="$PORT" \
        SERVED_NAME="$SERVED_NAME" MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" \
        BAT_TOKENS="$BAT_TOKENS" KV_ARGS_EXTRA="$KV_ARGS" \
        bash scripts/serve_a2.sh ) || { echo "⛔ shadow 的 DRY_RUN 失败（rc=$?）—— 上面就是原因" >&2; exit 2; }
    echo "[DRY] ↑↑↑ 以上是真实挂载清单 ↑↑↑"
    exit 0
fi

cd "$REPO"
MODEL="$MODEL" IMAGE="$IMAGE" GPU_UTIL="$GPU_UTIL" PORT="$PORT" \
SERVED_NAME="$SERVED_NAME" MAX_LEN="$MAX_LEN" MAX_SEQS="$MAX_SEQS" \
BAT_TOKENS="$BAT_TOKENS" \
KV_ARGS_EXTRA="$KV_ARGS" \
    bash scripts/serve_a2.sh
