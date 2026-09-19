#!/usr/bin/env bash
# 在容器内生成 vllm-ascend 的 patch 系列（带 git 历史，git am 可复现）
#
# 用法（容器内）：
#   bash /tmp/make_series_va.sh <输出目录> [前一系列目录]
#
# ## 为什么还要传「前一系列目录」
#
# v8（Engram device-index）不是重写，而是**叠在旧系列 0007 之上的增量**：
#   * 改：`model.py`（+313 行，设备路径 + 入图缓存）、`engram_hbm.py`（+31 行，能力探测只做 register）
#   * 新增：`engram_device_index.py`（host-mapped 表 + 设备侧哈希）、`engram_graph.py`（每 shape 一张图）
#
# 要让 **0001..0008 的提交历史与正文逐字节不变**，正确做法是：
#   ① 先把**旧系列** `git am` 进 worktree（此时树 = v7 的最终态）
#   ② 再把当前工作树的文件覆盖上去（未变的文件覆盖后无差异，自动不产生 hunk）
#   ③ **追加一个 commit**（0009）承载 v8 的增量
#   ④ `git format-patch` 全量导出 0001..0009
#
# 所以本脚本需要旧系列目录作为输入（默认 `/tmp/seriesin/old_series`）。
# 若旧系列已经不在手上：用本包 git 历史的**上一个版本**取 `patches/vllm-ascend/` 即可
# （这正是本次 v8 的做法：`git show <上一个发布 commit>:patches/vllm-ascend/<file>`）。
set -euo pipefail

SRC=${SRC:-/vllm-workspace/vllm-ascend}
BASE=${BASE:-46856f89e79c3011401e33663c60da37cd486d53}
WT=${WT:-/tmp/rel-va-$$}   # 独立目录，避免反复运行时 git worktree 的 prunable 陷阱
OUT=${1:-/tmp/rel-out-va}
OLD=${2:-/tmp/seriesin/old_series}

echo "[series] src=$SRC base=$BASE wt=$WT out=$OUT old=$OLD"
[ -d "$OLD" ] || { echo "[series][FAIL] 前一系列目录不存在：$OLD（见脚本头部说明）" >&2; exit 2; }
ls "$OLD"/*.patch >/dev/null 2>&1 || { echo "[series][FAIL] $OLD 里没有 *.patch" >&2; exit 2; }
cd "$SRC"

# --- 0) worktree + 先把旧系列 am 进去（保住历史） ---
git -C "$SRC" worktree prune
rm -rf "$WT" "$OUT"
mkdir -p "$OUT"
git worktree add --detach "$WT" "$BASE" >/dev/null 2>&1 || {
  echo "[series][FAIL] git worktree add 失败：$WT（先 git worktree prune）" >&2; exit 2; }
echo "[series] worktree ok"

cd "$WT"
git config user.name  "chiro2001"
git config user.email "chiro2001@163.com"

git am --keep-cr "$OLD"/*.patch >/dev/null
echo "[series] git am 旧系列 ok：$(git log --oneline "$BASE"..HEAD | wc -l) 个提交"

# --- 1) 用**发布包里的 patches/files/** 覆盖 worktree（含新增文件）---
#
# ⚠️ 不要从"活着的容器"取文件！容器是**按门控 env 挂载**的，例如
#    `indexer.py` 只在 CAND_MODE≠0 时才挂（默认 0 ⇒ 容器里是 stock 版），
#    从容器取会把 0004 的补丁**静默回退**掉。发布包的 patches/files/ 才是权威载荷。
#
# 下面是 Dockerfile 里 `inst <src> <target>` 的同一张落位表（保持同步！）：
PAYLOAD=${PAYLOAD:-/tmp/pkg/patches/files}
[ -d "$PAYLOAD" ] || { echo "[series][FAIL] 载荷目录不存在：$PAYLOAD" >&2; exit 2; }

A=vllm_ascend
MAP=(
  "ascend_forward_context.py:$A/ascend_forward_context.py"
  "dsa_v1.py:$A/attention/dsa_v1.py"
  "rope_dsv4.py:$A/ops/rope_dsv4.py"
  "token_dispatcher_moemask.py:$A/ops/fused_moe/token_dispatcher.py"
  "engram_gate.py:$A/models/deepseek_v41/engram_gate.py"
  "engram_hbm.py:$A/models/deepseek_v41/engram_hbm.py"
  "engram_hash.py:$A/models/deepseek_v41/engram_hash.py"
  "engram_jit_kernel.py:$A/models/deepseek_v41/engram_jit_kernel.py"
  "engram_plan_kernel.py:$A/models/deepseek_v41/engram_plan_kernel.py"
  "engram_device_index.py:$A/models/deepseek_v41/engram_device_index.py"
  "engram_graph.py:$A/models/deepseek_v41/engram_graph.py"
  "model.py:$A/models/deepseek_v41/model.py"
  "indexer.py:$A/models/deepseek_v41/indexer.py"
)
for pair in "${MAP[@]}"; do
  src=${pair%%:*}; tgt=${pair#*:}
  [ -f "$PAYLOAD/$src" ] || { echo "[series][FAIL] 缺载荷 $PAYLOAD/$src" >&2; exit 22; }
  mkdir -p "$(dirname "$WT/$tgt")"
  cp -f "$PAYLOAD/$src" "$WT/$tgt"
done
echo "[series] 载荷已覆盖 ${#MAP[@]} 个文件（来自 $PAYLOAD）"

# --- 2) 追加 v8 增量提交（0009） ---
git add -A
if git diff --cached --quiet; then
  echo "[series] NOTE: 工作树与旧系列无差异（可能重复运行）"
else
  git commit -q -F - <<'EOF'
feat(engram): device-side table lookup over host-mapped DRAM, captured in ACLGraph

Engram 的 INT8 表仍然常驻 host DRAM（206 GiB，进不了 HBM），但改由**设备算子直接索引**：
用 `aclrtHostRegister(..., MAPPED)` 把表所在的 mmap 注册成设备可寻址，
`torch.index_select` 即可直接读 host DRAM。于是 `d2h` 同步 / 分片 / all_gather /
all_to_all / broadcast / h2d 六条 host 路径整体消失，查表变成主图里的一张 ACLGraph。

新增文件：
  * engram_device_index.py —— HostMappedSafetensors / HostMappedEngramTable /
    DeviceNgramHash（向量化历史）/ 能力探测（**只做 host_register**，见下）
  * engram_graph.py —— 每个 batch shape 一张 ACLGraph，零拷贝 + 指针校验

三个关键设计点（都有实测支撑）：
  1. decode 与 prefill 分流：整表 gather 在 n=6/288 时都是 0.083 ms、**与表大小无关**，
     而 n=393216 时是 16.75 ms —— 代价按行发生。所以 decode 走整表直索（天然可捕获），
     prefill 仍走分段。
  2. 零拷贝捕获：图直接捕获在模型自己的常驻 buffer 上；曾试过"拷进私有 buffer"，
     **实测更差**（H2D 会阻塞等设备队列排空，route 0.35 -> 2.0 ms）。
  3. 每 batch shape 一张图 + 指针校验：bucket key 必须含 (n, n_reqs, block_width)；
     每次重放前比对四个输入的 data_ptr + shape，不一致就退回 eager。

门控：V41_ENGRAM_DEVICE_INDEX=auto（默认，探测通过才启用，否则静默回退 host 路径）
      / 1（强制，探测失败即抛错，A3 验收用这个）/ 0（强制关闭）

实测（A3-node1，同会话 A/B）：
  每步同步 host 时间 3.379 ms -> 0.058 ms（d2h 1.667 + hash 0.074 + route 1.638）
  decode 并发 1：29.5 -> 28.4 ms/step；并发 4：35.3 -> 32.1 ms/step
  单卡：n=12 时 2.024 -> 0.695 ms（含 0.285 ms ACLGraph 固定底噪）
  逐位一致：CPU/NPU/图三阶段 + 多设备回归 + 真实 layout 哈希，全部逐位相同

注意（两个掉过的坑，已写进注释，勿重复）：
  * 能力探测**只做 host_register**；在 worker 初始化期做 `int(t[0])` 会 segfault，
    端到端可读性交给独立进程工具 tools/probe_a2_hostmap.py。
  * 曾配套加过 model_runner_v1.py 的 device_metadata「自愈护栏」，**已撤销**：
    它的判据在正常运行中也会命中，提前释放 device metadata，
    64 并发实测 57/64 + 服务挂（ScatterElements 0x91 -> ERR00100 -> HCCL watchdog）。
EOF
  echo "[series] $(git log --oneline -1)"
fi

# --- 3) 导出全量系列（0001..0009），保留真实 commit hash/作者/日期 ---
git format-patch --no-signature -o "$OUT" "$BASE" >/dev/null
echo "[series] produced:"
ls -1 "$OUT"
git log --oneline "$BASE"..HEAD
