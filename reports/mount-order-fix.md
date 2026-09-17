# 启动器两处 bug 修复（`PYTHON_PGO` 与 `DRAFT_GRAPH` 接线）

> 2026-09-16 17:10 CST｜A3-node1｜由「实测 PGO 臂无法起服」暴露

---

## Bug：`PYTHON_PGO=1` 时启动器报 `MOUNTS: unbound variable`

### 现象

faB 臂（`PYTHON_PGO=1`）起服**容器从未出现**，`serve.out` 只有两行：

```
serve_a21.sh: line 187: MOUNTS: unbound variable
serve_a21.sh: line 192: MOUNTS: unbound variable
```

### 根因：条件块引用了**尚未初始化**的 `$MOUNTS`

我此前把两个条件挂载块（`HCCL_DET` guard、`PYTHON_PGO`）插在了 `MOUNTS=` 初始化**之前**：

```bash
# ✗ 错误顺序
if [ -n "$HCCL_DET" ]; then
  MOUNTS="$MOUNTS -e HCCL_DETERMINISTIC=$HCCL_DET"   # ← 此时 MOUNTS 还没定义
fi
if [ "$PYTHON_PGO" = "1" ] && [ -f "$PGO_LIB" ]; then
  MOUNTS="$MOUNTS -v $PGO_LIB:...:ro"                # ← 同上
fi
MOUNTS="-e V41_ENGRAM_JIT=..."                       # ← 初始化在这里
```

脚本有 `set -u` ⇒ 一旦条件为**真**，`$MOUNTS` 展开即报错。

### 为什么 faA（`PYTHON_PGO=0`）没暴露

**bash 不 evaluation 未命中的 `if` 分支体**。
faA 的 `HCCl_DET` 为空、`PYTHON_PGO=0` ⇒ 两个条件都是假 ⇒ 块体从未执行 ⇒ 不报错。

**⇒ 「同一脚本的另一个臂跑通了」不能证明这段代码没问题——只证明了那个臂没走到它。**

### 修复

把 `MOUNTS=` 初始化**上移到两个条件块之前**（`serve_a21.sh:190-200`），
并加注释说明这个顺序约束：

```bash
# [MOUNTS-ORDER-FIX] MOUNTS must be initialised BEFORE the conditional blocks below,
# otherwise `set -u` makes $MOUNTS unbound whenever their condition is true.
MOUNTS="-e V41_ENGRAM_JIT=..."
# [HCCL-DET] ...
# PYTHON_PGO ...
```

---

## 附带修复：`DRAFT_GRAPH` 接线（Phase 2）

新增开关与三个整文件挂载（`serve_a21.sh`）：

```bash
DRAFT_GRAPH=${DRAFT_GRAPH:-0}   # 0=stock(eager draft) 1=启用 draft ACLGraph
if [ "$DRAFT_GRAPH" = "1" ]; then
  MOUNTS="$MOUNTS -v $P/probe_draft/dsa_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py:ro"
  MOUNTS="$MOUNTS -v $P/probe_draft/dspark_proposer.py:/vllm-workspace/.../spec_decode/dspark_proposer.py:ro"
  MOUNTS="$MOUNTS -v $P/probe_draft/llm_base_proposer.py:/vllm-workspace/.../spec_decode/llm_base_proposer.py:ro"
  MOUNTS="$MOUNTS -e DSPARK_DRAFT_METADATA_MODE=sync"
fi
```

**注意**：`DRAFT_GRAPH=1` 会**覆盖** F3 的 `probe_dsa/dsa_v1.py` 挂载 ——
那是**故意的**，因为 `probe_draft/dsa_v1.py` 是「stock + 0004 + F3」的合并版（线 3 已做等价性验证）。

---

## Bug 3：dummy 起服漏传 `V41_DUMMY_WO_A_FIX`（我的测试脚本疏漏）

### 现象

`dgvB`（`DRAFT_GRAPH=1` + `LOAD_FORMAT=dummy`）起服失败：

```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
  File ".../attention/dsa_v1.py", line 1644, in _forward_o_proj
       o_proj_input = torch_npu.npu_transpose_batchmatmul(
```

### 排查过程（**以及一个被排除的错误假设**）

1. 先怀疑 `probe_draft/dsa_v1.py` 的 F3 兜底不完整 ⇒ **排除**：
   `_DUMMY_WO_A_FIX` 与 `_wo_a_2d` 的计数与可用的 `probe_dsa` 一致。
2. 再怀疑 `_forward_o_proj` 的 `if/elif` 链结构不同 ⇒ **排除**：
   逐行 diff 两个文件的 `_forward_o_proj`（1477-1660 vs 1498-1681）**完全相同**。
3. 全文件 diff ⇒ **只有 0004 的三处 hunk（46 行）**，即 `probe_draft` 是正确合并版。
4. **真因**：我的测试脚本 `dgvB_only.sh` **没有传**
   `EXTRA_ENV="V41_ENGRAM_WITH_DUMMY=1 V41_DUMMY_WO_A_FIX=1"`。

   `V41_DUMMY_WO_A_FIX` 是**专为 `LOAD_FORMAT=dummy` 加的兜底**（dummy 不调 `weight_loader`
   ⇒ `wo_a.weight` 停在声明的 2D 形状 ⇒ `npu_transpose_batchmatmul` 报维度错误）。
   我写新脚本时照抄了别人的 `RUN_ID/MAX_SEQS/LOAD_FORMAT` 却没照抄 `EXTRA_ENV`。

**⇒ 这不是补丁的问题，是我的测试脚本疏漏。**

### 教训

**dummy 实验的环境变量是一组，不能只抄一半。**
一个 dummy 会话至少需要：

| env | 作用 | 漏掉的后果 |
|---|---|---|
| `LOAD_FORMAT=dummy` | 不读权重 | — |
| **`V41_ENGRAM_WITH_DUMMY=1`** | 让 Engram host 路径在 dummy 下也建起来 | `[bneck]` 相位无数据 |
| **`V41_DUMMY_WO_A_FIX=1`** | `wo_a` 2D 形状兜底 | **`_forward_o_proj` 维度崩溃** |
| `V41_ENGRAM_PAD_SKIP`（可选） | — | 无 |

**建议固化**：把这些写成启动器里的 `DUMMY_OK=1` 一键开关，避免每次手抄。

---

## 方法论（第四次同类教训）

| # | 问题 | 教训 |
|---|---|---|
| 1 | `open()` 读运行时标志做图内开关 | **追踪期可见的运行时 Python 值**会炸 |
| 2 | `GATE_HOIST` 的 `tuple(tensor.shape)` 做缓存 key | 同上（unbacked SymInt） |
| 3 | **本轮** `MOUNTS` 顺序 | **dummy ≠ 真权重**、**臂 A 跑通 ≠ 臂 B 跑通**：条件分支的**执行**与否才是判据 |
| 4 | **本轮** 漏传 `V41_DUMMY_WO_A_FIX` | **dummy 的 env 是一组，不能只抄一半** |

---

## 证据

| 内容 | 路径 |
|---|---|
| 报错原文 | `/tmp/faB_serve.out`（两行 unbound variable） |
| 修复后启动器 | `serve_a21.sh:190-200`（`[MOUNTS-ORDER-FIX]`） |
| 备份 | `serve_a21.sh.bak_mountorder` |
| Phase 1 产物 | `probe_draft/{dsa_v1,dspark_proposer,llm_base_proposer}.py` + `make_draft_files.py` |
