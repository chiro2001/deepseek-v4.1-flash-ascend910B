# 启用 DSpark draft 图捕获：规划

> 2026-09-16 17:00 CST｜用户决定「dspark 应该需要使能其图捕获」
> 依据：`reports/two-corrections-draftgraph-gatehoist.md`（draft 从未入图）、
> 线 3 的 `eager-op-attribution.md`（eager 提交 718 次/前向）

---

## 1. 现状（已核实）

| 事实 | 证据 |
|---|---|
| **硬禁在 `dspark_proposer.py:75`** | `self.use_cuda_graph = False`，vendor 注释 *DSpark runs eager only (Ascend cudagraph unsupported on this path)* |
| 因此 `llm_base_proposer.py:604` 的 `ACLGraphWrapper` 分支**永不执行** | 日志 `Wrapping draft model` = 0 次；`aclmdlRIExecuteAsync` = 118 次 / 118 步 ⇒ 每步只 1 次重放（只有 target） |
| **解除硬禁的补丁已存在** | `wt-graph/patches/draft_graph/fix2_files/.../dspark_proposer.py:100` 改成按 `enforce_eager` 计算 |
| **四个前置缺陷的修复也已存在** | `0002`（capture metadata / D1）、`0004`（start_pos 常驻 buffer / D2）、`0005`（executor contract / D3）、`0006`（target_positions 常驻 buffer / D4） |
| **历史上强开 draft 图会崩** | A2 报告：`ms/round 28.1 达标但 tok/round 崩到 1.0`（draft 全被拒）—— 正是 D2/D4 的形态 |

---

## 2. ⚠️ 集成风险（必须先解决，否则会静默回退其它补丁）

**`0004` 改的文件在 A3-node1 上已被 F3 占用**：

| 文件 | 谁在改 |
|---|---|
| `attention/dsa_v1.py` | **F3（`O_PROJ_2D`）已在用** + `0004` 也要改它 |
| `spec_decode/dspark_proposer.py` | A3-node1 **未挂载** ⇒ 可直接用 fix2 版 |
| `spec_decode/llm_base_proposer.py` | A3-node1 **未挂载** ⇒ 可直接用 fix2 版 |

**已核对可行性**：`0004` 的 3 个锚点在含 F3 的 `probe_dsa/dsa_v1.py` 里**都在**
（`self.start_pos_prefill: torch.Tensor = torch.zeros(` ×1、`has_prefill = self.num_prefills > 0` ×2、`start_pos=self.seq_lens[:num_reqs] - seq_lens_q,` ×1），
且 F3 的改动位于 line 147-160 与 1571+，**与 0004 的三处（617 / 1231 / 1354）不重叠** ⇒ **可共存**。

**⇒ 绝对不能直接挂 `probe_draftmeta/dsa_v1.py`**（它是基于 stock 的整文件，
挂上去会**丢掉 F3 + 所有 dummy 兜底**）。

---

## 3. 执行计划（五个阶段）

### Phase 1：离线集成（无 GPU，可在任何机器）
1. 把 `0004` 的三个 hunk 合并进 `probe_dsa/dsa_v1.py`（**保留 F3 与 dummy 兜底**）
2. 从 fix2 取 `dspark_proposer.py` 与 `llm_base_proposer.py`（**不加探针**，只取功能修复）
3. 三者都过 `py_compile`；写**锚点唯一性自检**（可重复执行的幂等脚本）
4. **关键自检**：确认合并后的 `dsa_v1.py` **仍含 F3 的 `_o_proj_2d_enabled` 与 `_DUMMY_WO_A_FIX`**

### Phase 2：启动器接线（A3-node1）
新增三个挂载 + 一个开关：

```bash
DRAFT_GRAPH=${DRAFT_GRAPH:-0}     # 0=stock(eager draft) 1=启用 draft 图
# 1) probe_dsa/dsa_v1.py           （已挂载，含 F3 + 0004）
# 2) probe_draft/dspark_proposer.py      → spec_decode/dspark_proposer.py
# 3) probe_draft/llm_base_proposer.py    → spec_decode/llm_base_proposer.py
```
并确保 `SPEC_EAGER_OPT=0`（`enforce_eager=false`）—— 这是 `use_cuda_graph` 计算的前置条件。

### Phase 3：起服指纹（**三条缺一不可**）
| # | 指纹 | 期望 |
|---|---|---|
| 1 | 日志 `[spec_decode/base] Wrapping draft model with ACLGraphWrapper` | **出现**（stock 下为 0 次） |
| 2 | `aclmdlRIExecuteAsync` 次数 ÷ 步数 | **≈ 2**（target + draft）；stock 下是 1 |
| 3 | draft 特征算子（`FloorMod`/`FloorDiv`/`SelectV2`/`ArgMaxV2`）的 `OP State` | **static**（stock 下全 dynamic） |

### Phase 4：正确性（**这是历史上崩溃的地方，必须先过**）
1. **确定性探针**（≤16384 区间，同 prompt 连发 N 次，用 `torch.equal`）
2. **接受长度 A**：与 stock 同会话对比 —— 历史上强开时 A 崩到 1.0
3. 若 A 崩 ⇒ 检查 D2/D4 的修复是否真的生效（`start_pos`/`target_positions` 地址是否稳定）
4. 通过后交正确性线做 GSM8K-100 + Vision

### Phase 5：性能（device 账目 + 真权重）
| 指标 | 判据 |
|---|---|
| eager 提交次数（`OP State=dynamic` 的算子总和） | 应从 **718/前向** 大幅下降 |
| `api_statistic` 的 `launch` / `aclrtLaunchKernelWithHostArgs` | 应从 **748/732 次/pass** 下降 |
| 128K ms/step | 真权重 8 发中位 |

**收益上界**：`launch` 的量级是 **5.75 ms/pass**（748 次 × 7.69 µs），
若 draft 占其中 ~78%（718/918 的 dynamic 占比）⇒ **理论上界 ≈ 4–5 ms/pass**；
实际取决于暴露率（部分提交已与设备执行重叠）。

---

## 4. 风险与回退

| 风险 | 缓解 |
|---|---|
| **D2/D4 修复不完整 ⇒ A 崩到 1.0**（历史形态） | Phase 4 的接受长度对比是硬门槛；不通过就不采纳 |
| 图捕获期缺 metadata（D1） | `0002` 的 capture metadata 修复 + `DSPARK_GRAPH_DEBUG=1` 的断言 |
| device-metadata 契约冲突（D3） | `0005` 的 sync 契约（`DSPARK_DRAFT_METADATA_MODE=sync`） |
| 与现有 6 个补丁冲突 | `DRAFT_GRAPH=0` 是默认，**一键回退**；F3 兼容性已在 Phase 1 自检 |
| 显存（多一张 draft 图） | 记录 `GPU KV cache size`，若跌破 3M 则否决 |

---

## 5. 与其它线的边界

| 线 | 角色 |
|---|---|
| **线 2（我）** | Phase 2/3/5 的 A3-node1 起服与验证 |
| **线 3（单算子）** | Phase 1 的离线集成（它熟悉那批补丁）+ 单卡预检 |
| **线 1（正确性）** | Phase 4 的精度门；**同时继续 A 的第三源**（线 3 已给出强候选） |

**并行不冲突**：Phase 1 是纯文件操作（不需要 GPU），我这边继续跑 `final_all.sh`。

---

## 6. 证据

| 内容 | 路径 |
|---|---|
| 硬禁与解除 | `dspark_proposer.py:75`（stock）/ `fix2_files/.../dspark_proposer.py:100` |
| 四个前置修复 | `patches/draft_graph/000{2,4,5,6}-*.patch` |
| 整文件（含探针） | `probe_draftmeta/{dspark_proposer,llm_base_proposer,dsa_v1}.py` |
| 历史崩溃形态 | `A2_复现报告.md` §3.2、`reports/draftgraph-defect-ledger.md` |
| eager 提交账目 | `A3-node2:~/handoff/reports/eager-op-attribution.md` |
