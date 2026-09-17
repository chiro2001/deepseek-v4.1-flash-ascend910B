# 决策：DSpark draft 默认入图（SPEC_EAGER_OPT=0）

> ## ⚠️ [更正 2026-09-16 16:50] 本决策**实测为 no-op**，理由如下
>
> **DSpark 的 draft 从来就不在 ACLGraph 里，且开关无法改变这一点。**
> `vllm_ascend/spec_decode/dspark_proposer.py:75` 无条件硬置：
> ```python
> # DSpark runs eager only (Ascend cudagraph unsupported on this path).
> self.use_cuda_graph = False
> ```
> 它覆盖了基类 `llm_base_proposer.py:218` 按 `enforce_eager` 算出的值，
> 于是 `llm_base_proposer.py:604` 的 `ACLGraphWrapper` 分支**永不执行**。
>
> **三条实测证据**：① 全日志 grep `Wrapping draft model` = **0 次**；
> ② `aclmdlRIExecuteAsync` = **118 次而前向约 118 步 ⇒ 每步只有 1 次图重放（只有 target）**；
> ③ draft 特征算子（`FloorMod`/`FloorDiv`/`SelectV2`/`ArgMaxV2`）**全部 `OP State=dynamic`**。
>
> 这也解释了 a22 子代理的 A/B/A/B2 为何测出「−0.41 ms 但落在 +0.91 漂移内、不可分辨」——
> **它测的是一个 no-op。**
>
> **⇒ 本决策作废**（保留 `SPEC_EAGER_OPT=0` 作为默认值本身无害，但**不要**把它当作收益项）。
> 详见 `reports/two-corrections-draftgraph-gatehoist.md`。
>
> ---

> 2026-09-16 09:40 CST｜决策人：用户｜落地：A3-node1 `scripts/serve_a21.sh`、A3-node2 `$W/scripts/serve_a22_v2.sh`

## 决策

**默认让 DSpark draft 入图**（`--speculative-config ... enforce_eager=false`），
即 `SPEC_EAGER_OPT=0`。两台机器的启动器都已改默认值（含注释说明原因）。

## 理由

| # | 依据 | 说明 |
|---|---|---|
| 1 | **目标机 A2 的 CPU 很弱** | 我们本机 A/B 只测到 −0.41 ms/step（4 点均值），落在会话漂移（±1 ms）内所以"不可分辨"。但那段差距的物理来源就是 **host 侧逐算子下发**——CPU 越弱，暴露越多。A2 上这段收益会被放大。 |
| 2 | **本机 A/B 方向一致** | A(36.58) / B(35.93) / A′(37.49) / B2(37.31) ms/step；B 的两轮都在 A 附近或更好，**没有任何一轮显示 draft 入图更差**。 |
| 3 | **正确性已验证** | 计数探针（`1 2 3…25` 续写 48 token）：A′ 与 B2 **逐字节一致且 sha 相同**（`f5617204e526f297…`，与 A3-node1 参考输出一致）；128K 接受长度与 A **逐位相同**。未发现 draft 入图导致的任何损坏。 |
| 4 | **架构上更干净** | 入图后 draft 的 3 层前向不再需要在每个 decode step 由 host 逐个下发算子，图形态与 target 侧一致。 |

## 注意事项

1. **回退方式**：`SPEC_EAGER_OPT=1 bash scripts/serve_a21.sh`（或编辑启动器默认值）。
2. **本机测量不足以证明收益**：在强 CPU 机器上这是"不可分辨"的量级。**A2 上的复现包应把这条作为"预期在弱 CPU 上收益更大"的项**，而不是已证实的定量收益。
3. **不要再用"128K 同 prompt 三次一致"做正确性判据**：本机已验证 >16384 上下文的输出本身非确定（见 `reports/ctx-nondeterminism.md`），该判据无判别力。正确性请用 **≤16384 的确定性区间** 或**计数探针**。
4. 相关证据：`reports/draft-graph-allgather.md`（完整 A/B/A/B2 表、启动器适配、依赖 md5）、`logs/perf/dg_ab/*.jsonl`。
