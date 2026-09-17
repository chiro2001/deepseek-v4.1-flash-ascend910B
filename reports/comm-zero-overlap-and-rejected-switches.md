# 通信零重叠 + 一批开关的代码级否决（2026-09-16 01:50–02:00）

> 场地：A3-node1 chips 8-15，容器 `dsv41-a21-perf`，端口 8020
> 配置基线：`static_kernel=1` + `npugraph_ex=1` + `enable_fused_mc2=1` + `multistream_overlap_shared_expert=false`
> + Engram(int8, gate CHUNK=0) + local-owner + hash fast + jemalloc + `SP_TOKENS=7`

---

## 1. 最重要发现：通信与计算 **零重叠**

用设备时间线（`/tmp/op_summary_fmc2.csv`，32K decode，86 step）把 COMMUNICATION 与其余算子的区间并集求交：

| 项 | 总时长 | ms/step | 占窗口 |
|---|---|---|---|
| busy（全部 task 并集） | 2508.8 ms | 29.172 | 84.8% |
| **compute**（非通信） | 2204.2 ms | 25.630 | 74.5% |
| **comm** | 304.8 ms | **3.544** | 10.3% |
| **重叠 comm∩compute** | **0.2 ms** | **0.002** | **0.0%** |
| comm 暴露（未重叠） | — | **3.542** | — |
| FREE | — | **5.247** | — |

**读数**：`hcom_*`（allReduce / allGather / alltoallv / broadcast）与计算**完全串行**——3.54 ms/step 的通信时间 100% 暴露在关键路径上。
结合 FREE 5.25 ms/step，**非计算时间合计 8.8 ms/step**（占 34.4 ms 的 26%）。

这与 `step_trace_time.csv` 的历史读数一致（ALLTOALL 代：`Overlapped=0.0`）。
另外把通信本身再拆：`compute_union + comm_union = 29.17 ms/step`，与 busy 完全相等 ⇒ **两个集合无缝拼接、零重叠**，这个结论是自洽的。

**含义**：任何"让通信与计算并行"的机制（多流、overlap、融合）都是**理论上的下一块大蛋糕**（上限 3.5 ms/step ≈ −10%），比继续榨 host 侧（1.6–2.2 ms）更有价值。

---

## 2. 逐条否决的开关（全部有代码级依据，省下多次重启）

### 2.1 MegaMoe（`enable_fused_mc2=2`）—— **被配置约束否决**

动态库 `cann_ops_transformer` 在本容器**存在**（`importlib.util.find_spec` 为真），看似可用。但
`vllm_ascend/ascend_config.py:763 _is_megamoe_supported_by_config()` 要求：

```
moe_intermediate_size ∈ [1024, 3072] 且 moe_intermediate_size % 512 == 0
```

本模型 `moe_intermediate_size = 2304`，`2304 % 512 = 256 ≠ 0` ⇒ 返回 False。
而 `ascend_config.py:581` 的逻辑是：

```python
if self.enable_fused_mc2 == 1 and _MEGA_MOE_SUPPORTED and not self._is_megamoe_supported_by_config(vc):
    self.enable_fused_mc2 = 0        # ← 静默退回 0，丢掉我们已拿到的融合路径
```

⇒ **`enable_fused_mc2=2` 会静默退回 0**（连 fused dispatch_ffn_combine 都没了）。**禁止设置**。

### 2.2 `mc2_comm_alg=hierarchy` —— **与 fused MC2 互斥**

`ascend_config.py:_validate_mc2_comm_alg()`：

```python
if self.enable_fused_mc2:
    raise ValueError("fused mc2 op cannot be used with hierarchy communication. "
                     "Please set additional_config.enable_fused_mc2 to 0.")
```

⇒ 只能二选一。已在 A3-node1 起 A/B（`FUSED_MC2=1` vs `FUSED_MC2=0 + MC2_HIER=1`）实测中。

**实测结果（2026-09-16 01:56 起服）：`mc2hier` 臂起服直接失败。** 根因不是互斥检查，而是算子层的硬约束：

```
(Worker_TP7_EP7) ERROR ... RuntimeError: npu_moe_distribute_dispatch_v2:
  .../MoeDistributeDispatchV2KernelOpApi.cpp:194 NPU function error: call aclnnMoeDistributeDispatchV2 failed
  [ERROR] ERR00100 PTA call acl api failed.
  Invalid_Input(EZ0004): Parameter params shape of MoeDistributeDispatchV2 is required, but it is empty.
  TraceBack: epWorldSize should be 16 Aligned, but got 8.
             [FUNC:CheckGroupAttrParams][FILE:moe_distribute_dispatch_v2_tiling.cpp][LINE:585]
  ... Get attr and set tiling data failed. [FUNC:MoeDistributeDispatchA3TilingFuncImplPublic]
```

⇒ **`MoeDistributeDispatchV2` 要求 `epWorldSize` 是 16 的倍数**（16/32/48/64…），我们 **EP=8** 不满足。
**MC2/hierarchy 这条路线在 EP=8 下物理上不可用，方向关闭**（不需要再试 `mc2_comm_alg=fullmesh*`；
它们是同一个 MC2 dispatcher 的不同 comm_alg，tiling 前置约束相同）。

注意区分：我们正在用的 `enable_fused_mc2=1` 走的是 **`DispatchFFNCombineW4A8`**（另一套融合算子，EP=8 可用），
**不受**这条 16 对齐约束限制 —— 所以现有融合路径依然有效，只是不能再往"非融合 MC2"方向走。

起服日志：`logs/perf/a21_mc2hier_20260916_0149_serve.log`（3138 行，8 个 rank 全部在同一处失败）。

### 2.3 `dynamic_spec_config`（动态推测长度）—— **在 async scheduling 下是死开关**

机制上看很对口：DSpark 的 confidence head 给出逐 token 接受概率 → 累计存活率 → 自适应 verify 长度。
我们的逐位置接受率显示 pos5/pos6 **恒为 0** ⇒ 每步有 2/8 的验证是纯浪费，正是该开关的目标。

但**消费路径在 async 下不存在**：

- `vllm_ascend/worker/model_runner_v1.py:2060 take_draft_token_ids()` 是唯一读取
  `dynamic_spec.num_verify_tokens` 并据此截断 draft tokens 的地方。
- 上游 `vllm/v1/engine/core.py:617 post_step()`：
  `if self.check_for_draft_tokens and not self.async_scheduling and model_executed:` → 我们 `async_scheduling=True` ⇒ **跳过**。
- 另一处 `core.py:723`（`step_with_batch_queue`，即 async 路径）**只在 `deferred_scheduler_output` 非空时**调用，
  且只用于 `update_draft_token_ids_in_output`（grammar bitmask），**不回传 scheduler 的 verify 长度**。

我们实测日志确认 async 是开的：
`[admission_gate] max_concurrent_batches=2 (async scheduling/PP)`。

⇒ 在无 structured output 的常规推理下，`dynamic_spec_config.method="dspark"` **不改变任何调度行为**。
（若要启用，必须 `--no-async-scheduling`，而 async 本身是当前性能配置的一部分 —— 需作为独立实验，且注意它还可能触发 16 步一次的 `.item()` 同步。）

### 2.4 `use_sequence_parallel_moe`（SP-MoE，可消 TP allreduce）—— **前置条件不满足**

`vllm/config/parallel.py:673`：

```python
return (self.all2all_backend in ("allgather_reducescatter", "deepep_*", "mori_*", "nixl_ep")
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
        and self.data_parallel_size > 1)      # ← 要求 DP > 1
```

我们是 `DP=1` ⇒ 不成立。而 `replace_allreduce` 这条快路径（可跳过 MoE 的 TP all-reduce）只有在 SP-MoE 下才有意义
（`PrepareAndFinalizeWithMC2.prepare(replace_allreduce=...)` 的注释明确写"输入已是 TP 分片"）。

### 2.5 `CPU_BIND=1` —— **实测更慢（否决）**

A3-node2 同配置对照：8K **39.86** vs 34.40、32K **40.68** vs 34.35 ms/step。KV 几乎不变（3,389,665 vs 3,388,441）。

### 2.6 非-mtpq（BF16 draft）—— **KV 被压到 3M 以下（否决）**

`v41-w4a8-engram-dr-vision` 起服实测 `GPU KV cache size: 2,700,814 tokens < 3M`。
mtpq 省下的 2.42 GB 正是 KV 跨过 3M 门槛的来源 ⇒ 该路线受 KV 约束否决。

---

## 3. `SP_TOKENS` 的"同一目录陷阱"（对新读者很重要）

`SP_TOKENS=7` 时的捕获桶大小 = `max_seqs × (S+1)` = `4 × 8 = 32`。
我此前做的"非 mtpq 对照"（`MODEL=v41-w4a8-engram-dr-vision`）实测 A ≈ 1.08–1.55，远低于常值 2.7–2.9。
**该会话的 `SP_TOKENS` 与基线相同（7）**，所以差异不能归给桶大小 —— 但那条对照本身因 KV 不达标已作废，其 A 数字**不可用于任何结论**（请求内容是 295K-token 古典中文续写，内容可预测性极低）。

---

## 4. 剩余可动的大块（按设备账排序）

| 项 | ms/step | 性质 | 候选手段 |
|---|---|---|---|
| **通信零重叠** | **3.54 暴露** | 结构性 | 多流 overlap / 融合进图 / `hierarchy`（A/B 中） |
| `DispatchFFNCombineW4A8` | 8.48 | MoE 融合算子 | 已是最优融合形态；只能靠 MC2 算法或 EPLB |
| `hcom_allReduce_` | 4.83（148 次/步） | TP 通信 | SP-MoE（需 DP>1，不可用）/ 减少 allreduce 次数 |
| FREE | 5.25 | host/调度 | 主要已被本地化（Engram route-pipe 已否决） |
| HcPre + HcPost | 3.08 | 逐元素算子 | 跨层融合（glm5next 有参考实现） |
| host post-D2H | 1.6–2.2 暴露 | host | 路线已多轮收窄 |

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 零重叠计算 | `/tmp/ov.py`（本次），输入 `/tmp/op_summary_fmc2.csv` |
| 逐位置接受率 | `logs/perf/a21/p42_t4_quote_{8192,32768,131072}_fmc2b_*.jsonl` 的 `accepted_per_pos` |
| MC2 sweep | `logs/perf/a21_sweep_mc2_*.log`、`logs/perf/serve_{fmc2,mc2hier}_*.log` |
| async 证据 | `logs/perf/a21_fmc2_0019_serve.log` 的 `max_concurrent_batches=2 (async scheduling/PP)` |
| 代码依据 | 容器内 `vllm_ascend/ascend_config.py:581,763`、`vllm_ascend/worker/model_runner_v1.py:2060`、`vllm/v1/engine/core.py:617,723`、`vllm/config/parallel.py:673` |
