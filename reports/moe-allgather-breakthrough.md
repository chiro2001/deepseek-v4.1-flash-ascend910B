# ★ 突破：MoE 改走 AllGather 路径，128K −4.07 ms（−10.3%）

> 2026-09-16 06:10–06:30｜A3-node1 chips 8-15
> **这是本阶段最大的单项收益**，且**推翻了此前"MoE 已到访存下限"的错误结论**

---

## 1. 起因：cannbot 措施3 的判据命中了 MoE 融合算子

对 `DispatchFFNCombineW4A8` 取完整的 profile 比例字段：

```
Task Duration(us)    : 250.5
aic_total_cycles     : 9240303
aic_mac_ratio        : 0.007     ← MAC 只占 0.7%
aic_scalar_ratio     : 0.444     ← ★ 标量 44.4%  > 30%（cannbot 措施3 判据）
aiv_scalar_ratio     : 0.336     ← ★ 标量 33.6%  > 30%
aic_mte1_ratio       : 0.039     aic_mte2_ratio: 0.045
aiv_vec_ratio        : 0.003
aic_icache_miss_rate : 0.023     aiv_icache_miss_rate: 0.096
cube_utilization(%)  : 85.376
```

cannbot 措施3 原文：

> **适用场景：op_summary\*.csv 中的算子的 `aic_scalar_ratio` 或 `aiv_scalar_ratio` 占比超过 30%。**
> 预期收益：算子性能提升，显著降低 scalar 耗时占比。

⇒ `DispatchFFNCombineW4A8` **是标量受限（scalar-bound）**，不是访存受限。

---

## 2. 根因：TP 把 token 切成「每 rank 1 个」，而标量开销是固定的

`PrepareAndFinalizeWithMC2.prepare()`（`prepare_finalize.py:288-301`）：

```python
if self.tp_size > 1:
    split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
    hidden_states = split_hidden_states[self.tp_rank]
```

单流 spec decode 每步 **8 个 token**（1 sampled + 7 draft），TP=8 ⇒ **每个 rank 只拿到 1 个 token**。

而 `DispatchFFNCombineW4A8` 需要处理 **48 个本地专家**（384 / EP=8）的分派结构：
expert histogram、prefix-sum、permute/unpermute 的索引计算**全是标量**，
且这些开销**与 token 数几乎无关**。

⇒ **为 1 个 token 付 48 个专家分派结构的固定标量成本** —— 这就是 44% 标量的来源。

---

## 3. 解法：强制走 `AllGatherCommImpl`

A3 的硬件 profile 把 `moe_comm_policy` 设为 `FUSED_OR_CAPACITY`，
而 `_select_fused_or_capacity_moe_comm_method` 在 `enable_fused_mc2==1` 时**无条件短路到 FUSED_MC2**
——于是永远走不到 AllGather。

`AllGatherCommImpl` 的语义是：**all-gather 让每个 rank 都持有全部 token**，
再各自计算本地专家。这样 **每 rank 的 token 数从 1 变成 8**，标量开销被摊薄。

### 3.1 实现（env 门控，20 行）

`probe_moe/ascend_forward_context.py`（容器内 `vllm_ascend/ascend_forward_context.py` 的补丁版）：

```python
def _select_fused_or_capacity_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity):
    # [V41-MOE-AG] 实验开关：强制走 AllGatherCommImpl
    import os as _os_ag
    if _os_ag.environ.get("V41_MOE_COMM_ALLGATHER", "0") == "1":
        return MoECommType.ALLGATHER
    if use_cann_megamoe(vllm_config):
        return MoECommType.FUSED_MC2
    ...
```

通过 `serve_a21.sh` 挂载（`AFC_FILE`）+ `MOE_AG=1` 注入 env。

---

## 4. 实测结果

### 4.1 三点对照（3 发中位，unprofiled 客户端墙钟）

| 上下文 | MC2（旧） | **AllGather（新）** | Δ |
|---|---|---|---|
| 8K | 34.14 | **32.91** | **−1.23（−3.6%）** |
| 32K | 36.93 | **35.58** | **−1.35（−3.7%）** |
| **128K** | **39.39** | **35.32** | **−4.07（−10.3%）** |

**128K 的 tok/s：73.4 → 77.4**（A=2.734）。

### 4.2 正确性

同 prompt / `temperature=0` / `seed=1234` / 48 token：

```
MC2      : ' 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49'
AllGather: ' 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49'
一致: True
```

### 4.3 硬指标（全部改善或保持）

| 指标 | MC2 | **AllGather** |
|---|---|---|
| **KV 容量** | 3,388,563 | **4,160,767（+22.8%）** |
| Maximum concurrency | 3.39× | **3.97×** |
| static kernel 静默降级 | 0 | 0 ✅ |
| DSpark | S=7 | S=7 ✅ |
| Engram local-owner | on | on ✅ |
| jemalloc | 命中 | 命中 ✅ |

**KV 容量 +22.8% 的原因**：AllGather 路径不再需要 MC2 的通信 workspace（HCCL buffer 等），
这部分显存回到 KV 池。

---

## 5. 为什么之前没发现

我在上一轮报告 `draft-eager-and-moe-roofline.md` 里写了：

> **`DispatchFFNCombineW4A8` 是纯访存受限**（`aic_mac_ratio=0.7%`）… 9.4 ms/step 已接近该形态下限。

**这个结论是错的**：我只看了 `aic_mac_ratio`（0.7%）就推断"不發 MAC ⇒ 访存受限"，
**忽略了 `aic_scalar_ratio`（44.4%）**。实际上它既不是 MAC 受限、也不是访存受限，
而是**标量（索引/地址计算）受限**。

教训：**判 roofline 必须把 `*_ratio` 全部看完**（mac / scalar / mte1 / mte2 / fixpipe / vec），
只看一个比值会得出相反结论。而 `aic_scalar_ratio > 30%` 这条判据**就写在 cannbot 措施3 里**，
我此前读过但没把它对到我们的算子上。

---

## 5.1 设备账对照（同口径，客户端实测步数）

| 项 | MC2 | **AllGather** | Δ |
|---|---|---|---|
| compute | 30.36 | **24.67** | **−5.69** |
| busy | 33.99 | **28.09** | **−5.90** |
| comm | 3.63 | 3.42 | −0.21 |
| **FREE** | 6.98 | **7.80** | **+0.82** |
| 折合步长 | 40.98 | **35.89** | −5.09 |

### MoE 算子的直接对照（每步）

| 实现 | 算子 | ms/step |
|---|---|---|
| MC2 | `DispatchFFNCombineW4A8` | **9.41** |
| **AllGather** | `GroupedMatmulSwigluQuantV2` | **3.10** |
| | `GroupedMatmul` | **1.50** |
| | `MoeInitRoutingV3` | **0.55** |
| | **小计** | **5.15** |

⇒ **MoE 从 9.41 → 5.15（−4.26 ms/step，−45%）**，且 comm 几乎不变
（AllGather 的通信成本被 MC2 的 all-to-all 抵消）。

**FREE 反而 +0.82** ⇒ 瓶颈转向 host/调度 ⇒ **此前因"设备已满"而否决的 host 侧优化值得全部重测**。

---

## 5.2 连锁重测（AllGather 形态下，此前被 MC2 互斥/形态差异否决的开关）

每个配置独立起服，8K/32K/128K × 3 发中位：

| 配置 | 环境 | 8K | 32K | **128K** | 判定 |
|---|---|---|---|---|---|
| **`ag_base`** | `MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0` | 31.99 | **32.15** | **35.10** | **★ 最优** |
| `ag_ms` | `MOE_AG=1 FUSED_MC2=0 MULTISTREAM=1` | 32.56 | 33.19 | 35.41 | ❌ 更差（多流重叠收益 < 关掉 fused 的损失） |
| `ag_rtcore3` | `+ EXTRA_ENV=MAX_RUNTIME_CORE_NUMBER=3` | **31.72** | 34.28 | 35.87 | ❌ 32K/128K 更差 |
| `ag_hccl2k` | `+ EXTRA_ENV=HCCL_BUFFSIZE=2048` | 32.14 | 34.19 | 36.35 | ❌ 全面更差 |

⇒ **`ag_base` 是明确最优**，8K 的 −0.27 在噪声内。这三项在 MC2 时代也都被否决过，
换成 AllGather 后**结论不变**。

---

## 5.3 新瓶颈已转移：`hcom_allReduce_` 成为最大单项

AllGather 形态下的算子 Top（每步）：

| 算子 | ms/step | 次/步 | 对比 MC2 时代 |
|---|---|---|---|
| **`hcom_allReduce_`** | **6.03** | 165.6 | ↑（MC2 时 4.48） |
| `GroupedMatmulSwigluQuantV2` | 3.10 | 40.0 | 新（替代了 DispatchFFNCombine） |
| HcPre | 2.41 | 80.0 | ↓（MC2 时 2.38） |
| `QuantBatchMatmulV3` | 2.22 | 210.2 | 持平 |
| `TransposeBatchMatmul` | 1.90 | 40.0 | 持平 |
| `QuantLightningIndexerV2` | 1.80 | 7.4 | 持平 |
| `MatMulV2` | 1.75 | 65.1 | 持平 |
| `GroupedMatmul` | 1.50 | 40.0 | 新 |
| `SparseFlashMla` | 1.32 | 37.2 | 持平 |
| `MoeInitRoutingV3` | 0.55 | 40.0 | 新 |

**`hcom_allReduce_` 6.03 ms/step 现在是最大单项**——因为 AllGather 路径下 MoE 输出的
all-reduce 不再被融合进 `DispatchFFNCombine`，变成独立的 165.6 次 allReduce。

（`comm` 的区间并集仍是 3.42 ms/step，说明这 6.03 内部有重叠；`comm∩compute` 仍未测得重叠。）

---

## 6. 由此打开的**新空间**（下一步候选）

既然 MoE 的真实瓶颈是"每 rank 的 token 数 vs 专家数"，那么还有几个维度可试：

| # | 方向 | 依据 | 预估 |
|---|---|---|---|
| 1 | **`MAX_SEQS` 与 AllGather 的组合** | AllGather 下 graph 捕获尺寸语义变化（`mc2_tokens_capacity` 不再约束） | 未测 |
| 2 | **`enable_shared_expert_dp=1`** | 在 AllGather 下共享专家可走 DP 复制权重路径；此前因与 TP 切分冲突未试 | 未测 |
| 3 | **重新评估 `TASK_QUEUE` / `HCCL_BUFFSIZE`** | 通信形态变了（all-gather 取代 all-to-all），之前的结论可能不再成立 | 未测 |
| 4 | **重测 `SP_TOKENS`** | 标量开销摊薄后，更大的 S 可能重新变得有利 | 未测 |
| 5 | **`multistream_overlap_shared_expert`** | 与 `enable_fused_mc2` 互斥，而 **AllGather 路径可能不需要 fused_mc2** ⇒ 也许可以同时开 | **最有希望** |

---

## 6.1 最终验收（三次独立测量的一致性）

| 会话 | 8K | 32K | **128K** |
|---|---|---|---|
| AllGather #1（06:24） | 32.91 | 35.58 | 35.32 |
| AllGather #2（06:52） | **31.99** | **32.15** | **35.10** |
| AllGather #3（07:44，最终配置） | 32.54 | 33.55 | **35.14** |

**128K 稳定在 35.1 ms/step（±0.2）**，A=2.763 ⇒ **78.6 tok/s**。

### 与突破前的对比（本阶段总账）

| 指标 | 突破前（MC2） | **现在（AllGather）** | 改善 |
|---|---|---|---|
| 8K | 34.14 | **32.54** | −1.60（−4.7%） |
| 32K | 36.93 | **33.55** | −3.38（−9.2%） |
| **128K** | 39.39 | **35.14** | **−4.25（−10.8%）** |
| **128K tok/s** | **70.1** | **78.6** | **+12.1%** |
| **KV 容量** | 3,388,563 | **4,160,155** | **+22.8%** |
| MoE 算子 | `DispatchFFNCombineW4A8` 9.41 | `GMM+routing` 5.15 | **−45%** |

---

## 6.2 这次突破的方法论意义

**我此前写下的"MoE 已到访存下限、9.4 ms/step 不可动"是错的**，根因是
**只看 `aic_mac_ratio`（0.7%）就推断"不發 MAC ⇒ 访存受限"**，
漏看了 `aic_scalar_ratio=0.444` / `aiv_scalar_ratio=0.336`。

而 **cannbot 措施3 早就把判据写清楚了**："`aic_scalar_ratio` 或 `aiv_scalar_ratio` 占比超过 30%"。
我读过这条却没想到去对我们的算子。

**推论**：现在设备账里的其它"看起来是下限"的项，都值得用同样的方式重新审一遍
——特别是新的最大单项 **`hcom_allReduce_` 6.03 ms/step（165.6 次/步）**：
它同样是"次数×时长"的账，需要检查是否存在**结构性的次数冗余**（例如每层 4 次是否都必要）。

---

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| 补丁文件 | `probe_moe/ascend_forward_context.py`（md5 `6cccd4259bd65c907ef9d9dd42a83dca`） |
| 原始文件备份 | `/tmp/afc_orig.py`、`probe_moe/ascend_forward_context.orig.py`（md5 `debc3f93e2f9c038b45e4b0fccdb5729`） |
| 启动器 | `scripts/serve_a21.sh`（`AFC_FILE` / `MOE_AG`，默认 `MOE_AG=1`） |
| AllGather 会话 | `logs/perf/a21_ag_0615_serve.log` |
| AllGather 测量 | `logs/perf/a21/measure_ag.log`、`logs/perf/a21/p42_t4_quote_*_ag_*.jsonl` |
| 标量判据来源 | `/tmp/op_summary_fixed.csv` 的 `DispatchFFNCombineW4A8` 行；cannbot `model-recommend-analysis/SKILL.md:231` |
