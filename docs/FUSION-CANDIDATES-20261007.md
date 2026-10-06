# 小算子海的融合候选清单与占比（2026-10-07）

> 数据源：`results/armF_r6_base`（交付口径 profile，72 步稳态，步长 40.02 ms ↔ 真实 24.59 ms，K=1.6275）。
> 工具：`tools/prof_fusion_families.py`（新增）。
> 口径：**聚合暴露** = `union − 与其它算子的重叠`，比"自身时长之和"准（后者把被覆盖的部分重复计算）。

---

## 0. 一页纸结论

每步 **2,888 个算子**中，**2,048 个（71%）属于可融合族**；这些族的**聚合真实暴露 4.71 ms = 19.1% 步长**。

| 融合族 | 次数/步 | 中位/算子 | 自身 ms | **真实暴露** | **占步长** | 4:1 融合可省（真实） |
|---|---:|---:|---:|---:|---:|---:|
| **F3 norm + quant** | 370.1 | **10.52 µs** | 3.804 | **1.127** | **4.6%** | **1.794 ms** |
| **F7 RoPE / 取表链** | 197.0 | 8.82 µs | 1.895 | **0.877** | 3.6% | 0.801 ms |
| **F4 MoE 路由链** | 129.0 | **13.10 µs** | 1.690 | 0.384 | 1.6% | 0.779 ms |
| **F2 dtype / 搬运链** | **491.1** | **1.44 µs** | 1.679 | **0.888** | 3.6% | 0.326 ms |
| **F6 indexer 小 kernel** | 24.0 | **19.72 µs** | 0.655 | 0.403 | 1.6% | 0.218 ms |
| **F1 位置 / 槽位链** | **469.3** | **1.54 µs** | 0.981 | 0.356 | 1.4% | 0.333 ms |
| F8 归约 / 逐元素杂项 | 155.0 | 3.84 µs | 0.815 | 0.437 | 1.8% | 0.274 ms |
| F5 spec-decode 后处理 | 42.0 | 10.67 µs | 0.615 | 0.236 | 1.0% | 0.207 ms |
| **合计** | **2,047.6** | — | **13.13** | **4.708** | **19.1%** | **≈4.73 ms（乐观）** |

**三个读法**：

1. **数量最多的不是收益最大的**：F1+F2 有 **960 个/步**（占总数 33%），但中位只有 **1.44~1.54 µs**，
   两者合计真实暴露 1.244 ms（5.0%）；
2. **F3 是单项最大**：只有 370 个/步，但中位 **10.52 µs**（是 F1/F2 的 7 倍），
   真实暴露 **1.127 ms（4.6%）**，而且**官方已有现成融合算子**；
3. **"4:1 融合可省"是乐观上界**（假设中位时长全部可省、且融合后的 kernel 免费）——
   现实中应打折，**建议按 40~50% 计，即 1.9~2.4 ms = 8~10% 步长**。

---

## 1. 逐族明细

### F3 norm + quant（370.1 个/步，自身 3.804 ms，暴露 1.127 ms）★ 首选

| 算子 | 次数/步 | 中位 | 合计 |
|---|---:|---:|---:|
| **`RmsNorm`** | **140.0** | **13.42 µs** | **2.012 ms** |
| `aclnnDynamicQuantV2` | 141.0 | 5.52 µs | 0.747 ms |
| `DequantSwigluQuant` | 43.0 | 11.98 µs | 0.561 ms |
| `RmsNormCast` | 43.0 | 10.08 µs | 0.424 ms |
| `RmsNormDynamicQuant` | 3.0 | 20.92 µs | 0.060 ms |

**融合方案**：官方有 **`npu_rms_norm_dynamic_quant`**（A3 支持），
我们在代码里**只在 w8a8 分支调用**（`_is_w8a8_dynamic` 门控），当前路径没走。

**为什么它最大**：
* `RmsNorm` 140 次/步 = 40 层 × 3.5 —— **每一处 norm 后面都紧跟一次 quant**，天然成对；
* 两者中位合计 **18.9 µs**，融合后理论上一半以下；
* `RmsNormCast` 43 次同类（norm + cast）。

⚠️ **精度风险**：官方融合算子**非逐位等价**（F2 实验实测 4~5% 元素差 1 LSB），
且当年在 prefill 上反而更慢 ⇒ **必须过"结构化任务逐字一致 + 100% 正确"的分级门**。

### F7 RoPE / 取表链（197.0 个/步，自身 1.895 ms，暴露 0.877 ms）

| 算子 | 次数/步 | 中位 | 合计 | 备注 |
|---|---:|---:|---:|---|
| `InplacePartialRotaryMul` | **145.0** | 8.86 µs | 1.285 ms | RoPE 应用 |
| `aclnnIndexSelect_GatherV3` | 41.0 | 8.78 µs | 0.527 ms | cos/sin 取表 |
| `aclnnEmbedding_GatherV2` | 8.0 | 3.32 µs | 0.052 ms | |
| `RotaryPositionEmbeddingV2` | 3.0 | 10.42 µs | 0.031 ms | |

**⚠️ 其中"取表链融合"已经做过**：主分支 README §4.1 记录了
**`rope 取表融合`：cos/sin 取表链 6 kernel → 2，实测 −0.45~0.62 ms**。
但现在 profile 里仍有 41 次 `GatherV3` ⇒ **说明融合只做到"6→2"，还有 2 个没合**。

**剩余空间**：`InplacePartialRotaryMul` 145 次/步（40 层 × kv+q 两处 + draft），
可与**同一条链上的 quant/norm 合并**（即并入 F3）。

### F4 MoE 路由链（129.0 个/步，自身 1.690 ms，暴露 0.384 ms）

| 算子 | 次数/步 | 中位 | 合计 |
|---|---:|---:|---:|
| `aclnnMoeInitRoutingV3` | 43.0 | 14.50 µs | 0.633 ms |
| `MoeGatingTopKHash` | 43.0 | 13.73 µs | 0.591 ms |
| `MoeTokenUnpermute` | 43.0 | 10.84 µs | 0.466 ms |

**融合方案**：三者是**严格相邻**的 gating → routing → (experts) → unpermute 链，
且**每层各一次**（43 = 40 主层 + 3 draft）。`MoeGatingTopKHash` 可与
`MoeInitRoutingV3` 合并（gating 输出直接作为 routing 输入，中间无其它消费者）。

**注意**：`MoeTokenUnpermute` 必须等在 experts 之后，**不能提前合并**。
⇒ **实际可合的是 gating + routing（86 个 → 43 个）**。

### F2 dtype / 搬运链（491.1 个/步，自身 1.679 ms，暴露 0.888 ms）

| 算子 | 次数/步 | 中位 | 合计 |
|---|---:|---:|---:|
| **`aclnnInplaceCopy_Cast`** | **172.5** | **1.28 µs** | 0.398 ms |
| `aclnnInplaceCopy_ViewCopy` | 37.0 | 8.50 µs | 0.352 ms |
| `aclnnInplaceCopy_TensorMove` | 43.5 | 5.12 µs | 0.256 ms |
| `ZerosLike` | 12.0 | 4.78 µs | 0.116 ms |
| `MaskedFill` | 50.0 | 1.44 µs | 0.104 ms |
| `Fill` | 60.0 | 1.30 µs | 0.084 ms |

**融合方案**（多为**纯 Python 侧**改动，零精度风险）：
* `Cast` 172.5 次：**dtype 提升**（int32→int64 等）。可在**上位张量创建时就用目标 dtype**，
  省掉这一层转换（`IDS64_HOIST` 就是这个思路，已在 A2 生效）；
* `Fill` 60 次 + `ZerosLike` 12 次：**常量张量预建**（capture 期建好、每步复用）；
* `ViewCopy` 37 次 + `TensorMove` 43.5 次：**避免中间 `contiguous()`/`.to()`**。

### F1 位置 / 槽位链（469.3 个/步，自身 0.981 ms，暴露 0.356 ms）

| 算子 | 次数/步 | 中位 | 合计 |
|---|---:|---:|---:|
| `SelectV2` / `SWhere` | 48.0 | 1.92 µs | 0.098 ms |
| `Less`（LtTensor） | 46.0 | 1.38 µs | 0.074 ms |
| `Abs` | 45.0 | 1.28 µs | 0.069 ms |
| `Neg` | 45.0 | 1.50 µs | 0.067 ms |
| `FloorMod` | 24.0 | 2.34 µs | 0.067 ms |
| `Arange` | 6.0 | 10.12 µs | 0.064 ms |

**融合方案**：整条链都是**从 `positions` 派生**（位置/槽位/掩码），
可下沉成 **1 个 kernel**。仓库里**已有先例**：`_compute_slot_mapping_kernel`（12 次/步）
就是把这类链下沉的结果。

**注意**：`Abs`/`Neg` 各 45 次，需逐个核实它们是否真在位置链上（也可能是其它用途）。

### F6 indexer 小 kernel（24.0 个/步，自身 0.655 ms，暴露 0.403 ms）

| 算子 | 次数/步 | 中位 | 合计 |
|---|---:|---:|---:|
| `QuantLightningIndexerV2` | 8.0 | **48.93 µs** | 0.359 ms |
| `_prepare_indexer_indices_kernel` | 8.0 | 19.56 µs | 0.154 ms |
| `_quantize_indexer_query_kernel` | 8.0 | 17.20 µs | 0.142 ms |

**融合方案**：后两个是**已下沉的 JIT kernel**，可与 `QuantLightningIndexerV2` 合并
（都是 indexer 的输入准备）。但**暴露 0.403 ms 里有 0.359 是 QLI 本身**——
它已经在主分支被优化过（`QLI 无候选快速路径`：99.3 → 50.3 µs）。

### F5 spec-decode 后处理（42.0 个/步，暴露 0.236 ms）

`MaskedFill`（57.6 µs × 3）、`IndexCheck`（8.34 × 21）、`ArgMax`（17.52 × 7）、
`ReduceSum`（17.43 × 5）、`GatherElements`（7.92 × 4）、`IndexFill`（22.49 × 1）。

**融合方案**：整条是**拒绝采样后处理链**，输入是 logits + 已接受 token + mask，
可融为 1 个 kernel。文档里早先估过 **0.3~0.5 ms**，但**实测暴露只有 0.236 ms**
（因为它落在步尾、其它流在跑）⇒ **优先级下调**。

### F8 归约 / 逐元素杂项（155.0 个/步，暴露 0.437 ms）

`Add`（57 × 9.00 µs = 0.479 ms）最大，其余 `Mul`/`ReduceMean`/`Pows`/`Sub`。
`Add` 57 次需核实用途（残差加？）；若能与 `HcPost` 合并可省更多
（但 `HcPost` 是官方融合算子，不宜动）。

---

## 2. 优先级排序（按"可回收 ÷ 难度"）

| 序 | 族 / 动作 | 可回收（真实） | 风险 | 难度 | 依据 |
|---:|---|---:|---|---|---|
| **1** | **F3：`RmsNorm` + `DynamicQuant` → `npu_rms_norm_dynamic_quant`** | **≤1.13 ms（4.6%）** | ⚠️ 中（非逐位） | 中 | 官方算子现成；我们只走 w8a8 分支 |
| **2** | **F2：Cast/Fill/ZerosLike 消除（纯 Python 侧）** | **≤0.89 ms（3.6%）** | ✅ **零** | **低** | `IDS64_HOIST` 已是同类先例 |
| **3** | **F7：RoPE 与 norm/quant 合并** | ≤0.88 ms（3.6%） | ⚠️ 中 | 中 | 取表已合过（6→2），剩 2 个 |
| **4** | **F4：`MoeGatingTopKHash` + `MoeInitRoutingV3` 合并** | ≤0.38 ms（1.6%） | ⚠️ 中 | 中 | 严格相邻、每层各一次 |
| **5** | **F1：位置/槽位链下沉成 1 kernel** | ≤0.36 ms（1.4%） | ✅ 低 | 中 | `_compute_slot_mapping_kernel` 先例 |
| **6** | F6：indexer 三件套合并 | ≤0.40 ms（1.6%） | ⚠️ 中 | 中 | QLI 已优化过 |
| **7** | F5：拒绝采样后处理融 1 kernel | 0.24 ms（1.0%） | ⚠️ 中 | 中 | 暴露已被覆盖大半 |
| **8** | F8：`Add` 用途核实 | ≤0.44 ms（1.8%） | ❓ 待查 | 待定 | 需先确认语义 |
| | **合计** | **≈4.71 ms（19.1%）** | | | 乐观上界；**现实按 40~50% ⇒ 1.9~2.4 ms（8~10%）** |

---

## 3. 已经做过、不要重复投入的

| 项 | 状态 | 出处 |
|---|---|---|
| `rope` 取表链融合（6 kernel → 2） | ✅ **已做**（−0.45~0.62 ms） | 主分支 README §4.1 |
| `QLI` 无候选快速路径（99.3 → 50.3 µs） | ✅ **已做**（−0.49 ms） | 同上 |
| `wo_a` 2D matmul | ✅ 已做（−0.31~0.76 ms） | 同上 |
| expert mask 范围比较 | ✅ 已做（−0.51 ms） | 同上 |
| `IDS64_HOIST` | ✅ 已实现（A2 生效；**本配置无对象**） | `armH-r6-flags` |
| `ENGRAM_PAD_SKIP`（消 `ZerosLike`） | ✅ 已转默认（−58 µs） | `R6-SUMMARY` |
| `ScatterNdUpdateSk` 换算子 | ❌ **三路径全否** | `LINE-A-SCATTER-MICROBENCH` |
| `HcPre` 减迭代 / 原生实现 | ❌ 不可减（eps 下限）/ 原生慢 35× | `HCPRE-AND-SCATTER-DEEP-DIVE` |
| `DequantSwigluQuant` 融合 | ❌ **暴露仅 0.025 ms** | `OPTIMIZATION-ROADMAP-CORRECTED` |
| `metadata` 提前 | ❌ 暴露仅 0.032 ms | 同上 |
| MegaMoE / MC2 | ❌ 双重封堵 / 慢 2~8× | `PREFILL-LEVERS-ALL-BLOCKED` |

> ⚠️ **注意 F3 里的 `DequantSwigluQuant`**：它在**单算子暴露度**分析里只有 0.025 ms
> （因为落点被覆盖），但**按族统计它属于 F3**。
> 两者不矛盾：**单算子暴露**衡量"单独删掉它能省多少"，**族统计**衡量"整族一起融能省多少"。
> 融合的收益来自**减少调用次数**，所以要看族的**次数**，不能只看单算子暴露。

---

## 4. 验收方式（每一项都必须过）

| 门 | 工具 | 判据 |
|---|---|---|
| **性能** | `[bneck] hp`（单流 ms/step） | 不看聚合 tok/s |
| **逐位一致** | `walk_blocks.py` | 分级门：结构化任务逐字一致 + 100% 正确 |
| **长文针** | `ced_pd_acceptance.py --mode needle` | 60K/74K/150K 全过（基线 24/24） |
| **带宽** | `hbm_bw_sample.py` | 未打爆（上限 1182 GB/s） |

**建议的执行顺序**：先做 **#2（纯 Python 侧，零精度风险）** 建立判据流程，
再做 **#1（收益最大，但需过精度门）**。

---

## 5. 复现

```bash
PROF=~/cedpd-repo/results/armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_\
1434_20261004195431495_ascend_pt/ASCEND_PROFILER_OUTPUT
python3 tools/prof_fusion_families.py  $PROF 24.59   # 按融合族的规模与暴露
python3 tools/prof_size_bucket_expo.py $PROF 24.59   # 按规模分桶（<2µs … >2ms）
python3 tools/prof_small_op_ranking.py $PROF 20 24.59 # <20µs 逐算子排名
```
