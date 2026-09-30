# 上游状态调研：SMLA 算子是否已修复 / DeepSeek 新开源算子库能否替换

> 调研时间：2026-09-30 21:00（CST）。所有结论标注【实测】/【推断】。
> 起因：我们在 A3（910B, arch22）上做 DCP8 时，SparseFlashMla 的 CSA 模板
> 在 `compress_ratio=1` + KV 分片下产生 NaN 与非确定（见
> `V41-CSA-KERNEL-SOURCE-ANALYSIS-20260930.md`）。本次核查上游是否已修、以及
> DeepSeek 当日新开源的算子库能否替换。

---

## 0. 三句话结论

1. **上游没有修**：`vllm-project/vllm-ascend` @ `a40b52df`（2026-09-30）的
   `csrc/attention/sparse_flash_mla/op_kernel/arch22/` 与我们镜像里的算子
   **逐字节相同**（4 个关键文件 md5 全同）【实测】。
2. **上游明确不支持**：`vllm_ascend/attention/dsa_v41.py:1104` 写着
   **`supports_dcp = False`**，且 RFC #16375 把 "DSA 的 DCP" 整段列为
   **未勾选的 TODO**，其中一条正是我们撞上的问题【实测】。
3. **DeepSeek 今日新开源的三个库都不能在 A3/910B 上替换**：TileKernels
   没有注意力算子；FlashMLA 与 DeepSelect 的 Ascend 后端都要求
   **Ascend 950 + CANN 9.2.0**，且用 `bisheng`/`.asc` 编译【实测】。

---

## 1. 上游仓库：`vllm-project/vllm-ascend`

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/vllm-project/vllm-ascend |
| 调研 commit | `a40b52df0ef06a6360e2e97ce9b37a8939240d3f`（2026-09-30） |
| 算子目录 | `csrc/attention/sparse_flash_mla/`（60 个文件，含 `op_kernel/arch22` 与 `op_kernel/arch35`） |
| 我们镜像里的算子来源 | `github.com/vllm-ascend/DSv4.1` @ `e43cf1e9` —— **该仓库在 GitHub 上返回 404（非公开）**【实测】 |

### 1.1 逐字节比对（决定性问题）

| 文件 | 容器 md5 | 上游 md5 | 差异行数 |
|---|---|---|---|
| `sparse_flash_mla_csa_kernel.h` | `52f628d180d62711d1b1b5963e9ec764` | 同 | **0** |
| `sparse_flash_mla_csa_block_vector.h` | `e679387ca9fe654f4e2b6a55b4db2729` | 同 | **0** |
| `sparse_flash_mla_csa_block_cube.h` | `636084421391f61683b6ea53ef35443f` | 同 | **0** |
| `sparse_flash_mla_common_arch22.h` | `9aa5a33538338d221e5ceb8799652018` | 同 | **0** |

⇒ **我们撞到的缺陷在上游 HEAD 上仍然存在**，不存在"升个版本就好了"【实测】。

### 1.2 上游已把 DSA 的 DCP 标记为不支持

```python
# vllm_ascend/attention/dsa_v41.py:1104
class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False          # ← 由 0deca31 “Add DeepSeek V4.1 framework support” 引入
```
【实测】全仓 `supports_dcp` 取值：`common_cp.py:112 = True`，
`dsa_v41.py:1104 = False`（其余走 `supports_dcp_with_varlen`）。

### 1.3 RFC #16375《DeepSeek V4.1 Roadmap》把 DSA-DCP 列为未完成

状态 **open**（2026-09-11 创建），"Distributed parallelism: DCP and SP" 小节**全部未勾选**，
其中两条与我们撞到的问题**一字不差**：

```
- [ ] Support decode context parallelism (DCP) for sliding-window and compressed sparse
      attention: define KV/index-cache partitioning, global versus local positions,
      sparse-index ownership, ...
- [ ] Make compressor state and cross-layer KV/index/candidate sharing DCP-aware.
      Define how index selection obtains globally consistent candidates when keys are
      partitioned across ranks.
```

⇒ 我们分析的"全局 vs 局部坐标"与"稀疏索引归属"正是上游**自己承认还没定义**的东西【实测】。

### 1.4 上游当前的并行方案是"复制 KV"，没有容量收益

PR **#17171**（open）《Adapt Ascend attention to upstream PCP decode sharding》：

```
... The full KV cache remains replicated.
```
它走的是 **PCP（prefill CP）+ decode sharding**，且**KV 仍全量复制**
⇒ 拿不到我们需要的 ~5~8× 容量【实测】。

### 1.5 一处**文档与实现矛盾**（值得作为 bug 报告的核心论据）

算子文档（`docs/aclnnSparseFlashMla.md:686-688`）声明：

```
- 确定性计算
    - aclnnSparseFlashMla默认采用确定性实现，相同输入多次调用结果一致。
```

而"确定性级别"这条链路在 A2/A3 上是**断的**【实测 + 源码审计】：

```cpp
// op_host/sparse_flash_mla_tiling.cpp:392
batchConsistency_ = (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL); // =3
// :2492  该位进入 tiling key
tilingKey = GET_TPL_TILING_KEY(..., static_cast<uint32_t>(tilingInfo->batchConsistency), ...);
```
但它**只进 host key，arch22 内核从不读取**（`grep batchConsistency op_kernel/arch22/` **零命中**），
且容器里 `.../kernel/ascend910_93/sparse_flash_mla/` **只编出 1 个二进制**
⇒ 两个 key 落到同一份二进制。

**单卡实测复核**【实测】：

| 臂 | `bit_identical` | `max｜Δlse｜` | NaN |
|---|---|---|---|
| A 默认 | **False** | **nan** | 4806 |
| B `torch.use_deterministic_algorithms(True)` + `HCCL_DETERMINISTIC=strict` + `LCCL_DETERMINISTIC=1` | **False** | **nan** | 5388 |

⇒ **A2/A3 的 CSA 路径当前没有可用的确定性分支**，与文档声明不符。
（脚本：`experimental/v41-dcp/probes/replay_det.py`）

---

## 2. DeepSeek 当日（2026-09-30）开源的算子库核查

`deepseek-ai` 组织在 2026-09-30 前后有多个仓库更新。逐个核查能否替换我们的 SMLA：

| 仓库 | 内容 | Ascend 后端要求 | 能否替换 SMLA |
|---|---|---|---|
| **TileKernels** | MoE 路由 / 量化 / Engram / mHC / RoPE / Rand / Modeling | **Ascend 950 + CANN 9.2.0**（`bisheng` 编译） | ❌ **没有注意力类算子**，且面向 950 |
| **FlashMLA** | DSA 稀疏 prefill/decode、融合 Q-norm/RoPE、dense attention | **Ascend 950 + CANN 9.2.0** | ❌ 型号不符（我们 910B / A3） |
| **DeepSelect** | DSA 的 **TopK**（Lightning Indexer 场景）与 sampler | Ascend（`csrc/ascend_kernels/*.asc`，`bisheng`） | ❌ 是 **TopK** 不是注意力；且是 950 路径 |

关键原文（各自 README）【实测】：

* TileKernels `README.md`：
  `- **[2026-09-30] Huawei Ascend support**: ...`，
  Requirements 里 `Ascend Backend - Ascend 950 NPU - CANN 9.2.0 or higher`；
  目录只有 `engram / mhc / modeling / moe / quant / rand / testing / torch / transform`，
  **无 attention**。
* FlashMLA `README.md`：
  `For the Huawei platform: - Huawei Ascend 950 NPU - CANN 9.2.0 and above
   (provides the bisheng compiler ...)`；
  新增内容为 “Release of Ascend Attention Kernels” 针对 **950**。
* DeepSelect `README.md`：
  `[2026.09.30] We've released TopK kernels for Huawei Ascend NPU`；
  算子目录 `csrc/ascend_kernels/kernel.asc`（`.asc` + bisheng = 950 路径）。

**补充观察**：FlashMLA 的 Ascend 部分只有 `csrc/ascend_kernels/prefill/sparse/`
（`kernel.h` / `kernel_body.h` / `*.asc` 实例化），**没有 arch22（910B）实现**；
而 `vllm-project/vllm-ascend` 的 SMLA 才有 `op_kernel/arch22`（910B）与 `op_kernel/arch35`（950）两套。

---

## 3. 对我们目标的影响

| 目标 | 影响 |
|---|---|
| ① 容量 4.90× | 不受影响，已完成 |
| ③ 性能 1.29× | 不受影响，已完成 |
| ② 正确性（T≥560） | **不能靠"升上游"或"换算子库"解决**：上游代码相同且明确 `supports_dcp=False`；DeepSeek 新库面向 950 |

**可行路径（按代价排序）**：

1. **推动上游修 SMLA 的 CSA+分片路径**（我们已有生产同源、20 秒可复现的用例包
   与源码级定位）。RFC #16375 已经把这件事列成 TODO，我们的材料正好可以
   直接喂给它（把"global versus local positions / sparse-index ownership"
   从"待定义"推进到"已定位"）。
2. **在 vllm-ascend 上做一次"复制 KV 的 DCP"**（等价于上游 PCP 思路）：
   正确性有保证，但**放弃容量收益**（回到 ~1×），只保留 `T` 缩短带来的收益。
3. **等/换 950 平台**：FlashMLA/DeepSelect/TileKernels 全部面向
   Ascend 950 + CANN 9.2.0。若将来在 950 上做，可直接用官方算子（含
   Ascend 注意力内核），但当前 A3 不可用。

---

## 4. 建议的下一步（给上游/算子团队的材料）

把 `V41-CSA-KERNEL-SOURCE-ANALYSIS-20260930.md` + 复现包投到
`vllm-project/vllm-ascend`，**引用其自身文档与 RFC**：

* 文档说"SMLA 默认确定性"，实测在 A2/A3 的 CSA+分片下不成立（附单卡 20 秒复现）；
* 源码级定位：`cmpS2IdLimit` 的全局坐标假设 + 向量/矩阵阶段缺少"实际条数"握手；
* 与 RFC #16375 的 "global versus local positions / sparse-index ownership" 对齐，
  说明我们已经把这两项**定位到具体行号**；
* 需求方向：给 CSA 增加"本 rank 可见上界/局部坐标"的显式入参，
  或让因果过滤与掩码一致（不产生未写洞）。

复现包（public-read）：
```
cos://uploads-new/share/dsv41-dcp8-smla-nondeterminism-repro-v2-20260930.tar.zst
88.91 MB，md5 = c4ac20dc9e78c26fa0b6db832e6bc89d
```

---

# 附：`gitcode.com/cann/cann-recipes-infer` 调研（2026-09-30 21:10）

## A0. 结论

**这是 CANN 官方的推理优化样例库，里面有 DSv4.1 的完整实现（含 8 卡 8CP 配置），
但它仅支持 Ascend 950，且 CP 只用于 Prefill、Decode 走 DP+EP —— 没有 DCP。**
它用的稀疏注意力算子是与我们**不同的一版**（CANNBot-DSL 的
`mixed_quant_sparse_flash_mla`），**A3 上不可用**。

## A1. 仓库与版本

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/cann-recipes-infer |
| 调研 commit | `2225cae19d7612c1242d0815271e86e13eea2c95`（2026-09-30） |
| 该 commit 标题 | `feat(dsv4.1): 打开 type-2 grouplist 开关 + swiglu 归一自定义算子` |
| DSv4.1 目录 | `models/deepseek_v4_1/`（含 `models/`、`config/`、`utils/`、README） |
| 技术文档 | `docs/models/deepseek_v4_1/` 下 **5 份**（CANN 优化实践 / 算子指南 / 通信 / 低时延 / 单卡） |
| 代码注册表 | `models/modules/registry.py`：`SUPPORT_PLATFORM = ["A3", "950"]`（**全仓库**级别）【实测】 |

## A2. ★ DSv4.1 只支持 950

* 5 个配置**全部** `platform_version: "950"`【实测】：
  ```bash
  $ grep -h platform_version models/deepseek_v4_1/config/*.yaml | sort | uniq -c
        5   platform_version: "950"
  ```
* README《硬件要求》：**产品型号：Ascend 950 系列**；镜像为
  `cann9.2.0.pt2.13.0_dsv4.1_aarch_a5_image_custom_20260930`（CANN **9.2.0**、A5）。
* 配置里还有 950 专属项：`engram_tp_size`、`enable_superkernel`、TileLang 后端等。

> ⚠️ 注意别被 `SUPPORT_PLATFORM = ["A3","950"]` 误导：那是**整个仓库**的注册表
> （其他模型如 Qwen/Llama 支持 A3），DSv4.1 这一支的配置与文档都是 950。

## A3. ★ 有 8CP 配置，但 **CP 只用于 Prefill，Decode 用 DP+EP**

配置 `config/deepseek_v4_1_flash_rank_8_8ep_8cp.yaml`：

```yaml
parallel_config:
  world_size: 8
  attn_tp_size: 1
  ...
  cp_size: 8              # ← 8 卡 CP
scheduler_config:
  block_size: 128
  max_prefill_tokens: 131072
  batch_size: 8
  cp_mini_batch: 1
```

但技术报告（`docs/models/deepseek_v4_1/deepseek_v4.1_flash_cann_tech_report.md`）
把边界说得很明确【实测·原文】：

> ### Prefill Context Parallel
> V4.1-Flash 在 **Prefill 阶段**采用 Context Parallel（CP）把单条请求的序列切分到
> 各卡并行计算，**Decode 仍按 DP 执行**，两阶段的并行方式相互独立。

> ### Decode 并行（DP+EP）
> Decode 阶段沿用 DeepSeek 系列的并行方案：
> - **Attention 采用 Data Parallel（DP）并行**；
> - MoE 采用 Expert Parallel（EP）并行；
> - LM Head 采用 Tensor Parallel（TP）并行。

⇒ **官方 recipe 在 Decode 阶段不做 token 级 KV 分片**。这与我们在
`vllm-project/vllm-ascend` 看到的 `supports_dcp = False`（§1.2）完全一致。

## A4. 他们的 Prefill CP 是"复制 + 跨卡收集"，没有容量收益

技术报告 §序列切分与段长 / 数据流【实测·原文】：

* **Zigzag 切分**：序列切成 `2*cp_size` 段，第 r 张卡持第 r 段与第 `2*cp_size-1-r` 段；
  每层**按段各执行一次注意力**，窗口索引/压缩长度/算子 metadata **均按段构建**。
* **跨段依赖**：
  * 滑窗：每段段首需前序 `sliding_window=128` 的 KV ⇒ 每层把本卡两段尾部各 128 行
    合并做 **1 次 AllGather**，各卡从全局结果取自己需要的尾窗写入**临时窗口 Cache**；
  * 压缩 KV 与 index KV：**"需要收齐到全域"**。
* 另有 `cp_tmp_cache`（`get_cp_tmp_cache`：**Full-length temporary cache**）与
  `gather_cp_segments`（`all_gather_into_tensor` + `reverse_index` 还原顺序）
  【实测·源码 `models/modules/common_modules.py:77-125`】。

⇒ 每个 rank 最终都要拿到全域数据 ⇒ **KV 是复制/收集的，不是分片**，
**没有 KV 容量收益**。这与我们 DCP8 追求的 4.90× 是**两条不同的路**。

## A5. ★ 他们的稀疏注意力算子与我们**不是同一版**

```python
# models/deepseek_v4_1/models/modeling_deepseek.py:1041
import custom_ops  # Registers the repository's KV quantization writer.
self.sparse_attn_ops = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla
```

| | 我们（A3 / arch22） | 官方 recipe（950） |
|---|---|---|
| 算子 | `_C_ascend::npu_sparse_flash_mla` | `cann_ops_transformer::ds41::mixed_quant_sparse_flash_mla` |
| 来源 | `vllm-ascend` 的 `custom_transformer` vendor（AscendC, arch22） | CANN 内置 `cann_ops_transformer.ops.ds41`（CANNBot-DSL） |
| KV 精度 | **BF16** | **FP8 E4M3（原始）+ FP4 E2M1（压缩）** |
| 额外参数 | — | `quant_mode`、`rope_head_dim`、`key_dtype`、`value_dtype` |

### A5.1 为什么不能用它替换

在**我们的容器**（CANN 9.1.0 / A3）里核查【实测】：

| 检查项 | 结果 |
|---|---|
| `cann_ops_transformer/ops/ds41/` 子模块 | **不存在** |
| `aclnn_mixed_quant_sparse_flash_mla.h` 头文件 | **不存在** |
| `opp/.../kernel/.../*mixed_quant*` 算子实现 | **不存在**（`find` 零命中） |
| 仅有的东西 | 一个 224 行的 Python/C++ 包装层 `ops/csrc/mixed_quant_sparse_flash_mla.cpp`，它调用的 `aclnnMixedQuantSparseFlashMla` **在本版本不存在** |

⇒ 那份 Python 包装即使被 import，也只会 JIT 编出一个**链接不到 aclnn 实现**的壳子。
**A3 + CANN 9.1.0 上没有这个算子。**

### A5.2 ★ 但它的设计正好指出了我们 bug 的修法

算子指南（`deepseek_v4.1_cannbotdsl_operator_guide.md`）§"Mixed Quant Sparse Flash MLA"
描述的新版实现【实测·原文】：

> 以 128 个 KV 位置为一个处理 tile，**按照 Query 行与两侧有效稀疏 KV 长度划分任务**；
> 一行先处理原始 KV tile，再处理压缩 KV tile。……**尾块按实际有效长度屏蔽多余位置**。
>
> 调度阶段**根据每行有效长度生成任务范围**。……各核将局部输出、最大值与指数和写入
> Workspace，Vector 核再按在线 Softmax 合并公式归约。

对照我们在 arch22 版本里定位到的问题（`V41-CSA-KERNEL-SOURCE-ANALYSIS-20260930.md`）：

| | arch22（我们） | CANNBot-DSL 新版（950） |
|---|---|---|
| 有效长度来源 | `actCmpS2Size` 由**全局**公式推出（`cmpMaskRight = cmpMaskS2Size − actS1Size`） | **按每行实际有效稀疏 KV 长度**生成任务范围 |
| 丢键后的处理 | `CopyInSingleKv` 静默返回 ⇒ `kvMergeGm_` 留洞；矩阵阶段按**期望长度**读 | **尾块按实际有效长度屏蔽** |
| 两阶段一致性 | 缺"实际条数"握手 | 任务范围本身即由有效长度决定 |

⇒ **新版算子从设计上就避免了我们的缺陷类别**，但它是 950/FP8-FP4 路径。

## A6. 对目标 ② 的影响（更新）

| 路径 | 是否提供 Decode 侧 KV 分片 | 容量收益 | A3 可用 |
|---|---|---|---|
| `cann-recipes-infer` DSv4.1（官方） | ❌ Decode 用 DP+EP | 无（CP 只省 TTFT） | ❌ 950 专属 |
| `vllm-project/vllm-ascend` PCP（#17171） | ❌ KV 全量复制 | 无 | 部分（但非 v4.1） |
| 我们的 DCP8 | ✅ 每 rank 1/8 KV | **4.90×** | ✅（我们实现） |

⇒ **三条独立证据（CANN 官方 recipe / vllm-ascend 上游 / RFC #16375）
一致表明：Decode 侧 token 级 KV 分片 + 稀疏注意力这条组合，目前没有任何官方实现。**
我们撞到的 `T≥560` 缺陷正处在这条"无官方支持"的路径上。

**可选路径（修订版）**：

1. **推动算子侧按新版设计修 arch22**：把"按每行有效稀疏长度划分任务"与
   "尾块按实际有效长度屏蔽"这两条搬到 arch22 的 CSA 模板（新版已有正确做法，
   属于**移植**而非从零设计）。这是我们现有工作的最短闭环。
2. **退回官方架构**（Prefill CP + Decode DP+EP）：正确性有保障，但**放弃 4.90× 容量**。
3. **等 950 平台**：可直接用官方 `mixed_quant_sparse_flash_mla`（FP8/FP4 KV）。

**给算子团队的补充材料**：可以直接引用其**自家** CANNBot-DSL 算子指南里
"按每行有效长度划分任务 + 尾块按实际长度屏蔽"的设计，说明 arch22 版本的
`cmpS2IdLimit` 全局公式 + 静默丢键是**落后于自家新版设计**的。
