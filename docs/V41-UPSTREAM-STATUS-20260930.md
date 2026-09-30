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
