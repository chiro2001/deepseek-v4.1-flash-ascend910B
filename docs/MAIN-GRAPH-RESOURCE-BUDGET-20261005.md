# ★ 主图不是算力/带宽受限，而是**标量 + 跨核同步**受限（2026-10-05）

> 数据：`k6full_1004_100156` PROF_000003，rank0，主模型 stream（158），626 步（batch 6）。
> 工具：`tools/resource_budget.py`（本轮新增）。
> 口径：【实测】= profile 计数聚合；【推断】= 由计数推出的机制。

## 0. 一句话

主图一步 **16.53 ms** 里：
* **AIC MAC（真正的乘加）只占 0.574 ms = 3.5%**；AIV 向量算术 0.798 ms = 4.8%；
* **AIC scalar 3.024 ms（18.3%）+ AIV 的"非算术占用"约 7.15 ms（43%）**；
* ⇒ 这个工作负载**既不是算力受限也不是带宽受限**，而是**被标量地址计算与跨核同步卡住**。

这与"armF 降 SyncAll 4→2 就拿到 −7.56 µs/op"的方向完全一致，也解释了为什么
纯 kernel 数学优化（gmm1 18.2 µs 地板、HcPre A1）兑现率低 —— **优化的对象找错了单元**。

## 1. 硬件单元账（stream 158，每步 ms；分母 = 该 stream 的任务时长合计 16.532 ms）

| 单元 | ms/step | 占任务时长 |
|---|---:|---:|
| **AICore 总** | 9.356 | 56.6% |
| ├ AIC **MAC**（乘加） | **0.574** | **3.5%** |
| ├ AIC **scalar** | **3.024** | **18.3%** |
| ├ AIC MTE1 | 1.641 | 9.9% |
| ├ AIC MTE2（**载入**） | **4.044** | **24.5%** |
| └ AIC fixpipe | 0.365 | 2.2% |
| **AIV 总** | **10.499** | **63.5%** |
| ├ AIV vec（向量算术） | **0.798** | **4.8%** |
| ├ AIV scalar | 2.790 | 16.9% |
| ├ AIV MTE2 | 1.067 | 6.5% |
| ├ AIV MTE3 | 0.342 | 2.1% |
| └ **未归因（≈同步/等待）** | **≈5.5** | **≈33%** |

> AIC 侧四项之和 9.65 ≈ 9.36 ⇒ AIC 的账是自洽的（各分量基本串行）。
> AIV 侧子分量只有 5.0 ms，而 AIV 总占用 10.5 ms ⇒ **一半的 AIV 时间不在做任何记在账上的工作**。

## 2. 逐算子族的"AIV 等待"（子分量是并行的，故用 `aiv总 − max(子分量)` 估计）

| 算子族 | AIV 总 | vec | aiv scalar | 载入 | **同步/其它** | 同步占比 | 调用/步 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GroupedMatmulSwigluQuantV2（gmm1）** | 2.769 | 0.009 | 0.497 | 0.369 | **2.272** | **82%** | 40 |
| **HcPre** | 1.840 | 0.065 | 0.372 | 0.122 | **1.467** | **80%** | 80 |
| GroupedMatmul（gmm2） | 1.342 | 0.010 | 0.523 | 0.029 | **0.818** | 61% | 40 |
| **QuantBatchMatmulV3** | 0.873 | 0.016 | 0.129 | 0.051 | **0.745** | **85%** | 88 |
| QuantLightningIndexerV2 | 0.323 | 0.021 | 0.053 | 0.001 | 0.271 | 84% | 8 |
| SparseFlashMla | 0.587 | 0.008 | 0.330 | 0.089 | 0.257 | 44% | 40 |
| MoeInitRoutingV3 | 0.253 | 0.013 | 0.062 | 0.010 | 0.191 | 76% | 40 |
| RmsNorm | 0.336 | 0.066 | 0.148 | 0.030 | 0.188 | 56% | 89 |
| HcPost | 0.495 | 0.310 | 0.148 | 0.115 | 0.185 | 37% | 80 |
| MoeTokenUnpermute | 0.205 | 0.012 | 0.070 | 0.057 | 0.135 | 66% | 40 |
| InplacePartialRotaryMul | 0.232 | 0.032 | 0.103 | 0.028 | 0.129 | 56% | 96 |
| DynamicQuant | 0.221 | 0.116 | 0.061 | 0.019 | 0.105 | 48% | 92 |
| Cast | 0.134 | 0.003 | 0.030 | 0.024 | 0.104 | 78% | 126 |
| Add | 0.124 | 0.004 | 0.010 | 0.021 | 0.103 | 83% | — |
| MoeGatingTopKHash | 0.126 | 0.031 | 0.047 | 0.013 | 0.080 | 63% | 40 |
| **前 15 名合计** | **9.85** | | | | **≈7.15** | **43% of 主图** | |

**读法**：`同步/其它` 是"该算子族的 AIV 上，既没在做向量算术、也没在做标量访存、也没在搬数据"的那部分时间；
在 MIX_AIC 算子里，这部分几乎必然是在**等 AIC 部分 / 等跨核 flag**。
（HcPost 是纯 AIV 算子，占比只有 37%，正好反衬：**MIX 算子的等待占比显著更高**。）

## 3. 一个具体机制（HcPre）：小 M 被切了 20 个 K 块 × 24 核

`csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp::CalcOpTiling()`：

```cpp
uint64_t kSize   = hcMult_ * d_;                                  // 4 × 5120 = 20480
uint64_t mDimNum = std::min(aicCoreNum_, CeilDiv(bs_, M_L1_MAX_SIZE));  // bs=6 ⇒ 1
uint64_t kDimNum = aicCoreNum_ / mDimNum;                         // 24 / 1 = 24
uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));        // 20480/1024 = 20
```

⇒ **M=6 时：M 方向只切 1 块，K 方向切 24 份（20 个 chunk）**，
而 `hc_pre_m_k_split_core.h` 的 K 循环里**每个 chunk 都有 AIC↔AIV 的 flag 往返**
（`:125 CrossCoreWaitFlag` / `:131/133 Set+Wait` / `:163 SetFlag`），
循环外还有一次 `:174 SyncAll<false>()`。
【推断】20 个 chunk 的握手就是那 1.47 ms/步（16 µs/次）的主要来源。

**可测的预测**：把 `kDimNum` 降下来（例如小 M 时 `mDimNum=4, kDimNum=6`，或干脆
`kDimNum=1` 单核跑完整 K），AIV 等待应按 chunk 数成比例下降；代价是参与的核心变少、
单核串行工作量变大。**两者要实测取平衡**，因为算术量本身极小（MAC 合计只占 3.5%）。

同族的其他候选也都在同一张表上：gmm1 2.272、gmm2 0.818、QBMV3 0.745 —— 三个都是一样的
"小 M + 多核 + 多次握手"形态（gmm1 的 group_list 标量扫描已在
`REPORT-FLOOR-20261004B.md` 里量到 ~10 µs/次）。

## 4. 这对"该投哪"的改变

| 旧排序（按独占贡献） | 新排序（按**可动的单元**） |
|---|---|
| gmm1 12.7%、MatMulV2 11.6%、HcPre 10.9%、通信 10.4% | ① **AIV 跨核等待 43%**（gmm1 2.27 + HcPre 1.47 + gmm2 0.82 + QBMV3 0.75 …） |
| | ② **AIC scalar 18.3%**（group_list 扫描、地址计算） |
| | ③ AIC MTE2 载入 24.5%（受权重体量约束，属硬件地板） |
| | ④ MAC 3.5% / vec 4.8%（**不值得再投**） |

⇒ 结论：**继续投入"减少算术"或"减少数据搬运"的收益上限很低**；
值得投的是**减少同步轮次 / 减少标量读点**，以及**减少 kernel 数量**（尾巴那 1000 个算子）。

## 5. 复现

```bash
BASE=$HOME/cedpd-repo/results/k6full_1004_100156/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1435_20261004042337710_ascend_pt/PROF_000003_20261004042337723_00001435KBJORREQ/mindstudio_profiler_output
python3 tools/resource_budget.py $BASE 158 626
```
