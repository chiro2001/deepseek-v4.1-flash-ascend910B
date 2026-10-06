# 修正口径下的真实算子排名 + HcPre 子计数拆解（决定下一步该做什么）

> 承接 `FRESH-CONC1-PROFILE-AND-HCPRE-E2E-READY-20261007.md`（K≈1 更正）与
> `HCPRE-FUSE-E2E-NEGATIVE-20261007.md`（融合负结果）。
> 本文用**纯 conc=1 的新 profile**（步锚 `HcPre/86`、1854 步、步长 25.458 ms、2821 算子/步）
> 给出真实排名，并拆开第 1 名的内部构成。全部为【实测】。

---

## 0. 一页纸

| # | 结论 |
|---:|---|
| 1 | **Top26 算子占 93.5% 步长**（23.796 / 25.458 ms）—— 与"2888 个小算子海"的印象相反，**大头是少数几个大算子** |
| 2 | **第 1 名 `HcPre` = 2.884 ms（11.3%）**，86 次 × **33.60 µs** |
| 3 | ★ **HcPre 的 33.6 µs 里，真正的数学只占 6.2%**（MAC 1.28 + vec 0.81 µs）；**同步/等待约 46%**、标量 32%、搬运 25% |
| 4 | **AivKernel 2.817 ms（11.1%）** 是第 2 名（engram wkv 的 all_gather）；但它是 COMMUNICATION 且已知与 HCCL 重叠 |
| 5 | MoE 三个 GEMM 合计 **6.0 ms（23.6%）** —— 已验证跑在权重带宽峰值，**没有余量** |
| 6 | ⇒ **单流要再降，最大的可动项就是 `HcPre` 的 ~78% 开销**（潜在 1.4~2.2 ms = 5.5~8.7%） |

---

## 1. 真实算子排名（Top26，占 93.5%）

| # | 算子 | 核 | 次/步 | 单次 µs | ms/步 | 占步长 | 块数 |
|---:|---|---|---:|---:|---:|---:|---:|
| **1** | **`HcPre`** | MIX_AIC | **86.0** | **33.54** | **2.884** | **11.3%** | **24** |
| 2 | `AivKernel`（engram all_gather） | COMMUNICATION | 94.0 | 29.97 | 2.817 | 11.1% | 8 |
| 3 | `GroupedMatmulSwigluQuantV2`（MoE w1/w3） | MIX_AIC | 43.0 | 64.45 | 2.771 | 10.9% | 24 |
| 4 | `MatMulV2` | AI_CORE | 111.0 | 20.62 | 2.289 | 9.0% | 23 |
| 5 | `QuantBatchMatmulV3` | MIX_AIC | **183.0** | 11.11 | 2.034 | 8.0% | 20 |
| 6 | `GroupedMatmul`（MoE w2） | MIX_AIC | 43.0 | 37.49 | 1.612 | 6.3% | 24 |
| 7 | `SparseFlashMla` | MIX_AIC | 40.0 | 32.93 | 1.317 | 5.2% | 24 |
| 8 | `RmsNorm` | AIV | 140.0 | 5.63 | 0.788 | 3.1% | **5** |
| 9 | `SparseFlashMlaMetadata` | AI_CPU | 3.0 | 236.25 | 0.709 | 2.8% | 48 |
| 10 | `MatMulV3` | AI_CORE | 46.0 | 14.86 | 0.683 | 2.7% | 24 |
| 11 | `HcPost` | AIV | 86.0 | 7.16 | 0.616 | 2.4% | **5** |
| 12 | `DynamicQuantV2` | AIV | 141.0 | 4.08 | 0.575 | 2.3% | **2** |
| 13 | `InplacePartialRotaryMul`（RoPE） | AIV | 145.0 | 3.44 | 0.499 | 2.0% | **5** |
| 14 | `MoeInitRoutingV3` | MIX_AIV | 43.0 | 10.39 | 0.447 | 1.8% | 48 |
| 15 | `QuantLightningIndexerV2` | MIX_AIC | 8.0 | 53.54 | 0.428 | 1.7% | 24 |
| 16 | `SparseAttnSharedkvMetadata` | AI_CPU | 2.0 | 213.59 | 0.427 | 1.7% | 1 |
| 17 | `QuantLightningIndexerV2Metadata` | AI_CPU | 2.0 | 186.76 | 0.374 | 1.5% | 48 |
| 18 | `QuantBatchMatmulV3`（另一形状） | AI_CORE | 43.0 | 8.13 | 0.350 | 1.4% | 9 |
| 19 | `InplaceCopy_Cast` | AIV | **174.0** | 1.94 | 0.337 | 1.3% | **1** |
| 20 | `MoeTokenUnpermute` | AIV | 43.0 | 7.11 | 0.306 | 1.2% | 5 |
| 21 | `ScatterNdUpdateSk` | MIX_AIV | 58.0 | 5.27 | 0.306 | 1.2% | 6 |
| 22 | `Add` | AIV | 57.0 | 5.04 | 0.287 | 1.1% | 25 |
| 23 | `InplaceCopy_ViewCopy` | AIV | 31.0 | 8.29 | 0.257 | 1.0% | 48 |
| 24 | `SparseAttnSharedkv` | MIX_AIC | 3.0 | 81.52 | 0.245 | 1.0% | 24 |
| 25 | `DequantSwigluQuant` | AIV | 43.0 | 5.53 | 0.238 | 0.9% | **1** |
| 26 | `Index` | AIV | 21.0 | 9.58 | 0.201 | 0.8% | 48 |
| | **Top26 合计** | | | | **23.796** | **93.5%** | |

### 1.1 三条读法

1. **前 7 名就占 61.8%**（15.72 ms）——**"小算子海"不是主要矛盾**；
2. **`QuantBatchMatmulV3` 有 183 次/步**（比 `HcPre` 的 86 次还多），是"次数最多的大算子"；
3. **块数列暴露了低效**：`RmsNorm` 只用 **5/48** 个 vector 核、`DynamicQuantV2` 只用 **2/48**、
   `DequantSwigluQuant` 与 `Cast` 只用 **1** 个核 —— 它们是**延迟受限、并行宽度极窄**的形态。

---

## 2. ★ `HcPre` 子计数拆解：78% 是开销

**单次 `Duration` 中位 = 33.60 µs**（86 次/步 ⇒ 2.884 ms/step）。

| 子计数 | 单次 µs | 占 Duration | 每步 ms |
|---|---:|---:|---:|
| **`aiv_time`**（vector 核总占用） | **25.98** | **77.3%** | 2.234 |
| ├ `aiv_vec_time`（**真正的向量数学**） | **0.81** | **2.4%** | 0.070 |
| ├ `aiv_scalar_time`（标量） | 5.62 | 16.7% | 0.483 |
| ├ `aiv_mte2_time`（载入） | 3.01 | 9.0% | 0.259 |
| ├ `aiv_mte3_time`（写出） | 0.94 | 2.8% | 0.080 |
| └ **未归因（≈同步/等待）** | **≈15.6** | **≈46%** | **≈1.34** |
| **`aicore_time`**（cube 核总占用） | **21.03** | **62.6%** | 1.809 |
| ├ **`aic_mac_time`（真正的乘加）** | **1.28** | **3.8%** | 0.110 |
| ├ `aic_scalar_time`（标量） | 4.98 | 14.8% | 0.428 |
| ├ `aic_mte2_time`（载入） | 5.47 | 16.3% | 0.470 |
| ├ `aic_mte1_time` | 1.31 | 3.9% | 0.112 |
| ├ `aic_fixpipe_time` | 0.63 | 1.9% | 0.055 |
| └ 未归因 | ≈7.36 | ≈22% | ≈0.63 |

### 2.1 汇总

| 成分 | 占 Duration |
|---|---:|
| **真正的数学**（MAC + vec） | **6.2%** |
| 标量（AIV + AIC） | 31.5% |
| 搬运（MTE1/2/3 + fixpipe） | 24.5% |
| **同步/等待（未归因）** | **≈46%（AIV）+ 22%（AIC）** |

⚠️ 两个"未归因"是**各自的资源内部**（`aiv_time` 与 `aicore_time` 都已含等待），不是简单相加。

### 2.2 形态确认

| 项 | 值 |
|---|---|
| 形状（93%） | **`6,4,5120;24,20480;3;24;6,4`** ⇒ **M=6**（1+SP_TOKENS=5）、hc_mult=4、hidden=5120 |
| 块数 | **24（100%）** ⇒ **用满全部 cube 核** |
| 每步字节 | ≈1.9 MB（x 245 KB + hc_fn 1.97 MB） |
| 按带宽所需 | **≈1.5 µs** |
| 实测 | **33.60 µs** ⇒ **高 22×** |

⇒ **M=6 的极小 batch 用满 24 个 cube 核做"K 切 20 块"的归约 ⇒ 同步淹没计算。**

### 2.3 这与既有结论一致

* 官方文档已定性：**AIV 同步占 83%**、tiling 把 K 切成 **20 块 × 24 核**；
* `HCPRE-AND-SCATTER-DEEP-DIVE` 的结论：**固定开销主导（101 次/步 × ~43 µs）**、
  **迭代数不可减**（eps 下限，12 次仍差 8e-2）、**原生实现慢 35×**；
* A1（自适应 K_L0 = 128/64/32）在服务内**从未执行**（静态内核缓存未失效），
  澄清后**效应也低于噪声底**。

---

## 3. MoE 三个 GEMM：23.6% 且已到峰值带宽

| 算子 | 次/步 | 单次 µs | ms/步 | 占步长 |
|---|---:|---:|---:|---:|
| `GroupedMatmulSwigluQuantV2`（w1/w3） | 43.0 | 64.45 | 2.771 | 10.9% |
| `GroupedMatmul`（w2） | 43.0 | 37.49 | 1.612 | 6.3% |
| （`QuantBatchMatmulV3` 里属于 MoE 的部分未单列） | — | — | — | — |
| **合计（两项）** | | | **4.383** | **17.2%** |

加上 attention 侧的 `QuantBatchMatmulV3`（183 次，2.034 ms）与 `MatMulV2`（111 次，2.289 ms），
**矩阵乘类合计 ≈ 11.09 ms = 43.6% 步长**。

**但 MoE 那部分没有余量**：微基准实测 g=48 时跑到 **1292 GB/s ≈ HBM 上限**，
且真实路径已完成 K=1.6275 更正后仍判为"跑在峰值带宽"（`BANDWIDTH-TWO-SCOPES`）。

---

## 4. 结论：下一步该做什么

### 4.1 按"可动性 × 量级"排序

| 序 | 目标 | 量级 | 可动性 | 前置 |
|---:|---|---:|---|---|
| **1** | **`HcPre` 的同步/标量开销**（78% 是开销） | **1.4~2.2 ms** | ⚠️ **中高**（kernel 级：tiling / 核数 / 同步结构） | **必须先用图内微基准验证方向**（见 §5） |
| 2 | 小核数算子（`RmsNorm` 5 核、`DynamicQuantV2` 2 核、`DequantSwigluQuant` 1 核）合并到别的算子 | 0.5~1.0 ms | ⚠️ 中（但 HcPre 融合已证明 E2E 会翻号） | 同上 |
| 3 | `Cast` 174 次（0.337 ms）+ `ViewCopy`（0.257）+ `TensorMove` | 0.6~0.9 ms | ✅ **低**（Python 侧，减的是**真实搬运**不是 launch） | 无 |
| 4 | `AivKernel` 2.817 ms（engram all_gather） | — | ❌ 已知 100% 与 HCCL 重叠 | — |
| 5 | MoE GEMM 4.383 ms | — | ❌ 已在峰值带宽 | — |

### 4.2 判据必须改（本轮负结果换来的教训）

> **算子级 standalone（eager / 固定 M / 单算子）的收益不能外推到图模式 E2E。**
> `hc_pre_norm` 的 standalone 是 **−0.459 ms**，E2E 是 **+0.359 ms**，**差 0.82 ms**。
> ⇒ 任何融合/换算子项，**必须先在"图内微基准"上验方向**，再投入工程量。

---

## 5. 建议的下一动作（二选一，都很便宜）

| 选项 | 内容 | 成本 | 产出 |
|---|---|---|---|
| **A** | **图内微基准**：把 `HcPre`（或 `HcPre+RmsNormCast`）单独放进一张 NPUGraph，replay 对比 `torch.ops._C_ascend.npu_hc_pre_v2` 原版 vs 变体；扫"核数 / K tile" | 1~2 h（不重启服务，用 tiny/die6） | 得到 **图模式下的核数→时长曲线**，直接判定 §4.1 第1项是否可做 |
| **B** | **F2 第一项**：`_forward_o_proj` 的 `output[...] = self.wo_b(...)`（消 480 KB × 37 次/步的 TensorMove） | 0.5 h + 一次重启 | 预期 **0.1~0.3 ms**，且**减的是真实搬运**，图模式下同样有效 |

**建议先 B（快、方向确定），再 A（决定最大的那一项能不能动）。**

---

## 6. 复现

```bash
# 真实排名（K≈1，步锚 HcPre/86）
ssh a3-21 'cd ~/tmp && ANCHOR=HcPre ANCHOR_PER_STEP=86 \
  python3 prof_top_ops_anchored.py ~/tmp/freshprof 26'
# HcPre 子计数拆解
ssh a3-21 'cd ~/tmp && ANCHOR=HcPre ANCHOR_PER_STEP=86 \
  python3 prof_op_subs.py ~/tmp/freshprof HcPre'
# 资源账 + 重叠矩阵
ssh a3-21 'cd ~/tmp && python3 res_acct.py ~/tmp/freshprof'
```
