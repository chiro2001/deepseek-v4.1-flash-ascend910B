# 真实主流算子序列 vs 我的微基准排布：逐项对照（2026-10-07）

> 数据源：`armF_r6_base` 交付口径 profile，**主流 = stream 109**，一个 step（步长 43242 µs，profile 口径）。
> 工具：`tools/shunt_dump_sequence.py`、`tools/shunt_block_structure.py`。全部【实测】。

---

## ① 主流算子：按（设备类型 × block）汇总

| 类型 | block | 个数/步 | 合计 µs | 占比 | 均 µs |
|---|---:|---:|---:|---:|---:|
| **AIC** | **24** | **278** | **14950.8** | **53.2%** | 53.78 |
| **AIV** | **48** | **553** | **7360.0** | **26.2%** | 13.31 |
| AIC | 23 | 45 | 1494.5 | 5.3% | 33.21 |
| AIC | 22 | 43 | 1208.3 | 4.3% | 28.10 |
| AIC | 16 | 52 | 906.0 | 3.2% | 17.42 |
| AIC | 20 | 43 | 650.8 | 2.3% | 15.14 |
| AIV | 1 | 362 | 486.1 | 1.7% | 1.34 |
| AIV | 24 | 69 | 400.4 | 1.4% | 5.80 |
| AIV | 16 | 43 | 241.8 | 0.9% | 5.62 |
| AIV | 4 | 51 | 193.8 | 0.7% | 3.80 |
| AIC | 6 | 8 | 72.4 | 0.3% | 9.05 |
| AIV | 3 | 43 | 64.8 | 0.2% | 1.51 |
| AIV | 2 | 24 | 31.9 | 0.1% | 1.33 |
| AIV | 20 | 6 | 24.6 | 0.1% | 4.10 |
| AIV | 6 | 4 | 18.7 | 0.1% | 4.67 |

**合计：1624 算子/步；AIC 型 ≈ 17.5 ms，AIV 型 ≈ 8.5 ms（比 2.05:1）**

**要点**：**AIC 侧 53% 的时长集中在 278 个 24-block 算子上**（`HcPre` / `SparseFlashMla` /
`GroupedMatmulSwigluQuant` / `GroupedMatmul`）；**AIV 侧 26% 集中在 553 个 48-block 算子上**。
两者都"宽"（24-block 与 48-block 都占满 24 核）。

---

## ② 真实序列的前 70 个算子（`*` = 类型切换）

| idx | T | core | blk | dur µs | gap µs | | 算子 |
|---:|---|---|---:|---:|---:|:-:|---|
| 0 | AIC | MIX_AIC | 24 | 38.2 | 0.0 | | `HcPre` |
| 1 | AIV | AI_VECTOR_CORE | 48 | 18.7 | 0.8 | * | `RmsNorm` |
| 2 | AIV | AI_VECTOR_CORE | 16 | 5.6 | 0.0 | | `DynamicQuantV2` |
| 3 | AIC | MIX_AIC | 20 | 15.4 | 1.2 | * | `QuantBatchMatmul` |
| 4 | AIV | AI_VECTOR_CORE | 48 | 12.4 | 1.2 | * | `RmsNorm` |
| 5 | AIV | AI_VECTOR_CORE | 4 | 3.7 | 0.0 | | `DynamicQuantV2` |
| 6 | AIC | MIX_AIC | 16 | 17.9 | 3.2 | * | `QuantBatchMatmul` |
| 7 | AIV | AI_VECTOR_CORE | 48 | 10.9 | 28.2 | * | `InplacePartialRotaryMul` |
| 8 | AIC | MIX_AIC | 24 | 55.3 | 1.0 | * | `SparseFlashMla` |
| 9 | AIV | AI_VECTOR_CORE | 3 | 1.3 | 1.0 | * | `Neg` |
| 10 | AIV | AI_VECTOR_CORE | 48 | 5.5 | 0.0 | | `InplacePartialRotaryMul` |
| 11 | AIC | AI_CORE | 22 | 28.0 | 0.0 | * | `MatMulV2` |
| 12 | AIC | AI_CORE | 23 | 16.5 | 0.0 | | `MatMulV2` |
| 13 | AIV | AI_VECTOR_CORE | 24 | 5.1 | 34.8 | * | `InplaceCopy_TensorMove` |
| 14 | AIV | AI_VECTOR_CORE | 48 | 17.2 | 0.2 | | `HcPost` |
| 15 | AIC | MIX_AIC | 24 | 38.8 | 1.0 | * | `HcPre` |
| 16 | AIV | AI_VECTOR_CORE | 48 | 10.0 | 0.8 | * | `RmsNormCast` |
| 17 | AIC | AI_CORE | 24 | 19.3 | 0.2 | * | `MatMulV3` |
| 18 | AIV | AI_VECTOR_CORE | 1 | 1.2 | 0.0 | * | `Cast` |
| 19 | AIV | AI_VECTOR_CORE | 48 | 15.5 | 0.0 | | `MoeGatingTopKHash` |
| 20 | AIV | AI_VECTOR_CORE | 1 | 1.3 | 0.2 | | `Less` |
| 21 | AIV | AI_VECTOR_CORE | 1 | 1.3 | 0.0 | | `GreaterEqual` |
| 22 | AIV | AI_VECTOR_CORE | 1 | 1.3 | 0.2 | | `LogicalOr` |
| 23 | AIV | AI_VECTOR_CORE | 1 | 1.4 | 0.5 | | `MaskedFillScalar` |
| 24 | AIV | MIX_AIV | 48 | 16.4 | 1.2 | | `MoeInitRoutingV3` |
| 25 | AIC | MIX_AIC | 24 | 184.8 | 3.0 | * | `GroupedMatmulSwigluQuant` |
| 26 | AIC | MIX_AIC | 24 | 109.8 | 3.0 | | `GroupedMatmul` |
| 27 | AIV | AI_VECTOR_CORE | 1 | 1.2 | 1.0 | * | `Cast` |
| 28 | AIV | AI_VECTOR_CORE | 1 | 1.2 | 0.0 | | `Abs` |
| 29 | AIV | AI_VECTOR_CORE | 48 | 10.1 | 0.0 | | `MoeTokenUnpermute` |
| 30 | AIV | AI_VECTOR_CORE | 48 | 9.0 | 4.2 | | `Add` |
| 31 | AIV | AI_VECTOR_CORE | 48 | 16.8 | 21.5 | | `HcPost` |
| 32 | AIC | MIX_AIC | 24 | 37.0 | 1.2 | * | `HcPre`（下一层开始） |

**（idx 32~63 是 idx 0~31 的重复 = 一层；末尾 64~69 进入 MoE/索引链）**

---

## ③ 真实的"同类型块"结构（这是关键）

| 项 | AIC | AIV |
|---|---:|---:|
| 块数/步 | **380** | **380** |
| 每块算子数：中位 | **1** | **2** |
| 每块算子数：均值 | **1.2** | **3.0** |
| p90 | 2 | 7 |
| 最大 | 2 | 35 |

块大小分布：

```
AIC: {1 个算子: 291 块, 2 个: 89 块}            ← 几乎全是单算子块
AIV: {1: 96, 2: 179, 4: 11, 5: 39, 6: 5, 7: 43, 9: 4, 30: 2, 35: 1}
```

**⇒ 主流是"AIC 单算子 ←→ AIV 1~2 算子"的极细粒度交替，760 次切换/步。**

前 24 个块的实际构成：

```
blk  T    #ops   dur µs  算子
  0  AIC    1     38.2    HcPre(24,38)
  1  AIV    2     24.3    RmsNorm(48,18) + DynamicQuantV2(16,5)
  2  AIC    1     15.4    QuantMatmulWeightNz(20,15)
  3  AIV    2     16.1    RmsNorm(48,12) + DynamicQuantV2(4,3)
  4  AIC    1     17.9    QuantMatmulWeightNz(16,17)
  5  AIV    1     10.9    InplacePartialRotaryMul(48,10)
  6  AIC    1     55.3    SparseFlashMla(24,55)
  7  AIV    2      6.8    Neg(3,1) + InplacePartialRotaryMul(48,5)
  8  AIC    2     44.5    MatMulV2(22,27) + MatMulV2(23,16)
  9  AIV    2     22.3    InplaceCopy_TensorMove(24,5) + HcPost(48,17)
 10  AIC    1     38.8    HcPre(24,38)
 11  AIV    1     10.0    RmsNormCast(48,9)
 12  AIC    1     19.3    MatMulV3(24,19)
 13  AIV    7     38.3    Cast(1) + MoeGatingTopKHash(48,15) + Less(1) + ...
 14  AIC    2    294.6    GroupedMatmulSwigluQuant(24,184) + GroupedMatmul(24,109)
 15  AIV    5     38.3    Cast(1) + Abs(1) + MoeTokenUnpermute(48,10) + ...
```

---

## ④ ★ 我的微基准排布（对照）

**算子选择（4 个代理）**：

| 代理 | 真实对应 | 我的 core | 我的 block | 实测 µs |
|---|---|---|---:|---:|
| `mix` | `HcPre`/`SparseFlashMla`/`GroupedMatmul` | **MIX_AIC** | **48** ⚠️ | 12.4~13.2 |
| `mm` | `MatMulV2`/`MatMulV3` | AI_CORE | 20 | 7.8~8.1 |
| `rms` | `RmsNorm`/`HcPost`/`MoeGating` | AI_VECTOR_CORE | 48 | 7.6~8.0 |
| `dqs` | `DynamicQuantV2`/`Cast`/`Neg` | AI_VECTOR_CORE | 4 | 3.5~3.6 |

**排布**：每"层" AIC 8 个 `[mix,mm,mix,mm,mix,mm,mix,mm]` + AIV 5 个 `[rms,dqs,rms,dqs,rms]`

| 臂 | 排布方式 |
|---|---|
| `interleaved` | 上述 13 个算子 **1:1 交替**（AIC,AIV,AIC,AIV…） |
| `free` | AIC 全放流 A、AIV 全放流 B，**仅首尾同步** |
| `bar32` | AIC 按 **32 个算子一格**，同格 AIV 取**累计时长匹配**者，**格间加屏障** |
| `split_matched` | 按累计时长切成 22 个阶段（目标 60 µs/阶段），阶段间屏障 |

---

## ⑤ 真实 vs 我的排布：关键差异（决定结论可信度）

| 项 | **真实主流** | **我的微基准** | 影响 |
|---|---:|---:|---|
| 算子数 | 1624 / 步 | 416 | 规模小 4× |
| **类型切换次数** | **760 / 步** | 416（interleaved） | 同量级 ✅ |
| **AIC 块大小（中位）** | **1** | 1（interleaved）/ 32（bar） | ⚠️ bar 臂块过大 |
| **AIV 块大小（中位）** | **2** | 1 / 6 | ⚠️ |
| **AIC:AIV 时长比** | **2.05 : 1** | 2.72 : 1 | ⚠️ 我的 AIV 偏少 |
| **`mix` 代理 block** | **真实 24** | **我用的 48** | ⚠️⚠️ **最需要复测的一项** |
| 真实 `dqs` 类占比 | AIV 中 1~16 block 合计 ≈ 2.0 ms | 我用 3.6 µs/个 | 量级接近 |

### 两个必须承认的偏差

1. **`mix` 用了 48-block 代理**（`moe_init_routing`），而生产是 **24-block** 的
   `HcPre`/`SparseFlashMla`/`GroupedMatmul`。**24-block 已经占满 24 核**，
   所以"宽算子不可重叠"的方向应成立，但**具体重叠系数/切换代价必须用 24-block 复测**。
2. **`bar32` 的块太大**（32 个 AIC 算子一格），真实是 1~2 个算子一块。
   所以"屏障无用"的结论**不能**直接外推到真实结构 —— 真实的块细得多。

---

## ⑥ 下一步该做的（据此修正）

| # | 动作 | 目的 |
|---:|---|---|
| 1 | **用 24-block 代理复刻真实块结构**（AIC 1 块 / AIV 2 块，760 次切换） | 验证切换代价 3.96 µs（trace）是否可复现 |
| 2 | 在该结构下测 `interleaved` vs `free` vs `分块屏障` | 给出**可信**的收益数字 |
| 3 | 测"**同类型块加大**"（把 2 个 AIV 块合并） | 验证不需第二流的路径 |

> 已入仓工具：`tools/shunt_dump_sequence.py`、`tools/shunt_block_structure.py`。
