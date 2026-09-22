# 113 — ★★★ 8 卡 backbone 的**设备侧 profiler 归因**（首次拿到）：前三名全是设备计算，我们的 int8 修复链只占 1.17%

> 2026-09-23 04:52–05:2x CST。采集：**主代理**起的 `r8-r8-prof` 臂（`dsa_dir_D=MERGED_FULL`、`V41_PROFILE=1`）；
> 导出：宿主 `msprof --export=on`（**离线，不占卡**）。臂产物：`shadow-pkg/results/r8_r8-prof_20260923_045235/`。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

**8 卡 backbone 的设备侧账第一次被看见**：总量 **3660.7 ms / 89 种算子**，换算 ≈**131 个 decode 步**
⇒ 每步设备 ≈27.9 ms（与 quote 口径 `hp=27.905` **自洽**）。
**前三名全是设备计算**（`HcPre` 11.08% / `GroupedMatmulSwigluQuantV2` 10.55% / `SparseFlashMla` 10.36%）；
★ **我们整条 int8 修复链（`_kv8_swa_rows_kernel`）只占 1.17%** ⇒ **它已不是瓶颈**（与 `fuse_tune` 的独立结论一致）；
★ **通信只占 1.09%** ⇒ **不是通信瓶颈**。

---

## 1. 【实测】设备侧 op 排名（`op_statistic`，TP0，按总时长降序）

| # | OP Type | Core | Count | Total (ms) | Avg (µs) | Ratio |
|---:|---|---|---:|---:|---:|---:|
| 1 | **HcPre** | MIX_AIC | 8944 | **405.66** | 45.4 | **11.08%** |
| 2 | **GroupedMatmulSwigluQuantV2** | MIX_AIC | 4472 | **386.24** | 86.4 | **10.55%** |
| 3 | **SparseFlashMla** | MIX_AIC | 4160 | **379.21** | 91.2 | **10.36%** |
| 4 | MatMulV2 | AI_CORE | 11508 | 333.14 | 28.9 | 9.10% |
| 5 | QuantBatchMatmulV3 | MIX_AIC | 19032 | 245.59 | 12.9 | 6.71% |
| 6 | GroupedMatmul | MIX_AIC | 4472 | 192.38 | 43.0 | 5.25% |
| 7 | **ScatterNdUpdateSk** | MIX_AIV | 10192 | **175.58** | 17.2 | **4.80%** |
| 8 | QuantLightningIndexerV2 | MIX_AIC | 832 | 140.60 | 169.0 | 3.84% |
| 9 | HcPost | AI_VECTOR | 8944 | 138.74 | 15.5 | 3.79% |
| 10 | MatMulV3 | AI_CORE | 4820 | 108.23 | 22.5 | 2.96% |
| 11 | RmsNorm | AI_VECTOR | 14560 | 88.32 | 6.1 | 2.41% |
| 12 | DynamicQuant | AI_VECTOR | 18824 | 84.33 | 4.5 | 2.30% |
| 13 | MoeInitRoutingV3 | MIX_AIV | 4472 | 75.33 | 16.8 | 2.06% |
| 14 | Cast | AI_VECTOR | 41165 | 72.90 | 1.8 | 1.99% |
| 15 | InplacePartialRotaryMul | AI_VECTOR | 15080 | 61.65 | 4.1 | 1.68% |
| 16 | Add | AI_VECTOR | 7297 | 46.93 | 6.4 | 1.28% |
| 17 | SparseFlashMlaMetadata | AI_CPU | 312 | 46.24 | 148.2 | 1.26% |
| 18 | QuantBatchMatmulV3 | AI_CORE | 4472 | 43.20 | 9.7 | 1.18% |
| 19 | **`_kv8_swa_rows_kernel`**（**我们的 int8 融合件**） | AI_VECTOR | 4040 | **42.70** | 10.6 | **1.17%** |
| 20 | allreduceAicpuKernel | AI_CPU | 81 | 39.78 | 491.2 | 1.09% |
| 21 | MoeTokenUnpermute | AI_VECTOR | 4472 | 37.41 | 8.4 | 1.02% |
| 22 | RmsNormCast | AI_VECTOR | 4472 | 27.66 | 6.2 | 0.76% |

---

## 2. ★★ 三条立刻可读出的结论

### 2.1 **我们的 int8 修复链已不是瓶颈**
`_kv8_swa_rows_kernel` **42.70 ms = 1.17%**（4040 次、10.6 µs/次）。
★ 与 `FUSE_TUNE` 的独立结论**互相印证**：融合件已 100% 兑现 026 的标定（19.6 µs/层），
**8 卡上 int8 相对无 int8 基线的全部残余只剩 1.09 ms**。
⇒ **int8 这条线已经收干**（剩下的都在我们的两条修复之外）。

### 2.2 **不是通信瓶颈**
`allreduceAicpuKernel` **39.78 ms = 1.09%**（81 次、491 µs/次）；
`communication_statistic_*.csv` 只有 **323 B**（几乎空的）。
⇒ 与 `PROF_int8` 早期分析（那次发现 `COMMUNICATION core 431.3 ms/43.6%`）**完全不同** ——
★ 那次是**另一种配置/相位**（`046`-era），**本代臂的 decode 稳态通信很小**。
⇒ 这条**否掉了**"通信是瓶颈"的旧印象。

### 2.3 前三名全是设备计算，且都是**每层固定次数**的形态
| 算子 | Count | 每步次数（÷131） | 解读 |
|---|---:|---:|---|
| `HcPre` | 8944 | **68.3** | 40 层 × 2 层 Engram 之类的高频**预处理** |
| `GroupedMatmulSwigluQuantV2` | 4472 | 34.1 | MoE 的 SwiGLU 量化融合 |
| `SparseFlashMla` | 4160 | 31.8 | 稀疏 MLA attention |
| `ScatterNdUpdateSk` | 10192 | 77.8 | ★ **写侧**（KV 写回）——与 `PROF_int8` 量的"写侧 9.7 µs/层"同族 |
| `QuantBatchMatmulV3` | 19032 | 145.3 | 量化 matmul（**次数最多**） |
| `Cast` | 41165 | **314.2** | ★ 最多的算子（每次 1.8 µs ⇒ 合计 72.9 ms） |

★ `Cast` 41165 次 × 1.8 µs：**数量级最大的就是"小算子太多"** —— 这与 `023` 的结论
（"读侧是**算子个数 × 每核延迟**主导，不是带宽"）是**同一族现象**，只是这次在 backbone 上。

---

## 3. 【实测】口径换算（必须先对齐窗口与步数）

| 假设窗口步数 | 每步设备时长 |
|---:|---:|
| 20 | 183.0 ms |
| 40 | 91.5 ms |
| 72 | 50.8 ms |
| **131** | **27.9 ms** |
| 200 | 18.3 ms |

★ 用锚点定步数：quote 口径 `hp = 27.905 ms/step` ⇒ `3660.7 / 27.905 = **131.2**` ⇒ **窗口 ≈131 步**，
与"2 发 × 每发 256 输出 token"（每步约 2 token ⇒ 每发约 130 步）**量级吻合**。
⇒ 每步设备 ≈27.9 ms 与 `ENGRAM_TUNE` 的 `G ≈ 29.5–30.6` 同量级（差 5–8%，含窗口边界）⇒ **自洽**。

---

## 4. 由此得到的**下一步靶子**（按设备占比降序）

| # | 靶子 | 设备占比 | 可动手的方向 |
|---:|---|---:|---|
| 1 | **`HcPre`** 405.7 ms（68 次/步、45.4 µs/次） | **11.08%** | 高频预处理算子；先查"能否合并/降频" |
| 2 | **`GroupedMatmulSwigluQuantV2`** | 10.55% | MoE SwiGLU+量化融合；属模型主体计算 |
| 3 | **`SparseFlashMla`** | 10.36% | attention 本体 |
| 4 | **`ScatterNdUpdateSk`** 77.8 次/步 | **4.80%** | ★ **写侧**；与 int8 写路径相关（`PROF_int8` 已量 9.7 µs/层） |
| 5 | **`Cast` 314 次/步** | 1.99% | ★ **"小算子过多"** 的典型；可查能否合并 |
| 6 | `QuantBatchMatmulV3` 145 次/步 | 6.71% | 量化 matmul |

★ **诚实标注**：以上是**设备侧占比**，而 `ENGRAM_TUNE` 已证明**步是设备限定的**（`G` 占 93–95%）
⇒ **这些占比可以直接近似为"对 `hp` 的贡献占比"**（【推断·强】）。
但**能不能真省下来**取决于每个算子的结构性（是否可融合、是否可降频、是否可换 dtype），
**本卷尚未对任何一条做"改完再量"的验证** ⇒ 全部标 **【未确认】**。

---

## 5. 副产品：一条踩坑记录（`msprof` 的语义）

`msprof --export=on` 的 `--output` **既是输入也是输出**：
- ❌ `--output=<空目录>` ⇒ `ERROR: The path "…" does not have PROF dir`
- ❌ `--export=on` 在 `_ascend_pt` 目录里裸跑 ⇒ 同上
- ✅ **`--output=<那个含 `PROF_*` 子目录的 `_ascend_pt` 目录>`** ⇒ 导在原地，
  产物落 `PROF_*/mindstudio_profiler_output/{op_summary,task_time,op_statistic,api_statistic,communication_statistic}_*.csv`

★ 另两条（都实测）：
1. **产物属主是 root**（worker 以 root 写）⇒ 读要用 `sudo`，分析前 `sudo chown -R` 一次更省事；
2. **原始 `*_ascend_pt` 目录是 4.0K 的占位**（前两次 `start_profile` 后没发请求就 `stop`）——
  **只有真发过请求的那次才产出 ~395 MB/rank**（本次 8 rank 合计 **3.1 GB**）。
  ⇒ 判据：`du -sh <rank>_ascend_pt`，**<1 MB 就是空采集**。
