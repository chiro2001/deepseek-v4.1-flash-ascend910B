# DSpark 的 decode 时延增量拆解（并发 4，A/B profiler 对照）

> 口径：CED-PD，A3-21，`MAX_SEQS=4`，2048 prompt / 256 输出 / **并发 4** /
> `ignore_eos`，**2 轮热身**后采集。两臂只差 DSpark 相关开关。
> profiler 为 rank0（tp0）完整 session，已上传 COS 并登记 links-server。

| | A 臂 | B 臂 |
|---|---|---|
| `ARM` | `delivery` | `draft_graph` |
| `SPEC` / `SP_TOKENS` / `DRAFT_GRAPH` | 0 / — / 0 | **1 / 7 / 1** |
| `V41_CED_ALLOW_DSPARK` | — | 1 |
| KV 组数 | 12 | **13**（+G12 草稿 SWA） |

## 0. 结论：+17.03 ms/step 的构成

**step 周期（主口径）**：A **28.19 ms** → B **45.22 ms**，**Δ = +17.03 ms（1.60×）**。

| 增量来源 | ms/step | 占增量 | 说明 |
|---|---:|---:|---|
| **① 草稿 3 层本体** | **+2.2** | **13%** | 直接新增的设备工作，可用 M 值精确分离 |
| **② target 计算放大** | **+11.2** | **66%** | M 从 4 → 32，逐算子变贵 |
| **③ 通信放大** | **+3.6** | **21%** | TP allreduce / EP aio 随 M 放大 |
| | **+17.0** | 100% | |

**关键读数：草稿本身只占 13%。87% 的增量来自"target 那一步变宽了"。**

## 1. 为什么"变宽"这么贵：M 4 → 32

推测解码把每步的行数从 `batch × 1` 抬到 `batch × (1 + SP_TOKENS)`：
`4 → 32`，**8 倍**。而产出只多 `A ≈ 2.4` 倍。

用 M 值把每个 kernel 归到 target / draft 后（`tools/ced_prof_split.py`）：

| | A（M=4） | B target（M=32） | B draft（M=28） |
|---|---:|---:|---:|
| 每步 core 时 | 27.45 ms | 29.09 ms | **2.17 ms** |
| 层数 | 40 | 40 | 3 |
| **每层均摊** | **0.686 ms** | **0.727 ms** | **0.723 ms** |

**三个数几乎一样。** 这就是全部答案：

- target 每层只贵了 **+6%**（M 涨 8 倍）⇒ target 是**访存/固定开销受限**，不是算力受限；
- 草稿每层与 target 每层**同价**（0.723 vs 0.727），所以加 3 层就是再加
  `3/40 = 7.5%` 的层成本；
- 于是 **45.22 ≈ 28.19 × (43/40) × 1.06 + 通信放大 ≈ 28.19 × 1.60**。

换句话说：**DSpark 在并发 4 下的代价 ≈ "多跑 3/40 层" × "行数放大带来的 1.06~1.6 倍系数"。**

## 2. 逐算子明细（同为 batch=4 的稳态窗口）

窗口：A `t≥1.9s`（跳过 prefill），B `t∈[1.9, 4.55]s`（**同为 4 路**，避开请求退出后的降批段）。

| 算子 | A ms/step | B ms/step | Δ | 每 op 的 µs/op 变化 |
|---|---:|---:|---:|---|
| `GroupedMatmulSwigluQuantV2`（MoE gmm1） | 3.131 | 5.414 | **+2.283** | 78 → 130 µs（+66%） |
| `hcom` + `AivKernel`（TP/EP 通信） | 4.966 | 8.551 | **+3.585** | 通信体量随 M 放大 |
| `GroupedMatmul`（MoE gmm2） | 1.766 | 3.152 | **+1.386** | 44 → 75 µs（+71%） |
| `aclnnMatmul`（o_lora / 稠密） | 3.420 | 4.612 | **+1.192** | 24 → 30 µs（+24%） |
| `SparseFlashMla`（attention 主体） | 1.555 | 2.551 | **+0.996** | 39 → 64 µs（+64%） |
| `InplacePartialRotaryMul` | 0.481 | 1.320 | **+0.839** | 3.5 → 9.1 µs（**+158%**） |
| `RmsNorm` | 0.594 | 1.382 | **+0.787** | 4.6 → 9.9 µs（**+115%**） |
| `ScatterNdUpdateSk` | 0.314 | 1.055 | +0.741 | |
| `HcPre` | 2.920 | 3.638 | +0.718 | 36.5 → 42.3 µs（+16%） |
| `HcPost` | 0.614 | 1.265 | +0.650 | |
| `QuantMatmulWeightNz` | 2.809 | 3.300 | +0.491 | |
| `SparseAttnSharedkv`（**仅草稿有**） | 0.000 | 0.389 | +0.389 | 草稿 SWA |
| `MoeGatingTopKHash` | 0.204 | 0.586 | +0.382 | |
| `allgatherAicpuKernel`（**仅草稿有**） | 0.000 | 0.380 | +0.380 | |
| `aclnnIndex` / `IndexSelect` | 0.046 | 0.667 | +0.621 | |
| `InplaceCopy` / `Add` / `DynamicQuant` | 1.301 | 2.323 | +1.022 | |
| `SparseAttnSharedkvMetadata`（**仅草稿有**） | 0.000 | 0.250 | +0.250 | |

**两类规律**：

1. **小算子被放大得最狠**（RotaryMul +158%、RmsNorm +115%）——
   它们原本只有 3.5–4.6 µs，M=4 时几乎全是固定开销；
2. **MoE / attention 只放大 1.6–1.7 倍**——它们本来就跑在访存带宽上，
   行数翻 8 倍不等于时间翻 8 倍。

## 3. 没有新增气泡

| | core 时/步 | step 周期 | core/step |
|---|---:|---:|---:|
| A | 27.45 ms | 28.19 ms | **97.4%** |
| B | 46.23 ms | 45.22 ms | **102.2%** |

`core/step > 100%` 说明多流之间还有重叠。**B 的"设备空闲"比 A 更少**
⇒ **+17 ms 全部是真实设备工作量，没有任何同步/调度气泡可挖**。

## 4. 附：B 的 step 周期随时间变化（批大小效应）

| 阶段 | 步区间 | ms/step | batch | M |
|---|---:|---:|---:|---:|
| 4 路稳态 | 12–72 | **45.0** | 4 | 32 |
| 退出中 | 72–96 | 41.4 → 37.3 | 3→2 | 24→16 |
| 单路 | 96–173 | **33.5** | 1 | 8 |

对比 A：**全程 28.2**（batch 恒为 4，M=4 不变）。

⇒ **批处理的边际成本，B 是 A 的 3.8 倍**（4 路 vs 1 路：B +11.5ms，A +4.0ms）。
这是"每步 8 行放大"的直接后果，也是并发越高 DSpark 越不划算的原因。

## 5. 优化空间在哪

按"能砍多少 × 可行性"排序：

| 机会 | 预估 | 依据 |
|---|---|---|
| **按批大小自适应 K**（低并发 K=7、高并发 K=0） | 避开全部 +17ms | 见 `CED-PD-DYNAMIC-SPEC-20260926.md`：框架已有 `num_speculative_tokens_per_batch_size` |
| **通信放大 +3.6ms** | 部分可砍 | 通信体量随 M 线性，若能只传 1 行（而非 8 行）需要改协议，属大改 |
| **小算子融合**（RmsNorm + RotaryMul + DynamicQuant 合计 +1.9ms） | 0.3–0.6ms | 这三个在 M=32 下都是 9–10 µs，融合能省 kernel 启动与中间写回 |
| **草稿 3 层本体 +2.2ms** | 至多 2.2ms | 已在图模式；要再降需改草稿结构 |

**不建议的方向**：调 `SP_TOKENS`。几何衰减决定 pos5/6 接受率≈0，
砍 K 只会等比例减少产出而几乎不省时间（因为省的是被拒绝的草稿行，
而 target 那 8 行是固定要跑的）。

## 6. 原始数据

两份 rank0 完整 session 已上传并登记 links-server（页面 b.chiro.work:18080）：

| 臂 | 对象 | 大小 |
|---|---|---:|
| A（SPEC=0） | `ced_dspark_AB_20260926_A_spec0_conc4_rank0.tar.zst` | 299 MB |
| B（SPEC=7） | `ced_dspark_AB_20260926_B_spec7_conc4_rank0.tar.zst` | 298 MB |

解包：`tar --zstd -xf <file>`，各含
`kernel_details.csv`（190/161 MB）、`task_time.csv`（105/83 MB）、
`operator_details.csv`、`api_statistic.csv`、`trace_view.json`。

分析工具（本仓库）：

```bash
# 每步周期 + 时序（用 HcPre 次数精确定步：SPEC=0 → 80/步，SPEC=7 → 86/步）
python3 experiments/dspark/step_period.py kernel_details.csv 80 "A"
# 逐算子 A/B 对照（自动按步数归一到 ms/step）
python3 tools/ced_prof_ab.py a.csv:80:1.9 b.csv:86:1.9:4.55
# 按 M 值把每步拆成 target / draft / 其他
python3 tools/ced_prof_split.py b.csv:86:1.9:4.55 \
    --target-m "32,48,96,144,192" --draft-m "28,42,84,126,168"
```
