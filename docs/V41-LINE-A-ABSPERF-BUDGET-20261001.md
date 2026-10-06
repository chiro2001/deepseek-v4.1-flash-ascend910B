# 线 A：TP8+DCP8 decode 步的设备预算与空闲归因（2026-10-01）

> 目标：**优化绝对性能**（不再以 DCP8/DCP1 比值为判据）。
> 环境：a3-21 chip 8–15，容器 `dsv41-abs`，SPEC=0，PREFIX=0，BAT_TOKENS=2048，
> MAX_SEQS=16，BAT_TOKENS=2048，DCP=8，`--no-async-scheduling`，
> `V41_DCP_REPLICATE_INDEXER=1`，KV cache 6,082,458 token。
> 所有结论标注【实测】/【推断】/【未确认】。

## 0. 三条最重要的话

1. **decode 步的设备空闲是 13.94 ms/step（占步跨 37.0%），其中 65% 集中在一串
   元素级小算子上**；而这些小算子自身的内核时间合计不到 1 ms。【实测】
2. **集合通信不产生空隙**（`hcom_*` 的归因空闲 0.227 ms/step，`idle≥50µs` 为 0）。
   减通信相位对墙钟的杠杆小于预期。【实测】
3. 因此**融合/减少元素级算子个数**（而不是减少算子时间、也不是减少通信字节）
   是线 A 当前最大的单笔杠杆。【推断】

## 1. 口径（不先看这节会误读所有数字）

### 1.1 profile 抬高墙钟 ~10%

同一实例同一 prompt：

| 状态 | ms/step |
|---|---:|
| 无 profile（3 轮中位） | **32.58** |
| profile 中（3 轮中位） | **35.88** |

⇒ 本报告的 span / idle 都同时给出 **×0.857** 折算列。

### 1.2 必须去重 hcom

`op_summary_*.csv` 把每个通信算子**记录两遍**：一条
`hcom_allReduce__<gid>_<seq>_<k>`，一条 `AivKernel`。
实测 named = 15955、AivKernel = 15955（完全相等）【实测】。
不去重会把通信量算成 2 倍。`lineA/tools/idle_report.py` 只保留 named。

### 1.3 只统计"干净单步"

按开始时间排序后以 1500 µs 间隙切簇，簇边界会把相邻步切碎/并拢。
只保留 **`QuantBatchMatmulV3` 条数恰为 208** 的簇（本部署的结构常量）。
本次 179 个簇里筛出 **81 个干净步**。

### 1.4 空闲必须用区间并集求补集

按开始时间排序后相邻两条之间的空当，可能只是另一条**更早开始、仍在运行**的
长算子 —— 那不是空闲。正确做法：先区间并集，取**补集**，再归因到
"空闲开始前最后结束的算子"。
（用"相邻配对"会把 void 算成 idle，本文早期版本犯过这个错。）

## 2. 总账【实测】

| | profiled | ×0.857 |
|---|---:|---:|
| span | 37.68 ms/step | 32.29 |
| busy（区间并集） | 23.74 ms/step | 20.35 |
| **idle** | **13.94 ms/step（37.0%）** | **11.95** |
| idle ≥ 50 µs | 9.09（占全部空闲 65.2%） | 7.79 |

### 空闲直方图

| 区间 | ms/step | ×0.857 | 条/step | 次均 |
|---|---:|---:|---:|---:|
| 0–5 µs | 1.771 | 1.518 | **1521.2** | 1.16 µs |
| 5–20 | 0.971 | 0.832 | 74.2 | 13.08 |
| 20–50 | 2.107 | 1.805 | 74.1 | 28.44 |
| 50–100 | 2.341 | 2.006 | 33.0 | 70.96 |
| 100–200 | 1.783 | 1.528 | 12.5 | 142.4 |
| 200–500 | 2.930 | 2.511 | 9.9 | 297.1 |
| 500–1000 | 0.889 | 0.762 | 1.0 | 911.8 |
| 1000–2000 | 1.148 | 0.984 | 1.0 | 1176.7 |
| **合计** | **13.940** | **11.946** | 1726.8 | |

* 0–5 µs 桶有 **1521 条/步**、次均 1.16 µs —— 这是**每个 kernel 边界的下限**
  （下发/依赖检查），几乎不可压缩。
* **>200 µs 的 12.9 条/步 = 4.97 ms/step** 才是可攻的目标。

## 3. ★ 归因表：谁结束后设备停摆【实测】

| 前算子（空闲前最后结束） | idle≥50µs ms/step | ×0.857 | 条/step | 次均 |
|---|---:|---:|---:|---:|
| **SelectV2** | **3.043** | **2.608** | 22.3 | 136.6 µs |
| **Cast** | 1.572 | 1.347 | 2.7 | 592.4 |
| **Sub** | 0.960 | 0.822 | 2.0 | 488.8 |
| **Add** | 0.793 | 0.679 | 5.1 | 154.7 |
| _compute_slot_mapping_kernel | 0.631 | 0.541 | 7.8 | 81.1 |
| Fill | 0.445 | 0.381 | 3.3 | 135.0 |
| FloorMod | 0.326 | 0.279 | 2.0 | 160.9 |
| GatherV3 | 0.257 | 0.220 | 1.1 | 233.7 |
| ClipByValueV2 | 0.157 | 0.134 | 1.3 | 122.0 |
| RmsNorm | 0.149 | 0.128 | 1.0 | 149.2 |
| QuantLightningIndexerV2Metadata | 0.135 | 0.116 | 1.5 | 93.0 |
| 其余 ≤0.134 | | | | |

## 4. ★ 优先级表：kernel + 归因空闲【实测】

| OP Type | kernel | idle | idle≥thr | 合计 | ×0.857 |
|---|---:|---:|---:|---:|---:|
| hcom_allReduce_ | 5.642 | 0.186 | **0.000** | 5.642 | 4.836 |
| **SelectV2** | **0.074** | **3.244** | **3.043** | **3.117** | **2.671** |
| HcPre | 2.243 | 0.079 | 0.000 | 2.243 | 1.922 |
| **Cast** | 0.477 | 2.176 | 1.572 | 2.050 | 1.757 |
| GroupedMatmulSwigluQuantV2 | 2.034 | 0.075 | 0.000 | 2.034 | 1.743 |
| QuantBatchMatmulV3（208 次） | 1.956 | 0.180 | 0.000 | 1.956 | 1.676 |
| MatMulV2 | 1.652 | 0.040 | 0.000 | 1.652 | 1.416 |
| SparseFlashMla（78 次） | 1.478 | 0.148 | 0.000 | 1.478 | 1.267 |
| GroupedMatmul | 1.324 | 0.043 | 0.000 | 1.324 | 1.134 |
| **Sub** | 0.173 | 1.183 | 0.960 | 1.132 | 0.970 |
| **Add** | 0.104 | 0.957 | 0.793 | 0.897 | 0.769 |
| hcom_allGather_ | 0.731 | 0.041 | 0.000 | 0.731 | 0.627 |
| RmsNorm | 0.532 | 0.152 | 0.149 | 0.681 | 0.584 |
| _compute_slot_mapping_kernel | 0.024 | 0.632 | 0.631 | 0.655 | 0.561 |
| MatMulV3 | 0.650 | 0.004 | 0.000 | 0.650 | 0.557 |
| Sort | 0.530 | 0.000 | 0.000 | 0.530 | 0.454 |
| SparseFlashMlaMetadata | 0.411 | 0.139 | 0.108 | 0.520 | 0.445 |
| HcPost（80 次） | 0.508 | 0.092 | 0.000 | 0.508 | 0.435 |

**读法**：`SelectV2` 自身只花 0.074 ms/step，但它结束后设备停摆 3.243 ms
（22.3 次/步、次均 137 µs）—— **是该算子内核耗时的 42 倍**。

疑似来自 `_v41_dcp_merge_attention` 的四个算子
（SelectV2 / Cast / Sub / Add）：`idle≥50µs` 合计
**6.368 ms/step profiled = 5.46 ms/step 无 profile 口径**。【推断】

## 5. 两个反直觉的负结果【实测】

1. **集合通信不产生空隙**：`hcom_allReduce_` 归因空闲 0.186 ms/step、
   `idle≥50µs` **恰好为 0**；`hcom_allGather_` 同理。通信是"跑得满满的长算子"，
   其它 kernel 在它边上并行。⇒ 减通信相位/字节对墙钟的杠杆
   **小于**减元素级算子个数。
2. **空闲不是"一条大尾巴"**：0–5 µs 桶占了 1521 条/步 —— 每相邻两个 kernel
   之间就有 ~1.16 µs 的边界开销，这部分近似不可压。

## 6. 对 merge 融合算子的意义（预测）

单卡实测 merge 融合算子省 **0.520 ms/step**（逐位一致）—— 那是**内核时间**口径。
本报告的归因显示：它真正要消的是**调度空隙**，量级
**5.5–7.5 ms/step（无 profile 口径）**。【推断】

**验证方法**（8 卡 before/after，同一个实例、同一 prompt）：

```bash
python3 lineA/tools/idle_report.py <mindstudio_profiler_output 目录>
```

只看两节：`[1]` 的 `idle (ms/step)` 与 `[3]` 里 SelectV2/Cast/Sub/Add 四项。
**注意必须换新 run_id 抓 profile**，且不要复用目录（msprof export 会覆盖）。

## 7. 待补的对照

* **DCP1 的 profile**：DCP8−DCP1 = 5.90 ms/step，但 §3 识别出的空隙有 7.5 ms
  ⇒ 其中必然有一部分 DCP1 也有。缺这一块就无法把"可回收量"钉死。【未确认】
* **步首 2 条 ~477 µs 的 group503 allReduce**：81 步里 79 步恰好 2 条，
  位置固定在步内索引 ~20（全场 3210 条算子里非常靠前），
  前一个算子 `MaskedFill`（81/81）、后一个 `Tile`（80/81）。
  不是重复计数（已用 hcom_dedup 验证）。是 DCP 引入的还是本来就有的，
  需要 DCP1 对照。【未确认】

## 8. 产物

| 文件 | 内容 |
|---|---|
| `lineA/tools/idle_report.py` | ★ 权威空闲报告（五节固定格式，内置 ×0.857） |
| `lineA/tools/budget.py` | 去重后的算子账（by_type / hcom 分组） |
| `lineA/tools/gaps.py` | 早期版本（相邻配对口径，**已被 idle_report 取代**） |
| `lineA/tools/idle_attrib.py` | 早期版本（同上） |
| `lineA/tools/hcom_dedup.py` | 证明 hcom 被记录两遍 |
| `lineA/tools/slowar.py` | 定位步首 2 条慢 allReduce |
| `lineA/tools/opfwd.py` / `commstats.py` | 按 forward 聚类 / 中位步通信账 |
| `lineA/tools/woa_m1.py` | wo_a 的 M=1/2/4/8 交错 A/B |
| `lineA/out/idle_report.txt` | 本文 §2–§5 的原始输出 |
| `lineA/out/{budget.json,budget.csv,gaps50.txt}` | 原始表 |

## 9. 附：wo_a 的 M 无关性【实测】

邻居的 14.38 µs 是 M=8 口径，本项目的 decode 是 M=1。在 chip4 上以
40 层真实权重 + 整段 ACL graph、4 轮交错重测（µs/op，数值全部 max|d| = 0）：

| 臂 | M=1 | M=2 | M=4 | M=8 |
|---|---:|---:|---:|---:|
| vendor | 14.64 | 14.64 | 14.67 | 14.70 |
| vendor_pad16 | 10.06 | 10.08 | 10.07 | 10.09 |
| triton_BM16 | 9.79 | 9.75 | 9.79 | 9.79 |

⇒ **wo_a 是权重带宽 bound，与 M 无关**，邻居的结论直接适用于生产 M=1。
但**图重放墙钟**口径的增益是 **1.45×**（不是邻居单算子口径的 1.96×）：
40 次/步 ⇒ pad16 省 **0.18 ms/step**，Triton 再省 0.01。
⇒ 只做 M16 补零（零开发量），不为 wo_a 引入 Triton 依赖。

## 10. 空闲真的是"设备没活干"吗 —— stream 校验【实测】

`op_summary` 的算子分布在 **6 个 stream** 上：

| Stream ID | 条/步 | busy ms/step |
|---|---:|---:|
| 114 | 2287 | 15.46 |
| N/A（`COMMUNICATION`，即 hcom） | 197 | 6.37 |
| 111 | 160 | 1.04 |
| 112 | 160 | 0.80 |
| 47 | 361 | 0.77 |
| 39 | 35 | 0.70 |

全部 stream 一起求并集 → busy **23.74** ms/step；
各 stream **分别**求并集再相加 → **25.14** ms/step。
前者小于后者 ⇒ 存在跨 stream 重叠，但**只有 1.4 ms**，绝大多数时间只有一条流在跑。
⇒ §2 的 idle **13.94 ms/step 是真实的设备空闲**，不是"另一条流在跑而没看见"。【实测】

## 11. ★ 522 个"单元素 kernel"/步 —— 下一层根因【实测】

> **★ 本节与 §13 是同一个根因，不要分别报预算。**
> 合并后的口径见 **§18.4**：省 6 kernel/次调用 ⇒ **约 1.5 ms/step**（不是 2.3）。

每步 3200 条算子里，**522 条（16.3%）作用在只有 1 个元素的张量上**。

| OP Type | 条/步 | ms/step | 次均 |
|---|---:|---:|---:|
| **Cast** | **205.0** | 0.240 | 1.17 µs |
| **FloorDiv** | **71.2** | 0.095 | 1.33 |
| Fill | 40.1 | 0.048 | 1.20 |
| SelectV2 | 37.1 | 0.063 | 1.71 |
| BroadcastTo | 24.4 | 0.032 | 1.33 |
| FloorMod | 23.4 | 0.049 | 2.09 |
| Mul | 22.5 | 0.032 | 1.41 |
| Sub | 17.6 | 0.026 | 1.49 |
| ClipByValueV2 | 14.6 | 0.028 | 1.91 |
| GreaterEqual | 12.7 | 0.017 | 1.31 |
| Add | 12.7 | 0.023 | 1.80 |
| Equal | 10.8 | 0.017 | 1.56 |
| 其余 7 类 | ≤7.8 | | |
| **合计** | **522** | **0.725** | |

形状证据（每步计数，`Input Shapes` 已去引号）：

| OP Type | 形状 | 条/步 |
|---|---|---:|
| Cast | `1` | **201.0** |
| FloorDiv | `1;` | **71.2** |
| Fill | `1;` | 40.1 |
| SelectV2 | `1;1;` | 27.3 |
| SelectV2 | `1;1;1` | 8.8 |
| SelectV2 | `1;;1` | 1.0 |

### 11.1 来源【推断，有代码定位】

`vllm_ascend/attention/dsa_v41.py` 的 `build()` 里，**复制态 indexer 的
`[T,2]` 槽位映射**那一段（约 4088–4140 行）在 decode 时 `T = 1`
（`pos = positions[:num_input_tokens]` 只有 1 个元素），整段算术
**全都退化成单元素 kernel**，每次调用约 14 个。

实测对得上：`SelectV2 "1;1;"` 27.3 条/步 ÷ 每次调用 2 条
⇒ **约 13.65 次调用/步**。

**为什么没被缓存住**：该段用 `shared[slot_key]` 去重，`shared` 来自
`kwargs["common_v41_metadata"]`；这个 dict 在 model runner 里是
**按 kv_cache_group 新建**的（每个 cache group 一个空 dict）
⇒ 缓存只在同一 group 内生效、**跨 group 完全不复用**。
（`common_v41_batch_metadata` 才是每步建一次的那个 dict，已经传进来了，
但这段代码没用它。）

### 11.2 两处**代数上恒等**的冗余【推断，零风险】

```python
within = global_g.remainder(span)
face_offset = (torch.div(within, storage, rounding_mode="floor") * storage
               + within.remainder(storage))
```

`(w // s) * s + (w % s) ≡ w`（`s > 0`、`w ≥ 0`）
⇒ **`face_offset` 恒等于 `within`**，那 4 个 kernel
（FloorDiv + Mul + FloorMod + Add）是纯冗余。

```python
group_complete = (... if ratio > 1 else torch.ones_like(pos, dtype=torch.bool))
repl_valid = group_complete & (block_numbers > 0)
```

`ratio == 1` 时 `group_complete` 恒为全 True ⇒ `repl_valid ≡ block_numbers > 0`，
`Fill` + `LogicalAnd` 也是冗余。

### 11.3 纪律（必须遵守）

同 overlay 已**三次**记录：`PACKDIRECT` / `SUBALPHA` / `contigw` 都是
"离线微基准逐位一致、真实路径破坏正确性"。
⇒ 上述改写**必须**用**短问答**（`17 × 23`，最灵敏）做 A/B；
只跑长针会误判成"没坏"。

## 12. 下一步（按证据排序）

| # | 动作 | 依据 | 预估 |
|---|---|---|---|
| 1 | 复制态 indexer 槽位映射的**结果缓存到 `common_v41_batch_metadata`**（键含 `data_ptr(block_table)`/`data_ptr(positions)`/`T`/`ratio`/`storage`/`dcp`/步序号），group 自己的 buffer 只 `.copy_()` | §11.1 | 去掉 ~11.6 次冗余调用 × 14 kernel ≈ **160 kernel/步** |
| 2 | 消掉 §11.2 的两处代数冗余 | §11.2 | **再 ~6 kernel/次调用** |
| 3 | `T == 1` 的 decode 快速路径（`req_indices ≡ [0]`、`query_lens ≡ 1`） | §11.1 | 再 ~5 kernel/次调用 |
| 4 | merge 融合算子 | §6 | 5.5–7.5 ms/step（需 before/after gap 判据核实） |
| 5 | wo_a M16 补零 | §9 | 0.18 ms/step |

**第 1–3 项全部只作用于 `index_is_replicated` 分支**
（即 DCP>1 且 `V41_DCP_REPLICATE_INDEXER=1`），对 DCP1 无影响
⇒ 是 DCP8 专属收益。

## 13. ★★ 决定性发现：空闲不是散布的气泡，是**步后段一个 13.6 ms 的洞**【实测】

> **★ 本节与 §11 是同一个根因，不要分别报预算。**
> 本节只描述"洞在哪、有多大"；**预算合并见 §18.4**。

把步按等长时间切 40 桶（`lineA/tools/idle_timeline.py`）：

| 桶（时刻%） | 设备占用 |
|---|---:|
| 0–21（0–55%） | **87–94%** |
| 22（56%） | 68% |
| **23–35（58–90%）** | **4–13%** |
| 36（91%） | 53% |
| 38–39（96–99%） | 98% |

即：**前半段几乎满负荷，58%–90% 这一段是一条约 12 ms 的连续空洞**，
里面只跑着零星小算子。用 `lineA/tools/phase_window.py` 精确量这个窗口：

```
窗口 0.56–0.92：长 13.57 ms/步，设备只用 1.745 ms
              ⇒ host/空隙 11.82 ms/步（占 profiled 步跨的 31%）
              每步 425 条算子，平均每条 31.9 µs 预算里只有 4.1 µs 是设备时间
```

窗口内的算子构成与「该算子前面等了多久（>50 µs）」：

| OP Type | 条/步 | ms/step | **wait>50µs** | 次均 |
|---|---:|---:|---:|---:|
| **Cast** | **85.3** | 0.108 | **3.789** | 1.26 µs |
| SelectV2 | 36.0 | 0.062 | 0.000 | 1.72 |
| **Fill** | 35.9 | 0.043 | **1.385** | 1.20 |
| FloorDiv | 33.8 | 0.046 | 0.553 | 1.37 |
| BroadcastTo | 25.5 | 0.035 | 0.839 | 1.38 |
| FloorMod | 23.5 | 0.049 | 0.000 | 2.09 |
| Mul | 20.8 | 0.034 | 0.001 | 1.63 |
| ClipByValueV2 | 17.7 | 0.033 | 0.148 | 1.86 |
| Sub | 17.5 | 0.026 | 0.534 | 1.48 |
| **`_compute_slot_mapping_kernel`** | 7.8 | 0.024 | **0.736** | 3.06 |
| IndexCheck / Index | 4.8 / 4.8 | 0.014 / 0.040 | 0.263 / 0 | 2.85 / 8.39 |

### 13.1 逐条 trace 证据

打印窗口内单步的算子序列（前 70 条）可见**两段**：

1. **`dsa_v41.py::build()` 的复制态 indexer 槽位映射**（Python 层，约 30 条）：
   `Range → Sub → FloorDiv → BroadcastTo+FloorMod (×3) → Equal → Mul/Add (×4/×3)
   → Cast → Fill → SelectV2 → IndexCheck+Index`，30 条只花 **0.49 ms**（16 µs/条）。
2. **`block_table.py` 的 `_compute_slot_mapping_kernel`**（Triton）：
   连续 8 次，每次 device 只有 **~3 µs**，但**两次之间隔 64–172 µs 纯空闲**。

### 13.2 ★ 一个**已存在但默认关闭**的优化

`vllm_ascend/worker/block_table.py` 顶部已经有 `[V41-SLOT-MAP-FUSED]`：
把 **12 次 per-group 启动折成 1 次二维 grid 启动**，
单卡实测 host 时间 **1.774 → 1.048 ms/step（−41%）**。

**门控默认值 `V41_SLOT_MAP_FUSED=0`（关）**，而本次 run 的 `inner.sh` 没有设它
⇒ **走的是上游逐组路径**。【实测】

## 14. 修正后的优先级

| # | 动作 | 依据 | 预估 |
|---|---|---|---|
| **1** | **开 `V41_SLOT_MAP_FUSED=1`**（先 `verify` 跑一遍逐元素比对） | §13.2 | **0.7 ms/step**，零代码改动 |
| **2** | 复制态 indexer 槽位映射**跨 group 缓存**（存进 `common_v41_batch_metadata`） | §11.1 | 从 ~8–13 次降到 ~2 次 |
| **3** | 消掉 §11.2 两处代数冗余 | §11.2 | 每次 −6 kernel |
| **4** | 查窗口里 **Cast 3.789 ms 的等待**（85 次/步；`aclnnInplaceCopy_CastAiCore`=`.copy_()` 到异构 dtype） | §13 | 未知，量级最大 |
| **5** | merge 融合算子 | §6 | 需 before/after 用 `idle_report.py` 核实 |
| **6** | wo_a M16 补零 | §9 | 0.18 ms/step |

**注意**：#1–#3 都在同一条 `index_is_replicated` 路径上，
和 merge kernel 是**不同**的两段代码 —— 不要把两者的收益相加后再归给其中任一个。

## 15. ★★★ 把 11.8 ms 钉到单一 stream：**stream 47 上 10.88 ms 的 host 下发**

### 15.1 窗口内 85 个 Cast 的真身【实测】

| Op Name | 条/步 | 前面等 >50µs |
|---|---:|---:|
| `aclnnDivMods_CastAiCore_Cast`（`remainder` 的 cast 阶段） | **45.7** | 0.476 ms |
| `aclnnInplaceCopy_CastAiCore_Cast` | 12.1 | 0.001 |
| **`aclnnGeScalar_CastAiCore_Cast`（`>=` 比较）** | **10.9** | **3.174 ms** |
| `aclnnGtScalar_CastAiCore_Cast`（`>`） | 6.8 | 0.139 |
| 其余 4 类 | ≤2.4 | 0 |

形状：**76.1/85.3 是 1 元素标量**。stream：**75.2 在 stream 47**。

### 15.2 stream 47 的下发节奏【实测，决定性】

只取窗口内 stream 47 的算子，测相邻两条之间的空隙：

```
算子数        339.4 条/步
设备时间      0.719 ms/步
相邻 gap 合计 10.881 ms/步      ← 全部是 host/调度，不是设备计算
gap 中位      6.5 µs   p25 1.5   p75 24.8   p90 69.2
gap 直方图    0–10µs: 52.6% | 10–20: 14.4% | 20–30: 12.8% | 30–40: 4.2%
              40–50: 1.9% | ≥50µs: 8.5%（p90 达 69 µs）
```

⇒ **一半的下发是正常的（<10 µs），另一半是 10–100 µs 的异常下发。**
10.881 ms/step 全部花在「把 339 个 1 元素算子逐个交给设备」上。

这 339 条就是 §13 里那两段代码：
`dsa_v41.py::build()` 的复制态槽位映射（Python 层，占多数）
＋ `block_table.py` 的 `_compute_slot_mapping_kernel`（Triton，8 次）。

### 15.3 结论与量级

* **DCP8 − DCP1 = 5.90 ms/step**（32.58 vs 26.68）。
* stream 47 的 host 下发 = **10.881 ms/step**（profiled）⇒ 折合 **9.3 ms**（×0.857）。
  即使 DCP1 也有一半，DCP8 独有的部分也 ≥ 剩余差距。
* ⇒ **这是线 A 目前唯一量级足够大的单一杠杆**，比 merge kernel（§6）更靠上游、
  也更"干净"（纯 host 下发，不涉及数值改写）。

### 15.4 可达路径（按风险排序）

| # | 做法 | 风险 | 依据 |
|---|---|---|---|
| a | `V41_SLOT_MAP_FUSED=1`（已存在的融合 kernel，先 `verify`） | **最低**（自带逐元素比对门） | §13.2 |
| b | 复制态槽位映射也做成 **Triton/AscendC 单 kernel**，照抄 `block_table.py::_compute_slot_mappings_multi_kernel` 的模式 | 中 | §15.2 |
| c | 跨 group / 跨层**缓存**计算结果（`common_v41_batch_metadata`） | 低 | §11.1 |
| d | 消掉 §11.2 两处**代数恒等**冗余（每次 −6 kernel） | 低（但需短问答 A/B） | §11.2 |

`d` 的量级值得单独说：`within.remainder(storage)` 每次调用 = `BroadcastTo + FloorMod + Cast`
三个 kernel，加上 `div(within,storage)*storage +` 又是三个 ⇒ **每次调用可省 6 个**；
按 ~13 次调用/步、每个 ~30 µs 计 ⇒ **约 2.3 ms/step**。

## 16. 一个**必须纠正**的判断：`V41_SLOT_MAP_FUSED` 对 DCP8 无效

我一度建议开 `V41_SLOT_MAP_FUSED=1`（`block_table.py` 顶部那个"12 次 → 1 次"的优化，
单卡实测 host 1.774 → 1.048 ms/step）。**读门控源码后撤回**：

```python
# block_table.py::_v41_fused_precheck
if bt.dcp_world_size > 1:
    return f"group{i} dcp_world_size={bt.dcp_world_size} > 1（走 _compute_dcp_slot_mapping）"
```

DCP 组走的是 **`_compute_dcp_slot_mapping`** —— 它**本身就已经是一个融合 kernel**了
（这正是每步只看到 7.8 次 `_compute_slot_mapping_kernel` 而不是 12×N 的原因）。
⇒ 那个 12→1 的优化**早就为 DCP 生效**，`V41_SLOT_MAP_FUSED` 只服务非 DCP 组。【实测】

## 17. 两个分支的 kernel 数静态对比（说明 §15.4(d) 的作用域）

| | 复制态分支（DCP8 专属） | 非复制态分支（DCP1 也走） |
|---|---|---|
| 主体 | `pos//ratio`、`global_g//span`、`within`、**`face_offset`**、`query_lens`、`req_indices`、`column.clamp`、`block_table[...]`、`group_complete`、`repl_valid`、2×`where→copy_` | `slot_mapping[:n]`、**`compressed_slot_mapping`**、`valid`、`valid_end`、`arange<n`、`positions%2==1`、`clamp_min`、2×`where→copy_` |
| 约计 kernel 数 | **24–26** | **~31** |

⇒ **两个分支的算子数本来就相当**，所以 §13 那条 13.6 ms 的洞**不是 DCP8 独有的**。

但 **§15.4(d) 的补丁仍然只作用于 DCP8**：`face_offset` 那 4 个 kernel 与
`group_complete` 那 2 个内核**只存在于复制态分支**，DCP1 根本没有这一段。
⇒ 该补丁是**纯粹的 DCP8 专属绝对收益**（约 2.3 ms/step）。【推断，待 A/B】

**要把"洞"里可回收的量钉死，仍然需要 DCP1 的 profile**（§7）。

## 18. ★★ 方法论更正：那 10.881 ms 不是「host 计算」，是「host 等 + host 下发」

**批评成立。** 我先前把 stream 47 的**下发空隙**直接当成 host 的**计算时间**，
这两者不是一回事。补做了两个此前没碰过的 host 侧数据源来分辨：

| 数据源 | 内容 |
|---|---|
| `mindstudio_profiler_output/api_statistic_*.csv` | msprof 在 **host 侧**采集的每个 API 调用的墙钟耗时 |
| `FRAMEWORK/torch.op_range` | **host 侧 torch 算子时间线**（二进制，格式见 §18.3） |

### 18.1 host 侧 API 账（`lineA/tools/host_account.py`）【实测】

按 400 个 decode 步摊：

| 类别 | ms/step | 次/step |
|---|---:|---:|
| **① 阻塞等待**（`aclrtSynchronizeEvent` + `aclrtSynchronizeStreamWithTimeout`） | **22.13** | 4.4 |
| ② 内核下发（`aclrtLaunchKernelWithHostArgs` 2.06 + `node launch` 2.70，两者重叠） | 4.77 | 868（重复计） |
| **③ 包装/拷贝等非阻塞 host 工作**（`aclnn*` 含 `GetWorkspaceSize`、`aclrtMemcpy*`、event） | **6.47** | 1207 |
| **acl 层非阻塞合计**（= ②中的 acl 部分 + ③） | **8.43** | ~1633 |

③ 的细分（top）：`aclnnInplaceCopy` 0.559 / `aclnnDivMods` 0.489 /
`aclnnInplaceCopyGetWorkspaceSize` 0.387 / `aclrtMemcpyAsync` 0.385 /
`aclnnInplaceFillScalar` 0.349 / `aclnnRemainderTensorScalar` 0.345 /
`aclnnSWhere` 0.307 / `aclmdlRIExecuteAsync` 0.202（**图重放只要 0.2 ms/步**）/
`aclnnMuls` 0.177 / `aclnnGeScalar` 0.169 / `aclrtGetStreamAttribute` 0.167。

**关键结构性事实**：acl 层非阻塞调用 **~1633 次/步**，而 decode 步里
**不在图内的 eager 算子只有 ~434 个/步**（`aclrtLaunchKernelWithHostArgs` 计数）
⇒ **每个 eager 算子要付 ~3.7 次 host API 调用**。
而这个数正好等于 §13 窗口里的 425 条算子 ⇒ **窗口里那些算子就是 eager 的**。

### 18.2 host 侧 torch 算子账（`lineA/tools/oprange.py`）【实测】

`torch.op_range` 覆盖 host 时间 **25.95 s**（与 device 侧窗口 25.95 s **完全一致**）；
区间并集 **19.22 s ⇒ 覆盖率 74.1%**。按 400 步摊：

| host 算子 | 条/step | ms/step | 次均 |
|---|---:|---:|---:|
| **`Event::synchronize`** | 1.0 | **20.64** | **20.6 ms** |
| `vllm::dsa_v41_forward`（range，含子节点） | 0.4 | 18.89 | 47.2 ms |
| `aten::copy_` | 89.9 | 1.62 | 18.0 µs |
| **`aten::item`** | **65.5** | 1.11 | 17.0 µs |
| `aten::_local_scalar_dense` | 65.5 | 1.07 | 16.3 µs |
| `aten::where` | 65.1 | 0.97 | 14.8 µs |
| `empty_tensor` | 341.4 | 0.93 | 2.7 µs |
| `aten::slice` | 254.3 | 0.87 | 3.4 µs |
| `aten::div` | 29.8 | 0.47 | 15.9 µs |
| `aten::remainder` | 24.2 | 0.33 | 13.8 µs |

**★ 交叉验证（两个独立文件）**：

```
torch.op_range  Event::synchronize   20.64 ms/step  (2064.15 单位 ×10ns)
api_statistic   aclrtSynchronizeEvent 20.61 ms/step  (20608.93 µs)
                                      ↑ 差 0.15%
```

⇒ **host 每步有 20.6 ms 在纯等设备**，这是 blocking，不是算力瓶颈。

### 18.3 `torch.op_range` 的二进制格式（逆出来的，供复用）

```
记录 = type(u16) + len(u32) + payload[len]
  type == 2 : payload = ascii 名字
  type == 1 : payload[0:8]  = ts0 (u64)
              payload[8:16] = ts1 (u64)
              payload[16:58]= 固定头（含 -1、pid×3）
              payload[58:]  = 内嵌的 type2 名字记录
时间戳单位 = **10 ns tick**（不是 ns）
```

### 18.4 更正后的结论

| 量 | 值 | 性质 |
|---|---:|---|
| device 空闲（§13） | 13.94 ms/step | 待解释 |
| host 阻塞等待 | **20.61 ms/step** | **等**（说明设备在跑） |
| host acl 非阻塞工作 | **8.43 ms/step** | **算**（下发 + 包装） |
| host torch op 里除 synchronize 外的实际工作 | ~1–2 ms/step | 算（很小） |

⇒ **那 10.881 ms 的空隙不是 Python CPU 时间**；量级上能对上的是
**acl 层非阻塞的 8.43 ms/step**（其中 launch 只占 2.06）。

⇒ **§15.4(d) 补丁的预算要下调**：省 6 kernel/次调用 × ~13 次/步 = 78 个 eager 算子，
每个约 19.4 µs（8.43 ms ÷ 434）⇒ **约 1.5 ms/step**（先前写的 2.3 偏高）。【推断】

> **★ 2026-10-01 晚：上述「每个约 19.4 µs」已被撤回。** 见 §19。
> 该数字把 acl 非阻塞总时间**平摊**到全部 eager 算子上，而实测显示 host 时间
> 并**不均匀**分布。撤回后改用 §19 的实测口径。

## 19. ★★★ 用 host 侧自耗时重算：19.6 µs/算子不成立

### 19.1 先修一个方法学错误：必须掐掉 prefill

`torch.op_range` 里 **`vllm::dsa_v41_forward` 有一次 6717 ms**（首次请求的 TTFT）。
不掐掉它，按 400 步摊会得到 **48 ms/step 的自耗时 > 36 ms/step 的步长** ——
数学上不可能，正是这个越界信号暴露了污染。

**正确做法**：只用 `[第一个 Event::synchronize 起, 最后一个 Event::synchronize 止]`
之间的区间（= steady-state decode 循环）。本次保留 931369/954754 条事件，
覆盖 **15.2 s**，而 399 个 sync 窗口合计 14.35 s，自洽。

### 19.2 host 侧自耗时（`lineA/tools/host_self.py`，decode-only）【实测】

| 项 | ms/step |
|---|---:|
| host 自耗时合计 | 29.919 |
| ├ `Event::synchronize`（纯等待） | **20.693** |
| └ **非同步的 host 自耗时** | **9.226** |

**独立交叉验证**：`api_statistic` 的 acl 层非阻塞 = **8.43 ms/step**
（下发 2.06 + 包装/拷贝 6.47），与本表的 9.226 差 **9%**。
两个不同采集器、两套独立口径给出同一量级 ⇒ 这个数字可信。

### 19.3 ★ 位置分布：host 的活**不在**那个洞里

把每步归一化后按位置分桶（非同步自耗时）：

| 位置 | ms/step | 占比 |
|---|---:|---:|
| 0.0–0.1 | 1.330 | 14.4% |
| 0.1–0.2 | 2.484 | 26.9% |
| 0.2–0.3 | 1.965 | 21.3% |
| 0.3–0.4 | 2.021 | 21.9% |
| 0.4–0.5 | 0.467 | 5.1% |
| 0.5–0.6 | 0.184 | 2.0% |
| 0.6–0.7 | 0.182 | 2.0% |
| 0.7–0.8 | 0.183 | 2.0% |
| 0.8–0.9 | 0.235 | 2.5% |
| 0.9–1.0 | 0.177 | 1.9% |

```
前 40%  = 7.80 ms/step（84.5%）
窗口 0.56–0.92 = 0.715 ms/step（2.4%）
```

⇒ **§13 那个 58%–90% 的设备空洞里，host 只花了 0.715 ms/step 的活。**
⇒ **「Python 层是空洞主要来源」这一条撤回**；空洞不是"host 在忙"造成的。

### 19.4 撤回「每个 eager 算子 19.6 µs」，换成实测的 top 消费者

host 非同步自耗时的构成（不是 kernel launch，而是**张量层开销**）：

| host 算子 | 条/step | ms/step | 次均 |
|---|---:|---:|---:|
| **`aten::copy_`** | 88.9 | **1.315** | **14.78 µs** |
| **`empty_tensor`**（分配） | **330.8** | **0.874** | 2.64 |
| `aten::slice` | 252.5 | 0.495 | 1.96 |
| `aten::as_strided`（视图） | 306.4 | 0.438 | 1.43 |
| `aten::fill_` | 66.0 | 0.258 | 3.91 |
| `Event::record` | 6.1 | 0.250 | **41.23** |
| `aten::_local_scalar_dense`（=`.item()`） | 65.2 | 0.248 | 3.81 |
| `aten::where` | 65.1 | 0.243 | 3.73 |
| `aten::div` | 29.6 | 0.243 | 8.21 |
| `aten::remainder` | 24.1 | 0.193 | 8.02 |
| `aclnnInplaceCopy`/`aclnnInplaceFillScalar`/`aclnnDivMods` 等 aclnn 包装合计 | ~200 | ~0.8 | ~4 |

**⇒ 全部 `aclnn*` 包装加起来只有 ~0.8 ms/step；真正的大头是
`copy_` + `empty_tensor` + `slice` + `as_strided` ≈ 3.1 ms/step**，
即**每步创建/切片了几百个小张量**。

### 19.5 ★ `.item()` 的单卡判别实验（用户建议的那条）—— **成立**【实测】

`lineA/tools/itemcost.py` 在 chip4 单卡实测（µs/次，7 轮中位）：

| 臂 | µs/次 |
|---|---:|
| A. `torch.div(a1,b1).item()`（1 元素） | **53.01** |
| B. `torch.div(a2,b2)`（512 元素，不读回） | 10.93 |
| C. `torch.div(a1,b1)`（1 元素，**不读回**） | 11.10 |
| E. 纯 `.item()`（标量已算好） | **17.48** |
| F. `a2.sum().item()`（512 元素） | 57.45 |

**⇒ `.item()` 的代价 = A − C = 41.91 µs**
（一致性检查：排队 50 个异步算子后再 `.item()`，增量 34.3 µs；
在长 kernel 后 `.item()`，增量 43.1 µs —— 三个口径同量级。）

**⇒ 65.2 次/步 × ~40 µs = 2.6 ms/step**【实测单卡 + 计数】

### 19.6 但 `.item()` 的**来源**仍是【未确认】

我在 `dsa_v41.py` 的热路径上**没有找到** `.item()`：该文件里所有 `.item()` 都在
诊断开关（`V41-DCP-*` 探针）里，而这些开关在本次 run 中都是关的。

⇒ 「每个 1 元素标量算子都伴随一次 `.item()`」这个因果链**只证明了计数吻合**
（`aten::div` 11752 ↔ `aclnnDivMods` 11752 等三组逐条相等），
**没有证明调用关系** —— 我那次"父节点"分析用的是**时间包含**，不是调用栈，
`aten::div` 里"包含"`aten::item` 很可能是时间重叠造成的假父子。

**要钉死来源，下一步需要**：`torch.profiler` 带 `with_stack=True` 跑一小段
decode，直接看 `.item()` 的 Python 调用栈；或按 §18.6 的 `perf_counter` 法。

### 19.7 修正后的账

| 量 | 值 | 口径 |
|---|---:|---|
| device 空闲（§13） | 13.94 ms/step | profiled |
| host 阻塞等待 | 20.69 ms/step | 纯等（说明设备在跑） |
| **host 非同步自耗时** | **9.226 ms/step** | 其中 84% 在步的前 40% |
| ├ 张量层开销（copy_/empty/slice/as_strided） | **3.12** | |
| ├ `.item()`（65 次 × ~40 µs） | **2.6** | 与上一行不重叠（`.item()` 的**等待**不在自耗时里） |
| └ aclnn 包装 + 其它 | ~3.5 | |

**⇒ 结论：device 那个 13.6 ms 的洞，host 侧只能解释一小部分。**
把两者并列看：host 在 42.5%–100% 期间**阻塞在 `Event::synchronize`**，
而设备在 58%–90% 期间**空闲** —— 这两个事实同时成立是矛盾的，
除非 `Event::synchronize` 等的**不是**这段设备工作（例如它在等下一轮的
输入就绪、或等一个跨 rank 的汇合）。**这是目前最需要解释的一个矛盾，
标【未确认】，建议作为下一轮的头号问题。**

### 18.5 意外发现：65 个「标量读回」/步【未确认，但计数精确吻合】

`aten::item` 的**立即父节点**计数与 api_statistic 的算子计数**逐条相等**：

| `aten::item` 的父节点 | 条数 | 对应的 `aclnn*` 调用数 | 是否相等 |
|---|---:|---:|---|
| `aten::div` | **11752** | `aclnnDivMods` **11752** | ✅ |
| `aten::mul` | **8076** | `aclnnMuls` **8076** | ✅ |
| `aten::floor_divide` | **2076** | `aclnnFloorDivides` **2076** | ✅ |

⇒ **每一个 1 元素标量算子都伴随一次 `aten::item`**（65.5 次/步）。
若这确实是 device→host 读回，则**每步有 65 个同步点**，
正是 10–70 µs 长空隙的形态。【未确认：需要 `time.perf_counter()` 或
一次最小实验（把该标量改成 2 元素张量看 `aten::item` 是否消失）来证实】

### 18.6 建议的确认实验（便宜，不改语义）

1. 在 `build()` 的复制态分支插 `time.perf_counter()` 累加器，
   跑 128 步打印合计 —— 直接给出**纯 Python 时间**（预计 ~1–2 ms/step）。
2. 把 `within`/`face_offset` 那段的标量改成 **≥2 元素张量**（或直接调 `.item()` 之外的路径），
   看 `aten::item` 计数是否归零 —— 若是，则 §18.5 成立。
