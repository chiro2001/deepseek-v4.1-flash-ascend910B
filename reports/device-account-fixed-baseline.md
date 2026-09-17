# 修复后基线：32K decode 设备账（方法自洽版）

> 采集：`a21_fixed_0309` 会话（静态内核已启用），32K，1 发 192 token
> 步数来源：**同会话客户端实测 70 步**（`p42_t4_quote_32768_fixed_prof.jsonl`），
> 而不是任何外部假设 —— 这是对 `profiler-overhead-analysis.md` §4 那个方法论错误的修正。

---

## 1. 总量

| 项 | 总 ms | **每步 ms** | 占比 |
|---|---|---|---|
| busy（所有 task 区间并集） | 2165.6 | **30.94** | 83.0% |
| **FREE** | 445.0 | **6.36** | 17.0% |
| comm | 231.1 | **3.30** | 8.9% |
| compute | 1934.5 | **27.64** | 74.1% |
| **comm ∩ compute（重叠）** | **0.0** | **0.000** | **0.0% of comm** |
| comm 暴露（未重叠） | 231.1 | **3.30** | — |

折合设备步长 37.29 ms/step（采集态；同会话客户端墙钟 43.08 ms/step，差值 5.8 ms 是 profiler 在 host 侧的开销）。

**两条结构性事实（在修复后的配置上重新确认）**：

1. **通信与计算零重叠** —— 3.30 ms/step 的集合通信 100% 暴露在关键路径上。
2. FREE 占到 17%（采集态），即设备仍有可观空档。

---

## 2. 算子账（每步耗时 Top 16）

| ms/step | 次/步 | avg µs | 算子 |
|---|---|---|---|
| **9.411** | 40.0 | 235.3 | **DispatchFFNCombineW4A8**（MoE 融合） |
| **4.484** | **165.6** | 27.1 | **hcom_allReduce_**（TP 通信） |
| 2.384 | 80.0 | 29.8 | HcPre |
| 2.171 | 210.2 | 10.3 | QuantBatchMatmulV3 |
| 1.900 | 40.0 | 47.5 | TransposeBatchMatMul |
| 1.792 | 7.4 | 241.2 | QuantLightningIndexerV2 |
| 1.711 | 65.2 | 26.2 | MatMulV2 |
| 1.338 | 85.7 | 15.6 | hcom_allGather_ |
| 1.268 | 37.2 | 34.1 | SparseFlashMla |
| 0.768 | 130.2 | 5.9 | RmsNorm |
| 0.737 | 3.7 | 198.3 | hcom_alltoallv_ |
| 0.691 | 45.6 | 15.2 | MatMulV3 |
| 0.606 | 80.0 | 7.6 | HcPost |
| 0.535 | 134.8 | 4.0 | InplacePartialRotaryMul |
| 0.450 | 131.1 | 3.4 | DynamicQuant |
| 0.383 | 202.4 | 1.9 | Cast |

（注意：这些是**采集态**数值，绝对值偏高约 10–20%；比例关系可用。）

### 2.1 通信合计

| 算子 | ms/step | 次/步 |
|---|---|---|
| allReduce | 4.48 | 165.6 |
| allGather | 1.34 | 85.7 |
| alltoallv | 0.74 | 3.7 |
| **合计** | **6.56** | **255** |

其中 allReduce + allGather = 5.82 ms/step，占 comm 的 89%。按 40 层折算，
**每层约 4 次 allReduce + 2 次 allGather**（TP=8 的通信是主要成本）。

---

## 3. 对目标的含义（诚实评估）

目标：128K 单流 > 110 tok/s。

`tok/s = A × 1000 / ms_per_step`，128K 实测 **A = 2.763**：

| 假设 | 需要 ms/step | 与当前（39.1）的差距 |
|---|---|---|
| A = 2.763（当前） | **≤ 25.1** | **−36%** |
| A = 3.49（历史最优样本） | ≤ 31.7 | −19% |
| A = 4.4（S=7 的理论上限附近） | ≤ 40.0 | 已达 |

而设备 busy 本身就有 **30.9 ms/step（采集态）**。即使把 FREE 清到 0、host 完全隐形，
也到不了 25。⇒ **必须同时做两件事**：

1. **降低设备 busy**（当前 30.9）：最大单项是 MoE 融合 9.4、TP 通信 5.8、HcPre/HcPost 3.0；
2. **提高接受率 A**（当前 2.76，逐位置上界约 2.87）：空间有限，除非改 draft 质量。

**结论：以当前算子组合与 A，128K > 110 tok/s 不可达。** 现实可达的中间目标见 §4。

---

## 4. 现实可达的中间目标与下一步

| 目标 | 手段 | 预估 |
|---|---|---|
| 128K ≥ 80 tok/s | 清掉 host 侧剩余暴露（D2H 之后 116% 暴露那段） | 39.1 → ~35 |
| 128K ≥ 90 tok/s | 上面 + 通信与计算重叠（3.3 ms/step 暴露） | → ~31.5 |
| 128K ≥ 100 tok/s | 上面 + HcPre/HcPost 融合（3.0 ms/step） | → ~28.5 |

按性价比排序的**下一步候选**：

1. **通信/计算重叠**（3.30 ms/step 全暴露）—— 已知 `multistream_overlap_shared_expert` 与
   `enable_fused_mc2` 互斥；需找第三条路（图内多流 / 把 allreduce 挪进融合算子）。
2. **TP allReduce 精简** —— 165.6 次/步、4.48 ms/step。查 `finegrained_tp_config`
   能否减少 o_proj / MoE 的 allreduce 次数。
3. **HcPre + HcPost 融合** —— 2.99 ms/step，glm5next 有可参考实现；**可通过热更新迭代**。
4. **DispatchFFNCombineW4A8（9.41 ms/step）** —— 已是融合形态，只能靠算法/EPLB。
5. **Free 6.36 ms/step（采集态）** —— host 侧已被多轮收窄，剩余与 D2H 后的暴露段重叠。

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 采集会话 | `logs/perf/a21_fixed_0309_serve.log`（静态内核警告 0，编译缓存命中） |
| 客户端测量（步数来源） | `logs/perf/a21/p42_t4_quote_32768_fixed_prof.jsonl` |
| 设备 profile | `logs/prof_fixed/`、`/tmp/op_summary_fixed.csv`、`/tmp/op_stat_fixed.csv` |
| 分析工具 | `scripts/dev_account.py`（本次新增，步数由同会话客户端实测给出） |
| 修复后基准 | `logs/perf/a21/measure_fixed.log`（8K 34.27 / 32K 35.85）、`measure_lws.log`（34.20 / 35.18）、`measure_lws128.log`（128K 39.10） |
| 方法论修正 | `a21_reports/profiler-overhead-analysis.md` |
| 静默降级修复 | `a21_reports/static-kernel-silent-disable-fix.md` |
