# N=8 多流 decode 超线性开销深挖（2026-10-04）

**执行**：`/root/tiny_fusion_ops` 子代理　**机器**：a3-21（**纯读 profile + 代码**；未启停任何容器/服务，未碰 19210）
**数据**：`~/cedpd-repo/results/k6full_1004_100156/prof/`
`dp0_pp0_tp0_dcp0_ep0_rank0_1435_20261004042337710_ascend_pt`（记作 **N1 窗口**）
`dp0_pp0_tp0_dcp0_ep0_rank0_1435_20261004043514275_ascend_pt`（记作 **N8 窗口**）
**标注**：【实测】有原始数字；【推断】由实测推导；【未确认】缺证据。

---

## 0. 相位预算 —— N=8 的产能到底丢在哪（追加，置顶）

**口径**：profile 窗口 = **04:35:14.3 → 04:35:33.5（19.2 s）**，
由 `serve.log:29237 「Starting profiler...」` 与第一波特请求
（`POST /tokenize` ×8，紧跟在 `Profiler started` 之后）对齐 ⇒ **这 19.2 s 恰好覆盖 8 流 bench 的整轮**
（含 ramp 与 tail）。token 数用 `Avg generation throughput` 的 10 s 窗口（原始日志行）做锚。

### 0.1 相位表【实测】（并发流数由 GMM1 首维严格反推：**verify 首维 = 36×流数**、**draft 首维 = 15×流数**；
单流 verify=36 / draft=15 ✓，8 流 verify=288 / draft=120 ✓）

| 相位 | 墙钟 | 占比 | 流数 | 步数 | 估算产出 token | 该段产能 |
|---|---:|---:|---:|---:|---:|---:|
| **ramp**：prefill-only 步 + 单流 decode | 0 → 7.60 s | **39.6%** | 1 | ~170 | ~130 | ~17 tok/s |
| **full**：8 流稳态 | 7.60 → 13.50 s | **30.7%** | 8 | 133 | ~2730 | ~471 tok/s |
| **tail-A**：流逐个收尾 | 13.50 → 15.40 s | 9.9% | 7→5 | 43 | ~550 | ~289 tok/s |
| **tail-B**：剩 2 流 | 15.40 → 19.02 s | 18.9% | 2 | 81 | ~260 | ~72 tok/s |
| final：剩 1 流 | 19.02 → 19.20 s | 1.0% | 1 | 4 | ~5 | — |
| **合计** | **19.2 s** | 100% | — | ~431 | **~3675**（理论 4096，差 10%，来自 A 的相位差异） | **191 tok/s** |

**原始 token 锚（`Avg generation throughput`，每 10 s 窗）【实测】**：

| 窗口 | prompt tok/s | gen tok/s | 该窗 token | 覆盖的相位 |
|---|---:|---:|---:|---|
| 04:34:5x–04:35:11 | — | — | — | bench 前 |
| 04:35:01–11 | 51.2 | 16.0 | 160 | bench 前/刚起 |
| 04:35:11–21 | 537.6 | 19.1 | 191 | **ramp（prefill 8.2k prompt + 单流）** |
| 04:35:21–31 | 409.5 | **335.5** | **3355** | full(5.8s)+tailA(1.9s)+tailB 前 1.2s |
| 04:35:31–41 | 0.0 | 80.6 | 806 | tailB 余段(2.5s)+收尾 |
| 04:35:41–51 | 0.0 | 0.0 | 0 | 结束 |

⇒ 整轮 ≈ 4096 token（8×512）在 **19.2 s** 内产出 = **213 tok/s（窗口平均）**；
bench 自报 311 tok/s 是其 `decode_wall=13.15 s` 口径（不含 ramp 的 prefill 段）。

### 0.2 prefill-only 步（admission gate）【实测·日志】

```
04:35:06  cumulative_prefill_steps=950
04:35:16/17/18/19/19/20/21/21   prefill-only episode ended (steps=1) ×8
04:35:22  prefill-only step #960  total_tokens=256  deferred_decode_reqs=1
04:35:24  prefill-only episode ended: steps=8 deferred_decode_reqs=28
          cumulative_prefill_steps=966
```

* **窗口内 prefill-only 步 = 966 − 950 = 16 步**，墙钟跨 **04:35:16 → 04:35:24 = 8.0 s**（与 ramp 的 7.6 s 吻合）。
* 这 16 步搬运的 prompt 量：`GMM1 b1=6144`（=6×1024 tok）280 个事件 = **7 个 1024-token chunk**，
  `GMM1 b1=1536`（=6×256 tok）360 个事件 = **9 个 256-token chunk** ⇒ **7×1024 + 9×256 = 9472 prompt token**。
* 关键：**gate 把每个 prefill 变成"独占步"，其间 decode 被推迟**（`deferred_decode_reqs` 累计到 28）。
  这 8 s 里只有 **1 条流**在 decode（GMM1 b1=36 = 单流），其余 7 条流在等自己的 prefill。

### 0.3 产能上限与缺口分解【推断，A=2.74 / 44.26 ms/步】

* **上限**（19.2 s 全程 8 流、44.26 ms/步、A=2.74）：`19.2/0.04426 = 434 步 × 8 × 2.74 = **9513 token** ⇒ **495 tok/s**`。
* **实测** ≈ 4096 token ⇒ **213–321 tok/s**（取决于是窗口平均还是 decode_wall 口径）。
* **缺口分解**（按"该相位若满 8 流应产出"vs"实际产出"）：

| 相位 | 应产出 | 实际 | 缺口 | 占总缺口 |
|---|---:|---:|---:|---:|
| **ramp** | 3764 | ~130 | **−3634** | **63%** |
| tail-B（2 流） | 1783 | ~260 | −1523 | 26% |
| tail-A（5–7 流） | 990 | ~550 | −440 | 8% |
| full（8 流） | 2872 | ~2730 | −142 | 2% |

⇒ **full 相位本身已经跑到产能的 95%；产能损失 63% 在 ramp、34% 在 tail、只有 2% 在算子。**

### 0.4 该压 ramp 还是压算子？（必答）

**先压 ramp，而且 tail 主要是 ramp 的次生现象。**
【推断】依据：8 条流的输出长度相同（512），却出现 13.5→19.0 s 的 5.5 s 拖尾
⇒ 说明它们**不是同时开始 decode 的**（ramp 期间只有第 1 条流在跑，其余 7 条被 prefill 挡在后面），
收尾自然被拉长。把 ramp 从 8 s 压到 ~1 s，理论上：

```
整轮 ≈ 1s(prefill 与 decode 重叠) + 187 步/流 × 44.26ms ≈ 1.0 + 8.3 = 9.3 s
⇒ 4096 / 9.3 ≈ 440 tok/s（≈ 当前 321 的 1.37×，≈ 窗口平均 213 的 2.07×）
```

对比：即便把**算子**全部优化 10%（44.26 → 39.8 ms/步），稳态上限只从 495 → 550 tok/s，
而在本窗口里只值 `(5.85/0.04426 − 5.85/0.0398) × 8 × 2.74 ≈ 420 token ≈ +23 tok/s` —— 远小于 ramp 的 +120 tok/s。

**优先级（按本窗口的收益量级）**：
1. **让 prefill 与 decode 重叠**（取消 prefill-only 独占步 / 打开 chunked prefill 混跑）→ 最大头；
2. **缩短 ramp**（prefill 吞吐 / 8 条流并行提交 / 前缀命中）；
3. **收窄 tail**（同批请求对齐结束、动态调 `SP_TOKENS`/批量重排）；
4. 最后才是算子级优化（本窗口只值 ~2%）。

---

## 1. 步时口径更正 + 结论速览

| 项 | N=1（单流） | N=8（8 流，T=48） | 比值 |
|---|---:|---:|---:|
| **真实步周期（设备侧）** | **27.90 ms**（625 步 / 17.44 s） | **44.26 ms**（122 步 / 5.40 s，T=48 纯相位） | **1.59×** |
| 每步 GMM1 | 43.2 | 43.1 | 1.00× |
| 每步 HcPre | 86.4 | 86.2 | 1.00× |
| 每步 device busy（去双记账） | 27.15 ms | 45.60 ms | 1.68× |

**三条最重要的更正/发现（都是【实测】）**：

1. **一步不是 40 个 GMM1，而是 43 个**：40 个是主 verify forward（T=48），
   外加 **3 个 DSpark draft 层**（T=40，正好对应 `dspark_target_layer_ids=[37,38,39]`）。
   `452 步 × 43 = 19436 = 实际 GMM1 总数`（精确闭合）；N=1 侧同理 `625 × 43 = 26875 ≈ 27004`。
   ⇒ 用「40/步」切会**系统性少算 7%**，并在相位交替处错位。
2. **N1/N8 两个 profile 窗口都不是单一并发相位**：N8 窗口 19.2 s 里先后出现
   T=6 → T=48 → T=42/36/30 → T=12，还夹着 256/1024/1536/6144 的 prefill chunk。
   「40/步切出 39.5 ms」正是**跨相位平均**的产物，不是 N=8 稳态。
3. **allreduce 双记账已定量确认**：同一 allreduce 事件在 `op_summary` 里各记一次，
   `Op Name` 分别是 `hcom_allReduce__503_NN_32`（Stream ID 空）与 `AivKernel`（stream 113），
   **(start, duration) 完全相同**。N8 的 T=48 窗口实测：sum=938.4 ms、union=469.2 ms
   ⇒ **恰好 50%**，即 sum 必须除以 2（`hcom_alltoallv_` 同样 50%）。

**因此：报告中"N=8 = 69.1 ms/step / 2.64×"不是设备稳态**，而是把
「整轮 total_decode(311–321 tok/s) ÷ 8 流 ÷ 平均接受长度」当成每步耗时得到的
**整轮平均**（含 ramp-up、prefill、并发不满的时段）。
`prof_n8.json` 自身的 per_stream_med=61.98 tok/s、accept_len=3.20 ⇒ 单流步时 **51.6 ms**；
设备侧 T=48 纯相位是 **44.26 ms**。
【实测】8 流稳态的真实代价是 **1.59× 步时换 8× token**，不是 2.64×；
【推断】真正压住聚合吞吐的是**流之间没有满叠加**（8 条流各自 decode 8.26 s，
而聚合窗口 13.15 s ⇒ 叠加度 ≈ 63%）。

---

## 2. 每步真实算子计数表（必答 1）

### 2.1 判定方法（不猜）

1. **图实例识别**：用 `GroupedMatmulSwigluQuantV2` 的 `Input Shapes` 第一维
   （= MoE 展开后 token 数）反推 batch：实测严格满足 **GMM1 第一维 = 6×T**（多数层）
   或 **3×T**（少数层），T = `HcPre` 的 batch = 6 × 并发流数（含 1 个 bonus token）。
2. **步边界**：在**单一 batch** 的 GMM1 序列里找 >5 ms 的 gap。实测 gap 间隔分布极窄
   （N=8：p25=44.0 / 中位 44.2 / p75=44.5 ms），说明这就是**步周期**。
3. **计数闭合校验**：`步数 × 每步 GMM1 = GMM1 总数`，精确吻合（见 §0）。

### 2.2 T=48 稳态（8 流）的一步结构【实测】

```
[verify forward] 40 层 × (2×HcPre + 1×GMM1 + 1×attn + 2×allreduce + …)   ≈ 31.7 ms
   └ 层间隔中位 0.793 ms ⇒ 40 层 = 31.7 ms
[draft head]      3 层 × (2×HcPre + 1×GMM1)           (T=40)              ≈  2.3 ms
[相位尾]          metadata/sampling/AI_CPU/搬运                          ≈ 10.3 ms
------------------------------------------------------------------------------
合计                                                                      44.26 ms
```

**每步计数表（N1 vs N8，按步归一，已去 allreduce 双记账）**

| 算子 | cnt/步 N1 | cnt/步 N8 | 说明 |
|---|---:|---:|---|
| GroupedMatmulSwigluQuantV2 | 43.21 | 43.08 | 40 verify + 3 draft |
| GroupedMatmul | 43.21 | 43.08 | 同上 |
| HcPre / HcPost | 86.41 | 86.16 | 2/层 × 43 层 |
| MoeGatingTopKHash / MoeInitRoutingV3 / MoeTokenUnpermute / RmsNormCast | 43.21 | 43.08 | 1/层 |
| SparseFlashMla | 40.19 | 40.07 | 40 个稀疏注意力层 |
| QuantBatchMatmulV3 | 227.08 | 226.41 | ~5.3/层 |
| RmsNorm | 140.67 | 140.23 | ~3.3/层 |
| hcom_allReduce_ | 89.42 | 89.16 | 2/层（TP） |
| MatMulV2 | 111.68 | 111.18 | |
| DynamicQuant | 141.68 | 141.25 | |
| InplacePartialRotaryMul | 145.70 | 145.25 | |
| ScatterNdUpdateSk | 58.28 | 58.07 | |

⇒ **算子计数完全不随并发变化（±0.3%），N=8 的额外开销 100% 来自"每个算子变慢"，不是"算子变多"。**
这是本次最重要的单条结论。

### 2.3 白拿的两个"坑"（供后来者）

* **`apex` 相位**：窗口里 GMM1 的 batch 首维取值全集是
  {15,30,36,45,60,72,75,90,105,108,120,144,180,216,240,252,288,1536,6144}。
  凡并发 sweep 的 profile，**必须先按 batch 分段**，否则步周期被平均成无意义的值。
* **prefill 混入**：`1536 / 6144` 就是 prefill chunk（BAT_TOKENS=8192）。

---

## 3. N=8 稳态独占 top-15（必答 2）

`scripts/excl_multi.py <dir> 0.7 15`（该脚本按 (start,end) 去重，**已正确处理 allreduce 双记账**；
它保留的那一份名为 `AivKernel`）。窗口 = 采集窗口尾部 70%。

**N=8 窗口**（uniq=1209589，steady=845357，U=11984.4 ms）：

| # | op | excl_ms | excl% |
|---|---|---:|---:|
| 1 | AivKernel（=allreduce 的有效副本） | 1900.2 | 15.86% |
| 2 | GroupedMatmulSwigluQuantV2 | 1661.6 | 13.86% |
| 3 | MatMulV2 | 1025.9 | 8.56% |
| 4 | HcPre | 1016.1 | 8.48% |
| 5 | GroupedMatmul | 672.6 | 5.61% |
| 6 | SparseFlashMla | 615.6 | 5.14% |
| 7 | HcPost | 383.5 | 3.20% |
| 8 | QuantBatchMatmulV3 | 302.8 | 2.53% |
| 9 | ScatterNdUpdateSk | 282.5 | 2.36% |
| 10 | RmsNorm | 267.4 | 2.23% |
| 11 | InplacePartialRotaryMul | 237.9 | 1.99% |
| 12 | MoeInitRoutingV3 | 200.0 | 1.67% |
| 13 | SparseFlashMlaMetadata | 199.6 | 1.67% |
| 14 | MoeInitRoutingV3 | — | — |
| 15 | MatMulV3 | 169.5 | 1.41% |

TOP-15 excl 合计 9031.5 ms / 75.36%。

**N=1 窗口**（U=11074.9 ms）对照：GMM1 1409.7 / MatMulV2 1284.3 / HcPre 1206.9 /
AivKernel 1152.7 / QBMV3 466.5 / GroupedMatmul 636.8 / SparseFlashMla 597.0 / RmsNorm 187.7 …
（注意两窗口 U 不同，**不可直接相减**；per-step 口径见 §3。）

---

## 4. Δ 表（必答 3）：每步 µs，N1 → N8

（N1：全窗口 625 步；N8：T=48 纯相位 [8.0,13.4]s，122 步；均已去 allreduce 双记账）

| op | µs/步 N1 | µs/步 N8 | **Δµs** | Δ% | 单次 µs N1→N8 | 单次倍数 |
|---|---:|---:|---:|---:|---:|---:|
| GroupedMatmulSwigluQuantV2 | 3155.3 | 6636.7 | **+3481** | +110% | 73.0 → 154.1 | 2.11× |
| GroupedMatmul | 1686.6 | 3222.1 | **+1536** | +91% | 39.0 → 74.8 | 1.92× |
| RmsNorm | 729.2 | 2157.8 | **+1429** | +196% | 5.2 → 15.4 | 2.97× |
| QuantBatchMatmulV3 | 2403.8 | 3720.2 | **+1316** | +55% | 10.6 → 16.4 | 1.55× |
| hcom_allReduce_ | 2885.9 | 3850.1 | **+964** | +33% | 32.3 → 43.2 | 1.34× |
| HcPost | 631.3 | 1593.3 | **+962** | +152% | 7.3 → 18.5 | 2.53× |
| ScatterNdUpdateSk | 324.2 | 1247.9 | **+924** | +285% | 5.6 → 21.5 | **3.86×** |
| SparseFlashMla | 1354.3 | 2145.2 | **+791** | +58% | 33.7 → 53.5 | 1.59× |
| InplacePartialRotaryMul | 524.2 | 1314.7 | **+790** | +151% | 3.6 → 9.1 | 2.51× |
| HcPre | 2707.4 | 3340.9 | **+634** | +23% | 31.3 → 38.8 | 1.24× |
| MatMulV2 | 2879.0 | 3459.6 | +581 | +20% | 25.8 → 31.1 | 1.21× |
| allgatherAicpuKernel | 0.0 | 443.0 | +443 | 新增 | — | — |
| MoeGatingTopKHash | 186.7 | 554.7 | +368 | +197% | 4.3 → 12.9 | 2.98× |
| DynamicQuant | 575.6 | 915.6 | +340 | +59% | 4.1 → 6.5 | 1.60× |
| RmsNormCast | 157.5 | 486.9 | +329 | +209% | 3.6 → 11.3 | 3.10× |
| MoeInitRoutingV3 | 433.9 | 755.8 | +322 | +74% | 10.0 → 17.5 | 1.75× |
| **合计** | **27148.2** | **45602.7** | **+18454** | +68% | — | — |

**读法**：Δ 的 top-3（GMM1+GroupedMatmul+RmsNorm）就占 Δ 的 35%；
top-10 占 Δ 的 **76%**。
另外 Δ 表暴露出三类"**每步新增**"的项（不计入 N1 但 N8 有）：
`allgatherAicpuKernel`（+1/步，+443 µs）、`TensorMove`（+48.9/步，+265 µs）、
`ViewCopy`（+10.9/步，+185 µs）。

> ⚠️ 设备 busy 在 N8 口径下 = 45.60 ms > 步周期 44.26 ms（103%），
> 因为多 stream 并行时「各算子时长之和」会超过墙钟。**不能把"busy 和"当步时**。

---

## 5. 为什么随并发涨：固定成本 vs 随流增长（必答 4）

### 5.1 关键框架：每步成本 = 固定项 + 随流线性项【实测数据拟合】

设备 busy：单流 27.15 ms/步，8 流 45.60 ms/步。解 `F + V = 27.15`、`F + 8V = 45.60`：
**V ≈ 2.64 ms/流，F ≈ 24.5 ms/步**。
⇒ **每步约 24.5 ms（占 8 流步时的 55%）与批量无关**。这直接回答"该压 ramp 还是压算子"：
固定项落在关键路径上，**省 1 ms 就是 N=8 步时省 1 ms**；而 ramp/tail 只影响窗口吞吐。
两者都值钱，但压的方向不同：ramp 抬吞吐上限，固定项抬 steady 上限。

### 5.2 top-5 Δ 的机理【推断】+ 代码指针

| 算子 | Δµs/步 | 单次 N1→N8 | 读法 | 机理假设 | 代码指针 |
|---|---:|---|---|---|---|
| GroupedMatmulSwigluQuantV2 | +3481 | 73.0 → 154.1 µs（2.11×） | 8× token 只花 2.11× 时间 ⇒ **权重流量主导**，多出的 token 几乎搭便车 | MoE 组 matmul 权重字节固定，M 维变大只增加 cube 占用不增加权重读 | `vllm_ascend/ops/fused_moe/token_dispatcher.py`（`V41_MOE_MASK_RANGE`）、grouped_matmul 自定义算子 |
| GroupedMatmul | +1536 | 39.0 → 74.8 µs（1.92×） | 同上（down proj） | 同上 | 同上 |
| RmsNorm | +1429 | 5.2 → 15.4 µs（2.96×） | 140 次/步，多数是 per-head 小张量 ⇒ **固定开销（下发/UB 配置）占大头** | 小张量 kernel 的固定成本不随行数变，8 流只把行数放大 | `vllm_ascend/ops/layernorm.py`、`models/deepseek_v41/model.py`（input_layernorm/rms_norm_cast） |
| QuantBatchMatmulV3 | +1316 | 10.6 → 16.4 µs（1.55×） | 227 次/步，小 GEMM 权重主导 | 权重字节不变，M 增大收益递减 | `csrc/gmm/*`、`patches/files/dsa_v1.py` |
| hcom_allReduce_ | +964 | 32.3 → 43.2 µs（1.34×） | 89 次/步（2/层）⇒ **延迟主导**（ring 起步+同步），消息变大只 +34% | TP allreduce 每次都要跨 8 die 建链，固定同步成本高 | `csrc/torch_binding.cpp` / HCCL 调用点 |
| ScatterNdUpdateSk | +924 | 5.6 → 21.5 µs（**3.86×**，最大单次倍数） | 唯一"单次涨幅接近 token 涨幅"的算子 ⇒ data-dependent 扫描 | MoE 路由写回按 token/专家数扫描，8 流时分散度上升 | `models/deepseek_v41/indexer.py`、moe 写回 |

补两条 AI_CPU（metadata）实测【实测】：

| 项 | N=1 | N=8（T=48） | Δ |
|---|---:|---:|---:|
| AI_CPU 总 | 7.03 次/步，**1147.3 µs/步** | 8.00 次/步，**1953.9 µs/步** | **+807 µs/步** |
| ├ SparseFlashMlaMetadata | 536.3 µs/步 | 719.6 µs/步 | +183 |
| ├ SparseAttnSharedkvMetadata | 323.9 µs/步 | 402.8 µs/步 | +79 |
| ├ QuantLightningIndexerV2Metadata | 287.1 µs/步 | 386.7 µs/步 | +100 |
| └ **allgatherAicpuKernel（新增）** | 0 | **444.9 µs/步**（1 次/步） | **+445** |

⇒ AI_CPU 合计 **~1.95 ms/步**，全部是 host 侧串行任务，是**纯固定成本**的典型候选（可尝试与计算重叠）。

## 6. 优化候选清单（必答 5）

按"本窗口收益量级"排序（口径：8 流、512 token/请求、A≈2.74）：

| # | 候选 | 预期收益 | 改动位置 | 风险 | 证据级别 |
|---|---|---|---|---|---|
| 1 | **prefill 与 decode 重叠**（取消 prefill-only 独占步，改 chunked prefill 与 decode 混批） | 窗口吞吐 **+100~130 tok/s**（ramp 8s 中大部分可回收）；steady ms/step 可能 +0~2ms | `patches/admission_gate.patch`、scheduler 侧 | 需重验长上下文正确性与 TP 同步 | 【推断】 |
| 2 | **缩短 ramp**（8 流并行提交 + 前缀命中 + 更大 prefill chunk） | 每省 1 s → **+45 tok/s**（窗口口径） | bench harness / prefix cache / BAT_TOKENS | 低 | 【推断】 |
| 3 | **收窄 tail**（同批请求对齐结束、按剩余 token 重排批） | 每省 1 s → **+45 tok/s** | 调度/批重排 | 中（可能影响 A） | 【推断】 |
| 4 | **HcPre+RMSNorm 融合**（另一条线） | **0.6~1.2 ms/步**（固定项，直接进 N=8 步时；×86 次/步） | `csrc/moe/hc_pre` + `model.py` | 数值尚未对齐 | 【实测·部分】 |
| 5 | **AI_CPU metadata 与计算重叠 / 合并 metadata 算子** | 最多 **~1.0~1.9 ms/步**（若完全隐藏） | 三个 metadata 算子 + allgatherAicpuKernel 的调度 | 中 | 【实测】+【推断】 |
| 6 | **GMM1/GroupedMatmul 的 M 维 padding / expert 分布优化** | 若 padding 膨胀 >10%，可回收 ~0.3~0.6 ms/步 | `token_dispatcher.py`（MOE_MASK_RANGE 已有）、grouped matmul tiling | 中 | 【未确认】 |
| 7 | **RmsNorm 小张量合并**（140 次/步 → 更少更大 kernel） | ~0.3~0.6 ms/步 | `ops/layernorm.py` / model.py | 中 | 【推断】 |

## 7. 必要但未完成的验证（诚实标注）

* N8 窗口的 T=48 相位只覆盖 8 流稳态的 **5.4 s**（122 步），
  而 bench 的 512-token 输出要跑 ~160 步；**相位早段上下文更短**，
  所以 44.26 ms 可能比「整轮 8 流平均」偏乐观。【推断】
* 【未确认】"69.1 ms/step"的原始出处（本次未在仓库里找到直接记录，
  `~/tmp/prof_n1.json` / `prof_n8.json` 只给出 total_decode 61.79 / 311.39）。
* 【未确认】`hcom_alltoallv_` 的 516 µs 大包（出现在相位边界）归属哪一段逻辑。
