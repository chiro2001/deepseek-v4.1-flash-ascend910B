# Prefill 瓶颈：**allreduce 占设备时间 42%**（2026-10-04 实测）

> 用户目标里的第三个弱点："prefill 速度（包含 pd 混布时候的速度）"。
> 本文给出**我们自己的 TP8 上的首批 prefill 测量**与**一次 profile 的归因**，
> 以及正在验证的第一条修法。

---

## 0. 基线（先前没有这个数）

方法：`~/tmp/prefill_bench.py`，单请求、`max_tokens=1`，**取冷启 rep0**
（rep1 会被前缀缓存污染，实测 138K/283K tok/s，不可用）。

| 上下文 | TTFT | **prefill** | 备注 |
|---:|---:|---:|---|
| 4K | 0.58 s | 7,088 tok/s | 含首次预热 |
| **32K** | **4.03 s** | **8,136 tok/s** | 4 个 chunk |
| **128K** | **15.71 s** | **8,341 tok/s** | 16 个 chunk |

**与历史基线对照**：

| 来源 | 32K | 128K |
|---|---:|---:|
| `docs/prefill-memory-headroom.md`（2026-09-19，发布默认） | 4.218 s（7,768 tok/s） | 18.165 s（7,215 tok/s） |
| **本次（2026-10-04）** | **4.03 s（8,136）** | **15.71 s（8,341）** |
| **改善** | **+4.7%** | **+15.6%** |

⇒ 累计优化已经把 prefill 推高了 5–16%（不是退化）。
**但 CED-PD 形态（P 只跑 20 层）是 12,536 / 13,676 tok/s ⇒ 仍差 1.5–1.65×。**

---

## 1. ★ Profile 归因：一个算子吃掉 42%

方法：`~/tmp/prof_prefill2.py`（**预热用前段 8K、profile 用后段不同区间**，规避前缀缓存），
32,776-token prompt，rank0 `op_summary`。

| 算子 | 调用数 | 总时间 | **次均** | 占比 |
|---|---:|---:|---:|---:|
| **`allreduceAicpuKernel`** | **324** | **3.42 s** | **10,553 µs** | **42%** |
| `SparseFlashMla` | 200 | 0.94 s | 4,696 µs | 12% |
| `QuantLightningIndexerV2` | 40 | 0.57 s | 14,286 µs | 7% |
| `HcPre` | 430 | 0.51 s | 1,176 µs | 6% |
| `ScatterNdUpdateSk` | 290 | 0.30 s | 1,026 µs | 4% |
| `HcPost` | 430 | 0.29 s | 665 µs | 4% |
| `GroupedMatmulSwigluQuantV2`（MoE gmm1） | 215 | 0.18 s | 839 µs | 2% |
| 其余 | — | ~1.8 s | — | 22% |
| **设备合计** | — | **8.12 s** | — | 100% |

（设备合计 8.12 s > 墙钟 4.61 s ⇒ 多流并行确实在起作用，重叠约 43%。）

### 1.1 同一个 allreduce，prefill 比 decode 慢 **52×**

| 场景 | 次均 | 说明 |
|---|---:|---|
| decode（N=8, K=7） | **199 µs** | T=64 行 ⇒ payload 650 KB，**远低于 8 MiB** |
| **prefill（T=8000）** | **10,553 µs** | payload = 8000×5120×2 = **82 MB**，**远超 8 MiB** |

⇒ 与我们在 2026-10-02 从 HCCL 源码里确认的门限一致
（`AIV_ALL_REDUCE_A3_GRAPH_ENTRY_SIZE = 4 MiB` / `..._ENTRY_SIZE = 1 MiB`）：
**payload 超过门限 ⇒ 掉进 AICPU-RPC 回落**（算子名直接就叫 `allreduceAicpuKernel`）。

**324 次调用 = 4 个 chunk × 80 次/chunk**（40 层 × 2：attention 出口 + MoE 出口），
与 decode 的结构完全一致 —— 只是 payload 大了 1000 倍。

---

## 2. 正在验证的修法：`MC2=1`（comm × matmul 融合）

**机理**：`enable_prefill_mc2` 打开后，MoE 的集合通信与 matmul 融合/重叠
（vllm-ascend 的 MC2 路径），把 allreduce 从关键路径上挪走。
**我们的配置此前是 `enable_prefill_mc2=false`**（`scripts/serve_v2.sh:90`，由 `MC2` env 控制）。

旁证：`docs/CED-PD-PROFILING-20260925.md` §5 已记录
「allreduce 是 AI_CPU kernel（`RunAicpuRpcSrvLaunchV2_allreduce`），
avg 11.45 ms、max 92.3 ms」，并指出 **`MC2=0 FUSED_MC2=0` 意味着它没有与 matmul 融合**。

**实验设计（单变量）**：

| 臂 | 配置 | 状态 |
|---|---|---|
| A | `BT_PERSIST=1 + SKIP_K0=1`（已有 bench） | 已完成 |
| **B** | **A + `MC2=1`** | **正在起服** |

判据（**两个维度都要看**，防"救 prefill 打坏 decode"）：
1. prefill：32K / 128K 冷启 TTFT 必须**变快**；
2. decode：N=1/8/16 的 `(ms/step, A, tok/s)` **不得回退**。

---

## 3. 若 MC2 不够，备选（按预期收益排序）

| # | 手段 | 机理 | 代价 |
|---|---|---|---|
| 2 | **`MC2_HIER=1`**（`enable_mc2_hierarchy_comm`） | 分层集合通信，减少跨 rank 流量 | 一行配置 |
| 3 | **分块 allreduce**（每块 < 8 MiB，走 AIV） | 微观基准显示 AIV 3.75 MiB 只要 **77.6 µs**；82 MB 切成 3.75 MiB × 22 块 ≈ **1.7 ms** vs AICPU 的 10.5 ms | ⚠️ **我们在 DCP merge 上试过并失败**（34.6 → 77.6 ms/步），需先搞清当时为何失败再重试，不能盲目重走 |
| 4 | **CED-PD 形态**（P 只跑 20 层） | 架构性 1.5–1.65× | 要 **16 个 die**，而 dies 0,1 属别人 |
| 5 | 增大 `BAT_TOKENS`（8192 → 16384） | 减少 chunk 数 ⇒ 减少固定开销 | ⚠️ 但 32K 实测「每 chunk 时间几乎恒定」（1.135 s/chunk，4 个 chunk），说明**成本随 token 线性**，提 BAT 大概率无收益；且历史记录 16384 会 OOM |

---

## 4. 诚实边界

1. 上面的 4K 那一行含首次预热，**不要**当稳态数；
2. prefill 的 profile 只有 **32K 一档**；128K/1M 是否同构**未验证**；
3. 「MC2 能省多少」目前是**机理推断**，没有实测数 —— §2 的实验就是为了补它；
4. CED 的 12,536/13,676 是**不同形态**（P 只跑 20 层），**不能**当作我们 TP8 的可达目标。
