# `allreduceAicpuKernel` 溯源：**prefill 专属**，与 decode 无关

> 2026-09-16 12:45 初版 / **13:00 按子代理溯源结论大幅更正**
> A3-node1 chips 8-15｜dummy 128K profile（`logs/prof_vmA`、`logs/prof_abA`）
> 详细报告：`A3-node2:~/handoff/reports/aicpu-allreduce-investigation.md`（子代理交付）

---

## 0. ⚠️ 三处更正（初版的三条推断都是错的）

| 我初版的说法 | 实际 | 证据 |
|---|---|---|
| 「是 decode 的，7.5 次/步」 | ❌ **全部在 prefill 块**，**116 个 decode step 里 115 个为 0** | 5184 次全部落在 2750–25606 ms 的 prefill 区间；64 个 `allgatherAicpuKernel` 把这段切成 64 个 chunk |
| 「7.5 次/步是真实节奏」 | ❌ **窗口平均假象**（42855.7÷5183=4.41 ms 只在 prefill 块内成立）；真实节奏是 **chunk 内 ~3.9 ms/次** | 同上 |
| 「busy > 墙钟 是因为 stream 10 混进来」 | ❌ **根因是 decode 窗口跨过了 prefill 块**；decode 段里 stream 10 恒为 0 | 剔除 prefill 后：vmA **33.30** / abA **34.09** ms/step，落在墙钟 33–35 ms 内 ⇒ 账目自洽（错误口径给出 222 / 76 ms/step） |

**⇒ 正确的做法是「先切段（剔除 prefill）再按 stream 分组」，切段优先。**

---

## 1. 精确公式（子代理给出，与两份 profile 都吻合）

```
allreduceAicpuKernel 次数 = (prompt_tokens / max_num_batched_tokens) × 81
81 = 40 层 × 2 个 RowParallel(wo_b / down_proj) + 1 个 lm_head
```

| | abA（32K prompt） | vmA（128K prompt） |
|---|---|---|
| `allreduceAicpuKernel` | 1296 | **5184** |
| `allgatherAicpuKernel` | 16 | **64** |
| 每 chunk 次数 | 81 | 81 |
| decode step 里出现 | **0 / 115** | **0 / 115** |

（32K/2048 = 16 chunk × 81 = 1296 ✓；128K/2048 = 64 chunk × 81 = 5184 ✓）

---

## 2. 机制（子代理从代码/库符号取得）

* 来源：**HCCL 的 AICPU unfold**（`libhccl*.so` 的 `HcclLaunchAicpuKernel` 与 `HCCL_OP_EXPANSION_MODE`）。
* **只走 eager 路径**：prefill 是 eager 执行 ⇒ 每次 RowParallel allreduce 都展开一个 AICPU kernel；
  **decode 走 aclgraph 重放（`OP State=static`）⇒ 完全没有它**。
* 同一层的两次 allreduce **背靠背下发**（相邻间隙 <50 µs 占 54%，40×64 组）；
  随后有 ~2–3 ms 空隙等下一层算完。
* **它量的是"等待"，不是"搬运"**：kernel p50 2.54 ms，而 `hcom` 真搬运只占 12%，
  起点晚 87%。
* **A3 上 `HCCL_OP_EXPANSION_MODE` 合法值只有 `AI_CPU | AIV`**，AIV 已开 ⇒ **没有开关可换**。
* ❌ 更正：`HCCL_OP_EXPANSION_MODE=HOST` 在 **A3 上非法**（历史文档里的 E3-B 那条不适用）；
  要做对照只能改 `AI_CPU`。

---

## 3. 唯一可用的用户侧杠杆（**只影响 TTFT/prefill，不影响 decode ms/step**）

```bash
--max-num-batched-tokens 2048 → 8192
```

前向 chunk 数 **64 → 16**，`allreduceAicpuKernel` **5184 → 1296（−75%）**。
⇒ **对 110 tok/s 的目标无直接贡献**（那是 decode 指标），
但**对长 prompt 的 TTFT 有价值**（128K prompt 的 TTFT 现在 23–30 s）。

**待评估的副作用**：`max_num_batched_tokens` 也决定 `_engram_max_tokens`（Engram 的 pad 容量）
和 decode 图的容量，改大可能影响显存与 capture 尺寸。**需要线 2 单独验证。**

---

## 4. 初版保留的有效部分

<details>
<summary>原始的统计数字（口径已更正，数字本身无误）</summary>

---

## 1. 事实（从 op_summary 直接统计，无推断）

| 项 | 值 |
|---|---|
| 事件数 | **5184** |
| 全部所在 stream | **10**（独立于主流 38/140） |
| Task Type | **AI_CPU** |
| OP State | `dynamic` |
| 首末时间跨度 | **22855.7 ms** |
| 时长总和 | **13800.1 ms** |
| **占用率** | **60.4%**（= 13800.1 / 22855.7） |
| 平均间隔 | **4.41 ms** ⇒ **7.5 次 / 33 ms 步** |

另一份较早的 profile（`abA`，窗口 6046 ms）里：

| 项 | 值 |
|---|---|
| 事件数 | 1296 |
| 中位时长 | **2.562 ms** |
| p90 / max | 3.474 / 14.549 ms |
| 同流邻居 | **1264/1296 的前一个事件就是它自己**（背靠背串行） |
| 该流上其它算子 | 仅 `allgatherAicpuKernel` 16 个（≈prefill 的 16 个 chunk） |

---

## 2. 为什么这值得追

1. **它是 profile 里按"时长总和"排第一的单项** —— 比任何 matmul 都大。
2. **它是 `AI_CPU` 类型**。cannbot 措施8 明确写：
   > 适用场景：AICPU算子，且对应有等价Aicore可以替换。
   > 预期收益：显著提升算子性能，对应算子性能提升~50%。
   > 约束：需 CANN 算子开发团队用 AscendC 重写；**非用户侧可配置**。
   ⇒ 我们大概率**改不了算子本身**，但可以查「**调用次数由什么决定、有没有用户侧开关能减少次数**」。
3. **它是串行的**（背靠背 1264 次），且独占一个 stream ⇒ 如果它在关键路径上，就是实打实的耗时；
   如果它只是"等待对端"（AICPU 忙等），那它占的是 AICPU 核资源，会与其它 AICPU 算子（如 MoE 的
   `MoeInitRoutingV3` = MIX_AIV…不含 AICPU）争抢。
4. **与 `hcom_allReduce_` 的次数对不上**：`hcom_allReduce_` 是 78–98 次/步，而它是 7.5 次/步
   ⇒ **不是一对一的控制面**，是另一条路径。

---

## 3. 待查问题（已派给单算子线）

1. 在 vllm-ascend / torch_npu / CANN 里搜 `allreduceAicpuKernel` 的注册与调用点，定位是哪个 op。
2. 7.5 次/步 的来源（draft 3 层？采样？logits allreduce？MoE AllGather 的 AICPU 部分？）。
3. 是"真计算"还是"忙等"（查 CANN 侧实现）。
4. 有没有用户侧能减少次数的开关。

---

## 4. 对当前优化的影响

**暂不做为优化目标**，因为：
* 若需算子团队重写，超出我们可行动范围；
* 它与主流并行，不会直接加到 ms/step。

**但要做两件事**：
1. **记账时把它排除在"device busy"之外**（否则会高估 busy）——
   这也是为什么 `dev_account.py` 报的 busy union (43.03 ms/step) 比实际墙钟步长 (33 ms) 大的原因之一。
2. **它可能是 `hcom_allReduce_` 时长的解释器**：如果 `hcom_allReduce_` 的 73.6 µs/op
   在等 AICPU 编排，那把 AICPU 链缩短就能同时改善两者。

---

## 5. 证据

| 内容 | 路径 |
|---|---|
| vmA profile（dummy 128K） | `logs/prof_vmA/`（5.7 GB，rank0 CSV 已导出到 `/tmp/vmA_rank0.csv`） |
| abA profile（较早，32K→128K） | `logs/prof_abA/`、`/tmp/abA_rank0.csv` |
| 对比脚本 | 内联 python（见本报告 §1 的口径） |

---

</details>

---

## 5. `dev_account.py` 的口径修正（**影响所有历史 busy 数字**）

它报 **busy union = 43.03 ms/step**，而同会话墙钟只有 ~35 ms/step ⇒ 账目不自洽。

**正确做法（两步，顺序不能反）**：
1. **先切段**：用锚点算子把窗口切成"纯 decode"与"prefill"两段，**只统计 decode 段**；
2. **再按 stream 分组**：主流（38/140）与辅助流（10）分开列。

剔除 prefill 后：vmA **33.30** / abA **34.09** ms/step ✓ 落在墙钟内。

⇒ 此前 `device-account-fixed-baseline.md` 里的 30.94 ms/step 等数字**需要按新口径复核**。
