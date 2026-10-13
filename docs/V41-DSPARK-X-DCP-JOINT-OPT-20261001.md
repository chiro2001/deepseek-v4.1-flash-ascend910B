# DSpark × DCP8 联合优化分析（2026-10-01）

> 触发问题（用户）：**"如果把 dspark（推测解码）和 DCP merge 一起考虑，是不是有更多融合/优化机会？"**
>
> 结论先说：**机会在"摊薄"，不在"融合"**；而 DSpark 本身**对 ms/step 是明确的代价**。

---

## 0. ⚠️ 重要更正（2026-10-01，用户指出）

**初版本文写了一句错的**：*"开 DSpark 后 step 墙钟 39.5 ms，与 A2 单实例基线 38.5 ms 同量级
⇒ 多算 7 个 token 近乎免费。"*

这句话的基线拿错了：那个 **A2 单实例 38.5 ms 本身就是开着 DSpark 的**（A2 生产口径
"除了 int8 kv 和 pd 分离，其他特性都开了，包括 draft 入图"）。它只能证明
**"CED 不引入额外开销"**，**不能**证明"DSpark 免费"。

**正确的同条件 A/B**（`docs/CED-PD-DYNAMIC-SPEC-20260926.md` §11.4 / `CED-PD-SPEC-MODE-20260928.md` §6）：

| 档 | ms/**step** | A | ms/token | decode tok/s |
|---|---:|---:|---:|---:|
| 静态 `SPEC=0` | **24.35** | 1.0（每步 1 token） | 24.35 | 41.06 |
| 静态 `SPEC=1 DRAFT_GRAPH=1`（SP_TOKENS=7） | **32.68** | 3.10 | 10.54 | 94.79 |
| 动态 K=7 | 30.15 | 2.62 | 11.51 | 87.07 |

⇒ **开 DSpark 让 ms/step 从 24.35 涨到 32.68 = +34.2%（+8.33 ms/step）。**
**对"每步时延"是明确的退化；对"每 token 时延 / 吞吐"是 2.31× 的改善。**

### 这 +8.33 ms/step 花在哪【推断+部分实测】

profiler 文档 §0 给了 step 的**工作单位定义**：

> 一个 decode step = 1 次 **40 层 target forward（M = 1 + SP_TOKENS = 8 行）**
> + 1 次 **3 层 draft forward** + 验证

| 组成 | 证据 |
|---|---|
| (a) target forward 的 M 从 1 变 8 | 激活/KV 写入/attention 行数 ×8（权重流量不变） |
| (b) **3 个 draft block 的前向**（有自己的一套权重） | `DeepseekV41DSparkModel`：*"Three serial draft blocks matching the checkpoint's `mtp.*` tree"*；profiler 里 `HcPre 86 次/step = 40×2 + 3×2` —— **draft 的 6 次确实在里面** |
| (c) 验证 / 采样 / 接受逻辑 | 每步多一组 host↔device 交互 |

**【未确认】三者的分摊比例** —— 需要一次专门 A/B（例如固定 SP_TOKENS 扫 1/3/7，
或关掉 draft 只跑 verify）才能拆开。**不要凭直觉分配。**

### 这条更正如何改变结论

| 指标 | DSpark 的影响 |
|---|---|
| **ms/step**（每步时延、单步响应性） | **+34%，明确更差** |
| ms/token、decode tok/s、同长度输出的总耗时 | **2.31× 更好** |
| DCP8/DCP1 的**比值**（两种指标下） | 大致不变（draft 无 DCP 开销，但 verify 有） |

⇒ **本线程早期目标里的验收口径是 `ms/step < 1.2×` —— 在那个口径下 DSpark 是退化项，不是优化项。**
只有把指标换成 ms/token 或吞吐，它才是 2.3× 的大杠杆。
**必须先确认指标是什么，再决定开不开 DSpark。**

---

## 1. 事实一：当前实例 **DSpark 是关的**（`SPEC=0`）

| 证据 | 值 |
|---|---|
| `dsv41-gen1` 起服日志 | `speculative_config=None` |
| `dcp_stage_capacity.sh` | `export SPEC=0 MAX_SEQS=16 ...` |
| 交付口径（CED-PD） | `SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1` —— **DSpark 才是交付默认** |

⇒ 这是目前**最大的一根绝对性能杠杆**，而它是免费的（不需要写代码）。

### 历史实测（CED-PD 拓扑，**非 DCP**，2026-09-26）

| ctx | ms/step | A | **ms/token** | 相对 `SPEC=0`（26.8 / 27.4 ms/token） |
|---|---:|---:|---:|---|
| 32K | 34.22 / 34.22 | 2.59 / 3.61 | **13.20 / 9.70** | **≈2.4×** |
| 144K | 36.43 / 36.56 | 3.17 / 3.07 | **11.47 / 12.09** | **≈2.3×** |

来源：`docs/CED-PD-DSPARK-ACCEPTANCE-20260926.md` §2。
另有同源 profiler：CED-PD D 侧 **39.5 ms/step**（profiler 开启，本身约 8% 开销），
device core 累加 **38.5 ms**，与 A2 单实例基线 **38.5 ms** 同量级，且"同样的算子清单、同样的次数"
⇒ **CED 的 128-token 重放不给 decode 添额外开销**。

⚠️ **但这不等于"DSpark 免费"** —— 那个 A2 基线本身也是开 DSpark 的。
正确的同条件 A/B 见上面的 §0：**DSpark 让 ms/step +34.2%**。

> ⚠️ **口径警告（该文档自己写的）**：`ms/token = ms/step ÷ A`。
> **A≈1.0 时 ms/token 会假装变快**（每步只出一个 token）—— `reports/draft-graph-negative-control.md`
> 的负控就是这么骗人的。**任何 DSpark 性能结论必须同时报 A。**

---

## 2. 事实二：**draft 本身完全不需要 DCP**（结构性）

```
DeepseekV41DraftSWASpec(AscendSlidingWindowMLASpec)      ← 不是 FullAttentionSpec
        ↓ spec_is_dcp_sharded() → False
        ↓ resolve_group_dcp()    → 1
  draft SWA 组：复制态、effective_dcp=1
  ⇒ 无 q head all_gather、无 merge allReduce、无 remap
```

代码依据：`core/deepseek_v41.py:69` 的类定义 + `patch_v41_dcp.py:spec_is_dcp_sharded`。
另外 `DeepseekV41DSparkAttention.__init__` 显式要求 **`compress_ratio == 0`**（"supports only
uncompressed draft SWA layers"）⇒ draft 连压缩注意力都不走，也就**不会碰到我们修的那条
`ratio=1` 稀疏路径**。

**⇒ 推论：DSpark 产出的 A≈3 个 token，几乎不携带任何 DCP 专属开销。**

---

## 3. 事实三：DCP 的全部开销都是**每步固定**的，会被 A 摊薄

| DCP 专属开销 | 每步 | 每 token（当前 A=1） | 每 token（A=3） |
|---|---:|---:|---:|
| merge allReduce（76 次/step） | 0.98 ms | 0.98 | **0.33** |
| q-gather（36 次/step） | 0.67 ms | 0.67 | **0.22** |
| `Sort`/remap（8 次/step） | 0.65 ms | 0.65 | **0.22** |
| merge 逐元素后处理（573 图节点） | ~1.72 ms | 1.72 | **0.57** |
| **合计** | **~4.0 ms** | **4.0** | **~1.34** |

**注意：ms/step 一点都不省**（这些是 per-step 成本）。省的是 **ms/token**。

---

## 4. 事实四：我新写的 merge 融合算子**对 T 是次线性的**（实测）

`ascendc/merge/bench_merge.py`（chip6）：

| T | pre | post | 合计/层 | ×38 层 | **每 token** |
|---:|---:|---:|---:|---:|---:|
| **1** | 7.08 µs | 6.61 µs | 13.68 µs | 0.520 ms | **13.68 µs/token** |
| **8**（DSpark 的 verify 尺寸） | 14.14 | 6.69 | 20.83 µs | 0.792 ms | **2.60 µs/token** |
| 16 | 19.99 | 6.39 | 26.38 µs | 1.002 ms | 1.65 µs/token |

⇒ **行数涨 8×，kernel 只涨 52%** ⇒ 每 token 成本降 **5.3×**。
DSpark 让这个算子（以及所有类似的固定开销项）**按 token 计的价值放大 5 倍**。

---

## 5. 只有"两者同时开"才成立的新机会

### ★ (a) M 补零到 16 的收益变大（邻居已实测，零开发量）

邻居 `op_peak` 实测 vendor matmul：**M=8 → 14.99 µs，M=16 → 10.45 µs（1.43×）**。
- `SPEC=0` 时每步只有 1 个 query token，**没有 M 可以补**；
- **DSpark 让 T=8** ⇒ 补零到 16 是自然的（多算 8 行无用结果，权重流量不变）
  ⇒ 每次 matmul 省 ~4.5 µs，40 层 × 若干次 matmul ⇒ **~0.3 ms/step**。

**这条只在 DSpark 打开时才存在。**

### ★ (b) draft SWA 与 target SWA **共享缓存槽**

`DeepseekV41DraftSWASpec` 的 docstring：
> *"DSpark SWA owned by G12, **aliasing target slots at distinct block IDs**."*

⇒ draft 不再需要独立写一遍 SWA；它复用 target 刚写的那块。
这已经是既成事实（不用改），但它意味着 **draft 的 KV 侧边际成本 ≈ 0**。

### ★ (c) 集合通信的"字节免费"窗口还有 6 倍余量

历史实测（`dcp2_perf` 子代理）：collective **16 B 与 1 MB 同价**，拐点在 **~8 MB**。
- 当前 `pack` 是 `[1,64,640]` fp32 = **164 KB**；
- DSpark 的 T=8 让它变成 **1.3 MB** —— **仍在拐点以下，仍然免费**；
- q-gather 从 8 KB 变成 64 KB —— 也在免费区。

⇒ **开 DSpark 不会让集合通信变贵**，而"优化集合通信"这件事的**每 token 价值下降 3 倍**
（因为它本来就是 per-step 的）。

### (d) 反过来：per-step 的优化（我的 merge kernel、QBMV3 三合一）**每 token 价值也下降 A 倍**

| 优化 | 省 ms/**step** | 每 token（A=1） | 每 token（A=3） |
|---|---:|---:|---:|
| 我的 merge 融合算子 | ~0.9–1.2 | 0.9–1.2 | **0.30–0.40** |
| 邻居 QBMV3 三合一 | ~2.2（40 层口径） | 2.2 | **0.73** |
| 邻居 wo_a Triton | ~0.6 | 0.6 | **0.20** |
| **开 DSpark（净效应）** | **+8.33（变差！）** | **−13.8（变好）** | 同上分摊 |

⇒ **两者是不同指标下的东西，不能混着做"加总"**：
* 若指标是 **ms/step**：上面三条优化都是**净收益**，而 DSpark 是 **+34% 的退化**；
* 若指标是 **ms/token / 吞吐**：DSpark 一个开关（2.31×）比上面三条加起来还大，
  但它们的收益也要按 A=3.1 折算（÷3.1）。

### (e) 一个**不能**融合的地方（避免走弯路）

merge 的 `pre`（`w = exp(clamp(lse−ori_lse))`、`scaled = out·w`）与 `post`
（`num = pack − alpha·ori_out`、`num/den`）**中间夹着 all_reduce，物理上不能合成一个 kernel**。
我当前的 2-kernel 设计已经是这个约束下的最优切分。
另外 `ori_out` 是 rank 不变的（复制态），所以 `Σ_r(pack_r − α·ori) = Σpack_r − dcp·α·ori`
—— 想"把它挪到 all_reduce 之前"反而要多加回 `(dcp−1)·α·ori`，**不划算**。

---

## 6. 风险与未确认项

1. **DSpark × DCP8 这个组合从未跑过。** 历史 DSpark 数字全部来自 **CED-PD（无 DCP）**；
   我们当前实例 `SPEC=0`。必须实测。
2. 已知代码门：`num_speculative_tokens` 必须 `0 < n < STATE_RING_ROWS(32)`（SP_TOKENS=7 OK）；
   `Aurora supports only DSpark speculative decoding`；`V41_CED_ALLOW_DSPARK` 只影响 CED 角色。
3. **【未确认】** draft 的 SWA 缓存组（G12）在 DCP 分槽（`plan_cache_slots`）下的容量账——
   它是复制态且 aliasing，理论上不额外吃容量，但要把 `KV cache size` 实测出来对账。
4. **【未确认】** `aux_hidden_state` 取自 target 层 37/38/39 的残差；DCP 下残差是复制的
   （TP all-reduce 后），所以应该没问题 —— 但 8 卡下没验过。
5. 历史有"**四并发时 DSpark 几乎没收益**"的记录（本线程早期）⇒ 高并发下的策略要另测。

---

## 7. 建议的执行顺序（与线 A 协调）

| # | 动作 | 预期 | 说明 |
|---|---|---|---|
| **0** | **先定指标**：ms/step 还是 ms/token/吞吐？ | — | 决定 DSpark 该不该开（见 §0） |
| 1 | 加载 merge 融合算子（`V41_DCP_MERGE_KERNEL=1`），同实例 A/B | **−0.9~1.2 ms/step**（两条指标都受益） | 已单卡验证通过 |
| 2 | 邻居的 QBMV3 三合一（**线 A 正在做**） | ~2.2 ms/step | 见子代理 |
| 3 | **若要**开 DSpark：`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`，**必须同时报 ms/step + A + ms/token** | ms/step **+34%**；ms/token **2.31×** | 需要一次独立重启 |
| 4 | M 补零到 16（**只在 DSpark 打开时才有意义**） | ~0.3 ms/step | 依赖 3 |

**建议**：把 1 与 2 合并成一次重启（都是 ms/step 的净收益）；
**DSpark（第 3 项）单独做一轮**，因为它会把 ms/step 拉高 34%，
和 1/2 混在一起测会污染归因。
