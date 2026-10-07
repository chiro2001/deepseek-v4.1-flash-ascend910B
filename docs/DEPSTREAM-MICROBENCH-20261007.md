# DepStream 微基准：双流回放能不能真的重叠（2026-10-07）

> 线名已由 U3 改为 **DepStream**（依赖驱动侧流化），命名理由见 `PLAN-UBATCH-AND-DEPSTREAM-20261007.md` §0.0。

> 目的：在动真机之前，用**微基准**回答 DepStream 的前提问题——
> **把 AIC / AIV 两类算子分到两条流上，设备到底会不会并行？profiler 上能不能看见？**
> 环境：a3-21 **chip 5**（`dsv41-op-hcfuse`，算子实验专用，全程未动 tp8k5 / tiny）。
> 形状全部取自**真实 profile**（`armF_r6_base`，stream 109，M=48）。全部【实测】。

---

## 0. 一页纸

| 问题 | 答案 |
|---|---|
| 一张 NPUGraph 能不能捕获"真算子的双流 fork/join"？ | ✅ **能**（真实算子，不是合成 matmul） |
| 双流下发后，AIC 与 AIV 真的并行吗？ | ✅ **交叠率 91~94%** |
| 生产环境的交叠率是多少？ | ❌ **只有 24%**（3.50 / 14.75 ms） |
| 设备侧"工作并集"被压缩多少？ | **1.26 ~ 1.45×** |
| 图重放壁钟被压缩多少？ | **1.05 ~ 1.49×**（含 ~70~100 µs/次重放的固定开销） |
| 跨流 event 有多贵？ | **≈0.64 µs / 对**（48 对实测）⇒ 272 次交替/步 ≈ **0.17 ms**，可忽略 |
| ⚠️ 这证明 DepStream 成立了吗？ | **没有**。微基准用的是**互相独立**的两条链；真实模型的相邻 AIC/AIV 是**硬依赖** |

**一句话**：**机制已验证、代价已量化**；剩下的唯一问题是"**真实模型里到底有多少独立工作可以挪**"——
那必须靠依赖审计（见 `PLAN-UBATCH-AND-DEPSTREAM-20261007.md` §3.4）。

---

## 1. 方法

### 1.1 真实算子与形状（M=48，取自 profile `stream 109`）

| 标签 | 算子 | 真实形状 | 实测 core / 时长（µs） |
|---|---|---|---|
| `AIC_mm` | `aclnnMatmul_MatMulCommon_MatMulV2` | `(48,1024)@(1024,5120)` bf16 | **AI_CORE** / ~9.5 |
| `AIV_rms` | `RmsNorm` | `(48,5120)` + `(5120,)` bf16 | **AI_VECTOR_CORE blk48** / ~26.8 |
| `AIV_dq` | `DynamicQuantV2` | `(48,5120)` bf16 | **AI_VECTOR_CORE blk16** / ~18.3 |
| `MIXV_route` | `MoeInitRouting` | `(48,5120)` + `(48,6)` int32 | **MIX_AIC**（*不是* MIX_AIV） / ~30.2 |

> ⚠️ 两个踩坑：
> 1. `npu_quant_matmul` 必须带 `pertoken_scale=` 才是生产调用形态；不带直接报
>    `NPU function error ... QuantMatmulKernelNpuOpApi.cpp:297`；
> 2. **profile 里的 `MoeInitRoutingV3` 名字与实际 kernel 名 `MoeInitRouting` 不同**，
>    且它的 Accelerator Core 是 **MIX_AIC** 而非 MIX_AIV —— 按名字猜 core 会出错。

### 1.2 三种测量口径（缺一不可）

| 口径 | 做法 | 作用 |
|---|---|---|
| 壁钟（eager） | 循环 40 次 + `synchronize` | ⚠️ **不可用**：被 host 派发掩盖（见 §2.1） |
| 壁钟（图重放） | 捕获成 NPUGraph，重放 50 次取均值 | 消除 host 开销，给端到端 |
| **设备并集（profiler）** | 逐 kernel 取 `[start, start+dur]`，算 AIC/AIV 的并集与交集 | **唯一能看见"谁和谁并行"的口径** |

### 1.3 双流图的捕获（关键写法）

```python
g = torch.npu.NPUGraph(); root = torch.npu.Stream(); s2 = torch.npu.Stream()
ef = torch.npu.Event(); ej = torch.npu.Event()
with torch.npu.graph(g, stream=root):      # 捕获根流 = root（不是默认流！）
    ef.record(root)                        # fork event 必须 record 在捕获根流上
    with torch.npu.stream(s2):
        s2.wait_event(ef)
        for _ in range(N): fb()
        ej.record(s2)
    for _ in range(N): fa()                # 根流跑另一支
    root.wait_event(ej)                    # 汇合
```

这与 `MULTISTREAM-GRAPH-CAPTURE-BREAKTHROUGH` 的规则一致，**这次是在真实算子
（`aclMatmul` / `RmsNorm` / `DynamicQuant` / `MoeInitRouting`）上复现成功**。

---

## 2. 结果

### 2.1 ⚠️ eager 壁钟不可用（方法论）

第一次用 eager 壁钟测，得到"**并发比串行慢**"的荒谬结果：

| pair | 串行 | 并发 | 比值 |
|---|---:|---:|---:|
| `AIC_mm` / `AIV_rms` | 1353 µs | 1502 µs | 0.90× |
| `AIV_rms` / `AIV_dq` | 1642 µs | 1809 µs | 0.91× |

**原因**：eager 下 host 派发 ≈ 7~10 µs/算子，而算子本身只有 9~27 µs
⇒ 测的是 Python 派发，不是设备。与 `DBO-RUNTIME-VERDICT` 同源
（"eager 下 84% 时间在 Python 侧 ⇒ DBO 成败只能在图模式下判定"）。
⇒ **微基准也必须图重放 + profiler 双口径。**

### 2.2 ★ 设备侧交叠率（profiler，12 次重放）

| pair | 串行 A 支 | 串行 B 支 | 串行并集 | 并发交集 | 并发并集 | **交叠率** |
|---|---:|---:|---:|---:|---:|---:|
| `AIC_mm` ∥ `AIV_dq` | 2.290 | 1.484 | 3.775 | **1.429** | **2.657** | **91.7%** |
| `AIC_mm` ∥ `AIV_rms` | 2.290 | 2.235 | 4.525 | **2.140** | **3.562** | **91.8%** |
| `AIC_mm` ∥ `MIXV_route` | 2.302 | 2.696 | 4.997 | **2.627** | **3.963** | **91.4%** |
| `AIV_rms` ∥ `AIV_dq` | 2.197 | 1.458 | 3.655 | **2.096** | **2.526** | **93.8%** |
| `AIV_rms` ∥ `MIXV_route` | 2.434 | 3.095 | 5.529 | 3.084 | 5.388 | 82.0% |

* **串行臂交集恒为 0.000** ⇒ 同一条流上的算子完全串行（符合预期）；
* **并发臂交集 = min(A,B) 的 91~94%** ⇒ 图里的双流真的并行执行；
* 对照组：**生产环境**（`armF_r6_base` 全步）AIC ∩ AIV = **3.50 / 14.75 = 24%**。

### 2.3 设备并集压缩 vs 图重放壁钟

| pair | 串行并集(ms) | 并发并集(ms) | **设备压缩** | 壁钟串行 | 壁钟并发 | **壁钟压缩** |
|---|---:|---:|---:|---:|---:|---:|
| `AIC_mm` ∥ `AIV_dq` | 3.775 | 2.657 | **1.42×** | 0.320 | 0.293 | 1.09× |
| `AIC_mm` ∥ `AIV_rms` | 4.525 | 3.562 | **1.27×** | 0.381 | 0.306 | **1.24×** |
| `AIC_mm` ∥ `MIXV_route` | 4.997 | 3.963 | **1.26×** | 0.492 | 0.330 | **1.49×** |
| `AIV_rms` ∥ `AIV_dq` | 3.655 | 2.526 | **1.45×** | 0.312 | 0.296 | 1.05× |
| `AIV_rms` ∥ `MIXV_route` | 5.529 | 5.388 | 1.03× | 0.480 | 0.477 | 1.01× |

> **壁钟 < 设备压缩的原因**：图重放有 **~70~100 µs/次** 的固定开销（本微基准每次重放只有
> 24+24 个算子、~300 µs 量级，开销占比 25~30%）。真实 decode 步长是 **24~40 ms / 1500 算子**，
> 该开销会被摊薄到可忽略。
> ⇒ **应以"设备并集压缩 1.26~1.45×"外推，而不是壁钟的 1.05~1.49×。**

### 2.4 跨流 event 的代价（决定"能不能切细"）

| 链长 | 无 event | 有 event（每对算子 fork+join） | 差值 |
|---:|---:|---:|---:|
| 16 对 | 0.298 ms | 0.298 ms | **0** |
| 48 对 | 0.757 ms | 0.818 ms | **0.061 ms** ⇒ **0.64 µs/对** |

⇒ 主流上有 **272 对 AIC↔AIV 交替**（`DECODE-PARALLELISM-WHAT-IS-HIDDEN` §4）
⇒ 全切流的 event 代价 ≈ **0.17 ms/步（0.4%）**，**可以忽略**。
**瓶颈从来不是 event，而是"有没有独立工作"。**

---

## 3. 这次证明了什么、没证明什么

### 3.1 证明了

1. **图内双流是真实的设备并行**，不是排队假象（91~94% vs 生产 24%）；
2. **硬件/编译器不阻碍 AIC∥AIV**：`AI_CORE ∥ AI_VECTOR`、`AI_CORE ∥ MIX_AIC`、
   `AI_VECTOR ∥ AI_VECTOR` 都测到 91~94%；
3. **生产只有 24% 不是设备限制**，而是当前下发形态（单流 + 少量硬编码侧流）的结果；
4. **event 代价可忽略**（0.64 µs/对）；
5. **eager 壁钟会把结论测反**，必须图重放 + profiler。

### 3.2 ❗没有证明（不要误读）

微基准里两条流跑的是**互相独立的两条链**（`matmul` 与 `rms_norm` 读不同张量）。
而真实模型里**相邻的 AIC/AIV 是硬依赖**：

```
HcPre(AIC) → RmsNorm(AIV) → DynamicQuant(AIV) → QuantMatmul(AIC) → RoPE(AIV) → …
```

**光把它们分到两条流上没有任何收益**——依赖链上仍然是严格交替。
⇒ **DepStream 的全部难点在"找到独立工作"，不在"切流"。**

### 3.3 因此下一步只有一个问题

> **主流这 25.4 ms 里，有多少工作是"不属于紧邻依赖链"的？**

已知可挪候选（`PLAN-UBATCH-AND-DEPSTREAM` §3.3）：
主流 AIV 窗口里 **AIC 空转 6.43 ms**、尾部 **7.71 ms gap**、
侧流已承载的 **3.98 ms AIC / 5.44 ms AIV**。具体能挪多少，必须做依赖审计。

---

## 4. 复现

```bash
# 全部在 a3-21 chip5（dsv41-op-hcfuse），不影响 tp8k5 / tiny
ssh a3-21 'docker cp ~/tmp/benchG.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 benchG.py"'
ssh a3-21 'docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 ovl2.py /tmp/u3prof"'
ssh a3-21 'docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 benchW.py"'
ssh a3-21 'docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 benchE.py"'
```

工具已入仓：`tools/benchG.py`（双流图捕获+profiler）、`tools/ovl2.py`（并集/交集分析）、
`tools/benchW.py`（图重放壁钟）、`tools/benchE.py`（event 代价）、`tools/benchA2.py`（eager 反例）、
`tools/opseq.py`（提取真实算子序列）、`tools/shapes.py`（提取真实形状）。

---

## 5. 待办

| # | 动作 | 判据 |
|---:|---|---|
| 1 | **Bench B：真实算子顺序回放**（一个 layer 全部算子，1 流 vs 2 流） | 若 2 流无收益 ⇒ 坐实"仅切流无效"，精力全部转到"找独立工作" |
| 2 | **依赖审计**（代码级 + 延迟注入探针） | 可挪候选合计 ≥0.5 ms |
| 3 | 候选足够 ⇒ 在图里挪到侧流 → profiler 验证交叠 → 精度门 | 交叠率 ≥50% 且答案稳定 |
