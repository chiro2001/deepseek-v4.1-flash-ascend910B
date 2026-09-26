# 120 — ★★ **预注册**：`HcPre` 的 22.74 µs AIV 里，**sinkhorn 循环**该占多少（`hc_sinkhorn_iters` 20→1 值多少）

> 2026-09-23 06:5x CST。执行：**主代理**（纯源码阅读 + 算术，**零占卡**）。
> 目的：在测量之前把预测**写死**，避免事后编解释（本仓纪律，见 `094` 的先例）。
> 标记：**【源码】= 仓内逐字 / 【推断】/ 【未确认】**。

---

## 0. 为什么这条线值得单独开（量级）

| 事实 | 值 | 出处 |
|---|---:|---|
| `HcPre` decode 每步调用次数 | **86.00 次/步** | `PROF_MINE §6`（decode 稳态 96 步窗） |
| `HcPre` 每步耗时 | **2.709 ms/步**（11.2% 设备账） | 同上 |
| 单次中位时长（bs=6） | **30.56 µs** | `PROF_MINE §1.5`（8080 样本） |
| 其中 **AIV** 时间 | **22.74 µs** | 同上 |
| 成本结构 | 斜率 **187 ns/token**、截距 **29.44 µs** ⇒ **93% 是固定开销** | 同上（5→8064 token 拟合） |
| 6 token 时做的算力 | 3 MFLOP ≈ **0.1 TFLOP/s**（离峰值 3 个量级） | 同上 |

⇒ **它是设备侧最大的"可控"单项**（MoE/attention 的 GEMM 是模型数学、通信是结构性的，都不能动）。
★ 它也是 `CANNBOT_DOC` 认定"唯一量级够（≈3.6–4.2 ms/步）+ A2/A3 通用"的**单条**。

---

## 1. 【源码】`hc_sinkhorn_iters` 是怎么用的 —— 逐字

### 1.1 它是**运行时 attr**，不是编译期常量

`csrc/moe/hc_pre/op_host/hc_pre_def.cpp`（逐字）：
```
this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
this->Attr("hc_sinkhorn_iters").AttrType(OPTIONAL).Int(20);
this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
this->AICore().AddConfig("ascend910b");
```
`hc_pre_tiling.cpp:86-87`（逐字）：
```
auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;
```
（`:51` `constexpr int64_t ITER_TIMES_ATTR_IDX = 1;`、`:54` `DEFAULT_ITER_TIMES = 20`）
模型侧（`graph_prep/src/vllm_ascend/models/deepseek_v41/model.py:381`，逐字）：`hc_sinkhorn_iters=self.hc_sinkhorn_iters,`

⇒ **这个数从配置一路传到 kernel tiling，改它不需要重编 kernel**（但**需要重编图**，因为它在 `npugraph_ex` 捕获的图里）。

### 1.2 循环体：**每轮 6 个向量算子 + 2 个 PipeBarrier**

`csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h:468-484`（逐字）：
```cpp
for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
    LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * hcMult, hcMult);
    Adds(reduceLocal, reduceLocal, hcEps, curRowFactor * hcMult);
    PipeBarrier<PIPE_V>();
    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                       curRowFactor * hcMult, hcMult);
    ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, hcMult, hcMult);
    Adds(reduceLocal, reduceLocal, hcEps, curRowFactor * hcMult);
    PipeBarrier<PIPE_V>();
    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, hcMult, hcMult);
}
```

### 1.3 ★ 关键：**每次迭代只处理 4–16 个浮点数**

`hc_pre_tiling.cpp:147-149`（逐字）：
```
rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;
```
decode 时 `bs_ = 6`、`aivCoreNum_` = AIV 核数（40/48，≫6）⇒
`rowOfFormerBlock_ = 1`、`usedAivCoreNums_ = 6`、`rowOfTailBlock_ = 1`
⇒ **每核 1 行** ⇒ `stage2RowFactor = rowFactor_ = 1` ⇒ 循环里 `curRowFactor = 1`
⇒ 每个算子的元素数 = **`curRowFactor × hcMult = 4`**（`hcMult=4`），最大的一处是 **16**（`hcMult × hcMultAlign`）。

⇒ **19 轮 × (6 算子 + 2 屏障) = 114 个算子 + 38 个 PipeBarrier，全程在 4–16 个元素上。**
⇒ **【推断·强】这部分成本**不可能**来自数据量或算力，只能是 `issue + 屏障同步` 的固定开销。**

---

## 2. ★ 预注册（测量前写死，允许被打脸）

### 2.1 预测

| # | 预测 | 量 |
|---|---:|---|
| P1 | 若 sinkhorn 循环是主成本 ⇒ **每次迭代**（6 算子 + 2 屏障）≈ **0.5–1.1 µs** | —— |
| P2 | `hc_sinkhorn_iters` **20 → 1** 省 **15–21 µs/次**（= 19 × P1） | —— |
| P3 | 折算到每步：**1.29–1.81 ms/step**（× 86 次/步） | —— |
| P4 | 扫 `iters ∈ {1,2,5,10,20,40}` 应**单调递增且近似线性**（斜率 = P1） | R² ≥ 0.98 |
| P5 | `iters=1` 的读数应接近"**不含 sinkhorn 循环**"的底：即 `HcPre` 的 prologue + `hc_fn` 权重读 + Cube 部分 | 期望 8–14 µs |

### 2.2 判别性预测（★ 这一条决定"关不关线"）

* **若扫描是平的（20 与 1 只差 <3 µs）** ⇒ sinkhorn **不是**主成本 ⇒ 主成本在
  **prologue（`hc_fn` 1.87 MiB fp32 权重读 = 80 次/步 ≈ 150 MiB/步）+ Cube 阶段** ⇒
  **这一格关闭**（那是"权重带宽"或"定制核"的活，属 C 层），**不许再拿它做 ≤24 的算账**。
* **若扫描近似线性且斜率 ≥0.5 µs/轮** ⇒ sinkhorn 是主成本 ⇒ 值得评估 `hc_sinkhorn_iters` 的
  **配置 A/B**（**但必须过精度门**，见 §3）。

### 2.3 与"要不要动"的关系

★ **`hc_sinkhorn_iters` 是数值参数，不是性能开关**：它控制一个不动点迭代的收敛程度。
**改小 ⇒ 路由权重矩阵的归一化不收敛 ⇒ 可能掉精度/掉投机接受长度 A**。
⇒ 因此**任何**利用这一格的方案都必须同时满足：
1. 题库 **≥10/10**（与现状不退化）；
2. `prefix-pair` 三发逐字相同；
3. **`accept_length` A 不明显下降**（它是投机解码的收益指标，掉 A 会吃掉 ms/step 的收益）；
4. 8K/32K/128K 三档 quote 不退化。
⇒ **只看 `ms/step` 得结论 = 判据绑错对象**（本仓已记多次同族错误）。

---

> ★★★ **2026-09-23 07:2x 更正（见 `logs/125`）**：本节原来写"`prbench-c1` 里跑不起来、被环境挡住"——**那是我的误判**。
> 真相：`dir(torch.ops._C_ascend)` 只回 2 个名字是**假阴性**；`HC_LINE` 指出并**主代理已独立复核**：
> ```python
> import vllm_ascend.vllm_ascend_C                     # ← 前置 1
> sch = torch._C._jit_get_all_schemas()                # ← 用这个，不用 dir()
> # 结果：_C_ascend schema 总数 = 68，含
> #   _C_ascend::npu_hc_pre_v2(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, Tensor? pre_mix=None, *, int hc_mult=4, int hc_sinkhorn_iters=20, ...)
> #   _C_ascend::npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor out
> ```
> 还需要 **`ASCEND_CUSTOM_OPP_PATH=<...>/_cann_ops_custom/vendors/custom_transformer`**（否则报 `aclnnHcPre ... not in libopapi.so`）。
> ⇒ ★ **单卡 c1 即可做 Hc 线的全部工作**（不需要生产镜像容器、不需要 8 卡臂）——这是一条**重要的流程解绑**。
> ★ 教训（与 `AGENTS.md §5b` 同族）：**`dir()` 不是注册状态的判据**；判"算子在不在"要用 **`jit_get_all_schemas()` 并在正确的 import 之后**。

## 3. 测量方案（已在写探针；**不占 8 卡**）

探针：`a2/agents/HC_PROBE/hc_probe.py`（扫 `hc_sinkhorn_iters` / `pre_mix` / batch 三组）。
装置：**图内斜率**（沿用 `GEOM_MICRO` 的方法）或简单 `event` 计时 × 多次取中位；
**必须在生产镜像里跑** —— 因为 `npu_hc_pre_v2` 是 vllm-ascend 自己在
`csrc/torch_binding.cpp:2894` 注册的私有算子（`TORCH_LIBRARY_EXPAND(_C_ascend, ops)`），
**只在生产镜像里被注册**；`prbench-c1` 里 `torch.ops._C_ascend` 是空壳（**【实测】**：
`n_ops = 2`），而同族的 `torch.ops.npu.npu_mhc_pre`（op-plugin 的 MHC 家族）**不是同一条 kernel**，
**不能当判据**。

★ **落地安排**（省机时）：`G2G3_ARM` 正在起 8 卡臂（`KEEP=1`，容器在生产镜像里、8 rank 齐）；
**在它自己的压测跑完、拆容器之前**，用 `docker exec` 在该容器里跑本探针（**单 rank**，只占 rank0 的设备时间）。
⇒ 这样**额外占用 = 0 张卡、0 分钟起服时间**。

---

## 4. 这一格若成立的后果（**它会改变 ≤24 的判决**）

按 `logs/119 §6` 的 A 层清单（≈1.04–1.47 ms/step）：

```
27.905 − 1.47（A 层上限） − 1.81（HcPre/sinkhorn 上限）
   = 24.63 ms/step   ← 距 24 只差 0.6
```
⇒ ★★ **在"8K 服务端 quote"这把尺子上，第一次出现"≥24 但贴线"的可能**；
⇒ 但仍需与 **A2-only 的 MC2（≤1.153）** 或 **C 层**叠加才能真正过线。
★ 且**必须**通过 §2.3 的四道精度门 —— 否则这个数**不算数**。

---

## 5. 附：顺带定位到的"prologue 里还有什么"（供 P5 对照）

镜像外的同族实现里，Part2 在进循环**之前**还要做（`hc_pre_m_k_split_core.h:440-467` 一带）：
`CopyIn(workspace…)`（跨核 workspace 取 `hcMix` 个 `hcMult×hcMult` 块）→
`ReduceSumARAPerf` → `MulABLastDimBrcInline`（乘 `rsqrt`）→ `Muls`（乘 `hc_scale[2]`）→
`AddBAFirstDimBrcInline`（加 `hc_base`）→ `SoftmaxFP32Perf` → `ReduceSumARAPerf` → `Adds` →
`DivABABrcInline` ⇒ **约 8 个算子 + 1 次跨核 `CopyIn`**。
⇒ `iters=1` 的读数 = 这 8 个算子 + Stage1（Cube/权重读）+ 发射开销，是解释 §2.2 的关键对照。
