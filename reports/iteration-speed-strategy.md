# 如何更快迭代：起服耗时拆分 + 三层测试金字塔

> 2026-09-16 10:40 CST｜回答用户提问：「保持 TP8 但可用 8chip/2chip/1chip 做算子组合优化，
> 另一条线完整起服保证精度」是否可行

---

## 0. 结论

**方向正确，但三点要修正**：

| # | 修正 | 依据 |
|---|---|---|
| 1 | **算子级工作根本不需要 8 卡 —— 单卡已经够，且快 20–40×** | 单卡 harness 全流程 **41 s**、一轮迭代 **20–30 s**；已独立复现 `wo_a` 结论 |
| 2 | **不要用 2chip / 4chip**（TP≠8 的数字不可迁移） | TP=2 时每 rank 专家数 192（vs 48）、通信量、每 rank token 数全变 ⇒ 是**另一个配置**，不是 TP8 的便宜版 |
| 3 | **8 卡起服本身可以从 8m29s 压到 ~6m** —— 不需要牺牲任何东西 | 见 §2：4 个 capture 里我们只需要 1 个 |

---

## 1. 起服耗时实测拆分（cmB 会话，总 **8 min 29 s**）

| 阶段 | 耗时 | 占比 | 可否压缩 |
|---|---|---|---|
| 容器启动 → engine init | **38 s** | 7% | 难 |
| 模型 init + **权重加载**（`Loading weights 65.2s` + draft `16.2s` + Engram int8 206 GB + ckpt 解析） | **3 min 19 s** | 39% | 需要换 ckpt 格式，风险高 |
| Triton warmup（`deepseek_v41_indexer ... 21.8s` 等） | **22 s** | 4% | 部分可缓存 |
| `torch.compile` | **22 s** | 4% | 已有缓存（`torch_compile_cache`） |
| profiling / warmup run | 7 s | 1% | — |
| KV cache 分配 | 1 s | <1% | — |
| **ACL 图捕获** | **3 min 03 s** | **36%** | ★ **可压到 ~1 min**（见 §2） |

证据：`logs/perf/cmB_1029_serve.log`（时间戳 02:29:14 → 02:37:43 = READY）。

---

## 2. ★ 可立刻拿到的加速：捕获从 4 个桶降到 1 个

日志原文：

```
Capturing CUDA graphs (decode, FULL):   0%|  | 0/4
  [op_compiler] static kernel compile start   ← 第 1 次
  [op_compiler] static kernel compile success
Capturing CUDA graphs (decode, FULL):  25%|  | 1/4 [01:21<04:04, 81.65s/it]
Capturing CUDA graphs (decode, FULL):  50%|  | 2/4 [01:22<01:07, 33.89s/it]
  [op_compiler] static kernel compile start   ← 第 2 次
Capturing CUDA graphs (decode, FULL):  75%|  | 3/4 [02:10<00:40, 40.45s/it]
  [op_compiler] static kernel compile start   ← 第 3 次
Capturing CUDA graphs (decode, FULL): 100%|  | 4/4 [03:03<00:00, 45.43s/it]
```

**4 个 capture、3 次静态内核编译、平均 45 s/个**，共 3 min 03 s。

`cudagraph_capture_sizes` = `[1,2,3,4,6,8,12,16,20,24,32]`，但实际只捕了 4 个 ——
因为 `MAX_SEQS=4`，decode 的 token 数可能落在 **6 / 12 / 18 / 24**（1~4 个并发请求 × (1+SP_TOKENS=6)）。

**而我们测的是单流**（`max_running=1`）⇒ **只需要 6 这一个桶**。
⇒ `MAX_SEQS=1` 预期把捕获从 4 降到 1，省 **~2 min 15 s（起服 −26%）**。

这**不会改变测量结果**（单流本来就只有 1 个请求），属于纯配置优化。
已排队验证（`exp_tools/boot_time_test.sh`：MAX_SEQS=1 vs 4，各测 time-to-READY + 捕获次数 + KV + 128K 性能）。

### 2.1 为什么"静态内核编译"这么贵

每次 capture 都会触发一次 `[op_compiler] static kernel compile`（121 项）。
缓存目录是 `$P/p36_static_kernel_a21/{compile_outputs,install}`，但**按 capture 的图**
编译 ⇒ 图变了（或捕获尺寸变了）就重新编。⇒ 减少 capture 数量 = 同时减少静态内核编译次数。

---

## 3. 单卡 harness 已证明什么、没证明什么

（`A3-node2:$W/layer_bench/`，单卡 `/dev/davinci7`，容器 `dsv41-a22-layer`）

### 3.1 已证明（可用）

| 项 | 单卡 | 主线 8 卡 | 判定 |
|---|---|---|---|
| `wo_a` `TransposeBatchMatMul [8,1,4096]×[1,4096,1024]` | 33.7 µs | **47.26 µs** | 同量级 ✓ |
| `wo_a` 2D 版 | **22.5 µs** | — | 省 11.3 µs → 40 层 **0.45 ms/step** |
| 主线实测该改动收益 | — | **0.3–0.8 ms/step** | **方向一致 ✓** |
| 2D vs 3D 数值 | `max_abs = 9.8e-4`, `cos≈1` | — | 数值等价 ✓ |

**⇒ 结构性改动（省 kernel / 合并 kernel / 换 kernel）的方向判定，单卡可信。**
而且它独立复用了 cannbot 的 `ops/torch-ops-profiler` 官方模板（`examples/layer_norm_profiler_reference/`，4 文件 640 行，改 0 行就跑通）。

### 3.2 没证明 / 不可用（必须回 8 卡）

| 项 | 为什么 |
|---|---|
| **通信** | `hcom_allReduce_` 3.0 ms/step、Engram `hcom_alltoallv_`、MoE AllGather —— 单卡没有 |
| **并发算子间的 AICore 争用** | 同一 kernel 单卡/8卡比值散布在 **0.71×–1.41×**：`wo_a` 单卡更快（无争用），共享专家/`wo_b` 单卡更慢（8 卡时它们与 gate/dispatch 并发） |
| **图模式特有现象** | capture 尺寸 padding 是 **5.9 ms/step** 的影响（见 `reports/sptok-sweep-allgather.md`），单卡 eager 完全看不到 |
| **端到端接受率/精度** | 需要真 tokenizer + 真调度 + 真 KV cache |
| **routed expert 的 `GroupedMatmulSwigluQuantV2`** | 单卡报 `AclNN_Parameter_Error(EZ1001)`（缺 modelslim 的 NZ 打包布局）—— 3.1 ms/step 的大头**目前单卡测不了** |

---

## 4. 建议的分工（三层 + 一条精度线）

| 层 | 场地 | 单轮耗时 | 判什么 |
|---|---|---|---|
| **T1 单卡** | A3-node2 chip7（或 chip6/7） | **20–40 s** | 算子级结构改动、逐算子耗时、数值等价、`rtol/atol` 判定 |
| **T2 8 卡短起服** | A3-node1 chips 8-15 | **~6 min**（MAX_SEQS=1 后） | 需要通信/并发/图模式的 ms/step 判定 |
| **T3 8 卡完整** | A3-node2 chips 8-15 | ~9 min + 测量 | 精度（GSM8K / Vision）、端到端验收 |
| **精度线**（独立、低频） | 任意空闲 8 卡 | — | 与性能线**解耦**，不阻塞 |

**关于用户提到的「dspark 问题」**：它更接近**正确性**而非精度 ——
我们已实测确认 **>16384 上下文的输出非确定**（阈值 = `candidate_topk_blocks×candidate_block_size = 16384`，
即跨层候选筛选路径的启动点）。根因分析（`reports/nondeterminism-rootcause.md`）
把范围压到两个假设：H1 块级 top-k 内核用了 UB 残留（0.45）、H2 消费端读到非本 forward 的候选行（0.30）。
判别实验已在跑（`exp_tools/cand_mode_bc.sh`：强制 `candidate_mode=3/4`）。
这条线**只能用完整服务验证**，正好落在"精度/正确性线"里。

---

## 5. 立即可执行的三件事

| # | 动作 | 预期 | 状态 |
|---|---|---|---|
| 1 | **`MAX_SEQS=1` 起服计时验证** | 起服 8m29s → **~6m10s** | 已排队（`boot_time_test.sh`） |
| 2 | **把 T1 harness 扩到能跑 MoE 部分**（解决 NZ 打包 / 用 W8A8 路径代替） | 覆盖 3.1 ms/step 的大头 | 待办（子代理已知缺口） |
| 3 | **T2 只用于「必须 8 卡」的判定**，其余一律 T1 | 迭代速度提升 10× | 纪律，需写进交接文档 |

---

## 6. 参考

| 内容 | 路径 |
|---|---|
| 起服日志（含完整阶段时间戳） | `logs/perf/cmB_1029_serve.log` |
| 单卡 harness | `A3-node2:$W/layer_bench/`（`run.sh`、`lb/`、`README.md`） |
| cannbot 单算子指南调研 | `A3-node2:$W/reports/cannbot-layer-guide.md` |
| harness 报告 | `A3-node2:$W/reports/layer-harness.md` |
| 非确定性问题 | `reports/ctx-nondeterminism.md`、`reports/nondeterminism-rootcause.md` |
| capture 陷阱 | `reports/sptok-sweep-allgather.md` §2 |
