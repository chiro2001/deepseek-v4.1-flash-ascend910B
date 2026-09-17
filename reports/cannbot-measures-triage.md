# cannbot 措施逐条 triage（2026-09-16）

> 目的：把 cannbot 里与本部署（A3 8 卡、vLLM + `npugraph_ex`/aclgraph、W4A8、DSpark）相关的手段
> **逐条列出并标注状态**，确保不遗漏、也不重复做已否决的。
> 来源：`model-recommend-analysis`（22 条措施）、`model-infer-*` 各 skill、`optimize-workflow` 六阶段。

---

## 1. 状态总表

图例：✅ 已用/已达标 ｜ ⛔ 本机不可用（有实测依据）｜ ❌ 已实测否决 ｜ 🔄 待试 ｜ ➖ 不适用

### 1.1 H2D/D2H 优化（`model-recommend-analysis` §H2D和D2H优化）

| # | 措施 | 状态 | 依据 |
|---|---|---|---|
| H1 | 批量 H2D/D2H（aclnnMemcpyBatch） | ➖/🔄 | 我们的 D2H 是每步小量（`input_ids[:8]`/positions/block_table），批量化的收益面很小；但 **Engram 的 3 次 D2H 已实测为架构性代价** |
| H2 | Pinned 内存 | ✅ | Engram 的 pinned staging 已用（`offload_pinned`） |
| H3 | Embedding 层优化 | ➖ | 不适用（无重复特征输入） |
| H4 | 异步 H2D（独立 stream） | ✅ | draft 的 D2H 走 `draft_token_ids_copy_stream` side stream |

### 1.2 NN 计算优化

| # | 措施 | 状态 | 依据 / 下一步 |
|---|---|---|---|
| N1 | **算子自动融合**（`TORCHINDUCTOR_NPU_BACKEND=ascendc`） | **🔄 待试** | 文档明确"必须在 `torch.compile()` 之前设置"。我们走 `npugraph_ex`，需验证是否冲突。**下一批扫描** |
| N2 | 手写融合算子 | 🔄 | 那就是 `npu_mm_all_reduce_base`（见 §2 P1） |
| N3 | 单算子 tiling key 优化 | ➖ | 需算子团队，非用户侧可配 |
| N4 | 静态图下沉 | ✅ | 已 `cudagraph_mode=FULL_DECODE_ONLY` + `max_cudagraph_capture_size=32` |
| N5 | **多线程并行调度**（`MAX_RUNTIME_CORE_NUMBER=3`） | **🔄 待试** | 文档"仅对图模式生效"，我们正是图模式。**下一批扫描** |
| N6 | **调度线程绑核**（`CPU_AFFINITY_CONF`） | **⛔ 本机不可用** | 见 §3 —— 三种格式实测亲和掩码不变 |
| N7 | 混合调度 | ➖ | 我们的 decode shape 已固定 |
| N8 | AICPU 算子转 AICore | 🔄 | 我们有 `allreduceAicpuKernel`/`allgatherAicpuKernel`（AI_CPU）！见 §2 P1 注 |
| N9 | 静态图多流并行（`ge.autoMultistreamParallelMode`） | ➖ | GE 选项，我们是 aclgraph；**但 Eager 侧可用 `torch_npu.npu.Stream` 手动多流** → 见 §2 P5 |

### 1.3 多实例并行

| # | 措施 | 状态 | 依据 |
|---|---|---|---|
| M1 | 多实例并行 | ➖ | 我们的目标是**单流**延迟，不是吞吐 |
| M2 | AICore 控核（`ge.aicoreNum` / `set_device_limit`） | ➖ | 单实例，无实例间争抢 |

### 1.4 其它 skill

| Skill | 状态 | 依据 |
|---|---|---|
| `model-infer-multi-stream`（overlap_pct） | 🔄 **部分** | 已实测 `comm∩compute=0%`；`npu_stream_switch` 是 GE 侧 API，aclgraph 侧要用手动 Stream —— 见 §2 P5 |
| `model-infer-graph-mode` | ✅ | npugraph_ex + static kernel + FULL_DECODE_ONLY 都在用 |
| `model-infer-fusion/torch_npu_list.md` | 🔄 **P1** | 融合算子全可用但 vllm-ascend 未接，见 §2 |
| `model-infer-kvcache` | ✅ | KV 3,388,563 > 3M |
| `model-infer-quantization` | ✅ | W4A8 + mtpq 已达标（精度 198/200、Vision 23/23） |
| `model-infer-prefetch` | ➖ | 面向 memory-bound 热点，我们是通信/调度 bound |
| `model-infer-superkernel` | ➖ | 仅 GE 图 + A3 Decode，**与 aclgraph 互斥** |
| `optimize-workflow` 六阶段 | ✅ 1–5 | 并行化/FA/融合/量化/图模式均已做；阶段 6 是总结 |

---

## 2. 待试清单（按我的优先级）

### P1. `npu_mm_all_reduce_base`：把 TP allReduce 融进 matmul —— **⛔ 已实测关闭（2026-09-16）**

**结论：本机 CANN 算子包对 `ascend910_93` 没有 `MatmulAllReduce`，该融合算子物理上不可用。**

实测（A3-node2 `dsv41-a22-bench`，8 卡 torchrun，真实形状）：

```
o_proj    [8,5120]x[5120,5120]  fused FAILED:
  npu_mm_all_reduce_base: MatmulAllReduceBaseKernelNpuOpApi.cpp:276
  NPU function error: call aclnnMatmulAllReduce failed, error code is 161001
  ERR00100 PTA call acl api failed.
  Execution_Error(EZ1009): Failed to execute operator MatmulAllReduce_10. Reason:
    1. SoC version ascend910_93 verification failed.
       This SoC is not configured through the AddConfig API of the OpDef class.
    3. The operator package to which the MatmulAllReduce operator belongs is not installed.
  Check nnopExecutor != nullptr failed
sh_exp_w2 [8,2304]x[2304,5120]  fused FAILED: 同上
```

根因确认（列 `ascend910_93` 的 `ops_transformer` 内核目录）：

```
all_gather_matmul        all_gather_matmul_v2
allto_all_matmul         grouped_mat_mul_all_reduce
matmul_allto_all         matmul_reduce_scatter
matmul_reduce_scatter_v2
                         ← 没有 matmul_all_reduce
```

⇒ Python 侧绑定存在（`hasattr(torch_npu, ...) == True`），**但底层算子内核没编进这个 SoC**。
这是**环境限制**，非配置问题，**无需再试**。

**可用的近亲**：`MatmulReduceScatter`（`matmul_reduce_scatter`）确实存在。
但它产出的是**每个 rank 一个 token 分片**，要求下游是 sequence-parallel ——
我们的 `Add`（残差）需要完整 hidden，且 `use_sequence_parallel_moe` 要求 DP>1。
⇒ 在本形态下不可直接替换 all-reduce。

- **依据**：decode 里 `hcom_allReduce_` **~87 次/步**、墙钟 **2.242 ms/step**、**零重叠**；
  payload 全为 `count=40960 = 8 rank × 5120 hidden`（TP all-reduce），每层 2 次：
  - `TransposeBatchMatMul → MatMulV2 → [AR]`（attention o_proj）
  - `QuantBatchMatmulV3 → [AR] → Add`（共享专家输出）
- **现状**：`torch_npu.npu_mm_all_reduce_base` 等 5 个融合算子**全部可用**，
  但 vllm-ascend 里 `mmrs_fusion` 是**死代码**（设置+透出，0 个消费点），`npu_mm_reduce_scatter_base` 封装从未被调用。
- **收益上限**：2.24 ms/step（若融合能让通信完全藏进 matmul）。
- **验证中**：`a21_scripts/bench_mar.py` 微基准（8 卡，真实形状 8×5120×5120）。
- **注意**：融合算子内部 `comm_mode` 默认 `"aiv"`，而我们现在是 AICPU 通信核 —— **可能顺带解决 N8**。

### P2. Engram route 的 host→device 下发暴露

- **依据**：FREE 分解里 `Fill → hcom_alltoallv_` **1.392 ms/step** + `hcom_broadcast_ → hcom_alltoallv_` 0.551
  = 1.94 ms/step，占 FREE 的 30%。
- **风险**：`route 流水化`（async_op）已实测**无收益**；这条要换个做法（预取/提前一步）。

### P3. HcPre + HcPost 融合

- **依据**：2.384 + 0.606 = **2.99 ms/step**（profiled），共 160 次/步。
- **参考**：`glm5next/model.py` 的"hc_post + 下一层 hc_pre(+RMSNorm)"融合写法。

### P4. 纯 env 扫描（零代码改动）—— **已完成，三项均无收益**

**实测（`sweep_a21_env.sh`，每项独立起服，8K/32K/128K × 3 发中位）**：

| 配置 | 8K | 32K | 128K | 判定 |
|---|---|---|---|---|
| base | 34.87 | 35.10 | **37.51** | 基准 |
| `MAX_RUNTIME_CORE_NUMBER=3` | 34.82 (−0.05) | 35.24 (+0.14) | 38.02 (+0.51) | 无收益 |
| `HCCL_BUFFSIZE=2048` | 34.37 (−0.50) | 35.70 (+0.60) | 38.78 (+1.27) | 无收益 |
| `TORCHINDUCTOR_NPU_BACKEND=ascendc` | 34.32 (−0.55) | 34.97 (−0.13) | 39.71 (+2.20) | 无收益且 128K 退化 |

⚠️ 注意：这三项**无法同会话配对**（都要重启），且都在 ±0.7 ms 的噪声内（8K/32K），
只有 128K 的退化（+0.5~+2.2）超出噪声。**结论：不采用**，也不必再试。


| 变量 | 依据 | 状态 |
|---|---|---|
| `MAX_RUNTIME_CORE_NUMBER=3` | cannbot N5，仅图模式生效 | **❌ 已试，无收益**（8K −0.23 / 32K +0.15 / 128K +0.51） |
| `HCCL_BUFFSIZE=2048`（当前 1024） | cannbot 建议大 batch 可调高 | **❌ 已试，无收益**（8K −0.69 / 32K +0.60 / 128K +1.27） |
| `TORCHINDUCTOR_NPU_BACKEND=ascendc` | cannbot N1 | **❌ 已试，无收益**（8K −0.74 / 32K −0.13 / **128K +2.20**） |
| `CPU_AFFINITY_CONF` | cannbot N6 | **⛔ 已验证不可用**（§3） |

### P5. 手动多流（aclgraph 侧）

- **依据**：`comm∩compute = 0`，而 `multistream_overlap_shared_expert` 与 `enable_fused_mc2` 互斥。
- **思路**：在 eager 边界（如图外）用独立 `torch_npu.npu.Stream` 把可并行的子图挪过去。
- **难度**：高（图内无法插流），优先级低于 P1。

---

## 3. ⛔ `CPU_AFFINITY_CONF` 本机不可用（实测，2026-09-16）

文档格式（cannbot N6）：`CPU_AFFINITY_CONF=<mode>,npu<id>:<start>-<end>`，mode=1 粗粒度 / 2 细粒度。

实测（A3-node2 `dsv41-a22-bench` 容器，读 `os.sched_getaffinity(0)`）：

| 值 | torch_npu 输出 | 实际亲和掩码 |
|---|---|---|
| （未设，基线） | — | **320 CPUs，[320..639]** |
| `npu_affine:2` | ⚠️ `dcmi get affinity cpu info by device id is not supported. The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.` | **320 CPUs，[320..639]**（未变） |
| `npu_affine:2,force:1` | 同上警告 | 未变 |
| `2,npu0:320-359,npu1:360-399` | 无警告、无 PTA 输出 | **未变** |
| `2` / `1` | 无输出 | 未变 |

⇒ **`npu_affine` 被 DCMI 禁用**（二进制内另有提示
`Failed to get affinity cpu info, maybe your hdk version is too low, please upgrade it`），
显式范围格式被静默忽略。**这条在本机不可用，无需再试。**

### 3.1 附带纠正一个旧结论

此前"`CPU_BIND=1` 更慢（8K 39.86 vs 34.40）"的对比是**跨机**的
（A3-node2 的 39.86 对上 A3-node1 的 34.40），**A3-node2 从未跑过无绑核基线** ⇒ 该结论无效。
不过 `CPU_AFFINITY_CONF` 这条替代路线已被本节的实测关闭，所以不影响后续决策。

---

## 4. 证据路径

| 内容 | 路径 |
|---|---|
| 措施原文 | `src/cannbot-skills/model/model-recommend-analysis/SKILL.md:220-252` |
| 六阶段 | `src/cannbot/plugins/model-infer-optimize/workflows/optimize-workflow.md` |
| 融合算子清单 | `src/cannbot-skills/model/model-infer-fusion/references/torch_npu_API/torch_npu_list.md:17-50` |
| CPU_AFFINITY_CONF 实测 | 本报告 §3；容器内 `strings libtorch_npu.so \| grep CPU_AFFINITY` |
| 零重叠与 allReduce 拆分 | `reports/comm-compute-overlap-cannbot.md` |
| 优化空间估计 | `reports/optimization-headroom-estimate.md` |
