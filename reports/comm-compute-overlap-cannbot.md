# 通信/计算零重叠：cannbot 的解法与我们的落地缺口

> 回答两个问题：**(1)** cannbot 怎么讲通信与计算算子的优化；**(2)** 是否用 AIV 通信
> 实测对象：A3-node1 chips 8-15，`a21_fixed_0309` 会话（静态内核已启用），32K decode 70 步

---

## 1. 事实：重叠是 0%，而且**不是**"通信没跑在 AIV 上"

### 1.1 测量

| 项 | ms/step | 说明 |
|---|---|---|
| comm | 3.301 | 集合通信区间并集 |
| compute | 27.635 | 非通信区间并集 |
| **comm ∩ compute** | **0.000** | **零重叠** |

### 1.2 AIV 通信**早就开着**

`serve_v2.sh:33` 有 `export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}`，
实测**运行中的 vllm 进程**（容器内 PID 1）环境：

```
HCCL_BUFFSIZE=1024
HCCL_OP_EXPANSION_MODE=AIV      ← AIV 通信已启用
TASK_QUEUE_ENABLE=1
```

（注意：`docker exec <c> env` 看到的是**新进程**的环境，不是 vllm 进程树的；
要看 vllm 自己的环境必须读 `/proc/<pid>/environ`。这一点容易查错。）

⇒ **所以 0% 重叠的根因不是"通信用的哪个核"，而是数据依赖**：
allReduce 消费 matmul 的输出，下一个 matmul 又消费 allReduce 的输出 —— 即使 AIV/AIC 是两个独立核，
也没有可交叠的独立工作。

---

## 2. cannbot 怎么讲

### 2.1 有专门的 comm-compute 知识库

| 路径 | 内容 |
|---|---|
| `ops/ascendc-perf-optimize/references/comm-compute/index.md` | 两种通算模式总览 |
| `.../comm-compute/pipeline_balancing.md` | 流水配平（fill/drain 分析） |
| `.../comm-compute/bound_diagnosis.md` | 瓶颈判别（**含"AIC MTE2 等待被误读为访存瓶颈"**） |
| `ops/ascendc-performance-best-practices/references/mc2/pipeline_balancing_design.md` | MC² 配平设计 |

### 2.2 两种标准模式（`index.md:101-103`）

| 模式 | 场景 | 同步方向 |
|---|---|---|
| **Pattern A**（通信后计算，如 alltoall+matmul） | **AIV 先行通信**，AIC 等数据就绪后计算 | **AIV → AIC**：AIV `CrossCoreSetFlag` → AIC `CrossCoreWaitFlag` |
| **Pattern B**（计算后通信，如 matmul+alltoall） | **AIC 先行计算**，AIV 等完成后通信 | **AIC → AIV**：AIC `NotifyComputeComplete` → AIV `WaitComputeComplete` + `BarrierAll` 对齐所有 rank |

**物理基础**：A2/A3 是 **AI Core 分离架构**（AIC 管矩阵、AIV 管向量），
`HCCL_OP_EXPANSION_MODE=AIV` 就是把通信展开到 AIV 核 —— 于是矩阵计算与通信**在核级别天然可以并行**。

### 2.3 目标值

`mc2/pipeline_balancing_design.md` 开篇：

> 将 MC² 通算融合算子从**均匀切分、通信与计算串行执行**改为**长短块配平、AIC/AIV 分离流水掩盖**，
> 通过长短块排布最大化通信与计算的重叠度，**预期通算掩盖率从 50%–70% 提升至 85%+**。

### 2.4 现成算子（`model-infer-fusion/references/torch_npu_API/torch_npu_list.md`）

| 算子 | 作用 | 对应我们的痛点 |
|---|---|---|
| **`torch_npu.npu_mm_all_reduce_base`** | **融合 mm 与 all_reduce，融合算子内部实现计算和通信流水并行** | **TP allReduce 87 次/步** |
| `torch_npu.npu_all_gather_base_mm` | allgather + matmul 融合 | if TP 切分在输入端 |
| `torch_npu.npu_mm_reduce_scatter_base` | matmul + reduce_scatter，支持 perchannel/pertoken 量化 | 同上 |
| `npu_alltoallv_gmm` / `npu_gmm_alltoallv` | MoE AlltoAllv+Permute+GMM 融合，**并与共享专家 MatMul 并行** | MoE（我们已用 `DispatchFFNCombineW4A8`） |

### 2.5 另一条思路：`Send/Recv` 替代 AllReduce

`model-infer-parallel-analysis/SKILL.md:352-356`：

> - 多流并行可以隐藏部分通信延迟（Send/Recv + Compute 并行）
> - AllReduce / AllToAll **通常在关键路径上，难以完全重叠**
> - 仓库实践：LongCat AFD **用 Send/Recv 替代 AllReduce 正是为了利用重叠**

其通信优化优先级排序：`是否跨节点 > 是否在热路径高频发生 > 原语类型 > 通信字节量 > 是否可 overlap`。
我们的 allReduce 是**节点内**（单机 8 卡）+ **每步 87 次**（热路径高频）⇒ 按此排序属于高优先级。

---

## 3. 我们这边的落地缺口（关键）

### 3.1 算子可用，但 **vllm-ascend 没接**

实测容器内 `torch_npu`：**5 个融合算子全部存在**

```
npu_mm_all_reduce_base           True
npu_all_gather_base_mm           True
npu_mm_reduce_scatter_base       True
npu_alltoallv_gmm                True
npu_gmm_alltoallv                True
```

但全树搜索：

| 搜索项 | 结果 |
|---|---|
| `mmrs_fusion` | **只有 4 处：设置（2）+ 透出（2），没有任何消费点** |
| `npu_mm_reduce_scatter_base` | 封装存在（`device_op.py:161`），**但从没被调用** |
| `npu_mm_all_reduce_base` | **全树 0 处** |

⇒ **vllm-ascend 的 mm+通信融合路径是死代码**：`ascend_forward_context.py:169` 会算
`mmrs_fusion = tp_world_size <= 8`（我们 TP=8 ⇒ True）并写进 forward context，
但**没有任何算子读它**。我们的 TP allReduce 是普通 `dist.all_reduce` / `torch.ops.vllm.all_reduce`。

### 3.2 我们实际在跑的通信

decode 窗口（70 步）里的 `hcom_allReduce_`：

| stream | 次数/步 | 说明 |
|---|---|---|
| `N/A` | 82.8 | 与 stream 79 是**同一逻辑算子的两次记录**（时长逐条相同） |
| `79` | 75.3 | 同上 |
| `38` | 7.5 | 另一类 |

按 payload 去重后：**~87 次/步**，payload 全是 `count=40960 = 8 rank × 5120 hidden`
⇒ 即 **TP all-reduce**，约**每层 2 次**（attention o_proj + MoE/共享专家输出）。
墙钟占用 **2.242 ms/step**。

profile 里**没有任何 `mm*allreduce` / `reduce_scatter` 融合算子** —— 只有
`allreduceAicpuKernel` / `allgatherAicpuKernel` 这类独立通信核。

---

## 4. 结论与可选路线

**结论**：0% 重叠是**依赖结构**导致的，AIV 通信已经开着；cannbot 给出的正解是
**把通信与计算融合进同一个算子**（`npu_mm_all_reduce_base`），让算子在内部对 K 分块流水，
从而由"跨核并行（被依赖串死）"变成"算子内流水（不被依赖串死）"。
而这个能力在本 build 里**没有被 vllm-ascend 接上**。

### 可选路线（按侵入性排序）

| # | 路线 | 侵入性 | 预估 | 备注 |
|---|---|---|---|---|
| **R1** | 给 MLA 的 o_proj / MoE 输出接上 `npu_mm_all_reduce_base` | **高**（改模型 forward，且在图内） | ≤2.2 ms/step | 需自行接线，用热更新迭代 |
| R2 | 检查是否有更新的 vllm-ascend 版本已接该融合 | 低（换镜像） | 未定 | 当前镜像 digest `2c906b38…` |
| R3 | 减少 allReduce **次数**（如 `finegrained_tp_config` 调整 o_proj TP 度） | 中 | 未测 | 比融合更简单，先试 |
| R4 | 用 Send/Recv 替代 AllReduce 做重叠（cannbot 的 LongCat 实践） | 高 | 未定 | 需重构并行策略 |

### 建议下一步

**先做 R3**（低成本、可立即验证）：查 `finegrained_tp_config` 的
`oproj_tensor_parallel_size` 能否减少 TP allreduce 的参与面；
同时确认这 87 次/步是否真的都是必要的（是否存在可合并的相邻 allreduce）。

**R1 作为中期目标**：它对应 cannbot 给出的 85%+ 掩盖率上限，是唯一能实质吃掉
那 3.3 ms/step 暴露的路子；我们已有热更新通道可以低成本迭代（但改的是图内代码，
需要重捕获）。

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 零重叠与 AIV 环境 | `logs/prof_extract/op_summary_*`、`scripts/dev_account.py`、`scripts/op_wall.py` |
| allReduce stream/payload 拆分 | 本报告 §3.2；`/tmp/op_summary_fixed.csv`、`/tmp/op_stat_fixed.csv` |
| 死代码证据 | 容器内 `vllm_ascend/ascend_forward_context.py:168-171`、`platform.py:585-624`、`device_op.py:161-180` |
| cannbot 出处 | `src/cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/{index,pipeline_balancing,bound_diagnosis}.md`、`ops/ascendc-performance-best-practices/references/mc2/pipeline_balancing_design.md`、`model/model-infer-parallel-analysis/SKILL.md:340-360`、`model/model-infer-fusion/references/torch_npu_API/torch_npu_list.md:17-48` |
| AIV 通信开关 | `delivery_20260914/assets/serve_v2.sh:33`；vllm 进程 `/proc/1/environ` |
