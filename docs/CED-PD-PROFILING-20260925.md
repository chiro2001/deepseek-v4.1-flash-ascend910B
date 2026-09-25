# A3 PD 的流级 profiling：现有数据盘点与分析方法（2026-09-25）

回答两个问题：**(1) 有没有 MULTISTREAM / DSA_OVERLAP 这几个临界开关对应的
A3 profiling 数据；(2) 能不能从中分析出流的行为。**

结论：**能力已验证（能，而且 4 份都已分析出结果，见 §3 / §4 / §7）；
但"同一角色、只改开关"的对照数据还没有** —— 现有 4 份 A3 捕获里，
P 侧两份都是 `MULTISTREAM=1 DSA_OVERLAP=1`，D 侧两份都是
`MULTISTREAM=0 DSA_OVERLAP=0`，没有同角色反差。

不过 4 份恰好构成一个 **2×2**：P 的两份在**相同开关**下差"20 层 vs 40 层"，
D 的两份在**相同开关**下差"有/无 128-token 重放"。§7 就是这个 2×2 的结果。

## 1. 现有捕获盘点（a3-21，全部是 4×144K needle）

| 捕获目录（容器内 `/opt/dsv41/results/`） | 角色 | MULTISTREAM | DSA_OVERLAP | 大小 | 已离线分析 |
|---|---|---|---:|---:|---|
| `ced_base_p_0925_181150/prof` | P（40 层） | **1** | **1** | 8.1 G | ✅ |
| `ced_prof_p_0925_171906/prof` | P（CED 20 层） | **1** | **1** | 4.3 G | ✅（见 §7.1） |
| `ced_base_d_ms0_0925_184318/prof` | D（40 层） | **0** | **0** | 811 M | ✅ |
| `ced_prof_d_0925_165058/prof` | D（CED，带 128-token 重放） | **0** | **0** | 1.2 G | ✅（见 §7.2） |

开关取值不是猜的：容器内 `vllm serve` 命令行直接写着
`"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true`（P）
与 `...:false,...:false`（D）。

另外 A2 侧还有 7 份更早（2026-09-22）的 op 级捕获
`a2/agents/SAFE_LEVERS/out/prof_*`，但那些是**单流**的局部 op 捕获
（`AI_VECTOR_CORE` × 800~4200、stream_id 恒为 13 或 47），不能用来观察多流。

**缺的那一格**：一次 `MULTISTREAM=1 DSA_OVERLAP=1` 的 **D** 捕获（或
`=0/0` 的 **P** 捕获）。要拿到它必须重启对应角色（约 6 分钟），本轮按
「先不重启实例」没有做。

## 2. 分析方法（已验证，可复用）

```bash
# 1) 起服时带 PROFILE=1（= vllm --profiler-config {"profiler":"torch",...}）
# 2) 采一段：
curl -XPOST http://127.0.0.1:18990/start_profile      # P
curl -XPOST http://127.0.0.1:18991/start_profile      # D
#    ... 发要测的请求 ...
curl -XPOST http://127.0.0.1:18990/stop_profile

# 3) 离线 analyse —— 不用停服务，也可以在别的机器上对拷贝做
docker exec <容器> python3 -c "
from torch_npu.profiler.profiler import analyse
analyse('/opt/dsv41/results/<run>/prof/<rank>_ascend_pt')"

# 4) 流级分析（本仓库工具）
python3 tools/ced_prof_streams.py <rank>_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv \
    --label "..." --compute-stream 47 --other allreduce --gap-ms 50
```

`task_time.csv` 每行一个 device task，关键列：
`stream_id`（**流号**）、`kernel_type`（AI_CORE / AI_VECTOR_CORE / MIX_AIC /
AI_CPU / SDMA_SQE / COMMUNICATION / EVENT_WAIT / NOTIFY_WAIT…）、
`kernel_name`、`task_time(us)`、`task_start(us)`。

原始数据（`PROF_*/device_N/data/`）里是 `stars_soc.data`（任务调度）、
`ffts_profile.data`、`aicpu.data`，`analyse()` 把它们解成上面的 CSV。
容器里还带 CANN 9.1.0 的 `msprof` 可做别的视角。

## 3. 已经看出来的东西（P：MULTISTREAM=1 DSA_OVERLAP=1，40 层，4×144K）

窗口 90.4 s、235 万个 task、**15 条流**。按忙时排序（节选）：

| 流 | 任务数 | 忙时 | 占窗口 | 主要类型 | **这个流在干什么**（由 kernel 名认出） |
|---:|---:|---:|---:|---|---|
| 47 | 175,996 | 81.6 s | 90.2% | AI_VECTOR_CORE / MIX_AIC | **主计算**：`DynamicQuant`、`QuantBatchMatmulV3`、`InplacePartialRotaryMul` |
| 40 | 30,024 | 80.9 s | 89.5% | EVENT_RECORD / NOTIFY_RECORD | **TP 通信**：`aiv_broadcast_bfloat16_t`、`aiv_all_gather_bfloat16_t` |
| 51 | 279,940 | 80.5 s | 89.1% | NOTIFY_WAIT_SQE / SDMA_SQE | SDMA + 通知同步 |
| 52 | 134,139 | 80.5 s | 89.0% | NOTIFY_WAIT_SQE / SDMA_SQE | 同上 |
| 50 | 624,033 | 80.5 s | 89.0% | NOTIFY_WAIT_SQE / WRITE_VALUE_SQE | 同步机器（**62 万个 task**） |
| 90 | 1,032,279 | 80.5 s | 89.0% | NOTIFY_WAIT_SQE / NOTIFY_RECORD_SQE | 同步机器（**103 万个 task**） |
| 39 | 25,920 | 79.6 s | 88.1% | EVENT_WAIT / AI_VECTOR_CORE | 计算（`DynamicQuant`、`QuantBatchMatmulV3`） |
| 36 | 23,040 | 78.8 s | 87.2% | EVENT_WAIT / AI_VECTOR_CORE | 计算（`QuantBatchMatmulV3`、`RmsNorm`） |
| **10** | 11,664 | **66.8 s** | 73.9% | NOTIFY_WAIT / **AI_CPU** | **allreduce**：`RunAicpuRpcSrvLaunchV2_allreduce` ×5,832 |
| 8 | 11,664 | 65.4 s | 72.3% | NOTIFY_RECORD / NOTIFY_WAIT | 给上面那条 allreduce 配对的通知 |
| 43 | 576 | 0.35 s | 0.4% | COMMUNICATION | **EP 通信**：`aiv_all_to_all_v_bfloat16_t`（几乎不用！） |

### 3.1 多流 overlap 是**生效**的，不是摆设

| 量 | 值 |
|---|---|
| compute（流 47 上的 AI/MIX kernel） | 78.4 s 忙（86.7% 窗口） |
| allreduce（任意流） | 67.9 s 忙（75.1% 窗口） |
| **两者重叠** | **65.2 s** |
| **allreduce 被 compute 覆盖** | **96.0%** |
| allreduce 忙但 compute 闲 | 仅 2.7 s |

⇒ 那 46.8% 的 allreduce 时间**几乎完全藏在计算后面**，不直接吃墙钟。
这修正了"看到 allreduce 占 46.8% 就以为它是瓶颈"的直觉。

### 3.2 计算流上的空洞**不是**通信造成的

流 47 上 >50 ms 的空洞只有 5 段、合计 5.7 s，其中最大的是 t=0 的 1.5 s 启动段，
之后是每个请求边界各约 1.2 s（22.7 / 45.1 / 67.5 s，对应 4 个请求之间的间隔）。
**这些空洞里 allreduce 的占用率是 0%** ⇒ 空洞来自请求之间，与通信无关。

## 4. 已经看出来的东西（D：MULTISTREAM=0 DSA_OVERLAP=0，40 层，4×144K）

窗口 90.4 s、13.6 万 task、**10 条流**（证明"关多流"≠单流，HCCL/SDMA 仍有自己的流）：

| 流 | 任务数 | 忙时 | 主要 kernel |
|---:|---:|---:|---|
| 141 | 99,348 | 0.82 s | `DynamicQuant`、`QuantBatchMatmulV3`（主计算） |
| 142 | 12,652 | 0.03 s | SDMA（**PD 的 KV 接收**） |
| 140 | 11,016 | 0.85 s | `aiv_all_reduce_bfloat16_t` |
| 47 | 10,532 | 0.89 s | `Cast`、`Fill` |
| 38 / 43 / 45 / 139 | 1,900 / 272 / 136 / 104 | 0.4 s | `ClipByValueV2`、`aiv_all_to_all_v`、AICPU |

**D 几乎全程在等 P**。逐请求看：

| 请求 | SDMA 突发 | 计算区 | D 实际干活 | 到下一个突发的间隔 |
|---:|---:|---:|---:|---:|
| 1 | 22.89 s（10 ms，2,839 SQE） | 22.95–23.21 s（0.25 s） | ~0.27 s | **22.36 s** |
| 2 | 45.26 s（9 ms，2,722 SQE） | 45.32–45.57 s（0.26 s） | ~0.27 s | **22.39 s** |
| 3 | 67.66 s（12 ms，3,634 SQE） | 67.72–67.94 s（0.23 s） | ~0.24 s | **22.31 s** |
| 4 | 89.98 s（12 ms，3,457 SQE） | 90.03–90.26 s（0.23 s） | ~0.24 s | — |

`step_trace_time.csv` 也自证这一点：`Computing 776 ms`、
`Communication 116 ms`、**`Free 66 482 ms`** / `Stage 67 374 ms`。

⇒ 端到端 22 s/请求里，**D 只占 0.27 s（1.2%）**，其余全是等 P 的 prefill；
PD 的 KV 交接本身（SDMA）只要约 10 ms 就能下发 2,700~3,600 个 SQE。

## 5. 对 A2「DRAM 取回很慢」的适用性（要点）

* 这 4 份捕获是 **A3 的 PD** 形态，**没有 DRAM KV 池**，所以不能直接给出
  A2 DRAM 取回的耗时。要那个数必须在 A2 上按 §2 采一段。
* 但 §3 的两个量对 A2 直接相关：
  1. **allreduce 是 AI_CPU kernel**（`RunAicpuRpcSrvLaunchV2_allreduce`），
     avg 11.45 ms、max 92.3 ms。A2 的 `MC2=0 FUSED_MC2=0` 意味着它**没有**
     与 matmul 融合，只是靠多流掩盖 —— 一旦 DRAM 取回把主流堵住，
     这条 allreduce 会立刻从"被掩盖"变成"暴露在关键路径上"。
  2. **流 50/51/52/90 上有 62 万~103 万个 NOTIFY_WAIT_SQE / WRITE_VALUE_SQE**。
     这套通知机器本身就是可观的开销；DRAM 路径上每一步都插通知的话，
     会按这个量级放大。
* 判定"DRAM 取回慢"的正确量是：**SDMA（或对应搬运）流的忙时/突发时长**
  与**主流空洞中该流的占用率**（本工具表 C/D 就是干这个的）。
  本次在 D 上看到的 SDMA 是 10 ms/请求量级 —— A2 上应该显著更大。

## 6. 下一步（都要重启，未执行）

1. 补 **同角色反差**（最高优先）：给 **D** 采一份
   `MULTISTREAM=1 DSA_OVERLAP=1`（乱码配置）的捕获，与现有 `=0/0` 的 D 捕获
   做流级 diff。这是回答"关多流到底关掉了什么、为什么它能修乱码"的**唯一**
   最短路径——§7.3 的推断必须靠它盖章。由于该配置会输出乱码，用**短请求**
   采集即可（只要流的行为，不需要答案对），成本 ≈ 一次 D 重启 + 几十秒采集。
   可选加做：P 的 `=0/0` 一份（P 侧从未关过多流，能顺带说明为什么 P 不受影响）。
2. A2 侧按 §2 采一段 DRAM 取回期的捕获，重点看搬运流的突发时长与主流空洞。

## 7. 补充分析：CED 臂的两份捕获也做完了（2×2 齐了）

用一次性容器（`docker run --rm`，只读挂载捕获、`--network none`、`nice -n 15`）
离线分析了 CED 的两份捕获，不需要停任何服务。四份构成完整的 2×2：

| 角色 | CED 臂 | 基线臂 | 开关 |
|---|---|---|---|
| P | 20 层（`ced_prof_p_...171906`） | 40 层（`ced_base_p_...181150`） | 两边都 `ms=1 DSA=1` |
| D | 带 128-token 重放（`ced_prof_d_...165058`） | 无重放（`ced_base_d_ms0_...184318`） | 两边都 `ms=0 DSA=0` |

四份都是**同一批请求**（4×144K needle，串行，`temperature=0`）。

### 7.1 P 侧：CED 把通信量正好减半，但"没被藏住的部分"没减半

| 量 | CED P（20 层） | 基线 P（40 层） | 比值 |
|---|---:|---:|---:|
| 采集窗口 | **45.5 s** | **90.4 s** | **1.99×** |
| 流数 | 15 | 15 | 1.00× |
| 任务总数 | 1,212,365 | 2,359,425 | 1.95× |
| 主计算流 47 忙时 | 35.19 s（占空 **77.4%**） | 81.59 s（占空 **90.2%**） | 2.32× |
| 其中 AI/MIX kernel | 33.6 s（73.9%） | 78.4 s（86.7%） | 2.33× |
| **allreduce 忙时** | 25.5 s（56.1%） | 67.9 s（75.1%） | 2.66× |
| **allreduce 调用次数** | **2,952** | **5,832** | **1.98×** |
| allreduce 被 compute 覆盖 | **95.3%** | **96.0%** | — |
| TP 通信流 40 忙时 | 34.73 s | 80.91 s | 2.33× |
| 通知流 50 / 90 任务数 | 315,869 / 522,512 | 624,033 / 1,032,279 | 1.98× |
| EP all-to-all 流 43 | 576 | 576 | 1.00× |
| compute 流空洞（>50 ms） | 5 段，合计 7.3 s（占窗口 **16%**） | 5 段，合计 5.7 s（占窗口 **6.3%**） |

三条读法：

1. **allreduce 调用次数 2,952 vs 5,832 = 正好 1/2**（40→20 层），
   通知流任务数同样 1.98× ⇒ CED 确实是在"少做一半的层"，
   而不是靠别的方式把时间挪走。
2. **窗口 45.5 vs 90.4 s = 1.99×**，与 TTFT 比（10.95 vs 21.82 s）一致。
3. **占空比从 90.2% 掉到 77.4%，空洞占比从 6.3% 涨到 16%**——
   这是 Amdahl：计算减半了，但**没被 overlap 藏住的那部分没减半**
   （5 段空洞时长几乎没变：7.3 s vs 5.7 s，而窗口缩短了一半）。
   20 层配置下 allreduce/通知的开销占比更高，**这是 CED 继续提速的下一步**。

（顺带：EP all-to-all 两边都是 576 次，说明这个模型的 EP 通信量与层数无关——
按 token 路由，不按层。）

### 7.2 D 侧：CED 的重放只多花约 1.3 s/请求，D 依然几乎全程在等

| 量 | CED D（带重放） | 基线 D（无重放） |
|---|---:|---:|
| 采集窗口 | 46.1 s | 90.4 s |
| 流数 | 10 | 10 |
| 任务总数 | 148,078 | 136,506 |
| 各流忙时合计 | **6.21 s**（13.5% 占空） | **4.79 s**（5.3% 占空） |
| 其中 SDMA 流 142 | 13,209 SQE / 0.03 s | 12,652 SQE / 0.03 s |
| 最大单流 | 流 47：1.84 s | 流 47：0.89 s |

**4 条请求下来，D 的设备总忙时：CED 6.21 s vs 基线 4.79 s**（每请求 1.55 s vs 1.20 s）。
也就是说 **128-token 重放给 D 增加约 0.35 s/请求的设备时间**，
相对 11 s/请求的端到端仍然是零头（3%）。
两个 D 臂的 SDMA（PD 的 KV 接收）都是 **0.03 s / 约 1.3 万个 SQE**，与层数无关
（KV 只传一次）。

### 7.3 对"临界开关"这个问题的直接回答

* **P 侧 `MULTISTREAM=1 DSA_OVERLAP=1` 的流行为**：15 条流，能逐条认领
  （计算 / TP all-gather / EP all-to-all / AI_CPU allreduce / 四条纯通知流）。
  overlap 是**真生效**的：96% 的 allreduce 被计算覆盖。
* **D 侧 `MULTISTREAM=0 DSA_OVERLAP=0` 的流行为**：仍然是 **10 条流**
  （HCCL/SDMA 有自己的流），但**没有任何一条承担计算与通信的重叠**——
  通信流 140 与 SDMA 142 只在自己那 10 ms 窗口里忙，其余 22 s 全空。
* **因此"关多流"在 D 上的作用可以描述为**：把"辅助流可能与主流并发读写同一批
  buffer"这件事去掉（上一份报告已定位到候选 buffer 的写入范围只有
  `candidates.shape[0]` 行、且 reset 是 no-op）。但这句还只是**合理推断**——
  要盖章仍需补一份 **D 的 `ms=1 DSA=1`** 捕获来做同角色 diff（§6.1）。
