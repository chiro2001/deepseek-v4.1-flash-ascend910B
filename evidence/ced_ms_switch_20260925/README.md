# MULTISTREAM / DSA_OVERLAP 开关：D 侧单变量 + 流级证据（2026-09-25）

## 摘要

| 项 | 结论 |
|---|---|
| 单变量隔离 | **只重启 D**、只改 `MULTISTREAM`/`DSA_OVERLAP`（P 与其余参数一律不动）：`0/0` → 144K 四针 **4/4 PASS**；`1/1` → **0/4 全乱码**，两次独立采集都复现 |
| 流结构变化 | `ms=0`：**1 条计算流**；`ms=1`：**3 条计算流** + allreduce 换流 + 多一条 AICPU 流（10 条流 → 12 条流） |
| 未证实 | **具体缺哪条同步**还没定；下述证据只能把范围收窄到"3 条计算流之间" |

## 1. 复现（唯一变量 = D 的这两个开关）

同一台 a3-21、同一份模型、同一批请求（4×144K needle、串行、`temperature=0`）、
同一端口与 `num_blocks=29076`；**P 全程没动**（它是 40 层、`MULTISTREAM=1 DSA_OVERLAP=1`）。

| 臂 | D 的 `MULTISTREAM/DSA_OVERLAP` | 第一条回答 | 判决 |
|---|---|---|---|
| A | `0 / 0` | `ZQ7K-3341`（正确） | **4/4 PASS** |
| B | `1 / 1` | `不以或少或无<｜box｜> splitNativeAbilityCapacity…` | **0/4 FAIL（乱码）** |

B 臂的乱码形态与用户报的 A2 症状一致：HTTP 200、`completion_tokens` 打满
`max_tokens`（64）、无 `finish_reason`、`u_fffd=0`（不是 UTF-8 解码问题）。

## 2. 流级对比（**步数已对齐**）

第一版对比有个坑：`ms=0` 那轮模型答对、7 个 token 就停；`ms=1` 那轮乱码、
打满 64 个 token。**两者 decode 步数差 8 倍**，直接比任务数会把"decode 更长"
误读成"开关导致工作量变化"。

所以补采了一份 **`max_tokens=8`** 的 `ms=1` 捕获（乱码不会早停 ⇒ 恰好 8 步），
与 `ms=0` 的 8~9 步对齐。两份都是 4×144K、窗口 ~90 s：

| 量 | `ms=0`（答案对） | `ms=1`（乱码，8 步） |
|---|---:|---:|
| 窗口 | 91.0 s | 89.1 s |
| **流数** | **10** | **12** |
| 任务总数 | 136,506 | 90,962 |
| 计算流 | **141** 一条：99,348 任务 / 0.82 s | **135**：46,944 / 0.36 s<br>**133**：5,120 / 0.36 s<br>**132**：5,760 / 0.37 s |
| allreduce 流 | 140：11,016 任务（COMMUNICATION **2,754**）/ 0.85 s | 134：5,248（COMMUNICATION **1,312**）/ 0.37 s |
| AICPU `batch_get` | 139：104 | 131：120 |
| SDMA（PD 的 KV 接收） | 142：12,652 | 142：15,160 |
| TP 通信 | 40：544 | 40：512 |
| EP all-to-all | 43：272 | 43：256 |

### 2.1 三条计算流各在干什么（由 kernel 名认领）

| 流 | 主要 kernel | 判读 |
|---|---|---|
| **135** | `DynamicQuant`、`QuantBatchMatmulV3`、`InplacePartialRotaryMul` | 主计算（混合） |
| **133** | `QuantBatchMatmulV3_…_24`、`RmsNorm`、`InplacePartialRotaryMul` | 注意力那条（带 rope） |
| **132** | `DynamicQuant`、`QuantBatchMatmulV3_…_0`、`DequantSwigluQuant` | MoE / 共享专家那条（带 SwiGLU） |
| 134 | `aiv_all_reduce_bfloat16_t` | all-reduce |

⇒ 开关打开后，原本在一条流上顺序执行的 attention / MoE / 主链路被**拆到 3 条流上并发**，
all-reduce 也换到独立流。

### 2.2 顺带排除一个误解

`ms=1` 的任务总数**比 `ms=0` 少**（90,962 vs 136,506），单步计算时间却更长
（1.09 s vs 0.82 s，按 32 步算 34 ms vs 26 ms）。
所以"打开多流 = 干活更多"是错的；它是把工作**重新打包**（kernel 更大更少）
并让通信离开关键路径。这与"多流本身是个性能优化"相符 —— 问题出在正确性。

## 3. 还没证实的部分（诚实边界）

**"缺哪条同步"没有定。** 要把范围再收窄，需要看 132/133/135 三条流之间的
`EVENT_RECORD`/`EVENT_WAIT` 配对图 —— 即"某条流写了 buffer、另一条读它却没有
中间的 wait 事件"。本次只做到"流清单 + 忙时 + 每条流的 kernel 归属"，
没有做事件依赖图。下一步用同一份 `task_time.csv` 可以做（
`stream_id` + `kernel_type` 里的 `EVENT_RECORD`/`EVENT_WAIT` 已经带 `task_id`，
可以重建 per-stream 时间线并检查跨流可见性）。

另外两点也要明说：

1. **两个开关是一起改的**（`MULTISTREAM` 与 `DSA_OVERLAP`），所以严格说结论是
   "这一组"，不能归给其中某一个。要分开需要两次额外重启。
2. 本证据只覆盖 **144K**。更短上下文是否也乱码没测。

## 4. 与 A2 的关系

A2 生产环境的 `inner.sh` 里正是 `MULTISTREAM=1 DSA_OVERLAP=1`，
而用户报的症状是乱码 —— 与本文 §1 的复现形态一致。
因此**A2 的乱码很可能就是这一族问题**，且它与 CED 无关（本次是标准 PD 基线复现的）。

⚠️ 但**不能反推成"关掉多流就好了"**：A2 的关键症状是"DRAM 取回时非常卡"，
那是另一件事；关多流只能去掉乱码这一项。

## 5. 文件与命令

* D（`ms=1`）64 token 捕获：`ced_base_d_ms1_0925_221830/prof/<rank0>_20260925142622665_ascend_pt`
* D（`ms=1`）8 token 捕获（步数对齐用）：`.../<rank0>_20260925145630876_ascend_pt`（62 M）
* D（`ms=0`）对照捕获：`ced_base_d_ms0_0925_184318/prof/<rank0>_20260925110101806_ascend_pt`
* 启动脚本：a3-21 `launch_d_ms1.sh`；采集：`cap_d_ms1.sh` / `cap_ms1_short.sh`
* 分析：`tools/ced_prof_streams.py`（流汇总 / kernel 归属 / 重叠 / 空洞）
  ```bash
  docker exec <容器> python3 -c "
  from torch_npu.profiler.profiler import analyse
  analyse('/opt/dsv41/results/<run>/prof/<rank>_ascend_pt')"
  python3 tools/ced_prof_streams.py <rank>_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv \
      --label "..." --compute-stream 135 --other allreduce
  ```
