# 用 `api_statistic` 定位 host 侧真开销（新工具，新发现）

> 2026-09-16 16:15 CST｜数据源：`logs/prof_tvA/.../api_statistic_*.csv`（140 行）
> 窗口：236.5 passes 的 dummy profile

---

## 0. 为什么这份数据重要

我此前的 CPU 竞争实验发现：
* h320 臂（100% CPU 竞争）里 **ms/step 从 33 → 81（+48 ms）**
* 但 **Engram 的相位只涨了 +2.5 ms**

⇒ **~45 ms 的 CPU 敏感损失在 Engram 之外。** `api_statistic` 正好是 **host API 计时**，能直接定位它。

---

## 1. Top-22（按总耗时）

| API | Time (ms) | Count | Avg (µs) | 每 pass |
|---|---|---|---|---|
| **`launch`** | **1360.6** | **176,926** | 7.69 | **748 次/pass，5.75 ms/pass** |
| **`aclrtSynchronizeStream`** | **925.2** | 1,320 | **700.92** | **5.6 次/pass，3.91 ms/pass** |
| **`aclrtLaunchKernelWithHostArgs`** | **887.0** | **173,178** | 5.12 | **732 次/pass，3.75 ms/pass** |
| **`aclrtSynchronizeEvent`** | **620.3** | 1,096 | **565.98** | **4.6 次/pass，2.62 ms/pass** |
| `aclnnInplaceCopy` | 292.4 | 29,638 | 9.87 | 125/pass, 1.24 ms/pass |
| `aclrtRecordEvent` | 173.6 | 24,486 | 7.09 | 104/pass, 0.73 ms/pass |
| **`aclnnGroupedMatmulSwigluQuantWeightNzV2 GetWorkSpaceSize`** | **163.6** | **900** | **181.78** | **3.8/pass, 0.69 ms/pass** |
| `hcom_allReduce_` | 161.1 | 2,240 | 71.91 | 9.5/pass, 0.68 ms/pass |
| `aclrtMemcpyAsync` | 122.1 | 19,410 | 6.29 | 82/pass, 0.52 ms/pass |
| `aclnnInplaceCopyGetWorkspaceSize` | 108.4 | 11,778 | 9.20 | 50/pass, 0.46 ms/pass |
| `aclnnInplaceFillScalar` | 87.1 | 16,512 | 5.28 | 70/pass, 0.37 ms/pass |
| `aclrtStreamWaitEvent` | 78.7 | 16,336 | 4.82 | 69/pass, 0.33 ms/pass |
| `aclnnIndexSelect` | 66.3 | 7,920 | 8.37 | 33/pass, 0.28 ms/pass |
| `aclnnIndex` | 64.0 | 3,900 | 16.42 | 16/pass, 0.27 ms/pass |
| **`aclrtGetStreamAttribute`** | 57.4 | **156,018** | 0.37 | **660 次/pass, 0.24 ms/pass** |

**总计 7199 ms / 236.5 passes = 30.4 ms/pass 的 host API 时间**
（含「等待设备」，不等于纯 CPU 工作；但**调用次数**是纯 CPU 开销的可靠代理）

---

## 2. 三个可行动目标

### 目标 1 ★★：`aclnnGroupedMatmulSwigluQuantWeightNzV2 GetWorkSpaceSize` — 0.69 ms/pass

* **900 次调用，平均 181.78 µs** —— **单次最贵的 host API**（除 sync 外）
* 它是 **workspace 大小查询**；输入 shape/权重不变时**结果恒定** ⇒ **应该可以缓存**
* `GroupedMatmulSwigluQuantV2` 每 pass 40 次
* **⇒ 查 torch_npu/vllm-ascend 里这个 `GetWorkSpaceSize` 是否已有缓存层；
  若无，加「按 (shape, dtype, weight_ptr) 缓存」**

### 目标 2 ★★：748 次 eager kernel 提交/pass（`launch` + `aclrtLaunchKernelWithHostArgs`）

* **两个 API 合计约 1480 次/pass、约 9.5 ms/pass**
* 这些是**图外**的 eager 提交（图内走 aclgraph replay，不产生 host launch）
* 大头来自：**DSpark draft（3 层 eager）+ Engram + 采样/logits 路径**
* ⇒ **减少图外算子数 = 直接减少 host 分发**

### 目标 3 ★：`aclrtSynchronizeStream` / `SynchronizeEvent` — 6.5 ms/pass

* 5.6 + 4.6 = **10.2 次同步/pass**，平均 566–701 µs
* 这些**大部分是合法的等待**（host 等设备），但**若某处能去掉一次同步就是净收益**
* ⇒ 查是谁在调（`torch.npu.synchronize()` 的调用点）

---

## 3. 次要但确定的项

| 项 | ms/pass | 说明 |
|---|---|---|
| `aclnnInplaceCopy` 125/pass | 1.24 | 大量小拷贝 |
| `aclrtRecordEvent` 104/pass | 0.73 | Engram 的 `_record_offload_use` 就在其中（实测 evt=0.105 ms） |
| `aclnnInplaceCopyGetWorkspaceSize` 50/pass | 0.46 | **同样是 workspace 查询**，与目标 1 同类 |
| `aclnnInplaceFillScalar` 70/pass | 0.37 | fill 类 |
| `aclrtGetStreamAttribute` 660/pass | 0.24 | 频次极高，单次便宜 |

**⇒ workspace-size 查询类（目标 1 + `InplaceCopyGetWorkspaceSize`）合计 1.15 ms/pass**

---

## 4. 用法（新工具，建议固化）

从 profile 目录直接读 `api_statistic`（host API 计时）：

1. 找到 `<profile_dir>/*/mindstudio_profiler_output/api_statistic_*.csv`
2. 逐行读 `API Name` / `Time(us)` / `Count` / `Avg(us)`
3. 每 pass 值 = `Count / passes`、`Time / passes`

**注意**：`Time(us)` 是**串行累加**（同一 API 所有调用的时长之和），
所以它**不是墙钟**；但**调用次数**与**单次均值**是可靠的结构指标。

**旧 profile 也能用**：`logs/prof_ag/`（真权重 32K）里同样有 api_statistic，
可以交叉验证这份 dummy 结论。

---

## 5. ★ 交叉验证：真实权重 profile 的同一分析（`prof_ag`，32K）

**结论一致，且数字更极端**（说明不是 dummy 特有）：

| API | Time (ms) | Count | Avg (µs) |
|---|---|---|---|
| **`aclrtSynchronizeEvent`** | **2936.6** | 437 | **6719.90** |
| **`aclrtSynchronizeStream`** | **2682.1** | 552 | **4858.96** |
| `launch` | 1163.0 | **108,621** | 10.71 |
| `aclrtLaunchKernelWithHostArgs` | 698.0 | **105,569** | 6.61 |
| `hcom_allReduce_` | 242.6 | 2,176 | 111.49 |
| `aclrtRecordEvent` | 167.8 | 19,484 | 8.61 |
| `aclnnInplaceCopy` | 139.5 | 13,088 | 10.66 |
| **`...GroupedMatmulSwigluQuantWeightNzV2GetWorkSpaceSize`** | **98.6** | **996** | **99.03** |
| `aclnnIndex` | 74.4 | 4,344 | 17.13 |
| `aclnnMul` | 71.9 | 1,470 | **48.93** |
| `aclnnQuantMatmulWeightNz` | 69.9 | 5,400 | 12.94 |
| `aclrtStreamWaitEvent` | 66.4 | 14,360 | 4.62 |
| `aclrtMemcpyAsync` | 54.3 | 7,839 | 6.93 |

### 5.1 两点关键差异

1. **`GetWorkSpaceSize` 在真实权重下同样是 ~99 µs/次、996 次** ⇒ **不是 dummy 特有**，
   是真实存在的 host 开销（**目标 1 确认**）。
2. **`aclrtSynchronizeEvent` 单次 6.7 ms / `SynchronizeStream` 单次 4.9 ms** ——
   比 dummy profile 高一个数量级。**这符合"真实负载下设备更忙 ⇒ host 等更久"**，
   但也说明**同步点的数量（437 + 552 = 989 次）值得审查**：每次同步都是一次潜在的
   host/设备往返。

**⇒ 目标 3（同步点）在真实负载下的价值可能比 dummy 显示的更高。**

---

## 6. ⚠️ 撤回一条来自线 3 的重大结论：「draft allreduce 收图可省 2.5 ms」

线 3 报告：`hcom_allReduce_(dynamic)` **7.47 次/前向 × 346.04 µs = 2.586 ms/前向**，
并推断「这是 DSpark draft 的 allreduce，因为是 eager 所以每次要完整 host 提交 + 通信建链」，
建议收进图，估省 **≈2.5 ms/pass**，称「比 lookup+hash+plan 加起来还大」。

### 6.1 我的独立验证：**该结论错误，高估约 27 倍**

用**严格 decode 识别**（连续 43 个锚点、相邻间隔 < 2 ms ⇒ 一个 decode step），
并只统计**每个 run 内部**（不跨中间的 prefill 段）—— 106 steps：

| 类别 | n | **次/step** | 总 ms | **ms/step** | p50 µs | p90 | max |
|---|---|---|---|---|---|---|---|
| **dynamic** | 636 | **6.00** | 9.7 | **0.0913** | **13.3** | 21.7 | 322.6 |
| **static** | 8374 | **79.00** | 111.7 | **1.0536** | 11.9 | 19.8 | 35.5 |
| **合计** | 9010 | 85.0 | 121.4 | **1.145** | | | |

（636 + 8374 = 9010 = 85 × 106 ✓ 账目闭合）

**⇒ dynamic allreduce 在 decode 下只占 0.091 ms/step，不是 2.5 ms。**

### 6.2 误差来源：**又一次 prefill/decode 窗口混淆**

线 3 的「7.47 次/前向、346 µs/次」是**全 profile 平均**（含 prefill）：
* 全 profile 均值 364 µs —— **被 prefill 的长尾撑起**
* decode 段 p50 只有 **13.3 µs**，与 static 的 11.9 µs **几乎相同**
* 长尾（p90 1.1 ms、p99 7.1 ms、max 13.7 ms）**100% 在 prefill 阶段**

**⇒ 根本不是"eager 提交慢 25×"，而是"prefill 阶段的等待"。**

### 6.3 这是同一个方法论错误的第三次

| 次 | 误判 | 真值 |
|---|---|---|
| 1 | `allreduceAicpuKernel` 60% 占用 | 全在 prefill，decode 为 0 |
| 2 | `Transpose` 1.215 ms/step | 我的补丁污染，且 decode 只有 2.79 次 |
| 3 | **dynamic allreduce 2.5 ms/pass** | **decode 0.091 ms/step** |

**根因都是同一个**：`GroupedMatmulSwigluQuantV2` 锚点在全 profile 里
**prefill 与 decode 各贡献 40 个**，所以「锚点数/40」这个分母**数的不是 decode step**。

**⇒ 纪律（已加入测量规范）**：
**任何「每步」统计都必须先用「连续 43 锚点、间隔 < 2 ms」或等价判据切出 decode 段，
并且只在每个 run 的边界内聚合**（跨 run 聚合会把中间的 prefill 算进来）。

### 6.4 附带：decode 下 allreduce 的真实总量

**85 次/step × 平均 13.5 µs = 1.145 ms/step**（dynamic 6 + static 79）

对照此前用 `dev_account` 得到的「82.5 次/步、3.0 ms/step」——
**3.0 也偏高**（同源误差）。**真实值是 1.145 ms/step。**

---

## 7. 证据

| 内容 | 路径 |
|---|---|
| api_statistic 原始 | `logs/prof_tvA/.../api_statistic_20260916074320.csv`；副本 `/tmp/api_tvA.csv` |
| **真实权重 api_statistic** | `logs/prof_ag/.../api_statistic_20260916064651.csv`；副本 `/tmp/api_ag.csv` |
| CPU 竞争实验 | `/tmp/cpu_contention.log`（base 33.27 → h320 81.37 ms/step） |
| Engram 相位基准 | `logs/perf/fq_real_serve.log` 的 `[bneck]` |
| 相位拆解 | `reports/engram-host-breakdown.md` |
