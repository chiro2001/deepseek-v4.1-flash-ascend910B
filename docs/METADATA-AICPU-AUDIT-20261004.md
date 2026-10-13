# Attention Metadata AICPU 内核：归因与优化路径审计

**日期**：2026-10-04　**执行**：子代理 `/root/ced_die_budget`（die5；未碰 tp8k5/19210）
**产物**：`~/tmp/metadata/`（工具 / 基准 / vendor / profiler 输出）
**标注**：【实测】有运行数据；【实测·代码】有明确代码路径；【推断】由实测推导；【未确认】无证据。

---

## 0. 结论速览

### 0.1 已确证（可直接依赖）
| # | 结论 | 依据 |
|---|---|---|
| 1 | **vendor 覆盖内置**：`ASCEND_CUSTOM_OPP_PATH` 里的自定义 AICPU kernel 生效 | 四格 sentinel 对照（§1） |
| 2 | **AICPU kernel 可在本容器重编，且 zero-change 逐字节可复现** | md5 `31f7a0ef…`；`tools/rebuild_aicpu.sh`（§5） |
| 3 | **metadata 不在 aclgraph 里**：走独立 eager stream + ExternalEvent | `DeviceMetadataExecutor`（§2.1） |
| 4 | **每步 7 次调用**（2 QLI + 3 SMLA + 1 Sharedkv + 1 独立 Sharedkv），**串行** | 6 任务簇跨度 p50 **0.933 ms**（§2.3） |
| 5 | **host 侧成本 1013 ms**，其中 `GetWorkspaceSize` 占 **91%**（919.7 ms / 4396 次 / 平均 209 µs） | `api_statistic`（§3） |
| 6 | 设备侧 717 ms，**92% 来自长上下文 decode 的大簇** | 分簇统计（§2.2） |

### 0.2 已排除（负结果，不要再重走）
| 假设 | 结果 | 数据 |
|---|---|---|
| AICPU 排队/核少 | ❌ | `Task Wait` p50 **0.3–1.5 µs**；AI_CPU 总 wait 仅 13.6 ms；`sum == union` |
| AICPU 与其它 AI_CPU 争用 | ❌ | `union(AI_CPU) == union(三类 metadata)`，**没有任何其它 AI_CPU 任务** |
| s1（query 长度） | ❌ | 干净图内 s1=6 → 41 µs |
| s2（KV 长度）1→262144 | ❌ | 40–47 µs 平（§4.1） |
| batch 1→16 | ❌ | 平 |
| HBM 带宽压力 | ❌ | 无负载 41.0/45.6 vs 带宽负载 **40.0/44.5** |
| 算力（matmul）压力 | ❌ | **37.3/41.1 µs** |
| 8 个并发提交者 | ❌ | **37.2 µs** |
| 与 AI core 时间重叠 | ❌ | 忙占比 p10=p50=p90 = **0.00** |
| 跨容器 AICPU 争用 | ❌ | 服务在 die8-15 运行时，die5 三次 SMLA = **40.8 / 42.6 / 41.3 µs** |
| 参数变体（cmp/ratio/mask/layout） | ❌ | 40–55 µs |

### 0.3 尚未解释（唯一剩下的问题）
**服务里每个 metadata 任务 p50 = 158–200 µs，隔离测量 = 37–45 µs（4.4×）。**
所有可复现的设备侧因素已排除 ⇒ 剩下两个候选：
* **(H1) msprof 的 AICPU `Task Duration` 被 host 延迟污染** —— 它记录的不是纯执行时间，而是"驱动 enqueue → 完成 ack"，其中包含 host 迟到/调度
* **(H2) 服务上下文里内核真的跑得久** —— 但没有任何可复现的机制

**判决实验（已交给主代理执行）**：在 `sleep=0` 与 `sleep=1000` 两档各取一次 profile。
若 1000 档的 `Task Duration` 显著上升 ⇒ **H1 成立**（则 4.6% 与 host 那 1000 ms 是同一笔成本的两个视角，应合并优化）。
若纹丝不动 ⇒ **H2 成立**（则必须走"减调用次数"或"内核内改动"）。

---

## 1. vendor vs builtin 的优先关系【实测·决定性】

方法：给三个 kernel 注入魔数触发器（`max_seqlen_q == 7777777` ⇒ 返回 `PARAM_INVALID`），重编 → 组装 vendor → 四格对照。

| vendor | 模式 | 结果 |
|---|---|---|
| `vendor_base`（无 sentinel） | 正常 | OK |
| `vendor_base` | 魔数 | **OK**（魔数本身无害） |
| `vendor_sentinel` | 正常 | OK（重建的 .so 正常工作） |
| **`vendor_sentinel`** | **魔数** | **FAIL**（AICPU 异常 507018）← sentinel 触发 |

⇒ **`ASCEND_CUSTOM_OPP_PATH` 里的 vendor 覆盖内置。**

能力清单【实测】：
| kernel | 我们 vendor | 内置 CANN |
|---|---|---|
| `QuantLightningIndexerV2Metadata` | ✅ | **❌ 无** |
| `SparseAttnSharedkvMetadata` | ✅ | **❌ 无** |
| `SparseFlashMlaMetadata` | ✅ | ✅ `opp/built-in/op_impl/host_aicpu/libcpu_kernels.so`（已注册） |

⚠️ 修正早期结论：SMLA **内置也有一份**（我最初查错了路径）。实测 vendor 优先。
⚠️ 注入的源码已**全部还原**；镜像自带 vendor `.so` 仍是原始 `31f7a0ef…`。

---

## 2. 执行模型与成本结构

### 2.1 metadata **不在图里**【实测·代码】
`vllm_ascend/worker/device_metadata.py::DeviceMetadataExecutor`：
```python
self.stream = torch.npu.Stream()          # 独立 worker stream
...
with torch.npu.stream(self.stream):
    task.run()                            # eager host 下发
    self._external_stage_ready[...].record(self.stream)   # ExternalEvent
```
⇒ 主 aclgraph 只通过 ExternalEvent 等它，**图摊不掉这 7 次下发**。

### 2.2 双峰
| 簇 | n | 总时长 | mean |
|---|---:|---:|---:|
| 小簇 <105 µs | 887 | 59.8 ms | 64–73 µs |
| 大簇 ≥105 µs | 3509 | **657.3 ms** | 174–199 µs |
⇒ 92% 的时间来自大簇。

### 2.3 ★ 簇内结构：串行、无"首任务延迟"
6 任务大簇（n=521）按启动顺序：
| 位置 | p50 | p10 | p90 | 主要 op |
|---:|---:|---:|---:|---|
| 0 | 158.3 | 58.4 | 203.9 | QLI |
| 1 | 172.8 | 142.8 | 206.2 | QLI |
| 2 | 200.1 | 167.0 | 236.9 | SMLA |
| 3 | 199.4 | 166.5 | 241.1 | SMLA |
| 4 | 195.3 | 158.6 | 236.2 | SMLA |
| 5 | 173.6 | 141.7 | 208.1 | Sharedkv |

**簇总跨度 p50 = 0.933 ms**（≈6×155 µs）⇒ **串行**，且各位置**均匀慢**（不是"第一个慢、后面快"的 host-latency 指纹）。
⚠️ 这一条**对 H1 不利**：若纯粹是 host 迟到，应只见首任务慢。但若 host 每步都迟到（7 个 task 各自等一次），则均匀也说得通 ⇒ 仍需 sleep 实验裁决。

### 2.4 大小簇按时间演化（RLE）
| tag | 簇数 | 起(s) | 止(s) |
|---|---:|---:|---:|
| 小 | 155 | 0.19 | 2.56 |
| 混合 | 8 | 2.56 | 3.46 |
| **大** | **1034** | **3.46** | **17.67** |
⇒ t≈3.46 s 后**持续 14.2 s** 稳定在大簇 ⇒ 与**上下文长度/阶段**相关，不是随机。
（628 个 6 任务簇 = 628 个 decode 步；簇间距 p50 = 23.3 ms）

---

## 3. host 侧成本（此前所有口径都漏掉的一半）【实测】
`api_statistic` 含 **host 侧** API 计时：
| API | 总时间 | 次数 | 平均 | min | max |
|---|---:|---:|---:|---:|---:|
| `aclnnSparseFlashMlaMetadata**GetWorkspaceSize**` | 486.1 ms | 1884 | **258.0 µs** | 14.9 | 856.8 |
| `aclnnSparseAttnSharedkvMetadata**GetWorkspaceSize**` | 246.5 ms | 1256 | **196.3 µs** | 12.0 | 592.4 |
| `aclnnQuantLightningIndexerV2Metadata**GetWorkspaceSize**` | 187.1 ms | 1256 | **149.0 µs** | 13.2 | 538.6 |
| 三者合计 | **919.7 ms** | 4396 | **209 µs** | | |
| 三个算子本体（入队） | 93.6 ms | 4396 | 21.3 µs | | |
| **metadata host 合计** | **1013.3 ms** | 4396 | — | | |

两个要点：
1. `GetWorkspaceSize` 是算子本体的 **3–6 倍**，占 host 成本 **91%** —— 对 aclnn 而言它跑的是**完整 host 侧 tiling**。
2. min/max 跨度极大（13 ↔ 857 µs）、Variance 3.6e4 ⇒ **很可能每次 cache miss**（decode 期间 shape 恒定，只有数值在变）。

⚠️ **口径提醒**：host 1013 ms（5.7%）与设备 717 ms（4.1%）**是并行发生的**，**不可相加**。

---

## 4. 隔离基准

### 4.1 干净图内（输入图外预分配）
| 条件 | QLI | SMLA |
|---|---:|---:|
| 无负载 | 41.0 | 45.6 |
| HBM 带宽负载 | 40.0 | 44.5 |
| matmul 负载 | 37.3 | 41.1 |
| 8 并发提交者 | 37.2 | — |
| 服务同期运行（跨容器） | 38.0 | 40.8 / 42.6 / 41.3 |

### 4.2 ⚠️ 一条测量教训（我踩过，务必别重踩）
早前用 `bench_graph.py` 得到"负载把 metadata 拉长到 241 µs（5.8×）" —— **是假象**：
该脚本把 `mk_inputs()`（`arange`/`full`/`zeros`）**也捕进了图**，压力下变慢的是那些构造算子。
**判据**：图内测量必须**输入在图外预分配**，图内只含被测算子（`bench_graph2.py` 已修正）。

---

## 5. AICPU 重编链（已打通并固化）

* 三个算子（+`StoreKvBlock`/`VllmQuantLightningIndexer`）在**一个** `.so`：
  `.../custom_transformer/op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so`（5.4 MB）
* 构建是**普通 g++ 共享库**（非 AscendC）
* **zero-change 重编逐字节一致**：md5 `31f7a0efc7a2b89e00ef6ae45ef77883`
* ⚠️ 坑 1：`.so` 是 CUSTOM_COMMAND，5 个 `.o` 只是 **order-only 依赖** ⇒ `ninja` 只重编 `.o`、**不重链 `.so`**（报 "no work to do"），必须手工执行 `build.ninja:9669` 的链接命令
* ⚠️ 坑 2：`docker exec` 的 argv 直传路径会报 `No such file or directory`；必须让**容器内** shell 展开 `$CC`
* 脚本：`tools/rebuild_aicpu.sh`、`tools/inject_sentinel.py`、`run_locked.sh`

---

## 6. 协作卫生
* **die5 锁**：脚本 `~/tmp/metadata/run_locked.sh` 已包 `acquire || exit 1` + `trap release`
* 全部 NPU 测量已串行化；本轮结束后 `die5_lock.sh status = FREE`，无遗留进程
* 未使用 `rm -f`（一律 `mv` 到 `.bak<时间戳>`）

---

## 7. 下一步（按优先级）

| # | 动作 | 谁做 | 判据 |
|---|---|---|---|
| 1 | **sleep 三档 0→300→1000→0** 看 `[bneck] hp` | 主代理（服务侧） | 300 档 Δhp ≥1.5 ms ⇒ host 在关键路径；≤0.5 ms ⇒ 被盖住。7 task/步 ⇒ 300 µs 期望 +2.1 ms、1000 µs 期望 +7.0 ms |
| 2 | **0 档 / 1000 档各取一次 profile** | 主代理 | 1000 档 metadata `Task Duration` 若显著上升 ⇒ H1（msprof 被 host 污染）；纹丝不动 ⇒ H2 |
| 3 | 内核内相位计时（`Prepare` / `BalanceSchedule` / `GenMetadata`，文件驱动开关） | 我（编译不占 NPU） | 直接回答"40 µs 花在哪"；需服务重启时带 `V41_HC_OPP_PKG` |
| 4 | 减调用次数（`_publish_task` / `DeviceMetadataStage` 合并） | 需先看 3 的结果 | 7 次/步 → 更少 |

---

## 8. ★ 内核内相位计时（已打通并实测出数）

### 8.1 机制发现：**AICPU 内核里读不到容器的 `/tmp`**
【实测】注入文件驱动开关后再跑，日志显示 `on=0 every=20`（默认值），而文件里写着 `50`：
```
[META-PHASE-DIAG] calls=1  on=0 every=20 (文件可见性诊断)
[META-PHASE-DIAG] calls=50 on=0 every=20
```
⇒ AICPU kernel 进程**读不到**容器 `/tmp`（沙箱/命名空间限制）。
⇒ **文件驱动开关只能用在宿主侧 Python（如 `device_metadata.py`），不能在 AICPU 内核内用**。
⇒ 内核插桩必须用**编译期常量**。

### 8.2 输出通道（已验证）
`KERNEL_LOG_ERROR` 从 AICPU kernel 落到 **容器内** `/root/ascend/log/debug/device-<N>/device-<pid>_<ts>.log`：
```
[ERROR] AICPU(11847,aicpu_custom_scheduler):2026-10-04-19:46:41.387.722
        [sparse_flash_mla_metadata_aicpu.cpp:71][record][tid:11855]
        [META-PHASE] n=200 avg_us: prep=5.3 sched=4.5 gen=0.8 total=10.5
```
（含文件:行、函数、tid、时间戳 —— 完整可用）

### 8.3 ★★ 结果：**SMLA 内核本体只跑 9.5 µs**
`every=200` 恒开版，800 次调用（eager）：
```
[META-PHASE] n=200 avg_us: prep=4.6 sched=4.1 gen=0.8 total=9.5   （4 批一致）
```
| 相位 | 平均 |
|---|---:|
| `Prepare`（attr 解析 + ParamsCheck + ParamsInit） | **4.6 µs** |
| `BalanceSchedule`（全部负载均衡数学） | **4.1 µs** |
| `GenMetadata` | **0.8 µs** |
| **内核本体合计** | **9.5 µs** |

**同一次运行的口径对照**：wall = 43.6 ms / 800 次 = **54.5 µs/次**（eager）。

⇒ **量化分解**：
| 层 | 每次 | 说明 |
|---|---:|---|
| 内核本体（本插桩实测） | **9.5 µs** | 我们自己的代码 |
| AICPU 框架开销（msprof Task Duration − 本体） | **≈28 µs** | 任务建立/上下文/调度/ack |
| host 下发（eager wall − Task Duration） | **≈25 µs** | aclnn 调用 + 入队 |
| **隔离合计** | **≈54–63 µs** | 与 `bench_graph2.py` 的 45.6 µs（图内）吻合 |
| **服务实测** | **175–199 µs** | 仍有 **≈115–140 µs 未解释** |

⇒ **优化含义**：
1. **算法/相位不是瓶颈** —— 把 `BalanceSchedule` 全优化掉最多省 4.1 µs（占服务 175 µs 的 2.3%）。
2. **host 侧 tiling（`GetWorkspaceSize` 平均 209–258 µs）比整个内核本体贵 20–27 倍。**
3. 真正的大头是 **AICPU 框架开销 + host 路径**，而它们在服务里比隔离时膨胀了 3–4 倍。

### 8.4 交付物
* `vendor_phase`（`~/tmp/metadata/vendor_phase`，md5 `6d399f2c…` 首版 / 终版见 `libtransformer_aicpu_kernels_phase.so`）
  —— 用 `V41_HC_OPP_PKG=<该目录>` 即可在起服时挂载（已验 vendor 覆盖生效）
* 注入器：`tools/inject_phase_probe.py` + `tools/inject_phase_probe3.py`（终版：恒开 every=200）
* 构链：`tools/build_phase_vendor.sh`（链接 + 组装 vendor + 源码还原）
