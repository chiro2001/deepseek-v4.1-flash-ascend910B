# DeepSeek-V4.1-Flash W4A8 —— 昇腾（Ascend）推理服务

在**昇腾 910C（A3）**与 **910B3（A2）**上拉起 DeepSeek-V4.1-Flash 的 W4A8
量化推理服务：完整补丁、一键起服、自检与验收口径。

**历史交付实测**：**单台 A3（8 卡 / 16 device）**可跑 1M 上下文的 PD 分离；
旧配置 prefill 相对全 40 层同置部署 **2.07×**（144K）。
2026-10-13 新最优栈的三形态复测见 §1.2.1，尚未全部验收，不继承旧配置结论。

> **历史基础包的优化工作（算子分析、补丁编写、性能调优、文档）均由
> `deepseek-v4.1-flash` 模型自主完成**，未经人工逐行改写。

---

## 1. 你能得到什么

### 1.1 硬件

| 形态 | 硬件 | 谁用得上 |
|---|---|---|
| **A3 单机 8+8 PD 分离 + CED** | **1 台 A3**（8 卡 = 16 device）全用 | 单机、要长上下文、要 PD 分离 |
| A3 单实例 TP8 | 1 台 A3 的 **8 个 device（= 4 块卡）** | 单机、不拆 P/D |
| A3 TP8 + DCP8（实验验收中） | 同一组 **8 个 device / 8 ranks**，不是64 ranks | 长 KV 容量；精度与性能须独立验收 |
| A2 单实例 TP8 | **1 台 A2**（8×910B3，1 卡 1 device） | A2 口径 |

**A2 与 A3 的差别**：A3 每块卡有 **2 个 die**（`/dev/davinci0..15`），A2 每块卡 1 个。
`DEVS=` / `TP=` 数的都是 **device**，不是卡 —— 详见 §2 的「术语」。

**前置**：模型权重（273 GB，见 §3.1）、`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`
底座镜像（或本仓 §3.4 的镜像包）。

### 1.2 性能（实测，A3 真权重）

#### 1.2.1 2026-10-13 新最优栈：三形态已测数据快照

a3-21 真权重；Engram开启、device index关闭、DSpark5真实草稿图、BI/custom OPP、
async；保留最优多流后逐形态验精度。冷prefill：PREFIX=0、SEQ=1、BAT=8192。
A2形态是在A3上模拟TP8同置，**不是910B3硬件实测**；CED用P8+D8，DCP8复用TP8
的同8 ranks。P只执行encoder与全局源，不可能运行依赖末层残差的draft；CED的D须真实入图。

| 形态/正式负载 | 输入/输出预算 | HTTP decode ms/step | A | HTTP decode tok/s | TTFT均值 | worker prefill wall合计 |
|---|---|---:|---:|---:|---:|---:|
| A2形态/A3模拟 v2：地火 | 2048/256 | 27.953 | 2.525 | 90.322 | 0.317 s | 1.377 s |
| A2形态/A3模拟 v2：中级编程 | 2048/自然结束≤1024 | 27.092 | 4.986 | 184.025 | 0.314 s | 1.389 s |
| A2形态/A3模拟 v4：地火 | 2048/256 | 28.071 | 2.525 | 89.941 | 0.310 s | 1.333 s |
| A2形态/A3模拟 v4：地火 | 32768/256 | 29.595 | 2.902 | 98.052 | 4.733 s | 36.927 s |
| A2形态/A3模拟 v4：地火 | 147456/256 | 32.382 | 3.197 | 98.743 | 22.988 s | 180.522 s |
| A2形态/A3模拟 v4：中级编程 | 2048/自然结束≤1024 | 27.219 | 4.986 | 183.167 | 0.312 s | 1.425 s |
| A3 TP8+DCP8 v3：地火 | 2048/256 | 54.736 | 2.646 | 48.340 | 1.038 s | 1.786 s |
| A3 TP8+DCP8 v3：地火 | 32768/256 | 56.235 | 2.731 | 48.563 | 16.339 s | 129.867 s |
| A3 TP8+DCP8 v3：地火 | 147456/256 | 58.337 | 2.931 | 50.243 | 74.471 s | 592.728 s |
| A3 TP8+DCP8 v3：中级编程 | 2048/自然结束≤1024 | 53.575 | 4.873 | 90.949 | 1.034 s | 1.862 s |
| A3 CED P8+D8 | 同负载 | 待测 | 待测 | 待测 | 待测 | 待测 |

已有指标的每行实际8条串行请求、clean 8/8；两种已测形态的编程均独立隔离评分8/8题、128/128用例。
三元组来自同一HTTP decode窗口与真实worker轮数，满足`tok/s=1000×A÷ms/step`；
异步原始worker token IDs不可用，不编造逐轮token计数。prefill列是**8请求合计**的
worker调度→末prompt CPU bookkeeping wall，不是纯设备kernel时间；TTFT是端到端口径。
含后续metrics RPC等待的server-fenced ms/step单独归档，不能与HTTP tok/s混用。

目前完整通过本轮验收的是 **A2形态/A3模拟 v4**。为避免把8请求合计误当成单请求
prefill延迟，其最新预填充数据换算如下；均值为原始wall合计÷8，速率为
8请求实际输入tokens合计÷原始wall合计，**不是由TTFT推导，也不是纯NPU kernel吞吐**。

| A2 v4正式负载 | worker prefill wall均值/请求 | worker wall口径prefill tok/s |
|---|---:|---:|
| 地火2048/256 | 0.166602 s | 12,292.791 |
| 地火32768/256 | 4.615818 s | 7,099.066 |
| 地火147456/256 | 22.565253 s | 6,534.649 |
| 中级编程2048/自然结束≤1024 | 0.178076 s | 11,500.736 |

| 形态/候选 | GSM100（严格100/100门） | Vision23 | 147456长针 | 当前结论 |
|---|---:|---:|---:|---|
| A2形态/A3模拟 v2 | 100/100 | 23/23 | 待补 | 2K/编程/基础精度通过；长上下文待补 |
| A2形态/A3模拟 v4 | 100/100 | 23/23 | 4/4 | 本配置精度、编程及地火三档验收通过 |
| TP8+DCP8 v3 | **99/100，FAIL** | 23/23 | 4/4 | **整体严格精度未达标** |
| TP8+DCP8 v6 | 待测 | 待测 | 待测 | 有界定序归约CPU候选通过；实机窗口待释放 |
| CED P8+D8 | 待测 | 待测 | 待测 | 旧canary失败；v4设备回归待测，无有效性能数据 |

DCP唯一GSM错题在同服务/同512预算的一次诊断重复中答对且正常stop，但**不覆盖**
正式99/100结果。后续定序归约候选尚未上设备，不宣称已修复。A2 v2有旧route探针，
新无探针长上下文窗口另测；跨进程、跨形态数据不视为严格单变量A/B，也不直接与下面
128K/1K或旧CED数字相减。DCP短负载实测慢于同置，不能只以KV容量潜力宣称更快。
新A2 v4关闭旧route探针、修正末prompt计时边界，并在A3的0–7组重新测量；
只有一处独立worker物理guard派生，模型/SDK/图/BI/async检查保留。它不继承v2精度，
也不作为与8–15组的严格同卡A/B；冷canary没有混入正式8请求结果。
A2 v4的GSM/Vision原答已按冻结评分函数独立重评分；四针实际输入147435–147438
tokens，全部HTTP200/stop，原始请求/响应SHA已复核，不以runner汇总中缺失的
finish_reason推断截断。见[精度审计收据](experiments/delivery_matrix/evidence/a2sim_tp8_v4/quality/strict_quality_audit.md)。
A2 v4地火2K/32K/147456每档正式8条，实际usage输入均精确达目标，全部HTTP200/length/clean；
主代理独立复核gzip双SHA、8rank图计数、请求tag/fence与三元组换算。
三档的完整性能、计数与原件SHA见
[地火三档实测索引](experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/DIHUO_RESULT_INDEX_20261013.md)，
编程性能与独立128用例评分见
[编程正式轮结果](experiments/delivery_matrix/evidence/a2sim_tp8_v4/CODE8_RESULT.md)。

CED首canary在实际KV传输/128-token重放后，D侧`_pool_kernel`出现MTE/SMMU故障，
没有有效输出或性能数据；[失败证据](experiments/delivery_matrix/evidence/ced_v3b_canary_failure_20261013/failure_diagnostics.md)
完整保留。新的四处masked残余地址clamp候选通过CPU负控，尚未实机验收或证明故障因果。

完整口径、配置、原件SHA与失败史见
[交付矩阵状态](experiments/delivery_matrix/STATUS.md)及
[DCP实际质量更正收据](experiments/delivery_matrix/evidence/dcp8_v3/quality/strict_quality_receipt_corrected.json)。
可复现入口在`experiments/delivery_matrix/`，当前仍是固定SDK/最优源快照的实验适配器，
不是已经完成的通用发布包。CED单侧DRAM卸载、共享Engram mmap与KV唯一存储为下一阶段；
已有离线mmap组件不等于真实TP8/PSS/容量验收通过。

截至已归档现场快照：A2 v4已完成验收后受控停服，项目容器保留；新CED/DCP未启动。
外部`fyf_1009`的`rain_fusion_att`先后占physical5/0/1/9，影响P、D两组，未操作外部容器。
CED新可写jobs配方和[DCP v6 CPU配方](experiments/delivery_matrix/evidence/dcp8_v6_cpu_prep_20261013/CPU_PLAN.md)
已审核，等待芯片窗口释放；尚不能宣称三形态全部达标或DRAM容量交付完成。

#### 1.2.2 2026-10-13 chip8–15 算子核查：当前 TP8 基线

本轮只核查既有算子优化，不恢复三形态/DRAM大目标。a3-21 physical8–15 的完整 TP8
保持 Engram开启/index0、DSpark5 draft图、BI/async/双流、SEQ1/PREFIX0；用历史 A2v4 的
**同八条冻结输入消息**校准，1条排除的warmup + 8条串行2048/256。

| 时间/窗口 | physical chips | HTTP ms/step | A | HTTP tok/s | TTFT均值 | worker prefill wall均值/请求 |
|---|---|---:|---:|---:|---:|---:|
| 2026-10-13 历史 A2v4/A3模拟参照 | 0–7 | 28.071337 | 2.524752 | 89.940586 | 0.310032 s | 0.166602 s |
| 2026-10-13 当前 A3 TP8 base2校准 | 8–15 | **28.634375** | **2.524752** | **88.172083** | **0.321815 s** | **0.198948 s** |

主代理从8个原始请求重算：HTTP200/length、usage2048/256/2304、clean、prompt/messages
SHA均8/8通过；**输出文本SHA也8/8等于历史**。每rank实际808 decode rounds，2040有效decode
tokens；worker prefill wall合计 **1.591584930 s**，该wall口径 **10,294.141 tok/s**。
server-fenced另为 **30.292077 ms/step**，含metrics RPC等待可能性，不与HTTP吞吐混用。
新旧HTTP步时约+2.01%、吞吐约−1.97%，但这是换物理组/换进程的同输入趋势，**不是单变量A/B或算子收益**。

八rank有实际 draft ACLGraphWrapper/FULL 包装与带3层attention metadata的捕获日志；测量期各rank
proposal selector计数为808 graph/8 eager。selector不等于逐次底层replay事件，未新增低层profiler。
[独立raw审计](experiments/operator_stack/evidence/operator_revalidation_20261013/baseline2_client/primary_raw_audit.json)、
[结果与口径](experiments/operator_stack/evidence/operator_revalidation_20261013/baseline2_client/RESULT.md)及
[draft图证据](experiments/operator_stack/evidence/operator_revalidation_20261013/draft_graph_evidence/README.md)已保留。

该历史 TP8 服务入口为 `a3-21:18960`（localhost绑定），SID `3cbd844d9f684c43867b4b85c9c78ebe`；
配置指纹 `79513a0f…0da4319` 与计划一致，实际运行身份另由SID/八rank/PID映射绑定。
最终只读快照（2026-10-13 12:15–12:17，Asia/Shanghai）确认目标容器running/restart=0、
端口LISTEN、physical8–15 Health均OK，八个worker的NSpid/cgroup均匹配目标容器；
两个身份GET各一次通过。该快照窗口内服务未重启或停止，未追加生成请求，见
[最终在线快照及SHA清单](experiments/operator_stack/evidence/operator_revalidation_20261013/final_live/final_live_snapshot_20261013.md)。
保留 `a2-tp8-a3sim` 适配器form只是复用TP8入口，**不是新的910B硬件或三形态验收**；本轮不重跑GSM/Vision/编程。
此后为下述原生 DP 矩阵受控停掉 TP8；历史在线快照不代表当前服务状态。

#### 1.2.3 2026-10-13 八 chip 原生 DP 矩阵（实测进行中）

范围是 a3-21 physical8–15：DP2TP4、DP4TP2、DP4TP2DCP2，三者均 `DP×TP=8`，
DCP 在 TP 内划分，EP 保持跨八 rank。Engram 开启；`V41_ENGRAM_DEVICE_INDEX=0`
表示关闭 device-index 快路径，**不是选择 NPU ordinal 0**。保留 DSpark5 draft图、BI、async。

| 拓扑/候选 | 启动/质量/性能状态 | 自动 profile KV backing/物理 rank | 每 DP 逻辑 token 等价容量 |
|---|---|---:|---:|
| DP2TP4 v3，util0.85 | 两副本已初始化；唯一 DP0 canary HTTP输出完整，但旧测量校验报 FAIL；无正式质量/性能结论 | DP0 6,936,860,672 B；DP1 6,936,319,744 B | 启动日志 DP0 849,714；DP1 849,648 |
| DP2TP4 v4 | 权重加载/worker-ready 阶段初始化失败，未到新增 RPC；无可用 API，未发 canary | 待实测 | 待实测 |
| DP2TP4 v5 | 加载线程16、host NUMA interleave0–3的恢复候选；仅 CPU 验证，外部占卡未启动 | 待实测 | 待实测 |
| DP4TP2 | 独立候选准备，未启动 | 待实测 | 待实测 |
| DP4TP2DCP2 | 独立 Engram/DCP/draft 接线候选，仅 CPU 验证 | 待实测 | 待实测 |

DP2TP4 v3 的两副本分别有 12,824/12,823 个 global block IDs；启动日志对
196,608-token 请求给出 4.32x KV 并发等价容量。该数值含多 KV group 账本，
TP/DCP rank 不再作为逻辑容量倍数；SEQ1 仍只允许每副本一个活动请求。
这里记录的是 util0.85 的实际自动分配池及启动账，不是长请求成功或全显存上限。

旧 canary 四 TP 都记录 135 verifier rounds、135 graph calls、1 eager call，HTTP200、
完整 DONE、usage2048/256/2304、clean=true；DP1 没有请求。旧 reducer 把 prefill 初始提案
和 decode 的 eager 总数一并要求为零而拒绝，因此不发布该 canary 性能数字。
v4 增加来自真实 counter.requests 的 phase proof，要求每个 decode round 实际入图且
无 eager fallback，initial prefill 单独对账；原 FAIL 保留，不因诊断改写。

v3 已受控停止，v4 的 `run.exit=1`、engine/ASGI ready 均为 false；目标容器仍运行，
不能把容器在线当成推理 API 在线。故障发生在权重加载阶段：DP1 父进程先进入 shutdown，
其后三个 worker 的 `terminated` 及 DP0 的 `BrokenPipe` 属于退出链后果；
DP1/TP1/EP5 的日志提前截止且无 Python traceback。后续只读 kernel 日志确认：
08:23:20 UTC，目标容器的 host PID1509209 被 NUMA node6 的 `global_oom` 杀死，
约束为 `CONSTRAINT_MEMORY_POLICY,nodemask=6`；因未保留完整 PID 对应表，不能精确
将该 host PID 指认到某个 TP rank。事件的触发 `cpuset` 属于外部容器，而被杀的
`task_memcg` 属于本项目；不能由此反推本项目 worker 自己设置了 node6 绑定。
全机可用约1.48 TiB不代表每个 NUMA 节点有余量；一次复查 node6 free约6.3 GiB，
但 free 也不等同于 available。恢复需先核内存策略与外部 OOM 风险，不清 pagecache。

后续 CPU-only 内存策略探针已验证 interleave0–3可执行、1MiB页面均匀分布；
v5 保留原数学配置，仅降低 loader线程并显式绑定 host 分配策略，加载前及 describe
校验实际策略，失败不回退。最新 CPU 回归148 passed /349 subtests；不是设备验收。
最新 chip8–15 被外部 `dsv4-flash-tp8ep8-npu4-7-20261013` 占用（每 chip约55GiB），
因此未启动 v5，未运行 NPU探针，不操作外部任务。见[故障及恢复边界](experiments/dp_matrix/evidence/V4_OOM_AND_RECOVERY.md)。

新精度门、地火/编程单流与副本并发、长请求验证均须取得各自新 SID 的原始收据。
当前状态与复现入口见[原生 DP 实验](experiments/dp_matrix/README.md)。

#### 1.2.4 历史交付配置（不同口径）

**PD 分离 + CED vs 全 40 层同置**（同机同请求，唯一变量是 P 的层数）：

| 上下文 | prefill tok/s（CED） | 全 40 层基线 | 比 | TTFT（CED） | TTFT（基线） |
|---|---:|---:|---:|---:|---:|
| 32K | **12,536** | 7,219 | **1.74×** | — | — |
| 144K | **13,676** | 6,599 | **2.07×** | **10.95 s** | 21.8 s |
| 1M | **9,897** | 3,557 | **2.78×** | **101.2 s** | 280.3 s |

**decode**（同一批请求，唯一变量是 P 的层数）：两臂几乎相同 —— 25.3 / 26.8 / 37.9 ms/step
（CED）vs 25.4 / 26.8 / 37.4（基线）⇒ **CED 只改 prefill，decode 不退化**。

**单机同置形态**（TP8+EP8、128K 上下文、单流独占、全补丁）：

| 项 | 值 |
|---|---|
| 128K 单流 decode | **30.2 ms/step**（8 发中位，区间 29.8–31.2；最好一发 26.9） |
| 128K 单流吞吐 | **90.1 tok/s**（另一臂 85.0；峰值 110.5 **不可交付**，见 §6） |
| 累计优化 | 未打补丁 39.10 → 全补丁 **31.39 ms/step（−19.7%）** |
| （另一口径）设备直索引入图 | 并发 1 时 29.5 → **28.4**、并发 4 时 35.3 → **32.1 ms/step** —— ⚠️ 这是 1K prompt 口径，**不要与上面 128K 那行直接相减** |
| KV 容量 | **4,010,585 tokens**（A3 交付默认：`BAT_TOKENS=2048` + `V41_KV32_REPACK=1` + `GPU_UTIL=0.92`；= 1M 上下文 × 3.82 并发。要旧口径/更快的 prefill：`BAT_TOKENS=8192 V41_KV32_REPACK=0` ⇒ 2,987,727 tokens，见 §4） |

**并发吞吐**（A3、1024 token prompt、256 输出、`MAX_SEQS=64 PREFIX=1`）：

| 并发 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 单流 tok/s | 90.3 | 89.2 | 80.0 | 59.3 | 42.5 | 31.3 | 20.2 |
| **总吞吐 tok/s** | 87.1 | 151.5 | 237.7 | 324.4 | 432.6 | 583.9 | **719.5** |

### 1.3 精度

| 项 | 结果 |
|---|---|
| **CED-PD 验收** | **21/21 通过**（144K/1M 四针、流式、多轮、缓存命中） |
| 四针答案 | 与 `SPEC=0` 交付口径**逐字节相同** |
| Vision | **23/23** |
| GSM8K | **198/200**（另两次 199、197） |

### 1.4 支持的功能

| 功能 | 状态 |
|---|---|
| **1M 上下文**（`MAX_LEN=1048576`） | ✅ 含 1M 四针与多轮验收 |
| **PD 分离 + CED**（单台 A3） | ✅ 见 §3.3 |
| DSpark 投机解码 | ✅（`SP_TOKENS=5` 或 7；**低并发推荐**，见 §6） |
| Engram（V4.1 的记忆模块） | ✅ INT8 表常驻 DRAM；可进一步走设备直索引 |
| Vision | ✅ |
| 前缀缓存 | ✅（服务端计数器为准；代理不透传 `cached_tokens`） |
| 自动工具调用（tool call） | ✅ |
| codex 直连（Responses API） | ✅ 需先跑 §3.6 使能补丁 |

---

## 2. 术语：**卡 ≠ die** —— `DEVS` 数的是 device

> ⚠️ 本文档里凡出现 `DEVS=`、`TP=`、"芯片号"、"卡号" 的地方，
> 数的都是 **`/dev/davinciN` 的编号（= die）**，**不是物理卡**。

| 机器 | 物理卡（`npu-smi` 的 `NPU` 列） | **device / die** | 关系 |
|---|---|---|---|
| **A3（910C）** | **8 块** | **16 个** | **1 块卡 = 2 个 die** |
| **A2（910B3）** | 8 块 | 8 个 | 本包按 8 个 device 使用 |

**在自己机器上核对**（两条都不写任何东西）：

```bash
# 物理卡数（去重后的 NPU 号）
npu-smi info | grep -oE '^\| [0-9]+ +Ascend9[0-9]+' | awk '{print $2}' | sort -un | wc -l
# device / die 数（= DEVS 的可选范围）
ls /dev/davinci[0-9]* | wc -l
```

⇒ **A3 上 8 个 device = 4 块卡**；PD 分离用满 16 个 device = **8 块卡全用**。

---

## 3. 快速开始

### 3.1 拿权重

```bash
pip install modelscope
modelscope download chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend --local-dir <目录>   # 200+ GB
# ★ 下完必须重组 Engram（分片不能直接起服）—— 脚本随权重一起下载
```

### 3.2 A3：单实例 TP8（不拆 P/D）

```bash
bash tools/list_chips.sh                    # ① 只读：看哪些 device 空着、属主是谁
DEVS="8 9 10 11 12 13 14 15" \              # ② 8 个 device = 4 块卡
  MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
  bash scripts/serve_a3.sh
```

`DEVS` 是**用户输入**：哪 8 个 device 能用取决于当前谁在跑什么，脚本不替你判断。
默认会**拒绝已被占用的 device** 并打印占用进程（确实要带占用起服务才加 `ALLOW_BUSY=1`）。

### 3.3 A3：单机 **8+8 PD 分离 + CED**（推荐）

P 只跑 20 个 encoder 层 + layer-20 全局源投影；D 做 **128-token 有界重放 + 全 40 层**。

```bash
export MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq
bash deploy/a3-ced-pd/launch/serve_p.sh        # die 0–7（= 卡 0–3）→ :18990
bash deploy/a3-ced-pd/launch/serve_d.sh        # die 8–15（= 卡 4–7）→ :18991
bash deploy/a3-ced-pd/launch/serve_proxy.sh    #           → :18992（客户端连这个）
bash deploy/a3-ced-pd/launch/smoke.sh          # 144K 四针冒烟
```

也可以直接下载**已构建好的镜像包**（3.8 MB，不需要仓库）：
[Release `a3-ced-pd-v3`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/releases/tag/a3-ced-pd-v3)
—— 或看 [`deploy/a3-ced-pd/README.md`](deploy/a3-ced-pd/README.md) 的完整硬门清单。

### 3.4 A2：单实例 TP8

```bash
bash tools/check_checksums.sh               # 发包前（10 秒，不起容器）
bash scripts/build_image.sh                 # 烘焙补丁，产出 dsv41-a2:v8（约 10–20 min）
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/serve_a2.sh
```

⚠️ **用发布包里的 `scripts/serve_a2.sh`，不要用镜像里那份** —— 后者是**构建时**的快照，
可能比包旧。脚本每次起服都打印自己的路径 + 版本 + md5，报障时先看这一行。

### 3.5 起服前/后自检

```bash
bash tools/selfcheck_pkg.sh                 # 起服前（30 秒，不起容器）：19 组自检
DRY_RUN=1 ... bash scripts/serve_a3.sh      # 干跑：只打印将执行的命令，不碰 docker
PORT=8020 bash tools/attach_test.sh         # 服务已在跑时（不起容器、不删容器）
# 完整验收：8K/32K/128K + Vision + GSM8K
```

起服后**必须**核对这几项（不看就别相信结果）—— 完整清单见 §7。

---

### 3.6 让 codex 直接连本服务（可选）

本包的 `tokenizer_mode=deepseek_v41` 只认 **chat-completions** 的块词汇表，
而 **codex 等 OpenAI 客户端**发的是 **Responses** 词汇表 ⇒ 直连不可用，
且**其中两条是静默的**：

| 缺陷 | 症状 |
|---|---|
| `input_text` 被渲染成字面量 `[Unsupported input_text]` | **HTTP 200**，但**用户的话根本没进模型** |
| codex 放系统指令的 `developer` 角色要求 content 非空 | **HTTP 500** |
| 控制 token 可从正文注入 | 可**伪造轮次边界** |

```bash
docker exec <容器> bash /opt/dsv41/tools/enable_codex_responses.sh on      # 幂等，带备份/回滚
docker exec <容器> bash /opt/dsv41/tools/enable_codex_responses.sh status  # PATCHED / STOCK
```

⚠️ **改的是容器可写层 ⇒ 要重启服务才生效**；容器重建会丢，重跑即可。
重启前**务必确认没有残留进程**（`VLLM::EngineCore` / `VLLM::Worker_*` 的进程名里
**没有** `vllm serve`，`pkill -f "vllm serve"` 杀不到），否则起服会卡在
`rtsMallocHost 207001`。**A3 真机实测通过**：单轮文本、工具调用、图片、多轮、子代理。

## 4. 关键调优旋钮

用户最常需要调的六个。全部有实测依据，**改一个测一个**。

| 旋钮 | 默认 | 调它会发生什么 |
|---|---|---|
| **`BAT_TOKENS`** | **2048**（A3 交付默认，2026-10-14 起） | ★ **容量杠杆**（prefill 分块大小同时决定 activation 峰值）。2048 把 activation 峰值从 3.21 GiB 压到 0.79 GiB ⇒ 自动 profiling 多给约 9,000 块 KV（A3：2.99M → 4.01M tokens，**+34.2%**）。**必须与 `V41_KV32_REPACK=1` 成对**：这份增益会把池推过 32 位块边界（33,994 块 > 上界），高块号回绕 ⇒ 长文**静默读错**（实测：越界组 24~25% 请求输出级偏离；A3 负控 12/12 FAIL 的池正是越界 2,342 块）；重排把上界 29,076 → **32,768 块**（+12.7%）才装得下，且重排载荷自带 `[V41-KV32-CAP]` 自动夹上限（小显存机器给得少就不夹、不会 OOM）。实测 A3：**4,010,585 tokens、长文针 24/24**。**旧说法「机制是 chunked prefill 每刀约 2% 偏离」已作废**（见 `docs/KV32-OVERFLOW-DOSE-RESPONSE-20261006.md` §6）。代价：prefill 分块更细（8K **6,478** vs ~7.2K tok/s，−11%）、首 token 更慢（300K 约 79 刀）。**要 prefill 优先**：`BAT_TOKENS=8192 V41_KV32_REPACK=0` ⇒ 2,987,727 tokens、prefill 最快。|
| **`GPU_UTIL`** | **0.92** | 调到 0.94 能多约 **+9% KV**（把 activation 余量也吃掉），但**长 prompt 首 token 从 1.1 s 涨到 8 s**（实测 6~7×）。这是"KV 容量换 prefill 速度"的主动取舍；它与 `BAT_TOKENS` 是**两条独立杠杆**，但**方向相反**（降 BAT 拿容量、升 GPU_UTIL 拿容量，两者都伤 prefill）—— 交付默认已用前者，别再叠加后者 |
| **`V41_KV32_REPACK`** | **1**（A3 交付默认，2026-10-14 起） | **KV32 slot 重排**：把 ratio-1 源的 index 平面挪进别的 slot 空闲区 ⇒ 四槽页步长全 131,072、池 stride 540,928 → **524,288**、块上限 29,076 → **32,768（+12.7%）**。它是 `BAT_TOKENS=2048` 的**必要配套**（不重排则 33,994 块越过 32 位边界被守卫拒绝起服）。载荷 `patches/files/deepseek_v41.repack.py` 同时带 `[V41-KV32-CAP]` 上限语义 ⇒ **不要手动钉 `KV_CACHE_MEMORY_BYTES`**（钉死会在小显存机器上 OOM）。已验证：A3 TP8 DCP1 **4,010,585**；DCP8 **18,864,624（+10.1%）**、长文 24/24。A2 上**未实测**（serve_a2.sh 默认仍关）。详见 `docs/KV32-SLOT-REPACK-A3-VERDICT-20261006.md` |
| **`DRAFT_GRAPH`** | **1**（CED-PD 形态）／0（其它入口） | 投机解码入图。CED-PD 自 2026-09-27 起默认 1；其它入口仍是 0，**低并发推荐显式开**。两种口径都必须同时有 `DSPARK_GRAPH_CAPTURE_METADATA=1`，否则**静默失效**（`A≈1.0` 而 ms/step 反而更好看） |
| **`PREFIX`** | **1**（CED-PD） | 前缀缓存。**仍然取决于业务是否高度复用前缀** —— 若输入输出比很大且命中率低（实测某负载仅 3.52%），`PREFIX=0` 反而 TTFT −14.6%。CED-PD 自 2026-09-27 起默认开；做无缓存性能对照时必须显式 `PREFIX=0`，且**不要与开缓存的 ms/step、接受长度混比** |
| **`SPEC_MODE`**（CED-PD 的 D 侧） | **on** | 推测解码三档：`on` 全开（固定 K）/ `off` 全关 / `dynamic` 按并发切 K（并发 1 走 K=7、≥2 走 K=0）。**高并发吞吐建议 `off`**：并发 4 时 DSpark 把 ms/step 从 28.2 抬到 41.1（1.45×），换接受长度 A≈2.4 —— 收益集中在接受长度高的请求上，**成本由全批承担**。`dynamic` 是高风险档（豁免了上游一道保护），必须用 144K/1M 正确性探针验收。详见 `docs/CED-PD-SPEC-MODE-20260928.md` |
| **`SPEC`** | 1 | 旧写法（等价 `SPEC_MODE=on`/`off`）。非 CED-PD 入口仍用它；CED-PD 的 D 侧请优先用 `SPEC_MODE` |
| **`ENGRAM_WKV_TP`** | **1**（A3 起服默认，2026-10-05 起） | engram gate 的 `wkv` 投影**按输出维分片 + all_gather**，替代原先每卡复制的 315 MB 权重。实测：**−0.57 ms/步（2.2%）** 且 **KV 容量 +5.9%**（2,821,337→2,987,727）；每个输出元素仍由一次 matmul 决定（语义等价），144K 验收 11/11。关掉它（`=0`）会退回旧行为，并发与容量都会变差。详见 `docs/ENGRAM-WKV-TP-20261005.md` |
| **`ENGRAM_PAD_SKIP`** | **1**（A3 起服默认，2026-10-05 起） | engram 的 padded 缓冲只清零「会被真正读到」的行。实测 `ZerosLike "8192,6144"` **1.86→0 次/步**（省 ~58 µs/步）；模型只读 `lookups[:n]` 而清零范围 ≥ n，**零精度风险**。详见 `docs/ARMH-MECHANISM-AUDIT-20261005.md` |
| **`CPU_BIND`** / **`DROPCACHE`** | A3: 0 / 0 | **A3 共用机必需**：`CPU_BIND=0` 关掉内部 NUMA 绑核（否则目标节点满时 `migratepages` 内核态空转、服务永不就绪、`docker stop` 都停不下来）；`DROPCACHE=0` 不清整机 page cache（会打到别人） |

**`MAX_SEQS` 不是性能杠杆，是并发容量**：32→64 在并发 ≤32 时只有 ±1%；
只在并发 64/128 时 +8.7%/+7.2%，而此时单流掉 34%。聚合吞吐封顶约 259 tok/s
（A2 口径）—— **上界是算力，不是槽位数**。

---

## 5. 与上游的差异（2026-09-26 重新核实）

### 5.1 上游现在到什么程度

**上游仓 `vllm-project/vllm-ascend` `main`**（核实于 `1e9e03bc5`）：

| 项 | 上游状态 |
|---|---|
| V4.1 框架支持 | ✅ **已合入**（PR #16544，2026-09-18） |
| Engram host offload | ✅ **已合入**（同一 PR） |
| 验证的量化形态 | **W8A8**（+ INT8 Engram） |
| 多机同置部署 | **A3 × 2 台**（DP4/TP8/EP32）或 **A2 × 4 台** |
| 单台 A3 | ✅ 有路径：TP8/**DP2**/EP16 + Engram host offload（`--engram-config`）。但文档明确这是**冒烟级验证**（模型加载 / decode 图捕获 / 自然文本 / 混合长度并发），"**不建立数据集精度或性能**"；示例用 **128K** 上下文、512 batched tokens、1 GiB KV/rank |
| **PD 分离（Prefill-Decode disaggregation）** | ❌ **文档原文：`Prefill-Decode disaggregation is not covered by this guide.`** |

**上游 CED 的定义**：文档把 CED 描述为**模型架构**（40 层 = 20 causal-encoder + 20 decoder，
1M 上下文，含 SWA Bounded Replay）。也就是说"能拆"是模型给的，
**怎么在框架里拆成可部署的 P/D 两段，上游没做**。

### 5.2 本包的增量

| 增量 | 说明 | 证据 |
|---|---|---|
| **① 单台 A3 上的 PD 分离 + CED** | P 只跑 20 个 encoder 层 + layer-20 全局源；D 做 128-token 有界重放 + 全 40 层 | prefill **2.07×**（144K）、验收 **21/21** |
| **② W4A8 形态** | 上游验证的是 W8A8；本包跑 W4A8 + INT8 Engram | 273 GB 权重；精度 Vision 23/23、GSM8K 198/200 |
| **③ 1M 上下文的完整验收** | 上游单机示例是 128K | 1M 四针 4/4、多轮 3/3、TTFT 101.2 s |
| **④ 14 个补丁** | 通信 / Engram / 算子 / 调度四类优化，**逐项核对过上游没有等价实现** | 累计 **−7.71 ms/step（−19.7%）** |

**④ 的 14 项优化**（按瓶颈分类，每项独立门控、可单独关做 A/B）：

| 类别 | 优化 | 收益 |
|---|---|---|
| 通信 | **MoE 走 AllGather**（TP=EP 时把标量开销按 token 摊薄） | 128K **−4.25 ms**，KV 池 3.39M→4.16M |
| 调度 | **admission gate**（prefill 不再饿死 decode） | 首 token 后不再长时间不出字 |
| Engram | **INT8 表 host 常驻 + local-owner 快路径** | 释放 HBM 给 KV |
| Engram | **★ 设备直索引**（表仍在 host DRAM，设备算子直接读） | 每步同步 host 时间 **3.379 → 0.058 ms** |
| Engram | **hash / plan 的 numba JIT** | 0.427→0.076 / 0.261→0.068 ms |
| Engram | **gate 分块**（去掉固定 2048 行 padding） | −1.56 ms |
| 算子 | **QLI 无候选快速路径** | −0.49 ms |
| 算子 | **rope 取表融合**（6 kernel → 2） | −0.45~0.62 ms |
| 算子 | **`wo_a` 2D matmul** | −0.31~0.76 ms |
| 算子 | **expert mask 区间比较** | −0.51 ms |
| 算子 | **`q_norm` + `wq_b.quantize` 融合**（`V41_QNORM_FUSE`，35 层，用官方算子） | **decode −0.62%** |
| 算子 | **`input_layernorm` + `wq_a.quantize` 融合**（`V41_LNORM_FUSE`，43 层，**自研三输出算子**） | **decode −0.90%** |
| 量化 | msmodelslim 的 V4.1 W4A8 支持 | 仅在重新量化时需要 |

> ⚠️ **结构差异（要移植的人注意）**：本包的 Engram 补丁面向**扁平文件布局**
> （`engram_hbm.py` / `engram_hash.py` / `engram_gate.py`）；
> 上游 #16544 把 Engram **重构成了包**（`engram/{common,embedding,hash_state,layer,npu,parallel}.py`）。
> ⇒ 这 7 个文件**在上游已不存在**，要往上游提需要按新结构重做。
> 另外 7 个文件（`model.py` / `indexer.py` / `ascend_forward_context.py` / `rope_dsv4.py` /
> `block_table.py` / `token_dispatcher.py` / `dsa_v1.py`）在上游存在但无我们的实现。

### 5.2.1 算子优化兑现复核（2026-10-13）

上面的历史基础包收益不等于当前 DSpark+BI 栈中每项优化的独立收益。
本轮已回查正式权重、实际消费者与配对收据，结论如下：

| 候选 | 历史实测结论 | 当前最优 DSpark 栈的边界 |
|---|---|---|
| 新 `tp8meta` slot batching | A=1 真模型同实例配对，11/12 更快 | 当前关闭；源码只支持 1 行，不能覆盖 6 行 verifier 图 |
| 新 exact MoE mask/cast | A=1 真消费者逐位门与质量门通过，同实例 12/12 更快 | 当前关闭；只选 `(1,6)` 输入，不能把旧收益归给 `(6,6)` verifier |
| Hostmeta / 新 Indexer 融合 | A=1 有正式配对；Indexer 增量仅约 0.025 ms，7/12 更快 | 当前 DSpark worker 没有这些 arm 的安装/切换；async 下需重核关键路径 |
| 旧 `V41_SLOT_MAP_FUSED` | 已在 async DSpark HTTP 窗口复测，约 −0.0755 ms，判噪声 | 不采纳；它不是上面的新 slot batching |
| W4A8 prefix / QKV merge与tail | 正式配对慢于各自基线 | 保持关闭，不重复原几何测试 |
| WOA Cube | 单算子较快，真实模型消费者精度门失败 | 拒绝，不以微基准代替整网兑现 |
| GMM2 sparse MSD：M1 / K5 | M1 无统一收益；K5 四个设备窗口候选均慢，约 1.216–1.324× native 时间 | 尚未通过真实 full43 消费者接缝，不能作为已交付优化 |

旧 `V41_MOE_MASK_RANGE=1`（连续 expert-range 比较）仍在基线中开启，**不是**新
`STACK_MOE_MASK_ENABLED` 概率融合；新候选不能靠翻环境变量沿用已有捕获图。
扩展至 DSpark 六行几何后，仍需新消费者数值门、重捕获和同实例单变量对照。
详细证据及口径见 [近期优化复核](experiments/operator_stack/evidence/operator_revalidation_20261013/history_recent.md)
与 [GMM2/M1/K5 候选复核](experiments/operator_stack/evidence/operator_revalidation_20261013/history_candidates.md)。
旧基础包的真权重窗口、dummy设备账、CPU相位及统计限制另见
[基础补丁证据分层](experiments/operator_stack/evidence/operator_revalidation_20261013/history_base_patches.md)。
本轮 chip8–15 校准已完成（§1.2.2），但没有当前服务的单项A/B；不能把历史设备收益求和或配置开关
当作当前独立收益。尤其 JIT plan 依赖 fast local-owner 前置，实际采用值未独立证实。

---

## 6. 已知限制

| 项 | 状态 |
|---|---|
| **110 tok/s** | **不可交付**：设备 busy 本身 30.9 ms > 达标所需的 25.1 ms |
| **接受长度 A** | **不能当绩效指标**；必须同时报 `(ms/step, A, tok/s)` |
| `DRAFT_GRAPH=1` | **推荐显式开**（低并发），默认仍为 0 —— 理由见 §4 |
| `V41_MOE_ZERO_INVALID` / `MOE_NF` | 实验项 / 负结果，默认关 |
| **KV 门槛 3Mi** | **A3 交付默认已满足**（BAT=2048 + repack ⇒ 0.92 下 **4.01M**）。旧口径（BAT=8192 无重排）只有 2.82M ⇒ 那才是"不再满足、需显式 `GPU_UTIL=0.94` 且接受 8 s 级首 token"的那一档 |
| A2 与 A3 的性能差 | 硬件（两边同为 1600 GB/s/die）只能解释 ~15%，其余在 host 侧 |
| **codex 直连** | **需先跑使能补丁**：不装的话 Responses API 要么 500、要么**静默丢内容** |
| `reasoning.encrypted_content` | **不支持**（vLLM 侧直接 `raise`）。当前不触发；**一旦上游开始产出，这条链会 400** |
| **PD 分离的 DSpark 收益只在低并发** | 并发 4 时 ms/step 1.45×，成本由全批承担。⇒ 高并发用 `SPEC_MODE=off`，或 `dynamic` 让 K 随并发自动切（后者必须补做正确性验收） |

---

## 6.5 发布前门禁（改过 scripts/ 或 patches/ 就跑）

```bash
# 从**全新 clone** 自检：挂载/执行的文件是否都在库里、交付默认的代码特征是否齐全
bash tools/audit_fresh_clone.sh        # exit 0 = 通过；KEEP=1 保留临时 clone 供复核
```

**为什么需要**：本仓曾四次出现“**工作区改了、git 里没有**”的静默缺口
（自定义内核、`ENGRAM_WKV_TP` 实现、EPLB 接线/哑开关、`device_metadata` 诊断件），
每一个都**不报错**、只是让部署的人悄悄少掉性能或功能。详见
`docs/DELIVERY-AUDIT-20261005.md`。

## 7. 起服后必查（5 项）

```bash
# ① 静态内核没有静默降级 —— 必须输出 0
grep -ac "static_kernel.py:650" <serve.log>
# ② KV 容量 —— A3 交付默认（BAT=2048 + repack + GPU_UTIL=0.92）实测 4,010,585 tokens
#    （旧口径 BAT=8192 无重排为 2,987,727；DCP8 形态 18,864,624）
grep -oE "GPU KV cache size: [0-9,]+ tokens" <serve.log> | tail -1
# ②b 长 prompt 首 token —— 8K prompt 应 ~1.1 s（0.94 时会是 8 s）
# ③ 口径对不对（性能口径 vs 生产口径不可混比）
# ④ Engram local-owner 是否切到 fast 路径
# ⑤ DSpark draft 图真的在出 token（A ≥ 1.3；A≈1.0 就是静默失效）
grep -a "SpecDecoding metrics" <serve.log> | tail -1
```

---

## 8. 目录结构（给人看的）

| 目录 | 内容 |
|---|---|
| `scripts/` | 起服脚本（`serve_a3.sh` / `serve_a2.sh` / `serve_a3_ced_pd.sh` …） |
| `deploy/a3-ced-pd/` | **PD 分离 + CED 部署形态**（启动器 + 镜像包 + payload 清单） |
| `deploy/a3-tp8-fusion/` | **解码路径融合优化叠加包**（自研三输出算子 + 2 个开关 + 可复现发布包） |
| `tools/` | 自检、探针、压测、诊断（40+ 个，见 `AGENTS.md` §4） |
| `patches/` | 补丁件与载荷（`PATCHES.md` 有逐项说明） |
| `quant/` | 量化链（5 级，含 msmodelslim 补丁） |
| `docs/` `reports/` | 设计、验收、性能、故障史的完整记录 |
| `AGENTS.md` | **开发者 / Agent 视角**：补丁明细、调试入口、必踩陷阱 |

---

## 9. 许可与来源

* 许可、来源与致谢见 [`LICENSE`](LICENSE) / [`NOTICE`](NOTICE)。
* **测试语料不随包分发**（有版权）：`tools/fetch_corpus.sh` 自行下载并校验 sha256。

更细的内容（补丁逐项明细、故障史、性能拆解）在
[`docs/`](docs/)、[`reports/`](reports/)、[`EXPECTED_PERF.md`](EXPECTED_PERF.md)、
[`CORRECTNESS_STATUS.md`](CORRECTNESS_STATUS.md)。
