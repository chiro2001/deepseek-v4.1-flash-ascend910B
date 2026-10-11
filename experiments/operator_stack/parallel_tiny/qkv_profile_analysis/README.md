原调度 QKV 合并的真实 profile 支持“失去 KV 投影与 Q norm 的重叠，KV tail 仍等 Q quant 后提交”的回退方向，没有发现新增 Slice/ViewCopy/Contiguous 设备 kernel。此结论只针对已冻结的 `src_qkv_profile_v5`；新 early-KV-tail 候选的收益须由主线程独立审计与性能配对决定。

已有关闭观察的 12 配对结果为 mask `(19.324495ms/step,A=1,51.747794tok/s)`、qkv `(19.555371ms/step,A=1,51.136845tok/s)`，qkv 0/12 更快。本采集先做的三组关闭 profile 校准也同向：mask `(19.176812,A=1,52.146311)`、qkv `(19.377154,A=1,51.607165)`，仅用于采集自洽，不替代 12 配对。两者均 speculative_decoding=false，不能当 DSpark 结果。

采集 `formal_qkv_regression_profile_v1` 正常 exit0，40 层 / hidden5120 / 384专家 / top6 / TP8 / W4A8_DYNAMIC / Engram层1、14，strict。profile_selection 仅 `tp8mask,tp8qkv × PipeUtilization`。16 个实际 worker 回执均 `execute_model` 驱动、manual0、warmup5/active10、schedule16/required16、worker_calls18；非daemon离线导出后，全部 16 目录各 10 完整步、每步80 HcPre、device8–15、6400层 attention window 验收通过。

QKV 必须按真实 NZ storage 识别：qa `[1,5120];[40,320,16,32];[1280];[1]`，kv `[1,5120];[16,320,16,32];[512];[1]`，合并 `[1,5120];[56,320,16,32];[1792];[1]`，Type=`QuantBatchMatmulV3`。mask qa/kv各40调用，qkv合并40调用/step；`[1280→4096]` quant含 q_b 与 indexer query，共48次，不能统称40次 q_b。512维 RMSNorm 全部44次，按每层投影后首个512 norm绑定真正KV消费者，排除额外4次 source norm。

下表是各 rank 的 10 步/400 层均值再取 rank 中位。它是 PipeUtilization 采集态的区间诊断，核时和事件可能重叠，不能作为可回收 E2E 预算。

| 实际记录 | mask | 原调度 qkv |
| --- | ---: | ---: |
| qa+kv / merged 投影累计核时，µs/step | 806.967 | 787.454 |
| 每层投影区间 span，µs | 23.229 | 19.687 |
| KV投影与Q norm重叠，µs/层 | 2.216 | 0 |
| Q norm与KV norm重叠，µs/层 | 0 | 0 |
| KV norm开始距投影结束，µs/层 | 1.101 | 7.811 |
| KV norm开始距Q quant结束，µs/层 | 5.112 | 0.631 |
| 投影开始→首个q_b候选结束 span，µs/层 | 35.838 | 38.079 |
| KV norm kernel，µs/层 | 4.399 | 5.888 |
| 图前 span，µs/step | 861.900 | 859.513 |
| 主图 span，µs/step | 18573.013 | 18801.013 |
| 主图 kernel gap，µs/step | 1114.525 | 1103.013 |

rank 对齐差值中位与“两个 rank 中位相减”不是同一统计量：投影累计为 -23.920µs/step，投影→首 q_b 为 +2.031µs/层，主图 span 为 +226.125µs/step。保留两种原始值及 rank 数组，见 [差分摘要](evidence/differential_summary.json)。主图扩大而图前基本不变，与原客户端回退方向一致；不能据此把所有增加都归因于一个算子。

设备 Slice 均7次/step，Cast均117次，ViewCopy/Contiguous/TransData均0；合并1792输出相关 copy kernel 为0。QuantMatmul总数248→208，原始累计时间3969.075→3986.432µs/step；减少40个调用不能当作减少等量时延。RMSNorm165次不变，其累计604.543→674.043µs/step，属于诊断观察。

真实 event 簇直接显示保留的空 stage：rank0/step5/layer0，合并投影结束20.25µs，主流 EVENT_RECORD在21.25；aux等待结束21.75后在22.0立即EVENT_RECORD，随后aux EVENT_WAIT持续至29.5（原行 duration=7.660µs），期间没有KV matmul；Q quant在26.75–29.0，KV norm直到29.75开始。mask对应 aux 在16.5–24.25执行KV matmul，与14.25–19.25的Q norm重叠。见 [原 task 行与 kernel 区间](evidence/event_clusters.json)。task_time没有event_id字段，因此没有把这些行强行命名成某个Python Event对象；源码与时序共同支持空 Part2/part3 门的解释。

主图 EVENT_RECORD均604次/step、EVENT_WAIT约484.9次/step，复用RESET主要在图后约603次/step，事件数没有因合并下降。WAIT的跨流区间 union 几乎覆盖整个主图，其重叠累计约82–83ms；它不是有用计算占用，更不是可删除等待。时间线用工具 calibrated timestamp，绝对浮点时标的亚微秒量化限制也不支持把20ns record行凑成端到端收益。

完整 CSV、trace 与16,256,196字节 full_analysis 留在远端 `/work/qkv_profile_analysis_20261011/`。主要 compact包606,355字节经SSH拉回，没有原CSV/大trace。第一轮输入软链被解析器拒绝，第二轮改远端本地实体快照，所有非daemon解析子进程确认NPU未初始化；未重启采集、改正式源码或改原CSV。原数据SHA、worker receipt、解析回执和工具SHA分别见 [完整 compact](evidence/compact_analysis.json)、[worker 验收](evidence/worker_validation.json)、[parse manifest](evidence/parse_manifest.json)、[来源清单](evidence/provenance.json)。

冻结源 SHA：formal_qkv_merge_patches=`13236e3fd61fa789b92371e71af0ba993479bbb5bdb60bd59b9d3848b2f01965`；native dsa_v41=`56d3ce83098babf585289d3e078d1efff0442547f61fb0955a11f0b42bfc6ba3`；bench snapshot=`205ce470168954681b653e1470f44e417f4b1860cead804d0d160c917b705b80`。这不是主线程随后修改的 early-tail 实现。其余SHA和原冻结补丁/原方法片段见 [源 SHA](evidence/source_sha256.json) 与 evidence/snapshot。
