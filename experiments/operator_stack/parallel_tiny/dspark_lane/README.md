本目录负责 DSpark 的专属 tiny 执行线，最新正式验收目标为包含 DSpark 的 **TP8 ≤20ms/step**。已完成的 v7 是 **Target[6] FULL 图 + DSpark draft eager（enforce_eager=true，DRAFT_GRAPH=false）** 的 TP1 dummy 闭环基线。它未启用草稿图；tiny 只用于执行机制、计量和候选准备，正式达标仍由真权重 TP8 证据决定。

v7 正常 exit0，2次warmup后，同一模型3组原生 A/A 的 token、Top5 key 和有限 logprob 均一致（最大delta0）；8条关闭 profiler/logprob 的真实串行请求，每条32个completion token中排除prefill seed1，剩31个decode token，16个实际target/verifier轮次，17次实际 `_propose`（initial1 + 每轮16，含terminal），验证80个真实提议。8条合并 decode 窗口为5.872809130s / 248 token / 128轮，三元组 **`(45.881321ms/step,A=1.937500,42.228514tok/s)`**；这是 draft eager 的 tiny 诊断，不代表正式TP8或DRAFT_GRAPH有效ON。8/8输出为周期2复读，仅2–3种token；clean-rate不作质量证明，空输出0，`candidate_admitted=false`。

完整前端/core ID映射、原生CPU token流与最终输出prefix绑定、首末边界和三元组自洽均通过独立CPU解析；136次实际draft调用中initial8、terminal8，raw接受125，末轮预算裁剪5，不能将raw接受直接放进输出A。见 [v7分析](evidence/dspark_lane_20261011_c4_baseline_v7_analysis.json)、[host提交区间](evidence/dspark_lane_20261011_c4_baseline_v7_host_spans.json) 和 [冻结源manifest](evidence/frozen_v7_manifest.json)。[frozen_v7](frozen_v7/) 保留本次运行的原始bench/worker/staging字节；vendor的2806文件源码树与完整trace留远端。

现成 `v41-tiny-dspark` 的配置为 40 层、hidden5120、8 专家/top2、q_lora512、o_lora512、3 层 MTP、target `[37,38,39]`、block size5、Engram关闭、dummy权重。原配置 SHA256=`ad72656e807079b39a5e94f05f5ffe45116725787239344adb4577fc4a2671c3`。它与未启用 MTP 的普通 `v41-tiny` 是不同入口。

`bench_dspark_tiny.py` 使用 `method=dspark,num_speculative_tokens=5,enforce_eager=true`，target 捕获 `[6]` 的 FULL_DECODE_ONLY 图，draft 保持原生 eager。worker 不安装旧 operator 优化臂。原生 target、3层draft、rejection verifier、scheduler 和 cache 均经真实引擎执行。

`dspark_worker.py` 在 `execute_model` 接收的真实 `SchedulerOutput` 上记录 target 轮次、computed/prompt 长度和实际 `scheduled_spec_decode_tokens`；在原生 `_bookkeeping_sync` 返回的 CPU 列表上记录生成/接受；在实际 `drafter._propose` 与 `sample_tokens` 记录调用与完成。DSpark直接调用 `_propose`，`.propose` 的外层hook不被消费。启动/capture 期间不记录计数；独立profile请求与无观察性能明确分开。稳态不新增 D2H 或同步；每请求末轮仅同步一次，覆盖原生 terminal draft 的设备完成。

逐请求计量约定：

- prefill/seed 的 `_bookkeeping_sync` 完成时刻开启 decode 窗口，包含初始 MTP draft；客户端收到终态 `RequestOutput` 并完成末轮设备 fence 后关闭窗口，包含原生路径末轮 draft、调度与 cache。
- `computed_before >= prompt_tokens` 的实际 target 提交才计 decode round，前端 `engine.step()` 调用次数只作为诊断旁证。
- `decode_tokens` 使用最终输出 token 数减 prefill/seed token 数；末轮预算裁剪量、原始接受数单列，不将它们混入输出 `A`。
- `A=decode_tokens/rounds`、`ms=1000*decode_s/rounds`、`tok/s=decode_tokens/decode_s`；三者逐请求自洽。原生全局 MeanAcceptance 不参与计量。
- dummy 输出不作质量、clean-rate 或有效草稿收益证据；若接受数为0且 `A≈1`，标记 `draft_ineffective`，不接纳该草稿性能结果。

启动器 `launch_dspark_tiny.py` 只允许 physical4：实时检查 npu-smi health/进程、privileged 容器 `/work` 与只读 `/model` 映射、容器没有另一个运行任务，再执行单进程 `visible4→logical0` SDK，最后启动一个私有基线。沿用 CANN Python 路径，未改容器全局环境。没有 reset、清 pagecache、删设备文件或停止别人的进程。新 job 根不可覆盖，每次 source SHA 留在独立 launch receipt；单个 SSH 载荷小于1MB。

CPU 发现与源码 manifest 在 [evidence/discovery_compact.json](evidence/discovery_compact.json) 和 [evidence/source_manifest.json](evidence/source_manifest.json)。完整 CPU 源快照留在 `runtime/evidence/dspark_lane/source_snapshot/`。`stage_ids_vendor.py` 复制已完成 IDS v3 的完整运行时，仅移除其 capture 引用探针并精确恢复原生 model/router SHA；2806 个 Python 文件的树 digest 为 `e552b649eb47a39aeeeb549ffdb0f80d8aad199fe5105df34296b5ba62366ecc`。平台 Factory 保留 `bias_vl/image_sentinel_lo` 参数传递，没有过滤 kwargs 或改共享 Factory。

已纠正的启动问题均保留独立终态：v1 缺 CANN Python 路径；v2/v3 提前 import NPUWorker 导致模型绑定上游 Factory；v4 已完成 main/MTP 加载和 target 捕获，但 topology 回执误访问 ACLGraphWrapper.model。v5 保存原生引用，随后发现frontend ID与随机化core ID不同；v6采用add_request返回的core ID和OutputProcessor精确映射，但外层`.propose`计数未消费，被 [覆盖门拒绝](evidence/dspark_lane_20261011_c4_baseline_v6_rejected.json)。v7改用实际`_propose`，完整计数门通过。CPU计数与图包装边界检查见 [evidence/cpu_counter_gate.json](evidence/cpu_counter_gate.json)。

额外同实例96token请求只做真实decode驱动的5warmup/10active PipeUtilization采集，不进入性能统计。解析已确认10个物理4设备步，每步80个target `HcPre`（M6/static/stream13）+6个draft `HcPre`（M5/dynamic/stream47），对应实际40+3层；ACL replay为1次/step，只有target，ACL host kernel launch为702次/step。完整CSV/trace留远端，CPU解析只读取该冻结profile，结果只能解释本私有旧helper基线；算子累计时间、API累计时间及EVENT_WAIT union均不能当作可回收端到端预算。

目前没有可采纳候选。token/Top5/logprob 的原生 A/A 之后仍须补 routes 与 cache 消费者精度门，候选须通过同实例 OFF/ON、ON原生 A/A 后才可进入关闭 profiler/audit 的至少8条串行配对性能。
