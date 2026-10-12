# 当前目标：正式 TP8 包含 DSpark，≤20ms/step

用户于2026-10-11最新明确：**目标是包含DSpark在内，推理20ms/step**。
此要求取代此前关闭DSpark的≤17ms目标，也取代刚才的24ms表述。
旧会话01a1275a-c0b2-7662-aaaa-49e6ca507155的工作继续，源码/证据/正式质量与交付要求保留。

- 正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8与完整视觉权重。
- DSpark使用checkpoint原生MTP3：draft为128专家/top3（TP8 local16），K5/block5。
  正式target与draft实测GMM容量分别为36/local48及15/local16，审计与候选须按role绑定。
- 必须实际启用DSpark/MTP，时延包含主模型、草稿生成、目标验证、调度和cache处理。
  不能把仅开启env、草稿失效或无DSpark的A1结果作为新目标验收。
- 正式客户端串行至少8请求、固定2K输入/256输出预算，审计/profiler关闭。
  按实际target decode/verifier轮次计step，不以前端异步poll或输出token数代替。
  每请求记录decode墙钟、decoded_tokens、rounds、draft调用/提议/接受计数。
  ms/step=1000*decode_s/rounds；A=decoded_tokens/rounds；tok/s=decoded_tokens/decode_s。
  不混prefill、capture/profile或全局累积平均；首/末步处理显式记录。
- 报告(ms/step,A,tok/s)与clean-rate/复读/空答，A虚高不能算优化。
- 保持HCCL_DETERMINISTIC=strict及原精度门，完整路由/token/Top5/logprob/cache和草稿/验证实际消费者覆盖。
- 最优正式服务模型归属、GSM8K100/100、Vision23/23及包含DSpark的≤20ms客户端验收。
- 用户授权所有空闲芯片，正式TP8主线程优先a3-21 physical8–15。
  启动时复查实时占用/健康；仅80C98001 Alarm已授权，不reset、不停止租户、不清pagecache。
  SSH每文件<1MB，大权重/trace留远端或COS。
- 专属DSpark tiny线由已核验gpt-6.1-sol/max代理负责，独占健康空闲physical4。
  tiny/dummy/缩层结果只验证机制与包含DSpark的tiny时延，不能替代正式20ms/质量门。
- 报告COS/links-server发布，源码/compact证据提交，从HEAD生成MANIFEST、自检及双远端push。

当前已验收无DSpark mask客户端为(19.289956ms/step,A=1,51.840451tok/s)，GSM100/100、Vision23/23。
它是旧基线，不满足新的包含DSpark目标。尚无正式DSpark开启的20ms验收或最终部署完成声明。

最新QKV early-kv-tail-v2完整审计通过，但12组性能mask(19.208722ms,A1,52.059685tok/s)、
QKV(19.307216ms,A1,51.794106tok/s)，0/12更快，拒绝部署；同样关闭DSpark。
Engram SDK普通NPUGraph32代可行；正常TP8初始化的v7 static_default已在8rank完成32代语义/native stream恢复，但epoch临时目的地址使alias拒绝。独立v7 static_keep的8rank语义/alias/32代/恢复全部通过；整job因5s正常shutdown grace后6个child被收口为-9而exit1。v8在两个fresh正常TP8 job（static_keep/nonstatic_keep）均实际exit0，16rank case共512epoch、13个live owner seal/严格alias/native恢复通过、全部自有child正常0退出；source/SDK/math未改。它仍是未加载权重的小型SDK能力，不是完整43层消费者或正式性能，原型默认关闭。
稀疏MSD固定M1/capacity6已在真实E48原参数、synthetic activation的9case数值门通过（SDK/raw/candidate均maxULP0）；planner/canary/seals与输入原地修改生命周期已验证。当前未有性能、实际模型消费者或K5收益证明。正式profile实际target GMM容量36/E48，draft容量15/E16，不能直接推广M1二进制。

用户已清除旧goal并要求重新创建；2026-10-11已成功创建包含DSpark的正式≤20ms/step新goal，
工具状态为active。当前执行与完成审计以该新goal及本文件为准；旧17ms/无DSpark结论均不是新完成条件。
新goal尚未达成，不能标记complete。

2026-10-12第二轮更新（会话01a128f6续接，deepseek-flash[max]子代理）：
用户指示「优先推tiny多线优化、子代理一律deepseek-flash[max]」，随后要求找到本会话并继续。
root已重建goal（thread 01a12c28-2efd-7812-bd43-54c3cc4d6a33，active）并继续正式线。

- 正式线：v11 job `formal_dspark_full43_diag_v2` 在 rank1 `before_graph` 新增失败——`seal_native_parameters`
  对 `V41_ENGRAM_HOST_RESIDENT=1` 下有意的CPU常驻 `engram.embed.weight` 调用`npu.get_npu_format()`，
  抛 `NotImplementedError`（CPU backend）。rank1此前已写出`retained_root_ownership.json`（v11 root修复生效）。
  失败证据冻结于 `evidence/formal_a321/full43_diag_v2/`；另收口了v10/v11两个卡在resource_tracker的残留driver
  （身份核验后终止，见 `full43_diag_v1_stranded_cleanup_20261012.json`）。
- v12 `full43_consumers_v12`：`tensor_layout_metadata`仅对npu张量查询format；CPU张量记为
  `format=None, host_resident=true`且绝不再查询；其他device直接拒绝；`before_graph`新增`weight_seals.json`。
  CPU契约21项本地(2.13.0+cpu)与容器同版本(2.10.0+cpu)全通过；freeze `64e5fae9…`（56文件，20 UNCOVERED，全flags=false）；
  overlay 156成员/613382B；远端快照`src_dspark_full43_diag_v3` manifest `c2bdb9c6…`。
- v3 job `formal_dspark_full43_diag_v3` 已在fresh physical8–15启动（audit=true、profiler=false），
  正经历冷页缓存下的权重加载（8 rank，各~200GB host RSS），尚无43层输出/数值/性能结论。
- tiny K5容量CPU线（deepseek-flash[max]）：编译3/3成功、228/228数值/负例/回归断言通过（含与physical6已读回
  planner帧SHA逐一相同），证据 `k5_capacity_numeric_cpu/evidence/k5_capacity_numeric_cpu_verification_20261012.json`；
  `change_header`未运行（需CANN 9.1.0源）、device桥与性能门待授权窗口。
- tiny 2K/256计时窗口准备（deepseek-flash[max]）：`fixture_timing_diagnostic_v3_2k_256/`，72项CPU门通过，
  staged freeze `5442666d…`，`--launch`被参考缺失硬阻断；参考token待geometry v3终态`requests.json`。
- tiny geometry v3数值门（deepseek-flash[max]监控中）：同一physical4/PID953926，18/20请求、7/9检查
  （3×OFF A/A、seed0/1 off_on与ON A/A全delta0），剩seed2三项；终态后立即聚合冻结并启动2K/256计时窗口。

2026-10-12第三轮更新（tiny geometry v3终态 + 2K/256计时窗口启动）：
- geometry v3终态 exit=0：20/20请求，9/9数值门全过（3×native_off_AA、3×off_on、3×native_on_AA，
  max_logprob_delta全0），failure_lines=[]、nonfinite=[]；3828个geometry step原件（141,359,997B）远端重构校验通过。
  证据：`evidence/formal_a321/tiny_geometry2k256_v3_terminal_20261012.json`（c459fa91…）、
  报告 `reports/a321-tiny-geometry2k256-v3-terminal-20261012.md`（a460c4fb…）、306文件交付清单（ce18c503…）。
- v3 `results/requests.json`（1,064,345B，a01abe2b…）经gzip+base64单次小载荷（31,176B）取回并通过SHA核验；
  已生成 reference_token_ids_2k256.json（c5d9e55c…），2K/256计时包final freeze=`83004ebe2a6c696b35b9876d5715436e62392e1e5b4c2fe4af570e74415454d4`。
- 2K/256计时窗口已在physical4启动：job `dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256`，
  supervisor PID 2251067（启动前实时核对：chip4 Health OK、无进程、HBM 3288MB基线）。
  deepseek-flash[max]子代理“tiny_timing2k_monitor”负责监控→取回→聚合→报告；tiny结论不得外推正式20ms。
- K5容量CPU线（deepseek-flash[max]）完成：编译3/3、228/228断言通过（含与physical6已读回planner帧SHA逐一相同）；
  证据 `k5_capacity_numeric_cpu/evidence/k5_capacity_numeric_cpu_verification_20261012.json`（eb6bf966…）。
  剩余缺口：change_header（需CANN 9.1.0源）、weights_cpu payload路径、AscendC device桥与性能门（待窗口）。
- 正式线：v12（host-resident Engram封印）与v13（dynamo模块属性别名）先后在真机验证生效；
  v14（编译容器list容忍+结构诊断）已冻结（freeze `0fb7094e…`，CPU契约22项），v5 full43 job
  `formal_dspark_full43_diag_v5` 已在fresh physical8–15启动（含v14全部修复），当前权重加载中。

2026-10-12第四轮更新（2K/256计时结果 + v15启动）：
- tiny 2K/256计时窗口终态 exit=0（job `..._fixture_timing_diagnostic_v3_2k_256`，physical4）：
  18请求=2warmup+16measured（8对交错OFF/ON），按臂窗口三元组 OFF `(46.236890ms, A=1.229656, 26.594704tok/s)`、
  ON `(38.780688ms, A=1.229656, 31.707958tok/s)`（-7.456ms，-16.13%，仅同实例tiny机制，不得外推）；
  两臂各1659轮（3×128+5×255），18/18 token与v3参考逐位一致，period2复读18/18、dummy clean=0；
  initial/terminal draft与末轮fence、helper/replay计数、43封印（1548次）全部核对通过。
  报告 `reports/a321-tiny-2k256-fixture-timing-20261012.md`（baac8ef8…），分析 JSON（1fef32be…）。
  包缺陷（未改冻结字节）：final freeze把reference并入runtime_source_files但RUNTIME_FILES未含该键，analyzer需一次性补键后运行；建议后续把REFERENCE_NAME加回RUNTIME_FILES。
- v14在v5 job暴露：draft侧wrapper调用返回裸tensor（8rank adapter_mismatch_draft.json均为[5,5120]bf16 npu tensor），
  说明该调用路径上的observer-buffer envelope未被实际执行；v5已按纪律收口（run.exit=143）。
- v15 `full43_consumers_v15`（freeze `12d7e76c…`，CPU契约22项+同版本容器全过）新增：
  after_normal_load身份/扩展tag审计（post_load_identity.json、extension_identity.json）、envelope幂等重挂、
  以及“wrapped调用返回裸tensor时由adapter交回自有observer/prepack缓冲并显式记录adapter_synthesis”的受控回退
  （绝不静默、绝不冒充graph输出）。v6 job `formal_dspark_full43_diag_v6` 已在fresh physical8–15启动，权重加载中。

2026-10-12第五轮更新（v16编译分派点修复）：
- v15 job `formal_dspark_full43_diag_v6`（干净exit=1）给出决定性诊断：
  `post_load_identity.json`（rank7）证明 `drafter.model` 就是draft根、实例`forward`上envelope tag在位，
  但同时暴露该实例带 `_compiled_callable/compiled/evaluate_guards` 等键——即 `@support_torch_compile`
  在**构造时**捕获`self.forward`编译进`_compiled_callable`，类`__call__`径直分派到该callable，
  实例级forward补丁被绕过（这解释了v5的裸tensor）。v6因`extension_identity.json`把bound method塞进json.dumps而
  在新审计点失败（把方法对象序列化——纯审计代码缺陷）。
- v16 `full43_consumers_v16`（freeze `da678620…`，CPU契约23项：新增编译分派envelope测试与JSON安全回执）：
  `extend_compiled_dispatch`把envelope挂到 `_compiled_callable`（经`TorchCompileWithNoGuardsWrapper.__call__`的真实调用点）
  与`aot_compiled_fn`（pass_self语义），并在shim内`enter_root`、记录`note_dispatch`计数；
  扩展回执经JSON安全投影（bound method→repr）。instance-forward envelope保留（do_not_compile/skip路径）。
- v7 job `formal_dspark_full43_diag_v7` 已启动（快照`74f694b1…`），等待权重+首轮profile_run。

2026-10-12第六轮更新（v8大突破：真实32轮完成；v9修正draft eager+收尾窗口）：
- v7 job（v16）通过profile_run并在`DSparkProposer._propose`暴露`assert isinstance(model,_HIDDEN_STATE_DRAFTER_TYPES)`
  与plain proxy互斥 → v17动态子类adapter（isinstance透明、共享__dict__、只覆写__call__）。
  失败证据：`evidence/formal_a321/full43_diag_v7/`（含extension_identity.json首次成功落盘）。
- v8 job（v17）**首次完成真实32轮诊断请求**：`diagnostic_request_completed=true`、38个epoch导出、37个真实spec轮、
  main 40/40 decoder输出完整、图导出完整（main 1684、draft 202 tensors，alias/equality全过）、
  envelope来源为wrapped_callable_envelope（无一例静默合成）。唯一不达标项：
  (a) 3个draft decoder.output与shared/cache.draft.read点epoch=0（draft `_runnable` ACL图重放不跑Python taps；
  router/prepack等图外taps正常）；(b) `worker_exit_codes_complete=false`——默认5s ProcessManager宽限使EngineCore被强杀，
  executor teardown未及写`owned_worker_cleanup.json`。
- v18修正（freeze `70b6196f…`，CPU契约25项）：`force_draft_eager`在before_graph即force draft eager
  （use_cuda_graph=False + `_runnable` unwrap）并写`draft_eager.json`回执；driver把shutdown宽限提升到150s
  （并记录shutdown_timeout_seconds）；sdk_executor在任何清理异常前先写`owned_worker_cleanup_error.json`。
- v9 job（快照`0d51db71…`）已启动，等待决定性结果。

2026-10-12第七轮更新（v9清理通过；draft tap仍空；v20加账本）：
- v9 job（v18）终态：`worker_exit_codes_complete=true`（8 rank 全exit0、owned_worker_cleanup.json写入、grace=45s）——150s
  shutdown窗口修复生效。但仍`passed=false`：draft_eager回执显示`use_cuda_graph_before=false`（draft本来就是eager路径），
  3个draft decoder.output仍为0。
- v10 job（v19）显式置`do_not_compile=true`（回执确认）后draft decoder.output/shared/cache.draft.read仍0；
  draft adapter的envelope计数=5（draft调用发生了），但被插桩的draft layer Python未执行→执行路径另有旁路。
  main 40/40、图导出、37轮、cleanup全部保持通过；v9/v10证据分别存于`evidence/formal_a321/full43_diag_v9|v10/`。
- v20（freeze `415bc9fd…`，CPU契约25项）加**逐代Python调用账本**：`enter_counts`（按qualified+owner）、
  `tap_counts`（按点）、`root_entries`（main/draft），随每个epoch记录落盘——用于一轮定位draft路径断点。
  v11 job（快照`dbc19618…`）已启动。

2026-10-13第八轮更新（full43真实权重诊断首次通过）：
- **job `formal_dspark_full43_diag_v17`（v26包，快照`src_dspark_full43_diag_v17`）passed=True、exit=0**：
  真实32代诊断请求完成；8/8 rank各37个完整generation（40 main+3 draft decoder.output全部copy_count=1/rows>0/any(valid)），
  main/draft图导出完整（1684/202 tensors，alias/equality全过），`owned_worker_cleanup.json` 8×exit0、grace=45s。
  证据冻结于 `evidence/formal_a321/full43_diag_v17/`；报告 `reports/a321-full43-diagnostic-pass-20261013.md`。
- 收敛链要点（详见报告）：host-resident CPU参数封印→dynamo模块属性别名→编译容器list容忍→post-load身份审计（JSON安全）
  →编译分派点(`_compiled_callable`/`aot_compiled_fn`)envelope→adapter动态子类(isinstance透明)→draft eager+150s shutdown窗口
  →draft `do_not_compile`+编译安全账本→draft六行观测窗与positions/valid独立补位（含一处`torch.bool`笔误修复+CPU回归）
  →观察者Python计数定位`last_skip=dtype`→tap内positions/mask dtype归一（int64/bool）→43/43全写入。
- 诊断通过**不是正式准入**：仍UNCOVERED = 前端scheduler/rollback对应、prepack前后顺序、17消费者设备覆盖、
  原生A/A数学与logprob、独立通用质量（GSM100/100、Vision23/23）、包含DSpark的≤20ms/step性能。
  下一阶段：audit模式补全17消费者/前端对应，随后进入通用质量与正式API性能窗口（≤20ms目标）。

2026-10-13第二轮更新（tiny多线收尾 + draft-graph正式候选接线）：
- tiny K5容量线设备阶段完成（deepseek-flash[max]子代理）：真实CANN 9.1.0源编译（change_header首次执行）、
  target36/E48与draft15/E16两几何各5 case+2负例全过、planner帧与冻结oracle逐字节相同、SDK-vs-raw/candidate-vs-raw
  0 ULP、raw native A/A逐位一致；最小基准（同口径operator window）raw native 0.143–0.149ms、candidate 0.182–0.189ms
  （比值1.22–1.32，候选更慢≈多一次planner launch）——**当前容量下全量切sparse无收益**，不构成收益声明；
  证据 `k5_capacity_numeric_cpu/evidence/k5_capacity_device_{numeric,timing}_20261012.json` +
  `REPORT_DEVICE_20261012.md`；仅physical4、无reset/清cache/停他人。
- tiny 2K/256计时已闭环（OFF 46.24→ON 38.78 ms/step，-16.1%，仅tiny机制）。tiny三线（geometry v3/K5 CPU+K5 device/2K计时）
  均已交付；tiny机制结论仍**不得外推**正式收益。
- 正式线 draft-graph 候选接线（本条为正式20ms目标的关键路径）：
  * 机制确认：正式栈里 draft graph 的调用点是 `runner.drafter._runnable`（`ACLGraphWrapper(_run_merged_draft)`，
    见 llm_base_proposer.load_model）；`drafter.model` 是nn.Module，从不接收wrapper调用。
  * 交付件来源：`patches/files/draft/{dspark_proposer,llm_base_proposer,dsa_v1}.py`（A2线四件套：CAPTURE_VALUE_FIX/
    SWA_INDICES_RESIDENT/CAPTURE_NCTX_FIX/DISPATCH_QUERY_LEN_FIX，A2同实例 36.9→24.9ms/step、Vision23/23、GSM8K 98.999%）。
  * 长驻A3容器无法新增bind-mount ⇒ 以**快照内overlay + sitecustomize meta-path finder**实现等价字节替换，
    严格env门控（`DSPARK_DRAFT_USE_CUDAGRAPH=1`/`DRAFT_GRAPH=1`才生效，否则回答None、eager路径字节不变），
    缺件fail-closed（gated缺失即exit2）。快照 `src_dspark_native_graph_v1`（freeze 16文件含overlay与3交付件SHA）。
  * 三处修正/排障：①observer source guard 增加overlay llm_base_proposer SHA（d5003b77 stock vs 36d7c887 overlay）；
    ②draft wrapper 追踪对象由 `drafter.model` 修正为 `drafter._runnable`（graph_aa_v1/v3 的 "used eager" 拒绝根因：
    observer从未拦截到真实wrapper，events=[]）；③未改动任何冻结eager交付物与既有快照。
  * 当前 job：`formal_dspark_native_graph_aa_v4`（AA审计屏：3对含token/routes/top5/logprob与实际草稿/验证流对比），
    已在fresh physical8–15运行。AA通过后进入 closed-loop 计时窗口（perf模式）与正式API客户端（launcher v3 支持--draft-graph）。

2026-10-13第三轮更新（draft-graph候选：AA通过 + 正式API客户端新基线）：
- **AA审计屏 v4 通过**（exit=0）：3对全过，tokens/routes(2094×40×6)/Top5/logprob **逐位相同
  （max delta 0.0）**、actual_round_streams_equal=true。同几何三元组（6请求/688轮同规模）：
  eager 59.62 ms/step,A=5.75,96.44 tok/s → **graph 44.89 ms/step,A=5.75,128.10 tok/s（−24.7%）**。
  证据 `evidence/formal_a321/dspark_native_graph_aa_v4/`（comparisons/requests/result/banks+launched）。
- **正式API客户端窗口（draft-graph ON，8×2K/256，串行，审计/profiler关）**：
  688实际rounds、2040 decoded tokens、decode合计31.983s →
  **`(46.48740792151163 ms/step, A=2.9651162790697674, 63.783213812996586 tok/s)`**、
  clean 7/8、复读1（与eager基线同clean口径）；每请求rank收据中`runtime_graph=True`、
  `actual_graph_submission=True`（真实图提交被逐轮计数）。
  对照eager正式基线 `(57.5816614535, A=2.9651162791, 51.4941077458)` →
  **−11.094 ms/step（−19.3%）、tok/s +23.9%、A不变**。距离≤20ms仍需−26.5ms。
- 新host边界分解（688轮精确重建，与eager同法）：initial 0.195、**execute→bookkeeping 34.319**、
  bookkeeping→sample 3.677（eager时16.850→graph后−13.2ms，正是draft图化收益所在）、
  间轮3.061、terminal 5.235 ms/round。当前最大项=execute→bookkeeping（73.8%）。
- 设备侧[bneck] hp：eager 53.02 → graph **41.01** ms/step（同口径−12.0ms）。
  下一步：draft-graph设备时间线剖析（job `formal_dspark_native_graph_profile_v1`，已启动；
  快照`src_dspark_native_graph_profile_v1`=graph快照+profile包，bench已切graph臂并断言
  `draft_graph_configured=True`；launcher v2要求AA收据同臂）。
- 证据冻结：`evidence/formal_a321/dspark_native_graph_api_v1/{launched.json,acceptance_small.json}`。
  （原始acceptance 34.4MB与其round spans留远端；compact仅保留聚合/逐请求metrics/分解。）

2026-10-13第四轮更新（逐相位设备归因 + 与1行基线对照）：
- 设备时间线`formal_dspark_native_graph_profile_v1`（draft-graph ON，profiler开启，8rank×15步，exit0；
  离线CSV解析无新NPU初始化）。逐相位（rank0中位，profiler开启）：pre_target span15.35ms/kernel2.84ms
  （**431个小算子**、wait_union仅8.4µs⇒launch-bound主机入队受限）、target_graph(6行)span22.69/kernel21.65
  （**160×hcom_allReduce=6.12ms**、Swiglu3.13、HcPre2.30、MatMulV2 2.15、QBM2.00、GroupedMatmul1.46）、
  verify1.62/0.43、draft_ctx4.54/1.71、draft_body2.31/2.44。与eager profile对照：仅draft两段下降
  （body14.15→2.31、ctx7.88→4.54），target与pre_target几乎不变。
- **决定性对照**：无DSpark mask服务 `[bneck] hp=**19.226**ms/step, n=1 padded=1`（1行图，A=1客户端19.289956）
  vs 本候选 `hp=**41.01**ms/step, n=6 padded=6`（DSpark 6行图）⇒ DSpark在同一栈上净增设备步 ≈ **21.8ms**。
  ⇒ ≤20ms/step 必须在“6行target图+草稿+验证+逐步元数据”整体上砍掉约26.5ms客户端时延，
  即结构化工程（通信合并/元数据风暴压缩/图化），非单开关可及。
- 报告 `reports/a321-draft-graph-candidate-20261013-v46.md`；证据目录同前。
- 并行（deepseek-flash[max]子代理 storm_probe）：归因431算子风暴到源码函数、区分静态/动态、给压缩候选。

2026-10-13第五轮更新（slot-map融合候选同实例配对：设备侧−1.2ms，客户端无收益）：
- storm_probe（deepseek-flash[max]）交付：pre_target窗口（15.45ms）设备kernel并集仅**2.11ms**、
  CPU叶子算子3.28ms、aclnn API 653次/2.64ms，其余~10.6ms为未追踪主机胶水⇒**主机launch-bound**；
  四大来源=12×`_compute_slot_mapping_kernel`（已有`V41_SLOT_MAP_FUSED`融合开关，默认off）、
  Engram设备查表图replay 87 kernel、DSA稀疏注意力元数据37算子、其余~250 eager小算子（含26×`.item()`与13×阻塞H2D）；
  与eager臂对照：405算子/步**逐键零差异**⇒draft入图改造未覆盖pre_target。证据
  `evidence/formal_a321/storm_probe_20261013.json`（137KB）+ 报告`.../reports/storm_probe_20261013.md`。
- 同实例配对（同一快照/负载/客户端，仅多`V41_SLOT_MAP_FUSED`）：
  * `=verify`（融合+逐组双跑+逐元素比对）：**8×256 token与基线逐位一致**、无回落/无不一致告警；
    客户端`(47.3090,A=2.965116,62.675)`（双跑开销所致，仅作数值门）。
  * `=1`：客户端`(**46.5040**,A=2.965116,63.760)` vs 基线`(46.4874,A=2.965116,63.783)`——
    **客户端无稳定收益**（Δ+0.0166ms，噪声级）；但设备侧`[bneck] hp` 41.006→**39.797**（−1.21ms）。
    ⇒ 与storm_probe结论一致：pre_target主机受限，设备侧省时不传导到客户端。
    按纪律记为"无客户端收益"，未部署；开关保持默认off。
- 证据：`evidence/formal_a321/dspark_native_graph_api_{v2_verify,v3_fused}/`。
- 下一批候选（按storm_probe排序）：①26×`.item()`+13×阻塞H2D（0.3–1ms）；②~250输入准备小算子折叠成
  元数据图replay（保守3–5ms，需设备验证）；③target图内160×allReduce 6.12ms的合并（结构性）；
  ④Engram pad常量化/DSA元数据小算子合并。全部需先数值门再同实例配对。

2026-10-13第六轮更新（宿主相位探针 + 逐轮预算 + 两个候选结论）：
- 新诊断设施 `src_dspark_native_hostprobe_v1`（freeze 17 文件）：`host_span_probe.py` 外层包裹
  execute_model/sample_tokens/_prepare_inputs/_calc_spec_decode_metadata/_bookkeeping_sync/
  propose_draft_token_ids/_sample/drafter._propose/block_table.compute_slot_mapping，仅 monotonic_ns
  计时+调用序列（含起止纳秒，随 counter 收据返回），**不读张量/不同步/不改数学**；env 门控
  `DSPARK_HOST_SPANS=1`。带探针的 AA 屏仍 3/3 通过 ⇒ 语义不变。证据 `evidence/formal_a321/hostsplit_hspan/`。
- 逐轮分解（AA rank0，8 轮，A=5.75）：轮窗口40.30ms = execute跨度11.58（prep 2.24 + csmeta 0.42 +
  目标图提交/前段8.92）+ sample跨度28.57（bookkeeping D2H 1.11 + propose 3.68（含_propose 2.96）+
  其余23.78（含_sample等待目标logits+拒绝采样+终端fence））+ gap 0.15。⇒ 宿主可串行段≈6.5–7.5ms/轮，
  其余为设备与设备等待。
- 量级标尺：无DSpark mask 19.226hp/19.289956客户端(A=1)；DSpark候选 41.006hp/46.4874客户端(A=2.965)。
  **每token**：DSpark 15.68 vs mask 19.29（已更优）；但≤20ms/step且A≈2.97 ⇒ 需6.74ms/token，
  即设备侧（目标图21.6+草稿6.8+验证/准备3.2）整体再减半——结构化工程。
- 候选结论：①`V41_SLOT_MAP_FUSED=1` 设备−1.21ms但客户端46.504 vs 46.487（噪声级）⇒**不部署**；
  ②`V41_MOE_COMM_ALLGATHER=0`（MC2/FUSED_MC2）在 `route_capture_patch` 断言 (255,2044) 失败——
  MC2按floor分片 prefill token 而正式路由审计合同要求ceil完整路由 ⇒ **被审计合同阻塞**，
  需先重做路由导出合同（新快照+重验证）；失败原文 `evidence/formal_a321/dspark_native_graph_aa_v7_moecomm_rejected/`。
  该job已按身份核验收口（exit=143，芯片释放）。
- 报告 `reports/a321-dspark-round-budget-20261013-v47.md`。下一步：目标图通信收缩（160→80 allReduce或融合MC2）、
  草稿图瘦身、宿主7ms清理；全部需数值门+同实例配对。

正式API接线注意：现有`serve_formal_tp8.py`与`run_formal_client_acceptance.py`
仍显式要求`speculative_decoding=false`和`A=1`，是旧无DSpark验收链，不能直接用于本goal。
新API链须保留实际external→core request映射、同boot/time namespace的客户端结束时钟，
从真实worker计数获取rounds并核对initial/terminal及TP8末轮fence/Gloo join。
最终至少8个串行计数请求，warmup不计入这8个；不能使用`1000/tok/s`替代DSpark ms/step。

真实API CPU桥已冻结并通过21测试/90反例；真实checkpoint tokenizer工作集为2warmup+8measured、每条2048 token/256预算，AutoTokenizer parity10/10。实际完整真实权重原生HTTP诊断 `formal_dspark_native_api_v2` 已完成：8个measured合计688 actual verifier rounds、2040 decoded tokens、39.61618308s，三元组为 `(57.5816614535 ms/step, A=2.9651162791, 51.4941077458 tok/s)`，clean7/8、复读1。audit/profiler关闭、原生K5 draft eager/target FULL[6]；包含initial/terminal已提交draft和8rank末轮device fence/Gloo join，finish RPC排除。10请求/80份rank收据与原HTTP字节、38份部署源SHA均离线核验，13证据篡改反例拒绝；client/service均exit0，自有服务正常关闭。它是集成诊断，尚无完整43层/17消费者门、正式20ms或通用质量通过，不能授formal receipt。

v1曾因私有ASGI缺原 `routes` 失败exit1；其serve_http前的service_started=true不作HTTP证据。v2独立透明facade转发原routes/state后得到实际HTTP输出，不改冻结桥/SDK/数学。完整API raw约22MB、bridge windows约21MB保留远端并绑定SHA，本地保留全部round长度和首末原payload的112423B compact。

新的prepack CPU组件18文本、16测试/41反例已冻结；GMM1 raw INT8须在实际prepack前snapshot，GMM2原x GM原位pack，K从实际loaded role几何交叉绑定，6个审计Tensor须穿过真实graph出口。SDK v9协议桥已冻结10文件；完整43层生产接线 `full43_consumers_v10` 仍在开发，禁止把CPU或部分覆盖当正式消费者通过。通用质量API基础包17文本、8测试/34反例已冻结，真实source snapshot/启动器正在准备，GSM100/100与Vision23/23尚须同候选新输出。

tiny2K/256 v2的callable RPC实际传wrapper导致strict worker identity拒绝；v3以真实string bound RPC保留原identity/audit/capture门，43CPU门通过。当前同一physical4/PID953926继续运行，已完成7请求，9项数值门已有native OFF A/A seed0/1两项通过（delta0）；其余7项未完成，部分逐步快照不是整组数值通过。

用户再次要求create_goal时，工具核实当前thread已有unfinished active goal并拒绝重复创建；保持原目标active，未为重建而把未达成目标标complete。后续继续完整消费者实机审计、同实例配对优化及正式质量/20ms/部署与交付门。

2026-10-12更新：远端22MB原HTTP acceptance经独立CPU重算，10请求/80rank窗口的全部6648个rank-round原payload通过，后端runtime imports/NPU/HTTP均0。688 measured墙钟精确重建：execute→bookkeeping35.445734ms/step、bookkeeping→sample返回16.850337ms、轮间1.644755ms、末sample→client3.429715ms、initial/首调度0.211120ms，总57.581661ms；这些是host边界，不是孤立设备kernel或draft成本。

full43 v10首诊断已冻结并在fresh physical8–15真实加载权重。实际rank5加载39.1183GB后于`before_graph`拒绝：完整视觉架构`AscendDeepseekV41ForConditionalGeneration`没有`.model`属性，其实际decoder通过`language_model.model`持有；无实际graph/43输出/数值/性能通过。原cleanup已对8个出生身份匹配child通过pidfd硬kill，但10s内全部尚未reaped、exitcodes={}，该job维持失败且run.exit缺失。后续只读确认原core/worker已不可见、8–15无NPU进程；仅说明当前资源释放，不将其冒充原reaped0。独立v11沿原vl_model源码/class/registered ownership精确修复，保留原conditional/vision对象、draft.model、原forward/数学与ABI；20项同版本CPU门通过，56文件已冻结，20项缺覆盖与所有actual/formal false仍保留，下一fresh窗由root统一启动。

tiny同一PID/job当前13请求/5项数值门通过，新增首seed OFF/ON与ON A/A的token/Top5/logprob/proposal/43routes/3attention/逻辑与逐步visible cache/parameter seal全部通过，delta0；剩余4门未完成，不报价性能。general质量launcher21文本/14CPU/59拒绝已冻，仅CPU只读核验90shards/4984 MTP+vision headers与20assets，未运行新质量请求。新增独立K5容量CPU优化线准备target36/E48、draft15/E16实际参数探针，不从旧M1模板外推收益。

用户最新临时目录约定已落地：后续本地使用`~/tmp/dsv41-<任务名>`，workspace及repo AGENTS均记录，三代理已ACK。46项旧产物共6,578,301,143B已逐内容哈希核验后迁移至`~/tmp/dsv41-history-20261012`与`~/tmp/dsv41-tp8-selfcheck-20261012`，旧路径保留链接，/tmp空闲6.7GB。冻结源码/当前远端NPU路径与证据原字节不变。

2026-10-13第七轮更新（async 长请求配对判定 + 目标图算子全谱 + 设备底限量化）：
- **async@256 vs sync@256 配对判定通过**（同快照`src_dspark_native_async_v1`、同负载47-token逻辑与2K提示、
  仅差`DSPARK_ASYNC_{COUNTER,SCHEDULING}=1`）：
  * sync256：`formal_dspark_native_graph_sync256_v1` exit0、
    `(45.75276821705427 ms/step, A=5.930232558139535, 129.61472691676502 tok/s)`，258 rounds，1530 tokens。
  * async256：`formal_dspark_native_graph_async256_v1` exit0、`async_reference_passed=True`、
    `(41.80449965909091 ms/step, A=5.795454545454546, 138.6323145287125 tok/s)`，264 rounds，1530 tokens；
    6/6 请求 token 与 sync 参考**逐位一致、max logprob delta 0.0**。
  * ⇒ **ms/step −8.62%、tok/s +6.96%**（同输出）；稳态轮中位 async 39.4–40.8ms vs sync 42.0–47.2ms
    （−8~11%），代价是每请求多 1 轮 overshoot（44 vs 43）。
  * 采样干扰声明：py-spy 在最后约20s attach（231 samples，仅作归因），上述计时聚合与中位均为整窗数据；
    py-spy 期间比例极小且 pair-2-a/b 中位并未变慢。
  * 证据 `evidence/formal_a321/dspark_native_graph_{sync256,async256}_v1/`（launched/result/requests/
    pair-0-a counters compact/bneck lines；async 另有 async_reference_comparison.json）。
- **目标图（model 48）rank0 全谱**（既有 profile CSV 离线解析，脚本`/work/tools/td48_analysis.py`）：
  每步 1986 kernel、union 21.59ms、dur_sum 24.86ms；前几名＝81×hcom_allReduce 3.32ms（唯一 collective，
  40层×2：attention AllReduce + MoE `maybe_all_reduce_tensor_model_parallel`）、
  GroupedMatmulSwigluQuantV2 40×3.16、HcPre 80×2.30、MatMulV2 97×2.15、QuantBatchMatmulV3 208×2.00、
  GroupedMatmul 40×1.48、SparseFlashMla 40×1.31、…；整步全流 union **27.29ms**（含 draft 2.6、engram 0.37、
  eager 2.76）。
- **底限量化（新的否决性事实）**：整轮设备忙时 ≥27.3ms；pre_target 窗口 15.4ms 中设备仅 2.1ms（纯宿主）、
  target 图 23.1ms 中 21.6ms 忙。⇒ 即使宿主暴露全部消除、调度完美重叠，轮时仍 ≥27ms；**≤20ms/step 需要
  6 行 forward 自身先砍 ~30%**（21.6→≤15）并把 pre/draft/sample/host/客户端 5.5ms 全部藏掉。这是"结构化
  重写级"差距，不是开关级。
- 工具新设施：`py-spy 0.4.2` 已放入容器`/work/tools/py-spy`（host 复制经 `docker exec -i`），可对 rank0
  worker 直采 Python 栈（本轮 231 samples 多为 shutdown 帧，未形成归因结论）；profile 启动器加
  `offline_tools/native_profile_launcher_v1/launch_formal_dspark_profile_graph_v3.py`（=v2+`--extra-env`，
  本repo同名文件，sha a1d33819…）。
- **async 正式 API 桥变体**：新快照`src_dspark_native_graph_api_async_v1`＝api_v1 + async 三件
  （counters 4877900b…/contract d9271e2a…/worker 8dc2192a…）+ 桥`runtime_guards.py`两常量改为
  async 值 + `service_bootstrap._guard_config` 改为"env门控的async允许"；`native_api_snapshot.json`
  五文件哈希与 frozen_counter_sha256 已同步。CPU 自检：常量一一绑定 True、守卫真值表 4/4 正确
  （sync 合同保持 REJECT）、`test_async_counter_cpu.py` 容器内 **8/8 通过**。
  正式 API 窗口已启动：job `formal_dspark_native_graph_api_v4_async`（8×2K/256、串行、审计/profiler关、
  `--draft-graph`+async env），冻结客户端等待器已挂（`run_native_api_client_when_ready.py`，port 18583）。

2026-10-13第八轮更新（**正式 API async 窗口完成：新最优 32.9705 ms/step**）：
- job `formal_dspark_native_graph_api_v4_async`（8 串行 × 2K/256、审计/profiler 关、`--draft-graph`+async env）
  客户端 exit0、自有服务正常关闭（job run.exit=0、8–15 无残留进程）：
  **`(32.97052108069164 ms/step, A=2.9394812680115274, 89.15483200333648 tok/s)`**、
  694 rounds / 2040 tokens / decode_elapsed 22.88154163s、clean 7/8、复读1（与 sync 同口径）。
- 对照上一版正式最优 sync graph `(46.48740792151163, A=2.9651162790697674, 63.783213812996586)`：
  **−13.5169ms/step（−29.08%）、tok/s +39.78%**；A 差来自 6 个额外 overshoot 轮。
- **等价性**：8/8 请求 token 流与 sync 正式窗口逐位一致（`client/token_identity_vs_sync.json`，
  逐请求 sha256 相同）；原始 34.4MB acceptance.json 留远端 sha 496ab192…。
- 报告 `reports/a321-async-api-window-20261013-v49.md`；证据
  `evidence/formal_a321/dspark_native_graph_api_v4_async/`（320KB）。
- 归因：正式负载（A≈2.94、86 轮）宿主占比高，async 重叠收益最大（−29%）；A/A 屏幕（A≈5.9、43 轮）
  只 −8.6%。结合设备底限 27.3ms ⇒ **当前 32.97 的剩余宿主暴露只有约 5–6ms，主线已转为设备侧**。
- 设备侧下一发已启动：`formal_dspark_native_graph_profile_mc2_v1`（profile 快照 + `--extra-env
  V41_MOE_COMM_ALLGATHER=0`，vendor MC2 路径的设备时间线诊断；audit-off，不作性能声明）。

2026-10-13第九轮更新（MC2 排除 + slot-map 复测仍中性）：
- `formal_dspark_native_graph_profile_mc2_v1`（audit-off 设备诊断；`enable_fused_mc2` 默认 0 ⇒ 实走
  MC2=Distribute/Combine，非 FUSED_MC2）exit0：rank0 目标图 union **24.72ms（基线 21.59，+3.13）**、
  整步全流 union **35.65ms（基线 27.29，+8.36）**；新增 MoeDistributeDispatch 40×1.68 +
  MoeDistributeCombine 40×1.34 + allGather 40×0.64 + MemSet 80×0.83 + PadV3 80×0.44，
  只换回 AllReduce 3.32→2.40。⇒ **MC2（非融合）净负，方向排除**（FUSED_MC2 需 contract 变更，另议）。
  证据 `evidence/formal_a321/dspark_native_graph_profile_mc2_v1/`。
- async 生效后设备省时应能直接传导的假设**被 slot-map 复测否定**：
  `formal_dspark_native_graph_api_v5_async_slotfused`（async + `V41_SLOT_MAP_FUSED=1`）exit0、
  `(32.89499576368876, A=2.9394812680115274, 89.35952717939799)` vs v4 `(32.97052108069164, …)`：
  **Δ−0.076ms/step（−0.23%，噪声级）**，token 8/8 与 v4 逐位一致。⇒ 该开关继续默认 off；正式候选
  维持 v4 async。解释：slot mapping 落在串行链的宿主段，其设备收益被残余宿主暴露吸收。
  证据 `evidence/formal_a321/dspark_native_graph_api_v5_async_slotfused/`。
- **当前正式最优：async 臂 `(32.9705 ms/step, A=2.9395, 89.155 tok/s)`，clean 7/8、复现 token 8/8 一致；
  距 ≤20ms 仍差 12.97ms（−39.3%）**。报告 `reports/a321-async-api-window-20261013-v49.md`（含第4/5/7/8节
  归因与 MC2/ slot-map 两个负结论）。
- 主线剩余（结构性）：① 目标图 kernel 融合/AllReduce 收敛（1986 kernel/步，21.6ms）；
  ② 串行链宿主段（pre_target 10.6ms 胶水 + draft host 3.7ms）折叠成 replay/更少算子；
  ③ 完整消费者审计与 GSM8K100/100、Vision23/23、模型归属与部署/交付门。

2026-10-13第十轮更新（**发现并修复桥尾部伪影：正式窗口 32.97→28.09 ms/step**）：
- 逐请求窗口分解（v4/v5 sync/async 四窗口）发现每请求 133–725ms 的 post 尾部（两臂等量、随请求序号
  递增）；定位为 `CounterBridge._write()` 每事件全量重写含 8rank receipts 的历史状态
  （`bridge_windows.json` 最终 33.2MB、indent=2 同步序列化），阻塞 ASGI 事件循环
  （v5-w5：响应就绪 +21.7ms → 发包 +643.9ms）。**这是诊断 harness 自身开销，非解码工作**。
- 修复：新快照 `src_dspark_native_graph_api_async_v2`，`_write()` 序列化视图剥离 `rank_receipts`
  （HTTP finish 仍返回全量 receipts；stage机/校验/事件字节不变）。CPU 自检 0.17ms/1716B。
- **v6 正式窗口**（同负载、同 env、`--draft-graph`、审计/profiler 关）exit0、服务正常关闭：
  **`(28.092118429394812 ms/step, A=2.9394812680115274, 104.63722326244132 tok/s)`**、
  694 rounds/2040 tokens、clean 7/8、复读1；窗口=pre 386ms+rounds 18849ms+post 61ms。
  token 8/8 与冻结 sync 基线逐位一致。对照旧最优 v4 `(32.9705, 2.9395, 89.155)` = **−14.8%**。
  证据 `evidence/formal_a321/dspark_native_graph_api_v6_async_states/`；报告
  `reports/a321-api-v6-bridge-tail-20261013-v50.md`。
- 剩余差距精确定位在设备侧：逐轮 27.2–27.4ms ≈ 设备全流 union 27.29ms ⇒ 到 20ms 还需设备再砍 ~7.2ms（−26%）。
- 新候选（env 门控、默认关闭时字节不变）：快照 `src_dspark_native_mls_v1`（bench/profile）与
  `src_dspark_native_graph_api_mls_v1`（API，含 v2 桥修复）加入
  `V41_MLS_SHARED_EXPERT`（multistream_overlap_shared_expert）与 `V41_MLS_DSA_OVERLAP`
  （multistream_dsv4_dsa_overlap，冻结合同原为显式 False）。
  **v7 = shared-expert 多流重叠已启动**（`formal_dspark_native_graph_api_v7_mls_shared`，port 18613）。

2026-10-13第十一轮更新（**multistream 重叠三连测：正式窗口降到 26.61 ms/step**）：
- 候选实现：快照 `src_dspark_native_mls_v1`（bench/profile）与 `src_dspark_native_graph_api_mls_v1`（API）
  ＝各自基快照 + env 门控 `V41_MLS_SHARED_EXPERT`（multistream_overlap_shared_expert）与
  `V41_MLS_DSA_OVERLAP`（multistream_dsv4_dsa_overlap）；两 env 均未设时 additional_config
  与基线逐字节相同（本地 4 组真值表验证）。CPU 层已核对 vendor 实现：共享专家走辅助流、
  DSA MLA prolog/compressor 走辅助流。`src_dspark_native_mls_v1` 的 freeze.json 曾因键名少了 `stack/`
  前缀而自检失败，已修复并复核 **18/18 文件 0 mismatch**。
- 正式 API 窗口三连（快照 api_mls_v1、async 三 env、8×2K/256、审计/profiler 关，均 exit0+服务正常关闭）：
  * v7 `V41_MLS_SHARED_EXPERT=1`：**26.947666858789624 / A=2.9394812680115274 / 109.08110462456402**
    （vs v6 −1.144ms，−4.07%）；token 8/8 与冻结 sync 基线逐位一致。
  * v8 `V41_MLS_DSA_OVERLAP=1`：**27.31953162824208 / 2.9394812680115274 / 107.5963273459924**
    （vs v6 −0.773ms，−2.75%）；token 8/8 一致。
  * **v9 两者全开（当前最优）：26.613435590778096 / 2.9394812680115274 / 110.45102606106579**
    （vs v6 −1.479ms，−5.26%；≈两个单开关贡献可加）；694 轮/2040 token 不变、clean 7/8、复读1；
    token 8/8 一致。证据 `evidence/formal_a321/dspark_native_graph_api_v{7,8,9}_*/`。
- 数值门（bench async vs sync256 参考：token 逐位 + Top5 logprob<1e-3）job `formal_dspark_native_mls_gate_v1`
  已启动。报告更新于 `reports/a321-api-v6-bridge-tail-20261013-v50.md`（第6–10节）。
- **本轮累计：32.9705 → 26.6134 ms/step（−19.3%）、tok/s 89.15 → 110.45（+23.9%），全程 token 逐位一致。**
  距 ≤20ms 还需 −6.61ms（−24.8%），全部在设备侧（目标图 21.6ms 为大头；MC2 与 slot-map 已排除）。

2026-10-13第十二轮更新（数值门 + AA 审计 + 设备剖析；候选链收口）：
- **数值门通过**：`formal_dspark_native_mls_gate_v2`（bench async，`V41_MLS_SHARED_EXPERT=1`+
  `V41_MLS_DSA_OVERLAP=1`，vs sync256 参考）exit0、`async_reference_passed=True`、**6/6 token 逐位一致、
  max logprob delta 0.0**；同 job 三元组 (42.135, A=5.7955, 137.545)。证据 `dspark_native_mls_gate_v2/`。
  透明记录：gate v1 因前一窗口 workers 显存未回落而在权重加载期 OOM（exit=1），已清理自有残留并等待
  HBM 归零后重发 v2 通过。
- **A/A 审计通过**：`formal_dspark_native_mls_aa_v1`（audit 模式同配置、3 对）exit0、`native_aa_passed=True`：
  3/3 对 tokens_equal、**routes 2094×40×6 逐位相等**、prefill/decode differing=0、logprob delta 0.0。
  证据 `dspark_native_mls_aa_v1/`。
- **设备时间线剖析**：新快照 `src_dspark_native_mls_profile_v1`（freeze 23/23 自检 0 mismatch）+
  job `formal_dspark_native_mls_profile_v2`（8rank×15步，profiler 开）exit0、离线解析 8/8。
  注意：**profiler 开启会扰动多流重叠**（该臂 all_union 28.0ms 反而高于基线 27.29；eager/pre 段虚高），
  故该剖析只作结构性参考：target(m48) union 仍由 21.65→**20.52ms**（−1.1ms）、m48 kernel 数 1986→1947；
  正式判定一律以 profiler-off 的 API 窗口为准。
- 资源：v6–v9 四个服务、gate、AA、profile 全部正常收口（各自 job run.exit=0；physical8–15 无我方残留）。
  **注意：收尾时发现 physical 8–9（NPU 4）出现其他租户新起的 VLLMWorker（117MB，非我方）**，
  后续启动前必须实时复查占用（启动器自带占用断言会拒绝）。
- 下一步（按序）：① 若 8–15 空闲继续设备侧候选（目标图 20.5ms union 为唯一大头：厂商融合/更长重叠；
  已排除 MC2、slot-map、upstream pass_config 在 Ascend 不生效）；② 为 v9 配置补通用质量门
  （GSM8K100/100、Vision23/23、模型归属）——质量 runner 尚未就绪，是当前最大缺口；
  ③ 交付门（COS/links-server、MANIFEST、双推）。

2026-10-13第十三轮更新（**通用质量门（async 臂）已构建并启动**）：
- 背景：旧质量包（`dspark_general_quality_api_cpu`/`launcher_cpu`）契约显式要求 `async_scheduling=False`，
  而我们最优/最终服务是 async 臂 ⇒ 质量门必须扩展。
- **v2（env 门控 async）已冻结并部署**（默认关闭时与 v1 语义逐字节等价）：
  * `dspark_general_quality_api_cpu_20261012_v2`：`contract.py` 加 `ASYNC_ARM_ENV=DSPARK_QUALITY_ASYNC_ARM`、
    `ASYNC_COUNTER_SHA=4877900b…`、`counter_sha()`；`validate_candidate` 改为"调度臂必须与门控一致"+
    async 时必须有 `DSPARK_ASYNC_{COUNTER,SCHEDULING}=1`；`verify_rank_descriptions` 增 `scheduler_async`
    与 arm 一致性；`worker_adapter.py` 增 `counter.async_mode`/arm/env 三项一致性。CPU 门：**9 测试/38 反例**
    （原 8/34 + 新增 async 正/反例），manifest sha `8c524e1786910094f7afab5ceb61049eee769d7c56a015885bbfae58eb8abab8`。
  * `dspark_general_quality_launcher_cpu_20261012_v2`：计数器校验改用 `counter_sha()`；`assets.py` 解析
    api 包兄弟目录（v1/v2 名兼容）；`prepare_final.py` 默认 pin v2 包与 async counter；
    `launch.py` 把 `DSPARK_QUALITY_ASYNC_ARM` 透传进容器守卫、且在容器内以 root 新建 job 目录再 chown
    （挂载根属 root）。CPU 门：**14 测试/59 反例**，manifest sha
    `3fdf0afba2bcbe169cfffd72eb75c9eec77b9edf9a6cefc91466ebf697ed1a32`。
  * `dspark_api_counter_bridge_cpu_20261012_v2`：async 桥包（21 文本）+ 新 frozen manifest。
- **候选种子/快照（CPU-only，无 NPU/HTTP）**：种子 `q2_candidate/candidate_seed_v1.json`
  （sha 81b8d800…，152 源文件映射、SDK manifest sha 004de45a…（10 个实际 SDK 文件只读校验）、
  checkpoint config sha 40ebd329…）；经 `prepare_final_entry.py --execute` 生成
  **`/work/src_dspark_general_quality_v1`**（219 文件；snapshot manifest sha
  `a5cbfd6b58f84dfba16f5dfad95b3f95cb78d8c9c0b6d5821cdc7d598458e4ea`；candidate_contract sha
  `6dfd2abf07c99d1bd3a73761ff40449b3f1fbeaa7caed5d15a4935ce403e8e8f`；counter=async 4877900b…）。
  快照内候选与 ≤20ms 窗口同源同参：`async_scheduling=True`、`enforce_eager=False`(draft graph ON)、
  `multistream_dsv4_dsa_overlap=True`、`multistream_overlap_shared_expert=True`、strict/权重/视觉/MTP 不变。
- **实际质量作业已启动**：`formal_dspark_quality_async_v1`（端口 18643，`--allow-known-alarm --stop-after`；
  服务与评测器均已拉起，评测器在等 ASGI 就绪）；通过后由冻结 orchestrator 跑 GSM8K 严核 100/100 与
  Vision 23/23（含 3 negative）并按原评分函数复核。证据目录 `evidence/formal_a321/quality_async_prep/`。

2026-10-13第十四轮更新（**质量门首跑：Vision 通过、GSM 99/100 —— 发现 DSpark 路径与纯贪心分叉**）：
- 首跑 `formal_dspark_quality_async_v1` 因候选 `startup_env` 缺引擎启动 V41/HCCL 标志（V41_MOE_COMM_ALLGATHER、
  V41_ENGRAM_DEVICE_INDEX、HCCL_BUFFSIZE、PYTORCH_NPU_ALLOC_CONF 等）而在加载期 OOM；修复=把 API 窗口
  的完整 24 项启动标志冻进候选 env，重建快照 `src_dspark_general_quality_v2`（sha 113c85ed…）。
- **`formal_dspark_quality_async_v2`（async+mls，当前最优）实跑结果**：
  * 服务启动正常（无 OOM；bneck hp≈27.9ms 与正式窗口一致）；评测器完成并产出
    `client/{gsm100.json,vision23.raw.json,strict_quality_receipt.json}`。
  * **Vision 23/23 PASS**（含 3 个 mandatory negative；vision 权重 266 张量/0.97GB 实测加载）。
  * **GSM8K 99/100**（`gsm8k-12` 失败：gold=13，本服务答 12）。严格门要求 100/100 ⇒ 评测器按设计判失败
    （`strict_quality_receipt.json` 已记录身份/candidate/rank 描述，全部一致）。
- **决定性对照**：历史同 checkpoint、无 DSpark 的 A1 服务（`formal_best_async_service_v5` 与
  `formal_best_strict_service_v4`，同一 canonical 题目/chat/512/温度0）在 `gsm8k-12` 上**答 13 且 100/100**
  （原文含 "break even at end of year 12), so he starts earning in year 13... Let's check:"）。
  ⇒ **DSpark 投机路径在该题上与纯贪心路径分叉**（同 prompt、同权重、同协议），这是首个把
  "接受前缀/rollback 审计"缺失变成可复现实证的反例。
- 正在跑同步对照：以冻结 sync DSpark（`src_dspark_native_graph_api_v1`，无 mls/无 async）建
  `src_dspark_general_quality_sync_v1`（snapshot sha 8b0b7efd…，counter=da845a58…）并启动
  `formal_dspark_quality_sync_ref_v1`（端口 18663）。判定：若该臂也答 12 ⇒ 分叉源于 DSpark 投机路径本身
  （与我的 mls/async 改动无关，需修 spec-decode 语义或如实上报）；若答 13 ⇒ mls/async 改动是诱因，
  需回退/修正这两个开关并重新验证 ≤20ms 与质量。

2026-10-13第十五轮更新（**同步对照完成：DSpark 分叉与优化无关；两臂质量一致**）：
- `formal_dspark_quality_sync_ref_v1`（冻结 sync DSpark、无 mls/无 async；快照
  `src_dspark_general_quality_sync_v1`，snapshot sha 8b0b7efd…，counter da845a58…）实跑：
  **GSM8K 99/100（同样 `gsm8k-12`，答 12，与 async+mls 臂逐字相同的最终推理）**、
  **Vision 23/23 PASS（含 3 negative）**、服务 exit0、评测器按严格门判失败（<100）。
- ⇒ 结论：**DSpark 投机路径在 `gsm8k-12` 上与纯贪心分叉（12 vs 13），该行为在冻结基线与我的
  async/mls 优化臂上完全一致** ⇒ ① 我的两个多流开关数值上忠实于冻结 DSpark 基线；
  ② 质量门的 GSM 缺口是 DSpark 实现层面的既有语义/数值问题，不是本轮优化引入的。
- 对照基准（同一 canonical 题、chat/512/温度0、无 DSpark）：`formal_best_async_service_v5` 与
  `formal_best_strict_service_v4` 均答 **13**（100/100）。三个窗口的三元证据链完整、可复现。
- 质量证据：`evidence/formal_a321/quality_gate_runs/{formal_dspark_quality_async_v2,
  formal_dspark_quality_sync_ref_v1}/`（launched/run.exit/evaluator.exit/strict_quality_receipt/
  gsm100/vision23.raw，共 240KB）。
- 下一步（下一轮首选）：定位分叉点——对该 prompt 逐 token 对比 DSpark 与纯贪心输出，找到首个不同
  token，检查该处 draft 接受/拒绝与 rollback 语义（goal 中"接受前缀/rollback 审计"仍未闭合）；
  若属可修实现缺陷则修复并复跑质量门；若属 6 行 verify 与 1 行 decode 的数值差异则如实上报并
  评估最小侵入的对齐方案（例如关键步回退纯贪心）。
- 资源：两臂服务与评测器均已退出，physical 8–15 无我方残留（npu-smi 复核 4/4 NPU 空闲）。

2026-10-13第十六轮更新（**决定性对照：当前流水线 A1 = 100/100，分叉确认由 DSpark 路径引入；定位到 token 19**）：
- 全量预测比对：DSpark 臂（async+mls）vs 历史 A1 arm（formal_best_async_service_v5）100 题中
  **预测差异仅 gsm8k-12**；原始文本 55/100 有细微分叉（最早在第 9 字符）⇒ 两条解码路径存在广泛
  但轻微的数值差异（符合"6 行 verify 批 vs 1 行 decode"的批量形状效应），只有一题翻转最终答案。
- **同流水线 A1 对照（决定性）**：新建诊断服务 `formal_dspark_a1diag_v1`（同一 frozen stack、同一官方
  API/编码器/canonical 8-shot、仅 `speculative_config=None`、1 行图；spec-off 身份自报），
  GSM8K 严核 **100/100（271.7s，含 gsm8k-12=13）**。
  ⇒ 当前流水线内：**A1=100/100，DSpark=99/100**；两 DSpark 臂（async+mls 与冻结 sync）**逐字节相同**，
  故分叉是 DSpark 解码路径（verify 批量形状/prefill chunk 2044 vs 2048 等数值形状效应）引入，
  与我的两个多流开关无关。
- 分歧定位：对 gsm8k-12 的回答 token 化 → **前 19 个 token 完全一致，第 19 个 token 分叉**：
  DSpark 输出 671("The"，"The tree costs $90 to plant.") vs A1 输出 58559("Carl"，"Carlos spends $90 initially.")；
  两分支语义等价、后续推理走向不同结论（12 vs 13）。A1 该题撞到 512 token 上限（仍在权衡中给出 13）。
- 证据：`evidence/formal_a321/divergence_audit_20261013/divergence_summary.json`（三臂原文+首个分叉位置
  + token 级定位）；`evidence/formal_a321/a1_control_20261013/gsm100.json`（A1 100/100 原始结果）。
- 逐轮 accept/rollback 审计（DSpark 侧第 19 token 所在轮次、发出 token 是 accepted-draft 还是 bonus、
  该步 top-8 logprob 与 A1 的边际）已就绪（脚本 `bench_diag_prompt.py` + `analyze_divergence.py`，
  prompt 精确重建 4113 token）：**因 physical 8–15 被其他租户占满（8×45.7GB VLLMWorker）而暂缓**，
  首次尝试被显存门（free 16.6GiB < 0.7*61.3）拒绝，未触碰对方。
- 待资源释放后的顺序：① 跑 DSpark 逐轮+logprob 诊断（64 token，含 rounds 轨迹与 top-8）；
  ② 视结果决定修法（若 accepted-draft 语义或 rollback 有异常则修 DSpark 实现；若证实为数值近并列，
  需评估"仅对近并列 step 用 1 行复算"这类保语义修正的代价，或如实上报该 1/100 局限）。

2026-10-13第十七轮更新（**分叉完整归因：量化网格上的近并列，语义无误**）：
- 复现与定位：单 prompt 同步 DSpark 诊断（`formal_dspark_gsmdiag_v1`，精确重建 prompt=1206 token/4113 字符
  官方 chat 编码、K=5、draft-graph、64 token、逐轮收据 + top-8 logprobs）：
  前 18 个答案 token 与 A1 逐位一致；**第 19 个 token 分叉**（DSpark 671 "The" vs A1 58559 "Carl"），
  且该 token 是第 5 轮的 **bonus token**（verifier 自身 argmax）。
- **同一步上报分布**：DSpark top1=671 −0.918505、top2=58559 −0.918505（**精确并列 0.0**）；
  A1 top1="Carl" −0.77259、top2="The" −1.02259（领先 0.25）。两条路径的 logprob 差均落在
  **同一 0.125 量化网格**（W4A8/qrot 输出特征）⇒ 该题把"2 步领先"变成"0 步并列"，
  DSpark 以最低索引打破并列取 671。
- **accept/rollback 语义核查通过**：逐步发出 token = 其上报分布 top-1；accepted 位置等于 draft 值、
  bonus 位置为回退采样；轮次流与输出流唯一差异为"首个生成 token 不在轮内"（既有 prefill seed 边界）。
  未发现未验证接受/跳位等错误。
- ⇒ 结论：**gsm8k-12 的 12 vs 13 是数值层近并列被两条计算路径（1 行 decode vs 6 行 verify）
  分别打破**，非 accept/rollback 缺陷；冻结 DSpark 基线同样 99/100，历史 A1 同样 13/100100。
  报告 `reports/a321-dspark-greedy-divergence-20261013-v51.md`；证据
  `evidence/formal_a321/divergence_audit_20261013/`（含 A1/DSpark 两侧同一步 logprobs）。
- 对目标的影响：GSM100/100 与"DSpark 恒开"在该 checkpoint 的量化网格下相互冲突（1/100 题）。
  后续选项：①继续性能主线（≤20ms 尚差 6.6ms，设备侧）；②与用户确认质量门口径（接受 99/100+根因，
  或要求模型级 logits 精度改动并重新验证全部门）；③评估"仅近并列步做 1 行复算"的保语义修正（高代价）；
  ④继续排查是否存在可关闭的 logits 量化开关（本轮未发现显式配置）。
- 资源：三个诊断服务全部退出，physical 8–15 无我方残留；期间因**自身遗留 EngineCore/workers**
  造成一次显存门失败，已用容器内 PID 正确清理（教训：宿主机 PID 与容器内 PID 不同命名空间）。

2026-10-13第十八轮更新（**v46 交付链完成：COS/links 发布、提交、MANIFEST、自检、双推**）：
- **报告发布**（COS + links-server，下载回读 SHA 逐字节核验 true，登记 4/4）：
  `a321-api-v6-bridge-tail-20261013-v50`（.md 10666B sha 3e5a1cdf… / .html 16397B sha 7ea329b7…）、
  `a321-dspark-greedy-divergence-20261013-v51`（.md 5007B sha 955bb209… / .html 6724B sha 4db8567a…）；
  回执 `evidence/formal_a321/publication_v46.json`。
- **提交** `8ac7ec8`（272 文件 / +154291 行 / 5.5MB）：本批次报告 v46–v51（含 HTML）、
  DSpark native graph API/AA/profile 证据 26 个目录（async/窗口/slot-fused/桥修复/mls 三臂/sync256/
  async256/gate/AA/MC2 profile/hostsplit/storm_probe/quality_gate_runs/a1_control/divergence_audit）、
  新源码（profile launcher v2/v3、API launcher v3/v4、draft_graph overlay+sitecustomize、
  full43 overlay v12–v26、dspark_lane 4 个脚本）、**质量包 v2**（api 17 文本 + launcher 21 文本，
  repo 内重生成冻结清单：api manifest 16280381…、launcher manifest a3bd1e99…，CPU 门 9/38 与 14/59 复核通过）。
- **MANIFEST**：从 HEAD 重生成 5981 项（`61a0f1f`），核对旧文件 0 漂移；**自检**：对 HEAD 全量
  `git archive` 逐项 sha256sum -c **5980/5980 OK（exit 0）**，`tools/selfcheck_pkg.sh` 在归档内
  **全部通过（exit 0，19 组含负控）**；回执 `evidence/formal_a321/selfcheck_v46_archive.{json,log}`
  经 `477cae5` 入库并刷新 manifest（`6ea6fbe`，5981 项，最终归档复核 0 失败）。
- **双推**：origin（GitHub）与 a321 内网镜像均 `e608030..6ea6fbe`，`ls-remote` 复核两远端同为 `6ea6fbe`。
- 说明：`full43_diag_v1..v17`（~11MB）与 sdk_init 原始目录等历史大件继续留在工作区（未入库，
  与既往"只提交紧凑件"纪律一致）；`coscli.log/coscli_output/` 为本地传输日志，保持未跟踪。
- **继续性能主线**：physical 8–15 空闲，已启动组合臂 v10（v9 + `V41_IDS64_HOIST=1` +
  `V41_SLOT_MAP_FUSED=1`；两者均为数值恒等类开关：前者把 40 层重复的 int64 cast 提升为一次
  〔注释称 ~0.20ms/步〕，后者此前实测设备 −1.21ms）。同时记录：`enable_fused_mc2=1`
  经查配置门 `moe_intermediate_size % 512 == 0` 判定**本模型不满足**（2304 % 512 = 256），
  该路径被厂商代码自动降级关闭，故 FUSED_MC2/MegaMoe 不可用（无需再试）。

2026-10-13第十九轮更新（**v10 组合臂：仍为噪声级；配置级调优判定触顶**）：
- job `formal_dspark_native_graph_api_v10_hoist_slotfused`（v9 + `V41_IDS64_HOIST=1` +
  `V41_SLOT_MAP_FUSED=1`）实跑：
  **`(26.435513501440923 ms/step, A=2.9394812680115274, 111.19440777465151 tok/s)`**、
  694 rounds/2040 tokens、clean 7/8、复读1、**token 8/8 与冻结 sync 基线逐位一致**。
- 相对 v9（26.6134）名义 −0.178ms，但**逐请求分解显示增益集中在 v9 的 2 个离群请求**
  （28.28/28.24 → 26.74/26.68）；其余 6 个请求两臂基本持平。按纪律记**噪声级、不作为稳定收益**。
- 配置级候选已穷尽：MC2（+3.1ms 负）、FUSED_MC2（配置门不满足）、slot-map（两轮均噪声）、
  IDS64_HOIST（≤0.2ms 级）、upstream pass_config（Ascend 不生效）、DSA/shared-expert 多流（已取 −1.9ms）、
  c8 系列（依赖 sparse_li_c8，非本注意力路径）。**距 ≤20ms 的 6.4ms 缺口属厂商内核级**
  （目标图 1986 kernel/21.6ms、81 次 allReduce 3.3ms、量化/HcPre 链融合），非开关可达。
- 证据：`evidence/formal_a321/dspark_native_graph_api_v10_hoist_slotfused/`（196KB）。
- 服务收口：wrapper 先被 TERM 未带走 python 子进程，随后直接 TERM entry 成功；
  physical 8–15 复核 4/4 空闲，无我方残留。

2026-10-13第二十轮更新（**批不变模式攻关：找到并修复厂商 3 处缺陷**）：
- 目标缺口（GSM8K 99/100，gsm8k-12 的 12 vs 13）根因是 6 行 verify 与 1 行 decode 的
  **批形状数值差异**。找到厂商正解：`VLLM_BATCH_INVARIANT=1`（worker init 自动注册
  batch-invariant mm/matmul/softmax/attention/add_rms_norm/reduce_sum；另有独立 CANN 算子包
  `cann-ops-batch_invariant`）。逐层打通三处厂商缺陷：
  1) **`reduce_sum` 拒绝 `axis=` 别名**（`vllm_ascend.batch_invariant` 影子 `torch.sum`，
     而厂商自家 W4A8 权重加载用 `.sum(axis=1)`）⇒ 快照新增 `stack/bi_axis_compat.py`（env 门控，
     axis→dim 映射；非末维回退原生 sum：厂商算子只实现最后一维，且加载期求和不属服务路径）。
  2) **CANN 侧算子包缺失**：容器只装了 torch 扩展（`batch_invariant_ops`），
     opapi 库 `libopapi_batchinvariant.so` 不存在。首次误装 910b 变体（报
     `aclnnReduceSumBatchInvariant not in libopapi_batchinvariant.so`），
     经 `get_soc_version()=253 / Ascend910_9382` 确认本机是 **910_93（A3）**，
     卸载后改装 `cann-ops-batch_invariant-A3-1.0.0`（run 包，--quiet --install-for-all），
     并以容器环境变量 `ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant`
     接入（不能改 LD_LIBRARY_PATH：会破坏 set_env.sh 的 libhccl 解析）。设备级验证通过：
     6 行与 1 行 reduce **逐位一致**、reduce_sum 支持负 dim/last dim。
  3) **A3 算子包缺 Meta（fake tensor）内核**：`npu_reduce_sum_batch_invariant` 在 meta 张量上
     返回输入形状而不是归约形状 ⇒ Dynamo 假张量追踪在 engram gate 的 `.sum(-1)` 处形状失配
     （`(s72,4,5120) 与 (s72,4) 相乘`报错）。`bi_axis_compat.install_meta_kernels()` 为 7 个算子
     注册正确的 Meta 形状（含 reduce/sum/mean/softmax/log_softmax/mm/matmul/add_rms_norm），
     容器内直测：`[BI-META] installed=7 skipped=` 且 fake 张量形状正确。
- 质量快照流水线沿用 v2 门控（`DSPARK_QUALITY_ASYNC_ARM=1`）：v5/v6/v7/v8/v9/v9b 六轮迭代，
  每轮修复一处厂商缺陷（v9b = 当前完整修复版，snapshot sha `9f9f6c36365bf8758923e420b3b0d341
  243077aea8be48351fe90df18da6374f`）。另发现并修复自身工具链问题：seed 内嵌 source map 与
  独立 manifest 需同步重钉（否则 `Seed candidate differs from its pinned frozen source manifest`）。
- **启动受阻（外部占用）**：physical14 被其他租户 qwen3-30B 测试进程占用（20:33 起，非我方），
  资源门正确拒绝启动（`Requested chip is occupied: 14`）；未触碰对方。待其释放后立即启动
  `formal_dspark_quality_async_v9c_bi`（命令已固定：source-dir /work/src_dspark_general_quality_v9b，
  manifest sha 9f9f6c36…，port 18773，--allow-known-alarm --stop-after）。
- 判据：若批不变模式生效，则 6 行 verify 与 1 行 decode 数值一致 ⇒ gsm8k-12 应给出与 A1 相同的
  13 ⇒ GSM8K 100/100；同时需复核 Vision 23/23 与（另起的）性能窗口是否受影响。

2026-10-13第二十一轮更新（批不变质量臂的启动受阻与看守重试）：
- v9c 首跑走过了全部历史失败点（批次不变 8/8 启用、`[BI-META] installed=7`、
  **0 次 Dynamo/形状错误**、权重加载完成），最终因**物理 14 被另一租户 python3（55GB，qwen3 测试）
  抢占**、rank6 的 HCCL 通信初始化报 OOM 而失败；我方已按自有 PID 清理残留
  （EngineCore/8 workers），确认 physical 8–15 一度回到 4/4 空闲，未触碰他人进程。
- 修复源码与证据已提交并双推：`dca0ba8`（bi_axis_compat shim + seed 构建器 + NOTE）
  → `180070b`（MANIFEST 5998 项，HEAD 全量自检 0 失败），origin 与 a321 均为 `180070b`。
- 另设看守脚本 `~/tmp/watch_fire_v9d.sh`：每 30s 检查 physical 8–15 的 4 个 NPU 块是否全部空闲，
  一旦空闲即以固定参数启动 `formal_dspark_quality_async_v9d_bi`（v9b 快照，port 18783），
  日志 `~/tmp/watch_fire_v9d.log`；最多 120 次重试。看守是"已验证的等待"：资源门与启动命令均已
  在真实环境跑通过（v9b/v9c 两次实际启动证据），唯一变量是外部租户释放芯片。

2026-10-13第二十二轮更新（批不变第 4 处厂商缺陷：NZ 权重冲突；看守机制生效并跑到 forward）：
- **看守（已验证的等待）生效**：`watch_fire_v9x.sh`（每 30s 轮询、递增 job-id/port）于第 11 次尝试
  捕获到 physical 8–15 全空闲并成功启动 `formal_dspark_quality_async_v9x2_bi`；批不变 8/8 启用、
  `[BI-META] installed=7`、**0 次 Dynamo/形状错误**，引擎一路走到**首次前向**。
- **第 4 处厂商缺陷**：`aclnnGroupedMatmulSwigluQuantWeightNzV2` 失败——
  `The dimnum of storageShape for second input (weight) must be 5, but got [141557760] (dimNum 1)`。
  根因：`batch_invariant.override_envs_for_invariance()` 强制 `weight_nz_mode=0`，
  使 W4A8 MoE 权重保持 ND 布局，而本栈仍按 NZ（5 维 storage）融合算子派发。
  修复（快照 v10 的 `bi_axis_compat`）：包装 `override_envs_for_invariance`，调用原函数后把
  `get_ascend_config().weight_nz_mode` 恢复为 1（本栈默认；BI 的 aten::mm/softmax 注册与权重布局无关），
  并在 stderr 打 `[BI-NZMODE] restored ...` 回执。快照 `/work/src_dspark_general_quality_v10`
  （manifest sha f07e41ff350c8167c28cd2c0a86d9a8241f47847d6ad4f69e102d23699506ca0）。
- **性能代价测量已备妥**：新建 API 快照 `src_dspark_native_graph_api_bi_v1`
  （= api_mls_v1 + bi_axis_compat + sitecustomize 钩子，manifest 已含两文件 SHA），
  待质量臂验证后即可用同一负载测 `VLLM_BATCH_INVARIANT=1` 的 ms/step 代价（判别"质量修复是否可负担"）。
- 资源：v9x2 服务已按自有 PID 收口；physical 8–15 曾回到 4/4 空闲，但**物理 14 反复被外部租户
  短作业抢占**（qwen3 测试、moe_compute_exp 等），v10 首次启动被资源门拒绝；已重启看守
  `watch_fire_v10.sh`（指向 v10 快照，port 从 18810 起，日志 `watch_fire_v10.log`）继续等待。

2026-10-13第二十三轮更新（节点重启 + 批不变第 5 处缺陷 + v52 报告发布）：
- **a3-21 于 ~13:56 UTC 重启**（SSH 拒连约 2 小时；`up 2 min` 确认）。重启后：我方容器
  `dsv41-real-stack-tp8-20261010-a321`（restart policy=no）需手动 `docker start`；
  **容器可写层内的 CANN 批不变算子包与 /work 全部快照/种子均持久保留**；全机 16 卡零进程、
  physical 8–15 health=OK。
- **第 5 处厂商缺陷（bool dtype）**：v13x1 批不变臂首次跑到"服务就绪 + 首个 HTTP 200"，
  但首个 decode step 抛 `aclnnReduceSumBatchInvariant ... Tensor self not implemented for DT_BOOL`：
  BI 把全局 `torch.sum`/`Tensor.sum` 换成只支持浮点的实现，而 engram 设备路径用 **bool 掩码求和**。
  该异常从 forward 逃出、正好落在引擎 device-metadata submit/release 之间，把
  `_submission_in_flight` 永久锁死（厂商注释已文档化该机制），后续请求全部报
  "The previous device metadata submission has not been released"。
  修复：`bi_axis_compat.reduce_sum` 对非 `float16/float32/bfloat16` dtype 回退原生 sum
  （加载期/辅助求和不在批不变契约内）。快照 v11（manifest sha
  `aac6ab23a2c3b9abc89d46e13a0f00350c5d05a2cc6cdb374e19200c671c0a05`），job
  `formal_dspark_quality_async_v14_bi`（port 18960）已启动。
- **看守机制三处教训（已修正）**：① 成功判据必须匹配 `"service_launch_requested": true`
  （曾误用 `performance_receipt_granted`，回执里只有 `formal_receipt_granted`，导致看守不退出、
  连环启动旧快照作业——已全部按自有 PID 清理）；② **容器内 npu-smi 看不到其他租户的进程**，
  占用判断必须用宿主 `npu-smi`；③ 即使"无进程"也可能被瞬抢，需接受"启动后 OOM→重试"循环。
- **v52 交付**：报告 `a321-batch-invariant-arm-20261013-v52`（MD 6579B / HTML 8536B）发布到
  COS + links（回执 `publication_v47.json`，4/4 下载逐字节校验 true）；提交 `881cd77`
  （报告+回执+quality_bi_prep）→ `c054e83`（MANIFEST 6001）→ `9363506`/`f181254`
  （v52 自检回执 + MANIFEST 6003；HEAD 归档全量 sha 校验 0 失败、`tools/selfcheck_pkg.sh` exit 0）；
  origin 已推至 `f181254`；a321 镜像待节点稳定后补推。

2026-10-13第二十四轮更新（**质量门全部通过：GSM8K 100/100 + Vision 23/23，性能代价≈0**）：
- 第 5 处修复（bool dtype 回退）后，`formal_dspark_quality_async_v14_bi`（快照 v11）一次跑通：
  **GSM8K 100/100（空答 0、错误 0）**、**Vision 23/23 PASS**、
  `actual_client_quality_passed=True`、服务与评测器 exit 0；决定性题 `gsm8k-12` 由 12 → **13**
  （文本变为 "he first earns money in year 13"）。证据
  `evidence/formal_a321/quality_gate_pass_bi_v14/`（含逐题 gsm100.json、vision23.raw.json、
  strict_quality_receipt.json）。
- **性能代价测量**（`formal_dspark_native_graph_api_v11_bi`，2K/256×8 串行）：
  **`(26.5523755957162 ms/step, A=2.7309236947791167, 102.85044684362282 tok/s)`**、
  747 rounds / 2040 tokens、**clean 8/8、复读 0**。与非 BI 最优（26.44–26.61 / A=2.94 / clean 7/8）
  相比：**ms/step 在噪声内相同，clean 反而更好**；A 降低是数值路径改变后的自然接受率变化。
  BI 的 token 流与旧非 BI 窗口不同（0/8 逐位相同）——数值路径确实改变，质量以官方判定为准。
- 交付：报告 `a321-bi-quality-pass-20261013-v53`（MD 4485B / HTML 6697B）发布 COS + links
  （回执 `publication_v48.json`，下载校验 true）；提交 `21a6608`（报告+回执+两证据目录+本文件）
  → `2313ab1`（MANIFEST 6027）→ `5a0f906`/`e131647`（v53 自检回执 + MANIFEST 6029；
  HEAD 归档全量校验 0 失败、selfcheck exit 0）；**origin 与 a321 均推至 `e131647`**。
- **当前最优正式服务**（含 DSpark、完整权重、strict）：
  `(26.55 ms/step, A=2.73, 102.85 tok/s)`，clean 8/8，**GSM100/100 + Vision23/23 全通过**。
  距 ≤20ms 仍差 **6.55ms**；该缺口与 v50 判定一致（目标图 1986 kernel/21.6ms、81×allReduce 3.3ms、
  量化/HcPre 融合），属厂商内核级结构化工作。**后续性能优化应直接在 BI 臂上进行**
  （BI 代价≈0 且质量更好，避免为性能牺牲质量的两难）。

2026-10-13第二十五轮更新（**配置空间穷尽 + 关键路径精确化 + v54 报告发布**）：
- **关键路径精确化**（v11_bi 的 747 轮精确收据 + 逐段分解）：
  `轮 25.66ms ≈ 目标图 21.65 + 草稿 ≈2.6–3.7 + 拒绝采样 ≈1–1.2`；
  `bookkeeping→sample_end 的 3.80ms` 经查**就是草稿提案**（`propose_draft_token_ids` 位于
  `_bookkeeping_sync` 之后），pre_target/slot-map/Engram 等全被 async 隐藏。
- **两条新否定证据**：
  ① `BI + V41_SLOT_MAP_FUSED + V41_IDS64_HOIST`（v14_bi_smallops2）与 BI 基线**逐位相同的空结果**
  （26.552488540829987 vs 26.5523755957162，747 轮/A=2.7309 一致）⇒ 这些设备侧小节省不上关键路径；
  ② `enable_reduce_sample` 与 DSpark 模型不兼容（起服报 `'DeepseekV41DSparkModel' object has no
  attribute 'logits_processor'`）；`V41_ENGRAM_GATE_HOIST` 与批不变模式冲突（缓存键含 `id()/shape`，
  Dynamo 追踪下产生数据依赖表达式 `u0`，profile_run 失败）。
- **配置空间总表**（报告 v54 第 3 节）：BI/多流（采纳）、slot-fused/ids64-hoist（空）、MC2（负）、
  FUSED_MC2（配置门不满足）、reduce_sample（不兼容）、kv_nz（PD 专属）、sp_by_pass（需 SP）、
  nz_mode2（仅 bf16）、draft_window（降低接受率）、c8 稀疏系列（需稀疏层）、upstream pass_config（Ascend 未注册）。
- **定量结论**：1 行（关 DSpark）实测 19.23ms/step ⇒ 该栈每步固定成本≈17.7ms、边际≈1.5ms/行；
  6 行要 ≤20ms 需把固定成本砍到 ≤12.7ms。报告 v54 给出 5 项厂商内核级融合清单与收益估算
  （量化链融合 1.5–2.5、allReduce 合并 ~1.6、HcPre/Post 链 ~1.0、采样段 1–2、pre_target 折叠 1–2，
  合计 6.1–9.1ms），并明确这不是配置或本仓 patch 可达。
- 交付：报告 `a321-remaining-gap-budget-20261013-v54`（MD 7142B / HTML 9772B）发布 COS+links
  （回执 `publication_v49.json`，下载校验 true）；证据目录
  `evidence/formal_a321/remaining_gap_20261013/`（逐轮分解 v11_bi/v9 对照、v14 空结果收据、
  RS 不兼容原文）；提交 `e8d4a3f`。
  （BI 代价≈0 且质量更好，避免为性能牺牲质量的两难）。
