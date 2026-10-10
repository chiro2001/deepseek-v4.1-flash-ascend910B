更新v21：双GMM prefix的v2作业退出1，plog定位chip8/AIV21的下投影GroupedMatmul MTE非法GM（507015，mix_aiv+0x4d74），不是HCCL任务；无候选性能结果。新 `/work/src_prefix_v3` 仅GMM1 prefix、GMM2保留counts的 `tp8prefixup/tp8prefixuproute`：真实8rank×40层独立GMM1比较通过；`formal_prefix_up_audit_v1`退出0，四原生控制/九配对路由、token、Top5/logprob delta0，3840消费者逐位及576 group坐标精确，局部有效token0–3/中位1。完整守卫已启动 `formal_prefix_up_perf_v1` 四方案12组关闭审计配对，之后七组微架构采集；性能pending，19ms未达，当前API未恢复。

更新v20：新增正式 `tp8prefix`（一次cumsum复用）和 `tp8prefixroute`（routing直接prefix）两bank，384/top6/48-local独立96组整数及量化行/scale/索引、6次变化输入图重放通过。v1正式作业在原生预热因wrapper关键字契约失败，候选未执行；验证私有作业父子归属后终止143，证据保留。v2修正关键字参数及加载前预检，预检通过；`/work/src_prefix_v2` 的 `formal_prefix_audit_v2` 已启动，整网精度/实际GMM消费者/性能均待完成。UB预算源码已补充，A8W4 post行预算为8.5*row*n+4*alignUp(row,8)+6*n+64，仍无UB容量occupancy或双缓冲收益。已验收API为后续试验释放，恢复记录保留。

更新v19：正式 `tp8metastack＋strict` 服务 v3 客户端已退出0，GSM8K100/100、Vision23/23；8条serial、2K输入/256输出测得 `(26.292508ms/step,A=1,38.033648tok/s)`，模型/API归属通过。新组合质量已经独立验收，仍未达到19ms。首轮runner在选择记录落盘前退出，未发请求；已增加1800s有界等待，同一服务重跑通过。下一项准备试验正式W4A8的prefix group-list；尚无性能结论。已有未完成goal不能被create_goal覆盖，继续沿完整目标推进。

# 两条算子线叠加与TP8目标

目标：使用正式权重和用户已授权的空闲设备启动隔离TP8，精度/路由/输出与真实覆盖通过，端到端decode ≤约19 ms/step，同时报告A和token/s。
已创建active goal；独立分支`feat/operator-stack-tp8-20261010`。

## 当前状态（优先于下方历史记录）

最新v18：`formal_slots_indexer_perf_v1`退出0，公平参考四方案12组关闭审计配对：base30.183590/A1/33.130585，core26.840930/A1/37.256533，meta25.724235/A1/38.873848，metastack25.694245/A1/38.919221。两slot候选11/12快于core，Indexer增量只有7/12快于meta/配对中位0.025080ms，整体中位选择metastack。已起正式`formal_best_strict_service_v3`（loopback18763/独立best-v3-tp8metastack模型名），8条serial＋Vision23＋GSM100客户端正在加载/验证。不能继承旧core质量结论。19ms仍差约6.7ms，goal active，后续W4A8前缀/metadata/buffer方案仍需试验。

最新v17：四bank `formal_slots_indexer_audit_v1`退出0，九个完整路由/Top5/logprob配对delta0、四原生控制通过；两个slot候选48份rank×12group整数消费者精确且有实际覆盖，Indexer检查通过。参考学完布局后停止重复登记，probe_v6 144/6重放通过。`formal_slots_indexer_perf_v1`已启动同实例四方案12组关闭审计配对＋单独CPU诊断，当前结果pending，不计审计态时延。此前meta有效结果26.234730/A1/38.117411，相对普通core增量需本次复测。客户端质量只覆盖旧core100/100和23/23；19ms及最终最优部署仍未完成，goal active。

最新v16：`formal_slots_batch_audit_v2`退出0，六配对/三原生控制delta0，24份rank×12group整数消费者精确且有实际合并覆盖。`formal_slots_batch_perf_v2`退出0，12组关闭审计/profiler：base31.147350/A1/32.105460、core27.185700/A1/36.784044、meta26.234730/A1/38.117411；meta12/12快于base，core有登记开销，增量需公平复测。CPU诊断15步×8rank已保存，rank0 metadata累计8.429→5.303ms/诊断步，不计可回收/E2E。新`/work/src_manyslots_v8`参考学完布局后停止重复登记，增加meta＋Indexer。probe_v6 144组/6重放退出0；四bank`formal_slots_indexer_audit_v1`运行，通过才起12组perf。W4A8源码证据确认融合GMM拒绝type2、counts模式有前缀扫描、A8W4有单缓冲队列；前缀模式/缓冲方案仍未实施。19ms/最终服务未完成，当前已验收客户端仍core100/100和23/23、26.744ms/A1/37.391tok-s。

最新v15：正式slot审计v1退出1，registered_groups0/contract回退1752，覆盖门拒绝候选并阻止perf。读取当前源码后改为worker在KV初始化后绑定`model_runner.kv_cache_config.num_blocks`，保留范围约束并增加contract诊断。`formal_slots_batch_probe_v5`退出0，144整数用例/6图重放/旧字段为空→实际V1池绑定回归均通过。`formal_slots_batch_audit_v2`已启动`/work/src_manyslots_v7`正式base/core/meta，仍待整网精度及12group实际覆盖；通过后才起12组perf_v2＋单独CPU诊断。当前最佳正式core客户端仍26.744ms/A1/37.391tok-s，质量100/100和23/23已过，19ms及最终服务未完成。

v14独立验证更新：`formal_slots_batch_probe_v4`退出0，INT32/INT64共144组CPU整数参考、尾部保护和6次变化输入图重放全部通过；独立事件INT32为801.356→392.164µs、INT64为945.504→458.397µs，均不计整网收益。`formal_slots_batch_audit_v1`已在同一正式实例建立base/core/meta三bank，源码`/work/src_manyslots_v6`，正在加载/审计。通过原生控制、6个配对、24份rank的12group消费者及实际覆盖后，守卫才启动12组关闭审计的perf＋单独cProfile。完整19ms目标仍active，最佳正式core质量100/100和23/23已验，但最终服务已为实验释放。

更新v14：`formal_core_strict_service_v2`客户端退出0，GSM8K100/100、Vision23/23，8条serial性能`(26.744123ms/step,A=1,37.391393tok/s)`，归属已核对；随后仅SIGTERM本任务API以继续实验。新的完整19ms goal已重新创建active。新增`tp8meta`实验候选，将12group slot坐标转换合并一次发射，保留独立输出与每步动态值；默认已验收core不开启。v1 Ascend标量offset store编译失败已修正；v2/v3近INT32上界差异定位到旧core helper与CPU整数参考不符，候选与CPU相符。v4在`/work/src_manyslots_v6`运行，两种dtype全144组CPU参考及6次重放通过才启动正式审计；INT32已72组/3重放通过，独立事件801→392µs不计E2E。正式19ms及最终最佳部署未完成。

客户端更新v13：core API v1归属验证及8条serial性能成功`(27.412337ms/step,A=1,36.479925tok/s)`。视觉23例被HTTP400缺chat template拒绝，不计为模型精度失败；启动入口已补生产deepseek_v41 tokenizer/parser，官方encoder文本预检一致。仅SIGTERM本任务API后重新检查无占用/Alarm，已启动`formal_core_strict_service_v2`（loopback18762，独立front-v2模型名）及串行验收runner。官方GSM8K train7473/test1319 JSONL已就绪，v2客户端pending。v12描述的是客户端结果落盘前的状态。

2026-10-10更新：用户提醒的变量为`HCCL_DETERMINISTIC`（启动参数`HCCL_DET`），当前strict原生/叠加精度全delta0。`formal_strict_profile_v2`退出0，七组×两臂×八rank的112份CSV全部离线解析；新增时间线分析显示chip8 core采集态gap主要在主图前（7.283ms），主图内部1.162ms，仅为诊断、不能当可回收收益。微架构wo_a MTE2 0.872；GMM up Scalar0.600；没有UB occupancy或双缓冲收益证据。新增报告v12。

`formal_core_strict_service_v1`已经过无占用/授权Alarm检查而启动，正式core、strict、auto端口、loopback18761、独立模型名。客户端runner已安排服务归属核验→8条serial性能→视觉23例→GSM8K100，结果pending；GSM8K支持官方JSONL以保持相同8-shot和test顺序。当前正式最快仍为`(26.622860ms/step,A=1,37.561704tok/s)`，19ms未达到，Goal保持active。以下保留历史进展，旧的“尚未起服/解析待完成”描述不代表当前状态。

用户已再次授权a3-21 physical chip8–15继续，仍为80C98001 Alarm；正式TP8已加载运行，不reset、不停止其他租户。生产容器为`dsv41-real-stack-tp8-20261010-a321`。

原生无候选bank的graph/eager/Engram子图关闭A/A均未通过完整路由与Top5 logprob门槛。最新`formal_attention_boundary_trace_v1`在8rank都发现layer0注意力输出进入HC post之前已不同，residual/post/comb一致；固定输入的HC pre80、HC post80、router40每rank重复5次全部稳定，共1600份真实输入。

`formal_attention_internal_trace_v1`已完成：8rank的归一化、Q/KV、可见cache、SMLA、O投影局部乘积均一致，首个偏离在wo_b AllReduce输出；固定局部向量重复10次，每rank后9次都与首轮不同，max abs0.0234375–0.03125。

更新：v1被错误的81次覆盖预期拦下；实测每步80次。修正后的`formal_fp32_decode_reduction_v2`退出0，三组A/A完整路由、token、Top5和logprob全一致/delta0，8rank×80个归约相对FP64参考转BF16逐位一致。新`formal_fp32_stack_graph_audit_v1`正在同worker图模式比较base/core/stack；尚未验收正式性能。后续受补丁源码与证据sha守卫保护。

最新：`formal_fp32_stack_graph_audit_v1`在建立候选前的base A/A拒绝通过，decode路由差1344、专家集合差888、共同logprob max delta1.624983，token相同。eager修复通过不替代图路径；正在准备真实向量的归约图重放及图捕获padding/dtype/守卫覆盖诊断。没有部署通过验收的正式服务。

用户提醒HCCL环境变量的历史解决记录后，已核对初始与更正记录，优先启动`formal_native_graph_strict_v1`：正式原生图、`HCCL_DETERMINISTIC=strict`在通信域初始化前设置，不启用FP32或候选bank，3组A/A结果待完成。真实向量的固定顺序普通/补偿求和六配置已全部通过独立与图重放精度，原FP32图重放4rank失败，仍不计整网收益。新增报告v9记录本轮顺序调整。

最新进展：`formal_native_graph_strict_v1`退出0，三组完整路由/token/Top5/logprob一致，delta0。`formal_stack_graph_strict_audit_v1`已启动同worker base/core/stack叠加审计，保持strict和原生归约。客户端工具已支持独立模型名、8条serial prompt与错误模型拒绝；GSM8K缓存/官方encoding/视觉图片在a3-21可用。正式性能与客户端质量尚未验收。新报告v10。

最新正式结果：strict三bank整网审计退出0，3次原生控制/6个配对全delta0，72份rank消费者Indexer/cache逐位一致。关闭审计/profiler的12组配对退出0：base `(30.598535ms,A1,32.681303tok/s)`、core `(26.622860ms,A1,37.561704tok/s)`、stack `(26.707870ms,A1,37.442147tok/s)`；core/stack均12/12快于base，Indexer增量未显示稳定额外收益，当前中位最快core。19ms未达到。`formal_strict_profile_v1`已启动base/core七组细节采集；API验证入口已准备但未起通过验收的服务，GSM8K依赖在独立venv安装中。报告v11。

Profiling v1已确认终止：EI0020、NPU socket16666占用、0采集目录。新`formal_strict_profile_v2`已启动，strict＋`HCCL_NPU_SOCKET_PORT_RANGE=auto`，不停止他人/不reset。解析器可按112目录验证两臂七组。仍待新的采集与客户端质量/服务验收。

正式七组×8rank的56CSV已完成shape分析：每步原始2356行包含重复通信逻辑事件，去重后2273执行记录；实际AllReduce82次/step，chip8累计2.937ms，不能当E2E。O投影小M的MTE2高、GMM Scalar高分开研究。正式HC/HcPost保持原生、TP1 selected-GMM/router禁用、BF16激活覆盖0；Indexer独立精度通过不替代整网。

最新报告：`reports/a321-fp32-reduction-validated-20261010-v8.md`；归约定位见v7、微架构统计修正见v6。没有有效正式TP8性能或最优服务验收，19ms目标保持active。

## 历史记录

两条线：

- 主线`aca0af9`：TP1 tiny mdfull已验证19.209ms/A1/52.058tok/s，含HC/router、激活、Q/KV多流、GMM1融合、metadata及多group slot启动。
- 950迁移线`3d2e4a7`：六方向已处理，最终仅Indexer INT8 K后处理融合保留，TP1配对收益约0.3–0.5%；A2验证未完成。其他负结果不重复筛选。

最新用户明确要求正式权重，禁止将tiny dummy结果作为TP8验收。正式checkpoint为`v41-w4a8-engram-dr-vision-qrot-mtpq`，40层/hidden5120/384专家top6、W4A8_DYNAMIC，Engram层1/14。90个index分片均存在。
TP1限定的selected-expert/GMM/路由路径不得直接声称TP8生效，需按本地专家布局核对覆盖与回退。

最新用户要求避开问题设备：a3-21 chip8–15全部排除，不再继续dummy重试。a3-21 chip0/1和5被占用，另有2/3/6历史初始化失败，当前没有单机健康空闲8chip。a3-22全部Health OK，但仅14/15空闲；920B-47是服务节点，没有可见NPU。正在准备a3-22健康14/15上的基础冒烟与正式权重单算子核验，完整TP8仍待资源。
不重置设备、不停止其他租户，仅创建本任务隔离容器。用户本轮明确授权最后8chip，包含14–15。

进展：所有8个chip均为同一80C98001；chip8单Vector/Cube精确通过。
首轮HCCL因设置CONNECT_TIMEOUT=30低于SDK最小120被配置校验拒绝，不算通信失败证据。
修正为120后，8个rank的Vector/Cube/HCCL AllReduce均通过；RAS Alarm仍保留，未宣称硬件已修复。
隔离容器`dsv41-operator-stack-tp8-20261010`，task label一致，挂载私有/work与只读tiny模型，只有本轮chip8–15可见。
TP1叠加新worker/图bank已完成接入，额外随机化attention权重，审计47/48输出交替以覆盖C2完成/未完成末步。
audit_v1在随机权重初始化缺少inference-mode时拒绝启动，无精度/性能结论；修正后audit_v2正在执行。
为避免测量干扰，仅停止chip4属于本任务的API；chip5另一条线服务保持运行。

TP1 audit_v2在原生RMS参考的autograd/inference上下文不匹配时失败，未放宽精度，wrapper audit增加inference-mode后启动v3。
`stack_model_audit_v3`三组整网对照完成：实际路由、47/48输出token（C2末步两种状态）和Top5 logprobs一致，max delta0；Indexer消费者点INT8/scale缓存和functional参考逐位通过。
审计时延约22ms包含clone/verify等，不作为性能结论。正式TP1配对需等待TP8冷编译结束，避免CPU编译干扰。
TP8 prototype `TP8StackWorker`仅启用兼容的HC/router、HCstatic/HcPost、metadata/多group slot、形状适配激活及Indexer；TP1的8-expert selected GMM与路由特化明确关闭，保留原生分布式MoE。
`tp8_stack_audit_pilot_v1`已启动，使用同一8个worker进程的tp8base/tp8core/tp8stack图bank；当前正在模型/静态核冷编译，尚无有效TP8性能结果。

`stack_model_perf_v1`关闭审计/profiler，同进程12组：主线19.338175ms/A1/51.711188tok/s，叠加19.101980ms/A1/52.350594tok/s，配对中位1.002596，10/12组更快。过程存在时延漂移，不能把两个整体中位差当作稳定可回收时间；额外收益小，尚无“明显”大提升。
TP8 pilot_v1完成8rank模型加载、native/core/stack图捕获与第一次请求，但返回路由未通过范围/唯一性检查，未计精度通过或有效性能。
安装态RoutedExpertsCapturer的single-DP分支只存本rank topk_ids；源码未重建TP8/SP的全token路由。
新增仅审计态的route_capture_patch：在global metadata token数与实际local shard形状一致时，收集每个rank真实ID再调用原capturer，不复制/编造缺失路由、不放宽检查、不修改路由算法。pilot_v2正在验证；正式计时不启用此观测修正。

pilot_v2/v3/v4仍只有256个prefill token有效，观测status显示8rank的patched capture调用数均0；不能当成模型精度失败或绕过路由检查。
已定位Ascend平台在init_device期间用`patch/worker/patch_routed_experts_capture.py`覆盖core capturer方法，之前安装时机过早。
修正安装到load_model入口（平台初始化之后、回调绑定之前），使用Ascend forward context的global num_tokens检查SP布局，只重建缺失prefill分片，保留原生decode路径。
TP8激活形状适配的96个独立数值case已通过原1ULP门槛；新增tp8act单独消融，并将同一适配加到tp8stack。各实际shape在capture前预编译，不在捕获期启动编译。
`tp8_stack_audit_v5`正在执行完整三组、native/core/act/stack四bank审计；当前还没有精度通过或正式TP8性能声明。

## 最新正式权重迁移结果（2026-10-10）

上面的dummy状态为历史记录；a3-21旧TP8容器已停止，8–15故障chip不再使用。
a3-22的健康chip14/15已完成Vector、Cube、2rank HCCL冒烟。严格设备透传的v3容器中，ACL将14/15枚举为0/1；详细计数采集改用仓库标准NPU容器配置，v4的msprof已确认实际Device Id=14。

正式checkpoint的90个分片、辅助文件和软链闭包预检查通过。加载所选真实参数时遵循模型参数dtype：Indexer k_norm的F32文件权重加载为BF16。未随机化权重、未使用dummy加载。

- 原Indexer融合：page0/page1累计128例，functional INT8/FP32 scale与INT8/FP16 cache逐位一致。
- 正式TP8候选激活形状`[1,576]`、`[2/8/16,4608]`的96例通过原1 ULP门槛；整网覆盖仍待8rank审计。
- HC static在`layers.15.hc_ffn_fn`、输入尺度100、无pre_mix用例失败，94例通过后第95例有1个BF16元素超出原门槛。顺序求和与Div-RN变体仍失败，正式入口全部保留原生HC。
- 复现显示pre系数最大差`1.1920928955078125e-7`触发BF16舍入边界。没有放宽容差。
- 新Indexer单系数变体：rank0标量触发Ascend MLIR编译断言；改为单元素Tensor后128例逐位通过，但三路径配对相对原融合版的加速中位`0.9995117`、4/8更快，未采用。
- 同进程三路径单算子事件计时：native 18.159349μs、原融合14.309020μs、单系数14.292750μs；原融合相对native配对加速中位1.271290，8/8更快。此前独立两路径配对为17.925640→14.301140μs、1.255940。两次结果不相减。
- `Default`和`MemoryDetail`的8张逐核CSV均非空，Device14、1个Vector block、1800MHz。Default核内6.436666μs，Vector1.160、MTE2 0.510556、MTE3 0.151667、Scalar3.272222、wait_ib2.491111μs；各计数可重叠。
- Torch profiler固定warmup5/active5，native每调用6个task、融合1个task；采集态kernel累计14.5204→5.056μs。这与事件时间、端到端step时间分开记录。

正式运行层已从serve_a3的DRY_RUN载荷清单构建，25个文件逐字节校验通过：
`local/dsv41-operator-stack-real:20261010-production`，digest `sha256:85d2d8b484a55f928729154f4e4f90eb4395dc143df5d95e10228d9f368a113d`。
`bench_real_tp8_model.py`使用正式权重的正常auto加载（拒绝dummy，记录有效loader）、Engram开启、384专家top6、4个同worker图bank。正式起服仍须使用serve脚本生成完整的Engram可写挂载，当前只读单算子probe不能直接当TP8服务。

当前缺少同机空闲健康8chip，正式TP8精度、客户端质量、端到端`(ms/step,A,tok/s)`与最优服务尚未验收；Goal保持active，未声称达到19ms。

最新复查a3-22已释放8–11，加14/15共6个空闲健康chip，仍缺2个。Occupancy仅有BasicInfo/二进制dump，解码的元数据没有容量占用值；TimelineDetail在kernel context/args dump及解析处失败，没有有效逐指令时间线。
