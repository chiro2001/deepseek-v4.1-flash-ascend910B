# 两条算子线叠加与TP8目标

目标：使用正式权重和用户已授权的空闲设备启动隔离TP8，精度/路由/输出与真实覆盖通过，端到端decode ≤约19 ms/step，同时报告A和token/s。
已创建active goal；独立分支`feat/operator-stack-tp8-20261010`。

## 当前状态（优先于下方历史记录）

用户已再次授权a3-21 physical chip8–15继续，仍为80C98001 Alarm；正式TP8已加载运行，不reset、不停止其他租户。生产容器为`dsv41-real-stack-tp8-20261010-a321`。

原生无候选bank的graph/eager/Engram子图关闭A/A均未通过完整路由与Top5 logprob门槛。最新`formal_attention_boundary_trace_v1`在8rank都发现layer0注意力输出进入HC post之前已不同，residual/post/comb一致；固定输入的HC pre80、HC post80、router40每rank重复5次全部稳定，共1600份真实输入。

`formal_attention_internal_trace_v1`已完成：8rank的归一化、Q/KV、可见cache、SMLA、O投影局部乘积均一致，首个偏离在wo_b AllReduce输出；固定局部向量重复10次，每rank后9次都与首轮不同，max abs0.0234375–0.03125。

更新：v1被错误的81次覆盖预期拦下；实测每步80次。修正后的`formal_fp32_decode_reduction_v2`退出0，三组A/A完整路由、token、Top5和logprob全一致/delta0，8rank×80个归约相对FP64参考转BF16逐位一致。新`formal_fp32_stack_graph_audit_v1`正在同worker图模式比较base/core/stack；尚未验收正式性能。后续受补丁源码与证据sha守卫保护。

最新：`formal_fp32_stack_graph_audit_v1`在建立候选前的base A/A拒绝通过，decode路由差1344、专家集合差888、共同logprob max delta1.624983，token相同。eager修复通过不替代图路径；正在准备真实向量的归约图重放及图捕获padding/dtype/守卫覆盖诊断。没有部署通过验收的正式服务。

用户提醒HCCL环境变量的历史解决记录后，已核对初始与更正记录，优先启动`formal_native_graph_strict_v1`：正式原生图、`HCCL_DETERMINISTIC=strict`在通信域初始化前设置，不启用FP32或候选bank，3组A/A结果待完成。真实向量的固定顺序普通/补偿求和六配置已全部通过独立与图重放精度，原FP32图重放4rank失败，仍不计整网收益。新增报告v9记录本轮顺序调整。

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
