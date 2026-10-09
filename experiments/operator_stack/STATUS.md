# 两条算子线叠加与TP8目标

目标：a3-21 physical chip8–15的隔离TP8，精度/路由/输出与真实覆盖通过，端到端decode ≤约19 ms/step，同时报告A和token/s。
已创建active goal；独立分支`feat/operator-stack-tp8-20261010`。

两条线：

- 主线`aca0af9`：TP1 tiny mdfull已验证19.209ms/A1/52.058tok/s，含HC/router、激活、Q/KV多流、GMM1融合、metadata及多group slot启动。
- 950迁移线`3d2e4a7`：六方向已处理，最终仅Indexer INT8 K后处理融合保留，TP1配对收益约0.3–0.5%；A2验证未完成。其他负结果不重复筛选。

当前模型按上下文暂用同一40层/hidden5120/8专家top2 BF16 tiny dummy；已询问用户是否切换正式checkpoint，若收到说明则调整依赖工作。
TP1限定的selected-expert/GMM/路由路径不得直接声称TP8生效，需按本地专家布局核对覆盖与回退。

资源初查：chip8–15无设备进程，但health均Alarm，已查询的8/9/14/15为80C98001 AIC RAS模块不可修复错误；其余4个health及功能冒烟正在核验。
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
