# tiny真实量化与正式前缀对齐通过，转向异步实际模型步采集

2026-10-11 · a3-21 chip8–15 · 当前目标正式客户端 **≤17ms/step**。

真实8层tiny已通过数值审计：三组八层原生A/A的完整路由、token、Top5/logprob全一致，最大差0；三个正式2048-token输入、含重复运行的七次前四层prefill路由比较逐位一致。每rank八个W4A8 MoE内层scheme、44个W8A8 Linear确认。它可用于对应算子与缓存几何的快速诊断，不能代替完整40层正式客户端验收。

## 对齐证据与性能边界

`tiny_tp8_alignment_audit_v3`退出0，源 `/work/src_tiny_align_v7`，真实权重来自正式checkpoint逐张量字节复制。TP8/EP8、384专家/top6、本地48专家、hidden5120、FFN2304、两张Engram int8表及原层号哈希种子保持。五cache组、三slot布局与正式十二cache组、四slot不同；后半段激活、截断模型输出以及自生成decode输入也不同。

七次前缀检查覆盖57,344个token×layer位置、344,064个专家ID，均无差异。三组原生A/A覆盖47/48输出的C2末步状态，路由顺序/集合、token、Top5键集合与logprob全一致。完整topology和比较JSON已归档；只使用路由对齐作为见证，没有把它扩张成全部中间激活一致的声明。

审计期间13.535782ms包含路由和Top5观测，不作为性能。随后独立 `tiny_tp8_profile_v2` 退出0，在同一模型实例中六组无审计原生重复测量为 **(12.898705ms/step,A=1,77.527163tok/s)**，只针对八层同步诊断，既不是新候选收益，也不是正式17ms结果。两次测量不相减。

构造/加载/预热耗时分别421.855秒、463.539秒。只缩层没有消除完整Engram映射和初始化固定成本，尚不能宣称快速重启目标完全实现。后续算子试验应优先复用驻留worker或导出真实输入和权重做单算子重放，减少整模型反复启动。

## 七组微架构与异步采集的修正

tiny无审计计时后已完成七组×八rank，共56份原始profiling目录：PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio。原始数据留远端；56份CSV已解析，全部计数窗口均为10步；原时间线分析器硬编码40层82次规约，因8层实际18次拒绝继续。现增加显式 `--layers=8` 并核对诊断result身份，每rank每step仍要求精确18次；默认正式40层仍严格82次。`formal_async_profile_controller_v2` 继续汇总时间线并启动正式异步采集。56份产物进一步验收通过：每份10个step、160次HcPre（每步16），trace JSON完整，op/api统计存在；PipeUtilization为47列，其他计数组为29/32/34/38列并带各自计数。每rank每step精确18次规约。UB容量occupancy不作声明。

rank0诊断中GMM1 Scalar比例0.6163、GMM1约52.57μs/调用，GMM2约33.64μs，HC pre约26.71μs；与原正式小M、48-local专家的控制热点相符。规约采集态含显著长尾/跨rank等待，不能将累计3812.85μs当成可回收端到端收益。

正式当前最快已是异步metastack，但旧七组计数来自同步路径且逐步RPC推进，会扰动图间隙。按 cannbot `model-infer-profiling` 技能核对安装态框架后确认：NPUWorker在每个真实 `execute_model` 前调用 `WorkerProfiler.step()`，现有TorchNPU包装器却没有推进torch_npu的schedule。

新CounterProfiler利用框架已有的worker步回调推进异步采样；前端仅在实际token到达后开启/关闭窗口，不逐步发RPC。step clock有界到 `warmup5 + active10 + 收尾1 = 16`，每rank落盘实际worker调用数、schedule步数和完整性。同步手动路径也补齐收尾步。前端轮询一万次不改变worker计数，禁止将手动advance混入异步worker驱动。

CPU集成预检使用安装态WorkerProfiler控制流与记录用替身backend，通过两种模式；新预检还断言没有初始化NPU设备。它只证明驱动和边界逻辑，不能代替实际NPU采集。正式NPU窗口将在后续任务验证CSV、真实decode锚点和八rank完整步数。

## 当前接续状态

`formal_async_profile_controller_v2` 会先验证tiny审计/无审计终态，解析56份目录并生成微架构摘要，再复查芯片无占用与已授权80C98001 Alarm，启动 **`formal_async_profile_v1`**：完整正式40层、strict、原生base与metastack、12组关闭profiler交错配对，之后七组×两bank×八rank异步采集。源快照 **`/work/src_async_profile_v3`**，实际runner748485已验证存活，完整正式作业已经启动。profiling与正式报价分开，不停止他人、不reset。

正式已验收客户端仍是 **(19.376255ms/step,A=1,51.609559tok/s)**，GSM8K100/100、视觉23/23。API为实验释放；新的性能、质量与最优部署验收尚待完成。当前未达到17ms，下一步依据正式异步关键路径选取候选，不复用已否定的prefix、FP32规约或TP1专用算子。

代码、紧凑证据与报告继续提交，从已提交源码生成MANIFEST、自检，双远端push并发布COS/links-server。完整17ms目标保持执行中。
