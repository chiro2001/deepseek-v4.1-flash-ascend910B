# a3-21 chip8–15：正式 TP8 恢复验证

2026-10-10 · `feat/operator-stack-tp8-20261010`

用户重新指定 a3-21 physical chip8–15 后，本轮按该设备组恢复正式权重验证。8 rank 的 Vector、Cube、HCCL AllReduce 全部通过，正式 checkpoint 与 Engram 挂载检查通过；正式 TP8 首轮作业通过通信初始化，在模型构造阶段因漏传 lazy 加载参数退出，已修正并启动第二轮。整网精度和端到端性能仍待作业结果，尚不能宣称达到 ≤约19 ms/step。

## 本轮设备与模型

| 项目 | 结果 |
|---|---|
| 设备 | a3-21 physical chip8–15，TP8，8 die / 4 卡 |
| Health | 全部 Alarm；逐 chip 为 `80C98001`，RAS State / module error can not be fixed |
| 本轮功能测试 | rank0–7 的 Vector、Cube、HCCL AllReduce 均精确通过 |
| 资源隔离 | 启动前检查 8 chip 无设备进程；等待其他用户 chip14 短测试结束后启动 |
| 正式权重 | `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq` |
| 模型定义 | 40 层、hidden5120、384 routed experts、top6、W4A8_DYNAMIC |
| Engram | 开启，int8；4 个表文件的 O_RDWR 访问通过，未写内容 |
| checkpoint 预检查 | 90 个分片、辅助文件、软链闭包通过 |
| 私有容器 | `dsv41-real-stack-tp8-20261010-a321` |
| 镜像 | `local/dsv41-operator-stack-real:20261010-production` |
| 镜像 digest | `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea` |

功能冒烟通过说明本轮这些操作可执行，不等同于硬件 Alarm 已消除，也不等同于整网验证已通过。没有设备 reset、停止其他租户或清整机 page cache。旧 dummy TP8 容器保持停止。

config SHA256：`40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。

index SHA256：`726bd76e20f31743de40531cdc38ca861abee62042573530f8fac46436e0ac09`。

## 启动和验证修正

1. `launch_healthy_probe.py` 新增显式 `--allow-alarm`，只接受已核实的告警码，仍拒绝占用；新增 Engram RW 目录挂载，避免同一目录的 RO/RW 重复目标。
2. `bench_real_tp8_model.py` 明确使用 int8 Engram，默认关闭 CPU NUMA 迁移；新增 `--arms` 便于独立诊断。先完成两次原生请求，再创建候选 graph bank，区分原生启动问题与候选捕获问题。首轮报 `Engram HBM shards require --safetensors-load-strategy lazy`，补齐生产脚本已有的 lazy 和128线程加载配置。
3. `launch_real_tp8_job.py` 提供实时占用等待、任务锁、容器/芯片/挂载校验、环境与源码 hash 记录，并保存实际子进程退出码。任务目录由容器创建时，单独调整本任务目录属主。
4. 微架构采集入口按 8 rank 分别输出，避免复用 tiny 单 rank 的固定目录与名称。配对计时结束后再采集，路由/clone 审计与 profiling 不同时启用。

这些是启动与观测能力修正，没有产生新的算子性能结论。

## 两条优化线如何叠加

| Bank | 正式模型上的候选 |
|---|---|
| tp8base | 当前生产载荷的原生路径 |
| tp8core | 不可变 metadata 缓存、动态坐标/slot 准备融合、多 group slot 启动融合、Q/KV overlap |
| tp8act | core + 带形状和 dtype 守卫的 BF16 clamp/SwiGLU |
| tp8stack | act + Indexer K 投影后的 RMS/RoPE/量化/cache store 融合 |

metadata 优化减少 CPU 重复准备和小 kernel 启动；Q/KV overlap 用独立流和事件重叠无依赖分支；激活和 Indexer 融合减少中间物化、launch 和同步边界。收益必须由本轮同一模型、同一组 worker 的配对结果证明。

正式 HC static、顺序求和、Div-RN 已在原精度门槛失败，保持原生 HC。TP1 的 8-expert/top2 router 与 selected GMM 不匹配正式 384-expert/top6，不启用。正式 W4A8 可能已有 act_quant 融合，BF16 候选的实际覆盖需由本轮 graph bank 统计确认，不能以独立用例通过替代覆盖。

## 采集计划与判断口径

按 CANNBOT `ops-profiling` 的计数组约定，分别采集 PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio。整网通过 torch_npu profiler 采集 CPU/NPU 与通信，每 rank 独立目录；每个请求先运行9步，再做5步 profiler warmup和10步 active采集。

先定位真实 W4A8 的 task、图间隙、同步、HCCL、MoE、attention 与 Indexer 热点；需要逐核/逐指令细节时再对选中热点 replay。缺失、NA 或工具失败的数据明确保留为未采得，不用0替代。

GM/L2↔UB 的流量和带宽不能代替 UB 容量占用；Cube 的 L1、L0A/B/C、MTE1 只用于实际 Cube 工作核。pipe/wait 计数可能重叠，累计 kernel 时间不等于端到端 step。UB 双缓冲优先考虑有多 tile 搬运循环且 MTE 在关键路径上的热点；此前单行 Indexer 的低搬运利用率不足以支持优先添加双缓冲。

## 当前验收状态与证据

- 首轮正式作业：`/work/results/real_tp8_audit_v1`，退出码1；8 worker 完成 HCCL/Gloo 初始化，进入 `REAL_WEIGHT_LOADER` 后被 Engram lazy 参数守卫拒绝，未完成权重加载。
- 第二轮正式权重审计：`/work/results/real_tp8_audit_v2`，已启动，等待结果。
- 正式端到端 `(ms/step, A, token/s)`：尚无本轮有效结果。
- 正式 TP8 API 与客户端质量：尚未完成。
- 源码自检：源码提交后生成 MANIFEST，整包 `tools/selfcheck_pkg.sh` 全部通过；日志随证据交付。初次未提交改动时的两项 hash 差异已解决。

紧凑证据位于 `experiments/operator_stack/evidence/formal_a321/`：allocation、checkpoint、8 个 rank 冒烟与正式作业 launch receipt，传输 hash 记录在 `fetch_manifest.json`。后续正式结果补入本报告；历史 tiny 与单算子数据不充当本轮整网结果。
