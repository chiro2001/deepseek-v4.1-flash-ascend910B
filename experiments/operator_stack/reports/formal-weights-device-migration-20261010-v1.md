# 正式权重迁移、健康设备验证与微架构优化尝试

2026-10-10 · `feat/operator-stack-tp8-20261010`

本轮已避开 a3-21 的故障 chip，换到 a3-22 的健康 chip14/15，完成基础计算、两 rank 通信、正式 checkpoint 检查及单算子验证。正式权重上的 Indexer 后处理融合可保留；HC static 及两种修正未通过原精度门槛，正式入口保留原生 HC。新增的单系数量化变体通过精度，但没有稳定性能收益，未采用。

**正式 TP8 的端到端目标仍未验收。** 最新资源复查中，a3-22 空闲健康 chip 为 `8,9,10,11,14,15`，仍缺同机的两个 chip。正式 TP8 的 `(ms/step, A, token/s)` 尚无有效结果，也未交付正式 TP8 服务。Goal 保持 active。

## 1. 本轮模型与范围

正式 checkpoint：

```
/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq
```

| 项目 | 核实结果 |
|---|---|
| 模型 | Deepseek V4.1，40 层，hidden 5120 |
| MoE | 384 个 routed experts，top6，intermediate 2304 |
| 量化 | Ascend `W4A8_DYNAMIC` |
| Engram | 层 1、14；辅助文件与 softlink 闭包通过检查 |
| index 文件引用 | 90 个分片全部存在，合计 525,654,346,514 bytes；不使用 `total_size=0` 推断大小 |
| config SHA256 | `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424` |
| index SHA256 | `726bd76e20f31743de40531cdc38ca861abee62042573530f8fac46436e0ac09` |

没有随机化正式权重，没有使用 `load_format=dummy`。单算子验证读取正式 HC、Indexer 参数；输入为合成数据，尚不是实际整网激活。所选 tensor 的内容 hash 已记录，未对全部大分片做内容 hash。

Indexer 的 K 投影是模型内明确分配为 BF16 的 `nn.Linear`；k_norm 文件权重为 F32，按原模型的 BF16 参数类型加载。验证使用相同的加载转换，不修改训练参数。额外三层 MTP 参数单独区分，主干 HC 数量是 80，而不是文件中包括 MTP 的 86。

历史 tiny 的配对结果仅用于解释两条线的关系：主线为 `(19.338175 ms/step, A=1, 51.711188 token/s)`，加 Indexer 为 `(19.101980, 1, 52.350594)`，配对加速中位约 1.002596。它不能作为正式 W4A8 的结果，也没有证明很大的叠加增益。

## 2. 换设备与启动修正

| 机器 | 状态与处理 |
|---|---|
| a3-21 | chip8–15 均有 RAS Alarm，全部排除；旧 dummy TP8 容器已停止。0/1、5 有现存任务；2/3/6 有历史启动失败，不再试用这些问题设备 |
| a3-22 | 所有 chip 的 Health 为 OK；先使用空闲 14/15，最新复查又释放了 8–11，目前共 6 个空闲健康 chip |
| 920B-47 | 服务节点，无可见 NPU |
| ysy21/22 | SSH 鉴权拒绝，未取得硬件信息 |

没有重置设备、停止其他租户或清整机 page cache。

启动时解决了三个实际问题：

- Engram 的绝对软链经过 `/home/l00886679/projects/dsv41/models/...`。只挂最终解析目录会缺少中间路径，按 `tools/model_mount_args.sh` 补齐所有字面跳转路径。
- torchrun 默认使用无法在容器解析的 `host22`，改为 `--master-addr=127.0.0.1`。
- 严格透传 `/dev/davinci14/15` 时，ACL 将它们枚举成逻辑 0/1；运行时过滤值必须为 `0,1`，物理映射单独记录。详细计数采集改用仓库标准 NPU 容器配置，过滤物理 `14,15`；有效 msprof CSV 的 `Device Id=14` 已验收。

两个 chip 的 Vector、Cube 和两 rank HCCL AllReduce 全部精确通过。独立采集脚本还补上框架的 `bootstrap_custom_op_env` / `enable_custom_op` 初始化；原生 HC vendor 包本来就在镜像中，直接加载基础动态库不足以注册完整运行环境。

## 3. 精度与优化尝试

| 方案 | 验证 | 结论 |
|---|---|---|
| 原 Indexer 后处理融合 | 4 组真实 wk/gamma，page0/page1 各 64 例，共 128 例 | INT8、FP32 scale、INT8/FP16 cache 全部逐位一致，保留 |
| 正式 TP8 激活形状 | `[1,576]`、`[2,4608]`、`[8,4608]`、`[16,4608]`，共 96 例 | 通过原全元素与最大 1 BF16 ULP 门槛，待整网覆盖与审计 |
| HC static | 真实主干参数；94 例后第 95 例失败 | 正式入口回退原生 |
| HC 顺序求和 | 按原生 row0→row1→row2→row3 求和 | 同一失败仍存在，不采用 |
| HC Sigmoid Div-RN | 根据原生 Exp/Adds/Div 实现修正 pre 的除法舍入 | 同一失败仍存在，不采用 |
| Indexer rank0 单系数 | 每行一次 Div-RN | Ascend MLIR TypeRange 断言，编译拒绝 |
| Indexer 单元素 Tensor 系数 | 避开 rank0 编译缺陷，再广播系数 | 128 例逐位通过，性能无稳定增益，不采用 |

Indexer 用例覆盖零输入、尺度 `0.001/1/100`、BF16/FP32 RoPE、1/4 行、无效 slot、不同 cache page 和 page 间距。最初的 RoPE 维数和 cache token 间距不符合原生接口约定，已修正测试夹具；没有改变精度门槛。实际支持的布局是页内 token 连续、页间可以存在间距。

HC 失败发生在 `layers.15.hc_ffn_fn`，尺度 100、无 pre_mix。一个 BF16 元素的绝对差为 0.25、相对差约 0.005181，超过原 `rtol=atol=0.004`；其余 FP32 输出仍按原 `1e-4` 门槛检查。保存了失败输入及结果用于复现。

复现中，pre 系数最大差只有 `1.1920928955078125e-7`，却触发 BF16 舍入边界。用原生 pre 做三种求和，均与原生 y 一致；用候选 pre 做三种求和，均有同一个元素不同。因此顺序求和不是这次差异的主因，修正 Div 也不足以解决。后续需要核对投影、归约、RMS、Exp 等中间边界，不能用放宽容差来迁移。

## 4. 同进程单算子性能

以下是 **Indexer K 投影之后的后处理**，不包含投影、整网、通信或客户端开销。真实权重生成 projected input，关闭 profiler 后在同进程正反交错计时，每个图 20 次调用、50 次重放、8 组配对。

| Case | Shape | DType | 原生后处理 μs | 原融合 μs | 单系数 μs |
|---|---|---|---:|---:|---:|
| 三路径同进程 | `[1,128]` | BF16 | 18.159349 | 14.309020 | 14.292750 |

原融合相对原生的配对加速中位为 **1.271290，8/8 更快**。单系数相对原融合的配对加速中位为 **0.999512，4/8 更快**，没有稳定改善，继续保留原融合。

之前独立两路径实验为 17.925640→14.301140 μs，配对加速中位 1.255940、8/8 更快。两次实验各自采用同进程对照，不跨会话相减，也不把单系数的整体中位差当作收益。

单算子的 `(ms/step, A, token/s)` 不适用；正式 TP8 这组三元组仍未测量。上述单算子速度不能外推为端到端提升或 19 ms/step 达标。

## 5. profiling 与 910C 微架构证据

### 采集方法与有效性

采用已读取的 cannbot torch profiler 参考：`torch_npu.profiler` 固定 `warmup=5, active=5`，原生和融合各独立采集。单算子逐核使用 `msprof op`，`--kernel-name=post_kernel --launch-count=1 --warm-up=5`，分别采集 `Default`、`MemoryDetail`。精度检查中的 debug-store kernel 不进入该窗口。

首次非特权隔离配置的通道 47/FFTS 采集失败，只留下 BasicInfo，不计有效数据。换为仓库标准 NPU 采集配置后，Default 与 MemoryDetail 各有 8 张非空 CSV；均为物理 Device14、1 个 Vector block、1800 MHz，并记录 SHA256。工具退出码 0 本身不是验收判据。

Torch profiler 显示原生每调用 6 个 task：RMSNorm、RoPE、DynamicQuant、2 次 Scatter、1 次 Cast；融合只有 1 个 `post_kernel`。5 个 active 调用为 30→5 个 task。采集态 kernel 累计为 **14.5204→5.056 μs/调用**，与无 profiler 的图事件时间、端到端墙钟分别记录。

### Default 逐核数据

| 指标 | 值 |
|---|---:|
| Task duration | 6.960139 μs |
| AIV 核内时间 | 6.436666 μs |
| Vector | 1.160000 μs，约 18.02% |
| MTE2 | 0.510556 μs，约 7.93% |
| MTE3 | 0.151667 μs，约 2.36% |
| Scalar | 3.272222 μs，约 50.84% |
| scalar wait_ib | 2.491111 μs |
| scalar wait | 0.652222 μs |
| aiv_icache_miss_rate | 0.092771（工具原值） |
| Vector 总资源冲突比例 | 0.011566 |
| L2 read hit rate | 61.904762% |
| L2 total hit rate | 60.000000% |
| GM→UB 数据 / 利用率 | 1.000 KB / 0.067328% |
| UB→GM 数据 / 利用率 | 0.250 KB / 0.019829% |
| UB Vector 读 / 写带宽 | 10.519556 / 7.852627 GB/s |

这些 pipe、wait 计数允许重叠，不能相加来分解总时延。带宽是工具逐核路径口径，不能称为整芯片 HBM 利用率。这里的 L2 命中率是独立 kernel 重放结果，不是正式模型命中率。

MemoryDetail 的 task 为 6.860137 μs，AIV 核内 6.322778 μs；GM→UB / UB→GM 为 0.150832 / 0.037708 GB/s，工具路径利用率约 0.068541% / 0.020186%；UB Vector 读/写为 10.709040 / 7.994071 GB/s。带宽吞吐不代表 UB 容量占用。

此 kernel 运行在 Vector core，使用 GM/L2↔UB 搬运。Cube 的 L1、L0A/B/C 与 MTE1 指标为 NA，不能以 0 或这些 NA 推断模型所有 buffer 的利用情况。

### 延迟与双缓冲判断

这个位置的 MTE2/MTE3 占比较小，Scalar、wait_ib 和指令供给更值得检查。每个 program 只处理一行的 128 维数据，没有跨多个 tile 的搬运循环，追加 UB 双缓冲的优先级较低。需要优先检查生成代码、控制/同步链与指令缓存，而不是仅扩充 buffer。

基于这一判断，实际尝试了“每行只求一次量化系数再广播”；精度通过但配对性能无改善，因此没有把理论上的指令减少当作实测收益。

Occupancy 与 TimelineDetail 也已尝试。Occupancy 保留 BasicInfo 和 dump，尚未得到可验收的 UB 容量占用值；TimelineDetail 明确发生 kernel context/args dump 与解析失败，未形成有效逐指令时间线。不能据此定位每一条指令的关键路径。原始 dump 保留在远端，较大文件不经 SSH 传输。

## 6. 正式 TP8 入口与交付状态

新增正式入口 `bench_real_tp8_model.py` / `real_tp8_worker.py`：按正式服务的正常 auto 权重加载并记录有效 loader、Engram 开启、384 专家 top6，禁止随机化验证权重和 dummy 加载。原生、metadata/QKV、激活、Indexer 组合使用同一组 8 个 worker 的图 bank；审计时完整检查实际 top6 路由、token/logprobs、缓存写入与每 rank 覆盖。正式计时关闭审计与 profiler。

TP1 的 8-expert router、selected GMM 与路由特化不迁入正式路径。HC 候选已明确关闭。共享激活的回退绑定原模块实现，避免用重新拼接的实现替代正式基线。实际 routed W4A8 路径若已融合激活或 dtype/layout 不匹配，会按守卫走原生，覆盖必须由 8rank 结果证明。

从 `serve_a3.sh` 的 DRY_RUN 载荷清单生成私有生产层，25 个文件逐字节校验通过：

```
local/dsv41-operator-stack-real:20261010-production
sha256:85d2d8b484a55f928729154f4e4f90eb4395dc143df5d95e10228d9f368a113d
```

该构建不加载模型、不使用 NPU。正式起服必须使用 serve 脚本的完整模型挂载与 Engram 可写目录，当前只读单算子 probe 不是正式 TP8 服务。

最新 a3-22 已有 6 个空闲健康 chip，仍不足 TP8。未抢占其他进程、未在故障设备上重试，也未将 tiny 或单算子证据算作正式整网通过。

## 7. 仍未完成的优化机会

1. **正式 TP8 的整网基线和叠加审计。** 先完成健康 8chip 的真实权重加载、质量、路由与缓存审计，再做同 worker 配对，报告完整 `(ms/step,A,token/s)`。
2. **正式 W4A8 的真实热点。** 采集 8rank 的 CPU/NPU task、HCCL 与图间隙，分清 metadata、launch/同步、MoE dispatch/GMM、attention/Indexer 和通信的实际占比。
3. **HC 的数值边界。** 继续核对原生投影、跨核归约、RMS、Exp 和 BF16 舍入；当前三种候选均拒绝，不能从 tiny 成功直接推到正式权重。
4. **激活的实际覆盖和收益。** 独立形状通过，但 W4A8 中可能已被原生融合，需要用真实 graph bank 验证覆盖，再按配对结果选择。
5. **有多 tile 的搬运流水。** UB 双缓冲应在真实 GMM/attention 等循环工作集上、按 MTE 和指令时间线判断；本次单行 Indexer 证据不支持优先添加。

已有负结果的上游 950 GMM/SMLA/投影/epilogue/UB→L1 方案没有重复筛选。A2 实机迁移仍未完成。

## 8. 证据与复现

仓库小型证据位于 `experiments/operator_stack/evidence/formal_a322/`、`formal_a322_profile/`。失败、复现、编译拒绝与有效结果均保留。远端根为：

```
a3-22:/home/l00886679/projects/dsv41-real-operator-probe-20261010
```

关键命令（健康设备映射和容器归属须先核对）：

```bash
bash /work/src/stack/run_stack.sh formal_model_contract.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --output /work/results/checkpoint_NEW.json
bash /work/src/stack/run_stack.sh verify_real_operators.py --section=indexer \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --output /work/results/indexer_NEW.json
bash /work/src/stack/run_stack.sh bench_real_indexer.py --compare-scalar --profile \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --output /work/results/threeway_NEW
bash /work/src/stack/run_stack.sh summarize_real_profiles.py \
  --root /work/results --physical-chip 14 --output /work/results/validation_NEW.json
```

采集参考技能：`/home/chiro/projects/dsv41/layer_bench/cannbot_templates/SKILL.md` 及其 `REFERENCE_PROFILER_AND_METRICS.md`；逐核命令参数以当前镜像 `msprof op --help` 为准。有效 Default/MemoryDetail CSV 的设备、核数、行数、频率及 SHA256 已校验，退出码 0 但没有所需数据的尝试不计成功。
