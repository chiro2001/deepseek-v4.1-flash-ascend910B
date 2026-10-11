# IDS64 hoist：真实 Cast 覆盖通过，tiny 端到端回退

2026-10-11，a3-21 physical chip4。独立 tiny 作业 `ids_audit_perf_v3` 退出 **0**。保持 `V41_IDS64_HOIST` 默认 **off**，本结果不支持正式开启，也不构成正式 TP8 或 ≤17ms 声明。

开关确实把每图 40 次路由输入的 int32→int64 转换移到模型入口的一次转换，40 个真实消费者的整数值不变。三组原生 A/A、三组重新捕获后的 A/A、六组 off/on 的完整有序路由、token、Top5 键集合一致，最大 logprob 差 **0**。关闭审计后的 12 组交错配对却全部 on 更慢。

| 同实例、profiler/路由导出/logprobs 全关 | ms/step | A | tok/s |
|---|---:|---:|---:|
| off | 28.862625 | 1 | 34.646883 |
| on | 28.983470 | 1 | 34.502425 |

配对 `off−on` 中位 **−0.108590ms**，on 获胜 **0/12**；20,000 次固定 seed bootstrap 的配对中位 95% 区间为 `[−0.126960, −0.087595]ms`，仅描述这 12 组 tiny 样本。两个臂各自中位之差与配对中位不是同一统计量。不能将减少的 Cast 次数或采集态累计算子时间当作端到端收益。

## 源码与真实消费者

正式容器只读核查记录在 [source_audit_final.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/source_audit_final.json)。正式和原 tiny 的消费者源码 SHA256 相同：`f32950da16926afe7569b1a6daa437860db2e83171d96623ac79f7734e52b004`。

- `vllm_ascend/models/deepseek_v41/model.py:57` 在模块导入时读取 `V41_IDS64_HOIST`，默认 0。
- 同文件 `:1441` 在层循环前把已处理 placeholder 的 `moe_input_ids` 转为 int64，然后向各层 MLP 传递同一张量。
- `vllm_ascend/ops/fused_moe/router/fused_topk_router.py:159` 只在 `sqrtsoftplus` 且 `tid2eid` 或 `bias_vl` 非空时消费 token IDs；`:164` 每层执行 `.to(torch.int64)`，随后 all-gather 或 pad/split，最终传入 `moe_gating_top_k_hash`。

这是 **token ID 输入**转换，选出的专家 ID/dispatch 索引没有因这个开关改变。int32→int64 是精确整数扩展；原来的 `-1→0` 处理仍在原位置。通信消费者看到的 dtype/shape 保持 int64。仅在运行中修改环境变量不会重读模块布尔值，已编译图也不会因此自动更新。

tiny 配置为 `num_hash_layers=0`、`vision_n_layers=32`，实测 40 层 `bias_vl` 都非空、`tid2eid` 都为空，因此 40 层全部覆盖该消费分支。其他配置的覆盖数量必须另测；源码中“40 次”注释不能替代实际见证。

## 模型和执行口径

| 项目 | 本次 tiny |
|---|---|
| 硬件 | a3-21 Ascend910；physical4 = NPU2/die0，PCI `0000:95:00.0` |
| 规模 | 40 层、hidden5120、64 heads、head_dim512；8 routed experts/top2、FFN256 |
| TP/EP | TP1/EP1；不覆盖正式 TP8 多 rank 规约 |
| 权重/量化 | 未量化 BF16 dummy；gate/HC 以固定 seed20261011 随机化，保持原生算子 |
| 输入/路由布局 | decode token IDs `[1]`、ND、stride1、连续；off 为 int32，on 共用 int64；路由 top2 `[1,2]` |
| 图 | FULL_DECODE_ONLY，capture `[1]`，各次重捕获使用独立 graph pool |
| 骨干编译 | 两臂均仅绕过骨干 Dynamo 的无 guard 包装；原生 forward 被完整 NPU 图捕获/重放 |
| 请求 | 三个 2048-token 文本输入；审计生成 47/48 tokens，性能生成48；temperature0 |
| cache/调度 | BF16 KV、block128、固定4GiB；MAX_LEN8192、MAX_SEQS1、BAT2048；同步单进程 |
| 其他 | Engram/DSpark/prefix 关闭；HCCL_DETERMINISTIC=strict，LCCL_DETERMINISTIC=1，OMP1 |
| 计时 | `engine.step()` 墙钟，跳过前9步，取后续 decode 步中位；A=1 |

骨干编译处理保证同实例重捕获能真实切换现有 Python 全局开关；它是本诊断的明确限制，不代表正式编译布局。dummy 模型不能用于真实文本质量或正式384专家/top6、W4A8、TP8、通信/Engram收益的外推。

## 精度和实际 Cast 证据

Capture-only 钩子只保存张量引用；capture 内不执行 D2H、`.item()`、`.tolist()` 或设备探针。整数检查在请求完成后、图外进行。完整路由文件保留远端，终态分析另外逐字节比较了全部 12 对 `.npy` 文件，覆盖 **1,005,280** 个重复计数的 token×layer 位置，不能解释为独立样本数。

| 每个 decode 图 | off | on |
|---|---:|---:|
| 真实路由 consumer | 40 | 40 |
| router material ID Cast | 40 | 0 |
| router 同 dtype no-op | 0 | 40 |
| 模型入口 material ID Cast | 0 | 1 |
| 路由 Cast 输出 storage 数 | 40 | 1 |

无审计性能图重新捕获前，同时关闭 ModelConfig、`routed_experts_initialized`、scheduler 路由导出和40个 router callback；每个性能输出断言 `routed_experts=None`、`logprobs=None`。性能结束后，两臂分别另采10步 PipeUtilization，窗口只用于覆盖与归因。

| 每步实际 kernel CSV 计数 | off | on |
|---|---:|---:|
| 全部 Cast | 162 | 123 |
| `[1]` int32→int64 Cast | 71 | 32 |
| HcPre | 80 | 80 |
| MoeGatingTopKHash | 40 | 40 |

两份 CSV 各有800个HcPre、400个hash router，Device_id全部为4；实际少 **39 Cast/step**。trace JSON、op/api统计齐全。采集态 scalar int32→int64 Cast累计90.130→37.929µs/step，只是 profiler 算子归因，不作为无 profiler 报价。

## 证据、失败历史和设备释放

[verification_summary.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/verification_summary.json) 是紧凑结论；[accuracy_gate.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/accuracy_gate.json)、[consumer_witnesses.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/consumer_witnesses.json)、[performance_pairs.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/performance_pairs.json)、[profile_validation.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/profile_validation.json) 保留门控和原始配对统计。

远端不可变根为 `/home/l00886679/projects/dsv41-tiny-prof-20261009/ids_lane_private_v3`，作业目录 `results/ids_audit_perf_v3`；源包、完整 routes、raw profiler/trace 均留远端。SSH只取139,067字节的终态紧凑包及76,202字节的失败/释放补充包，没有传输≥1MB文件。

- `v1` 退出1，NPU初始化前缺 `acl`；私有 PYTHONPATH 追加 SDK 搜索路径后新建v2，未覆盖v1。
- `v2` 完整精度门通过，性能预热断言拒绝残留路由导出；定位独立 `routed_experts_initialized` latch。未报价。失败日志、全部精度门及源码哈希在 [history/v2](../../runtime/evidence/ids_lane/ids_audit_perf_v3/history/v2)。
- `v3` 新建快照，完整重跑精度后关闭所有导出状态，12组性能及两份10步profile成功，退出0。解析器适配了实际为事件数组的 trace JSON；仅重新解析文件，没有重跑NPU或改采集窗口。
- 新建非 privileged `dsv41-tiny-ids-20261011-c4-v1` 曾创建排队；最终只运行 `sleep infinity`。唯一 NPU 作业实际在旧 privileged c4 容器下用私有根和进程 `visible=4` 启动，没有并发第二份作业。

终态即时复查 physical4 Health OK、NPU2/die0无进程，HBM3291/65536MB，记录见 [chip4_release.json](../../runtime/evidence/ids_lane/ids_audit_perf_v3/chip4_release.json)。主线程已获释放确认；此 lane 不再启动 chip4 作业。未提交代码，未修改正式源码、STATUS或MANIFEST。

`launch_ids_job.py`、`bench_ids_model.py`、`ids_worker.py`、`ids_patches.py` 和 `instrument_snapshot.py` 提供私有模型复现入口；`analyze_ids_profiles.py` 做终态覆盖校验，`collect_ids_evidence.py` 只取紧凑证据。将来复现须先有新的芯片分配与实时健康/占用核查。
