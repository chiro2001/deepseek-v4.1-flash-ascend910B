# 正式 TP8 含 DSpark 20 ms/step：缓存修复与 SDK/GMM 门进展

2026-10-11。新 goal 为 active，正式包含 DSpark 的 ≤20 ms/step 尚未达成。
本轮没有授予完整消费者审计、正式性能或最终部署通过。

## 已完成与下一步

| 线路 | 本轮实际结果 | 下一步 |
|---|---|---|
| DSpark tiny，physical4 | 固定 before-capture O 投影缓存；20 请求、9 项数值检查全部通过；仍全部周期 2 复读 | 新诊断窗口关闭计时中的逐层 D2H/hash/audit，测原生 OFF/ON 的包含 DSpark 时延 |
| SDK 正常 TP8 worker，8–15 | 8 个正常 worker 初始化成功；修正 overlay 身份误拒绝后，已核实 rank6 的既有 context/stream 和 logical6→physical14；在懒加载对象源码检查处失败，尚未 compile/capture | 新版本只修 SDK 懒代理类与 backing module 的来源检查，再使用全新 worker |
| 稀疏 GMM，physical6 | 修复 HostArgs 第 8 参数；9 case、324 事件样本、406 次数值检查通过，maxULP=0；没有统一的 candidate-vs-raw 收益 | 推迟当前机制的整网接入；保留原生 fallback 的 CPU 设计与真实几何资料 |

三条子代理线路均正常交付，继续使用已核验的 gpt-6.1-sol/max。
每次 NPU 启动均重新检查占用和健康；本轮结束的 physical4、6、8–15 作业已释放。
8–15 的 Alarm 仅放行用户已授权的 80C98001。没有 reset、停止其他租户、清 pagecache 或删除设备内部文件。

## tiny 固定缓存与真实回放

`graph_fixture_v1` 在编译、profile 和 capture 之前，为 target40 与 draft3 按原生 permutation 创建 43 份派生 O 投影缓存，注册非持久 owner buffer，保留原始参数与算子顺序。
此前 target replay 后派生缓存出现非有限值；本轮 capture 后及每请求后，43 份缓存与源参数的 pointer/layout/value/finite 检查均保持，共 860 次逐项 seal。

20 请求为 2 warmup + 18 审计请求；9 项 A/A、OFF/ON 检查全部通过。完整 token、43 层路由、3 层 attention、proposal、logical cache bitwise 一致；从实际请求重算 Top5/logprob 最大差值为 0。

冷 helper 调用/返回为 1/1，实际 draft entry 为 1，descriptor 为 6×1。
每个 OFF 请求有 17 次 helper 返回、16 次 target replay、0 次 draft replay；每个 ON 请求有 1 次 initial eager helper 返回、16 次 target replay 和 16 次 draft replay。
各请求 initial/terminal `_propose` 均为 1/1；ON 的 terminal replay 为 1，终态 device fence 已完成。

全部 20 请求仍周期 2 复读，性能评估禁用，tiny/formal admission 均为 false。
该数据含大量诊断 D2H，不报价；dummy 的接受长度不能当正式吞吐收益。
新的低负担窗口仍保留固定 owner fixture、实际 rounds/propose/replay 和 terminal fence，并在计时窗外核验全部参数与缓存。

证据：`parallel_tiny/dspark_lane/evidence/dspark_lane_20261011_c4_graph_fixture_v1_analysis.json`。
194 个取回文件 SHA 校验通过；本轮 10 个 launched source、v7 冻结 37 文件均保持。

## M1 事件门：数值通过，收益不统一

第一个事件窗口在 raw native stop-event 同步失败，无完整 case 或性能样本。
实际源码差异是 HostArgs 第 8 参数误填输出 GM 本体，而 v5 成功路径要求输出 tensor-list 的 GM descriptor。
原失败源码和日志已冻结。

独立 v2 修正该参数。使用 v5 的 27 组实际参数/descriptor 快照编译检查，共 1801 项通过；27 个误放输出 GM 的反例均拒绝，编译原 v1 在 offset56 正确拒绝。
上机先完成一次无 Event 的 raw numeric smoke，再执行 9 case 的 3 路轮换 warmup/采样。

SDK/raw/candidate 数值完成分别为 135/136/135；raw 含 smoke1。planner 接受 135，324 测量样本按同 case/round 配对保存。
406 次数值均 maxULP=0，406 次 args/descriptor seal 通过；完整 E48 seals、依赖 SHA 与 canary 门通过。
cleanup=0，owner_retained=false，physical6 健康释放，无 retry/timeout。

| Case | raw operator 中位 µs | candidate operator 中位 µs |
|---|---:|---:|
| R1_e0 | 30.44 | 33.02 |
| R1_e47 | 39.29 | 33.04 |
| R3_e0 | 30.95 | 33.07 |
| R3_e47 | 39.50 | 32.80 |
| R3_three_slots | 37.33 | 45.24 |
| R6_e0 | 30.53 | 32.74 |
| R6_e47 | 39.64 | 33.10 |
| R6_mixed | 38.09 | 38.96 |
| R6_six | 46.72 | 52.08 |

candidate 仅 3/9 case 的 operator 中位比 raw 更快，fresh-inclusive 为 4/9。
candidate 比 SDK 的 operator 中位 9/9 更快，但该设备/队列区间包含 host 提交空闲：SDK 提交中位 93.13–109.67 µs，raw 12.53–13.84 µs，candidate 15.40–20.18 µs。
这不能解释为数学核或已入图整网的可回收收益。planner 包含在 candidate 区间；输入每次在同 GM 地址恢复，stop 完成后才进行 D2H/数值/封印检查。

完整结果与可重算脚本：`parallel_tiny/gmm_lane/sparse_msd/m1_event_perf_v2/`。

## K5 实际几何与覆盖率

正式 profile 的 target 为 40 个 GMM1/GMM2，容量 M36/local48；draft 为 3 个，容量 M15/local16。
实际 SDK `patch_speculative_config.py` 会把 draft 的私有 HF/model_arch config 转为 checkpoint 原生的 128 专家/top3；target 保持 384/top6。不能把 main 的 top6 或 M1/capacity6 二进制推广给 draft。

6 个测量请求共 48 nominal round；仅 42 个完整接受且 6 个输入 route 齐备的轮次，与 8 rank 的实际 route SHA 逐项吻合。最后 6 轮尾部缺项已排除。
42 轮含 14 个唯一 route SHA，共 13440 个 layer×rank 投影。local R 中位4、均值4.5、最大19；active experts 中位3、均值2.7213、最大10。
精确命中已测获益 R1/R3/R6-e47 的仅 32/13440=0.238%，去重后 10/4480=0.223%。更广的 late-expert 模式没有性能证明。

这些是受限的实际 target route 投影，未覆盖未接受/未导出轮次、draft route 或 actual GMM prepack consumer。
完整原生 fallback 的 CPU 协议、坏 frame/counts 反例已通过；没有设备实现或角色 tiling 数值门。
当前候选整网优先级已降低。

同一正式 profile 的 80 个 rank-step，target GMM1/GMM2 kernel duration-sum 的 rank 中位分别为 3.115124/1.489683 ms，draft 为 0.140471/0.107669 ms。
这是 profiler 诊断，不是端到端时延或可回收预算；HCCL 父子 duration 没有累加。

## 正常 TP8 SDK 身份门

测试使用标准 `WorkerWrapperBase.init_worker → init_device`，不调用 `load_model`，每个新 worker 仅执行一个 variant。
v2 的 8 个 worker 正常初始化；首个已保存 rank0 在 ACL mapped inode 检查拒绝，probe allocation=false。
只读 CPU 检查确认路径、inode 和 SHA 一致，但 overlay stat 为 0:259，`/proc/maps` 为底层 253:2。

v3 保留 NOLOAD，通过 kernel `map_files` 全文 SHA/inode、实际可执行段与 ELF offset 的逐位比对、符号地址段归属及 overlay mountinfo 证明加载对象；56 项 CPU 检查通过。
新 TP8 job 中，已保存 rank6 的 actual context/stream、logical6→physical14、TP rank6/world8、idle/capturing=false 均通过，probe allocation=true。
随后 `inspect.getfile` 对 `_LazyNpuGraphEx` 实例触发 TypeError：0 generation、0 copy，未到 compile/capture，不能判定 alias 能力。
其他 7 rank 的 case 结果不由首个失败推定。所有 8 个新 worker 已退出并释放。

下一版本仅修 SDK 懒代理类/真实 backing module 的来源检查，保留原 factory/config 调用，不过滤 kwargs、不修改 SDK、不复用失败 worker。
完整 43 层/17 点消费者、实际 INT8 prepack 次序、cache reader/writer 和 rollback 审计仍待完成。

## 正式完成条件

最终使用完整40层/hidden5120/真实W4A8_DYNAMIC、Engram int8、完整视觉、原生MTP3与严格 HCCL 确定性。
正式 API 至少 8 个串行计数请求、2K 输入/256 输出，关闭 audit/profiler；按实际 target/verifier decode rounds 计步，包含 initial 和 terminal 已提交草稿完成。
必须自洽报告 `ms/step=1000*decode_s/rounds`、`A=decoded_tokens/rounds`、`tok/s=decoded_tokens/decode_s`，同时给 clean/复读/空答。
完整消费者审计、GSM8K100/100、Vision23/23、服务归属、包含 DSpark ≤20 ms/step 及最优部署全部实际达成，才完成 goal。
