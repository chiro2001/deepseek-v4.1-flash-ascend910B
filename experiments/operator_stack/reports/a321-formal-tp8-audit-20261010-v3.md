# a3-21 正式 TP8：完整模型运行、实际覆盖与路由一致性诊断

2026-10-10 · `feat/operator-stack-tp8-20261010`

**正式 checkpoint 已在 a3-21 chip8–15 完成加载、图捕获和12次请求。叠加方案尚未通过整网精度验收，19 ms/step目标未达到。** 首组对照的 token 相同，但 decode 专家路由和 Top5 logprob 不满足原门槛；同一个原生 bank 的前后请求也存在差异，需先排除 bank 重建与观测代码的影响，不能直接归因于 core 优化。

## 已完成的正式模型工作

模型为 `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`：40层、hidden5120、384 experts、top6、W4A8_DYNAMIC，Engram int8开启。使用正式文件，不随机化权重，不使用 dummy。视觉模块按生产图片额度4创建并加载，本轮请求为文本，图像质量未测试。

本轮三次启动均保存退出码和原始日志：

| 作业 | 结果与修正 |
|---|---|
| real_tp8_audit_v1 | 8 worker通信初始化通过；缺 lazy 参数，被 Engram 守卫拒绝 |
| real_tp8_audit_v2 | Engram映射完成；图片额度0使视觉模块未创建，loader加载 `aligner.w1.bias` 时退出 |
| real_tp8_audit_v3 | lazy、128线程加载、视觉额度4、Engram RW挂载对齐；完整模型加载成功，4个bank均捕获并完成warmup；首组整网路由对照失败，退出码1 |

L1、L14 Engram表已注册为设备可寻址；权重loader记录加载耗时50.05秒。静态核启用标记出现1次，`static_kernel.py:650`回退标记为0。原生 HC 在每个rank捕获80次；正式384/top6路由保持原生，未启用TP1的8/top2特化。

## Alarm设备的实际可用性

这8个chip持续报告 `80C98001` RAS Alarm。本轮8-rank Vector、Cube、HCCL AllReduce精确通过；正式模型也完成实际请求。**本轮证据证明这些工作可以执行，不能解释为Alarm已修复。** 没有reset设备、停止其他租户或清page cache。

每次启动前检查占用。v3结束后，另一用户的SGLANG TP8作业占用了chip8–15，新的原生单图A/A控制作业正在等待空闲；占用检查未绕过。该SGLANG主进程启动时间在v3结束约8分钟之后，不用它解释本次已发生的精度差异。

## 两条线的实际覆盖

| 项目 | 正式模型中的结果 |
|---|---|
| HC static / HcPost候选 | 正式数值门槛已失败，全部保持原生 |
| TP1 router / selected GMM | 384/top6契约不匹配，未启用 |
| metadata / slot准备 / QKV overlap | core候选已捕获和执行；整网一致性尚未验收 |
| BF16 clamp/SwiGLU候选 | routed/shared在所有bank和rank的实际调用及选择均为0；正式W4A8走原生融合路径，不能计额外收益 |
| Indexer后处理融合 | tp8stack每rank捕获4次，8rank均有覆盖；INT8量化、FP32 scale、INT8/FP16 cache独立检查通过 |

12次请求各检查8rank，共96次rank审计；Indexer functional bitwise和consumer-point cache bitwise检查全部通过。该结果来自正式整网激活与真实cache布局，而不只是上一轮的合成独立用例。它仍不能替代整网路由/logprob门槛。

## 整网失败的定位

本次请求prompt为2048 token，输出47/48 token交替；已保存的路由shape为 `[2094/2095,40,6]`。所有ID在0–383，每个top6无重复，路由不是无效值或越界值。

| 对照 | token | prefill不同token-layer数 | decode不同token-layer数 | top6集合不同数 | 共同token最大logprob差 |
|---|---|---:|---:|---:|---:|
| 原生warm0 → 原生pair0 | 相同 | 0 | 1380 | 879 | 1.124988556 |
| 原生pair0 → core pair0 | 相同 | 0 | 1344 | 812 | 0.624981880 |
| 原生pair0 → act pair0 | 相同 | 0 | 1347 | 862 | 0.624891281 |
| 原生pair0 → stack pair0 | 相同 | 0 | 1370 | 871 | 0.624998093 |

这些对照的Top5 token集合也不完全一致。原门槛为完整路由一致、Top5集合一致、最大logprob差<1e-3；没有放宽。首个base/core差异位于第2048行、layer4，最初是top6内部顺序互换；后续也有专家集合变化，不能只按无序集合将它忽略。

原生前后对照之间曾重建其他bank，且原生bank含实验观测包装，因此这不是“未改生产模型原生A/A也已证明不确定”的结论。下一步用独立 `FormalControlWorker`：不安装候选算子包装、不重建多bank、仅保留审计路由导出修正，运行同一prompt的3次A/A对照。这个区分决定后续该修模型原生路径还是修bank管理。

审计期间的时延受路由导出、logprobs与cache快照影响，不作为正式性能，不发布本轮有效 `(ms/step,A,token/s)` 或加速比。采集态累计kernel时间也不能充当端到端step时间。

## 新增诊断方案与原理

1. **生产原生单图A/A**：`bench_formal_native_control.py` + `formal_control_worker.py`，不导入算子bank补丁。保持同一组worker、正式权重和原数值门槛，保存token、路由、logprobs对照；可另用 `--eager` 区分图路径影响。当前单图作业因设备占用等待，尚无结果。
2. **候选bank独立内存池**：实际 `ACLGraphWrapper` 捕获使用 `self.graph_pool`，并将输出和workspace转成弱引用。现有多bank共享pool，因此将其作为地址/生命周期调查点；新增 `--isolated-pools`，每个候选bank建立独立NPU graph pool，并在切换bank时恢复其pool。该开关默认关闭，尚未上板验证，不能宣称它已修复差异。
3. **七组计数与8rank分目录采集**：入口已准备，性能计时完成后才启动profiler，审计与profiling分开。本轮尚未采得正式整网七组数据。待一致性问题厘清后再采MTE、L2、UB、Cube L1/L0、通信和CPU/图间隙，避免继续外推tiny或独立单行kernel的结果。

UB双缓冲的判定保持上一轮证据边界：单行Indexer的低搬运利用率不支持优先增加buffer；应在正式GMM/attention等多tile循环中确认MTE位于关键路径。未采得的容量occupancy、NA计数和工具失败不写为0。

## 交付与下一步

代码在独立分支；新增完整模型加载修正、显式Alarm使用/占用等待、正式TP8服务worker、原生A/A控制、graph pool诊断和分rank计数入口。服务worker限制TP8、单序列、capture_sizes=[1]、A=1；尚未通过客户端验收，未部署未通过精度的候选方案。

先等待这组chip空闲并完成生产原生A/A；若单图稳定，优先验证bank隔离与事件/参数恢复；若单图也失败，再独立比较eager并定位第一个偏离的decode层。通过原门槛后，关闭审计，进行同进程配对计时和热点profiling，再选最佳服务与客户端质量验收。目标保持active。

紧凑证据：`experiments/operator_stack/evidence/formal_a321/results/real_tp8_audit_v3/`，包含bank覆盖、96次审计摘要、路由差异诊断、原始日志与退出码。原始requests留远端，110,145,043 bytes，SHA256为 `4b68bffe9adfda1600100ec8e814d01218b7abe08347aa1d0bf619643f41ad0d`；大于1MB的原始数据不经SSH传输。
