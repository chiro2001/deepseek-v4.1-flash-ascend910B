# 包含 DSpark 的真实 HTTP 基线与生产审计进度

2026-10-11。完整真实权重 TP8、包含 DSpark 全部执行开销的 ≤20 ms/实际 verifier step 目标保持 active。原生 HTTP 集成诊断得到 **57.581661 ms/step、A=2.965116、51.494108 tok/s，clean7/8**；性能与质量均未达验收要求。

## 实际 HTTP：2 warmup 加 8 个串行计数请求

实际 job 为 `formal_dspark_native_api_v2`。使用 40 层、hidden5120、384 专家/top6、W4A8_DYNAMIC、Engram int8、完整视觉和原生 MTP3 权重。draft 原生 128 专家/top3；DSpark K5、draft eager、target FULL[6]。HCCL strict，audit/profiler 关闭，本次没有候选优化。

工作集由正式 checkpoint tokenizer 准备，2 warmup 与 8 measured 均为 2048 输入、256 输出预算。每条排除一个 prefill seed，decoded_tokens=255。8 measured 合计 **688 actual rounds、2040 decoded tokens、39.61618308 s**。

| measured | actual rounds | ms/step | A | tok/s | clean |
|---:|---:|---:|---:|---:|:---:|
| 0 | 48 | 60.265070 | 5.312500 | 88.152224 | 是 |
| 1 | 148 | 56.389065 | 1.722973 | 30.555091 | 是 |
| 2 | 50 | 56.591812 | 5.100000 | 90.119045 | 是 |
| 3 | 85 | 55.482356 | 3.000000 | 54.071243 | 是 |
| 4 | 142 | 53.717714 | 1.795775 | 33.429841 | 是 |
| 5 | 46 | 61.702470 | 5.543478 | 89.842081 | 否，复读 |
| 6 | 61 | 64.421528 | 4.180328 | 64.890231 | 是 |
| 7 | 108 | 59.595788 | 2.361111 | 39.618758 | 是 |
| 合计口径 | 688 | **57.581661** | **2.965116** | **51.494108** | **7/8** |

合计按 sum(decode_s)/sum(actual rounds) 计算；warmup 独立，不计入 8 条。三元组逐请求与合计均满足 `ms=1000*decode_s/rounds`、`A=decoded_tokens/rounds`、`tok/s=decoded_tokens/decode_s`。ms/step 中位 58.093800，单请求 tok/s 中位 59.480737，不能把二者与合计口径混用。高 A 需质量复核；clean 是非空/指纹/复读启发式，不是语义正确性。

## 计数、身份和完成边界已复核

新离线 verifier `validate_native_api_diagnostic.py` 校验 10 个请求共 80 份 rank 收据、原 HTTP 响应字节、全部 per-round 输出长度、首末 payload、external→core 映射、同 boot/time namespace CLOCK_MONOTONIC 和 38 份部署源 SHA。13 个独立证据篡改反例全部拒绝，涵盖外来 clock、错 core、遗漏首末 draft/fence/join、虚构 rounds、seed 混入 decode、错误三元组、finish RPC 混入时钟及伪造正式通过。

decode 从末次 prefill bookkeeping、初始草稿提交前开始，包含初始与末轮已提交的原生草稿。每 rank 的末轮 device fence、sample timestamp、Gloo TP8 join 先于 HTTP 输出。客户端 `response.read()` 完成后立刻保存 end_ns，再发 finish RPC，因此计数取回 RPC 不进入 decode_s。零 token cleanup 不计 step，末轮输出裁剪不伪造额外 rounds。

例 measured0：48 verifier rounds、49 draft calls、245 proposals、240 verified draft tokens；末轮只裁剪 1 token。8 rank 原 core ID、首末 token/proposal 与每轮长度均一致。

保留首末轮样本的 host wall-clock 区间如下；它们不隔离设备 kernel，也不能把该样本中位当作全窗口分解。

| 区间 | 64 个 rank 样本中位 |
|---|---:|
| 首轮 execute 开始→bookkeeping | 35.084315 ms |
| 首轮 bookkeeping→sample 返回 | 16.591345 ms |
| 末轮 execute 开始→bookkeeping | 35.967620 ms |
| 末轮 bookkeeping→sample 返回 | 18.447815 ms |

末轮全部 rank join 到客户端 read 的每请求中位为 286.990910 ms，属于此次真实 API 墙钟，未从 quote 中扣除。该固定尾部开销与目标/草稿执行路径均需后续同实例候选验证；仅改变前端计时无法证明 ≤20 ms。

## v1 启动失败与 v2 修正

v1 已加载真实模型，但原 `serve_http` 访问 `.routes` 时，私有 ASGI 隔离器缺该属性而失败 exit1，没有 HTTP 推理输出。v1 在进入 serve_http 前写的 `service_started=true` 不作为 HTTP 成功证据。

v2 用独立透明 facade 转发原 app 的 `.routes` 与 `.state`，保留原 ASGI 调用、三个私有端点、403 隔离与 clock/ID 算法；冻结桥、原 SDK 与模型数学不变。实际完成 10 个 HTTP 输出后 client/service 均 exit0。仅对已验证 PID/PPID/start_ticks/cmdline 的自有 frontend1483147 使用 pidfd SIGTERM，SDK 正常 shutdown，不停止其他租户、不 reset。后续 fresh 只读快照确认 physical8–15 无运行进程，健康码仅为用户授权的80C98001；下一次启动仍必须重新检查。

## 尚需完整生产消费者审计与候选优化

SDK v8 static_keep/nonstatic_keep 双 job 的 16 rank/512 epoch 已通过，但未加载模型，只提供小型 SDK 捕获/alias 能力。v9 独立协议桥已冻结 10 文件并严格读取该原证据，不改写旧 single-chip 收据。完整 43 层/17 消费者生产接线 `full43_consumers_v10` 尚在开发。

INT8 prepack CPU 组件已冻结 18 文本，16 测试/41 反例通过。GMM1 在量化后、真实 prepack 前做 raw INT8 snapshot，MSD workspace 双 bank；GMM2 从 GMM1 原 INT8 输出、npu_grouped_matmul 前绑定原 x GM 原位 pack。K 从实际 loaded W2/FFN 分区与输出几何交叉确定，6 个审计 Tensor 须保留在真实 graph 出口。CPU _version/rollback 或 Python 引用不能代替实际内容/设备 cache 证明。

通用质量 API 基础包已冻结 17 文本，8 CPU 测试/34 反例通过，保留官方 FastAPI/AsyncLLM/serve_http、实际 ASGI lifespan 与 v4 worker 窗口 inactive。完整 source snapshot/启动器仍在准备；GSM canonical 8-shot/前100 test 严格100/100、Vision 原20问答加3 negatives严格23/23尚须同候选新输出，不复用旧 A1 或脚本默认80%门。

三个代理已回传任务 ACK，继续已核验的 gpt-6.1-sol/max。tiny physical4/PID953926 的同一个 2K/256 v3 job 继续运行：快照已完成7请求、9项数值门中 OFF A/A seed0/1两项通过，logprob delta0，token/Top5、43路由、3 attention、逻辑 cache/逐步 visible cache 与参数 seal 全过；剩余7项未完成。它不报价正式收益，不重启、不修改42份冻结源。

下一项实际工作是以 fresh TP8 运行已可执行的完整权重消费者通路，并严格列出缺失覆盖；随后完成43/17、prepack/cache/rollback数值门及同实例配对优化。正式目标仍要求至少8串行请求、包含 DSpark≤20 ms、GSM100/100、Vision23/23与最优正式部署。

## 证据与交付范围

- validated：`evidence/formal_a321/native_api_v2_validated.json`；所有 actual/full-consumer/performance/formal 的通过范围明确分开，正式 admission 均 false。
- 远端完整 acceptance：22,145,365 B，SHA256 `1ea53d1bf189b8444dd1f4fdd7856b7bd5ef8d28a7fc8b3d577e20935f492a12`；bridge windows：21,306,200 B，SHA256 `ad89b26b2e8a4e027c882316eeaad7e49e6029a3cb1435919c6e33494d66710e`，均留远端。
- 本地 `native_api_v2_compact.tgz`：112,423 B，SHA256 `04e86e3683b80622fa59707d0fe1213ae74f79ba03600c5386bf8fd3bc9d8fc5`。各原文件与完整 raw 源 SHA 由 manifest/index 绑定；未保留的中间完整 payload 不冒充本地已审计。
- prepack manifest：`bf2becb2f9e9e87e1d72ff1bbb23c88a5414f1d914e5ce8dec7b035182a902bd`；general API manifest：`5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a`，root逐文件SHA复核。
- 当前 thread `01a128f6-00b4-7761-8a4b-eb78c0303c52` 已有 unfinished active goal；再次 create_goal 被接口拒绝，保持原目标 active，未把未达成目标标 complete。

本报告仅记录实际诊断与准备结果；完整消费者、正式20ms/质量、最优部署和全部交付完成后才能 complete。
