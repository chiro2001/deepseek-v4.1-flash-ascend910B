# 正式 full40 TP8 DSpark 原生 ON 准备

最新目标：**包含 DSpark 的完整正式推理 ≤20 ms/step**。本目录只有 CPU 准备与入口代码；NPU 权限为 0，physical8–15 的实际启动由主线程执行。没有继承此前 A=1 的质量、时延或安全 receipt。

入口为 [`bench_formal_dspark.py`](../../bench_formal_dspark.py)，CPU 计数模块为 [`formal_dspark_counters.py`](../../formal_dspark_counters.py)。bench 顶层只导入 stdlib、CPU contract 和 CPU counter，**不会提前导入 worker、NPUWorker、模型或 FusedMoEFactory**。新 worker 由 LLM 的标准 `worker_cls` 字符串加载；API 签名门位于正常 Ascend 平台初始化后、实际权重加载前。

## 固定配置

只读实际 checkpoint：`/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`，config SHA256 `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。

[`discovery/formal_checkpoint_cpu.json`](discovery/formal_checkpoint_cpu.json) 已从现有正式容器用 stdlib 检查 90 个 safetensors shard header、index/config、辅助 symlink closure。40 层、hidden5120、384 专家/top6、W4A8_DYNAMIC、Engram[1,14] 与视觉配置均保留；MTP 有三层、4718 个 tensor、target ids[37,38,39]、trained block5。MTP 专家为 W4A8、attention 为 W8A8、main_proj 为 FLOAT。未执行或哈希全部权重内容。

原生 `method=dspark` 自动复用 target checkpoint 及量化配置。默认 K5，必须 K≥checkpoint block size；target `FULL_DECODE_ONLY` capture `[1+K]`，固定同步调度、单请求、strict、prefix off。draft graph 默认 ON，同时设置 `DSPARK_GRAPH_CAPTURE_METADATA=1`、sync metadata、capture value/nctx/resident indices/query-length 四件套；没有 dummy、旧 prefix、woa、负 QKV 或 tiny cache plan。

## 主线程启动

先把新入口、counter 和整个新目录加入**完整且兼容的正式冻结源**，复用现有 `run_stack.sh` 的 SDK 环境/PYTHONPATH；不要只复制不完整的 tiny vendor 片段。以下路径为新冻结源和新 job 的命名示例，容器和设备授权由主线程现有 launcher 负责。

```bash
STACK_SRC_ROOT=/work/src-dspark-native-aa-v1 \
STACK_PHYSICAL_CHIPS=8,9,10,11,12,13,14,15 \
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
HCCL_DETERMINISTIC=strict \
bash /work/src-dspark-native-aa-v1/stack/run_stack.sh bench_formal_dspark.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/results/dspark-native-aa-v1 \
  --mode audit --pairs 3 --speculative-tokens 5 --draft-graph 1
```

默认 `--mode prepare` 只做 CPU checkpoint/启动计划准备，不导入 vLLM 或 torch。`--mode audit` 使用同一 LLM 实例、同一组 worker 的 DSpark ON 原生 A/A，每对包含两个完整请求；两条 warmup 在计数窗之外。完整 accepted target 路由、token、Top5/logprob 通过原有 `formal_comparison` 严格比较，另比较实际 verifier 提议及原生 raw token 流。

audit 退出码只表示 `native_aa_passed` 控制门；即使 exit0，`passed/full_consumer_audit_passed/safe_formal_receipt/formal_model_validated` 仍为 false。完整 draft/cache/device-consumer 专门证明属于后续阶段，**不会阻止先运行原生 ON 基线和定位失败**。audit 的计时是诊断数据，没有正式性能 claim。

`--mode perf --pairs 10 --audit-root <完整正式DSpark消费者审计目录>` 需要完整 actual main/draft/verifier/cache 审计及相同源码、checkpoint、DSpark config 的独立 receipt。当前 CPU 准备没有该 receipt，因此 perf 在 LLM 创建前拒绝；不能以旧 A1 receipt 启用。

## 实际轮次与时钟

计数绑定真实 `NPUModelRunner.execute_model → _sample/rejection_sampler → _bookkeeping_sync → drafter._propose → sample_tokens`。`scheduled_new/cached_reqs` 中 CPU computed/prompt 长度用于排除 prefill；原生同步 bookkeeping 已产生的 CPU token 列表用于记录 raw 生成及接受数，不新增每轮 D2H。计数只在显式配对请求 tag 后激活，load/profile/capture/warmup 都在窗外。

decode 起点是 terminal-prefill bookkeeping 已给出 seed、但 initial draft 尚未执行的 CPU monotonic 时间；终点为最终客户端 RequestOutput。按准确 output budget 判定的末轮 sample 后，仅在当前 rank 设备同步一次，确保已提交的 terminal draft 也完成后才返回客户端。steady rounds 不新增同步或 D2H，末轮同步明确记录在 receipt；计数 RPC 的返回延迟不进入时间窗。

worker/client 必须在相同 boot ID、time namespace 和 CLOCK_MONOTONIC。8 rank 的真实 scheduled/proposal/token/接受流必须一致。末轮 budget 截断单列，不能用前端 poll 次数、输出 token 数或全局 MeanAcceptance 作 rounds。

```text
decoded_tokens = 最终请求 token 数 - prefill seed 1
rounds = 实际已完成的 target/verifier decode 轮次
A = decoded_tokens / rounds
ms/step = 1000 × decode_elapsed_s / rounds
tok/s = decoded_tokens / decode_elapsed_s
ms/step = 1000 × A / tok/s
```

收据含每请求与 aggregate 三元组、initial/terminal draft 调用、实际验证提议、raw/usable 接受数、末轮 clip、clock_scope、8 rank 源码函数/文件 SHA。A≈1 且接受数为 0 会设置 `acceptance_zero_diagnostic_required=true`，需要检查实际草稿/图是否失效，不能据此宣布 ≤20 ms 达标。

## SDK 与未完成的门

实际源/API receipt 与 device audit 分开。早 import worker 会在平台 patch 前绑定旧 factory，已在 tiny 线观察到 `bias_vl` TypeError；本入口通过标准 LLM 初始化加载 worker。真实 worker 仍检查当前 factory 是否接受视觉 bias，记录 `sdk_rankN.json`；不复制 tiny 适配、不删除 kwargs 或绕过视觉偏置。

load 完成、capture 之前保存原生 target、full40 backbone、draft model/backbone 引用。capture 后拓扑从这些实际对象及当前注册 cache tensor 读取，不假设 ACLGraphWrapper 有 `.model`。cache 地址/布局 metadata 不构成 device-consumer proof。

当前 CPU 计数契约 **16 项测试通过**，涵盖真实 rounds、seed/prefill 排除、末轮 clip、8 rank/clock 一致性、missing sampler/verifier、graph ON 被 eager 替代、缺少末轮 fence，以及 wrapper 不暴露 `.model` 的拓扑回归。CPU 成功不代表 NPU/A/A/正式性能通过。

当前支持的首轮范围是同步调度；async placeholder ABI 会明确拒绝。原生 target/verifier 路由导出是审计证明的一部分，draft 每层设备路由、cache 写读/别名与完整 actual-consumer marker receipt 仍待后续 NPU 阶段。所有现有正式 runner、ARM、launcher、STATUS、MANIFEST 和其他 agent 文件均未修改。

```bash
python3 experiments/operator_stack/parallel_tiny/dspark_formal_prep/test_counters.py -v
python3 experiments/operator_stack/bench_formal_dspark.py --help
```
