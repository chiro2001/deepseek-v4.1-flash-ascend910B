# 含 DSpark 的 tiny 同实例时延与 TP8 SDK 参数 ABI

2026-10-11。正式完整 TP8、真实权重、包含 DSpark 全部执行开销的 ≤20 ms/step 目标继续 active，尚未达成。

## tiny：已测到同实例图执行时延下降

physical4，TP1、完整40层 tiny dummy/BF16、原生 MTP3。固定 before-capture O 投影 owner fixture；原始参数及数学顺序保持。
本轮 128 输入/32 输出，2 warmup + 8 个交错 OFF/ON pairs，每 arm 8 个计数请求。

| Arm | decode 窗总秒数 | 实际 rounds | decode tokens | ms/step | A | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 5.826804710 | 128 | 248 | 45.521911794 | 1.937500 | 42.561920703 |
| ON | 4.923480720 | 128 | 248 | 38.464693123 | 1.937500 | 50.370868521 |

三元组按相同窗口的总量计算：`ms=1000*decode_s/rounds`，`A=decode_tokens/rounds`，`tok/s=decode_tokens/decode_s`。
ON 的观测 step 时延少 7.057218671 ms，约 15.50%。该差值只适用于此同实例、128/32 tiny 机制实验。

每请求 17 次真实 `_propose`，包括 initial 与 terminal 各一次。OFF 有 17 次 helper 返回、16 次 target replay、0 次 draft replay；ON initial 为 NONE/eager helper，随后 16 次 target 与 draft replay。
每个 decode 窗包含已提交 initial/terminal 草稿完成，terminal fence 每请求一次。
计时中的诊断 D2H/hash 为 0，route/logprob audit、first-value trace、profiler 关闭；原生 CPU bookkeeping 与 host counter/observer 开销保留。额外显式 native fence 观测为 0。

19 个实际源码 SHA guard 通过。18 请求前后所有 43 份源参数/缓存/finite seal 通过，共 1548 次逐项检查；16 计数请求对应 1376 次。
同 pair token 完全相同，且与冻结 128/32 数值参照一致。完整数值门来自先前的 20 请求/9 检查、logprob delta0 审计，本计时任务没有重复执行逐层消费者审计。

18 输出仍全部周期 2 复读，clean-rate=0，tiny/formal admission=false。不能以 dummy 接受长度或该差值推定正式吞吐/20 ms 达标。
physical4 终态健康、无模型进程；无 reset、清 pagecache 或停止其他租户。

紧凑证据：`parallel_tiny/dspark_lane/fixture_timing_diagnostic_v2/evidence/dspark_lane_20261011_c4_fixture_timing_diagnostic_v2_analysis.json`，SHA256 `030baa7b62d63e1318a17bef69b00bee8b4a9d153ea5518686e7a868cb2d0904`。
精确冻结交付清单共 85 文件，全部源 SHA/原字节 JSON 核验；SSH 包 196205 B。

## 2K/256 与 context 图优化的边界

原模型真实 sliding_window=128、K5、三层 DraftSWASpec。2K prefix 需要新的 per-group block table、actual sparse indices、逻辑位置与 rollback 验证；不能直接借用旧 128/32 的 cache 假设或固定 rounds。
当前只做 CPU 扩容准备，不授予新几何的设备/时延通过。

实际 native caller 顺序为 combine → set_inputs_first_pass → _run_merged_draft → 三组 project/store → 串行 MTP40→41→42。
combine 位于 outer draft graph 外；如果 _run_merged_draft 已由 outer graph 包装，KV 前置 hook 自然包含其中，不能再嵌套 NPUGraph。
正式 eager/profile 的 bank 显示 draft graph 未配置；这些源码与 bank 信息没有替代实际消费者或捕获证明。
最小候选须保留原位数学顺序、每 gid 的 slots/positions/owner 绑定及 dynamic/prefill 原生 fallback，当前仍为独立 CPU 设计。

## TP8 SDK：已进入真实生成内核，首参 ABI 仍拒绝

`formal_dspark_sdk_init_static_default_v4` 使用 8 个全新标准 WorkerWrapper，正常 init_device 后借已有 context/stream，不 load_model。
45 文件严格冻结、65 CPU 检查通过。实际 lazy registrar、backing module、compiler 来源门已通过；原 SDK factory/config/kwargs 调用保持，没有修改 SDK 或过滤参数。

首个完整 raw RPC 为 rank7：real-warmup 中 compiled → 外/内 AOT → npu_fx_compiler.run_kernel → generated kernel line90 报 `TypeError: expected Tensor()`。
line90 为 `assert_size_stride(args[0], (6,64), (64,1))`，确认首参数不满足 Tensor；当前没有实际参数类型记录，不能猜成 None/list。
0 完整 epoch、0 copy_calls、alias_error=null；没有语义或 alias 通过/拒绝结论。

8 rank 的实际 Dynamo/AOT/generated kernel/result 文件均已取回。其他 7 rank 的异常字符串为空，real-warmup 中被终止，不继承 rank7 的具体错误原因。
8 个模型未载入的 worker 已终态退出、physical8–15 无占用。资源门仅放行用户已授权的 80C98001。
本轮 SSH compact 90752 B，SHA256 `bff2f0cdbfc29ebfbd6088f18cc0514d7271caed8eedaa4bfae4936a98a3302c`，58 文件逐 SHA 校验。

正在用同版本 CPU torch 追踪外/内 AOT、CopyMode invoke 包装、`_boxed_call` 与 runtime_args，取得可复现 ABI 证据后才建立新版本/新 worker。
不绕过 generated kernel 的 Tensor assert，不重用失败 worker，不修改旧 v1–v4 源码。

## 正式验收继续保持原范围

完整40层/hidden5120、主模型384/top6、原生 draft128/top3/MTP3、真实W4A8_DYNAMIC、Engram int8、完整视觉；HCCL_DETERMINISTIC=strict。
先完整 main/draft/verifier/cache、实际路由与 INT8 prepack consumer/接受前缀/rollback 审计，再同实例候选验证。
正式 API 至少8个串行计数请求、2K/256、audit/profiler关闭，按实际 decode rounds 计步，包含所有已提交首末草稿完成；同时报告三元组与 clean/复读/空答。
GSM8K100/100、Vision23/23、服务归属、包含 DSpark ≤20 ms/step、最优部署与交付全部实际达成后才 complete。
