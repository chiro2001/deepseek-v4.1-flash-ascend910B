# 私有 draft 图候选的原生消费者诊断

当前已经验证的第一个坏 operand 是 **MTP0 的 `_dummy_wo_a_3d` 派生缓存权重**。graph_diag_v7 的第一实际 decode、原生 OFF 路径中，BF16 `[8,4096,512]` 存在 2,144 个非有限值。此前同一审计请求的 prefill 中，同一 cache 指针、stride 和格式与原始 wo_a 的原生 loader permutation 逐字节一致；原始 wo_a/wo_b、raw attention 保持有限。

graph_diag_v6 已证明该坏缓存之后的原生 O 投影输出 `[5,5120]` 全部非有限，并确认此前实际 HcPre-attn 的四个输出、query、sink 和 665 个按实际 sparse indices 选择的 cache 槽均有限。graph_diag_v8 的同次 native target replay 前后快照进一步证明：重放前三层 cache 均正确，重放成功返回并 device fence 后 MTP0/MTP1 分别出现 2,144/672 个非有限值，MTP2 保持正确；具体覆盖 kernel 和 allocator/lifetime 机理仍未确定。

新固定 fixture 按同一原生公式在所有 compile/profile/capture 前初始化 target40+draft3 cache，注册到每层 self_attn 的非持久 `_tiny_fixed_native_wo_a_3d` buffer，并使实际消费者绑定同一 Tensor。初始化器 SHA256 为 `7dfa11336f7be4c8c55d0f09eb302673ef31ac51d582686d31cffc432af6fe78`；原始参数没有变化，也没有在 warmup/capture 后回填缓存。纯 eager 的 native_diag_fixture_v1 已通过 finite/owner/source/value seal；graph_fixture_v1 已以 exit0 完成 20 请求、9 项原生 A/A 和 OFF/ON 数值检查。所有诊断时间都没有性能资格。

| 运行 | 结果 | 能证明的范围 |
| --- | --- | --- |
| native_diag_v1 | exit1 | 诊断代码读取 inference tensor 的版本计数器失败；不是质量结果 |
| native_diag_v2 | exit0 | fresh spec.enforce_eager=true、无 draft ACLGraphWrapper/entry；32 token、16 decode、17 `_propose`，91 次真实 router 调用的输入、参数封印和即时返回值没有被 finite 门拒绝 |
| graph_diag_v3 | 启动前拒绝 | fresh4 门发现未知 SDK 探针占用；只完成私有 staging，没有启动自己的 SDK 或模型 |
| graph_diag_v4 | exit1 | MTP0 router 算术之前，hidden 的 25,600 个值和 logits 的 40 个值已全部非有限，gate 参数有限且值/地址封印保持 |
| graph_diag_v5 | exit1 | 第一 decode 的 context projection、三层 shared-KV，以及 MTP0 实际 sparse reader 输入、读槽和输出均有限；下游 router 输入仍非有限 |
| graph_diag_v6 | exit1 | 实际 Torch dispatcher 的 HcPre 四输出有限，O 投影输入和原始 wo_a/wo_b 参数有限；原生 O 投影输出首先被拒绝 |
| graph_diag_v7 | exit1 | 同一指针的原生 wo_a 派生 cache 从 prefill 的正确字节变成 first decode 的 2,144 个非有限值；第一 decode 的 BMM 在执行前被 finite 门拒绝 |
| graph_diag_v8 | exit1 | 同次 native target replay 区间中覆盖了 MTP0/MTP1 cache；原始参数值、cache 指针/stride/format保持 |
| native_diag_fixture_v1 | exit0 | compile/capture前显式owner固定43 cache，32 token/16 decode/17 draft；target40各1次原生consumer、draft3各17次consumer finite/封印保持，73类实际边界无nonfinite |
| graph_fixture_v1 | exit0 | 20请求（2 warmup + 18审计）；3组原生OFF A/A、3组OFF/ON、3组ON A/A全部通过；43 cache在capture后及每请求后owner/source/view/value封印保持；dummy输出仍周期2复读，性能阶段禁用 |

graph_fixture_v1 的 9 项比较中，token、原生 proposal、43 层有序 route IDs/weights、3 层 draft attention 消费者值和逻辑 cache 均逐字节相同，Top5 logprob 键相同，重新从请求文件计算的最大 logprob 差值为 **0**。每个审计请求保留 17 次实际 step 快照；1032 个参数 view 封印及 43 层 router 参数值封印通过。参数 view 封印不等于全参数字节值证明。最新保留的 73 类 first-value 记录包含 206 个 tensor summary，均有限；非有限证据文件为 0。

冷 draft capture 的原生 helper 调用/成功返回均为 1，真实 entry 为 1，descriptor 为 `num_tokens=6, num_reqs=1, uniform=True`。以下计数在 20 请求的 raw 和请求记录之间逐项核对：

| 每请求计数 | OFF | ON |
| --- | ---: | ---: |
| target decode / 实际 draft 调用 | 16 / 17 | 16 / 17 |
| 原生 Python helper 调用 / 成功返回 | 17 / 17 | 1 / 1 |
| target replay / draft replay 成功返回 | 16 / 0 | 16 / 16 |
| initial / terminal 实际 draft 调用 | 1 / 1 | 1 / 1 |
| initial / terminal draft replay | 0 / 0 | 0 / 1 |
| 请求末尾 `read_receipts` 计入的 device fence | 1 | 1 |

ON 的单次 Python helper 是 initial NONE/eager 调用；热 graph replay 不执行 Python helper 或 Python consumer 守卫。最终非 capture 的 O 投影 Python consumer 守卫累计为 target40 各 20 次、draft3 各 180 次；每请求后的 43 个全缓存封印另外检查 replay 后的缓存状态。initial 和 terminal 调用的完成都绑定到请求末尾 fence。诊断还有其他同步，`steady_extra_sync_count=0` 只描述原计数器，不能据此声称没有额外诊断同步。

20 请求的 32-token 输出都在首 token 后进入周期 2 复读，数值一致性不能作为输出质量验收。`diagnostic_completion.json` 明确记录性能评估禁用、tiny mechanism 和正式候选均未准入；本轮不报价，也不构成正式 TP8 ≤20 ms/step 结果。最终 fetch 中 physical4 Health OK、无 NPU 进程行，容器仅剩 `sleep infinity`。194 个取回文件的 SHA 全部校验，SSH 压缩包为 320,561 bytes；冻结 v7 的 37 个文件 SHA 保持不变。

终态证据见 [graph_fixture_v1 compact analysis](evidence/dspark_lane_20261011_c4_graph_fixture_v1_analysis.json) 和 [最终 fetch 清单](evidence/dspark_lane_20261011_c4_graph_fixture_v1_fetch.json)。具体覆盖 kernel 和 allocator/lifetime 机理仍未确定；固定 fixture 只证明这轮诊断中未再观测到派生 wo_a cache 损坏。

native_diag_v2 与 graph 诊断采用相同的 router 前输入 D2H 检查，但 draft 的编译配置及 warmup 历史不同，因此尚不把异常归因于 capture 污染。所有诊断读值都会增加同步，只作为坏值定位证据。native_diag_v2 的 module forward hook 只观测到编译启动，不能声称覆盖每次运行上下文；新 native worker 已把上下文守卫移到实际 `_run_merged_draft` 入口。

`native_first_value_trace.py` 保持原 compiled HC 方法，通过 Torch OpOverload/OpOverloadPacket 的实际调用处观察 HcPre/HcPost/rms_norm_cast；capture 内跳过读值。cache 选槽严格依据实际 sparse indices 的 `-1` padding ABI，不按有限性筛选。诊断对参数和相关路由张量同时检查有限性、地址/view 以及字节值哈希；派生 wo_a cache 使用原生 loader 的 permutation 做只读字节比较。

CPU 门使用真实 CPU tensor，覆盖非有限值留存、inference activation、scalar、负零、参数 `.data` 修改、原生 cache padding、越界槽位、ABI 参数绑定和 capture 禁止读值。这些 CPU 检查不构成 NPU 或完整消费者精度证明。

完整 `.pt`、权重和 profiler trace 留远端，SSH 只取小于 1 MB 的源文件和 compact JSON。所有 NPU 作业仍需 physical4 Health/occupancy/已有容器映射的 fresh 门；不 reset、不清 pagecache、不停止其他进程。

已提交的 v7 基线 37 文件清单不包含这些活候选和诊断文件，并保持原 SHA。正式 TP8 的 20 ms/step 验收仍由主线程进行。
