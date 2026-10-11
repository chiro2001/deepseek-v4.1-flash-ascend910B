DSpark 的最小完整路径应在同一请求中证明下列实际消费者；配置开关和已构造的 module 数量不足以证明执行覆盖。

| 位置 | 原生行为 | 取证要求 |
| --- | --- | --- |
| target `execute_model` | 40层前向/缓存更新，输出进入 target 层37/38/39的 residual aux states | 真实 SchedulerOutput，排除 prefill/profile/capture |
| DSpark `set_inputs_first_pass` | 将实际 context hidden/position/slot mapping 填入每组常驻 buffer | 实际 context 行数、query 行数、组别、shape/dtype/stride/format、generation 与 cache 地址 |
| `combine_hidden_states` | `[N,15360]→[N,5120]` main projection 与 main norm | 原生 A/A 及候选 OFF/ON 输出和参数快照 |
| `precompute_and_store_context_kv` | 对三层独立 `wkv→kv_norm→RoPE`，写入三层 draft SWA cache | 三层实际 producer 与每组 scatter 消费者；不能只记录第一个 module |
| `DeepseekV41DSparkModel.forward` | 3个串行 delayed-mHC draft block，norm/Markov head 生成块提议 | 实际三层 forward/attention/MoE route；草稿 token 与 logits/Top5/logprob |
| target verifier/rejection sampler | 验证本轮真正调度的提议，接受前缀与 bonus token | `scheduled_spec_decode_tokens`、原生 CPU sampled list、raw接受数和预算裁剪量 |
| scheduler/cache | rejection rollback、下一轮 cache/slot/metadata 更新和终态释放 | 所有实际轮次、首轮 initial draft、终态边界与缓存 generation |

原生源码中三层 context KV 的权重不同，因此不能将相同 `context_states` 误解为三个输出相同。当前只准备覆盖与精度取证，未合并投影、跨轮缓存或重排流。

图候选必须共同使能 `DRAFT_GRAPH=1` 与 `DSPARK_GRAPH_CAPTURE_METADATA=1`，并保留已验证的 capture 代表值、context-KV写入和真实 nctx 修复。已被既有文档否证的 hoist-only、pad-slot、max-seqlen、no-topk-share 等开关，不作为本轮重试候选。capture 区间不得 D2H；实际逐请求 `_propose` 窗口才能取 token/消费者内容证据。

候选精度门采用同实例原生 A/A、OFF/ON、ON原生 A/A：token一致、route顺序与消费者值一致、Top5 key集合一致、有限 logprob 的差值低于原基线容差；cache地址和有效内容须同时覆盖。关闭全部 profiler/audit 后才比较至少8条固定预算、同实例交错配对的串行请求，使用实际 target/verifier 轮次与自洽三元组。正式验收仍为真实 TP8 的20ms/step。
