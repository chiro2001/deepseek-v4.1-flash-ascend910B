# Frozen Dihuo 输入路径独立只读审查

日期：2026-10-13。只读审查 `experiments/delivery_matrix/frozen_dihuo_inputs.py`、`benchmark.py` 的新增 opt-in 分支及已归档的 A2v4 2K Dihuo 请求字节。未运行服务、未访问 API/NPU/SSH，未修改源码或测试。新测试/计划由 workload agent 维护，本报告不替代其验收。

## 总结

**方案边界适合本次 2K×8 单轮复验，核心安全顺序正确：** frozen 模式先按显式 manifest SHA 校验 manifest 和八个 gzip/raw/prompt/messages，再开始任何 HTTP 请求；随后用同一个已有 `verify_model_ownership`、`verify_service_identity`、warmup、armed request、SSE 和 counter fence 路径。它复用的是输入消息，不是旧输出或旧性能结论。默认非 frozen 分支仍调用原本的本地 corpus/suffix + tokenizer 校准路径。

首轮发现的严格失败关闭项已反馈 root；复读当前工作区确认 helper/benchmark 已落实成对 SID/runtime fingerprint、`is not None` 配对与空值拒绝、manifest 文件 symlink 拒绝、`requests/` 声明根约束、逐行历史 token provenance。**当前 benchmark 通用聚合仍不要求输出 completion 恰好等于 max_tokens，也不强制 `finish_reason=length`；root 指定由本次独立 2K×8 acceptance 门强制 8/8 新鲜 SSE usage 为 `(2048,256,2304)` 且 `length`。** 源码 hash **不是** benchmark CLI 的内置准入参数：root 指定部署计划在 execute 前冻结 benchmark/workloads/frozen helper 三源 SHA，远端 `sha256sum -c` 通过；benchmark 只把执行时三源 hash 写进结果。本次只读审阅未检查该部署计划/执行回执，因此将其作为独立准入合同记录，不宣称已由本 review 验证通过。

## 已核实的实现边界

- `run()` 在 GET `/v1/models` 之前调用 `load_frozen_dihuo_inputs()`；loader 验证 manifest pin、固定八个 artifact 名称/顺序、compressed/raw 字节数与 SHA、schema/source/workload fingerprint、八个不同 prompt SHA、每个 request 的用户消息/hash、sample/request 顺序、seed、请求 token cap、`ignore_eos`、历史 usage 和完成状态。失败不会返回 samples，故不会进入 HTTP 读身份、warmup 或生成阶段。
- 尺寸上限是 manifest 512 KiB、每个压缩输入 `<1,000,000` 字节、每个解压 JSON ≤8 MiB；gzip 只读到上限 +1 后拒绝超限。artifact 文件名由代码固定，单文件解引用后也检查 parent 仍是 request root；request root 必须与声明路径解析后相同，manifest 文件本身拒绝 symlink。显式指定的 bundle 父路径本身经 alias 到真实位置仍作为 caller 指定的 bundle root。
- frozen 模式只返回原消息、prompt 和 source provenance；历史 response/output/latency/counter/quality 未进入 sample。helper 只复制固定 7 项 input metadata（sample/suffix、corpus 偏移/长度、历史 `tokenizer_count`、历史 exactness），不会转发 `source_metadata` 中额外的结果/sentinel 字段。当前 A2v4 2K 八个已归档 receipt 的键集合与此一致。loader 校验历史 usage/clean 仅作输入收据准入，不是把旧测量当成本次结果。`dihuo_source.original_source` 是历史来源路径/hash metadata；证据另标 `corpus_file_read_this_run=false`，不是本轮实际 corpus 读取或质量收据。
- frozen 模式不构造本地或远端 tokenizer。历史 `tokenizer_count=2048` 只作为输入来源证据；本次 prompt 长度必须由 stream 最终 SSE usage 和当前 server counter fence 再验证。每请求 `source_metadata` 内的 `tokenizer_count_scope` 与 `input_tokens_exact_scope` 都标明 historical provenance，避免下游误读为本次 tokenizer 测量。
- measured 请求按输入序号设置 `seed=request_index`，使用 `max_tokens=args.output_tokens`（该复验要求 256）和 `ignore_eos=True`；arm 同样声明 256。SSE 开 `stream_options.include_usage`，最终 `usage_tokens` 传入 `counter_metrics_delta`，其中本次 `prompt_tokens` 必须等于 2048，usage 三字段必须为当前响应的真实一致值，worker/counter 和 terminal fence 必须匹配当前请求/服务 SID。
- 通用 benchmark 的 `max_tokens` 是上限语义；当前通用 `summarize_group` 对输出只要求本次真实 usage、prompt 长度、finish reason 存在、TTFT 等有效项，并不对所有历史/新任务强行要求每请求恰好 256 completion tokens 或 `finish_reason=length`。这次 2K×8 若要求新鲜输出严格 `(prompt=2048, completion=256, total=2304)` 且 8/8 `finish_reason=length`，应由本次专用 acceptance 门显式强制；不要把历史 receipt 的 256/length 当成本次结果，也不要把通用 cap 偷换成所有场景固定输出。

## 执行前必须满足的准入项

1. 已落地：frozen 模式必须提供非空 manifest 路径及 SHA-256；`is not None` 配对检测让空值 fail closed，不会退回旧 tokenizer/corpus 路径。
2. 已落地：对 frozen replay，SID 必须是 32 位小写 hex、runtime fingerprint 必须是 64 位小写 SHA-256；身份 GET 后、任何 warmup/generation 前分别精确比较，并输出两个 pinned 值。model-id 与 identity 两个只读 GET 都在八件输入先验后。
3. 已落地：manifest 文件拒绝 symlink，请求目录拒绝解析出 manifest parent，单个 artifact 解析后必须仍在 request root。manifest 父路径本身经 alias 到真实位置允许作为操作者明确指定的 bundle root。
4. 部署源代码准入在外部启动计划：执行前冻结 benchmark/workloads/frozen loader 三源 SHA，并对部署字节 `sha256sum -c`；benchmark 结果记录三份**实际执行** SHA。CLI hash 仅为事后证据，不是源码准入门。
5. 当前通用 client 仍按 API `max_tokens` cap 口径；单独的本次 acceptance 必须检查 8/8 新响应 SSE usage 为 `(2048,256,2304)`、`finish_reason=length`，同时 counter request_id、seed、max-token arm、terminal fence 与当前 SID/runtime 对齐。历史 `usage_tokens` 只用于历史输入 receipt 准入，不作为本次输出。
6. 已落地：每请求来源元数据标明 `tokenizer_count` / `input_tokens_exact` 为 historical source provenance；本次实际 prompt count 只来自新 SSE/server usage。不得以历史 count 充当当前 tokenizer 调用或当前模型 tokenizer 证明。

## 对 2K×8 的判断

对于这一次复验，**单个 SHA-pin split manifest + 八个 bounded gzip 逐件验证 + 复用既有 bench 请求/usage/counter 链**是最小且可靠的设计。它避免把历史内容塞进 benchmark 源码，也不重跑大型输入矩阵。普通 benchmark 默认流程不应切换到 frozen 输入；frozen 模式只允许 `tasks=dihuo, rounds=1`，不接受 code 混跑或多轮回放。

本结论限定于已提供的 A2v4 输入材料和当前源码审阅；不声称 workload agent 的新测试/计划已通过，不认为源码 hash 已内置到 CLI，也不把历史 A2v4 的 warmup、生成文本、时延、counter 或质量门继承到新 2K×8 运行。
