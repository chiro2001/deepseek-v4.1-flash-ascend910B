# 正式TP8原生图：strict环境变量通过三组A/A

2026-10-10 · a3-21 physical chip8–15 · 正式40层/W4A8/384专家top6权重

`formal_native_graph_strict_v1`已正常退出0。仅在新进程与通信域初始化前设置`HCCL_DETERMINISTIC=strict`，保留原生归约和单个原生图，不启用FP32修复或候选bank。三组同worker、同prompt请求的完整路由、token、Top5集合和logprob全部一致，最大logprob差0。当前2K/47-token测试恢复了可靠的原生图对照；不把它外推为所有上下文和请求形态的保证。

## 通过的具体范围

正式checkpoint为`v41-w4a8-engram-dr-vision-qrot-mtpq`，配置sha `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。正常auto/lazy与128线程加载，完整视觉权重和Engram int8保留。

模型为TP8/EP8、max_num_seqs=1、max_model_len8192、KV4GiB、block128、FULL_DECODE_ONLY capture `[1]`，static kernel开启，Q/KV多流关闭。2次warmup、1次reference、3次重复；每请求2048输入token、47输出token。每次路由矩阵均为`[2094,40,6]`，原始ID范围与专家唯一性通过。

|三组A/A指标|结果|
|---|---|
|生成token|全部一致|
|prefill/decode完整路由|全部一致，差异0|
|专家集合|全部一致|
|Top5候选集合|全部一致|
|logprob最大绝对差|0|
|原门槛|全部通过，未放宽`<1e-3`|

启动记录确认8–15无其他占用、逐chip仍为已授权80C98001；未reset或停止其他租户。当前结果是精度控制，不是性能数据，`performance_claim=null`。

## 与此前试验的关系

原BF16/AIV基线和FP32修复的整网图模式都曾未通过原生A/A。局部固定输入已把首个偏离定位到layer0 O投影AllReduce。FP32修复在eager中通过了三组A/A和FP64参考，但它没有完全解决图路径。真实向量独立试验的固定顺序AllGather求和六种配置通过精度与图重放，尚未接整网，不能替代本次原生图证据。

本次`strict`直接通过了当前正式原生图门槛，验证了用户提醒的环境变量方向。历史文档中`true`的GSM8K退化、`strict`的重复批次更正仍保留；本轮需要继续验证当前模型的客户端质量与性能，才能选择交付配置。

## 已启动叠加审计

`formal_stack_graph_strict_audit_v1`正在同一组正式worker中比较`tp8base/tp8core/tp8stack`，三套都使用`strict`和原生归约。继续检查捕获候选前后的base A/A、三组配对的完整路由/Top5/logprob与消费者cache审计。47/48输出交替覆盖C2末步两种状态。

正式HC static/HcPost保持原生；TP1的8expert/top2 router/selected-GMM不用于正式384/top6；BF16激活候选实际覆盖0；Indexer融合只在stack臂选择。不会把没有实际覆盖的项计作收益。

本轮还补齐客户端工具：GSM8K可显式指定独立服务模型名；性能工具支持`--concurrency 1 --prompt-count 8`，以max_num_seqs=1服务采集8条不同prompt的单流统计，并可拒绝模型名不匹配的服务。模拟错误服务的负控已证明在tokenize/生成之前拒绝访问。a3-21已核实本地GSM8K缓存、官方encoding、carrots/corn图片存在。

叠加精度通过后，关闭审计/profiler完成同进程配对，再启动选定臂的正式TP8 API，验证模型归属、GSM8K、视觉与客户端时延。仅在实际`(ms/step,A,tok/s)`达到≤约19ms/step且质量通过时完成目标。当前没有有效正式端到端性能或最佳服务验收，Goal保持active。

## 紧凑证据

- `evidence/formal_a321/results/formal_native_graph_strict_v1/`：启动配置、结果、三组比较、退出码。
- `evidence/formal_a321/results/formal_stack_graph_strict_audit_v1/launched.json`：正在执行的叠加审计来源与环境。
- 历史核对与取值限制见[v9](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-hccl-determinism-followup-20261010-v9.html)。
