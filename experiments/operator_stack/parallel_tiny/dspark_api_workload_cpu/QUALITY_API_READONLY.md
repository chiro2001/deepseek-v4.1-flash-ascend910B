# GSM100 / Vision23 与正式 DSpark API：只读接线结论

本文件只读取脚本及已有正式 client 的来源，未运行任何评测、HTTP 或服务。这里只列 **100/100、23/23 的验收门**，不声称 DSpark 已取得这些结果。6 个本地脚本均冻结了源码 SHA 与函数片段；其中 4 个底层 client 与 `/work/client/` 的已有正式副本逐 SHA 相同，见 `evidence/quality_script_source_guards.json`。

| 脚本 | 实际请求/口径 | DSpark / A=1 假设 | 新 API 的影响 |
|---|---|---|---|
| `serve_formal_tp8.py` | 选择 audited/perf arm 后启动官方 OpenAI API | `verify_selection` 明确拒绝 `speculative_decoding=true`，要求所有 arm 的 A=1；命令未加 speculative-config，capture size 为 `[1]` | 不能拿来启动本次正式 DSpark 候选；只读保留，后续使用新独立 launch/selection |
| `run_formal_client_acceptance.py` | `/v1/models` + 进程所有权；bench8；GSM；Vision | 入口要求 `selection.measured.A==1`，所有权记录 spec=false，timing 写 `1000/per_stream_med`；模型名前缀及官方 api_server 进程名也硬编码 | 新 orchestrator 需绑定 DSpark 候选的实际 source/config/weights/process identity；不能保留 A=1 换算或旧前缀假设 |
| `tests/acc_eval.py` | 官方 `encoding.encode_messages`，chat 模式的 **预编码字符串**经 `/v1/completions`；默认 SSE、512 输出预算；串行 conc=1 | 无 A=1/spec=false 条件，不计算 DSpark rounds/A；准确率判据独立于投机解码 | 可用原底层脚本对 general API 验收；固定 2K token-ID/256/min_tokens=256 的私有 endpoint 不兼容这些请求 |
| `tests/vision_accuracy_check.py` | `/v1/chat/completions`，image_url data URI；默认 SSE，失败可回退非流式；23 cases，默认 max_tokens=32；附带 4-token 图像/文本 token-cost 探针 | 无 A=1 限制。`--expect-dspark` 只记录/警告 metrics，不会据此拒绝质量 PASS；记录的 `1+accepted_tokens/draft_tokens` 不是真实每请求 A | 需要完整 vision/general API。metrics 的“mean_acceptance_length”不能代替 v4 actual-round A 或性能三元组 |
| `tests/t_vision.py` | 调用上一个脚本，归一 `cases/pass_n/verdict` | 不传 `--expect-dspark`；自身退出码只要求至少 19/23，不等于正式 23/23 | 新正式质量报告必须检查 **23/23**、原始 returncode/verdict、空答/错误与 mandatory negatives，不能只看 wrapper exit=0 |
| `tools/bench_concurrency.py` | streaming token-arrival，`/tokenize` 与 `/metrics`；按 counters delta 估 A | `A=1+n_spec*accepted/draft`，默认 n_spec=7；不是逐请求 actual rounds；旧 orchestrator 强制 A=1 | 正式 DSpark K=5 的性能使用已冻结 API bridge + v4 counters；不能引用默认 7 或按 1000/tok-s 换算 |

GSM 门保留原任务：canonical train 的前 8 条作为 8-shot，test 的前 100 条，官方 encoder 的 chat 模式，temperature=0、conc=1、max_tokens=512；要求 `n=100, correct=100, empty=0, errors=0`。脚本本身即使准确率不足也可能正常退出，正式报告必须读取 summary。其断点文件按 idx 复用结果，后续 DSpark job 要用新的 output/tag，不能复用 A1 服务的旧 `.jsonl`。

已有 canonical 数据仅记录来源与 SHA，原文件留在远端：

- `/work/client_data/gsm8k/train.jsonl`：7,473 行，SHA `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`。
- `/work/client_data/gsm8k/test.jsonl`：1,319 行，SHA `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
- 来源为官方 `openai/grade-school-math` 的 `grade_school_math/data/{train,test}.jsonl`；source metadata SHA `7ad55d39dadf0737401faccad1f7dd7894f75595967dce1e961e0ed32ff514fc`。

Vision 内置 20 个 carrots/corn 短问答加 3 个 negative，共 23 项。正式门应要求 `cases=passed=23`、`empty_or_garbled=errors=0`、mandatory negative 全通过、verdict=PASS 和退出码 0；正式模型的 vision 配置、真实 vision 权重、官方图片和图像预处理须在新实际 job 中验证。脚本的默认 80% 门和 wrapper 的 19/23 门不能替代该正式门。`--expect-dspark` 警告及累计 metrics 也不授真实 DSpark 消费或性能证明。

后续连接新正式 API 的最小步骤（本轮均未执行）：

1. 为新候选冻结完整 checkpoint/tokenizer、SDK/operator 源码与 TP8/DSpark K=5/graph/vision 配置，并记录实际服务的模型名、端口、PID、源码和加载配置；保留 load_format=real、full40、完整 vision/MTP 权重与原严格精度要求。
2. 私有性能服务按现有 bridge 的 2 warmup + 8 measured 协议完成文本性能验证，所有 counter window 必须 finish。性能使用 v4 的 actual rounds 与 `len(output_tokens)-1`，每请求及 aggregate 三元组自洽。
3. 在后续已分配的独立 general-quality job 中，使用同一候选的 checkpoint/source/DSpark/vision 配置，提供 `/v1/models`、`/v1/completions` 和 `/v1/chat/completions`；绑定该 job 的实际身份。不要只放开 private ASGI 白名单：私有 AsyncLLM adapter 还要求 armed 的固定 window，通用质量请求的 prompt/预算/stream/image 语义不同。
4. 原底层 GSM/Vision 脚本可传新 general API 的显式模型名、canonical GSM 数据、官方 encoder 与图片目录运行；用新的输出目录，保留原评分、预算与所有负例。新的 orchestrator 替换旧 selection 的 spec=false/A1 条件和旧所有权硬编码，不修改旧 serve/client 或已冻结 private bridge。
5. 分别收集实际 GSM100/100、Vision23/23 的原始输出与严格 summary，绑定候选 source/config/process identity；单独审查 DSpark 实际调用与性能证据。private 文本 clean/repetition、2048/256 prompt 清单或任何本包 CPU 成功均不能授这两项质量门。

private performance 服务当前只接受三条 `/private/dspark/*` 路由；公开 completion/chat、stream 与多模态会被拒绝。它的文本洁净度检查和本包准备输入，都不能代替 vision/general API 验收。当前实际 GSM/Vision/HTTP/performance/formal admission 均为 false。
