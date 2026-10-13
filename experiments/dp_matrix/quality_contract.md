# Native-DP 矩阵质量合同

状态：规范合同，定义每种新 DP 几何的验收门；本文不是现场运行收据，也不表示任一 DP 服务已经通过。

## 范围与隔离

本合同适用于新矩阵的三种八芯片几何，均限于 physical chip 8–15：

| layout | DP | TP | DCP |
|---|---:|---:|---:|
| dp2-tp4 | 2 | 4 | 1 |
| dp4-tp2 | 4 | 2 | 1 |
| dp4-tp2-dcp2 | 4 | 2 | 2 |

每种几何必须以自己的实际服务实例、served model、service-instance ID（SID）、runtime fingerprint 和 layout 身份独立验收。每个 suite 前后读取只读 GET /dp/identity；所有 suite 的身份必须一致。身份中的实际 worker/rank、TP/DCP/EP 通信组必须与该几何相符；physical 8–15 的映射另以 PID、NPU 与 cgroup 现场证据确认，配置声明或旧计划本身不算实机证明。

已有 A2-sim TP8 质量结果只可作为旧服务上的测试口径与输入溯源，不能替代任一新 DP 几何的验收，也不能向其他 SID、fingerprint 或 layout 借用 PASS。三个 layout 之间同样不得互相继承质量结论。

## 透明路由与取证

GSM、Vision、Needle 可继续调用各自冻结的只读旧客户端，但 base URL 必须指向仅绑定 loopback 的透明 header-injecting proxy。代理仅对生成 API 请求增加 X-Data-Parallel-Rank；不得改写 JSON body、路径、方法或已有语义头，不得注入 X-DP-Matrix-Tag，也不得添加私有 transfer 字段。代理按首次见到的原始请求 body SHA-256 在 DP rank 间轮转，并将该 SHA 固定映射到同一 rank，以使相同 body 的客户端重试仍落到原 rank。/tokenize 等非生成预检请求只透明转发并记录，不要求 rank 响应头，也不计作 DP 命中；147456 Needle 的实际生成请求才计入路由验收。每个 suite 中每个 DP rank 至少要有一个成功生成请求；rank 由请求头与响应头共同证明。

不得为质量请求调用 POST /dp/arm 或打开 metrics window。生成请求显式携带 rank、没有 measurement tag 且没有活动 window 时，应走普通生成路径。代理应对响应做 tee 式取证：在流式转发的同时记录原始响应 body 字节数及 SHA-256，不得为捕获 SSE 而先缓冲整条响应。每次 attempt 至少记录 suite/case、请求 body SHA-256、选定 DP rank、HTTP status、原始响应 SHA-256 与字节数、usage、finish_reason、是否完整消费 [DONE]、响应是否正常关闭、重试关系，以及生成响应头 X-DP-Matrix-Rank。审计行必须计算 `route_matches_response`；对每个成功生成响应，该值必须为 true，响应 rank 必须存在且等于所选 rank。缺失、越界、错配、HTTP 错误、解析错误或未完整读完均判该 suite 不通过。

质量评分必须使用冻结的 evaluator、canonical 输入和阈值。重试按原客户端策略记录，不得用重试覆盖或删除失败 attempt；同一请求 body SHA 的重试必须保持 rank。最终评分只用该 suite 规定的原始/最终 evaluator 结果，不因输出 SHA 不同而判质量失败或成功，也不得用跨配置输出哈希相等代替质量评分。

## 四项硬门

### GSM100

- 输入：canonical GSM8K test 前 100 行及原 evaluator 的前 8 条 train 示例（8-shot）。test JSONL SHA-256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`；train JSONL SHA-256 为 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`。
- 冻结 evaluator `acc_eval.py` SHA-256：`aad813de1b1af090c4186b0a443b7839339928f90b63e7c3a654d0e028b8e6e2`；wire sidecar SHA-256：`f5bc086e69fed7706933390cb4287460c3aa536a378f1d73b3700c7c3eeb985d`。独立评分器 `scoring.py` SHA-256：`4369e659ed465d8fe0b3812da0743a375acaedde95b8ad0e1514d36751399832`。
- 固定口径：chat，temperature=0，max_tokens=512，conc=1，串行 prefill；保持冻结 evaluator 的原 retry policy 与模型请求语义。
- PASS：100/100；empty=0、errors=0。所有最终有效响应 HTTP 200，usage 非空，finish_reason=stop，若为 SSE 则完整消费 [DONE] 并正常关闭；代理记录 100 个 case 的请求/响应及 rank 绑定。独立 canonical 评分按冻结 evaluator 原规则执行。

### Vision23

- 输入：原始 23-case 顺序、问题和图片；冻结 evaluator `vision_accuracy_check.py` SHA-256：`af13bc4785bfcddcd46a398fe7611c6942a4875e1eee8c767ce31148ebee675f`。
- 冻结图片：`carrots.jpeg` SHA-256 `5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a`；`corn.jpeg` SHA-256 `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`。
- 固定口径：max_tokens=32、temperature=0、expect-dspark、min-hit-rate=1.0。必须运行并通过 evaluator 定义的全部 mandatory negative checks，不得删减或放宽规则。
- PASS：23/23；empty、garbled、errors 均为 0；mandatory negative checks 全通过；代理对 23 个 case 保存请求、路由 rank、原始响应及响应 rank 证据。

### Code8

- 冻结输入：八题 split manifest SHA-256 `6ec9d9267695a4b6e7d5e27a1beaf0823df8498c9762b4c7204d2892474a8704`，包含每个 split artifact 的完整 SHA-256；本地独立评分原收据 SHA-256 `9f232da2041c374ffe8d9451070f981f09a2b5944b63b374dbe1ad42a6d57c7e`。
- 冻结历史 scorer 元数据：benchmark.py SHA-256 `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`；workloads.py SHA-256 `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`。独立评分环境为 bubblewrap 私有 network namespace、只读系统 Python/stdlib/libs、512 MiB 内存、2 CPU 秒、3 秒 wall timeout，并使用任务专属 TMPDIR。
- 旧 TP8 benchmark 会调用 /delivery/identity 与 /delivery/arm，不得直接用于新 DP 服务。必须使用 DP 专用只读 replayer 消费冻结 split manifest，按冻结请求提示词生成八条普通 HTTP 请求；不得调用旧 delivery 私有 API。保留原 2,048-token 输入语义、自然结束及每题最多 1,024 个输出 token。生成的每份输出均独立归档并绑定 split artifact、prompt、输出和 scorer SHA。
- 评分必须在隔离环境内对新输出运行冻结任务/用例规则；不得在宿主机直接执行不可信生成代码。PASS：8/8 题通过，128/128 用例通过，failed case 列表为空。
- 历史远端 scorer 的 8 个 `sandbox_error` 必须原样保留为历史事实；既不把它们改写成 pass，也不覆盖本地 bwrap 评分。工作区当前 `experiments/delivery_matrix/benchmark.py` SHA-256 是 `b39d9c3ddc44701d7463f0ee2b2a36b55e01fd6fe69fab85b11db20e8651f666`，与冻结历史 SHA 不同，不能冒充冻结 Code8 scorer。SHA 匹配的原件已核实存在于 `/home/chiro/tmp/dsv41-ced-pd-v4-bench/client_source_ced_20261013_v2/delivery_matrix/`（benchmark `33c458…`、workloads `62db350…`）；DP 专用回放器只调用该冻结 client 的普通 stream 函数，独立评分器调用该冻结 workloads 的 bwrap 函数。当前门为 NOT RUN；若这些冻结文件后续不可用，则记 BLOCKED，不得换成不同字节后宣称 PASS。

### 147456-token Needle

- 冻结 runner `ced_pd_acceptance.py` SHA-256：`d12d746f1831431e6c6aa55447ae3b2d9a04e15b58800d505aaad508b7eff71e`；raw API capture wrapper SHA-256：`df0452fd6735e89a8780da8849913c946f14ec14aede8b695eb20aa5c8a71076`；语料 SHA-256：`a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`。
- 固定四针 A–D、context_tokens=147456、max_model_len=196608、max_tokens=64、temperature=0、stream=false。请求直接走标准 tokenize/chat API，不得携带 transfer/private-contract 字段。
- PASS：四条均 HTTP 200、finish_reason=stop、完整响应正常关闭，答案逐字精确匹配各自 needle，且实际 usage.prompt_tokens ≥ 145982（147456 的 99%）。只看请求中声明的 context length 不满足此门。

## 总体 PASS 与收据

一个 layout 的质量总 PASS 必须来自同一个 SID、runtime fingerprint 与服务配置下，GSM100、Vision23、Code8、四针 Needle 四项全部 PASS；任何身份变化、配置/服务重启、缺少响应 rank 证明或 suite 未运行，都不能把整体标记为 PASS。未运行、证据缺失、冻结 scorer 不可用统一记为 NOT RUN/BLOCKED，不得折算为通过。

新收据应按 layout 和 run 分开保存，并至少归档：suite 前后 identity 原件及 SHA、physical placement 证据、冻结输入/源文件哈希、每个 request/response 的路由与 wire 记录、每项 scorer 原始结果及哈希、重试与错误 attempt、总门判定。当前文件只约定验收标准；没有发起 HTTP/NPU 操作，也没有声称任何新 DP 服务已经通过。

## 参考历史收据

旧 A2-sim TP8 原始质量收据：`experiments/delivery_matrix/evidence/a2sim_tp8_v4/quality/quality_stage_receipt.md`；严格审计：`experiments/delivery_matrix/evidence/a2sim_tp8_v4/quality/strict_quality_audit.md`；Code8 结果与冻结 split 说明：`experiments/delivery_matrix/evidence/a2sim_tp8_v4/CODE8_RESULT.md`。这些文件只用于核对上述冻结口径和 SHA，不是本合同的 DP 服务 PASS 证据。
