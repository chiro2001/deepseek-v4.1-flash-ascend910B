# DSA overlap off: DSA-OFF 图模式 1M D 对照

归档位置：`ced_prompt_tail_dsaoff_graph_20260924_1211/`。原始文件与完整 P/D/proxy 日志清单 `SHA256SUMS` 含105项，本地逐项校验通过。P 保持容器 `dsv41-ced-ab-p-graph-20260924-070428`、ID `0db3d62627d8680aa17e0398e1f26cec8b1a654bdc6724ed247ec1235e06739d`。D 为 CED decode TP8，GRAPH=1/EAGER=0、`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`、`DSA_OVERLAP=0`、`MULTISTREAM=1`；runner SHA 为 `bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`。代理连接 P/D ports 18990/18991，通过 18992。

## 1M D 请求

四条请求均逐字节使用原始 request SHA `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`，max_tokens=64、temperature=0、stream=false、无logprobs，body实际1,019,789 tokens。服务端日志时间为UTC；下表列出的日志时刻均为UTC。传输耗时按TP0→TP7排列。

| API ID | curl wall / A3 start–end (+08) | 实际输出 | P 延迟释放 | D KV transfer | replay / eager |
|---|---|---|---|---|---|
| `chatcmpl-1212c816-e473-4132-874b-475aaa3eab4e` | 101.858518s / 12:28:34–12:30:15 | `RB9N-6014` | 04:30:11，7979 blocks | 04:30:14–15；118.11/120.76/118.95/120.76/119.70/118.11/118.93/132.72 ms | 04:30:15，`1019718..1019845`；8个chunk128、8个tail eager |
| `chatcmpl-d9cafa32-64be-410b-b0ec-e872253c3bc3` | 101.776889s / 12:31:20–12:33:02 | `RB9N-6014` | 04:32:58，7979 blocks | 04:33:01；121.04/121.17/121.30/122.66/119.68/121.86/121.49/134.54 ms | 04:33:01，`1019718..1019845`；8个chunk128、8个tail eager |
| `chatcmpl-5eb3459b-025f-4e2f-bbec-e5dd11f365d1` | 101.739844s / 12:34:07–12:35:48 | `RB9N-6014` | 04:35:44，7979 blocks | 04:35:47–48；121.60/121.19/120.56/124.88/121.80/128.12/121.34/133.35 ms | 04:35:48，`1019718..1019845`；8个chunk128、8个tail eager |
| `chatcmpl-029ad711-38c2-46c1-a4d5-eb2f628f3758` | 101.199682s / 12:36:54–12:38:35 | `message.content=null` | 04:38:32，7979 blocks | 04:38:35；132.05/130.10/129.63/129.46/129.94/131.20/128.65/143.30 ms | 04:38:35，`1019718..1019845`；8个chunk128、8个tail eager |

四个响应均 HTTP 200、U+FFFD=0。前三条 usage 为 prompt/completion/total `1,019,847/7/1,019,854`；第四条为 `1,019,847/1/1,019,848`。四条的APIServer access行没有 API ID 或时间戳；D日志中的 POST 200按串行请求顺序对应 1M A/B/C/D，行号为1018、1065、1097、1128。P delay、D transfer/replay行本身带API ID，可直接对应。

## D4 响应字段与日志边界

D4 原始响应SHA为 `df44b591c0af4cb25a22e177faebfb97e7d6eac9a8902fe2fa093a536d0e686f`。`choices[0]` 有 `index/message/logprobs/finish_reason/stop_reason/token_ids/routed_experts`；message 的 `content/refusal/annotations/audio/function_call/reasoning` 均为 null，`tool_calls` 字段缺失；choice 的 `logprobs/token_ids/stop_reason/routed_experts` 均为 null，`finish_reason=stop`。D日志在该 API ID 附近只记录 KV transfer、CED-D replay、chunk和prompt-tail处理；没有记录该请求的生成 token ID、sampler选择或EOS事件，不能据 content=null 或 completion_tokens=1 推断实际 token。

D启动配置为 `FULL_DECODE_ONLY`，capture sizes 为 `[1,2,3,4,8,12,16,20,24,32]`。日志记录 graph capture 在04:16:50完成、耗时20秒、占0.77GiB；全日志仅有一条TP0 `Replaying aclgraph`（04:22:25），无API ID，时序靠近此前短针请求；1M A–D附近没有可按API ID确认的aclgraph replay行。日志另有多条 `causal_conv1d_update_npu` 不可用、回退PyTorch实现会同步并stall decode-FULL图捕获的WARNING。这是日志事实；这些记录不能独立证明1M各请求实际是否执行了graph decode，也不能分解首token时间。

P/D/proxy完整日志按 `ERROR/Traceback/OutOfMemory/OOM/RuntimeError/Exception`扫描均为0匹配。第四条与前三条的输出不同只作观测记录；不从wall或日志间隔推断精度差异原因。

## 验证边界

另有短针和144K A门控记录：短针精确返回`ZQ7K-3341`；144K A精确返回`ZQ7K-3341`，并从完整D日志确认8/8 replay chunk与tail-eager标记。短针replay为`0..20`、只有21 tokens，因此chunk128标记数为0是预期；第一次对短针的本地摘要计数器曾误要求8个chunk128标记，未触发重发，已写明原因。
