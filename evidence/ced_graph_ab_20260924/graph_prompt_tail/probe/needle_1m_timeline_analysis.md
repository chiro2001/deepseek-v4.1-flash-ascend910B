# A3-21 1M CED P/D 时间线与资源审计

日期：2026-09-24。此审计只分析已归档的请求时间戳及 P/D/proxy 日志，不发请求或改变运行实例。

host 时间戳是 `+08:00`；P/D logger 的 `INFO 09-24 HH:MM:SS` 按 UTC 读，再加 8 小时对齐。所有请求串行提交到同一 P、D、proxy 实例，没有在两次完整 D 请求之间重启；D1 与 D2 中间插入了一次 max1/top5 诊断。

## 请求与日志事件

| 请求 | API ID | host start–end；curl wall | P 延迟释放 7,979 blocks | D KV transfer；replay；eager 标记 | 输出 |
| --- | --- | --- | --- | --- | --- |
| A | `chatcmpl-1f1d6122-4a0f-4507-9af9-23318369c345` | 09:37:33–09:39:16；102.622 s | 09:39:12 | 09:39:15；`1019718..1019845`；09:39:15 | `ZQ7K-3341`，正确 |
| B | `chatcmpl-4299d746-36dd-4f6b-a16e-7fb202800944` | 09:40:30–09:42:12；102.171 s | 09:42:09 | TP0–3/5–7 09:42:11，TP4 09:42:12；`1019717..1019844`；09:42:12 | `VX2M-8890`，正确 |
| C | `chatcmpl-f0159097-5352-433b-9fa8-53a883ceb43e` | 09:43:21–09:45:03；101.895 s | 09:44:59 | 09:45:02；`1019719..1019846`；09:45:02 | `HT4P-5527`，正确 |
| D1（首次） | `chatcmpl-d1f40b8b-cf56-4611-9ed9-272b3120020f` | 09:46:19–09:48:32；133.175 s | 09:48:29 | 09:48:31；`1019718..1019845`；09:48:32 | `Tech-D9q7Wm`，错误 |
| max1/top5 | `chatcmpl-6d7c6bec-af98-45c2-a4f5-066dac48cb8d` | 10:17:02–10:19:09；101.609 s | 10:18:52 | 10:18:54；`1019718..1019845`；10:18:55 | `RB`，预期答案的前缀 |
| D2（同 SHA 重发） | `chatcmpl-cfd16b81-c2b3-482c-a3a2-aac60b41d89c` | 10:25:04–10:29:25；224.115 s | 10:28:56 | 10:28:59；`1019718..1019845`；10:28:59 | `RB9N-6014`，正确 |
| D3（追加复测a） | `chatcmpl-42db0df7-4ecb-4dab-9341-db0dd4858c51` | 10:55:17–10:56:59；101.673 s | 10:56:55 | 10:56:58；`1019718..1019845`；10:56:58 | `RB9N-6014`，正确 |
| D4（追加复测b） | `chatcmpl-81d4a2f7-beca-4a35-b391-bd7be8db3a80` | 10:59:25–11:01:06；101.600 s | 11:01:03 | 11:01:05；`1019718..1019845`；11:01:05 | `Uhq9-3DqT`，错误 |

每条请求均有 TP0–TP7 八条 chunk-128 replay 记录和八条 `one-token prompt tail forced eager` 标记。D logger 的 API-server HTTP 200 行没有 request ID 或时间戳；表中通过串行顺序配对，不能据此分解阶段耗时。proxy 日志也没有逐请求时间戳或请求 ID。

### 每个请求的 P→D transfer 耗时

以下为 D 端 Mooncake 日志按 TP0 至 TP7 排列的单 rank `took` 时间，单位毫秒：

- A：109.37 / 108.11 / 107.50 / 109.38 / 108.39 / 109.35 / 105.65 / 107.15
- B：105.01 / 105.35 / 102.90 / 104.76 / 109.38 / 104.63 / 103.54 / 102.21
- C：107.37 / 107.69 / 110.22 / 108.66 / 106.90 / 107.73 / 107.53 / 105.30
- D1：128.20 / 137.24 / 128.28 / 135.42 / 132.01 / 129.29 / 126.30 / 123.41
- max1/top5：155.62 / 142.85 / 143.09 / 144.78 / 150.48 / 145.82 / 141.13 / 140.83
- D2：145.68 / 144.67 / 140.21 / 143.64 / 144.18 / 144.48 / 143.52 / 142.88
- D3：134.05 / 135.25 / 135.41 / 137.69 / 136.59 / 139.15 / 139.69 / 139.75
- D4：123.89 / 126.90 / 125.48 / 127.42 / 127.61 / 128.42 / 128.38 / 131.49

P 侧 `Delaying free of 7979 blocks` 是 request 进入 `FINISHED_LENGTH_CAPPED` 后安排异步远端传输时的延迟释放记录（实现位置：`experimental/ced/mooncake_hybrid_connector.py:1518`）。它可作为 P 侧 handoff 准备的日志锚点，不是精确的 prefill 起点。P 该行到 D 首条 transfer 打印相差 2–3 秒；D transfer 与 replay 通常在同一秒或下一秒。日志只到秒，transfer 的毫秒值来自各 rank 的独立计时。

`.start`/`.end` 文件记录的是整秒外包络，curl `wall` 是单独的毫秒计时。A/B/C/D1 的外包络时长为 103/102/102/133 秒，与 curl wall 接近；max1 与 D2 的外包络分别为 127 秒和 261 秒，比 curl wall 多 25.39 秒和 36.88 秒。因此外包络不作为阶段计时。不同 wall 值只记录为观测，不解释精度变化。

## 首 token 与图解码

max1/top5 请求在 `max_tokens=1` 下返回 `RB`，其 logprob 为 -0.000223，top-5 为 RB / R / 【 / 抱歉 / EOS。它说明该请求返回的首 token 候选强烈偏向正确前缀，但响应没有首 token 独立时间戳。`one-token prompt tail forced eager` 是 prompt 未缓存尾 token 的处理标记，不能当作首 token 计时。

D 启动命令配置为 `FULL_DECODE_ONLY`、EAGER=0。归档的 `aclgraph_replay.log` 只有更早短请求的一条 TP0 `Replaying aclgraph` 记录；上述各条 1M 请求均没有带各自 API ID 的图 decode replay 时间。PROFILE=0，也没有 profiler trace。因此现有证据不能拆出首 token 延迟或逐 token graph decode 耗时；full D 响应包含多个 completion token，但不能仅凭最终文本测出后续 token 走图的具体时间。

P之外的三份完整D日志（早期Graph+tail、DSA overlap off Graph+tail、正确CED eager诊断）都各有8条 `causal_conv1d_update_npu` 不可用的启动WARNING，内容是回退PyTorch同步实现可能stall decode-FULL。该warning跨模式都出现，不是某个输出模式独有的标记；日志也没有将它关联到某个特定请求。

## HBM、KV 与负载观测

启动日志原值：P Available KV cache 15.16 GiB，D 15.15 GiB；双方 `GPU KV cache size` 均为 3,322,350 tokens，最大 1,048,576-token 请求的并发报告均为 3.17x。rank 级 Current KV cache 约 15.15–15.17 GiB。P 启动 free device memory 为 60.88–61.12 / 61.27–61.28 GiB，D 为 60.88–61.13 / 61.27–61.28 GiB；weights 37.43 GiB、peak activation 3.15 GiB、non-torch 0.62–0.63 GiB、NPU graph memory P 0.77–0.78 / D 0.79 GiB，warmup 后 torch reserved / allocated 53.43 / 52.87 GiB。原始行在 `P/serve.log` 和 `D/serve.log`。

`probe/meta/npu-smi.txt` 是单张没有内嵌采样时间的快照：全部16颗 Health=OK、AICore=0%；P chips0–7 HBM 64,891–65,147 / 65,536 MB，进程内存62,058 MB；D chips8–15 HBM 59,204–59,471 / 65,536 MB，进程内存56,386 MB。它无法关联到任何一条 1M 请求，Health=OK 也不等价于 ECC 计数器为0；快照没有 ECC 计数器字段。没有 CPU 利用率样本或按请求的 NPU 负载、带宽、HBM 时序。现有 route-probe 行是算子路径计数/耗时样本，不是 CPU/NPU 利用率。

完整 P/D/proxy 日志及 npu-smi 快照中，ERROR、OOM、Traceback、ECC、独立词 `EE` 均无匹配。P/D 保持同一进程和启动配置；日志没有 per-request HBM 分配/负载测量，因此不能说不同输出对应的显存或 CPU/NPU 负载相同。

## D eager 诊断臂（DSA overlap 保持开启）

P container ID 与 graph+tail run 相同。D端只切为`GRAPH=0 EAGER=1 V41_CED_GRAPH_PROMPT_TAIL_EAGER=0`，保留`DSA_OVERLAP=1 MULTISTREAM=1`及其余模型、BF16 KV、Engram、CPU_BIND、CED consumer连接器配置。硬门确认D实际命令有`--enforce-eager`、无`FULL_DECODE_ONLY`，runner SHA回到`67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e`，CED connector/DSA/scheduler replay patch正确挂载。

原始D request SHA `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`，max64/temp0/nonstream/no-logprobs，连续提交4次，4次答案均精确为`RB9N-6014`，usage均1,019,847/7/1,019,854，U+FFFD=0：

| 请求 | API ID | start–end +08 | curl wall | P handoff | D transfer/replay | 输出 |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | `chatcmpl-277ba5fe-fbdd-4cfa-9a19-d1e692d4076b` | 11:46:27–11:48:23 | 116.170616 s | 11:48:05 | 11:48:08；`1019718..1019845` | 正确 |
| 2 | `chatcmpl-0b49049d-6f2a-4650-8699-909239437f50` | 11:49:28–11:51:10 | 102.116898 s | 11:51:06 | 11:51:09；`1019718..1019845` | 正确 |
| 3 | `chatcmpl-2d13983a-f446-4f90-9c0b-66dc61e31d03` | 11:52:12–11:53:54 | 102.093542 s | 11:53:50 | 11:53:52–53；`1019718..1019845` | 正确 |
| 4 | `chatcmpl-97c1e932-0643-4188-a1b8-12e92ed72929` | 11:54:54–11:56:36 | 102.270511 s | 11:56:32 | 11:56:34；`1019718..1019845` | 正确 |

每条请求均有8个Mooncake worker transfer记录、8条chunk128复用记录、APIServer 200；prompt-tail marker为0，符合该臂未加载runner patch的配置。最终P/D/proxy health均200，error/OOM/Traceback扫描计数为0，服务保持运行。每条的原始request/response、SHA、逐rank transfer和replay excerpt在`probe/ced_prompt_tail_d_eager_cedrole_20260924_1129/`。

曾有一台`V41_CED_ROLE`为空的错误启动尝试；它没有CED connector/DSA/scheduler replay挂载，未发任何model request，已作为configuration mistake单独归档，不计入四次精度样本。停掉后未做设备reset，待8–15 HBM自然回落并确认无holder后才启动正确eager D。

## 判读边界

1M A/B/C均正确；四次完整同SHA图+tail D中两次正确、两次错误。顺序为：D1错 → 插入max1/top5并返回正确前缀 → D2正确 → D3正确 → D4错。D1/D2间有max1诊断，D3/D4则为紧邻串行复测。四次图+tail D replay区间相同、日志无服务错误；wall、KV transfer时长和请求顺序不解释输出变化。后续单变量D eager诊断臂中，同SHA四次完整D均正确。此结果仍是有限样本，只表明在本轮条件下两臂结果不同，不据此单独确认某个机制。下一步按主Agent指示测试DSA overlap关闭的图+tail配置。

原始请求/响应、状态、hash与逐请求 replay摘录在`ced_graph_prompt_tail_probe_20260924_090512/needle_1m_graph/`；D3/D4的完整服务日志快照在`needle_1m_graph/live_logs_after_repeat_a/`与`needle_1m_graph/live_logs_after_repeat_b/`。
