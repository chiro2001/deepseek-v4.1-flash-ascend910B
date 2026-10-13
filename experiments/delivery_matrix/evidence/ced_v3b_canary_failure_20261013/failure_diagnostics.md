# CED v3b first-canary failure and archive receipt (2026-10-13)

采集到本机时间约 2026-10-13 06:16 CST。目录内包含 bench 原始 canary receipt、P/D
实际 plan 与服务日志、proxy 日志、D rank3 Ascend device/plog 日志以及校正后的
watchdog 上下文。原始远端文件仍留在原路径；本地文件均为字节拷贝，并逐项校验 SHA-256。

## 结论边界

- CED v3b 的唯一 code canary **失败**，不能进入 code8、Dihuo 或质量/性能轮。bench
  未重试，也未发送 stop/SIGTERM；本记录中的服务/API退出是 canary 后服务自身崩溃与
  runner 收尾。没有重启 P/D、停止容器、杀进程或 reset NPU。
- 首个有具体设备诊断内容的故障来自 D TP rank 3 的 `_pool_kernel`：Ascend plog 报
  SMMU fault、AIVEC/Vector Core exception，MTE 文本为 “MTE accesses an invalid GM
  address or the cross-device memory access times out”。随后 HCCL watchdog 报错；HCCL
  callback 明确指出 fail task 不来自 HCCL。底层内存访问故障的具体错误地址/上游算子
  关系尚未裁定，本收据不声称已证明源码根因或证明待建补丁可解决。
- D 的 KV transfer 已实际到达全部 8 个 decode ranks，且随后开始 CED 128-token replay；
  因而此次不是“KV 完全没传到”的证据。bench receipt 没保存动态 `kv_transfer_params`
  字典/完整 `remote_block_ids`，不从日志反推这些未保存字段，也不为取字段重发请求。
- `d3/run.exit` 文本为 `0`，但它只表示 runner wrapper 收尾返回 0；与 D API/EngineCore
  在请求中崩溃、HTTP SSE error 500、`/delivery/metrics` backend 500 同时发生，绝不能
  将此退出码解释成 canary 或服务成功。

## 唯一 canary 与协议证据

原始 bench receipt：[`code_canary_ced_20261013_v2.json`](code_canary_ced_20261013_v2.json)，
SHA-256 `d76a4b56ce047312ddba029b963a7a4bc860cc34517fdffc23076c860f48b235`，140,954 B。
bench 的摘要原件 [`incident_summary.json`](incident_summary.json)，SHA-256
`2514188542dd1502e042ab7df57a7e47c3a4f26ba256980f5decdc200d224510`。

- Tag/request ID `dm-canary-3e48e9fe86e8454ab214308d2910f6a1`，sample
  `code-01-lru_cache`，请求 max_tokens=128，prompt SHA-256
  `53b396533a39a5533fc4cc7194c223d02fcc47ddf04e625c02714d6ac209d4c8`。
- `/delivery/arm` 成功，P 和 D 各 rank 0–7 均 `armed=true`、`async_mode=true`，同 tag；
  P ack 绑定 SID `53af26c59ea9402ba83283a4865f8057`，pair SID
  `3762dc9b083a4b33c045c877273d19a7cb5fc927dee2960b0181948d81424f41`。canary 前的
  pair `/delivery/metrics` 成功且各 rank 空闲/零计数。
- canary 后单独读取旧 P `/delivery/metrics` 的原文已保存在
  [`prefill/metrics_after_canary_20261013.json`](prefill/metrics_after_canary_20261013.json)，
  SHA-256 `667a6f7b7bd47273e1fba3376577187512aeb3ac1afb90a8e87cd43495bae7a1`（4,501 B）。
  P rank 0–7 各有且只有一条上述 tag，`prefill_calls=1`、`pending=false`、`errors=[]`；
  因而旧 P 不能与新 D 的空计数/新 proxy 严格同 tag 对齐。
- proxy 的 `/v1/chat/completions` HTTP 状态是 200，但 SSE 内嵌
  `InternalServerError`/500 “EngineCore encountered an issue”。输出为空，finish_reason
  与 prompt/completion/total usage 均为 null；output SHA 是空字节 SHA
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。该 canary 被明确
  排除在正式 code8 统计外。之后 proxy `/delivery/metrics` 返回 HTTP 502，嵌套 D backend
  `/delivery/metrics` HTTP 500/EngineCore error。
- identity receipt 中 P 为 MooncakeHybridConnector producer、kv_port=19090，D 为
  consumer、kv_port=19091，两者 prefill/decode dp=1,tp=8；pair 形态 P8+D8/DCP=1，实际
  overlay SHA `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`、BI OPP
  artifact fingerprint `0167b055e442435b2790f1a77933fe389bdaa45db67a6f2045646606609d4a6b`
  匹配。启动身份与计划详见
  [`ced_v3_pair_ready_20261013.md`](../ced_preflight/ced_v3_pair_ready_20261013.md)。
- 精确动态 KV 参数没有落在 canary JSON 或 proxy access log。代理代码在向 D 转发前校验
  transfer contract；proxy/D 访问日志及 D per-rank Mooncake 日志证实这次请求确实被转发
  并发生 transfer，但不能恢复实际 block ID 列表。不得把此限制改写成 KV canary 未发出，
  也不得凭服务配置冒充动态 contract 原件。

## 首发错误的时间线

D 原始服务日志位于本目录 [`d3/run.log`](d3/run.log)，SHA-256
`f994496ac288e3c23390872d5b1018d5a83bc0efb5f3409d9cee65f25ee94f90`，232,380 B；原远端
位置为 D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/run.log`。日志中 canary
request id 为
`chatcmpl-dm-canary-3e48e9fe86e8454ab214308d2910f6a1-804095ba`（P/transfer）和
`chatcmpl-dm-canary-3e48e9fe86e8454ab214308d2910f6a1-bb3c9472`（D/replay）。

1. P prefill 日志记录为该请求延后释放 27 个 blocks。D 日志按 TP rank 0–7 逐一记录
   Mooncake KV transfer 完成，时长 166.62–200.64 ms，均带相同 canary tag。
2. EngineCore 记录 CED replay positions `1919..2046`；8 个 worker 均记录
   `replay chunk=128: reusing global source and C2 ring`，并记录
   `[CED-GRAPH] one-token prompt tail forced eager`。
3. D rank3 plog 在 2026-10-12 21:58:01.082 UTC 开始报告 stream 47/task 13623 失败，
   早于 21:58:02 HCCL watchdog 信息。**这个设备/plog 是首因证据：**
   [`debug/plog/plog-1239342_20261012215801250.log`](debug/plog/plog-1239342_20261012215801250.log)，
   SHA-256 `498cde0367f2e5eefc6be3acf6484fb82f18e3661ae5025dac8b137742d9dab4`，7,211 B。
   其报告要点：`device_id=11, stream_id=47, task_id=13623`、SMMU fault=1、`chipId=5,
   dieId=1`、AIVEC error `0x800000`、`_pool_kernel`/program id 566/hash
   `12281891151718203137`/PC offset `+0x62c`，MTE invalid-GM/cross-device-timeout 文本、
   Vector Core retCode `0x31`、ACL event code `507035` 和 `ERR00100`。错误 kernel/task
   的具体 invalid address 与生成该地址的上游源码位置还未确认。
4. 随后 rank3 先报告 `ProcessGroupHCCL.cpp:2011` watchdog terminated；其他 ranks 接着
   出现相同 watchdog error。HCCL callback 原文称 `the fail task is not from HCCL`，所以
   HCCL 是之后的传播/故障通知，不应被写成首因。
5. 21:58:41 EngineCore 发现 VllmWorker-5 died；21:58:42 engine fatal traceback 落在
   shared-memory broadcast `acquire_read` → `RuntimeError("cancelled")`，上层成为 SSE
   `EngineDeadError`。D API 随后 shutdown；runner 最终写 `run.exit=0`。

watchdog 上下文的 newline 物理行号校正在
[`rank3_watchdog_context_lines_1113-1253.txt`](rank3_watchdog_context_lines_1113-1253.txt)，
恰含事件行前 100 行和后 40 行（含事件行共 141 行），SHA-256
`5db0e46ee79522844bed36ec0dd7ba71bffea456a73b1a090a152aaf0073c6f0`。此前基于 Python
`splitlines()` 的旧范围片段因为日志含 embedded `\r` 而错位，已保留并标 superseded：
[`rank3_watchdog_context_lines_1311-1407_SUPERSEDED.txt`](rank3_watchdog_context_lines_1311-1407_SUPERSEDED.txt)，
不是 watchdog 上下文。不要用其替代完整原始 D 日志。

## 远端原路径到本地证据副本的映射

所有远端源文件仍保留在原位置。D 文件从现有 D 容器通过 tar stream 取回；P/proxy 文件
从 a3-21 host 的现有任务挂载路径取回；没有经 `/tmp` 中转。表中 SHA 同时在来源端与本地
副本端核验相同。

| 内容 | 原件路径 | 本地证据路径 | 字节数 | SHA-256 |
|---|---|---|---:|---|
| Bench canary | `/home/chiro/tmp/dsv41-ced-runtime-20261013/bench_v2/evidence/code_canary_ced_20261013_v2.json` | `code_canary_ced_20261013_v2.json` | 140,954 | `d76a4b56ce047312ddba029b963a7a4bc860cc34517fdffc23076c860f48b235` |
| D run log | D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/run.log` | `d3/run.log` | 232,380 | `f994496ac288e3c23390872d5b1018d5a83bc0efb5f3409d9cee65f25ee94f90` |
| D run.exit | D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/run.exit` | `d3/run.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| D runner.pid | D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/runner.pid` | `d3/runner.pid` | 8 | `9378f8144cadd791f3d9cc8bd2b08651644dfa4508f36fdc573724f4cec150c7` |
| D plan | D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/service/plan.json` | `d3/service/plan.json` | 7,169 | `59fd95fab2a7ceb8250996de27b7ddcb21ee3e9b093c09bdf4dce6caa221a6d5` |
| D service selection | D 容器 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3/service/service_selection.json` | `d3/service/service_selection.json` | 12,632 | `e6ac340d38ba19fad59171caa61f4a36167fd69e0fc02da774a63ebf2ab9440b` |
| D rank3 plog | D 容器 `/root/ascend/log/debug/plog/plog-1239342_20261012215801250.log` | `debug/plog/plog-1239342_20261012215801250.log` | 7,211 | `498cde0367f2e5eefc6be3acf6484fb82f18e3661ae5025dac8b137742d9dab4` |
| D rank3 device log | D 容器 `/root/ascend/log/run/device-11/device-1239342_20261012214429596.log` | `run/device-11/device-1239342_20261012214429596.log` | 28,067 | `495741fb4da8abc24acc375dc686776bea53803ec4e1bec08305f20f3f67113d` |
| D EngineCore vLLM log | D 容器 `/root/ascend/log/vllm_ascend/vllm_ascend_20261012_214935_1239181.log` | `vllm_ascend/vllm_ascend_20261012_214935_1239181.log` | 10,806 | `1720cea45877cd5e898a7eaa8c1aab2c81700f76e5d9e50ae3780626eb3a7d83` |
| P run log | a3-21 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/run.log` | `prefill/run.log` | 143,797 | `62787884fc6dc3f23cac8fa3c96f49c61795c1f532467e17eed60516614ec3b0` |
| P plan | a3-21 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/service/plan.json` | `prefill/plan.json` | 6,937 | `e5a6234cf8227d64bcf287a483fb1b17d4a9659696751eed4af0ab9394b4a748` |
| P service selection | a3-21 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/service/service_selection.json` | `prefill/service_selection.json` | 12,539 | `214bddc0e8fe596b6fdfca354b743e0c252101a2b6188df9c54e93ccf47c7588` |
| P post-canary metrics receipt | P container `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/metrics_after_canary_20261013.json` | `prefill/metrics_after_canary_20261013.json` | 4,501 | `667a6f7b7bd47273e1fba3376577187512aeb3ac1afb90a8e87cd43495bae7a1` |
| Proxy log | a3-21 `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy_18992/proxy.log` | `proxy/proxy.log` | 8,730 | `4652354a0b5fefc3ef3d8d6d72fc9acc4df35d5f20bff61127d6fd45c40142bc` |

The two context files are derived from the preserved `d3/run.log`; their current hashes are
listed separately above. No source files or service plans were rewritten during collection.

## 崩溃后现场与未操作事项

复核时 P 容器仍 running，`127.0.0.1:18990` 和 proxy `127.0.0.1:18992` 监听；D 原容器
仍 running、restart count 0，但 D API/EngineCore 已退出、`127.0.0.1:18991` 无监听。D
容器 namespace 中旧服务 runner/API PIDs（1238305/1238311）已退出；没有对 D 容器做
restart/stop，没有杀进程，没有 NPU reset。

只读 `npu-smi info` 显示 D 对应 host NPU4–7 均 Health=OK、`No running processes found`，
每 die HBM 使用约 2,875–3,127 / 65,536 MB；P 对应 NPU0–3 仍有八个
`VLLMWorker_TP`，每 die HBM 使用约 59–60 GB。D 侧未残留该服务 worker/HBM 进程占用。

P、D、proxy 完整日志与 actual plans 都保留在本目录；D 的完整 run log/plan/selection/exit、
rank3 plog、device log 和 EngineCore vLLM 日志的原始 SHA-256 与字节数见文件树本身。原
远端 task/cache 和日志没有清理。新 D candidate 在主代理完成 source-only 修改、来源/哈希
门和 CPU negative controls 并明确放行之前不启动；任何后续单变量候选不得继承本次
canary、质量或性能结论。
