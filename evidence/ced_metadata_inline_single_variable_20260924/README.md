# A3-21 metadata-inline D 臂：短/144K/1M 混合序列单变量取证（2026-09-24）

本目录归档 2026-09-24 在 a3-21 上完成的一组真实权重实验证据：在同一个 metadata-inline D 实例上按固定顺序发送 22 token、144K、以及 6 次同一份 1M 请求，逐次记录返回内容、usage、HTTP 状态、D 侧 replay 区间与 P 侧 KV block 释放日志。本文件只写实测事实，不给因果结论。

## 原始远端路径与采集方式

- 本组远端运行目录（只读访问）：`/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_20260924_1355`
- 采集方式：`ssh a3-21` + `tar` 流式拷贝；远端只被读取，没有写入、重命名或删除任何文件。远端未安装 `rsync`，因此未使用 `rsync`。
- 归档时远端状态（`docker ps`，只读查询）：本组 D 容器 `dsv41-ced-metadata-inline-d-20260924-1355` 与其 proxy **已不在运行列表**（该次采集的原始记录 `RUN_SUMMARY_D1_D6.md` 也写明取完证据后停止 inline D 与 proxy）；P 容器 `dsv41-ced-ab-p-graph-20260924-070428` 仍在运行，本次归档没有停止或重启任何服务。

## 实验目的

在同一 D 进程上先做短请求基线、再做 144K，然后连续发送 6 次相同的 1M D 请求，观察混合序列下长请求的逐次返回行为，并把 D 侧 replay 区间与 P 侧 KV block 释放日志对应起来。样本量小。

## 服务与精确启动配置

- D 容器：`dsv41-ced-metadata-inline-d-20260924-1355`（inspect 记录 `Id=c8646177…`，创建于 2026-09-24T05:58:04Z），chips `8 9 10 11 12 13 14 15`，HTTP 18991、KV 19091（`driver.log`、`live_final_d6_20260924/D_container.inspect.json`）。
- P 容器：`dsv41-ced-ab-p-graph-20260924-070428`（inspect `Id=0db3d626…`，创建于 2026-09-23T23:04:45Z），`V41_CED_ROLE=prefill`，即本组 P 是跨臂复用的长驻实例，因此 P 日志计数含更早的实验。
- proxy 容器：`dsv41-ced-metadata-inline-proxy-20260924-1355`（inspect `Id=36a03ef6…`，端口 18992）。
- 镜像（三者相同）：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`，镜像 ID 前缀 `sha256:1f2c08195c5b…`；模型根 `/home/l00886679/models/out/v41-flat-verify3`（真实权重）。
- 运行参数：`TP=8 DP=1 PORT=18991 SERVED_NAME=deepseek-v41-ced-pd`、`MAX_LEN=1048576`、`MAX_SEQS=4`、`BAT_TOKENS=8192`、`GPU_UTIL=0.92`、`BLOCK=128`、`KV_DTYPE=bfloat16`、`GRAPH=1 EAGER=0`、`PREFIX=0 SPEC=0 SP_TOKENS=7`、`ENGRAM=1 ENGRAM_STORAGE=int8`、`NPUGRAPH_EX=1 STATIC_KERNEL=0 CPU_BIND=0`、`MULTISTREAM=0 DSA_OVERLAP=0`、`LOADER_MT=1 LAZY=1`、`ENGRAM_DEVICE_INDEX=0`、`PROFILE=0`、`V41_KV_TIER=off`。完整清单见 `serve_cmd.txt` 与 `inner.sh`。
- D 容器的 CED 开关（`live_final_d6_20260924/D_container.inspect.json` 的环境变量原件）：`V41_CED_ROLE=decode`、`V41_CED_METADATA_INLINE=1`、`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`、`V41_CED_CAPTURE_DECODE=0`，各 snapshot 开关为空串。
- 运行期关键文件 SHA-256（`live_final_d6_20260924/runtime_hashes.txt`）：`attention/dsa_v41.py=72788c49…`、`worker/model_runner_v1.py=bd250a59…`、`distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py=a93bd605…`。
- 起服侧动作（`driver.log`）：admission gate 以 mount 模式现场应用（live tree 命中 15 处）；应用固定 Core 版本的 128-token replay 调度补丁；应用固定 runner 版本的单 token prompt 尾部 eager 补丁；等待就绪 331 s。
- 采集时的健康与名册：`live_final_d6_20260924/service_state_pre_stop.txt` 记录 D/P/proxy 均 running 且 `p_health=200 d_health=200 proxy_openapi=200 proxy_docs=200`；`metadata_worker_ranks.txt` 与 `prompt_tail_worker_ranks.txt` 各 8 行（TP0–TP7）。

## 请求

- 发送顺序：`short22_inline_01` → `144kA_inline_01` → `1mD1_inline` … `1mD6_inline`，全程非流式、无 logprobs。
- 6 次 1M 请求逐次复用同一份原始请求体，SHA-256 `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`；每次发送前重新校验（各 `*.request.sha256` 记录该值）。
- 仓库内保存的是压缩后的 `*.request.json.gz`；**压缩前的 SHA-256 见同名 `.sha256`**（例如 `1mD1_inline.request.sha256` 对应远端未压缩的 `1mD1_inline.request.json`）。归档时逐个解压重算过 SHA-256，与旁证一致。

## 逐次结果

| 顺序 | 请求 | prompt_tokens | content | completion_tokens | finish_reason | u_fffd | HTTP | wall (s) | D replay positions | summary 的 exact_answer |
|---:|---|---:|---|---:|---|---:|---:|---:|---|---|
| 1 | `short22_inline_01` | 22 | `ZQ7K-3341` | 8 | stop | 0 | 200 | 4.611694 | 0..20 | true |
| 2 | `144kA_inline_01` | 144462 | `ZQ7K-3341` | 8 | stop | 0 | 200 | 10.369423 | 144333..144460 | true |
| 3 | `1mD1_inline` | 1019847 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.492258 | 1019718..1019845 | null |
| 4 | `1mD2_inline` | 1019847 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.542457 | 1019718..1019845 | null |
| 5 | `1mD3_inline` | 1019847 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.573675 | 1019718..1019845 | null |
| 6 | `1mD4_inline` | 1019847 | `null` | 1 | stop | 0 | 200 | 101.142135 | 1019718..1019845 | null |
| 7 | `1mD5_inline` | 1019847 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.540291 | 1019718..1019845 | null |
| 8 | `1mD6_inline` | 1019847 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.515998 | 1019718..1019845 | null |

补充说明：

- 8 次全部 HTTP 200、`cached_tokens=0`、`u_fffd=0`。按发送顺序，1M 长请求的第 4 次（`1mD4_inline`）返回 `content=null`、`completion_tokens=1`、`finish_reason=stop`，其完整 `choices` 字段另存于 `probe/inline_single_variable/D4_choices_full.json`。
- 22 token 与 144K 两次的 `summary.json` 记为 `exact_answer=true`（内容就是校验码 `ZQ7K-3341`）；6 次 1M 的 `exact_answer` 均为 `null`，即探针未对该用例自动判定对错，本目录只记录响应原文。
- 响应体 SHA-256 的完整值见各 `*.response.sha256`：`short22=146ec76c40f0b1c6…`、`144kA=7ebbf19e6c993f08…`、`1mD1=1921a59f26ee94a5…`、`1mD2=e0f88c96bdfb6b93…`、`1mD3=e0d13428c4ac183b…`、`1mD4=0925b739126e20e7…`、`1mD5=70532d511df21745…`、`1mD6=bf803c7748c7b3b1…`。
- 8 条 D replay 记录见 `probe/inline_single_variable/live_final_d6_20260924/D_replay_all.txt`（`[CED-D] replay ... positions=... before uncached last token`），与上表一致：短请求 replay 覆盖 `0..20`，144K 覆盖 `144333..144460`，6 次 1M 都是 `1019718..1019845`（128 tokens）。

## P 侧 KV block 延迟释放

P 是跨臂复用的长驻实例，日志计数为累计口径。按请求提取的 P 侧行（`probe/inline_single_variable/live_final_d6_20260924/{short22,144kA,D1..D6}_P_api_match.txt`，每文件 1 行）为：短请求 `Delaying free of 7 blocks`、144K `Delaying free of 1140 blocks`、6 次 1M **各 1 条 `Delaying free of 7979 blocks`**。整份最终 P 日志 `live_final_d6_20260924/P_serve.log` 里 `Delaying free of 7979 blocks` 累计 30 条（含更早实验）、`Force freed` 0 条；D/proxy 日志中均无 `Force freed`。

## 已知限制

- 响应中 `token_ids` 为 `null`，本归档**没有 sampler 侧 token id**；`content=null` 只说明消息内容为空，**不能据此断言模型输出了 EOS 或零个 token**（本次 usage 记为 `completion_tokens=1`、`finish_reason=stop`）。
- 本组每个长度只有 1 次（1M 为 6 次串行重复），样本量小；本目录不给出失败率，也不对 `content=null` 的原因下结论。
- `live_after_short22_inline_01/` 与 `live_after_144kA_inline_01/` 只含 `proxy.log` 与 `service_state.txt`（另有 0 字节的 `D_replay_excerpt.txt`、`D_serve.log`、`P_serve.log`）；原始记录 `RUN_SUMMARY.md` 说明这两次抓取时 D/P 的 docker log 为空，因为 vLLM 写的是宿主挂载的 `serve.log`，完整宿主日志在 `live_final_20260924/`、`live_final_d6_20260924/`。
- `live_after_1mDN_inline/D_replay_excerpt.txt` 是**累计**口径（含此前请求的行），逐请求对应关系请用 `live_final_d6_20260924/` 下的 `{short22,144kA,D1..D6}_*_match.txt` 与 `D_replay_all.txt`。
- 除 `*.request.json` 外，其余文件（response、summary、headers、status、transport、日志）**原样保留、未压缩**：本组最大的日志为 `probe/inline_single_variable/live_final_d6_20260924/P_serve.log`（约 470 KB），远低于 90 MB 上限。
- `probe/inline_single_variable/SHA256SUMS` 是**远端生成的清单**（覆盖远端未压缩的 `*.request.json`）。为与仓库既有惯例一致（参见 `evidence/ced_graph_ab_20260924/graph_prompt_tail/probe/.../SHA256SUMS.remote`），本目录把它改名为 `SHA256SUMS.remote`，内容未改；在仓库内对它跑 `sha256sum -c` 会得到 217 项 OK、8 项（已压缩的 `*.request.json`）报告缺失，这是压缩后的预期结果。压缩前该清单 225 项全部通过（归档时已验证）。

## 目录结构

- `inner.sh`、`serve_cmd.txt`、`driver.log`、`serve.log`：D 服务启动脚本、参数汇总、起服日志与运行日志。
- `probe/inline_single_variable/*.{request.json.gz,request.sha256,response.json,response.sha256,summary.json,headers,status,transport.txt,start.txt,end.txt,curl.stderr}`：每次请求的原始字节与旁证（`request.json` 已压缩，未压缩原件不入库）。
- `probe/inline_single_variable/D4_choices_full.json`：`1mD4_inline` 响应 `choices` 数组的完整字段。
- `probe/inline_single_variable/RUN_SUMMARY.md`、`RUN_SUMMARY_D1_D6.md`：本次采集当时的原始小结（d7953e5 该臂的阶段性记录），本 README 未改动它们。
- `probe/inline_single_variable/live_after_1mDN_inline/`、`live_after_short22_inline_01/`、`live_after_144kA_inline_01/`：每次请求后的 D/P/proxy 日志与状态快照。
- `probe/inline_single_variable/live_final_20260924/`、`live_final_d6_20260924/`：过程与收尾的完整宿主日志、容器 inspect、NPU 与监听快照、逐请求 match 文件、`runtime_hashes.txt`、`service_state_pre_stop.txt`。
- `probe/inline_single_variable/SHA256SUMS.remote`：远端清单（见上）。
- `SHA256SUMS`：本目录所有入库文件的 SHA-256，可用 `sha256sum -c SHA256SUMS` 校验。

## 相关证据

- 同项目另一组 metadata-inline D 臂（同一 1M 请求串行 8 连发）：`evidence/ced_metadata_inline_repeat8_20260924/`，远端路径 `/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450`。
- 本组远端路径见本文件开头。
