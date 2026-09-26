# A3-21 metadata-inline D 臂：同一 1M D 请求串行 8 连发（2026-09-24）

本目录归档 2026-09-24 在 a3-21 上完成的一组真实权重实验证据：在固定的 P/D/proxy 三件套上，把**同一份 1M D 请求体**串行发送 8 次，逐次记录返回内容、usage、HTTP 状态、D 侧 replay 区间与 P 侧 KV block 释放日志。本文件只写实测事实，不给因果结论。

## 原始远端路径与采集方式

- 本组远端运行目录（只读访问）：`/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450`
- 采集方式：`ssh a3-21` + `tar` 流式拷贝；远端只被读取，没有写入、重命名或删除任何文件。远端未安装 `rsync`，因此未使用 `rsync`。
- 除顶层 `serve.log` 外，`probe/` 全部文件与 `inner.sh`、`serve_cmd.txt`、`driver.log`、`BUILD_INFO.txt` 在本地与远端**逐字节一致**（两端对这批文件的联合聚合值均为 `sha256=ab2f983abc01095311482aa9b7bc1faaac59ae80d62e7a19e8eb6331de2d53e4`）。
- 顶层 `serve.log` 是 D 服务**实时日志的前缀快照**：拷贝时远端仍在被 `/metrics` 轮询追加（远端当时 279679 字节）。本地副本 279600 字节，已验证与远端第 1..279600 字节完全一致（本地 `sha256=1593b381904db5473678f046253e45da5bd6d5eab3d21d2e13cf50f285bbff4d`），所以它是可验证的一致快照，但不等同于远端现行文件的最终内容。
- `BUILD_INFO.txt` 在远端即为 0 字节占位文件。
- 归档时远端状态（`docker ps`，只读查询）：本组 D 容器 `dsv41-ced-metadata-inline-repeat8-d-20260924-1450` 与其 proxy 仍在运行；P 容器 `dsv41-ced-ab-p-graph-20260924-070428` 仍在运行。本次归档没有停止或重启任何服务。

## 实验目的

在固定 P 实例、固定镜像、固定解码图配置下重复发送完全相同的 1M D 请求，观察返回是否稳定、以及在重复采样中是否再次出现 `content=null`。8 次是同一请求的串行重复，样本量小。

## 服务与精确启动配置

- D 容器：`dsv41-ced-metadata-inline-repeat8-d-20260924-1450`，chips `8 9 10 11 12 13 14 15`，HTTP 18991、KV 19091（`driver.log`）。
- 镜像：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`；模型根 `/home/l00886679/models/out/v41-flat-verify3`（真实权重，`LOAD_FORMAT=<real weights>`）。
- 运行参数：`TP=8 DP=1 PORT=18991 SERVED_NAME=deepseek-v41-ced-pd`、`MAX_LEN=1048576`、`MAX_SEQS=4`、`BAT_TOKENS=8192`、`GPU_UTIL=0.92`、`BLOCK=128`、`KV_DTYPE=bfloat16`、`GRAPH=1 EAGER=0`、`PREFIX=0 SPEC=0 SP_TOKENS=7`、`ENGRAM=1 ENGRAM_STORAGE=int8`、`NPUGRAPH_EX=1 STATIC_KERNEL=0 CPU_BIND=0`、`MULTISTREAM=0 DSA_OVERLAP=0`、`LOADER_MT=1 LAZY=1`、`ENGRAM_DEVICE_INDEX=0`、`PROFILE=0`、`V41_KV_TIER=off`。完整清单见 `serve_cmd.txt` 与 `inner.sh`。
- KV 连接：`--kv-transfer-config {"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"19091",...}`（`serve_cmd.txt`）。
- 起服侧动作（`driver.log`）：admission gate 以 mount 模式现场应用（live tree 命中 15 处）；应用固定 Core 版本的 128-token replay 调度补丁；应用固定 runner 版本的单 token prompt 尾部 eager 补丁；等待就绪 301 s。
- **本组没有直接取证容器级 `V41_CED_*` 环境变量**：归档内容里没有 D 容器 inspect，`live_after_repeat8_D*/D_container.log` 为 0 字节。metadata-inline 通路的**直接证据**是 D 侧日志中的 `[CED-META] inline metadata group=...` 标记：每个请求后的快照 `D_inline_markers.txt` 都含 200 条，覆盖 `Worker_TP0_EP0`…`Worker_TP7_EP7`（`D_inline_worker_ranks.txt`）。本目录不据此推断未取证开关的取值。

## 请求

- 8 次请求使用同一份原始请求体，SHA-256 `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`；每个 `repeat8_DN.request.sha256` 记录的就是该值，8 份逐字相同。
- 每次 `prompt_tokens=1019847`、`cached_tokens=0`、非流式、HTTP 200。
- 仓库内保存的是压缩后的 `repeat8_DN.request.json.gz`；**压缩前的 SHA-256 见同名 `.sha256`**（`repeat8_DN.request.sha256` 对应远端未压缩的 `repeat8_DN.request.json`）。归档时逐个解压重算过 SHA-256，与旁证一致。

## 逐次结果

| # | content | completion_tokens | finish_reason | u_fffd | HTTP | wall (s) | D replay positions | 响应体 SHA-256（前 16 位） |
|---|---|---:|---|---:|---:|---:|---|---|
| D1 | `RB9N-6014` | 7 | stop | 0 | 200 | 105.370867 | 1019718..1019845 | `b85e65a296c97700` |
| D2 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.469440 | 1019718..1019845 | `24246f4688b4be2f` |
| D3 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.382553 | 1019718..1019845 | `f3c4735123e96a0c` |
| D4 | `null` | 1 | stop | 0 | 200 | 101.097201 | 1019718..1019845 | `901c817a4589a591` |
| D5 | `RB9N-6014` | 7 | stop | 0 | 200 | 102.688735 | 1019718..1019845 | `10be102cbbcd8d66` |
| D6 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.339587 | 1019718..1019845 | `f72b36c6f0729af3` |
| D7 | `RB9N-6014` | 7 | stop | 0 | 200 | 101.339809 | 1019718..1019845 | `79de9268686145bf` |
| D8 | `null` | 1 | stop | 0 | 200 | 100.963459 | 1019718..1019845 | `222e6d96df2fec2c` |

补充说明：

- 8 次全部 HTTP 200、`cached_tokens=0`、`u_fffd=0`；`summary.json` 的 `exact_answer` 字段 8 次均为 `null`（探针未对该 1M 用例自动判定对错），本目录只记录响应原文。
- D4 与 D8 的完整响应体见 `repeat8_D4.response.json`、`repeat8_D8.response.json`：两者 `choices[0].message.content` 为 `null`、`usage.completion_tokens=1`、`finish_reason=stop`、`token_ids=null`。
- D 侧 replay 区间来自每次请求后的日志快照 `live_after_repeat8_DN/D_replay_excerpt.txt`，8 次全部是 `positions=1019718..1019845`（128 tokens，"before uncached last token"）。
- 响应体 SHA-256 的完整值见各 `repeat8_DN.response.sha256`。

## P 侧 KV block 延迟释放

P 服务是跨臂复用的长驻实例，其宿主日志中的计数是累计口径。按请求顺序抓取的快照中，`Delaying free of 7979 blocks` 的行数为 31、32、33、34、35、36、37、38（`live_after_repeat8_D1..D8/P_serve.log`），即每次 1M 请求各新增 1 条、合计 8 条，与 8 次请求一一对应；8 个快照中 `Force freed` 行数均为 0。每次对应请求的 P 侧行见 `live_after_repeat8_DN/P_api_match.txt`（例如 D1 的 `chatcmpl-0e864727-...-924ace06`，D8 的 `chatcmpl-ce107fe6-...-aaa3a1e9`）。

第 8 次请求后的服务状态见 `live_after_repeat8_D8/service_state.txt`：`P=running D=running proxy=running`，`p_health=200 d_health=200 proxy_openapi=200`。

## 已知限制

- 响应中 `token_ids` 为 `null`，本归档**没有 sampler 侧 token id**；`content=null` 只说明消息内容为空，**不能据此断言模型输出了 EOS 或零个 token**（本次 usage 记为 `completion_tokens=1`、`finish_reason=stop`）。
- 8 次是同一请求的串行重复，样本量小；本目录不给出失败率，也不对 `content=null` 的原因下结论。
- 本组没有保存 D 容器 inspect / 进程环境，容器级 `V41_CED_*` 取值未直接取证（见上文）。
- 顶层 `serve.log` 为前缀快照；远端同一路径的文件在本目录生成后仍会被 `/metrics` 轮询继续追加，因此后续远端内容不在本快照内，本目录的 `SHA256SUMS` 固定的是本地副本。
- 除 `*.request.json` 外，其余文件（response、summary、headers、status、transport、日志）**原样保留、未压缩**：本组最大的日志为 `probe/inline_repeat8/live_after_repeat8_D8/P_serve.log`（约 550 KB），远低于 90 MB 上限。

## 目录结构

- `inner.sh`、`serve_cmd.txt`、`driver.log`、`serve.log`：D 服务启动脚本、参数汇总、起服日志与运行日志（顶层 `serve.log` 为前缀快照）。
- `probe/inline_repeat8/repeat8_DN.{request.json.gz,request.sha256,response.json,response.sha256,summary.json,headers,status,transport.txt,start.txt,end.txt,curl.stderr}`：每次请求的原始字节与旁证（`request.json` 已压缩，未压缩原件不入库）。
- `probe/inline_repeat8/live_after_repeat8_DN/`：每次请求后的 P/D/proxy 日志快照与取证（`P_serve.log`、`D_serve.log`、`proxy.log`、`D_replay_excerpt.txt`、`P_api_match.txt`、`D_inline_markers.txt`、`D_inline_worker_ranks.txt`、`service_state.txt`、`D_serve_cmd.txt`、`D_inner.sh`、`D_driver.log`、`D_container.log`、`P_container.log`）。
- `SHA256SUMS`：本目录所有入库文件的 SHA-256，可用 `sha256sum -c SHA256SUMS` 校验。

## 相关证据

- 同一批实验中的另一组 metadata-inline 单变量臂（22 token / 144K / 1M 混合序列）：`evidence/ced_metadata_inline_single_variable_20260924/`，远端路径 `/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_20260924_1355`。
- 本组远端路径见本文件开头。
