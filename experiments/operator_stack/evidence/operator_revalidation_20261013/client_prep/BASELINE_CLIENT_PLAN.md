# 2026-10-13 full-TP8 base1/base2 客户端现态校准计划

## 范围与放行状态

本文件最初是 CPU-only 准备。site 首次 base1 启动曾因 rank4 free-memory guard 自然 exit 1，未 ready；当时未发请求。root 后续确认 base2 ready、独立核验 SID/fingerprint 和八 rank 配置，并授权唯一客户端窗口；已完成的一次结果见 [`baseline2_client/RESULT.md`](../baseline2_client/RESULT.md)。本计划仍作输入/协议说明；没有后续自动重跑授权。

目标窗口仅为地火 2,048 输入 / 256 输出：8 条 measured HTTP 请求，串行、1 round；保留 A2v4 协议中的 1 条 excluded warmup。不是质量/代码/多模态矩阵，也不是同进程 A/B。结果标为**新 full-TP8 base2 的一次现态校准**，不能与 A2v4 跨进程/跨配置相减成算子收益。

root 在放行前核验了 served model `dsv41-operator-revalidation-tp8-20261013`、SID/fingerprint、`src_v11/generic-v7` 冻结源与实际八个 rank；physical 8–15；BI/async/双流/draft graph、SEQ=1、PREFIX=0、Engram index=0。实际 form 为 `a2-tp8-a3sim`。不要把旧 A2v4 SID/fingerprint 当成本次身份。客户端保存的 GET model/identity 与放行 pin 一致；详见结果 compact receipt。

## 冻结参照输入与字节 SHA

2K A2v4 参照档为 [`DIHUO_RESULT_INDEX_20261013.md`](../../../../delivery_matrix/evidence/a2sim_tp8_v4/client/DIHUO_RESULT_INDEX_20261013.md)，冻结请求收据在 [`dihuo_2k_a2sim_v4_20261013/`](../../../../delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/)。本机 CPU 校验确认 8 个 gzip artifact SHA 与 manifest 一致，解压后 `.request.prompt_sha256` 也与 manifest 一致；未读取或复制 prompt 正文到本计划。

| 序号 | A2v4 prompt UTF-8 SHA256 | 请求 gzip artifact SHA256 |
|---:|---|---|
| 0 | `407fe12d7b6921e391d10143dd45aab1f0259b58292878d5e8217dfb308202bd` | `c4a2ee604ca5b5b9097f58c0048265740a157d7646b46b2fdb419378b91da161` |
| 1 | `67e56a5c7b3f6922d402e19c39c7ee4a28a90a0a0400b74983192f199d7453b3` | `575ed90736ac7eb3403e588bd1a188905bede7713002482bf3514f6b3ae5bee3` |
| 2 | `ac8fa5fa5bf6e2deca4623091b759e2539a9d05cff50aceecf78895d78138970` | `4659011ec2d791e9dde59d6b1caebb3aad081ef2d8e2a0f5a3fb7f7b4ce0ffbf` |
| 3 | `f8211101041d3c5bd99a4035c309041cc65126f00e461113a33c45e72f18c153` | `23b81740d3dda8a69963372d54c5eb94d8e9d0f214931170752c6b3881327732` |
| 4 | `8459c6fadddd3304a224f908054d819fd0984920e06d3a5e11c5ab5bfb7e839c` | `c5b97208f2e2636c01afd4c288cf21cf63df819cdf52a41689d554563e575196` |
| 5 | `4921f678afdcb27732cadb2337fcf3b69d94ec3c0fecb556f32c9646dc5f62eb` | `47297fe7fc61a7208cd0072f01cd8462d9cb1b5e5bbe6c4803417f72974ede27` |
| 6 | `1d0e24f3524b78b20713e6b5b89e4b71bcacf1fa6cadb062922a104b6e8dee79` | `b0ffda01baad49aab9ead87a8f544d527582ba131b2b4124bccf7eec8a96a5f4` |
| 7 | `3de9a7bbceb895172ab8be541760d74d93ae8cb3cb9349c0ffbd47a5794a3ad0` | `eb5c86cc644464f6ac96c1f241ebd29029b4c81c23640907160a09310ab929a7` |

冻结输入来源及客户端字节：

- A2v4 2K `manifest.json` SHA256：`afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec`；完整 gzip 大小与 SHA 另见 [ARCHIVE_RECEIPT.md](../../../../delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/ARCHIVE_RECEIPT.md)。
- 地火源文件 SHA256：`00aefd13430d4be45424db3c3f832b6c012d9c60c199518f45feed5d0a57e52b`。五个 suffix SHA：`continue.txt` `ed8a9f2f9caec4937119347ae212e3e1de7c26623f7e8ebc070197d8503201e9`；`extract.txt` `6a78065d11b7ed4c17ee20faea356bc6a480230c8bb43fd96421a34f293b2fd3`；`qa.txt` `7e79b50c9ec1d8c69ef3b501ce86b53612c35f4f08f2ed3cd2b3e97b4eeb0bea`；`quote.txt` `ac277599ba94093a47bee48e44de7c3afa3991e25b0300de44fead93098ad39a`；`summarize.txt` `e538ee1a79ee4a750d9b90e1e25454fddce07009d9c85d322989f42f5b8bfede`。
- A2v4 历史结果使用 client v2：`benchmark.py` SHA256 `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`；`workloads.py` SHA256 `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`。新 frozen-input 入口新增 `frozen_dihuo_inputs.py` 并修改 `benchmark.py`，**不能复用 v2 源包**；实际部署前由 root 冻结三文件 SHA，并在 site staged source 上执行 `sha256sum -c` 外部门。CLI 不自 pin 源码 SHA。

## 最小执行命令（只在新 ready gate 放行后运行）

新入口直接读取 A2v4 冻结 request messages，不重新生成 prompt、不读取 Dihuo corpus，也不调用 tokenizer。先把 `manifest.json` 与 `requests/dihuo_round00_request00.json.gz` 至 `request07.json.gz` 按原目录结构放入独立任务输入目录；8 个 gzip SHA 必须与本计划表一致，manifest SHA 必须为下列 pin。输入加载器会再次验证全部压缩/解压/prompt/messages SHA、顺序、预算、seed、`ignore_eos`、usage 与请求完成状态；任何错误都在访问服务前失败。

实际执行前将冻结 client 三文件暂存到 task root 的 `client_src/`，并用 root 提供的 `CLIENT_SOURCE.sha256` 执行 `sha256sum -c`。SID/fingerprint 采用 root 对新 ready 服务核验后提供的 32/64 位小写十六进制值。本次这些部署/身份门均已通过。

```sh
TMPDIR=/home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/tmp \
PYTHONDONTWRITEBYTECODE=1 \
python3 /home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/client_src/benchmark.py \
  --base-url http://127.0.0.1:18960 \
  --model dsv41-operator-revalidation-tp8-20261013 \
  --tasks dihuo --input-tokens 2048 --output-tokens 256 \
  --rounds 1 --warmups 1 --request-timeout 600 \
  --frozen-dihuo-manifest /home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/inputs/manifest.json \
  --frozen-dihuo-manifest-sha256 afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec \
  --expected-service-instance-id 3cbd844d9f684c43867b4b85c9c78ebe \
  --expected-runtime-fingerprint 79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319 \
  --api-key '' \
  --json-out /home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/evidence/dihuo_2k_base2_frozen_20261013.json
```

`--warmups 1` 是排除在 measured 8 条之外的一条单 token warmup，与 A2v4 相同。Frozen mode 在任何 GET 前完成 manifest+8 文件门；bad pin/损坏文件不触发任何 GET、tokenizer、warmup 或 generation。之后执行 `/v1/models` 与 `/delivery/identity` 两个 ownership GET；SID/fingerprint 若与显式 pin 不同，在 tokenizer/warmup/generation 前拒绝。此次保持原 client 的 8 条 serial 测量协议：每条 request 的 before metrics → arm → SSE → after metrics 与 actual-round 差分不变；没有同进程 A/B。仅执行这一个授权窗口。

上面是已完成的一次调用的复现命令，输出为 `/home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/evidence/dihuo_2k_base2_frozen_20261013.json`；不要再次执行。唯一窗口已用完。

## Frozen 输入门与当前响应验收

- Frozen gate 必须验证 A2v4 manifest pin 和所有 8 个 gzip 原始字节；同时验证解压 JSON SHA、`prompt_text`/`prompt_messages` SHA、唯一有序 prompt marker、source/workload fingerprint、2048/256 预算、每条 `seed=request_index`、`ignore_eos=true`、历史 usage=2048/256/2304、`finish_reason=length` 与历史 clean 状态。它不依赖本地语料或 tokenizer。
- Gate 只输出 input-only samples 白名单（sample/suffix/offset/slice/tokenizer provenance、messages/text 和历史字段作用域）；旧 output、质量、TTFT、latency、`A`、worker counters、前后 metrics snapshots 均不进入新测量。历史 tokenizer_count 与 usage 只是输入来源证据，不是新测量结果。
- 新请求仍须独立验收：measured 8/8，HTTP 200、无 SSE error、当前 usage 精确 `(2048,256,2304)`、`finish_reason=length`、8 个输出 clean；当前 `/delivery/metrics` 的 8-rank actual decode rounds/tag/window/fence/TP join 仍必须有效并一致。不得复用历史 actual rounds、输出、评分或性能字段。若任何当前输出不满足精确 usage/finish gate，则本次不能作为有效 2K 校准。
- 该 frozen input manifest SHA `afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec`，仍对应 [`ARCHIVE_RECEIPT.md`](../../../../delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/ARCHIVE_RECEIPT.md)。root 已在 CPU-only 路径用真实归档 8 件调用 loader PASS；新自包含 synthetic fixture 测试覆盖哈希/预算/顺序/source/symlink/size 负控、bad pin 与 SID/fingerprint 前置顺序、历史计时字段隔离和 8 条串行计数协议。

## 指标与结果验收口径

- **HTTP/SSE 口径**：记录每条 `ttft_s`、`request_wall_s`、SSE `decode_s` 与最终 usage。TTFT 是 HTTP 请求到首个 SSE 内容，不等于 prefill time；`request_wall_s` 也不直接等于 decode。要求每条 HTTP 200、无 SSE error、`finish_reason=length`、usage=(2048,256,2304)，输出非空且 `output_cleanliness` clean。
- **prefill**：只报告 server-sourced cumulative `prefill_seconds` delta；普通 v4 路径按每请求 ranks 的最大值聚合，边界 first scheduled prefill → terminal prefill CPU bookkeeping。TTFT 与 prefill 统计分列，不能互推。如果当前服务没有该计数源，就写 null/ unavailable，不以 TTFT 或请求总墙钟补值。
- **actual rounds 与 decode 三元组**：不是以 completion tokens 推估。client 对每条请求读取 before/after `/delivery/metrics`；校验 ranks 0–7 的请求轮数差、request tag、窗口 fingerprint、terminal fence/TP join 及 pending/errors，再累加实际 `decode_rounds`。按实际 rounds 算 `A=decode_tokens/rounds`；async raw worker decode token 若为空，client 按冻结协议从 SSE completion usage 减一个 seed token 归一化。HTTP decode tuple 为 `(client_decode_ms_per_step, A, decode_tok_s)`，其中 `client_decode_ms_per_step = SSE decode seconds ×1000 / actual rounds`。**server-fenced ms/step 另列**：终端 prefill bookkeeping 到 worker completion fence，可能含 metrics RPC 时间，不与 HTTP tuple 混算。
- **rank counter clean/raw SHA**：每条请求保留八 rank before/after 完整快照与差分；要求 ranks 0–7 的 `decode_rounds` 差值一致、draft graph/eager 等可用字段一致，无请求 tag/window mismatch、pending/error 或不闭合的 terminal fence。输出 measured request count、每 rank round 合计、draft graph/eager 合计、clean/8。跑后计算 raw JSON SHA256，并保留 client/source SHA；raw 大文件留任务 TMPDIR，evidence 仅记其路径、大小、SHA 与紧凑摘要。
- **评分**：Dihuo 没有正确性/答案语义评分器。`8/8 clean` 只代表 `output_cleanliness()` 的非空、替换字符与重复长块完整性筛查；不代表回答质量。本轮不跑 GSM8K、Vision、code 或其他质量矩阵。

A2v4 可比参照为 8/8 clean、aggregate actual rounds 808（ranks 0–7 各 808）、TTFT 平均 0.310032 s、server prefill 合计 1.332814 s、HTTP decode tuple `(28.071337 ms/step, A=2.524752, 89.940586 tok/s)`，server-fenced `29.646913 ms/step`。这是另一个 SID/fingerprint、A2-sim v4 在 physical0–7 的历史值，只作跨实例同 workload 生成法趋势参考，不是本次 service 的 baseline 或 perf admission。完整结果见 [A2v4 2K RESULT](../../../../delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/RESULT.md)。
