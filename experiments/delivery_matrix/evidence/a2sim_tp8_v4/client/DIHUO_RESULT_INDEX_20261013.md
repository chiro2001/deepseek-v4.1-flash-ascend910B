# A2 simulated TP8 v4 Dihuo 三档结果索引

本索引汇总 A2 v4 在 2K、32K、147456 三档各 8 个串行正式请求。三档均为 benchmark exit 0、8/8 clean；每个请求均 HTTP 200、`finish_reason=length`，最终 SSE usage 为目标 prompt token 数 / 256 completion tokens / 二者之和。每档另有 1 个被排除的 warmup，不计入下表。三个档位按序完成，未重跑或并发。

## 绑定身份

- Model / form：`dsv41-delivery-a2sim-tp8-v4` / `a2-tp8-a3sim`
- SID：`9d2ba214732b4eed84dc4f416f27c6ae`
- Runtime fingerprint：`61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`
- 三档均使用冻结 client v2：`benchmark.py` SHA-256 `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`；`workloads.py` SHA-256 `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`
- 服务身份确认 prefix caching 关闭；三档所有 measured 请求的 prompt usage 都与精确 tokenizer 校准目标一致。

## 结果摘要

| 目标输入 | 每请求 HTTP usage (prompt/completion/total) | TTFT 均值 (s) | Server prefill 合计 (s) | HTTP decode (tok/s) | A | Client decode (ms/step) | Server-fenced (ms/step) | 每 rank draft graph / eager |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2048 / 256 / 2304 | 0.310032 | 1.332814 | 89.940586 | 2.524752 | 28.071337 | 29.646913 | 808 / 8 |
| 32,768 | 32768 / 256 / 33024 | 4.733177 | 36.926546 | 98.051972 | 2.901849 | 29.595011 | 30.479052 | 703 / 8 |
| 147,456 | 147456 / 256 / 147712 | 22.988100 | 180.522022 | 98.743217 | 3.197492 | 32.381892 | 34.002879 | 638 / 8 |

`draft graph / eager` 是按每个请求 before/after 的 rank 累计计数差计算；表中每 rank 的合计值在 ranks 0–7 一致。`Server prefill` 为 `GET /delivery/metrics` 的累积 `prefill_seconds` 请求差分，边界是 first scheduled prefill → terminal prefill CPU bookkeeping；`server-fenced` 单独报告 terminal prefill CPU bookkeeping → worker completion fence，可能包含 metrics RPC delay。它们均不由 TTFT 推导。HTTP decode 与 client decode 是独立 HTTP/SSE 口径。

32K 档曾有一条进度消息误抄 2K 的 usage 数字；该档未重跑。原始 receipt 的 `config.input_tokens_target` 和 `workload_manifest.input_tokens_target` 均为 32768，精确 remote tokenizer 查询 148 次，8 条 HTTP usage 的 prompt count 均为 32768。root 已独立审核三档全部分拆件，均 PASS；147456 此处是八个正式性能请求，不是独立精度门的四针。

## 逐档归档

- 2K：[完整结果](dihuo_2k_a2sim_v4_20261013/RESULT.md) · [归档收据](dihuo_2k_a2sim_v4_20261013/ARCHIVE_RECEIPT.md) · raw：a3-21 `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_2k_a2sim_v4_20261013.json`，2,439,806 bytes，SHA-256 `7ca31c18ea7832f95396b9109a2378b0c23ae411e2ae52dd02e2085c7812a057`
- 32K：[完整结果](dihuo_32k_a2sim_v4_20261013/RESULT.md) · [归档收据](dihuo_32k_a2sim_v4_20261013/ARCHIVE_RECEIPT.md) · raw：a3-21 `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_32k_a2sim_v4_20261013.json`，6,133,283 bytes，SHA-256 `86886e31ebd27556fabe3f8db4c7d1c4bc1cbe7fba660d8c12314b39cb6227d6`
- 147456：[完整结果](dihuo_147456_a2sim_v4_20261013/RESULT.md) · [归档收据](dihuo_147456_a2sim_v4_20261013/ARCHIVE_RECEIPT.md) · raw：a3-21 `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_147456_a2sim_v4_20261013.json`，16,657,226 bytes，SHA-256 `07d19104aeff8ced36aaf922893d386b041c1d675a954cd55e02df375a5277d2`

三份完整原始 JSON 保留在远端任务 TMPDIR，不经 SSH 整体传输；每档仅逐件传输小于 1 MB 的 gzip request receipts、manifest 与 identity snapshot。各档旧/新路径、大小和 SHA-256 逐件核验记录在对应 `ARCHIVE_RECEIPT.md` 中。
