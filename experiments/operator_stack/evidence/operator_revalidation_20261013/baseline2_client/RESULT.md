# Full-TP8 base2：地火 2K 唯一客户端校准

日期：2026-10-13。root 核验 base2 ready、资源/源码/IPC 门与服务身份后，授权并发起了一次客户端测量；客户端没有启动服务或初始化 NPU，未重试、未额外发 canary/质量请求。首次 base1 因一个 worker 的 free-memory guard 自然 exit 1、未 ready（按 spawn 顺序疑似 rank4，失败日志未显式打印 rank/physical）；本结果仅对应后续 base2。

## 身份与输入门

- Model/form：`dsv41-operator-revalidation-tp8-20261013` / `a2-tp8-a3sim`；SID `3cbd844d9f684c43867b4b85c9c78ebe`；runtime fingerprint `79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319`。client 自身的 model/identity GET 与 root 放行 pin 相同。
- Root 放行配置：a3-21 D 容器 physical8–15、TP8/DCP1、`src_v11/generic-v7`、BI/async/双流/draft graph、SEQ1/PREFIX0、Engram index0、base2 arm。A2v4 的旧 SID/fingerprint 不用于本次绑定。
- 外部冻结的 client source 三 SHA 通过 staged `sha256sum -c`：`benchmark.py` `b39d9c3ddc44701d7463f0ee2b2a36b55e01fd6fe69fab85b11db20e8651f666`；`workloads.py` `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`；`frozen_dihuo_inputs.py` `605cecfb4853e3657de80b95296ae48e862ea91067fe63b3c3ae12793ddeaa2c`。
- 使用 A2v4 2K frozen input manifest，SHA256 `afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec`；8 个 gzip 与输入 gate 全部通过压缩字节、解压字节、messages/text SHA、顺序、budget/seed/ignore_eos/usage/clean 等验证。Measured 8 条请求的 prompt SHA 与 message SHA 均逐项等于 A2v4 原件；未重新生成 prompt、未读 corpus/tokenizer、未复用历史计时/输出/quality/counter。

## 测量结果

1 条 excluded warmup 后完成 1 round、8 条串行 HTTP measured request，8/8 HTTP 200、无 SSE error、finish=`length`、usage=`2048/256/2304`、8/8 clean。Dihuo 的 clean 是空输出/替换字符/重复长块的文本完整性检查，不是语义正确性评分。

| 请求 | actual rounds | A | HTTP ms/step | HTTP tok/s | TTFT (s) | server-fenced ms/step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129 | 1.976744 | 29.431483 | 67.164274 | 0.322713 | 30.653841 |
| 1 | 91 | 2.802198 | 28.997775 | 96.634924 | 0.322191 | 30.697334 |
| 2 | 93 | 2.741935 | 28.899912 | 94.876949 | 0.324083 | 30.549348 |
| 3 | 73 | 3.493151 | 28.559412 | 122.311716 | 0.320801 | 31.029640 |
| 4 | 111 | 2.297297 | 28.408765 | 80.865793 | 0.323639 | 30.040291 |
| 5 | 130 | 1.961538 | 28.392346 | 69.086874 | 0.323559 | 29.575253 |
| 6 | 89 | 2.865169 | 28.200136 | 101.601231 | 0.322353 | 30.227506 |
| 7 | 92 | 2.771739 | 27.982578 | 99.052314 | 0.315183 | 29.917819 |

Aggregate：808 actual decode rounds；ranks 0–7 各 808 rounds、808 graph-selector proposal calls、8 eager-selector calls；HTTP SSE completion 总计 2048 tokens，扣除每条一个 seed 后有效 decode tokens=2040，故 `A=2040/808=2.524752`。HTTP decode 三元组为 **(28.634375 ms/step, A=2.524752, 88.172083 tok/s)**，tok/s 与 `A`/HTTP ms-step 自洽差为0。TTFT mean/median/min/max 为 **0.321815 / 0.322533 / 0.315183 / 0.324083 s**。

另列不可混算的 server 计数：server-fenced **30.292077 ms/step**，scope 是 terminal prefill bookkeeping→worker completion fence，可能含 metrics RPC delay；server prefill 累计 **1.591585 s**，由 `/delivery/metrics` `prefill_seconds` max-rank delta 提供，scope 是 first scheduled prefill→terminal prefill CPU bookkeeping。prefill 不从 TTFT 推导。

## Draft graph 证据边界

`identity.rank_descriptions[].draft_graph=true` 只证明配置。另有 raw per-request `/delivery/metrics` 的 `draft_graph_calls`/`draft_eager_calls` 差分：八 rank 每请求 graph-selector 计数分别为 129/91/93/73/111/130/89/92，与各自 actual rounds 相同；eager-selector 每请求均为1，八 rank合计一致。

计数来源是实测服务的 delivery counter：[`worker.py`](../../../../delivery_matrix/worker.py) 对实际 `initial_draft` 与 round proposal records 中的 `runtime_graph` 布尔值求和；[`formal_dspark_counters.py`](../../../formal_dspark_counters.py) 包住实际 `drafter._propose` 返回后调用 `draft_finished`，并从 `_dspark_use_graph_rt(drafter)` 采样 selector。这比 identity 配置标记强，支持“实际 proposal 调用被 runtime graph selector 标记”的结论。**它不是底层 Graph.replay 调用/捕获完成事件或图ID回执；因此本结果不宣称独立证明了 graph capture/replay 已完成。** raw metrics 内所有请求的 terminal fence 与 terminal TP join 均为 true，8-rank decode-round 差分一致。

后续只读源码/日志审计补充：八 rank 均有实际 FULL ACLGraphWrapper 包装、带三层 draft attention metadata 的捕获及完成日志；同服务亦有通用 ACL graph replay 日志。见[后续 draft 图证据](../draft_graph_evidence/README.md)。上段保留客户端计数证据的原始边界，不表示后续审计仍缺少捕获证明；808 个 selector 计数仍不能等同于逐次独立采集的底层 replay 事件。

## 归档与趋势边界

- 远端 raw JSON 保留在 `/home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/evidence/dihuo_2k_base2_frozen_20261013.json`，1,091,384 bytes，SHA256 `b6dd175184cce2c2fab235cc0dc47bb3a2adb8886b78f5735cab80e208211084`；没有把该 >1 MB 文件传回本机。
- 远端 split gzip request receipts 均 <1 MB（最大 15,388 bytes；最大解压 188,137 bytes）。原件双 SHA、prompt/output SHA 在 [`request_manifest.json`](request_manifest.json)；其 SHA256 `87c66ad1d78bc6eabc25af32faa34af2d86db10a3aa4fd1589fbe454fb30b39c`。远端 `sha256sum` 与 manifest 字段一致；首次结果归档时本地没有正文或 gzip，后续私有审计副本的位置及边界见下一项。
- 仓库内只存去敏 compact summary [`compact_summary.json`](compact_summary.json)，28,747 bytes、SHA256 `90e72c166c939691e0b3522cc61d45be4f6fdbaa768168d57dfe56c529a6dcdd`，split manifest [`request_manifest.json`](request_manifest.json) 与本报告；这些仓库文件均不含 `prompt_text` 或 `output_text`。按 root 后续独立复核授权，8 个 gzip 请求体及 byte-identical manifest 另存于本机私有临时目录 `/home/chiro/tmp/dsv41-operator-revalidation/raw-audit/base2/`（manifest 在目录根，request gzip 在 `requests/`）；每个请求 gzip <1 MB，压缩及解压 SHA/size 均与 manifest 一致。该临时审计副本含正文，仅供 root 私有复核，不在 repo/Git；远端 >1 MB raw JSON 仍保留原地、未传。
- A2v4 2K 旧参照为 `(28.071337 ms/step, A=2.524752, 89.940586 tok/s)`、server-fenced `29.646913 ms/step`、server prefill 1.332814 s。新旧服务 SID、runtime/config、物理设备范围不同，仅作同输入跨实例趋势；不能当同进程 A/B、算子收益或 performance admission。
- 本次唯一授权生成窗口已用完；客户端没有补发生成、轮询或质量请求。详见[客户端准备与输入门计划](../client_prep/BASELINE_CLIENT_PLAN.md)。后续现场仅做最终只读在线核验（models/identity GET 各一次，无生成），服务保持在线；见[最终在线快照](../final_live/final_live_snapshot_20261013.md)。
