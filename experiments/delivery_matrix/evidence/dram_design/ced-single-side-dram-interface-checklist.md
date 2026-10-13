# CED 单侧 DRAM 接线接口清单（只读源码审查）

日期：2026-10-13。目的：把 P 写 DRAM、D 优先读 DRAM、Mooncake 兜底/清理、异步释放和 CED 状态恢复所需的接口分开列清。未改实现、未导入 Torch、未启动服务或 workload。本报告将 a3-21 native ABI、当前交付 overlay 和较早的 DRAM 原型分开；原型源码不能当成当前 P 的运行时回执。

## 证据身份

| 来源 | 身份 / SHA-256 | 使用边界 |
|---|---|---|
| a3-21 实际 native offload 源快照 | `/home/chiro/tmp/dsv41-offload-research/ced-actual-offload-abi/SOURCE_MANIFEST`，manifest SHA `886fddfbde73f51c4aa65db0f30888e14719f7fba2766516bd1f4c4aa2920a3b`；vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，vLLM-Ascend `e43cf1e9f5d9bead076853aa6bcacb671465de94` | 13 个实际容器源码文件已逐个 SHA 校验；NPU handler、offload adapter、core mapping 的具体行号见下文。 |
| Ascend MultiConnector | vLLM-Ascend HEAD `e43cf1e9f5d9bead076853aa6bcacb671465de94`；`ascend_multi_connector.py` SHA `b1e25b9e2213c58bf4a05a4f6cf857ddc7d8ef3b51d9b5cb46d55ef5128ae540` | 实际版本代码来自该精确 git object。 |
| upstream MultiConnector | vLLM HEAD `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`；`multi_connector.py` SHA `32194ed05223ee482c7f0f5a9bf6786eb7e4f76eb714d11fa33dbce38935b462` | 实际版本代码来自该精确 git object。 |
| 当前 CED Mooncake overlay | `operator-stack-tp8/experimental/ced/mooncake_hybrid_connector.py` SHA `e83ed49d500138c02cf97e2078d633845578d4a4ddfc7c8fd16f9bb2e859a4e4`；亦绑定在 `ced_runtime_20261013_v1/manifest.json` | 当前交付 overlay 的静态实现；不是本轮 live P done-count receipt。 |
| CED DRAM 适配原型 | `ced-dram-offload` HEAD `8125757eceded47b6d8831ae5cbf60ac7dd3c686`；`experimental/ced/dram/connector.py` SHA `56c911312b4b4efcccb1967f25b039de49b3e55595273a2d1613b5a5b11edff4` | 只作为接口/状态流参考；不在当前 `ced_runtime_20261013_v1/manifest.json` 文件集内。 |

## 一、D 命中选择、Mooncake 抑制与释放 P delayed-free

### 实际选择语义

- `AscendMultiConnector.__init__()` 直接通过 base `MultiConnector` 构造 `_connectors`（`e43cf` `ascend_multi_connector.py:18-30`；`6e448d` `multi_connector.py:169-185`），子项配置顺序即查找顺序。它没有 `abort()` 或 `get_finished()` 覆写；这两项走 base aggregator。
- `AscendMultiConnector.get_num_new_matched_tokens()` 只对 `has_preempted_request()` 分支额外置顶，否则回落到 base（`e43cf` `ascend_multi_connector.py:140-160`）。因此 preempted/unhashed state 是一个高于普通配置顺序的独立分支，必须一并测试。
- base `MultiConnector.get_num_new_matched_tokens()` 查询每个 child，记录**首个** `toks > 0` 的 index，但不在命中后 break（`6e448d` `multi_connector.py:385-404`）。后续 child 若返回 `None`，`:395-398` 会立即把结果改成 `(None, False)`，覆盖前面已见的 DRAM positive hit。仅把 DRAM 排在 Mooncake 前面，不能保证 DRAM hit 总能被选中。
- 选中后，Ascend override 给所选 child 传真实 `blocks` 和 `num_external_tokens`，其余 child 传 `blocks.new_empty()` 和 `0`，除非 child 显式要求 full blocks（`e43cf` `ascend_multi_connector.py:127-138`）。`build_connector_meta()`、worker `start_load_kv()`、`get_finished()` 仍按 child 循环，不会因某 child 未获选自动跳过其生命周期（`6e448d` `multi_connector.py:293-340,422-433`）。

### 推荐的选择契约

1. lookup 阶段保存显式 per-request `selected_source` / `load_owner`（例如 `ced_dram` 或 `mooncake`），并把“DRAM positive 已确认”作为可短路后续 lookup 的判定；不能只依赖列表顺序。若必须调用后续 child，也要避免 Mooncake `None` 覆盖已确认 hit。
2. 仍需在 allocation/meta 阶段告诉未获选 Mooncake child“零本地块、零 token”，以完成请求清理；但这一步只能生成**空接收/控制 ACK**，绝不能带任何非空 `local_block_ids`。如果为短路 lookup 而过早清空 `do_remote_prefill` 或 `remote_request_id`，会同时丢失 P 侧清理所需的消息身份。
3. 所有 child 的 `request_finished_all_groups()` 会被聚合；多个异步 child 的 pending 计数由 `MultiConnector._extra_async_saves` 管理（`6e448d` `multi_connector.py:483-541`），P 页面要等 DRAM store 与 Mooncake send 两个独立 barrier 都完成才可复用。

### 当前 CED Mooncake 空接收路径与 P free ACK

当前交付 overlay 的 D `update_state_after_alloc()`（`operator-stack-tp8/experimental/ced/mooncake_hybrid_connector.py:1594-1638`）在 `do_remote_prefill` 与 remote metadata 完整时，将 `num_external_tokens == 0` 转为每组空 `local_block_ids`，仍加入 `_reqs_need_recv`，随后清掉 `do_remote_prefill`。worker `_transfer_kv_cache_all_groups()` 遇到本地总块数为 0 就早退、不搬字节（`:777-794`）；但 `_handle_request()` 的 `finally` 仍调用 `_send_done_recv_signal()`（`:730-758,1162-1180`）。这条路径同时满足“无 Mooncake KV pull”和“通知 P 可以清理远端发送状态”的设计目的。

P delayed-free 的当前 overlay 源码流程为：

1. P 的 `request_finished_all_groups()` 对 `do_remote_decode` 请求裁剪/筛选 remote block IDs；有实际发送块时返回 `delay_free_blocks=True` 并将 request 放入 `_reqs_need_send`（`operator-stack-tp8/experimental/ced/mooncake_hybrid_connector.py:1668-1734`）。
2. Worker 从 metadata 先登记 `reqs_to_process`，再登记 `delayed_free_requests`（`:2331-2342`；tracker 方法见 `:279-320`）。
3. D 发 `DONE_RECVING_MSG` 后，P listener 按 `remote_port_send_num` 累计各端口；达到计数时才调用 `update_done_task_count()`，清理 delayed set，并给 D 回 ACK（`:454-481`）。P 的 Mooncake `get_finished()` 从 tracker 取完成集合并作为 `finished_sending` 返回（`:2170-2189`）；scheduler 收到后释放请求块。

**重要风险：当前 P done-count 不能据源码宣称已在 DRAM-hit 路径正确运行。** `KVCacheTaskTracker.update_done_task_count()` 只接受已经出现在 `reqs_to_process` 中的 ID，否则仅记录错误并丢弃该完成信号（当前 overlay `:287-300`）；而 DONE listener 会在该调用后仍 ACK D。`add_delayed_request()` 也仅在 request 已登记时入队（`:315-320`）。因此，过早的 D empty-recv ACK 可能让 D 收到 ACK，但 P 尚未 latch 完成状态；随后 P 仍可能加 delayed free 并等到超时。`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 的 force-free 是泄漏兜底，不是成功传输/正确释放回执（`:321-342`）。应为 early-done 增加可幂等 latch，或在 P `reqs_to_process` 建立之后再允许完成 ACK；并核对多端口/PP 的 `remote_port_send_num` 计数。当前源文件无运行时的零接收 hit → P tracker 移除 → P scheduler free 证据。

同一 overlay 的 D 本地 task tracker 也有注册时序风险：`start_load_kv()` 先把 receiver task 放入并发队列（`:2318-2329`），之后才对 `metadata.reqs_in_batch` 调 `add_req_to_process()`（`:2331-2335`）。空接收很快完成时，`_handle_request()` 可能在 tracker 注册前调用 `update_done_task_count()`；该方法会因 request 不在 `reqs_to_process` 只记录错误而丢弃本地 `finished_recving`（`:287-300,730-758`）。所以需要分别验证 P source-block release 和 D receiver tracker cleanup/finished-recving 记账；D 收到 P 的 ZeroMQ ACK 只代表远端收到了消息，不等于 D worker 的 finished counter 已记账。推荐先注册两端 tracker，再提交 no-op task，或在两侧实现可幂等 early-completion latch。

### Abort 与 `get_finished()`

- 在实际 vLLM `MultiConnector`、Ascend wrapper 和归档 NPU offload connector 中没有通用 `abort()` 钩子。`request_finished_all_groups()` 是 request finalization/延迟释放协商，不是取消 DMA。
- NPU offload worker 的 `get_finished()` 只轮询 transfer end event；完成后回收 event/stream/descriptor buffers（归档 `ced-actual-offload-abi/vllm-ascend/.../native/cpu_npu.py:224-247`）。`wait(job_ids)` 会 `event.synchronize()`（`:249-253`），`shutdown()` 也同步 drain（`:255-264`）。没有中途取消接口。
- Core offload worker 将完成的 load job ID 映射为 `finished_recving`（归档 `vllm/distributed/.../offloading/worker.py:346-381`）；store 完成记在 `completed_jobs`，native worker 自身不产生 `finished_sending`（`:349-354`）。CED DRAM 原型用 `job_id -> req_id` 表和额外 barrier 补发 `finished_sending`（`ced-dram-offload/experimental/ced/dram/connector.py:172-210`），但该文件不是当前 overlay manifest 的运行时绑定。
- D 在 load 中途 abort 时，目标 NPU page 仍可能由 DMA 写；要等 event 完成，worker 返回 `finished_recving` 后由 scheduler 在 request 已 finished 的分支 free destination blocks（当前 CED scheduler overlay `scheduler.py:3038-3088`）。不能在 abort 时直接复用目标块。若实际跳过 Mooncake load，也要保留/生成 P 的 no-op done ACK；单纯从 `_reqs_need_recv` 删除请求会造成 P delayed-free 等到超时。

## 二、NPU native handler 与 `CanonicalKVCacheRef.mapping` 缺口

实际 ABI 快照（manifest 上述 SHA）给出的数据布局：

- Ascend adapter 的 `register_kv_caches()` 为 Attention cache 构造 `[num_blocks, page_size_bytes]` 的 `int8` view，并在 `offloading_connector.py:218-223` 创建 `CanonicalKVCacheRef(tensor_idx, page_size_bytes)`，**未传 `mapping=`**。组装 `group_data_refs` 后直接初始化 worker（`:300-314`）。
- `CanonicalKVCacheRef.mapping` 字段存在且可选，`CanonicalPageMapping` 含 `runs`、canonical/local page bytes、`num_writers`、`writer_index`（`vllm/v1/kv_offload/base.py:475-515`）。核心兼容注册路径会从 specs/shape/stride 派生并附 mapping（`vllm/distributed/.../offloading/worker.py:69-74,211-229`）；这条能力没有接入上面的 Ascend adapter。
- `NPUOffloadingWorker` 要求每个 NPU tensor 是二维 `int8`；CPU tensor 是 pinned `[num_cpu_blocks, npu_page_size_bytes * blocks_per_chunk] int8`（`native/cpu_npu.py:267-324`）。这是本进程每 worker 的 pinned tensor，不等同于 P/D 共享内存/共同可查的跨进程 DRAM pool。
- `SingleDirectionNPUOffloadingHandler.transfer_async()` 按 `group_sizes`、`block_indices` 和 `tensor_idx` 生成指针，每 ref 将 descriptor size 设为 `data_ref.page_size_bytes`，再以 `torch.ops._C_ascend.swap_blocks_batch(src,dst,sizes,direction)` 做**整页连续复制**（`cpu_npu.py:95-222`）。没有消费 `ref.mapping.runs` 的 gather/scatter。
- D2H (`npu_to_cpu=True`) 先让 copy stream 等当前 model stream，之后还串行等待前一 transfer 的 `end_event`；H2D 方向同样受单 handler 的前序 event 排序。每 batch 记录 `start_event/end_event`，`get_finished()` 只按队首 `end_event.query()` 完成并回收 staging descriptors（`:187-247`）。这些 async 生命周期必须覆盖未来拆成多 run 的所有 copy descriptor；不能在某一 run 完成时提前 `finished_recving` 或 free block。

若接入 per-ref canonical sharing，至少还需要：在 Ascend adapter 为 ref 填入经 rank/spec 几何认证的 mapping；按每个 `CopyRun` 的 local/canonical offsets、fragment size/count/strides 生成 gather/scatter operations；store 与 load 使用逆向 run 映射；按 `writer_index/num_writers` 选每 ref writer，而不是一个 aggregate rank-0 writer；给所有 run 定义共享 backing tensor/页分配/异步完成 barrier，并在写之前验证实际 KV bytes 相等。现在的 `NPUOffloadingSpec` 和连续 copy descriptor 都没有这些接线；不要把 metadata 中的 `runs` 当成已经节省了 host bytes。

## 三、P 侧 Engram token mirror 与 compressor ring

### Engram token mirror

`ced-dram-offload` 的 DRAM 原型只把 KV bytes 纳入 pool；P-hit 的 Engram 历史是**另一个 token mirror**：

- 保存入口（patch 参考，不在当前 runtime overlay manifest 内）：`patches/files/engram_hash.py` `PagedNgramHistory.update(input_ids, positions, request_ids, block_table, block_size)`（`:427-488`）把 `input_ids` 经 `token_map` 压缩后，按 `block_table[request_ids, positions // block_size]` 与 `positions % block_size` 写 worker-local page mirror；图像/无效 token 写 `-1`。
- 注册入口同样是 patch 参考：`PagedNgramHistory.__init__()` 仅在 `CED_DRAM_STRICT_REGISTER=1` 时 `register_history(self)`（`:397-399`）。helper `engram_history.py:7-21` 用 `WeakSet` 保存当前 worker 的所有 history，并将恢复页广播给每个 `restore_token_pages()`。
- 恢复入口：候选 `CEDOffloadingConnector.update_state_after_alloc()` 从完整 prompt token IDs 取 checkpoint 前一个 group-2/SWA 页（`:127-146`），放入 metadata 的 `engram_seed_pages`（`:155-167`）；worker `handle_preemptions()` 再调 `restore_history_pages()`（`:187-193`）。恢复 API `restore_token_pages(pages, block_size)` 对完整页做校验并重新压缩 token IDs（`patches/files/engram_hash.py:401-425`）。它不是从 DRAM KV page 自动反推 token mirror。
- 精确到请求的实施契约：`(physical_group2_page_id -> exactly block_size raw token IDs)`，与同一请求的 DRAM boundary/块代际绑定；没有 prompt IDs、history 未在本 worker 注册或页映射不一致时必须 fail closed。只保存 KV run 不会恢复 Engram mirror。

**版本限制：**当前 `ced_runtime_20261013_v1/manifest.json` 记有 `best_model_and_engram_unchanged=true`，而 `PagedNgramHistory` / `restore_token_pages` 来自较早 CED patch。不能据该原型宣称当前 P 已安装或调用此 mirror 接口；现场必须确认加载文件/注册 receipt。

### Compressor state ring

- 实际 vLLM-Ascend HEAD 中 `DeepseekV41CompressorStateSpec` 是 `AscendCircularBufferSpec`（`core/deepseek_v41.py:68-77`；该类继承 `AttentionSpec`，`core/circular_buffer.py:27-42`），block size 为 32 行 FP32 packed KV/score。它不是普通可丢弃的 Mamba state。
- 候选 CED DRAM connector 将 `prefill_groups()` 的 `recurrent` 加入 `ced_excluded_groups`（`ced-dram-offload/experimental/ced/dram/connector.py:71-78`）；`CEDNPUOffloadingSpec.create_worker()` 对这些 group 保留 group 位置但置空 refs（`:30-48`）。当前因此**没有 compressor ring store/load/restore 接口**。
- 较早 compressor patch 在 ratio-2 residual 中仅当 segment `start_pos` 落在 pair 中间时读 ring（`a2/patches/kv8-int8-pkg/vllm_ascend/ops/triton/compressor/compressor_triton.py:244-255`），而 CED DRAM prototype 限制 boundary 对齐到 `_ced_alignment`（`:107-125`；旧 serving receipt 是 1024）。由此可推导 aligned/even boundary 的首段 residual 应为零，但这只是候选源码推导，不是当前实际 runtime 的 `positions/start_pos` 一致性或恢复成功 receipt。
- 仅当未来允许奇数 checkpoint 时，必须增加**按 request + boundary 的 compressor snapshot**，保存每个 ratio-2 source 的必要 KV/score ring 行及其 circular slot/absolute-position 元数据；D 在新 request 的 state page 上恢复后才能安全执行 residual。不能在 P request 结束时把最终 32-row ring 直接冒充历史 checkpoint ring，因为它可能已覆盖目标 boundary 之后的数据。对 1024 对齐契约，保持 group1 excluded 是较小接口；仍须对 worker 新 page 的首次读取路径 fail-closed 验证。

## 四、当前可实施的接口顺序与未确认风险

建议实现/验收接口按以下顺序绑定：

1. **P store commit：**在 async NPU store 的每个 end event 全部成功后才发布 DRAM hit key；带 store/job generation、group mask、boundary、mapping hash。P 的 `request_finished_all_groups()` 需等待 DRAM job barrier；若同时启用 Mooncake send，MultiConnector 两个 `finished_sending` 都到达后才 free source pages。
2. **D lookup/select：**新增显式 `selected_source=DRAM` 与 `remote_cleanup_required` per-request 状态；对 DRAM-confirmed hit，不能让后置 Mooncake `None` lookup 覆盖已确认结果，也不能让 `has_preempted_request` 特例无意抢占。D 所选 DRAM child 接收真实分组/块映射。
3. **D async load/abort：**生成 worker job IDs 并保留目标 blocks 到 NPU end event。abort 只设 tombstone/不再 resume；它不取消 DMA。event complete 后由 `finished_recving` 走 scheduler 的 aborted-request 分支释放目标 blocks。
4. **D→P release：**若原始 remote params 建立了 P delayed-free，即使 DRAM 满足全部/部分 hit，也必须产生一个幂等 no-op `DONE_RECVING` 或显式等价 cancel ACK，带 `remote_request_id` 和完整 `remote_port_send_num`。P、D 两端的 `reqs_to_process` 都应先登记再允许快速 ACK；或两端都必须 latch 早到的完成消息。以 P 实际 tracker removal + scheduler free receipt、D 的 `finished_recving` 入账/abort 收尾 receipt验收，不能以 D 收到 ZeroMQ `ACK` 代替。
5. **状态恢复：**Engram 与 compressor 各自声明保存格式、page group/代际、checkpoint boundary 和 restore worker。Engram prompt-token mirror 不同于 KV；compressor group1 当前排除，只有改变对齐契约时才引入单独 snapshot protocol。

目前没有证明 a3-21 的 CED DRAM shared pool 跨 P/D 可见：实际 NPU offload spec 使用进程内 pinned CPU tensor；实际 source snapshot也没有 P2 shared quota worker。当前 CED overlay manifest 只绑定 scheduler、runner、DSA 和 Mooncake connector；并不包含上述旧 DRAM adapter、token-mirror patch 或 compressor snapshot。尚未验证：P 收到空接收 ACK 时 tracker 已登记、`remote_port_send_num` 全部计数对齐、store hit 发布晚于 D 可见、D abort/event 与 scheduler free 的真实联动、实际跨 TP KV bytes 相等、Engram mirror 在当前 worker 注册、以及 compressor 对齐 boundary 的绝对位置契约。本报告不把源码可用性推成服务端已经接线或运行成功。
