# CED 单侧 DRAM 模式范围更正（addendum）

日期：2026-10-13。此 addendum 更正 [`ced-single-side-dram-interface-checklist.md`](ced-single-side-dram-interface-checklist.md) 对 P 写入、D 读取同一 DRAM KV pool 的讨论范围；原报告保留不改，其 SHA-256 为 `f14196db626dafec6b28c820926f8053dfdcc6482ff99adb0ed879625e1d3dac`。旧报告中 P→D 共用 DRAM 的内容只作为另一种潜在架构分析，不代表本用户所需的单侧部署。

## 单侧模式的真实语义

`off | prefill | decode` 选择的是**哪一侧安装 DRAM KV store/load handler，并创建一个仅在本侧 TP8 ranks 间使用的唯一 KV pool**；非选中侧不得创建、写入或查询这个 DRAM KV pool，仍走原有 Mooncake P/D 协议。P/D 两侧共用的 Engram mmap 是独立状态平面：它不使 KV pool 跨 P/D 共享，也不能作为 DRAM KV 文件/容量的来源证明。

| 模式 | P（prefill） | D（decode） | KV DRAM pool 边界 |
|---|---|---|---|
| `off` | 不安装 DRAM KV handler；照常处理 P 与 Mooncake send | 不安装 DRAM KV handler；照常 Mooncake receive | 不创建 DRAM KV pool。若平台要求共享 Engram mmap，仍由它自己的配置管理 |
| `prefill` | 安装 DRAM handler，在 P 本侧 TP8 ranks 间创建唯一 pool；本地 DRAM hit/load、store 均在此 pool 内 | 不安装/访问 P 的 DRAM pool；保持原 Mooncake P→D 传输 | 只属于 P；不得让 D 的 DRAM connector读取它 |
| `decode` | 不安装/访问 D 的 DRAM pool；按原 Mooncake producer/延迟释放协议发送 | 安装 DRAM handler，在 D 本侧 TP8 ranks 间创建唯一 pool；D-only hit/load/store 仅访问此 pool | 只属于 D；不得让 P 的 DRAM connector写入它 |

这里“唯一池”指 selected side 上本地 TP8 ranks 共用的一个容量/配额/页分配域，不是每 rank 各自独立的 NPUOffloadingSpec pinned tensor。归档实际 NPU ABI (`SOURCE_MANIFEST` SHA `886fddfbde73f51c4aa65db0f30888e14719f7fba2766516bd1f4c4aa2920a3b`) 显示 `NPUOffloadingWorker` 为每个 worker 分配自己的 pinned CPU tensor，NPU spec 没有把 `SharedOffloadRegion` 接入；因此现有 ABI 本身还不足以满足跨本侧 TP8 唯一池契约，需独立实现/绑定该 pool。

## P-only DRAM hit：恢复 P 本地状态，D 仍 Mooncake

P-only 模式的 hit 是 P 自己对其 DRAM 池的命中；它不向 D 提供 DRAM read。可沿用的恢复接口是候选 P DRAM connector 中的 `get_num_new_matched_tokens()` → `update_state_after_alloc()` → `build_connector_meta()` → worker `handle_preemptions()`：在分配 P 本地块之后，用**P 本地**的 group-2 block IDs 和 `request.prompt_token_ids` 还原需要的 Engram token mirror 页（原型 `ced-dram-offload/experimental/ced/dram/connector.py:107-146,155-167,187-193`）。完整 token page 的 restore API 是 `restore_history_pages(page_to_raw_token_ids, block_size)`，而 worker-local Engram history 必须在同一 worker 中注册（原型 `engram_history.py:7-21`、`patches/files/engram_hash.py:397-425`）。这些是 `ced-dram-offload` HEAD `8125757eceded47b6d8831ae5cbf60ac7dd3c686` 的原型接口；当前 `ced_runtime_20261013_v1/manifest.json` 声明 `best_model_and_engram_unchanged=true`，不能宣称已绑定到现行 P runtime。

P continuation 仍应遵守 boundary 对齐并保留一个 P-local compute tail（原型 `ced-dram-offload/experimental/ced/dram/connector.py:107-125`）。该原型把非 prefix-cacheable compressor state group 排除 (`:71-78`)，所以不从 KV pool 复制 compressor ring。对旧 compressor 实现与 1024 对齐边界的“首段 residual 为零”只是静态推导；在绝对 positions/start_pos 和新 state page 首读路径被验证前，不能移除 fail-closed 对齐门。P 的 DRAM store event 和 P 的 Mooncake send completion 是两条独立释放 fence：P-only 模式仍把 KV 发给 D，必须同时等本地 DRAM store barrier 与正常 Mooncake P→D done/ACK 后再复用 P source blocks。

## D-only DRAM hit：D 本地恢复，P 仍 Mooncake

D-only 模式的 KV 数据只能来自 D 自己的 TP8 DRAM pool；不得把 P-only pool 当作远端来源。若 D 的本地 DRAM miss，则完全照常走 Mooncake pull。若 D 本地 DRAM hit，数据读取改为 D-local DRAM，但**仍须完成 P 的 Mooncake delayed-free 控制生命周期**：

1. D 显式选择 `selected_source=decode_dram`，停止该 request 后续 Mooncake lookup，避免错误发起 P→D KV pull；D 命中的 group/page geometry 与其本地 TP8 pool 绑定。
2. 如果该请求已携带有效 `do_remote_prefill` / remote request / endpoint 参数，D 不能删除这些参数后直接结束 Mooncake child。需保留每组空 `local_block_ids` 的 no-op receive，或执行显式、幂等、计数匹配的 `DONE_RECVING`/cancel ACK，把 P 的远端 delayed-free 栅栏关闭。
3. 当前交付 Mooncake overlay 源码有可复用的 no-op 形态：未选中的 Mooncake child 接受 `num_external_tokens=0`，创建空组接收 metadata；transfer 对零本地块早退，但在 `finally` 仍发送 `DONE_RECVING_MSG`（`operator-stack-tp8/experimental/ced/mooncake_hybrid_connector.py:1594-1638,777-794,730-758,1162-1180`）。这证明源码设计提供了“不读 KV、发清理消息”的接口，不证明当前 P/D 实例已正确更新 P done counter。
4. ACK 应绑定 P 的 `remote_request_id`、所有 `remote_port_send_num` / PP rank 计数和两侧 task registration。当前 overlay 的 P tracker 可能丢弃在 `reqs_to_process` 注册前到达的 done；D worker 也先 enqueue receive、后登记本地 process tracker（同文件 `:287-320,2318-2335`）。应先注册/有 early-done latch，并分别验证 P 的实际 source-block free 与 D 的 local receiver bookkeeping；ZeroMQ ACK 不是两侧 page-freed receipt。

D-only 的 DRAM load 仍是异步 NPU transfer：abort 不取消已提交的 DMA。目标块要等 D-local offload worker event 完成并返回 `finished_recving` 后再释放；这与给 P 的 Mooncake done ACK 是两个不同的事件。P side 不创建/查询 D 的 DRAM pool；它只按 Mooncake completion 释放自己的发送页。

Engram mmap 是独立共享状态：D-only 模式的 D 本地 DRAM hit 仍需确认该请求在 D 的 Engram history/token mirror 与当前 group-2 physical page ID 一致；需要 token-page seed 时，应在 D 的本地 allocation/restore hook 中从请求 token IDs 重建该 mirror，而不是依赖 P DRAM 读或跨侧 KV pool。此 D-only Engram restore hook 尚未被当前 overlay/runtime receipt 证实。

## 仍未确认的事项

- 当前实际 P/D 容器是否已有本侧 TP8 唯一 DRAM pool、P 与 D 模式是否严格隔离：无运行时 receipt。
- P-only 原型 Engram restore 与 current `ced_runtime_20261013_v1` 模型/Engram 是否接线：manifest 说 Engram 未改，不能推定已生效。
- D-only DRAM hit 的 empty-recv ACK 是否能覆盖全部 P ports、P tracker 是否入账且 scheduler 实际释放：无 live done-count/free receipt。
- P/D 共用 Engram mmap 与 single-side DRAM pool 的 backing、权限、生命周期必须分开核验；共享前者不代表共享后者。

本 addendum 未改服务或实现，也未修改原接口清单；所有模式讨论均为对用户语义的范围澄清与静态接线建议，不是运行成功/容量/性能声明。
