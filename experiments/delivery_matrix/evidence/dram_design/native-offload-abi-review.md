# Native offload ABI 与 CED state-group 只读审查

日期：2026-10-13。范围：核对原生 CPU/NPU offload 的 replicated-layout / `CanonicalKVCacheRef` ABI、CED P-only 基线的 KV group 组成，以及 compressor ring 在 P-hit 边界的状态要求。只读代码与既有日志；没有运行设备、改实现或重写历史 receipt。

## 证据来源与版本边界

工作区 `/home/chiro/projects/dsv41` 本身不是 Git checkout，因此没有可报告的 workspace HEAD。分析用到的 checkout / snapshot：

| 来源 | 本机路径与 HEAD | 证据用途与限制 |
|---|---|---|
| CED 插件、实验日志及历史 overlays | `/home/chiro/projects/dsv41/ced-dram-offload`，`8125757eceded47b6d8831ae5cbf60ac7dd3c686`，检查时 worktree clean | `evidence/ced_dram_20261009/v10-p-serve.log` 是带真实 `vllm serve` 输出的历史 serving artifact，但日志本身没有标主机名；旁边 README 的 a3-21 声明只明确指向 probe8/hccl8。`results/ced_dram_dry.log` 是 dry-run 挂载清单，不是运行 receipt。 |
| Ascend fork 代码 | `/home/chiro/projects/dsv41/upstream-v41/vllm-ascend-fork`，`fed4b2b80ee68b88486567a9d70541d11579d33c`，检查时 worktree clean | 用于原生 NPU worker、V4.1 cache / compressor 的源码交叉审查；不能据此宣称它就是 10 月 9 日容器内源码。 |
| 配额实现的较新副本 | `/home/chiro/projects/dsv41/operator-stack-tp8`，HEAD `cf16be3821b91f0f66ce87e46b84b534677a9f4c`，worktree 有大量未跟踪文件 | 只用作配额修复对照；v10 dry-run 挂载的是 `ced-dram-offload/a2/patches/.../p2_pool.py`，不能用较新副本反向证明旧运行采用了修复。 |
| `graph_prep` vLLM 源码树 | `/home/chiro/projects/dsv41/graph_prep/ref/vllm`，无 `.git` / 无可读 HEAD | 静态参考快照。下文凡标为 `graph_prep/ref` 的 vLLM ABI 行锚均是源码参考，不是 a3-21 容器源码身份证明。 |

v10 P serving artifact 报告 vLLM `0.27.1+empty`，run id 为 `ced_dram16k_20261009_v10`；日志没有记录运行主机名、镜像 ID、vLLM/vllm-ascend Git SHA 或完整挂载文件哈希。相邻 dry-run 清单名为 a3-pd-dry、列出期望镜像与挂载路径，但其 max_len 为 1,048,576，而 serve artifact 命令为 32,768；不能用 dry-run 替代实际运行配置，更不能证明容器实际映射到的文件字节。因此，本地 checkout HEAD 只标识本次审查读取的代码，不冒充历史运行时 HEAD。可直接用作 serving 配置证据的是日志中的版本/组回执：`ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1-6,609-614,1198-1207`。

## Offloading ABI 结论

1. `replicated_layout` 是整体布局开关，不是 per-group/per-layer 复制描述。参考快照 `graph_prep/ref/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:113-136,181` 只在单组、精确 `MLAAttentionSpec`、纯 TP-only、单节点 mp 等条件全满足时置位；`graph_prep/ref/vllm/v1/kv_offload/config.py:71-75` 也把 per-ref replication metadata 写为后续计划。v10 serving artifact 的回执是 `replicated_layout=False, num_copies=8`（`ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1199`）。CED 的多组布局不能安全地把这个整体 bool 直接改成 True。

2. `CanonicalKVCacheRef.mapping` 目前是 mapping metadata / 几何证书，不是已经生效的复制 ABI。参考快照中 `CanonicalKVCacheRef.mapping` 定义见 `graph_prep/ref/vllm/v1/kv_offload/base.py:460-515`；`graph_prep/ref/vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py:387-444` 按 spec、shape、stride 推导 mapping，`:355-375` 校验本地 page 覆盖与选定 writer 的 canonical 区间 tiling。这里没有跨 TP 读取/比较 KV 实际内容的字节相等检查，因此“shape/stride 可拼成同一页”不证明真实 KV bytes 相等。

3. copy/load worker 当前没有消费 `ref.mapping`。Ascend 适配层只在 `upstream-v41/vllm-ascend-fork/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py:216-223` 将 mapping 传入 `CanonicalKVCacheRef`；参考 CPU handler `graph_prep/ref/vllm/v1/kv_offload/cpu/gpu_worker.py:283-353,394-399` 仍按每 ref 的 page size 生成连续复制 descriptor；Ascend NPU handler `upstream-v41/vllm-ascend-fork/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:125-187,207-215` 同样按 ref page size 做连续 DMA。两处都没有按 `mapping.runs` gather/scatter canonical bytes。

4. writer 选择仍是 aggregate rank 0。参考 worker `graph_prep/ref/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:54-57,301-343` 以 `_is_store_writer` 门控整笔 store；配置复制时只有 rank 0 写，其余 rank 只确认 job。它没有按 ref 的 `writer_index` / `num_writers` 逐页选 writer。即使 metadata 已有 per-ref writer 信息，这条 handler 路径仍未连接。

5. CED 当前选用的 single-tier NPU spec 也没有共享 mmap 支撑。`ced-dram-offload/experimental/ced/dram/connector.py:30-54` 的 `CEDNPUOffloadingSpec` 继承 `NPUOffloadingSpec`；`upstream-v41/vllm-ascend-fork/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/npu.py:42-59` 说明该 spec 沿用 CPU spec 的 CUDA-alike shared-region gate，故 replicated layout 在此路径 fail closed。tiering spec 的 shared-region override 在同文件 `:62-71`，不是“打开 aggregate bool 即可”的 CED single-tier 接线。

安全适配的最低条件：

- 复制标志/布局下沉为 per-ref/per-layer；对目标 TP ranks 的**实际 KV bytes**执行一致性门，门未通过就维持独立槽位。
- store/load handler 逐 ref 执行 mapping runs 的 gather/scatter，并定义 canonical 页的分片组装、重复字节选择、load 逆映射和异步生命周期。
- writer 按 ref 的复制关系选择；不能让全局 rank 0 写入取代某些 ref 所需的跨 rank 数据组装。
- 把 CED P2 容量配额与 TP 内容去重当成两个独立问题：前者按组做隔离，后者要另行证明实际字节相同。

## P2 配额：保留语义，同时别把旧 v10 路径说成已含后续修复

v10 真实日志报告 group quotas，并出现 `abandoned=False` 的 quota-short 记录；它说明当时有池压力，不是 store 完成/DRAM 命中收据（`ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1204-1207,1565-1580`）。P2 设计的必要语义仍是：按组权重分配容量；共享 backing tensor 的行数按所引用组的最大区间上界计算；相同 tensor 的不同组区间须互不冲突；配额不足只淘汰本组，不能淘汰时放弃本次 store。

版本差异需要特别保留：v10 dry-run 清单指向 `ced-dram-offload/a2/patches/kv8-offload-pool/p2_pool.py`，但 v10 serve artifact 没有该文件的运行时哈希。当前 `ced-dram-offload` HEAD 该路径中的 `prepare_store()` 将 unit 口径的 `deficit` 原样传给 entry 口径的 `policy.evict()`（`:346-367`），并且 `_allocate_blocks` 有 assert 路径；较新的对照 `operator-stack-tp8/a2/patches/kv8-offload-pool/p2_pool.py:405-472` 才显式按 `ceil(deficit / bpc_g)` 折算并捕获分配短缺、回滚部分分配。不要把后者的修复归给 v10 历史执行；也不要把当前本地旧路径未经哈希核对就断言为容器实际字节。共享张量行数/冲突计算的参考位置：`ced-dram-offload/a2/patches/kv8-offload-pool/p2_pool.py:422-461`。

## CED 12/13 groups 与 compressor state

v10 P-only 的运行时组清单是 12 组：group 0 `DeepseekV41FullSpec`（block 128、8 refs）；group 1 `DeepseekV41CompressorStateSpec`（block 32、3 refs、prefix-cacheable=false）；groups 2–11 是各含 4 refs 的 SWA 组（block 128）。这是实际 v10 日志 `ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1201` 的描述性 spec 清单，不等同于 vLLM 公共 `KVCacheSpec` 的 Python 类型名。

P 的 DRAM 参与清单是 group `(0,2,3,4,5,6)`；group 1 和上半部 SWA `7–11` 被排除，日志见 `ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1206-1207`。按 V4.1 layer-plan fixture/config，layer 0–19 包含 ratio-2 source 层 `2,8,14`，它们在 P forward 中写各自 compressor ring，但这些 ring 不进入 P DRAM（来源计划见 `upstream-v41/vllm-ascend-fork/vllm_ascend/models/deepseek_v41/model.py:489-529`；不是 v10 日志里的完整模型 config dump）。group 1 的三个成员是 `model.layers.{2,8,14}.self_attn.compressor.state_cache`：较早 v9 rank0 metadata/mock artifact 中 group 1 的三个组件名和 `prefix_cacheable:false` 可见于 `ced-dram-offload/evidence/ced_dram_20261009/v9-rank0-meta.json`；v10 只确认组数/类型标签/块大小，不在同一日志内重复列出层名。

源码交叉核对：生产测试 fixture 配置 `compress_ratios=[0,0]+[2]*18+[1]*20`、`kv_source_layer_ids=[2,8,14,20]`（`upstream-v41/vllm-ascend-fork/tests/deepseek_v41_utils.py:276-277`；另见 `upstream-v41/vllm-ascend-fork/tests/ut/models/test_deepseek_v41_cache.py:74-75`）。ratio 2 的 source 才构造 state cache：`upstream-v41/vllm-ascend-fork/vllm_ascend/models/deepseek_v41/compressor.py:22-45`；cache grouping 将所有 compressor states 放入独立 CircularBuffer 组，见 `upstream-v41/vllm-ascend-fork/vllm_ascend/models/deepseek_v41/cache_config.py:47-55,78-94`。当前 fork 测试明确 `12` 组为 `[8,3]+[4]*10`，state spec 的 `block_size=32`，生产布局也断言 12 组、state 成员只落在前三个 full-KV slots：`upstream-v41/vllm-ascend-fork/tests/ut/models/test_deepseek_v41_cache.py:156-221`。这些是当前 fork 的源码测试，不替代 v10 runtime 的完整 config dump。

state 不是 MambaSpec，也不是任意可丢弃的“recurrent tensor”：v10 把它识别为 `DeepseekV41CompressorStateSpec`；历史 patch 定义将该类作为 Ascend circular-buffer spec 子类，并要求 32-row packed FP32/FP16 KV-score ring（`ced-dram-offload/a2/patches/kv8-int8-pkg/vllm_ascend/core/deepseek_v41.py:134-144`）。对应 cache module 文档为每个 active request 分配一个 private circular page（`ced-dram-offload/a2/patches/kv8-int8-pkg/vllm_ascend/models/deepseek_v41/compressor.py:38-48`）；标准 vLLM/当前 fork 对应类型为 `CircularBufferSpec`。它为每个 ratio-2 source request 保存投影 KV 与 gate score；不是 Mamba SSM state。P layer 0–19 中确实有这三个 state owner（2、8、14），但 P DRAM group list 明确将 group 1 排除。

### state 的读写与 P-hit checkpoint 边界

- 写入调用链：ratio-2 source 在 `ced-dram-offload/experimental/ced/dsa_v41.py:574-618` 生成 `wkv`/`wgate` 投影后调用 `compressor.pool_projected()`；历史 patch 的 `ced-dram-offload/a2/patches/kv8-int8-pkg/vllm_ascend/models/deepseek_v41/compressor.py:111-127` 把 `state_cache.kv_cache[0].squeeze(-2)`、C2 metadata 交给 `compressor_from_projected()`。
- 实际 ring 操作：补丁算子 `ced-dram-offload/a2/patches/kv8-int8-pkg/vllm_ascend/ops/triton/compressor/compressor_triton.py:88-197` 在 ratio-2 组的 residual 分支读先前 KV/score 行；`:335-361` 的 `_cache_update_kernel` 将本段最后至多 32 个 token 的投影 KV/score 写到 `slot = token_pos % 32`；`:660-743` 的 `compressor_from_projected()` 按顺序 launch K2 pooling 再 launch K3 更新。这些源代码是本地 patch tree 的静态证据，未声称与容器逐字节一致。
- 当前 DRAM hit boundary 的运行时契约要求 `boundary < request.num_tokens` 且 `boundary % _ced_alignment == 0`，见 `ced-dram-offload/experimental/ced/dram/connector.py:107-125`。v10 日志实测 `_ced_alignment=1024`（`ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:1202,1207`），是 ratio 2 与 32-row ring 的整倍数。算子 residual 只在 `start_pos` 落在 ratio-2 pair 中间时读取前一 token；对齐到 1024 的续算起点是偶数，按当前算子逻辑 residual 为 0，因此不需要跨 checkpoint 取回 group 1 的先前投影行。
- 这是**源码推导，不是恢复成功 receipt**。还需对实际 runtime 版本验证绝对 `positions/start_pos` 与 boundary 一致，以及 group 1 新分配的 circular page 在 aligned resume 时没有隐藏的先读路径。若以后允许奇数 token checkpoint，则最小恢复对象是每个 ratio-2 source 的前一 token 两条 ring-dtype 向量（KV 与 score；当前 patch 默认 FP32、可选 FP16），写回其 circular page 的 `(boundary-1) % 32` 行；仅有 token IDs 不足以还原这些投影，除非重算到 source layer。

现有 P-hit 处理恢复的是另一种状态：`ced-dram-offload/experimental/ced/dram/connector.py` 的 `CEDOffloadingConnector.update_state_after_alloc()` 用 group 2 的前一完整 SWA 页及 prompt IDs 生成 Engram history seed（`:127-146`），再由 `handle_preemptions()` 调用 `restore_history_pages()`（`:187-192`）。没有同类代码恢复 compressor-state group。对当前 1024 对齐契约，state ring 可能无需跨边界恢复，但必须把 Engram history mirror 和 compressor ring 分开表述。

12→13 的差异：V4.1 cache builder 只有存在 draft-only SWA names 时才追加 draft group（`upstream-v41/vllm-ascend-fork/vllm_ascend/models/deepseek_v41/cache_config.py:109-121`）；P-only v10 日志明确 `draft groups=()`、`total_groups=12`（`ced-dram-offload/evidence/ced_dram_20261009/v10-p-serve.log:612`）。因此 13 组是带 DSpark/Draft-SWA 的本地/D 侧形态，不是该 v10 P receipt 的形态。CED Mooncake connector 也把 D 本地组数写成 `12 + len(self.ced_draft_swa_groups)` 的契约，见 `ced-dram-offload/experimental/ced/mooncake_hybrid_connector.py:2221-2247`；这些 draft-only 页不由 P 提供。

## MultiConnector 顺序提示

静态 vLLM 参考 `graph_prep/ref/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:385-416` 会依配置顺序查询所有 child、选第一个 positive hit；未选中的 child 仍在 `update_state_after_alloc` 收到 `num_external_tokens=0`。child 的 worker load/wait/save 也逐个调用（`:293-313`）。因此 D 侧要让 DRAM 命中优先，配置顺序应在 Mooncake 前；但 Mooncake zero-external 分支不等于“完全无副作用”：CED fork `ced-dram-offload/experimental/ced/mooncake_hybrid_connector.py:1596-1640,1648-1659` 会构建空组 receive metadata并清除 `do_remote_prefill`，其 worker 对零本地块早退且发送清理通知（`:779-795`）。这不搬 KV 数据，但会触碰请求参数和 remote cleanup，需由组合路径实测确认。

## 审查结论

当前 native offload ABI 可安全描述“metadata 具备 canonical mapping 能力”，不能描述“CED TP8 已按 mapping 去重”。v10 的运行时收据反而明确显示 `replicated_layout=False`、8 份副本、P2 group 1/state 排除；其 quota-short 日志不是容量/命中/续算达标证据。后续若只允许 1024-token 对齐的 P-hit boundary，ratio-2 ring 的跨边界 residual 从静态机制看可不恢复；仍需对被固定的实际 runtime source 做端到端续算验证，并维持禁止奇数 boundary 的 fail-closed 条件。
