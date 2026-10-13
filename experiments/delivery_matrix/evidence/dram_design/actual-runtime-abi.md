# a3-21 实际容器 native offload ABI（只读源码审查）

日期：2026-10-13。目标是用 a3-21 容器内实际 checkout 的源码核对 native KV offload ABI；这份记录不代表启动服务、压测或验证设备运行。未导入 Torch、未读取权重、未变更容器或服务。

## 来源身份与采集范围

| 项目 | 证据 |
|---|---|
| 容器 | `dsv41-real-stack-tp8-20261010-a321`（a3-21） |
| vLLM HEAD | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vLLM-Ascend HEAD | `e43cf1e9f5d9bead076853aa6bcacb671465de94` |
| 本地源码快照 | `/home/chiro/tmp/dsv41-offload-research/ced-actual-offload-abi` |
| 清单 | 同目录 `SOURCE_MANIFEST`：13 个文件的相对路径、容器路径、字节数、SHA-256 和采集方式 |
| 清单 SHA-256 | `886fddfbde73f51c4aa65db0f30888e14719f7fba2766516bd1f4c4aa2920a3b` |

13 个小源码文件均小于 1 MiB；6 个文件仅在与 `ced-v2-base.DoURUn/connectors` 缓存哈希一致后复用，另外 7 个从容器以只读 tar 流复制。此清单绑定的是本次读取到的文件字节；它不能替代对容器完整 worktree、所有运行时覆盖层或历史服务启动时文件的清点。

## ABI 发现

1. **核心结构有 mapping 元数据，但这本身不是已经工作的去重拷贝 ABI。** 实际 `vllm/v1/kv_offload/base.py:475-515` 定义 `CanonicalPageMapping`，含 canonical/local page size、`runs`、`num_writers`、`writer_index`；`CanonicalKVCacheRef.mapping` 可为 `None`。`vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py:252-315,355-375,387-444` 按 KV spec 与 tensor 几何推导 mapping，并校验本地覆盖及所选 writer 对 canonical page 的区间铺满。这里证明的是可推导的页几何/区间关系，不是跨 rank 实际 KV 内容逐字节相等。

2. **上游兼容注册路径会附 mapping；Ascend 的 layout-adaptation 路径没有附。** 核心 `worker.py:69-74,211-229` 计算 mappings 并在创建 `CanonicalKVCacheRef` 时设置 `mapping=mapping`。实际 Ascend `vllm_ascend/.../native/offloading_connector.py:202-223` 的 `add_view()` 只传 `tensor_idx` 和 `page_size_bytes`；组装引用见 `:300-314`，没有映射字段。该结论仅针对进入此 adapter 分支的布局。

3. **两个被审查的 copy handler 都执行连续页拷贝，不读取 `mapping.runs`。** CPU `vllm/v1/kv_offload/cpu/gpu_worker.py:310-353,390-400` 根据 `tensor_idx` 生成整页源/目标指针，并把每个传输长度设为 `data_ref.page_size_bytes`。Ascend NPU `vllm-ascend/.../native/cpu_npu.py:137-178,198-207` 使用相同的整页长度描述并调用 `swap_blocks_batch`。因此 mapping 字段存在不等于 handler 已实现 gather/scatter 或 canonical page 组装。

4. **writer 门控是全局 aggregate rank 0，不是 per-ref writer 选择。** 核心 `worker.py:54-57,292-344` 初始化一次 `_is_store_writer`：当 `replicated_layout` 为真时只允许 parallel rank 0 提交整笔 store，否则各 rank 都写；非 writer 只 ack。虽然后续 mapping 有 per-ref `writer_index/num_writers`，这条 job 门控未按 ref 消费它们。

5. **实际 Ascend NPU spec 没把 shared mmap 接入该 worker。** `vllm-ascend/.../native/npu.py:58-93` 的 `NPUOffloadingSpec` 继承 CPU spec，并创建 `NPUOffloadingWorker` 时只传 KV caches、chunk size 和 block 数，没有传 `SharedOffloadRegion`。基类 `vllm/v1/kv_offload/cpu/spec.py:90,144-173` 仅在 `_uses_shared_region()` 时打开 mmap，而该默认门是 `current_platform.is_cuda_alike()`（`:144-147`）；NPU spec 本文件没有覆盖它。CPU worker 的非 mmap 分支则分配独立 CPU tensor（`vllm/v1/kv_offload/cpu/gpu_worker.py:472-512`），NPU worker 亦自行分配 pinned CPU tensor（`vllm-ascend/.../native/cpu_npu.py:270-310`）。因此不能把 CPU CUDA-alike 的共享 mmap 能力推定为当前 Ascend NPU 路径能力。

6. **本次采集的真实 NPU 路径中没有 CED P2 配额 pool/worker 接线。** NPU spec 文档写明它“使用 vLLM CPU manager + Ascend transfer worker”（`npu.py:58-82`）；被采集的 CPU manager 委托 cache policy 的通用接口见 `vllm/v1/kv_offload/cpu/manager.py:30-57,182-209`。这 13 个文件未包含 `NPUHostSharedWorker` 或按 group 分区的 P2 配额 pool。此结论限定在清单覆盖的源码，不能扩展成“容器任意路径均不存在此符号”。

## 结论与适用边界

此实际源码快照支持如下表述：核心 offload 类型允许表达 canonical mapping；核心兼容注册逻辑能填入该 metadata。它**不支持**“a3-21 Ascend NPU copy 已按 mapping 去重”或“Ced 当前 worker 使用 P2 group-quota shared host pool”的表述。对审查到的 Ascend layout adapter / NPU handler，应按独立 pinned CPU tensor 与连续页复制理解。若要实现 TP 去重，还需将 mapping 传入适配路径、按 runs gather/scatter、按每 ref writer 选择与组装，并验证跨 rank 实际字节；若要声称共享 quota pool，须另行绑定实现与运行时字节哈希。

## BI OPP 可复现性边界（沿用只读 preflight，不作超范围推断）

既有 `ced_preflight/image-and-opp-readonly.md` 记录：基础镜像缺少 BI OPP；a3-21 容器 writable layer 有约 214 MiB / 617 个文件的 vendor 树；所选 A3 `.run` SHA-256 为 `9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e`，并核验了 `version.info` 和三项关键库的哈希。该证据没有解包 `.run`，也没有取得完整内部 manifest。因此 `.run` 是否还会改 vendor 树之外的 `config.ini`、loader 注册文件或其他 CANN 配置仍未核实；不能据此承诺“只挂载 vendor 目录并设置 `ASCEND_CUSTOM_OPP_PATH`”就足以保证算子可被找到、加载和执行。安装后关键库哈希相同也不是完整安装副作用相同的证明。

如需达到 OPP 复现保证，后续还需针对确切 A3 安装包只读取得其内部文件清单/安装脚本行为，并在目标镜像中核验 loader/config 发现路径及实际算子注册/执行 receipt；在此之前保持该保证为未证实。本轮未执行 `.run`、未复制 OPP、未冷编译或启动 NPU workload。
