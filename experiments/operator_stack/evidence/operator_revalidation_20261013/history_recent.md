# a3-21 算子优化历史复核（2026-10-10 至 13）

日期：2026-10-13。只读审计 `experiments/operator_stack/reports` v1–v56 的状态摘要、相应 `evidence/formal_a321` 收据与本地冻结源码。未 SSH、未连接 NPU/服务、未发请求、未更改或重新捕获任何作业。临时工作路径为 `/home/chiro/tmp/dsv41-operator-history-recent`；本报告按要求用 `apply_patch` 新建。

## 结论先行

- **当前最优已通过服务不是 tp8meta/tp8mask。** 冻结 `best_service_20261013/env_manifest.json` 记录 `STACK_TP8_ARM=tp8base`、`STACK_METADATA_MANY_SLOTS_ENABLED=0`、`STACK_MOE_MASK_ENABLED=0`；同时 `V41_MOE_MASK_RANGE=1`、`VLLM_BATCH_INVARIANT=1`、DSpark graph 和 async 调度开启。当前正式窗口是 8 个 HTTP 请求、2K/256、(26.5524 ms/step, A=2.7309, 102.8504 tok/s)，747 个实际 verifier rounds、GSM8K 100/100、Vision 23/23；20 ms 仍未达到。
- `V41_MOE_MASK_RANGE=1` 是**旧 token-dispatcher 连续 expert-range 比较**（避免 `Index`/`IndexCheck`），不是新 `tp8mask` 概率掩码融合；同理旧 `V41_SLOT_MAP_FUSED` 是 block-table slot-map kernel 融合，不等于新 `STACK_METADATA_MANY_SLOTS_ENABLED` 的 host metadata batching。当前环境里旧 MoE range 开、新 mask/metadata arms 关。
- 历史上最像值得补测的是：①新 `tp8meta` slot batching 与 DSpark 的交互；②新 exact `tp8mask` 与 DSpark+BI 的交互；③`tp8hostmeta` 在当前 async best 下是否还暴露于关键路径。**现成 TP8 多臂 runner 是 A=1、无 DSpark；slot/mask 分别还有 `n==1` / `(1,6)` 几何守卫，而 profiler 已见 DSpark target 6 行、top-k `(6,6)`；full43 diagnostic 冻结为 base arm，也不是可现场切换的 DSpark arm。** 这两项需先泛化候选几何并通过六行消费者门，再接入当前 DSpark worker、fresh load/capture；不得只改 env 后沿用旧图。
- Prefix/W4A8、QKV merge/QKV-tail、WOA Cube 和 quant-GEMV 已有正式负收益或实际消费者精度拒绝；没有源码/几何实质变化时不建议重跑。Indexer 融合曾通过消费者门，但在 slot stack 上的全模型增量仅约 0.025 ms、7/12 配对更快，不应列为主候选。

## 如何判读旧结果

| 收据类型 | 本审计采用的口径 | 不能替代什么 |
|---|---|---|
| `bench_real_tp8_model.py` 多臂结果 | 同一 TP8 模型实例/worker 的内部 `engine.step` 或 token-arrival 配对；A=1、无 speculative decoding；不是 HTTP。结果文件有 `same_model_instance=true`、`same_processes_per_rank=true`。 | 不能直接代表 DSpark rounds，也不能从 microbench 事件时间推算 HTTP 延迟。 |
| `formal_dspark_native_graph_aa_*` | server-side AA/terminal device-fence + rank join 的 DSpark 等价/round 证据；适合核语义，不是 HTTP 客户端报价。 | 不证明 2K/256 长请求收益或质量门。 |
| `formal_dspark_native_graph_api_*` / API acceptance | 真实 API/HTTP；v11 BI 收据 scope 是 terminal-prefill bookkeeping 到完整私有 HTTP response read，按实际 verifier rounds 计算 A。 | 不应与内部 `engine.step` 数字直接相减；不同服务窗口也不能当同进程单变量 A/B。 |
| 单算子/图事件探针 | 设备事件同步后的 kernel/调用计时，另记 CPU submission headroom。 | 不等于图内有效端到端节省；若真实消费者不通过，性能作废。 |

全模型/HTTP decode 结果一律保留三元组 `(ms/step, A, tok/s)`，并附 rounds/tokens；A=1 的旧结果没有包含 DSpark 草稿及拒绝采样成本。

## 候选逐项复核

| 优化 | 冻结实现、开关与测试几何 | 正确性门与比较类型 | 收益与当前判定 |
|---|---|---|---|
| **新 slot batching (`tp8meta`)** | `tp8_patches.py` 的 `tp8meta` arm 把每步 12 组 raw-slot → `(physical block, offset)` 准备从 12 次调用合为一次；`STACK_METADATA_MANY_SLOTS_ENABLED=1` 才传 `tp8_slot_batches.REGISTRY.prepare`。正式 checkpoint、40层/384专家 top6、TP8、真实 W4A8/Engram；无 DSpark、A=1、2K/48。**实现守卫 `SlotBatches.record()` 要求 `n==1` 且 `num_reqs==1`；当前 DSpark target 图为 6 行，不能视作几何兼容。** | Probe v4 以 CPU floor/remainder 参考覆盖 INT32/INT64 各 72 例、尾部保护与 6 次变化输入图重放；audit v2/full stack audit 检查完整路由、Top5/logprob 和 24 个 rank×12 group 消费者坐标。 | v18 同实例四臂内部 `engine.step`：base 30.1836、core 26.8409、meta 25.7242、metastack 25.6942 ms；slot 候选 11/12 快于 core，配对加速中位 1.0313。v19 的 `tp8metastack` 另有 HTTP 客户端 26.2925/A=1/38.0336 tok/s + GSM100/Vision23，但它是 slot+Indexer 组合，不是 slot 单变量 HTTP 收益。**新 slot batching 在当前 DSpark best 没启用，且无 DSpark 证据；历史 A=1 收益不得外推到 6 行 target。先扩展并通过新几何/消费者门，再接入 DSpark worker 与重捕获。** 证据：v14–v19；`results/formal_slots_batch_probe_v4/`、`formal_slots_batch_audit_v2/`、`formal_slots_indexer_perf_v1/`；形状见 `evidence/formal_a321/fusion_scan_20261013/fusion_scan_rank0.json`。 |
| **旧 slot-map fuse (`V41_SLOT_MAP_FUSED`)** | `patches/files/block_table.py` 的 12×`_compute_slot_mapping_kernel` 合并，独立于上行 `tp8meta`。verify 模式检查 8×256 token 输出逐位一致。 | v46 verify 门通过；v49 在 async DSpark API 窗口用同 workload/token 流复测。 | 设备 `[bneck] hp` 41.006→39.797 ms，但 HTTP async v4→v5 32.970521→32.894996 ms，仅 −0.0755 ms（−0.23%）、token 8/8 相同，判噪声、不采纳。v54 又确认 BI+旧 slot-map+IDS64 捆绑与 BI 基线差为精确零。**此旧开关已实测，不列入重跑队列；不能与新 `tp8meta` 混名。** 证据：v46、v47、v49 §8、v54；`dspark_native_graph_api_v4_async/` 与 `...v5_async_slotfused/`。
| **Indexer 融合** | `indexer_patches` 的 fused arm 与 12-group slot batching 一起在 `tp8metastack` 中安装；真实层、KV/index/cache 生产布局，TP8/W4A8。 | 独立 page0/page1 整数/reference 和实际 rank 消费者逐位检查通过；v17 三方案 audit 的路由/token/Top5/logprob delta=0。 | v18 同实例 A=1 内部配对：slot 25.7242，slot+Indexer 25.6942 ms；增量配对中位约 0.0251 ms，只有 7/12 更快。**已测试且边际很小，不优先重测**；当前 best 的 `V41_QLI_NO_CANDIDATE=1` 是已有 baseline 开关，不代表新 `tp8metastack` fused arm 已装入 DSpark worker。证据：v17/v18；`results/formal_slots_indexer_perf_v1/`、`banks.json`、`comparisons.json`。
| **Hostmeta 静态几何缓存 (`tp8hostmeta`)** | 初始化缓存 cache-kind、compress-ratio、storage-block、slot/SMLA/QLI 共享键；动态 position/query/seq/slot/skip 仍逐步读取。真实正式 W4A8/TP8/40层，A=1、无 DSpark。 | v23 六完整配对的路由/token/Top5/logprob delta=0；24 rank 收据的 288 个 group 坐标精确，spec-build 覆盖非0。 | v23 同实例内部 12 对 metastack 25.7192 vs hostmeta 25.6363 ms，9/12 更快，配对中位约 0.1004 ms；v24 另有 HTTP client 26.1330/A=1/38.2657 + GSM100/Vision23，但没有 HTTP 单变量 paired baseline。**低优先；当前 async DSpark 可能已隐藏 host准备，只有 profiler/host timeline 显示仍暴露且完成 DSpark 接线后才值得测。** 证据：v23/v24；`results/formal_hostmeta_audit_v1/`、`formal_hostmeta_perf_v1/` 与 v24 client receipt。
| **W4A8 prefix (`tp8prefixup`, `tp8prefixuproute`)** | 仅 GMM1 prefix / routing-prefix 变体；保留原生 GMM2；真实层/正式权重/TP8。曾试过 GMM1+GMM2 prefix，出现 MTE 非法 GM，随后限制为只改 GMM1。 | 8rank×40层有效行的 INT8/scale/output 与原生逐位；3840 GMM 消费、576 cache 坐标精确，完整三组路由/logprob 通过。 | v22 同实例内部 A=1 12 配对：`prefixup` 26.9695 vs metastack 25.7289 ms、12/12 更慢；`prefixuproute` 25.8527，3/12 更快且配对中位仍慢约 0.1833 ms。**正式精度通过不等于优化成功；无新 kernel/几何变化不重测。** 证据：v20–v22；`formal_prefix_up_perf_v1/`。
| **WOA Cube** | BN64/BK256 和 BN128/BK512 两个正式权重 `wo_a` GEMV 候选；层0/20权重按真实 TP8 切片，BF16，范围96个独立消费者输入。 | 独立 96/96 在 ≤1 BF16 ULP 内（80例逐位）；但完整真实模型 `language_model.model.layers.0.self_attn.wo_a` 首个消费者最大 184 ULP，严格门拒绝。 | 单算子配对中位约 1.13× 快；真实消费者失败后性能控制器明确 `do not run timing`，没有整网收益、客户端或部署。**除非数值算法已修，不重试。** 证据：v29；`formal_woa_gemv_probe_v1/` 与 `formal_woa_model_audit_v1/rejection.json`。
| **MoE mask：旧 range vs 新 exact mask** | 旧 `V41_MOE_MASK_RANGE=1` 在当前 best 打开，替代 `expert_map[topk_ids]` 的 Index/IndexCheck 为范围 compare；新 `tp8mask` 是独立 `STACK_MOE_MASK_ENABLED=1` + `STACK_TP8_ARM=tp8mask`，将概率 `masked_fill` 替换为精确 mask/cast custom op。**当前 exact-mask 守卫只接受 `ids.shape == weights.shape == (1,6)`；DSpark target 的 top-k ids/weights 实测为 `(6,6)`。** | 新 mask 独立 96例/192个 FP32/BF16 输出比较逐位；正式 audit 每 rank 40 个不同层消费者逐位、960 检查，并通过路由/token/Top5/logprob。 | 新 tp8mask 同实例内部 token-arrival 12对 metastack 19.3729 vs 19.2287 ms、12/12更快，增益0.1498ms；v32另有 **HTTP** 8×2K/256 客户端 `(19.289956,A=1,51.840451)`、GSM100/Vision23。**历史只覆盖A=1/无DSpark，不能外推到 DSpark 六行 `(6,6)` 几何。** 当前 best `V41_MOE_MASK_RANGE=1` 但 `STACK_MOE_MASK_ENABLED=0`、`STACK_TP8_ARM=tp8base`；新 mask 未启用，若再测须先泛化精确 shape/消费者守卫、在当前 worker 接线并重捕获，再做 DSpark+BI 单变量验证。旧 range-mask 的 −0.51ms 数据来自 2026-09 dummy/profile，与这次新 exact-mask 的 +0.15ms 不能相加。证据：v30–v32、v53；`formal_moe_mask_probe_v2/`、`formal_moe_mask_audit_v1/`、`formal_moe_mask_perf_v1/`、v32 HTTP client；DSpark target 形状见 `evidence/formal_a321/fusion_scan_20261013/fusion_scan_rank0.json`。
| **Quant GEMV / INT8投影** | 两个正式层0/20的 q_a/KV 复制权重与 q_b TP8切片，试 ND/NZ、tiling、scale 顺序；计划 120 cases/candidate。 | v3 到 17/20权重、102/120例时所测结果在原≤1ULP门内，但没有完成完整独立精度，更未到真实模型消费者。 | 同芯片配对所有已测方向更慢：q_a speedup 0.764、KV 0.654、q_b 0.707；ND/NZ适配额外毫秒级，候选被 `rejected_for_timing.json` 拒绝，无整网或HTTP结果。**不重测既有tiling，除非算法/ABI变更。** 证据：v32/v33；`formal_quant_gemv_probe_v3/`、`rejected_for_timing.json`。
| **QKV merge / early-KV-tail** | 两层/真实 W4A8 NZ 权重，Q_a+KV INT8权重拼接、single M1原生量化 GEMM；tail版只调整KV norm/RoPE/SWA提交时序，非M1回退。 | v35真实V4.1调用链改对后，8rank×40唯一层的Q_a/KV消费者逐位，完整路由/token/Top5/logprob通过；v36又通过patch/source SHA守卫和rank审计。 | v35 same-instance内部 token-arrival A=1：QKV 19.5554 vs mask19.3245，0/12更快；v36 early-tail19.3072 vs mask19.2087，0/12更快。独立投影约7–9µs收益没有转为整网收益；两版部署均 skipped。**没有源差异时不重测。** 证据：v34–v36；`formal_qkv_merge_perf_v2/`、`formal_qkv_tail_perf_v1/`。
| **DSpark draft graph / async / BI** | 现 best 的 full40+MTP3、K=5、draft graph、async、BI是多个互补优化/质量模式，不是单一算子臂。 | v46 AA server-fenced 3对 token/routing/Top5/logprob全同；v49 HTTP 8×2K/256并按实际 verifier rounds计A、token流与sync参考逐位；v53 GSM100/Vision23通过。 | v46 graph首窗HTTP46.487/A2.965/63.783 vs eager旧HTTP57.582（跨窗口）；v49 async HTTP32.971/A2.939/89.155 vs sync46.487（不同job，token 8/8一致）；当前 v11_bi HTTP **26.552/A2.731/102.850**, 747 rounds/2040 tokens、clean8/8、质量全过。BI vs v9/v10是跨快照/不同A，不能称算子增益；但它已在现 best。**draft graph/async/BI无需重做历史确认，20ms缺口仍在。** 证据：v46/v48/v49/v52/v53/v56；`dspark_native_graph_aa_v16_async/`、`dspark_native_graph_api_v4_async/`、`dspark_native_graph_api_v11_bi/` 的 raw HTTP/acceptance、`quality_gate_pass_bi_v14/`。

## 当前 best、旧配置与 full43 的 arm 边界

`best_service_20261013/env_manifest.json` 和 `BEST_SERVICE.md` 给出的真实 DSpark+BI 配方是 base arm、slot/mask candidate 两项均关、旧 V41 range mask 开；它的当前 HTTP 三元组在上表。不能从 `V41_MOE_MASK_RANGE=1` 推断 `STACK_MOE_MASK_ENABLED=1`：`token_dispatcher_moemask.py` 的旧 range比较与 `formal_moe_mask_patches.py` 的新概率融合处理的是不同调用点。

历史 `tp8meta/tp8mask/tp8hostmeta` arm 是 `launch_real_tp8_job.py` + `RealTP8StackWorker`/`tp8_patches` 的 A=1多图/多bank实验；`tp8_patches.create()` 会重置并重新 `capture_model()`。对应 `ServingRealTP8StackWorker` 明确拒绝 `speculative_config is not None`，所以这个 runner不支持 DSpark。

full43 v26 的真实 DSpark诊断已观察 8rank×37个完整43输出 generation，但 freeze 把 `STACK_TP8_ARM=tp8base`、`STACK_METADATA_MANY_SLOTS_ENABLED=0`、`STACK_MOE_MASK_ENABLED=0`、`DRAFT_GRAPH=0` 固定；报告明确所有formal/performance admission仍false。`prepare_full43_audit_snapshot_v17.py` 按 freeze/hash构造 source snapshot，`launch_full43_audit_v2.py` 只接 mode/snapshot，不提供 arm 参数；它也不是当前 `DRAFT_GRAPH=1` 的 best HTTP服务。故**不能在已捕获的 full43/DSpark 图上翻环境变量来切换新arm**。若要做 DSpark+BI 下的 slot batching 或 exact mask：需要主代理新接入受控 arm 的 DSpark worker/contract，生成新源码 freeze/snapshot，fresh load并重捕获，再走同实例单变量 A/B + full consumers + HTTP三元组/质量门。没有这些前置，`full43` pass只是base-arm诊断，不给候选背书。

### 额外几何门：A=1 候选不等于 DSpark 六行候选

离线 target profiler 证据 `evidence/formal_a321/fusion_scan_20261013/fusion_scan_rank0.json` 记录 `MoeGatingTopKHash` 输入 token 行数为 6、输出 ids/weights 均为 `(6,6)`；`MoeInitRoutingV3` 输入 `(6,5120)` 与 `(6,6)`，扩展出 `(36,5120)`。与此对照：

- `tp8_slot_batches.py::SlotBatches.record()` 仅当 `n==1` 且 `num_reqs==1` 才接受候选，否则走原生 fallback；历史正式实验也是 A=1。
- `formal_moe_mask_patches.py::dispatch_mask()` 仅当 ids/weights 同为 `(1,6)`、dtype/连续性满足守卫时调用 exact custom op，否则也走原生路径；DSpark 捕获到的目标几何为 `(6,6)`。

因此旧多臂数据可以证明对应 A=1 形状上正确/有收益，却**不能证明六行 DSpark 下会命中、正确或有同样收益**。候选排期必须在 DSpark worker 集成之外，先把形状和分组/行映射守卫扩展到 target 的真实输入；之后做 fresh load/capture、六行 full-consumer 数值审计，再测 HTTP/质量。单纯设置 `STACK_METADATA_MANY_SLOTS_ENABLED=1` 或 `STACK_MOE_MASK_ENABLED=1` 不会绕过这些 fail-closed shape guards。

## 建议补测次序

1. **首选：新 `tp8meta` slot batching × 当前 DSpark+BI。** 历史 A=1内部候选增量最大（约1.12ms），当前 best明确关闭；但 `record()` 的 `n==1 && num_reqs==1` 守卫与 DSpark 六行图不兼容。先扩展并审计真实几何，再接入 DSpark worker、fresh load/capture。旧 `V41_SLOT_MAP_FUSED` 已在 v49长请求HTTP窗口复测为噪声，不重跑那个旧flag。
2. **次选：exact `tp8mask` × 当前 DSpark+BI。** 历史正式 A=1 HTTP质量与内部配对都通过，当前只保留旧 range mask；exact custom op 的 `(1,6)` 守卫不覆盖 DSpark 的 `(6,6)`。先泛化并过六行 full-consumer/token/logprob门，再比较完整 HTTP verifier rounds；不沿用 A=1收益。
3. **条件项：hostmeta。** 先用当前 async best的时间线确认 metadata host准备仍在露出关键路径；如果已经被 async藏住，0.10ms旧收益不值得重占卡。若需测，独立于 slot batching和mask，一次只变一项。

暂不排期：Indexer（slot stack增量0.025ms/7 of12）、W4A8 prefix（正式配对慢）、WOA Cube（真实消费者精度拒绝）、quant GEMV（已测配置均慢/探针不完整）、QKV/early tail（0/12更快）；除非算法、源码或几何有实质变化。

## 主要原件路径

- 当前 best 环境/三元组：`evidence/formal_a321/best_service_20261013/{BEST_SERVICE.md,env_manifest.json,env_exports.sh}`；v56状态汇总 `reports/a321-status-and-evidence-20261013-v56.md`。
- Slot/indexer/hostmeta：`reports/a321-formal-quality-slot-batching-20261010-v14.md`、`a321-slot-formal-pass-cpu-20261010-v16.md`、`a321-slot-indexer-combined-pass-20261010-v17.md`、`a321-fair-stack-performance-20261010-v18.md`、`a321-formal-client-pass-20261010-v19.md`、`a321-hostmeta-small-gain-strict-comm-20261010-v23.md`、`a321-hostmeta-client-pass-async-20261010-v24.md`；对应 `evidence/formal_a321/results/{formal_slots_batch_perf_v2,formal_slots_indexer_perf_v1,formal_hostmeta_perf_v1}/`。
- Prefix/WOA/Mask/Quant/QKV：`reports/a321-w4a8-prefix-probe-20261010-v20.md`、`a321-prefix-up-formal-pass-20261010-v21.md`、`a321-prefix-negative-microarch-20261010-v22.md`、`a321-formal-woa-cube-20261011-v29.md`、`a321-formal-moe-mask-probe-20261011-v30.md`、`a321-formal-moe-mask-audit-pass-20261011-v31.md`、`a321-formal-mask-client-pass-20261011-v32.md`、`a321-quant-gemv-negative-20261011-v33.md`、`a321-qkv-merge-independent-pass-20261011-v34.md`、`a321-v41-qkv-audit-pass-20261011-v35.md`、`a321-qkv-tail-and-sdk-followup-20261011-v36.md`；对应 `evidence/formal_a321/results/{formal_prefix_up_perf_v1,formal_woa_model_audit_v1,formal_moe_mask_perf_v1,formal_quant_gemv_probe_v3,formal_qkv_merge_perf_v2,formal_qkv_tail_perf_v1}/`。
- DSpark/async/BI/full43：`reports/a321-draft-graph-candidate-20261013-v46.md`、`a321-async-dspark-arm-20261013-v48.md`、`a321-async-api-window-20261013-v49.md`、`a321-batch-invariant-arm-20261013-v52.md`、`a321-bi-quality-pass-20261013-v53.md`、`a321-remaining-gap-budget-20261013-v54.md`、`a321-status-and-evidence-20261013-v56.md`、`a321-full43-diagnostic-pass-20261013.md`；对应 `evidence/formal_a321/dspark_native_graph_api_v11_bi/`、`quality_gate_pass_bi_v14/`、`full43_diag_v17/`。
- 当前/历史 arm 定义：`experiments/operator_stack/parallel_tiny/dspark_formal_prep/worker.py:9–24`、`contract.py:74–91`、`tp8_patches.py:19–61,104–145`、`real_tp8_worker.py:46–72`、`launch_real_tp8_job.py:175–183,239–246`；旧 range-mask 实现 `patches/files/token_dispatcher_moemask.py:55–68`，新 exact mask `experiments/operator_stack/formal_moe_mask_patches.py:51–70`。
