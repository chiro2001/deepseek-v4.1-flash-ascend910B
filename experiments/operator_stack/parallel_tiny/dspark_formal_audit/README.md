# DSpark actual-consumer/cache 审计准备

本阶段为 **CPU/source guard 原型**，NPU 权限为 0。目标仍为正式 full40 TP8、包含全部 DSpark 开销的 ≤20 ms/step；本目录没有执行消费者审计，也没有性能或正式安全 receipt。

原生 ON 基线、图候选与消费者审计分开。当前正式 stock DSpark 硬禁 draft 图，主线程已选择真实 eager draft 先完成原生 ON A/A；图实现候选由 tiny 专线先验证。本目录不改终态/活快照中的 bench、counter、worker、SDK、ARM 或 launcher。

## 已封存的实际来源

生产 container 为 `dsv41-real-stack-tp8-20261010-a321`，正式冻结源 `/work/src_dspark_native_v2`。`discovery/source_receipt.json` 记录只读取得的 SDK/生产源码、原始路径和 SHA。单文件均小于 1 MB；源码只解析为 AST，没有导入 torch/vLLM 或执行设备逻辑。

[`stock_graph_support.json`](discovery/stock_graph_support.json) 将三种证据分开：v2 请求的 `enforce_eager=false`/target `FULL_DECODE_ONLY[6]`、stock 构造器第75行无条件 `self.use_cuda_graph=False`、实际8 rank bank 的 `draft_graph_configured=false`。stock DSpark 文件 SHA 为 `4bbd1d555781bcf4e143a499d56d8b962229a5d0b2b233e13242627184aa47ac`，明确注释此路径只支持 eager；没有配置层开图方法。

[`formal_v2_terminal_compact.json`](discovery/formal_v2_terminal_compact.json) 保存 exit1、完整拒绝 traceback、请求配置、实际 bank 和原始大文件 SHA。v2 bank 没有记录实际 draft compilation_config 的完整值，因此不会从请求配置推断该值。

## 新独立 job 的最小接线

[`consumer_source_plan.json`](consumer_source_plan.json) 已逐一核验17个实际函数的完整文件 SHA、函数 AST SHA 和行号。[`formal_dspark_audit_patches.py`](../../formal_dspark_audit_patches.py) 只生成守卫/绑定计划及验证 CPU 协议，不安装运行时 patch。

| 组 | 真实消费者点 | 必须绑定的实际数据 |
| --- | --- | --- |
| main/MTP 层与路由 | 共享 `DeepseekV41DecoderLayer.forward` 的实际43层出口；`AscendFusedTopKRouter._compute_routing`→实际 AllGather/MC2 dispatcher | load 时 native 模块/模型来源、参数 ptr/version/shape/stride/format；实际 input/output、token/position、top6 IDs/weights；不能以模块注册数量代替 device payload。 |
| aux/context KV | target aux residual→`combine_hidden_states`→各层 `_project_shared_kv`→`_store_standard_swa_kv` | checkpoint[37,38,39]→runner[38,39,40]→实际进入layer_idx[37,38,39]；三层不同 WKV/norm、RoPE位置、每组真实 context slots 与写入值。 |
| cache 写读 | `DsaAttnKvPlan.dsa_kv_compress_scatter`、target long/indexer scatter；main SparseFlashMla 与 draft `_forward_attention` 的真实 cache operands | 实际 cache plane/storage/逻辑group/物理stride、block_table、有效slot、seq_len/indices、写入后与读取时的有效内容。多个 layer 的相同 ptr 不表示同一逻辑 cache owner，需 group/slot/storage-offset 对应。 |
| verifier/rollback | `_sample`→原生 rejection output→CPU bookkeeping→`Scheduler.update_from_output` | coreID映射、真实调度提议、raw接受前缀/bonus、末轮budget裁剪、logical computed rollback与下一轮实际读写边界。 |

独立 job 在标准平台初始化/load 后、capture 前保留 native target/backbone/draft 引用，按精确 source guard 在该进程内重建/绑定方法。每个 observer 使用 capture 前预分配的固定地址、有界 copy/epoch/positions 槽。capture 中只允许原生设备 copy/marker 操作，不 D2H、不日志、不 Python 预算计数；新传输形态的实际 GraphEx 别名/marker 保留仍须独立验证，不使用此前未通过的 opaque `None` mutation/wait-copy。

真实 decode/propose/sample 窗口在图外分代；上一代全设备 fence 与导出完成后才复用槽，审计 job 可每代同步当前 rank 设备以覆盖 hidden streams。固定device epoch、真实position/token payload、原始consumer地址/布局与实际cache有效值共同构成后续覆盖证据。所有 audit 拷贝、重算和 fence 的时延均不计正式 perf。

原正式 Indexer 框架见 `discovery/indexer_consumer_framework.py.txt`：其方法是在 cache 写入点保存有效slot的值，再在隔离 scratch 中按原生 norm/RoPE/quant 重算并 bitwise 比较。旧 eligibility 只有 rows1，且使用 capture 内 clone；本次 target6/K5 不继承旧通过结论，需要固定 buffer 与逐代有效坐标对应。

## CPU 原型检查的边界

`TensorSeal` 检查 ptr、storage ptr/offset、shape、stride、dtype、device、format；权重还检查 version。mutable cache version 不是 device epoch。

`accepted_cache_transition` 区分 verifier 输入和输出：target 输入是上一actual token+drafts，cache 提交的是上一actual token+已接受 drafts；bonus token 已被预测，但到下一轮才作为输入计算。rollback 按 raw 接受数，末轮前端 budget 裁剪单列，不混为 rejection。

`GenerationEnvelope` 要求从 `LLMEngine.add_request` 返回值及 `OutputProcessor.external_req_ids` 获取实际 external→core 映射，不猜前缀；拒绝未 fence 导出、未导出复用、旧 epoch 与 CPU callback 伪装的 layer payload。即使合成数据通过结构检查，返回仍固定 `device_audit_passed=false`。实际 device exporter 尚未实现，`install_in_new_job` 明确拒绝安装。

严格原生 A/A 门仍为完整 token/route顺序、Top5 keys、有限 logprob delta<1e-3；整数/坐标/运输后的cache值需 bitwise；BF16计算按原独立消费者门验证。不能用全局 MeanAcceptance、类名、注册数或 Python 回调数替代这些门。

```bash
python3 experiments/operator_stack/formal_dspark_audit_patches.py \
  --discovery experiments/operator_stack/parallel_tiny/dspark_formal_audit/discovery \
  --output /tmp/dspark_consumer_source_plan.json
python3 experiments/operator_stack/parallel_tiny/dspark_formal_audit/test_contract.py -q
```

6项CPU反例检查通过；源码/函数 AST 守卫通过。后续实际固定buffer capture 别名、43层设备payload、三层context KV、cache读写/rollback对应和NPU数值门仍未验证。

## stock v2 独立兼容副本

`stock_compat_v2/` 是供主线程新快照使用的 counter/bench 副本，原冻结文件没有被覆盖。`stock_graph_observer.py` 只观察实际 FULL forward context、已存在entry、地址对应及该entry `NPUGraph.replay` 成功返回，记录图提交；它不授予层/cache消费者覆盖。`use_cuda_graph` 只作为 configured 字段。initial-prefill draft 的 NONE/eager 单列，`FULL_DECODE_ONLY` 只要求 steady decode proposals 真正 replay。相关 SHA 与复制规则见 `stock_compat_v2/receipt.json`，CPU核验见 `stock_compat_v2/cpu_validation.json`。
