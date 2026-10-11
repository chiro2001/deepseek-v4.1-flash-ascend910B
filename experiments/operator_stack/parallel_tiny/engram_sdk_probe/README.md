# 真实 SDK / AOT / GraphEx 小张量探针

2026-10-11，A3-21 physical4，单进程、单运行时设备 `npu:0`。

已验证 eager NPUGraph 的双流 external-event 路径 **32/32 代通过**；每代再插入 host `consumed.wait(main)` 的变体在第 2 代被 SDK 拒绝。真实 AOT/GraphEx 路径在 producer 首次调用时报 `TypeError: expected Tensor()`，尚未执行 GraphEx 的真实 opaque-op 回调，也未进入外层 capture/replay。其实际地址别名、执行流和 submission-stream completion fence 前置门均未通过。

所有原始结果保留 `safe_formal_receipt=false`、`formal_model_validated=false`、`prototype_executed=false`。本探针没有运行正式 Engram 权重、完整投影或正式模型，没有性能结论。

## 探针及冻结版本

[`probe_sdk.py`](probe_sdk.py) 使用 BF16 `[1,64]` 输入与 `[64,64]` identity weight，通过逐代变化且可精确表示的数据检查 MatMul、缓存、consumer 输出和设备 epoch。`publish`、`wait_copy`、`mark` 都是返回 `None` 的真实 `torch.library.custom_op`，Fake 实现只返回 `None`。

当前源码冻结为 `source_v6`：

```text
probe SHA256: bd5d995abc63dde901b558c0e44d31266d3cc14ac990e9ecd3b9007f900cc167
prototype context SHA256: 987653dd4a6887c2c0f0cc0d407509dc68cb3cbd5f36f65ec8a68ab003cb455c
```

context SHA 只说明探针编写时的原型上下文；探针没有导入或执行该原型。此前 `source` 至 `source_v5` 使用的上下文 SHA 为 `3f8176a05201d61fe5cb4b39c94c0715e95cb6f776ee827ab0a9e0110cd01ccc`，其原始结果和源码未重写。

实际 SDK 为 `torch=2.10.0+cpu`、`torch_npu=2.10.0.post4`。npugraph_ex、ExternalEvent、NPU FX compiler、AclGraph、static-kernel、compiler-config、正式 compiler-interface 和 runner 文件的路径、大小与完整 SHA 均保存在各阶段 `environment.json`。

## 分阶段结果

原始文件位于 [`runtime/evidence/engram_sdk_probe`](../../runtime/evidence/engram_sdk_probe/README.md)。以下 exit code 是独立进程的终态。

| 阶段 | 冻结源码 | exit | 已完成代数 | 结论 |
| --- | --- | ---: | ---: | --- |
| `sdk` | `source` | 1 | 0 | source-freeze 使用 `inspect.getfile` 读取 lazy wrapper 失败，尚未初始化 NPU；后续改为 `nge.__file__`。 |
| `sdk_v2` | `source_v2` | 137 | 0 | reset 后每代 host wait-before-current-record 在第 1 代收到 single-record/multiple-wait 拒绝；仅结束已核对的本任务进程，未修复或复用事件。 |
| `sdk_record_first` | `source_v3` | 0 | 32 | pristine ExternalEvent 首次 wait-before-first-record 通过；随后 host record-before-wait，包括 host consumed.wait，32 代通过。 |
| `npugraph_direct` | `source_v4` | 1 | 0 | consumer 在默认流 capture 被 SDK 明确拒绝；后续使用独立非默认 main stream。 |
| `npugraph_direct_v5` | `source_v5` | 0 | 32 | eager NPUGraph aux/main 各一次 capture，consumer-first，无每代 host consumed.wait，32 代 reset/replay 通过。 |
| `npugraph_hostwait` | `source_v5` | 1 | 1 | 第 1 代通过，第 2 代每代 host consumed.wait 被 single-record/multiple-wait 拒绝；进程直接终止，没有复用。 |
| `graphex_static` | `source_v5` | 1 | 0 | producer 首次编译运行 `expected Tensor()`；AOT functionalization v2 未显式对齐，不能作为完整的正式配置拒绝证据。 |
| `graphex_static_v6` | `source_v6` | 1 | 0 | 显式关闭 functionalization v2、启用正式 static options 后，producer 首次调用仍 `expected Tensor()`，GraphEx 真实回调尚未执行。 |
| `graphex_nonstatic_v6` | `source_v6` | 1 | 0 | 只关闭 static kernel 的诊断对照，同样在 producer 首次调用失败；该对照也未取得 GraphEx 回调或捕获证明。 |

`sdk_record_first` 的原始逐代字段 `wait_enqueued_before_record=true` 是 `source_v3` 的记录错误。该阶段的真实顺序以顶层 `enqueue_order=producer-first` 和冻结源码为准；原始证据保留，`source_v4` 起已修正字段生成。

## 32 代 eager NPUGraph 的证明范围

每代先观察旧 producer/main 普通 Event 双 fence 完成，再 reset 本任务自己的 ready/consumed ExternalEvent。输入 copy 完成后记录普通 input Event；consumer 图先 replay，producer 故意延后 5 ms。在 producer 提交前，main 上普通 consumer completion Event 在 32 代中都未完成。producer 提交后同步双 completion fence，才进行 D2H 检查。

32 代 cache、out、consumer 和 marker 均逐位一致，epoch 每代加 1，七个显式静态 tensor 地址均稳定。capture 的真实 `wait_copy` 和 `mark` 回调流与 main completion Event 的提交流相同。

这是 eager NPUGraph 的证据。`source_v5` 只记录回调流整数，没有记录 AOT/GraphEx opaque-op 的实际 tensor 参数地址，不能用于新的 submission-stream 安全 receipt。

## AOT/GraphEx 失败与未完成的门

`source_v6` 使用 `torch.compile(fullgraph=True, dynamic=False) → aot_autograd → nge.get_npu_backend`，保留与正式 compiler-interface 相同的双 AOT 链。选项为 `force_eager=True`、`inplace_pass=False`、`clone_input=False`、`clone_output=False`；static 阶段另有 `static_kernel_compile=True`、`_vllm_aclnn_static_kernel_sym_range=[1]`、`LOCAL_WORLD_SIZE=1`，并显式设置 `torch._inductor.config.enable_auto_functionalized_v2=False`。

探针在自己的独立进程内临时 hook `_NpuFxCompiler._get_compiled_gm`，保存真实 `compiled_gm.get_code()`，随后恢复方法。没有修改 SDK 文件、正式 compiler-interface 或现有正式进程。

v6 的 AOT 图使用 `higher_order.auto_functionalized`。保存的实际生成 kernel 在 `publish` 前包含 `as_strided → clone → as_strided` 的 cache 处理。`clone_input=False` 没有消除这一 functionalization clone。

但该生成代码尚未运行到真实 `publish` 回调：所有 v6 `OP_CALLS` 都来自之前的 eager 小 MatMul warmup。它们记录 stream、phase、data_ptr、shape、stride、dtype、device、NPU format 及与显式静态 tensor 的逐字段对照，即使失败也会在 finally 保存。原始 `real_callback_static_bindings_match=true` 只覆盖这些 eager warmup 回调，**不能读作 AOT/GraphEx 地址 guard 通过**。

当前尚无法将 `expected Tensor()` 进一步归因为调用适配错误、编译器对 None op 的限制或其他内部处理；本轮没有独立无 custom-op 的 GraphEx control。可以确认的是冻结的确切链未跑通，不能推广为整个 SDK 都不支持该能力。生成代码中的 clone 是编译证据，**不是实际 opaque-op alias guard 拒绝证据**。

因此以下项目仍未取得真实证明：

- AOT/GraphEx 真实 `publish/cache`、`wait_copy/out`、`mark/marker/epoch` 参数与原型严格静态地址 guard 一致。
- GraphEx 外层 capture 的实际 `wait_copy`/`mark` 流与普通 completion Event 提交流一致。
- GraphEx 下 32 代 marker/epoch、双 fence、稳定别名和 wait-before-current-record replay。
- 新 context SHA 的正式原型或正式模型安全性。

## 资源与证据卫生

全部 NPU job 都使用已授权的 privileged 容器 `dsv41-tiny-goal20-20261010-c6`，每个进程同时覆盖 `ASCEND_RT_VISIBLE_DEVICES=4` 和 `ASCEND_VISIBLE_DEVICES=4`，并断言 `device_count_before_init=1`；运行时 logical0 对应 physical4。其他芯片未被本任务启动进程。

启动前保存完整 npu-smi 并检查 card2/chip0 健康且无进程。最终两个 v6 job 均 exit1；退出后完整 npu-smi 显示 physical4 健康且无进程，container process snapshot 中也没有 probe。[`sdk_scope_release.json`](../../runtime/evidence/engram_sdk_probe/remote_host/evidence/sdk_scope_release.json) 保存了释放收据。physical4 已转交主线程，后续本任务 NPU 权限为 0。

未执行设备 reset、清 pagecache 或删除设备内部文件。只对本任务的 ExternalEvent 做了代际 reset。失败进程不复用。源码、log、exit、prelaunch/postlaunch、FX 图、实际 kernel 与紧凑 JSON 均已取回；90 个原始文件共 423746 bytes，单文件最大 20832 bytes。完整清单及原始远端路径见 `collection_receipt.json`，逐文件校验见证据目录 `SHA256SUMS`。
