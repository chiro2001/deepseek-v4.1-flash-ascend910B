# main/draft GMM 原生 prepack 审计组件：CPU 准备

本包只新增独立 component，没有安装任何 hook，也未修改 qkv/full43、模型、SDK、权重、旧源/冻结包或 root 文件。最后一轮 **16 项 CPU torch 测试、41 个拒绝反例通过**。19 个真实源码文件、原17点计划和历史 M1 事实均有 SHA；NPU/服务/真实 HTTP 为 0，actual graph、prepack order、device/consumer、formal、性能及收益全部 false。

| role | 真实 routed experts / topk / local E | decode token extent / expanded M | INT8 raw K 来源 |
|---|---|---|---|
| main，global0–39 | 384 / 6 / 48，EP8 | 6 / 36 | GMM1=`loaded w13.shape[1]`；GMM2=`loaded w2.shape[1]` |
| draft，global40–42、MTP local0–2 | 128 / 3 / 16，EP8 | 5 / 15 | 同样按实际该 role 的 loaded 权重/FFN绑定，不能套用 main 默认 |

`contracts.bind_owner` 接受 retained decoder、真实 quant layer、selected router 和 W4A8 scheme，检查其 owner/submodule 归属、role/global/local index、EP/route gate、original single NZ bank/scale/assist 的对象及元数据。实际 ModelSlim 路径为 `layer.quant_method.quant_method = scheme`，不是假定 quant_method 本身就等于 scheme。GMM2 的 K 还须与 loaded `intermediate_size_per_partition` 和真实 GMM1 INT8 输出末轴相符。历史 profiler 的 K=2304 只作旁证；CPU额外用 FFN1792 反证代码没有硬编码通用2304。

真实消费边界：

- GMM1：`AscendW4A8DynamicFusedMoEMethod.apply_gmm1_act_quant` 中，`_quant_hidden_states` 返回实际 INT8 后、`torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(...)` 前。不能在 apply 入口复制 BF16 输入当成 raw INT8。其 CANN `GMMA8W4PreProcess` 从原 xGM 读取，将 high/low INT4 写入 `workSpaceGM` 与 `workSpaceGM + workSpaceOffset1` 双 bank；不能宣称它使用 GMM2 的原 xGM 写回路径。
- GMM2：`apply_gmm2` 消费 GMM1 返回的实际 INT8 `hidden_states`、`act_out_scale`，tap 在 `torch_npu.npu_grouped_matmul(...)` 前。CANN ND A8W4 MSD selector 调用 `op1.Init(x, x, ...)`，因此原 x GM 是 pack 输出地址。NZ输入/其他模板不能从该 ND 源码路径推断。
- 原生 `apply_moe_mlp` 的顺序为 GMM1 → 原 `current_stream().record_event()` → GMM2；本组件不改其事件或数学次序。CPP `aclCreateTensor` 使用原 storage.data、storage_offset、shape/stride/format；`EXEC_NPU_CMD` 使用原 `getCurrentNPUStream().stream(false)` 和原 workspace allocation。Python getter 为真实 `torch_npu.npu.current_stream` 来源；profiler stream47 不是该 handle。

以上原函数/参数、INT32 NZ format29、INT8 ND format2、borrowed stream、原 GM/storage 地址来源与 prepack target/workspace 来源见 `evidence/installed_sources.json`。真实 workspace 地址和实际 kernel/tiling 目前没有观察到，明确保留为设备门的缺口，不能从源码参数名编造这些地址或授实际消费者证明。

最小 full43 接缝是 `pure_graph_factory.make_pure_graph_entry(stage, actual_native_globals)`：

```python
g1, source_g1, receipt_g1 = make_pure_graph_entry('gmm1', native_gmm1.__func__.__globals__)
q, scale, *raw_g1 = g1(scheme, mlp_input,
    audit_positions=positions_g1, audit_valid=valid_g1,
    audit_epoch=device_epoch, audit_buffers=buffers_g1)

# 在原位置保留 before_gmm2_evt = current_stream().record_event()
g2, source_g2, receipt_g2 = make_pure_graph_entry('gmm2', native_gmm2.__func__.__globals__)
out, *raw_g2 = g2(scheme, mlp_input, q, scale,
    audit_positions=positions_g2, audit_valid=valid_g2,
    audit_epoch=device_epoch, audit_buffers=buffers_g2)
# 原q/scale/out继续原生模型；raw_g1/raw_g2各6个Tensor必须一起传出outer graph。
```

`audit_buffers` 是预分配的六个独立 Tensor：raw INT8 `[M,K]`、positions INT64 `[M,1]`、valid BOOL `[M]`、epoch/copy_count/row_count INT64 `[1]`。snapshot 先完整复制原始 capacity 的 INT8 字节，不按valid mask清洗、cast、zero或NaN修复；epoch最后写。原consumer call/kwargs 的 AST逐项相同。GMM1 返回原 q/scale 加6个snapshot输出，GMM2返回原 out加6个输出。原载体没有用clone替代，也没有人为的加零数学依赖。纯 Tensor 源中没有 hostledger、data_ptr/version查询、fileSHA、D2H或SDK import。

`native_adapter_factory.PreparedStageAdapter.call_eager` 提供 capture 外的 owner/GM/参数检查与生命周期协议接缝，适用于已得到实际 raw INT8 的独立调用；它的 Python状态不进入 replay graph。BF16到INT8的消费入口应使用上面的 source-derived纯Tensor边界，不能在量化前凭空创建 raw lease。生产 eager adapter还要求实际 selected OpOverload schema、borrowed stream来源及具有原job/rank/source引用的 fresh TP8双SDK门证据；qkv v9 loader的 `protocol_passed/as_dict()` 只作先决条件，不授本包prepack/device/formal通过。

动态/prefill fallback 边界由 full43 外层在 capture 外按实际 extent/role/source 选择：不等于本包固定 M36/M15 的动态/prefill几何、其他format/overload、非SILU、expert weight list、EPLB/LoRA，继续调用 **原生方法**。不强制padding、改counts、转dtype、NaN sanitize或复用错误role。fallback必须记为本component未覆盖，不能授prepack/full43 coverage；该extent的实际审计需要另一个对应的审计点。

`raw_lifecycle.py` 是宿主控制协议，任何native consume之后都将原lease视为consumed；即使 `_version`、shape/ptr/format完全一样也不能复用。独立 A/A 必须调用prior-work fence，再从真正before-pack的固定snapshot用`copy_`恢复同一个原GM；或在prior fence后调用新的真实producer/refeed，并绑定新generation。后者的实际device producer/epoch证据必须由full43取得；Python callback或一个generation标签本身不授设备内容证明。compiled replay不能靠host Python状态作每代证明。

历史 M1（合成 INT8 activation、真实 E48 W2、capacity6/K2304）在同一原GM上，SDK与raw调用后的字节SHA改变，而下一调用前恢复原始字节；原未使用行保持。该回执没有实测 ATen `_version`，不补造该事实。本轮另以真实CPU torch + ctypes raw pointer write验证“内容变化但 `_version` 不变”，CPU mock保留原before-pack副本；after-pack clone、错role/shape/owner、packed x 未恢复、错误stream标签、丢输出/alias均拒绝。历史实验不能代替本轮正式main/draft消费者验证。

准备状态及限制：

- `evidence/CPU_validation.json` / `.log`：16 CPU、41拒绝反例、四个main/draft×GMM1/GMM2正例和四个纯Tensor source-entry正例；所有消费者、quant、workspace、stream都是明确 CPU fake。
- `evidence/private_gmm{1,2}_tensor_entry.py.txt`：最小私有源码接线样本，原数学/调用参数AST保持。
- `evidence/frozen17_and_M1_provenance.json`：原17点计划 SHA与旧M1 raw生命周期来源；只扩展、不替换计划。
- 未 capture/replay真实graph、未取得真实pack target/workspace/stream/device epoch、copy-before-pack次序或消费者精度。Python/CPU顺序不能证明编译器未移动tap；即使双SDK toy门全过，实际INT8消费者仍须新正式权重审计窗验证。
- 原权重、原数学和所有严格精度门不变，audit时延不算perf，没有任何收益声明。
