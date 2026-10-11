# Native bounded-copy / epoch exporter v1

当前 NPU 权限为0。唯一获准的 physical7 GraphEx static case 已终态失败：`torch.npu.set_device(0)` 映射到 devId7 时报507033、`TsdOpen failed / tdt error=1`，exit1。配置32代，实际0代；probe张量分配、MatMul、compile、capture、epoch和copy均未到达。没有观察到ATen或alias拒绝，不能据此判断native copy支持与否。未启动nonstatic/keep-mutations、没有重试/reset；physical7已经释放，终态Health OK/Error Code NA且本任务进程为0。

[SDK终态compact](sdk_terminal_compact.json)与[只读初始化对照](init_comparison.json)给出精确口径；[原始归档](evidence/sdk_default_static_p7_20261011/collection_receipt.json)保留source_v1、argv、完整失败栈、pre/post健康及释放快照。远端与归档的source_v1不改写。

只读对照未发现device node缺失或SDK/driver版本差异：正式a321与成功p4的c6相关节点存在且major/minor/mode相同，torch2.10+cpu、torch_npu2.10.post4、CANN9.1.0、driver26.1.1及核对的SDK/worker源码SHA相同。镜像、显式Devices/默认可见芯片、network mode存在差异，但没有因果证明。生产worker在set_device之后import torch_npu._inductor；p7在之前修改配置，而p4 GraphEx v6也采用这个先后顺序并成功初始化，因此顺序差异单独不足以解释p7。指定p6证据证明初始化/原生权重分配先成功，之后分别在format守卫/workspace allocation失败；它们不构成数学或模型通过。

`exporter.py` 已实现真实 ATen `copy_/zero_/fill_/add_` 的固定地址有界观察模块：capture 前分配 values、positions、valid、row_count、copy_count、epoch；每次写 payload/extent 后增加 device copy_count，最后写 device epoch。返回的 carrier 是原Tensor，不改模型数值或地址。没有 opaque None custom op、ExternalEvent wait-copy 或隐藏全局 buffer mutation。

SDK 门 `gate_sdk.py` 已可执行：真实BF16 MatMul→producer snapshot→原生cache index_copy/index_select→cache reader snapshot；32代改变输入/positions/epoch，每代旧fence后reset，回放后当前rank全设备fence再D2H。验证payload、cache写读、返回值、device count/epoch、静态地址。实际 backend runtime 的 TorchDispatchMode 记录真实ATen copy源/目的地址/shape/stride/format/stream，Fake不记录；生成Dynamo/AOT/实际kernel源码保留。

唯一case运行的是归档source_v1版本，SHA见终态compact。当前本地gate仅补充独立进程identity（PID/boot ID/monotonic起点）、默认alias=false及精确device-initialization阶段标签；此修订尚未执行SDK。exporter源码与失败case保持同一SHA。当前安装入口需要当前冻结gate/exporter源码的两份fresh static/nonstatic实际receipt，不复用历史失败进程或旧源码receipt。

正式对齐选项为force_eager、inplace_pass=false、clone_input/output=false、functionalization_v2=false；static另有static_kernel_compile及sym_range[1]。static/nonstatic必须独立fresh进程。outer AOT默认行为可能functionalize buffer并copyback；所有值正确仍不能自动授予无alias。显式`--outer-keep-mutations`是独立审计编译对照，不修改正式compiler或活源。

## 主线程最小命令

以下仅为未来独立job的命令模板，本轮不继续运行。需主线程重新分配芯片并通过既有资源守卫；每个job终态后才可启动下一条。原始结果目录由程序创建，不能预建或复用。

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_VISIBLE_DEVICES=7 SDK_NATIVE_COPY_PHYSICAL_CHIP=7
export HCCL_DETERMINISTIC=strict LOCAL_WORLD_SIZE=1 OMP_NUM_THREADS=1
export NATIVE_COPY_RUNTIME_ROOT=/work/native_copy_gate_static_default_v1
bash /work/native_capture_copy_v1/run_gate.sh \
  --stage graphex --static-kernel --physical-chip 7 --generations 32 \
  --output /work/results/native_copy_static_default_v1
```

随后fresh进程做nonstatic同门：换新的runtime/output目录并去掉`--static-kernel`。若默认AOT出现临时alias/copyback，新独立对照仅增加`--outer-keep-mutations`，仍先static后nonstatic各fresh进程。

可先用`--stage npugraph`或`--stage eager`定位native复制/原生capture API；它们不能替代GraphEx门。

读取`result.json`、`progress.json`、environment/source SHA、实际copy_calls及生成kernel。失败条件包括：编译/真实调用拒绝、未观察实际copy、临时dest/copyback alias、epoch旧值、缺失/重复device写、payload/positions/valid/cache/returned值不符、静态地址变化或fence失败。失败进程不修复/复用，不清pagecache/reset设备。

`passed`只表示32代该阶段硬件数值/epoch语义；`actual_copy_alias_gate_passed`另列真实copy地址门。两者同时成立仍只给小SDK能力证据，`device_audit_passed/safe_formal_receipt/formal_model_validated`始终false，不能报20ms通过。

## 接入17 hook / 43层的规则

每个main0–39/MTP40–42模块的实际Tensor出口使用独立`PointSpec`和预分配observer，保留load期native引用/source/权重ptr-version-layout。通用函数返回Tensor carrier，私有审计版将其穿过实际调用点；此版本没有改任何现有hook或17点的SDK源码。

None-return cache writer不改成新的opaque mutation：在原生scatter前观察实际updates Tensor，完整post-write/cache消费值在reader前再次采样。reader观察真实cache有效行，但carrier使用q等有界Tensor；不复制整张cache。CPU bookkeeping/scheduler只做accepted-prefix/bonus/rollback对应，不能充当device覆盖。有效行包含真实positions/block+offset/逻辑cache group，不能凭注册数量或类名授device receipt。

当前`hook_carrier_policy`已给出上述接线策略；只有SDK alias/当代marker门通过后，才在新的独立job中把observer接入17点及43层，并跑原严格原生A/A/route/Top5/logprob与cache有效值门。audit时延不计perf。

`consumer_bindings.py`接收load期保留的真实native main/draft引用，不猜graph wrapper。它对43个独立对象的原forward、main/draft reader、三个distinct wkv/norm/RoPE producer作source SHA守卫；draft全局键40–42与prefix `mtp.0/1/2`产生的真实attention/MoE局部索引0–2分别记录。worker/proposer/sampler/cache plan和frontend scheduler绑定分开，所选AllGather/MC2必须来自实际MoE调用传入的对象。cache绑定同时保留逻辑group、prefix、buffer index、storage offset/stride/format，允许合法共享GM指针；weight seal保留实际参数对象和ptr/version/storage/layout，复查变化即拒绝。

`adapters.py`提供route原生integer IDs/weights、native scatter之前updates和reader之前有界有效cache行的接线；无Nonzero、整cache clone或D2H。accepted-prefix mask包含last actual+accepted drafts，bonus属于下一轮输入。`install_from_loaded_refs`必须先满足两fresh SDK门，才绑定native对象、封存weight元数据和分配observer；当前失败receipt在绑定、分配、class mutation之前即拒绝。它只准备43层输出入口，其余实际17点调用接入及逐代device导出仍需独立审计job完成，注册数量/source plan/CPU mock始终不是device覆盖。

CPU ATen测试在正式容器的CPU torch2.10运行，显式关闭NPU backend autoload，覆盖carrier恒等、32代地址稳定、旧/缺/重复写、错误payload、越界cache坐标、integer route IDs、reader/writer及bonus mask。stdlib对象绑定/门反例在本机运行，覆盖错误源、错role/对象、producer共享、weight版本变化、重复进程/epoch、缺fence、临时alias、仅Python回调或缺epoch copy。数量与源码SHA见[CPU validation](cpu_validation.json)；所有mock记录明确无NPU导入/作业及device receipt。

本目录最终精确清单为[freeze.json](freeze.json)和[SHA256SUMS](SHA256SUMS)，包含新增adapters、consumer bindings、当前测试和原始SDK证据。所有`device_audit_passed/actual_copy_alias_gate_passed/safe_formal_receipt/formal_model_validated`仍为false，无正式≤20ms结论。
