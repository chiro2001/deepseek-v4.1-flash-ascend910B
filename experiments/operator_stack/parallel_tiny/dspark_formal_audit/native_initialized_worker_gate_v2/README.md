# 正常已初始化 worker 中的 native copy/epoch 小门 v2

本目录是CPU可审阅、可由未来新正式审计worker调用的准备代码。当前NPU权限为0，本目录未执行任何NPU case。v1已冻结，全部保持原样；不重试physical7，不改根counter/bench/launcher、活profiling、SDK或设备文件。所有`device_audit_passed/full_consumer_audit_passed/safe_formal_receipt/formal_model_validated/actual_prepack_consumer_order_validated`均为false，没有正式≤20ms结论。

`run_in_initialized_worker(worker, *, controller_path, controller_sha256, output=None)`在正常`NPUWorker.init_device`已经返回后调用。它没有set_device/init/reset/empty_cache、context/stream creation或物理芯片选择。正常SDK继续拥有当前context与stream。入口核对真实TP rank/world、worker device、已有context非空、实际runtime logical→physical映射和current stream；仅借用同一个scope。

controller文件及source manifest必须逐字节SHA匹配。manifest包含本候选九个源文件、实际torch/torch_npu/NPUWorker/NPUModelRunner/TP getter/GraphEx源码，以及主线程提供的准确ACL/runtime库路径与SHA。库只允许`RTLD_NOLOAD`，同时核对当前进程实际mapped inode；不加载新库，不换toolkit路径。库符号、context、stream或mapping检查失败即终态，不修复、不降级、不自建context。

## 最小新快照接线

只在主线程另行分配8–15并批准的新正式审计job中使用。保持正式checkpoint config SHA `40ebd329…d424`、full40/hidden5120、384/top6、Engram[1,14]、vision、原量化、DSpark K5/block5、PP1/DP1/TP8和严格确定性环境。该小门可以安排在normal init之后、模型/请求执行之前的idle窗口，不计入性能测量。

可运行的最小init-only入口是`init_only_driver.py`，支持run_stack直接脚本路径及package import。标准源码已核对并归档于`evidence/sdk_source_reference`：`WorkerWrapperBase.init_worker`（230–319）只构造worker，`init_device`（327–331）只转发normal init，`__getattr__`（333–334）转发RPC。`WorkerProc.__init__`（608–670）在640调用init_device、648强制load_model，READY在885之后，因此stock MultiprocExecutor不能在load前接收该RPC。新driver用八个fresh spawn进程中的原标准Wrapper分段API，保持完整formal config；没有修改/override SDK load_model，也不声称模型权重已加载。

主线程另行完成host资源/80C98001门后，首case命令为（`NEW_JOB/gate`必须尚不存在）：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_gate_v2/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

host launcher须传入相同的`ASCEND_RT_VISIBLE_DEVICES`与`ASCEND_VISIBLE_DEVICES`8–15；driver只校验它们，不改可见设备。driver沿AA v3的`formal_prep.contract.model_kwargs/startup_env(audit=False,draft_graph=False)`创建完整EngineArgs config，只将worker_cls设为本包sdk_worker私有subclass；为本job设置私有VLLM/Ascend/Triton/TMP cache。默认ABI路径为当前toolkit的`lib64/libascendcl.so`和`lib64/libruntime.so`，可用`--acl-library/--runtime-library`显式指定准确已加载路径；NOLOAD/mapped-inode不匹配即失败，没有路径fallback。正常init默认等待600秒。raw_initialized/raw_results每个真实RPC先落盘再检查，失败不会抹去已收到的原文。

既有AA v3的contract/worker依赖由主线程冻结验证；本包只交付自身完整freeze，不修改这些模块。其余三个variant须换独立NEW_PRIVATE_SNAPSHOT/NEW_JOB及fresh worker进程，不能在首case进程里继续。

新快照自己的worker模块可声明：

```python
from parallel_tiny.dspark_formal_audit.native_initialized_worker_gate_v2.worker_hook import make_worker_class
from your_new_snapshot_formal_worker import FormalDSparkWorker
NativeGateWorker = make_worker_class(FormalDSparkWorker)
```

新的`worker-cls`指向该快照模块的`NativeGateWorker`。factory创建私有subclass；super正常init成功返回后只记录worker/PID/boot/init-return生命周期，不patch已有class。与本包一起使用package import，避免同一文件被flat/package双重导入。

controller流程：

1. 对八个新worker执行`describe_native_initialized_copy_gate(library_paths={"acl": ACTUAL_LOADED_ACL_PATH, "runtime": ACTUAL_LOADED_RUNTIME_PATH})`。这一步只读身份与源码/库文件SHA，既不dlopen也不查询/选择物理芯片，不能授device receipt。
2. controller在同一共享容器路径调用`controller_cpu.write_controller_bundle(descriptions, metadata_root=NEW_METADATA_ROOT, output_root=NEW_OUTPUT_ROOT, checkpoint_config_path=ACTUAL_CONFIG_PATH, audit_job_id=NEW_JOB_ID, nonce=NEW_NONCE, variant=VARIANT, physical_chips=EXPLICIT_MAIN_THREAD_ASSIGNMENT)`。checkpoint路径必须为实际正式文件。八个描述的源码manifest必须相同；PID/boot/init-return必须唯一且与这些新worker一致。controller/root路径不能预建或复用。
3. 对八个worker执行`run_native_initialized_copy_gate(controller_path=..., controller_sha256=...)`。可直接使用collective RPC的相同kwargs；每个worker从已核对的实际TP rank得到controller绑定的`NEW_OUTPUT_ROOT/VARIANT/rankR`。显式传output时也必须完全匹配该路径。每rank都必须返回本轮真实receipt；只调用一个rank或CPU模拟不能当TP8能力通过。
4. 任一失败保存raw RPC receipt/result/traceback/FX/kernel，结束整个新审计job。成功也结束本轮worker；不在该worker切换variant或继续模型执行。

`static_default/nonstatic_default/static_keep/nonstatic_keep`分别需要独立的新TP8 worker进程/审计job。每个生命周期仅一次case，失败后及成功后都拒绝第二次调用；controller不能改一个metadata bool或换nonce复用进程。keep仅为独立AOT对照，不改变正式compiler或活源。

## 小SDK门的真实内容

每rank只分配独立小probe Tensor：BF16 `[6,64]`×identity MatMul、producer bounded snapshot、原生cache index_copy/index_select、reader snapshot。六个observer buffers（values/positions/valid/epoch/copy_count/row_count）作为显式Tensor输入，全部作为Tensor tuple输出。没有None custom op、ExternalEvent或隐藏全局buffer mutation。

GraphEx为fullgraph `torch.compile → outer aot_autograd → actual npugraph_ex`，正式对齐force_eager、inplace_pass=false、clone_input/output=false、functionalization_v2=false；static增加static_kernel_compile和sym_range[1]。临时compiler flags/源码保存hook仅限新进程，finally恢复CPU状态；任何异常仍终止job。

32代同时改变输入、positions、epoch及native cache槽，每代先完成上一代当前rank全设备fence，再reset/写入/回放；当前代全设备fence后才D2H导出，之后才可复用。数值、extent、实际device count/epoch、原buffer与返回tuple的地址/layout、原cache写读和返回值逐代核对。没有capture内D2H、日志或Python预算计数。

实际runtime TorchDispatchMode只记录real NPU Tensor的native copy，排除FakeTensor。alias门同时检查cold-capture实际destination、返回tuple地址、borrowed stream和positions/valid/epoch的原input source ptr；源为functionalization temporary/copyback也拒绝。count/rows若出现copyback而不是原生add/fill同样拒绝。数值通过与alias通过分别保留，二者同时成立也只是小SDK能力，不能授完整模型receipt。

## 最小消费者tuple与生命周期

`hook_tuple_adapters.py`只有可调用函数，不安装17点/43层hook。`decoder_output_tuple`返回原hidden/mix加六个buffer输出；`cache_write_input_tuple`在原scatter之前观察实际updates并返回原updates；`cache_reader_input_tuple`在原reader前采样有界block+offset行并返回原q。私有outer graph必须把全部observer Tensor输出继续传出，不能只保留Python引用或丢弃tuple。合法shared cache ptr需以role/global layer/logical group/prefix/storage offset/stride/format区分；draft全局40–42与MTP局部0–2分别保留。

`activation_before_mutating_consumer_tuple`必须置于真正GMM/pre-pack消费边界之前，实际copy到独立有界GM buffer，epoch最后写；再把原x交给原consumer。ND MSD原生PreProcess可以直接把INT4 high/low pack写回INT8 x，PyTorch `_version`不一定变化。激活/cache不能套用immutable weight version seal，不能在step后只保存x引用。SDK与raw每次独立调用都须先在旧fence后向同一GM ptr恢复原输入；复用已pack x会使控制失真。

future 43层binder的MoE几何必须按role与实际module/config/loader区分：checkpoint原字段main `n_routed_experts=384/num_experts_per_tok=6`，draft `dspark_n_routed_experts=128/dspark_num_experts_per_tok=3`。EP8实际local counts分别48与16，已见实际expanded GMM rows分别target6×6=36、draft5×3=15。不能以target E48/top6填draft的route/count门；当前小SDK门仍6×64、不含模型值，完整binder尚未安装。

本BF16 toy并未验证真实compiled INT8 MSD消费者的copy-before-pack次序。实际device审计还须验证该次序与stream、原始/pack后的payload、每代marker及实际消费；编译器是否移动tap不能由Python代码顺序或CPU反例推断。该项receipt始终false，直到新的实际consumer门取得证据。

完整43层/17点接线可沿v1冻结source guard与actual native references规划；但不能调用v1旧独立初始化receipt来放行本v2，也不能继承注册计数、旧A1/无DSpark数据或现有native A/A结果作为完整消费者证明。新job仍须逐代证明main/draft路由、aux37–39入口、三个独立WKV/norm/RoPE producer、实际cache writer/reader、per-group元数据、rejection、bookkeeping与scheduler rollback。accepted cache inputs是last actual+accepted drafts，bonus是下一轮输入。

CPU验证数量/源码SHA见`cpu_validation.json`，精确冻结清单见`freeze.json`与`SHA256SUMS`。stdlib假阳性反例和CPU ATen/raw-memory反例都明确无NPU backend导入、无设备作业及device receipt。
