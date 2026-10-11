# SDK 小门的外 AOT / 原 SDK 调用 ABI 修订 v5

本目录只修订 v4 的 fw_compiler 返回边界。原 SDK factory、生成 kernel 的 `assert_size_stride`、两层 AOT、CopyMode、static/keep 选项、正常 init-only、强映射 ABI 身份、lazy source getter 和 controller/new-job 绑定均保留。v1–v4 冻结内容不变；本 revision NPU 权限/作业为 0，43 层消费者继续暂停。

实际 v4 rank7 在 `real-warmup` 进入生成 kernel，第90行 `assert_size_stride(args[0], (6,64), (64,1))` 抛出 `expected Tensor()`。source gate 已完成，但没有 epoch/copy 证据、alias_error 为 null。实际 receipt 没有保存首参的具体类型，因此只确认非 Tensor，不把 CPU 的 list 类型当成实际 NPU 观测，也不把该原因继承给其他 rank。

## 最小 ABI 修正

同版本 CPU PyTorch 2.10.0 使用原19输入/同程序/外内 AOT复现：原输入均为 Tensor，SDK-like backend 的 Dynamo unboxed `forward` 携带 `_boxed_call=True`；外 AOT 直接以 `compiled_fn(args_list)` 调用它，最内 kernel 得到 `(list_of_19_tensors,)`，原 Tensor assert 同样失败。仅传播最外 CopyMode invoke 的 boxed flag 仍失败。

新 `tensor_call_abi.boxed_sdk_compiler_adapter` 是私有 fw_compiler 返回接口：

1. 新 adapter 声明 `_boxed_call=True`，只接受一层 list，arity 必须等于该 FX examples 的长度（当前19）。所有元素须为真实 Tensor，拒绝少参、多参、嵌套 list、None、scalar、bool、FakeTensor 和错误层数。
2. adapter 保持所有原 Tensor 对象与顺序，以 `sdk_callable(*actual, **kwargs)` 原样调用原 SDK Dynamo 返回。SDK 函数和其 advertised flag 不变，所有 keyword 值与顺序保留；不修改 SDK、不筛 kwargs、不移除 kernel assert。
3. 最外 CopyMode/Dynamo invoke 保持 unboxed 调用，增加同样的 Tensor/arity 门。首次 `real-warmup` 仅记录每边界的 arity、type、Tensor mask、object IDs 与 keyword 名/类型，不读取 Tensor 值。capture 中不增加 ABI 日志或 D2H。失败仅保留参数引用，类型结构在 capture context 已退出的异常处理阶段写入 receipt。

采用真正 v5 helper 的 CPU 双 AOT 集成已通过：最内 kernel 接收19个 Tensor，原 assert 通过，程序返回14项。76 项单测均通过（old65 + 新 ABI11），另保存当前错误、仅invoke flag错误、prototype boxed adapter成功和实际 v5 helper成功的原始 CPU结构。CPU 与源码准备不能放行真实 device/semantics/alias 门。

## 主线程同选项的新 job

主线程用新的 fresh8–15 资源/host 门与独立 worker，在新快照中运行同一个 `static_default`，先核验真实 ABI，再判32代与alias：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v5/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

`NEW_JOB/gate` 必须尚不存在；host提供相同ASCEND可见设备8–15。driver仍从output parent取得真实job ID，绑定完整路径/nonce/PID/boot/init-return；每worker仅一次case，无load_model、复用、SDK修复或cache清理。本次 candidate 源集为12文件。其余variants各须独立fresh job，不能通过换variant掩盖未核验的调用ABI。

CPU ATen 与双 AOT 使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`python -B`，前后确认未导入真实torch_npu。所有原日志、原始结构、source SHA、旧包完整性、validation与strict freeze见 `evidence/`、`cpu_validation.json`、`freeze.json` 和 `SHA256SUMS`。

本 revision 尚无真实NPU ABI/warmup/capture/replay结论。`device_audit_passed/actual_copy_alias_gate_passed/full_consumer_audit_passed/safe_formal_receipt/formal_model_validated/actual_prepack_consumer_order_validated` 均为false，没有正式性能结论。
