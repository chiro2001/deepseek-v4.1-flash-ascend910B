# 正常已初始化 worker 的映射 ABI 身份与 copy/epoch 小门 v3

本目录仅修订 v2 的已映射 ABI 身份守卫和审计 job 标识，复用已验的标准 init-only 与 Tensor tuple 小门。CPU 检查共 56 项通过：原 contracts 24、bootstrap 4、ATen 9，加映射身份证明及反例 19。NPU 权限和本 revision 的 NPU 作业均为 0。所有完整设备、消费者、正式模型及性能 receipt 仍为 false；43 层消费者接线继续暂停。

旧 `native_capture_copy_v1` 的 60 个文件与 `native_initialized_worker_gate_v2` 的 31 个文件保持冻结内容。旧 v2 的真实首 case 在映射身份阶段拒绝，尚未分配 probe，0 epoch / 0 copy；这不构成 ATen 或 alias 能力不支持的证据。该 job 已由主线程结束，不能复用其 worker。

## ABI 身份守卫

旧 v2 把 `/proc/self/maps` 的 device 与路径 stat device 强制相等。主线程的只读 CPU 证据确认同一个 ACL/runtime inode 和完整 SHA，在当前容器分别呈现 maps device 253:2 与 overlay stat device 0:259。v3 用完整证明处理这两个视图：

1. 在自身 `/proc/self/maps` 中定位主线程给定的准确路径，拒绝缺失、deleted 或多种映射身份。
2. 打开该映射的 `/proc/self/map_files/<range>`，记录 kernel link、fstat inode/device/extent，全文计算 SHA。maps inode、map_files inode 和当前路径 stat 必须相符，完整内容必须匹配 controller 冻结 SHA。
3. 从自身 `/proc/self/mem` 读取这个对象的全部 `r-x` 段，按 maps 的 file offset 与该 mapped FD 中的 ELF 切片逐字节比较，同时记录两侧 SHA。缺段、错误 offset/extent、可写 executable、短读或任意内容差异均拒绝。
4. 只使用 `RTLD_NOW | RTLD_NOLOAD` 获得已有对象句柄，取 ACL 的 `aclrtGetCurrentContext/aclrtGetDevice` 和 runtime 的 `rtGetDevicePhyIdByIndex` 地址。每个地址必须位于本对象的 executable 段。发现阶段不调用这些符号，不加载新库，没有路径 fallback。
5. device 视图不同必须有覆盖准确路径的最长匹配 overlay mount、匹配 fstat device 和原 mountinfo 行；同时保留 maps/fstat device、mountinfo SHA、inode、完整文件 SHA、全部 executable 比较与符号地址。权限失败直接拒绝，失败 RPC 也保留已经获得的证明字段。

`mapped_object_proof.py` 只使用标准库，只读取自身 PID 的 proc 条目，不调用设备 API。它返回的结构结论不能授设备或正式模型 receipt。实际 worker 中只有证明成功后，原有 runtime snapshot 才查询正常 init 已建立的 context、logical→physical 映射和 current stream。没有 init/set_device/reset/create context/create stream 或修复路径。

## 公共清单与每 rank 绑定

公共 `source_manifest` 保持十个候选源文件、实际 SDK 源文件，以及 ACL/runtime 的 `path/bytes/sha256`。PID、ASLR 地址、mapped segments 和符号地址只存在 descriptor 的 `loaded_abi_proofs` 及独立 `metadata/loaded_abi_proofs/rankR.json` 中。八 rank 公共清单必须相同；各 rank 的证明必须匹配该 descriptor PID 与公共文件身份。

controller 的完整 SHA 同时绑定 proof 文件 SHA、rank、PID、boot ID、normal-init-return、nonce、完整 job/output/controller 路径。driver 用 `output.parent.name` 得到真实 NEW_JOB 标识，`NEW_JOB/gate` 中的 `gate` 仅是输出子目录。执行前重新证明当前映射，并核对本 rank 的 controller-bound discovery proof；不同库、段、地址、mount 证据或文件变化均拒绝。NOLOAD 句柄保留到该 worker 退出。

## 新快照首命令

由主线程重新完成 fresh 8–15 资源及 host 健康门后，在独立新快照和新 job 中执行。`NEW_JOB/gate` 必须尚不存在：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v3/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

host launcher 继续提供相同的 `ASCEND_RT_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES` 8–15；输出子目录契约不变。默认 ABI 路径沿用 v2，可用 `--acl-library/--runtime-library` 指定准确已加载路径；未匹配即拒绝，不搜索替代库。其余 `nonstatic_default/static_keep/nonstatic_keep` 各须独立 fresh job/worker，不能在首 case 的进程重试或切 variant。

driver 仍用八个 fresh spawn worker 的标准 `WorkerWrapperBase.init_worker -> init_device`，同线程 RPC，正常 init 返回后确认 model 未加载。stock `WorkerProc` 会在 init 后强制 `load_model`，所以这里复用标准 Wrapper 分段 API；不改 SDK loader。完整 formal EngineArgs/config 和环境沿原 AA v3 依赖，只更换本包私有 worker class。checkpoint SHA、TP8/full40/hidden5120、main384/top6、DSpark K5、严格确定性和已有 scope 守卫保持。主线程管理 host launcher 及依赖冻结；本目录不启动 NPU、创建容器或更改芯片分配。

## 保持的小 SDK 内容与边界

每 rank 仍为 BF16 `[6,64]` × identity、producer bounded snapshot、native cache index_copy/index_select、reader snapshot。六字段 buffers 显式输入和 tuple 输出；32 代改变输入、positions、epoch 和 cache 槽。每代先旧设备 fence 再 reset/replay，新全设备 fence 后才 D2H。数值、extent、write count/epoch、原地址与返回 tuple、stream 和 marker 源地址均核对。GraphEx/AOT/static/keep 对照选项、FakeTensor 排除、copyback 拒绝和一 case 生命周期均沿 v2。

成功也只证明这个小 SDK case；完整模型、真实 INT8 pre-pack copy-before-consumer 次序、17 点/43 层接线及正式性能结论均未取得。`device_audit_passed/full_consumer_audit_passed/safe_formal_receipt/formal_model_validated/actual_prepack_consumer_order_validated` 保持 false。

CPU ATen 检查在 CPU PyTorch 2.10.0 中通过内存模块注入运行，使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1` 和 `python -B`，前后断言 torch_npu 未导入；未调用 SDK 入口。原始 CPU 日志、receipt、root 只读支持证据引用 SHA 和完整源清单分别见 `evidence/`、`cpu_validation.json`、`freeze.json` 与 `SHA256SUMS`。
