# 正常已初始化 worker 的 SDK lazy source getter 修订 v4

本目录只修订 v3 的 GraphEx 懒代理源码身份获取，并保留标准 init-only、强映射 ABI 身份、controller/new-job 绑定、原 SDK factory 调用和 copy/epoch 小门。CPU 共 65 项通过：原 37 项、映射身份 19 项、实际 SDK lazy 类/getter 9 项。NPU 权限与本 revision 的 NPU 作业均为 0；43 层消费者工作继续暂停。v1/v2/v3 冻结文件保持原样。

主线程取回的 v3 `static_default` 原始 rank6 receipt 表明：strong 库身份和 borrowed scope 已过，probe allocation 已到；对 `npugraph_ex` 懒代理实例执行 `inspect.getfile` 时抛出 TypeError，尚未进入 compile/capture，0 epoch / 0 copy。这不能作为 alias 能力拒绝。此处只引用 rank6 的原始证据，不把该 case 的结论继承给其他 rank。

## 唯一 SDK 接线修订

实际 SDK 在 `torch_npu.dynamo` 中定义 `_LazyNpuGraphEx`，并将其实例注册为顶层 `sys.modules['npugraph_ex']`。该实例不是 ModuleType。正常 child import 加载后，实例的 `_npugraph_ex` 字段持有真正的 `torch_npu.dynamo.npugraph_ex` backing module。

新 `graphex_source_identity.py` 按以下事实核验来源：

1. 只对实际注册的 lazy **类**执行 `inspect.getfile`；核对类的 namespace、registrar 中的类对象及其源码 SHA。公共 manifest 增加 `npugraph_ex_lazy_proxy`；不对 lazy 实例调用 `getfile`。
2. 保留原来的 `import_module('npugraph_ex')` 和 `import_module('npugraph_ex.npu_fx_compiler')` 顺序。之后读取已加载的 `_npugraph_ex` 字段，与 canonical `sys.modules` 的真正 ModuleType 核对；不强制切换 import 路径。
3. 对真正 backing module 和 compiler module 分别核对 `__file__`、`__spec__.name/origin` 与完整 SHA，并检查 lazy getter 返回的 factory/config 与 backing module 的原绑定一致。
4. 来源检查不调用 compiler factory，不修改 SDK proxy/factory，不筛选或转换 kwargs。后续仍执行原 `nge.get_npu_backend(compiler_config=config)`；GraphEx/AOT、static/keep 选项、warmup/capture/replay 与 tuple 程序保持。

新公共 candidate 源集为十一文件，增加 getter helper。公共 SDK 文件身份仍不含 rank/PID/ASLR 地址；ABI proof 按 rank 单独 SHA 绑定。v3 的 self/map_files 全文 SHA/inode、current stat、实际 r-x 与 ELF offset 逐字节比较、NOLOAD symbol 归属和 overlay mount/device 证明全部保留。正常已有 context、logical→physical、stream、idle、单 case/fresh worker 守卫及 `NEW_JOB/gate` 输出契约也保留。

## 主线程新 job 首命令

主线程完成新的 fresh8–15 资源/host 健康门后，使用独立新快照与尚不存在的 `NEW_JOB/gate`：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v4/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

driver 继续取 output parent 名作为真实 audit job ID，并与完整 output/controller 路径、nonce、PID/boot/init-return 绑定。host launcher 继续提供相同的 ASCEND 可见设备 8–15；该 driver 不分配替代芯片，不修复 scope，不复用已结束的 v3 worker。其余三个 variant 各需独立 fresh job。`run_stack.sh`、正式 AA v3 配置及依赖冻结由主线程管理；没有 load_model 或正式性能测量。

## CPU 证据与边界

getter 反例通过只读收集的实际 SDK 源码 SHA 构造原 `_LazyBackend/_LazyNpuGraphEx` AST 类，配合 CPU ModuleType fixture 检查旧错误、正确类 getter、三份源码 SHA、module origin、实际 class/backing identity 与原 factory 参数传递。没有导入真实 torch_npu、没有执行真实 SDK factory或获得设备 receipt。三份完整源文件及收集 receipt 位于 `evidence/sdk_source_reference/`；rank6 原始失败的引用 SHA 位于 `evidence/cpu_graphex_source/`。

CPU ATen 9 项使用 CPU PyTorch 2.10.0 的内存模块注入，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`python -B`，前后断言 torch_npu 未导入。全部 CPU 日志、receipt、验证汇总与严格文件清单见 `evidence/`、`cpu_validation.json`、`freeze.json` 和 `SHA256SUMS`。

本 revision 未在真实初始化 worker 中运行 getter/compile/capture/replay。`device_audit_passed/actual_copy_alias_gate_passed/full_consumer_audit_passed/safe_formal_receipt/formal_model_validated/actual_prepack_consumer_order_validated` 均为 false。tiny SDK 门仍是 BF16 `[6,64]` 与 32 代 observer/cache/epoch/alias 验证；真实 INT8 pre-pack 消费者、17 点/43 层及正式性能结论尚未取得。
