# Native bounded-copy / epoch exporter v1

当前只有 CPU 检查通过，NPU 权限为0。主线程未分配 physical7，本任务没有启动该芯片。

`exporter.py` 已实现真实 ATen `copy_/zero_/fill_/add_` 的固定地址有界观察模块：capture 前分配 values、positions、valid、row_count、copy_count、epoch；每次写 payload/extent 后增加 device copy_count，最后写 device epoch。返回的 carrier 是原Tensor，不改模型数值或地址。没有 opaque None custom op、ExternalEvent wait-copy 或隐藏全局 buffer mutation。

SDK 门 `gate_sdk.py` 已可执行：真实BF16 MatMul→producer snapshot→原生cache index_copy/index_select→cache reader snapshot；32代改变输入/positions/epoch，每代旧fence后reset，回放后当前rank全设备fence再D2H。验证payload、cache写读、返回值、device count/epoch、静态地址。实际 backend runtime 的 TorchDispatchMode 记录真实ATen copy源/目的地址/shape/stride/format/stream，Fake不记录；生成Dynamo/AOT/实际kernel源码保留。

正式对齐选项为force_eager、inplace_pass=false、clone_input/output=false、functionalization_v2=false；static另有static_kernel_compile及sym_range[1]。static/nonstatic必须独立fresh进程。outer AOT默认行为可能functionalize buffer并copyback；所有值正确仍不能自动授予无alias。显式`--outer-keep-mutations`是独立审计编译对照，不修改正式compiler或活源。

## 主线程最小命令

主线程先按现有资源守卫确认并授权physical7：健康状态、已授权告警、无租户进程和设备映射。以下每条只在授权后执行；每个job必须终态才启动下一条。原始结果目录由程序创建，不能预建或复用。

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

CPU测试在当前容器的CPU torch2.10运行，显式关闭NPU backend autoload；7项反例通过，含重复GM写、旧epoch、错payload、越界、carrier恒等、32代reset地址稳定、None writer采样策略。`cpu_mock_receipt.json`明确NPU导入/作业/设备receipt均为false/0。
