# Launch args v3：CPU 契约修正

本目录冻结CPU阶段的安装SDK读取、ARM静态审查、CPU链接和mock。原v3 GMM/planner对象、17数学体、tiling、loader v2与历史p6失败终态均保持冻结。新独立so为8440B，SHA256为 `5af9db3a35f3cc652bb7643b71f64fb621c70e66e89cd6896b0d0c49787571f5`，见 [CPU_contract.json](evidence/build/CPU_contract.json)。随后独立授权的 [physical6 HostArgs R0门](../launch_args_v3_npu/README.md) 已成功提交、同步和读回frame，未启动GMM数学。

## 已确认的参数契约缺陷

[p6参数日志摘录](evidence/sdk_audit/p6_observed_args_excerpt.log) 显示：旧planner将16B普通host栈数组交给 `aclrtLaunchKernel`；SDK记录 `isNoNeedH2DCopy=1,argsSize=16,hasTiling=0`，随后 `halMemHostGetDevPointer` 返回drv11，设备para_base仍为host栈地址 `0xffffe48a5bf8`。首次R0 planner同步返回507035，SDK提示scalar GM地址无效或跨设备访问超时。这个调用把未映射host参数buffer交给no-copy路径；旧数组活到同步点，单纯延长变量寿命不能修复地址空间错误。counts的384B pinned host buffer并不注册独立的参数栈。

安装SDK静态绑定也证实旧GMM wrapper的 `aclrtLaunchKernelV2` 转到 `rtsLaunchKernelWithDevArgs`：`libacl_rt_impl.so` 的 `0x877e4` 调用该符号。普通LaunchKernel的 `0x82758` 调用 `rtLaunchKernelByFuncHandleV3`，与p6真实no-copy记录一致。没有把507035直接归因为硬件或数值，也没有宣称修正后运行已通过。

## 支持的Host与WithConfig路径

[SDK头与调用契约](evidence/sdk_audit/launch_memory_contract.json) 保存声明、源码SHA及实际绑定。`aclrtLaunchKernelWithHostArgs` 的实现 `0x87c7c` 转到 `rtsLaunchKernelWithHostArgs`。其普通rtArgsEx分支在 `0x505f8` 清零offset31的 `isNoNeedH2DCopy`；[公开结构定义](evidence/sdk_audit/rtArgsEx_layout_excerpt.txt) 明确0表示需要H2D复制。调用方提供host参数字节，参数字节里的指针值仍为原GM地址。

`aclrtLaunchKernelWithConfig` 接受opaque `aclrtArgsHandle`，不能把16B原数组强转成此句柄。[安装SDK launcher](evidence/sdk_audit/installed_launcher_excerpt.txt) 的实际顺序为：

```cpp
aclrtKernelArgsInit(function, &sdk_args_handle);
aclrtKernelArgsAppend(sdk_args_handle, host_bytes, bytes, &parameter_handle);
aclrtKernelArgsFinalize(sdk_args_handle);
aclrtLaunchKernelWithConfig(function, blocks, stream, config, sdk_args_handle, nullptr);
```

源码为安装态 `mskl/launcher/code_generator.py:1036/1054–1058`，使用普通host `vector<void*>` 参数并持有SDK member handle。Finalize在提交前执行，用于完成参数构造，不是释放API。该样例结束时卸载binary，公开头没有独立ArgsDestroy；v3包装借用合法SDK handle，不伪造内部结构或destroy入口。`InitByUserMem`需要SDK计算的actualArgsSize与合法用户host memory，本最小修正采用SDK管理内存或明确HostArgs接口，避免手工页注册及未验证的内部布局。

保守生命周期契约为：host参数源保持至该次stream完成；SDK args handle及其function/module保持至stream完成；GM pointees、pointer-list和workspace也保持至完成。同一个active句柄的参数不并发修改。注册blob生命周期仍由loader v2管理至全部SDK handle关闭。不能把普通host栈当作device/no-copy地址；若另行采用no-copy路径，调用方必须有可被该device访问的GM地址或成功注册/映射的host区域，并持有至任务完成。

## 精确最小修正与范围

[launch_v3.cpp](launch_v3.cpp) 的推荐最小入口用明确host API替换旧提交调用，维持全部参数字节：

```cpp
Gmm2PlannerArgsV3 args{counts_gm_address, frame_gm_address};
int status = gmm2_launch_v3_host_args(planner_function, stream,
                                    &args, 16, 1, 0);
// 检查status，同步成功后才读回128B frame/执行CPU oracle。
```

没有inline host数据指针需要设备重定位，placeholder array为nullptr、count为0。planner仍为两条64位GM地址、16B、block1、schedule0；GMM仍为原九条64位地址、72B、block24、schedule1，workspace在offset64。GMM的ffts/list/counts/token_scale/workspace取值、单tensor list结构和112MiB+128B容量/别名guard继续由冻结host契约验证，本v3只修参数传递方式。

`gmm2_launch_v3_with_config` 给持有合法SDK argsHandle的调用方提供Append→Finalize→WithConfig序列，逐项检查SDK错误并返回failed_stage1/2/3。它不初始化context、选卡、分配GM、注册ELF或运行数学；调用方负责句柄创建、所属function一致性及stream完成后的module收口。两种入口均不调用旧LaunchKernel/V2。

[36项mock](evidence/build/mock_run.log) 通过，验证payload顺序/大小、调度、host入口零placeholders、WithConfig真实构造次序、逐阶段错误传播及错误profile在SDK前拒绝。mock没有链接或加载libascendcl/libruntime；它模拟复制，不作为SDK运行或planner可见性证明。

实际CPU根 `/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011`，so在 `build/libgmm2_launch_args_v3.so`。复现命令（复制本目录源到该新根的source/）：

```bash
python source/build.py --root /work/gmm_lane_sparse_msd_launch_args_v3_cpu_repro \
  --frozen-root /work/gmm_lane_sparse_msd_build_20261011_v3_repro \
  --loader-v2-root /work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011
```

40份小证据、4份实际构建源码SHA通过校验；证据包64709B，没有安装/覆盖SDK或正式so，没有commit。后续仅在新的资源消息后验证host-copy参数提交、参数地址/stream行为及R0/1/3/6、expert0/47 frame；出现首次SDK错误仍须终态清理，不放宽type/SHA/pair/owner guard。GMM数学、≤1ULP、收益、capture和正式DSpark/MTP≤20ms仍未验证，M1capacity6不外推多token verifier。
