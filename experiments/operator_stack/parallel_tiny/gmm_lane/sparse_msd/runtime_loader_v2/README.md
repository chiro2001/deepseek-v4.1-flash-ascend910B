# 独立 loader v2：CPU 阶段完成

此目录修复 v3 最小注册门的明确阻断：`aclrtBinaryLoadFromFile+MAGIC` 被 SDK 返回107000，原文为 `RT_LOAD_BINARY_OPT_MAGIC does not support in load from file.`，并未进行 kernelName 查询。旧 so/source、GMM/planner对象、17个原生函数体与 tiling 保持冻结，旧运行失败证据在 [p8 runtime_gate_status.json](../evidence/runtime_registration_p8_20261011/runtime_gate_status.json)。CPU修复阶段无NPU启动；随后新授权的physical6门已通过Data注册与真实函数配对，首个R0 planner同步返回507035后终态停止，见 [p6 runtime_gate_status.json](evidence/runtime_registration_p6_20261011/runtime_gate_status.json)。physical6已销毁自身context，fresh smi确认无进程、Health OK、已释放；没有重试或迁往8。

[loader_v2.cpp](loader_v2.cpp) 改用 `aclrtBinaryLoadFromData`，从只读文件取得完整host blob，显式配置 `isLazyLoad=0` 和原 vector/AICORE MAGIC。新so `libgmm2_sparse_msd_loader_v2.so` 为29600B，SHA256：`b2e6c3033f372a0542d96f9aa2053f230617052aacd2b37bd484a5293b4cf884`。它仅引用四个SDK符号：Data加载、函数查询、函数地址查询、卸载；没有device/context初始化、分配、kernel提交或File加载。

## SDK 约束审查

[安装SDK头摘录](evidence/sdk_constraints_complete/acl_header_excerpt.txt) 提供Data加载声明、MAGIC值和选项布局；[SDK SHA与ARM地址证据](evidence/sdk_constraints_complete/SDK_constraints.json) 保存三个实际runtime后端的静态窗口，不加载这些库。

`libruntime_v201.so` 的具体窗口 [magic_parse](evidence/sdk_constraints_complete/libruntime_v201.so.magic_parse.6540112.0.asm) 表明：

- `0x4eb278` 将option type2派发到MAGIC解析；type1接受non-lazy值0/1。
- `0x4eb370–0x4eb374` 读取loader的file-mode字段，非零跳到 `0x4eb8c4` 的file拒绝错误。
- 非file分支读取option value并比较 `0x43554245`（AICORE）、`0x41494343`（Cube）、`0x41415246`（Vector），随后 `0x4eb394` 将有效MAGIC存入loader。

v100/v200后端保留同类的MAGIC解析与file拒绝窗口。ARM窗口中的导出符号标签来自最近的dynamic symbol，定位依据是实际指令地址和错误/解析字符串引用。随后physical6实测Data加载、三种基础kernelName查询与vector/mix严格配对均成功，三个handle及77352B blob均保持至关闭完成。

## 类型、生命周期与失败处理

v2先读取并校验三个冻结ELF，再开始任何SDK调用：

| 项目 | planner | native/candidate |
|---|---|---|
| ELF64小端machine | `0x1029` | `0x1029` |
| 实测冻结flags | `0x930000` | `0x940000` |
| 参数bytes | 16 | 72/72 |
| 类型配置 | explicit Vector MAGIC | explicit AICORE MAGIC |
| 原生metadata | 真实单kernel ABI | 两个 `_mix_aic/_mix_aiv` section，KTYPE4、task ratio1:2 |
| runtime地址要求 | AIC空、AIV非空 | AIC/AIV均非空 |

完整blob小于1MB，存在有效`.text`；固定kernelName保持原v3值。源里保留`aclrtGetFunctionAddr`的严格vector/mix配对，没有通过删除类型检查来接受加载结果。原decoder/数值/调度对象均没有重编译或修改。

每个blob由library对象持有，其地址和字节在SDK handle存活期间保持稳定。关闭按candidate→native→planner执行，所有handle成功卸载后才删除owner并释放blob。加载失败后的回滚会保留原SDK错误和具体阶段；若任何卸载失败，`*owner`仍非空，全部blob继续持有，status同时记录primary与cleanup错误。部分关闭的owner不能返回可用function handles；已有owner不能被另一次open覆盖。调用方应在全部kernel完成后close，检查返回值，保持失败owner直到其SDK handle真正关闭。

[loader_v2.h](loader_v2.h) 使用独立 `gmm2_sparse_loader_v2_*` 名称，避免覆盖旧入口。成功后返回原 `Gmm2SparseFunctions` ABI，可交给冻结host adapter；此CPU阶段没有提交该adapter的GMM计算。

## CPU 验证与复现

[CPU_contract.json](evidence/build/CPU_contract.json) 与 [338项mock结果](evidence/build/mock_run.log) 通过。mock只链接普通C++与本地SDK函数替身，不链接或加载libascendcl/libruntime；验证实际冻结ELF解析、显式MAGIC/non-lazy、原kernelName、逐模块加载/查询失败、vector/mix拒绝、partial SDK handle回滚、卸载失败的blob保留、状态记录及owner保护。四个v3对象和13份冻结源码SHA均不变。源码SHA也与远端实际构建一致。

在独立新根目录保留如下源码层级：`source/runtime_loader_v2/*`、`source/host_adapter.h`。后者必须与v3冻结header SHA一致。运行：

```bash
python source/runtime_loader_v2/build.py \
  --root /work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_repro \
  --frozen-root /work/gmm_lane_sparse_msd_build_20261011_v3_repro
python source/runtime_loader_v2/inspect_sdk_constraints.py \
  --root /work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_repro
```

脚本固定两种visible guard为空，拒绝覆盖既有build证据。实际CPU根为 `/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011`，so在其 `build/` 下。没有全局SDK/so安装、正式源码修改或commit。

## physical6 实测与剩余阻断

[p6运行事件与清理](evidence/runtime_registration_p6_20261011/run_receipt.json) 证实count1、logical0→physical6，动态visible两个变量均为6，容器全局8–15配置不变。Data/non-lazy/MAGIC和基础kernelName在真实SDK通过，实际函数大小为：

| 函数 | AIC | AIV |
|---|---:|---:|
| planner | 地址空，0B | 地址非空，368B |
| native | 地址非空，5152B | 地址非空，8440B |
| candidate | 地址非空，8748B | 地址非空，8440B |

原冻结so/对象与源码SHA在运行前后不变。首个R0 case的counts H2D、frame memset及planner提交均返回0；第一次 `aclrtSynchronizeStreamWithTimeout` 返回507035（vector core exception、InnerCode0x715005e、retCode0x31）。fault kernel为 `gmm2_sparse_row_plan`，PC offset0x74；SDK原文为“scalar instruction accesses an invalid GM address or cross-device memory access times out”，见 [精确SDK错误](evidence/runtime_registration_p6_20261011/runtime_precise_error.log)。错误同时记录host argument映射drvRetCode11、para_base为host栈地址 `0xffffe48a5bf8`，下一步应CPU审查launch args内存契约，尚未把根因判为硬件或数值。

本次只有1次planner提交，frame未读回，其余R1/3/6和expert0/47样例未运行。stream、384B GM buffers、384B pinned host buffer、v2 handle/blob、context及ACL finalize清理均返回0，owner为空、live binaries/blob bytes为0。全日志1672143B留远端，compact证据包29888B。GMM数学、大权重和变异对象启动均为0，无模型精度或收益结论。

当前门已按SDK初次失败终态关闭。后续需要独立CPU launch ABI修复和主线程新的最小窗口，保持现有类型、SHA、配对及blob生命周期检查。GMM精度、实际MoE/capture及完整TP8 DSpark/MTP≤20ms目标仍未验证；M1capacity6只覆盖符合几何的draft/target步。

独立 [launch args v3 CPU修正](../launch_args_v3/README.md) 已准备明确HostArgs/SDK WithConfig路径，消除host栈交给dev/no-copy入口的参数契约缺陷；36项mock通过。它未改变此目录的注册代码、旧运行证据或编译对象，修正运行尚待授权。
