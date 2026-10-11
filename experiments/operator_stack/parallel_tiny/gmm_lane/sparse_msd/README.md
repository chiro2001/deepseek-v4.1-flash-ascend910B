# GMM2/down sparse MSD：CPU 与 HostArgs R0 冻结

当前交付为独立设备 planner、私有原生 MSD 调度补丁和独立 host adapter。CPU 编译、tiling 一致性、编译后稀疏分支与 ABI 契约通过。新的 physical6 HostArgs R0 门已成功：Data 注册、真实 vector/mix 地址及大小检查、一次 planner 提交、同步及128B frame读回全部通过，frame逐位等于CPU oracle，前后canary完整；全部清理成功，fresh smi确认physical6无进程且Health OK。该冻结交付的GMM数学启动和权重字节仍为0，精度及性能尚待验证。所有修改仅在本目录及远端私有 build 根目录，无全局 SDK/so 安装或正式源码修改。

当前正式目标为 TP8 包含 DSpark/MTP 草稿、target 验证、调度与 cache 的完整推理≤20ms/step。此候选固定 M1 路由容量6，只能后续报告符合该 profile 的 draft/target 单token步；多token verifier 的覆盖需要另行证明。

历史physical8 File+MAGIC加载返回107000；历史physical6普通LaunchKernel使用host栈/no-copy=1时，首个R0同步返回507035。这两次失败终态均保留，详见 [loader v2实测记录](runtime_loader_v2/README.md)。

独立 [launch args v3](launch_args_v3/README.md) 保留16/72B参数布局，CPU链接与36项mock通过。新 [HostArgs R0实测](launch_args_v3_npu/README.md) 确认SDK实际记录no-copy=0、argsSize=16、physical6/stream46，并成功读回frame。其成功只覆盖R0 planner；后续已授权的M1 capacity6数值门使用独立目录与证据。

## 冻结实现与证据

- [prepare_source.py](prepare_source.py) 复制安装态完整 GMM 源，只改变两个 MSD 类的 AIC `Process` 调度；AIV 保持原 48-group 遍历。`pre` 源与 17 个原生函数体逐字节不变，SHA 和实际 diff 见 [source_contract.json](evidence/v3/source_contract.json)、[native.diff](evidence/v3/native.diff)。
- [planner.cpp](planner.cpp) 一次读取原 `INT64 counts[48]`，生成最多六个非空专家槽；仍保留完整 E48 NZ 参数，按原专家编号访问原生权重、scale 和 assist。候选 raw 名为 `gmm2_sparse_msd`，没有调用旧 WeightNzV2 API。
- [编译后分支契约](evidence/v2/compiled_branch_ablations/compiled_branch_contract.json) 比较真实 ELF `.text`：原命令重建逐位相同；只改变状态字地址、rowBegin 地址各改变 1 字节；只改变 activeSlots 上界也改变机器代码；关闭稀疏分支使 AIC `.text` 从 8748B 降到 5152B。对照仅用于 CPU 代码生成证明，禁止启动这些变异对象。SDK objdump 无法解码 ISA，未把 `<not available>` 当作反汇编证据。
- [调度契约](scheduling_contract.json) 覆盖 700 个分布、336000 个 core/geometry 比较、762750 个 tile/pointer 顺序比较；稀疏 AIC 与原生 AIV 的任务顺序一致。这是 CPU 调度证明，设备握手仍须运行验证。
- [最终产物](evidence/v3/frozen_artifacts.json) 确认 v3 完整重建与 v2 candidate/native、v1 planner 对象逐位一致，AIC `.text` 也与已证明的产物相同。当前实际编译实例为 `A8W4_TEMPLATE=1` 的 API 反量化，AIV:AIC 为 2:1；Vector 反量化类有源码补丁，尚无独立编译覆盖。

| 产物 | 大小 | SHA256 |
|---|---:|---|
| planner | 34560B | `f921e2d9c895aca5958177a17bd398f058c7a1d8910564bcd54c4dd98b51b770` |
| candidate GMM | 23488B | `bd08a12b15d14f44a2c63a2ff91ec2977a0ed2b62645ed7d43273ee2fb024970` |
| native control GMM | 19304B | `1e8a606d459bd4560b0e19efbb1894d98a1826bf0048eb9fa16d930b7d46ec0d` |
| private host adapter | 13392B | `02816eb5b23f93d7784258578a9aecf8df7dc35b4bc3feacb15af3020ad10ac8` |

native/candidate 的 SDK tiling 均为 2032B，逐位相同，SHA256 为 `5a596a726706e39d34be71b61a6ed563325c1d482953937050a7b8373e9a263c`，见 [tiling_byte_equality.json](evidence/v3/tiling_byte_equality.json)。CPU host 输入要求 `group_list_type=1`；生成 tiling 的 nominal 字段仍为 0。原 MSD 类直接按 counts 读取，该 nominal 字段未参与此解释。候选保留原 host tiling，使用新入口的 counts 契约，避免用该 nominal 字段消除稀疏分支。

## 输入、布局与 planner

固定 profile 为 row capacity6、E48、K2304、N5120、single tensor、quant group1，输出 BF16。x 为 `[6,2304] INT8`、counts 为 `[48] INT64`、token scale 为 `[6] FP32`。权重的 Torch INT32 逻辑形状为 `[48,2304,640]`，NZ storage 为 `[48,80,144,16,8]`；编译用 INT4 descriptor 为 `[48,80,144,16,64]`，逻辑 `[48,2304,5120]`。scale 的原 INT64 位模式按 UINT64 读取，形状 `[48,1,5120]`；assist 作为原 bias 参数为 `[48,5120] FP32`。

planner frame 共 128B，`uint32_t[32]`：

| word | 内容 |
|---|---|
| 0–3 | totalRows、activeSlots、status、totalExpertNum=48 |
| 4–27 | 六条 `{expert,rowBegin,rowCount,reserved}`，每条 16B |
| 28–31 | padding=0 |

inactive expert 为 `0xffffffff`。有效专家按原 dense counts 的专家顺序生成；counts 每项非负且总行数≤6。非法 count 令 status=1；status 非0会回退原 dense 调度，因此调用方仍必须提供合法的原生 counts。这不是输入拒绝通道，adapter 不读回设备 counts。原生数值路径保持两份 INT4 dot、各自 FP16 反量化，再 FP32 高低合并及 assist/token scale；所有舍入、激活和输出代码保持原样。

## Workspace、别名与 raw ABI

见 [workspace 契约](evidence/v3/workspace_contract/workspace_contract.json) 与 [runtime ABI 契约](evidence/v3/runtime_abi_contract.json)。生成 wrapper 首先 `SetSysWorkspaceForce(base)`；DAV2201 的 `GetUserWorkspace` 返回该系统 base+16MiB，并忽略传入形参，所以 raw 函数的第二次调用没有再加一次 16MiB。

| 相对原始 workspace base | 用途 |
|---|---|
| `[0,16MiB)` | SDK system reservation/KFC |
| `[16MiB,16MiB+128)` | device planner frame |
| `16MiB+128` 起 | 原 native user scratch，内部相对布局保持原样 |

SDK JSON 的原 workspace 为 112MiB，host adapter 要求至少 **117440640B，即 112MiB+128B**，并向 GMM 传原始 system base。实际编译的 KFC 参数为 core24、MIX_NUM2、MAX_MSG_COUNT64；复制原结构到普通 CPU 编译器测得 `sizeof(KfcMsg)=128`、另两种消息结构为64。清理 message 区上界786432B，UB消息区上界847872B，均小于16MiB。

[host_adapter.h](host_adapter.h)、[host_adapter.cpp](host_adapter.cpp)、[host_loader.cpp](host_loader.cpp) 借用已初始化的调用方 context/stream，不选卡、不分配、不同步。调用方先用 `gmm2_sparse_msd_pack_lists` 构造五个 32B 的单 tensor pointer-list，复制到各自 GM list 地址，随后才能提交 GMM。pointer-list 为 `{24,1<<32,0xffffffff,data_address}`，对应原生 `GetTensorAddr`/DAV2201 `ListTensorDesc` 格式。x/weight/assist/scale/y 均继续指向完整原生 tensor，所有 tensor/list/workspace span 必须互不交叠，且容量、对齐及地址加法须通过 guard。

实际 ELF `__CCE_KernelArgSize` 为 GMM 两个分支各72B、planner16B。GMM 的九个64位参数依次为 `ffts_addr,x_list,weight_list,assist_list,scale_list,counts,token_scale,y_list,workspace_base`；planner 为 `counts,frame_address`。FFTS/C2C 地址由调用方当前 context 的 `rtGetC2cCtrlAddr` 获取。native/candidate 均 blockDim24、schedule mode1；candidate 之前在同一 stream 以 blockDim1 提交 planner。计时须包含这次 planner 提交。

`gmm2_sparse_msd_open` 使用 SDK JSON 中的基础 kernelName，并要求 mixed 函数同时有 AIC/AIV 地址、planner 只有 AIV 地址。相关 ACL 声明和动态导出符号存在，私有 so 已通过 `--no-undefined` 链接。[30项 mock 契约](evidence/v3/host_mock_contract_result.json) 验证容量/别名/溢出拒绝、参数顺序、指针布局、stream 与提交失败传播；该测试没有链接或加载 `libascendcl/libruntime`。实际注册首次调用 planner 的 `aclrtBinaryLoadFromFile` 就被 MAGIC 选项拒绝，尚未执行 `BinaryGetFunction` 或 mix 配对，见 [运行终态](evidence/runtime_registration_p8_20261011/runtime_gate_status.json) 与 [精确SDK错误](evidence/runtime_registration_p8_20261011/runtime_precise_error.log)。v3 so/source/对象保留冻结，新 loader 仅在独立 `runtime_loader_v2/` 做 CPU 修复。

## CPU 复现与历史

在具备安装态 CANN9.1 的容器中，将本目录复制到一个新的 `gmm_lane_sparse_msd` 私有 build 根。以下命令只编译及运行普通 CPU mock；脚本固定两个 visible guard 为空，拒绝覆盖已有构建日志：

```bash
python source/build_cpu.py --root /work/gmm_lane_sparse_msd_build_repro --stage all
python source/prove_compiled_branch.py --root /work/gmm_lane_sparse_msd_build_repro
python source/verify_workspace.py --root /work/gmm_lane_sparse_msd_build_repro
```

`source/` 需包含本目录的脚本/CPP/header 及 `evidence/v2/compile_input_{native,candidate}.json`。`--stage planner|full-gmm|host` 可分项构建。planner target 为 `dav-c220-vec`，GMM native/candidate target 为 DAV2201/`Ascend910_9382`。完整成功日志在 [v3/logs](evidence/v3/logs)；失败历史在 [v1](evidence/v1)、[v2/branch_evidence](evidence/v2/branch_evidence)，大 `.i` 留远端。

历史包括错误 `-mcpu` 参数、nullable list checker、缺少 `param_name`、v5 nominal groupListType guard 被常量消除，以及 SDK 不支持 IR/assembly dump。v5 的成功编译仅作历史，不作为稀疏分支覆盖。v6 修正 counts 契约后的原对象与完整 v3 重建才是冻结候选。

OPC 加载 libruntime 并调用 `GetVisibleDevices` 进行只读枚举，空 visible guard 产生枚举错误，但编译继续成功；日志记录 destructor `WaitMonitorExit used count=0`。未命中显式 set_device、TSD、设备 malloc 或 kernel launch 日志，见 [runtime_enumeration_contract.json](evidence/v3/runtime_enumeration_contract.json)。该枚举不等于 NPU 初始化，没有解除 visible guard。

远端私有根为 `/work/gmm_lane_sparse_msd_build_20261011_v1`、`v2`、`v3_repro`，宿主对应 `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/` 同名目录。历史证据包133355B、最终证据包93287B、统一构建源码包14936B；补充 workspace 验证脚本单独传输，所有文件及包均低于1MB。原 `.i` 与完整原生源保留在远端，13份构建源码/模板的本地与远端 SHA 一致。

## 资源授权后的最小门

1. physical6仅已授权最小M1 capacity6数值门。每次启动前fresh smi/health、无进程、Health OK；进程双visible env为6，count1、logical0映射physical6，先核对Data注册及mixed配对。新device异常立即终止后续并清理，不换chip重试。
2. 用实际已 dispatch 的 x/counts/token scale 与完整 E48 NZ 参数，先核对 raw native control 对原 SDK API，再做 native A/A 和 candidate 配对。覆盖R0–6、专家0/47、六槽、混合count、量化边界；有效输出要求原生≤1ULP。设备 frame 的128B读回只放在 probe/capture 外，检查真实专家和rowBegin、planner写入可见性。
3. 检查原 `SwigluQuant→GMM2→MoE` consumer、capture/replay 的动态 counts/专家变化和 sparse 分支实际覆盖。无审计性能配对纳入 planner 成本；独立 probe 完成后仍需正式 TP8 整网门。

v3入口的File加载参数组合不受支持；独立 [loader v2](runtime_loader_v2/README.md) 的CPU链接、338项mock及Data真实注册/配对已通过，保留blob生命周期、ELF类型和runtime配对检查。HostArgs R0成功已冻结，当前交付没有GMM精度或收益结论。
