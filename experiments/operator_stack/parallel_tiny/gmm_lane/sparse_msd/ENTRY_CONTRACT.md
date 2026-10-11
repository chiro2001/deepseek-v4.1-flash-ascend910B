# 授权后 GMM2/down probe 的入口

当前 NPU 启动权限恢复为0。冻结 so 为远端 `gmm_lane_sparse_msd_build_20261011_v3_repro/libgmm2_sparse_msd_probe.so`。本次临时 physical8 注册门证实单visible设备、physical8→logical0和合法context，随后首次 `BinaryLoadFromFile+MAGIC` 返回107000，尚未查询kernelName或启动planner/GMM。context/finalize清理成功，fresh smi确认physical8已释放，见 [运行终态](evidence/runtime_registration_p8_20261011/runtime_gate_status.json)。v3产物保持冻结；以下调用是v3契约记录，加载修复在独立 `runtime_loader_v2/` 做CPU准备，后续运行需主线程新的资源消息。

先完成一个只打开对象、核验函数配对、关闭对象的小门：

```cpp
Gmm2SparseLibrary *library = nullptr;
int status = gmm2_sparse_msd_open(
    "planner_cpu_rebuild.o",
    "op_build_native/Gmm2NativeMsdDown_control_v6.o",
    "op_build_candidate/Gmm2SparseMsdDown_candidate_v6.o", &library);
// status 必须为0；SDK错误或 AIC/AIV 配对失败应直接报告。
Gmm2SparseFunctions functions{};
if (status == 0) status = gmm2_sparse_msd_functions(library, &functions);
// 获取/关闭不提交 GMM；BinaryLoad 本身需要调用方合法 NPU context。
gmm2_sparse_msd_close(library);
```

`open` 使用原 SDK JSON 中的基础 kernelName。当前物理 ELF 符号有 `_mix_aic`/`_mix_aiv` 后缀；基础名的 runtime 配对解析仍待验证。已知更早的阻断是 SDK `binary_loader.cc:131` 拒绝 file loading 的 MAGIC 选项，原文为 `RT_LOAD_BINARY_OPT_MAGIC does not support in load from file.`，并非 kernelName 失败。本次已按失败终态停止，没有重试或放宽配对检查。

数据 probe 需重新 open 并保留句柄至 stream 完成。调用方提供 [Gmm2SparseBuffers](host_adapter.h) 的全部 GM span，使用原完整 E48 参数与实际已 dispatch 输入；权重最少283115520B（270MiB）、scale1966080B、assist983040B、workspace117440640B、五个32B list，加上很小的 x/counts/token_scale/y。共享同一个足量 workspace 做顺序 A/B 可控制额外内存；candidate 前必须重新运行 planner，因为 native scratch 从原 user 边界开始，可能覆盖旧 frame。

在 capture 外安装一次稳定的五个 pointer-list，再在同一 stream 提交：

```cpp
uint64_t lists[20];
status = gmm2_sparse_msd_pack_lists(&buffers, lists);
// 将每条32B list复制到 buffers.{x,weight,assist,scale,y}_list 的 GM地址；
// 每条data_address是原 tensor GM地址。保证复制完成且host源仍有效。
// buffers.ffts_address由当前context的 rtGetC2cCtrlAddr 获取。
if (status == 0) status = gmm2_sparse_msd_launch(&functions, &buffers, stream, 0); // raw native
// 检查status、同步并归档原生输出，然后再提交candidate。
if (status == 0) status = gmm2_sparse_msd_launch(&functions, &buffers, stream, 1); // planner + sparse
```

所有提交错误必须逐项检查；调用方在各次输出比较前同步该 stream。adapter 的提交成功只表示 runtime 接受任务，不证明设备计算成功。把真正的数据运行错误与 loader 错误分别归档。

valid profile 为 `row_capacity=6, experts=48, k=2304, n=5120, group_list_type=1`。counts 为原生 `[48] INT64` 非负计数、总行数≤6，frame 仍保留 E48。设备 status 非0回退原 dense 调度，不用于拒绝非法 counts；数据必须由原 native dispatcher 提供，或由图外 CPU oracle 生成合法的 probe counts。

核验顺序：原 SDK API与 raw native、native A/A、candidate配对≤1ULP，再检查 planner 的128B frame、实际 MoE consumer、capture/replay覆盖。R0–6、专家0/47、六槽、混合count和量化边界都需覆盖。frame/输出读回和日志放在 capture 外。最后进行包含 planner 成本的无审计性能配对；本M1候选可报告符合 profile 的 draft/target步，当前不能声明多token verifier或完整TP8 DSpark/MTP的覆盖与收益。
