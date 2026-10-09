# SMLA UB→L1 直写能力门禁：当前标准 A3 路径不适配

目标是迁移950已测的 Vector gather 后直接写共享L1、通知Cube消费的通路，移除UB→GM→L1中转。当前A3为Ascend910_9382 / DAV_2201，CANN9.1.0。门禁结果不支持在当前标准SDK接口下实现相同的直写通路，因此保留已通过30case和整网审计的原生SMLA GM中转路径。

安装态 `tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:123` 的 `DataCopyUB2L1Impl` 明确标为software-emulated；代码先 `GetKfcClient()->AllocUB` 获得GM workspace，然后 `copy_ubuf_to_gm`，再发送KFC消息从GM拷到L1。ND→NZ和DataCopyPad同样为软件中转。`dav_3510`分支则在对应混合核配置下调用实际 `CopyUbufToCbuf`。公开DataCopy/TSCM函数名不等于片上直写。

最小编译核验使用同一安装态SDK/Bisheng、C++17与标准include路径：

|检查|目标|结果|
|---|---|---|
|纯GM→UB→GM控制用例|dav-c220-vec|编译成功|
|950 `CopyUbufToCbuf` 原语|dav-c220-vec|未声明，编译拒绝|
|同一原语|dav-c220-cube|未声明，编译拒绝|

控制用例成功排除了基本include/类型/编译目标配置错误。未启动不受支持的直写kernel，也没有虚构候选性能A/B；此方向在能力门禁处停止。结论只覆盖当前标准SDK路径，不把未经验证的非公开指令、自定义混合核或其他硬件配置判为绝对不可能。

复现：`scripts/probe_ub_l1.py`。完整编译命令、diagnostic、SDK源码路径、行号摘录和SHA256在 `evidence/final/ub_l1_capability.json`；大源码/编译产物仅留独立远程build/ub_l1，不覆盖框架或SDK。
