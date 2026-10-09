# 单 token clamped SwiGLU 融合

基于既有 `both` HC/router 图，routed 激活链为两个
`Slice→Clip→ViewCopy` 加 `SwiGlu`，合计1254.896 μs/步。
shared 的逐操作激活另为459.818 μs/步。这些是采集态覆盖成本，尚非新收益。

目标为 DAV_2201 / Ascend910_9382 的 BF16 `[2,512]` 和 `[1,512]`。
小输出使用每行一个 Vector program，直接加载 gate/up，融合 clamp 与激活。
保留 routed 原地修改输入的副作用；shared 不修改输入，并复现各 BF16 中间 cast。
保留真实 limit/alpha/beta；禁止通过低精度近似或改变激活定义获得收益。

已按 cannbot `triton-op-designer` / `triton-op-coding` 读取 sketch DSL、
elementwise、基础/API与案例；硬件依据为 npu-arch 的 DAV_2201 参数。
参考 elemwise-concat 的直接切片加载，以及 elemwise-zeros 的少核策略。
本 checkout 无 `.claude/template` 类别经验文件。草图在 `clamped_swiglu.sketch`。

## 初始静态检查

- 所有 load/store 带mask，输出索引与输入两半映射分开；各元素只有一个写者。
- 无核间同步、原子、reduce；无需双缓冲，UB tile小于本机192 KiB。
- 默认BLOCK=256，非大shape；无GPU专用warps/stages参数。
- kernel无return/break/continue/while；host只做元信息、分配与启动。
- 通过显式stride保持输入副作用，未用contiguous复制替代输入。
- 禁用FP融合，以保留shared的运算和舍入边界。
- 未做数据依赖D2H或把tensor值烘焙成constexpr。

精度门槛在测试前固定：特殊值分类和输入clamp副作用必须一致；
native相对比较使用全元素BF16混合容差，并同时记录逐位一致性、ULP差和
CPU高精度误差。模型接入必须另过非恒定权重、实际路由和logprobs审计。

第一版flat索引通过139个精度用例，但routed图时延224.365 μs，远慢于native
25.013 μs，已拒绝并保留源码/数据。v2调整行/tile划分（BLOCK 1024→256，
目标routed grid 1→2）：每个program固定行，连续访问两个半行。
没有删除输入修改副作用；编译器具体降低原因尚未通过指令trace确认。
