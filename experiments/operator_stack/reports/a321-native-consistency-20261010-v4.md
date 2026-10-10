# 正式TP8恢复：原生图与eager一致性控制

2026-10-10 · a3-21 physical chip8–15 · `feat/operator-stack-tp8-20261010`

**本轮已证明：不安装候选算子包装、不重建其他bank，原生整网图和eager路径仍未通过原A/A门槛。** 差异集中在decode，prefill路由完全一致，输出token相同，Top5集合与logprob不稳定。不能把此前叠加对照失败直接归因于新算子，也不能宣称正式精度或19ms目标达标。

## 恢复环境和比较方法

继续使用用户指定的chip8–15和正式checkpoint `v41-w4a8-engram-dr-vision-qrot-mtpq`，40层、hidden5120、384 experts、top6、W4A8_DYNAMIC、Engram int8和完整视觉模块。8个chip在启动前无设备进程，告警仍为已授权的 `80C98001`。没有reset、停止其他用户或清page cache。

上一轮暂停时中断的 `formal_native_graph_control_v1` 不算结果。本轮新建 `formal_native_graph_control_v2`；worker只保留审计路由导出修正，不导入候选bank包装。两次warmup后，生成一份reference，再重复同prompt三次。每次2048 token输入、47 token输出、40层top6；要求完整路由一致、Top5集合一致、最大logprob差<1e-3。

随后独立运行 `formal_native_eager_control_v1`，关闭整网图和静态核，其他正式配置相同。此时Engram自己的子图仍按生产默认开启，因此不能称为所有组件都无图。

## 本轮控制结果

| 路径/重复 | token相同 | prefill不同数 | decode不同token-layer数 | top6集合不同数 | 共同token最大logprob差 |
|---|---|---:|---:|---:|---:|
| 原生图/0 | 是 | 0 | 1343 | 851 | 0.562463760 |
| 原生图/1 | 是 | 0 | 1357 | 860 | 0.874831200 |
| 原生图/2 | 是 | 0 | 1371 | 876 | 0.874914169 |
| eager/0 | 是 | 0 | 1331 | 776 | 0.749944687 |
| eager/1 | 是 | 0 | 1362 | 854 | 0.874942780 |
| eager/2 | 是 | 0 | 1339 | 821 | 0.687136650 |

6组对照全部失败，路由shape均为 `[2094,40,6]`，ID范围和top6唯一性均通过。由于Top5集合不相同，表中的差值只统计共同token，不能当成完整Top5的通过指标。原生图首次偏离出现在第一条decode输入的layer3或layer1；eager首次偏离出现在layer4、layer1、layer6。

此结果降低了“仅由多bank共享内存池或整网图捕获引起”的解释优先级；没有排除Engram子图、原生数值归约、通信或状态生命周期问题。独立pool开关保持诊断用途，尚未采纳。

## 新增诊断与原理

- `formal_comparison.py` 统一保存紧凑失败证据；先检查原始路由值，避免int16收窄把越界ID伪装成合法值，并拒绝非有限logprob差。已用上一轮真实数据复现1380个decode差异、879个集合差异和1.124988556的共同token差值；同输入正例、越界和非有限负例均通过预期检查。
- 正式叠加benchmark在候选创建前，以及每个候选捕获前后增加原生A/A检查。这样可以定位状态从哪个阶段开始偏离，门槛未放宽。
- `formal_native_eager_probe_v1` 在同一组worker中先复现原生控制，再关闭Engram子图，保留相同hash/gather/dequant和正式表；比较关闭前后的A/A，排查该组件的图生命周期。
- 随后仅在真实eager decode中保留HC、384×5120路由投影的输入与输出，模型forward中不做D2H。请求结束后，用固定输入重复原生方法5次，检查重复值与实际forward值。这是单算子数值诊断，不是新的计算算法或性能计时。

组合诊断正在运行，Engram子图与单算子重复结果尚待完成。原生控制请求不安装这些探针，避免用有探针的结果代替原生A/A证据。

## 性能与交付边界

整网精度尚未验收，本轮没有有效的正式 `(ms/step,A,token/s)` 或新加速比，没有部署未通过的候选服务。正式96次Indexer逐位检查与BF16激活覆盖0的上一轮结论继续成立，不能替代本轮整网门槛。

源码与控制证据已提交；整包自检通过。待组合诊断完成后追加结果并发布报告，然后按首个实际偏离点推进修复和性能计数采集。目标保持active。
