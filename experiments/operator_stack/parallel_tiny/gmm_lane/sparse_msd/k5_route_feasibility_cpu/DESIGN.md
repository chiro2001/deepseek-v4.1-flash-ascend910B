# 两role候选的CPU协议设计

当前机制保持低优先级：M1 candidate对raw只有e47的3/9 case更快，真实完整接受target输入route子集中，精确对应R1/R3/R6单个e47的投影仅32/13440（0.238%），去重为10/4480（0.223%）。这些只是模式重合，没有实际GMM consumer或K5时延证明。SDK对prebound host路径的差距不能作为已入图GMM收益。此目录没有构建或挂接K5设备候选，NPU调用/权限均0。

| role | x/y容量 | 本地E/counts长度 | global E/top-k | 每step GMM2 | CPU协议frame前缀 |
|---|---:|---:|---:|---:|---:|
| target | 36 | 48 | 384/6 | 40 | 640B |
| draft | 15 | 16 | 128/3 | 3 | 384B |

上述几何以全部8rank的正式profile step9为准。draft的128/top3通过实际`patch_speculative_config.py` override和`draft_model_config.hf_text_config`消费者源码绑定；CPU仅执行该函数AST，registry是标明的stub。target和draft的communication cache必须按num_experts、num_local_experts、top-k等实际shape分开，当前实际源码已经包含此guard。

未来设备实现需分别冻结两role的原生SDK tiling、完整weights/scale/assist、counts INT64/type1、token scale FP32、ND activation、NZ权重和BF16输出。旧M1 capacity6/E48 objects和2032B blob不能直接用于任一role。新增工作区需容纳原生workspace以及对应frame前缀，system base保持512B对齐，candidate math scratch从frame之后开始；原生control保留自身布局。frame前缀数值是本CPU协议设计，不是已提取的SDK workspace或已验证的设备ABI。

`planner_protocol_cpu.py`的8-word header为magic/version/E/capacity/R/active/status/reserved，slot为local expert ID/row_begin/row_count/reserved。target最多36个非零slot，draft最多15个；不能沿用6槽截断。planner保留原local expert ID和完整参数，并由同stream顺序提交后candidate读取。所有17原生数学函数、两份INT4独立dot和FP16舍入、FP32合并、assist/token处理保持原样；只扩展AIC调度及入口frame检查，AIV保持原生全groups路径。

candidate必须在任何MMCompute或handshake之前完成整frame验证，逐项与原counts的非零专家、完整前缀和行数吻合。header错误、slot超界/重复/乱序/缺项、错误offset/count、非零reserved或padding、frame长度错误，均回退对应role的**全部原生groups**（target 0–47、draft 0–15），不能执行一部分稀疏slot再回退，也不能把坏专家ID取模或截断。native counts自身非法时属于原输入契约错误；fallback不能把negative、overflow、超容量或错长度counts合法化。

CPU门完成1266项调度代数等价检查、6680个坏frame全native回退反例、8个非法native输入拒绝，并验证13440条实际target route投影。测试使用多个singleM/singleN/core组合，保留MSD两份展开的原tile/core分派顺序。这不是设备fallback、数值或capture证明；真实role tiling仍须独立绑定。

下一实证优先项是实际prepack设备GMM counts与activation消费者：本数据源只覆盖完整接受target输入route，不能补齐其他6轮或draft的任何route。应在capture外绑定完整GM生命周期及只读参数seal，分别做native控制/SDK/candidate数值门；事件测量先通过数值和owner/工作区门，再在同实例同stream配对，计入planner且保留原kernel内部KFC清理。正式graph场景需另证实际已入图执行和关键路径，profile duration sum存在重叠，不能把step9的target 1.498976ms、draft 0.1058675ms当可回收E2E预算。
