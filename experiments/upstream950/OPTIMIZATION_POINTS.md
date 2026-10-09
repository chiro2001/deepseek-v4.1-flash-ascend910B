# 950 已有优化向 A2/A3 迁移：候选与证据

日期：2026-10-09。结论：优先验证稀疏专家列表与 SMLA 的索引/页表批量预取；Indexer 后处理融合、输出投影融合和权重面板优化作为后续独立切口。UB→L1 直写属于需要先验证硬件与同步能力的候选。

本次完成了独立 A3 tiny 环境、服务冒烟和 GMM 接口验证；没有修改原优化线，也没有将这些候选接入模型。本报告不预测端到端收益。下文热点成本来自另一条线已有的 chip4 `both` 采集态数据，只用于排序，不是新 chip5 的性能测量。

## 1. 来源与现有基线

上游已固定到以下提交，完整 URL 见 `results/upstream_revisions.json`：

|仓库|提交|
|---|---|
|cann-recipes-infer|`4c3d1e1258053c4d810322198a7fd1722c89897e`|
|cann-recipes-train|`a446bab76f2da1e18fe3e16f8d609b0cc6c34779`|
|ops-transformer（recipes 引用的算子源码）|`59dd0c236a6d57f03d26474d8c18c0df1bdc911d`|

主要参考：

- [V4.1 CANNBot-DSL 算子优化](upstream/cann-recipes-infer/docs/models/deepseek_v4_1/deepseek_v4.1_cannbotdsl_operator_guide.md)：Attention Prologue、Indexer、QLI/QSLI、SMLA 和输出 Epilogue 的实际优化与单变量对照。
- [950 低时延 TP 实践](upstream/cann-recipes-infer/docs/models/deepseek_v4_1/deepseek_v4.1_low_latency_tp_guide.md)：稀疏 Group List、权重亲和切分及框架优化；其中 `Future Plan` 项不能当作已落地效果。
- [950 AutoFuse](upstream/cann-recipes-train/docs/llm_pretrain/ascend950/ascendc_autofuse.md) 与 [A3 V4 AutoFuse 实践](upstream/cann-recipes-train/docs/llm_pretrain/deepseek-v4_torchtitan_npu_autofuse.md)：证明自动融合思路不局限于 950；训练整网收益不能外推到 tiny decode。
- [上游融合分析技能](upstream/cann-recipes-infer/.agents/skills/model-infer-fusion/SKILL.md)：按模块匹配参考链路，并查本机接口约束。

新环境是 `a3-21` 的物理 chip5，Ascend910_9382 / DAV_2201。CANN 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4、vLLM 0.27.1；框架 HEAD 为 vLLM `6e448d0e…`、vllm-ascend `e43cf1e9…`，213 份安装态相关源码已只读取证并记录 SHA256。

tiny 为 BF16/TP1，40 层、hidden5120、64 heads、head_dim512、SWA128、topk512；`q_lora_rank=512`、`o_lora_rank=512`、8 个路由专家/top2、中间维256。因此 attention/KV 主维度有参考价值，投影低秩维及 MoE 形状仍与上游生产模型不同。950 文档中 QA rank1280、输出投影 rank1024、384 专家/中间维2304 的代码不能照形状直接替换。

## 2. 候选优先级

|优先级|切口|现有热点范围|迁移状态|最小实验|
|---|---|---|---|---|
|P0，低成本先试|GMM 稀疏专家列表 `group_list_type=2`|GMM1+GMM2 约1.816 ms/步|A3 BF16 ND/NZ 接口与输出已验证；A2有官方文档/源码支持，未实机验证|先比 GMM，再将列表生成计入完整路由链|
|P0，重点算子开发|SMLA 索引/页表批量预取到 UB|SMLA 约1.732 ms/步|已在 arch22 源码定位逐项 GM 读取；A2/A3 候选|仅改变地址取值位置，保留原 gather 和计算|
|P1|Indexer K 的 Norm→RoPE→INT8量化→cache融合|4 个 K 源层；Q端8个索引源层|融合范围明确；需实现保持当前 INT8 精度的 Vector 后处理|先做 K 后处理，保留投影输出的 BF16 边界|
|P1|Attention inverse RoPE + 八组 wo_a 输出前处理|wo_a 约1.526，wo_b约1.442 ms/步；不能把全段当可回收时间|950有融合先例；本机八组路径未融合|先融合 inverse RoPE 与第一层投影输入准备，再测 wo_a tiling|
|P2|Q/KV联合投影、q_b权重面板预排与消费顺序预取|q_b约1.169 ms/步；QA/KV另计|需按 BF16/INT8 重做布局和资源预算|固定矩阵乘语义，只变面板布局或预取一项|
|P2，先验证能力|SMLA gather 的 UB→L1 片上交接|与同一段 SMLA 重叠，不能和寻址收益相加|950有实测；A2/A3的同核混合执行、直写与同步尚待最小验证|先编译/运行最小交接用例，验证合法性后再做算子|

## 3. P0：稀疏专家列表是最容易先验证的迁移项

上游 `models/deepseek_v4_1/models/modeling_deepseek.py:562` 构造路由参数，选择 `expert_tokens_num_type=2`；随后 GMM 使用 `[expert_id, token_count]` 二维表及 `group_list_type=2`。目的在于少 token 时跳过空专家遍历。

本机 `vllm_ascend/ops/fused_moe/token_dispatcher.py:489` 仍请求 count 输出、`group_list_type=1`；`routed_experts.py:209` 和 `:272` 将该值传入两次 GMM，已有可替换的参数通路。

[GroupedMatmulV5 文档](upstream/ops-transformer/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) 的 A2/A3 部分明确支持类型2，非零组必须前置，并按 expert ID 有序，零组放在后面；group size总和需满足输入行数约束。`grouped_matmul_tiling.cpp:1314` 同样明确检查 910B/910_93 支持。当前 torch_npu docstring 只写0/1，属于文档版本差异，不能据此否定实际内核。

本次在新环境运行了兼容性验证：

- BF16，8专家；GMM1 `[2,5120]×[8,5120,512]`，GMM2 `[2,256]×[8,256,5120]`。
- 两组选中专家 `[2,5]`、`[0,7]`，随机非恒定权重。
- ND及FRACTAL_NZ各4项；NZ实际格式编号29；`split_item=2`与当前模型一致。
- 稀疏表与完整表计算结果全部逐比特一致，最大绝对差0。

证据：`results/sparse_grouplist_compatibility.json`、`results/sparse_grouplist_nz_compatibility.json`。它们只证明当前 A3 的接口和这8个用例通过，没有证明加速、完整模型正确性、量化路径或A2实机结果。

下一步必须把表生成耗时算进去。优先让原路由阶段直接产出正确有序的稀疏表；若要追加sort/重排task，收益可能抵消。不得用D2H读取expert IDs来构造表。当前tiny只有8专家，384专家的生产收益与tiny可能明显不同。

## 4. P0：SMLA批量寻址与索引预取

950文档的 SMLA 对照把物理地址从GM逐项读取改为成块预取至UB；另一连续TND实验进一步直接预取索引。QSLI则先搬页表与候选数组，由Vector批量生成地址。可迁移的核心是减少“取出索引→查询页表→发出KV搬运”的串行等待。

本机源码中的直接证据：

- `sparse_flash_mla_csa_block_vector.h:533`：`topkGm_.GetValue(...)` 逐项读取稀疏索引。
- 同文件`:549`：`cmpBlockTableGm_.GetValue(...)` 查询物理页。
- `ProcessVec0L`循环调用这些函数之后才调用`CopyInKv`。
- 已有profiling中SMLA具有高Scalar占比；它支持优先调查这个位置，但不能单凭占比断言可回收多少时间。

第一版仅将当前tile所需的索引和页表条目预取到UB，保留现有地址计算、双缓冲gather及softmax，方便归因。随后才比较Vector批量页号/gather与下一tile预取。

当前是Paged KV，不能套用连续TND的 `base+index` 地址公式。必须保留页表、物理stride、负索引、tail、有效长度、重复索引、跨页和64bit地址语义。不能缓存上一decode步的动态索引。

**已存在的优化要保留**：本机 `CopyInKv:594` 已经在符合条件时用双burst合并两个KV块，异常stride/tail时回退。成对搬运不是本轮新增机会，不应重新实现后宣称迁移收益。现有SWA路径也已有部分UB索引读取，需把改动局限于仍逐项读取GM的路径。

## 5. P1：Indexer后处理与输出投影

### Indexer

上游 `indexer_prologue_k` 以独立Cube投影和一个Vector kernel完成Norm、RoPE、量化、cache写入。它仍保留投影结果落GM，并非整条投影都在单kernel完成。K数据和scale直接写cache，消除额外散写。

本机 `models/deepseek_v41/indexer.py:97` 的链条是 `wk→RMSNorm→partial RoPE→dynamic INT8 quant→K scatter→scale cast/scatter`。可先把这段Vector后处理与两份cache写入融合。Q端在`:134`做投影/RoPE，另算weights并调用已有`quantize_indexer_query`，后续再考虑联合调度；已有query量化kernel无需重复开发。

950使用MXFP4，本机使用INT8。迁移融合范围时继续使用本机INT8定义、scale精度、cache布局和slot映射。不得通过切换到FP4来声称等价加速；RoPE配对、BF16中间舍入及量化阈值均需对齐。

### Attention输出投影

950的`attn_epilogue`融合inverse RoPE、输入量化、八组MM1、中间量化和MM2，并优化权重面板预取。现有 `dsa_v41.py:602` 独立执行inverse RoPE，再调用 `_forward_o_proj`。tiny真实wo_a为八组 `[1,8,4096]×[8,4096,512]`，wo_b为 `[1,4096]×[5120,4096]`。

本机`V41_O_PROJ_2D`仅在`n_local_groups==1`进入，TP1 tiny是八组，不能把开关生效当作本轮方案。先做inverse RoPE与分组输入准备融合/专用tiling；若继续跨wo_a/wo_b融合，需保留wo_a输出的BF16舍入。不能将两层权重代数合并。

## 6. P2：权重流水与片上交接的迁移边界

950 Attention Prologue已验证联合QA/KV输入复用、扩大DMA面板、四Buffer循环、归一化阶段提前预取后继权重、连续panelpack等措施。本机QA与KV仍从同一输入分别投影，q_b是较重权重搬运热点，可分别试联合投影或权重面板重排。

这些设计需按本机BF16/INT8字节数和24 AIC/48 AIV重算。DAV_2201为UB192 KiB、L0C128 KiB；950文档方案使用UB256 KiB，部分L0C预算到256 KiB。QLI的两份M96×N256 FP32结果已需192 KiB，不能照搬到本机L0C。面板粒度、buffer深度、bank错位和TopK直方图的统计边界均需重新验证。预排必须一次性进行并随权重版本失效，不能每步转换布局。

SMLA片上交接方面，本机`CopyOutMrgeResult:651`将gather后的KV写GM，Cube侧`csa_block_cube.h:467`再读入L1。950的UB→L1实验移除了这个中转。但其Vector指令、混合kernel的共享L1访问、NZ转置和跨核通知不能视为已获A2/A3支持：现有通用`ub_to_l1`封装本身不足以证明相同执行路径合法。先做DAV_2201最小编译/运行验证，再决定直写或保留GM中转的流水改造。

950的4-buffer、UB错位和“更多buffer”不是统一配方；文档中8-buffer方案反而更慢。本轮先定位首发/排空/供数，再选择buffer数量。

## 7. 已筛除或需单独开展的项

|950方案|当前结论|
|---|---|
|原生MXFP8/MXFP4、A4W4 Cube|不能等价替换本机BF16/W4A8；当前mixed-quant API虽列A2/A3支持，但仅是TurboQuant模式3，不能据此调用950的FP8/FP4模式1/2|
|MegaMoE整核|当前上游接入prefill、要求EP>1且TP=1、W4A8 MXFP4；不符合本tiny TP1/BF16本地路径。激活融合由另一条线推进|
|CCU展开/UBC/URMA Engram offload|依赖950通信硬件及完整多rank/Engram环境；本tiny无法验证|
|DSpark、FP4 KV、减少Sinkhorn迭代|改变功能、精度或实验范围；不属于当前语义保持的算子候选|
|已有npugraph_ex、QLI无candidate路径、HC/router|已具备或已有优化线验证，保留作基线，不作为本轮新收益|
|训练FSDP/反向/mHC backward融合|没有当前推理链中的对应计算；仅借鉴AutoFuse的Vector融合方法|

`npu_mla_prolog_v3`虽在本机存在，内置文档要求Hcq1536、qk-nope128、独立Wuk吸收等旧MLA语义，与本tiny q_rank512、head_dim512且末64维RoPE的V4.1不同，不能直接替换。Q/KV融合应按V4.1重做，或在后续核验新的V4.1专用接口。

## 8. 下一轮实验与采纳门槛

建议先用低成本的稀疏Group List验证建立新泳道A/B流程，再投入SMLA寻址优化。新候选使用独立命名空间、独立.so或进程内patch，不覆盖镜像框架源码。

1. 同一chip5、同一模型进程保存native/candidate图，单变量交错测量；避免chip4并行压测时的卡级功耗/温度干扰。
2. 保持权重、请求、KV状态、CPU绑定一致，计时关闭profiler和审计；JIT/图捕获置于预热之外。
3. 非恒定权重和数据对齐，验证边界、图回放、输出生命周期，再进行模型输出/路由审计。
4. 性能报完整`(ms/step, A, tok/s)`；本tiny关闭DSpark，A=1。单算子更快和task变少均不代替端到端证据。
5. 单独在A2上验证后才报告A2收益，再进入真实TP8/W4A8/长上下文场景。

服务身份和32输入/16输出token冒烟通过；图捕获日志确认router40次、HC80次、物理可见设备5。原chip4容器仍保留。完整实验状态见`progress.md`。
