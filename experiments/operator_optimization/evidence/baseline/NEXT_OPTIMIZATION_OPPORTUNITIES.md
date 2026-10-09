# 下一轮优化机会：基于优化后 tiny 的热点与源码

日期：2026-10-09。当前基线为已接入 HC + router 的 `both`：无 profiler decode 中位 **26.122 ms/步、38.282 token/s**。

本次只重新分析已有20步 matched profiling，并只读核对当前容器源码；没有重启服务、修改运行中的模型或新增NPU测试。以下时间均为**已有采集态累计时间**，用于排序和确定覆盖范围，不是新增实测收益，也不能直接当作可回收墙钟。

**推荐先做专家激活融合，再验证现有 Q/KV 多流重叠路径；之后针对 tiny 的8组 wo_a 开发或选择合适的分组 GEMV。** MoE路由与GMM适合作为后续一条完整优化链推进。SparseFlashMla值得深挖，但需先补逐核等待与控制流证据。

## 1. 优化后的热点重新排序

已按Type、shape、core重新聚合 native/both，两份CSV均为20步，分别43960/45560条，Device_id=4；SHA256与上一轮验收一致。完整结果保存在 `results/opportunities/optimized_hotspots.json`。

|位置|both调用/步|采集态累计 ms/步|观察|
|---|---:|---:|---|
|wo_a，TransposeBatchMatMul|40|1.526|8组、小M，MTE2约64.4%，MTE1约28.3%|
|wo_b，`[1,4096]×[5120,4096]`|40|1.442|MTE2约89.2%，大权重搬运|
|routed GMM1|40|1.246|仅4个Cube block，MTE2约81.7%|
|q_b，`[1,512]×[32768,512]`|40|1.169|MTE2约86.8%，大输出投影|
|`[1,5120]×[512,5120]` BF16|121|1.149|聚合包含不同模块；不能全部归为Q/KV|
|lm_head|1|1.014|MTE2约99.5%，大权重流式读取|
|SparseFlashMla，合并所有shape|40|1.732|主要shape的AIC Scalar约60%，Cube约1.6%，ICache miss约21%|
|专家clamp链中的ViewCopy|80|0.776|每次约9.71 μs，48个Vector block，Vec约0.27%|
|新HC project_hf32|80|0.717|AIV Vec约40.1%，MTE2约36.2%|
|HcPost|80|0.589|AIV Vec约62.8%，已有实质Vector工作|
|routed GMM2|40|0.570|24个Cube block，Scalar约59.9%，Cube约4.0%|
|新HC finish_hc|80|0.540|5个Vector block；保留完整20次Sinkhorn|
|MoeInitRoutingV3|40|0.461|48个Vector block，Vec约0.33%|

低利用率不是同一种问题：ViewCopy和单token路由有明显固定开销；GMM需检查小M tiling和核分工；大投影主要受搬运影响；SparseFlashMla的Scalar/ICache特征需要进一步解释。

## 2. 优先级与实验成本

|优先级|机会|当前覆盖时间/范围|第一项实验|成本与适用范围|
|---|---|---|---|---|
|P0|routed clamp + SwiGLU融合|**1.255 ms/步，280个task/步**|将两个切片clamp和SwiGLU合成一个kernel|中等；tiny可先实现，生产还需量化/padding验证|
|P0后续|shared expert激活融合|约0.460 ms/步，280个task/步|按实际alpha/beta和BF16舍入语义融合|中等；与routed分开验证|
|P1，低成本先试|已有Q/KV多流重叠|Q/KV前处理链；收益取决于实际重叠与事件开销|`multistream_dsv4_dsa_overlap` OFF/ON匹配A/B|低；框架已有实现，但当前配置关闭|
|P1|8组wo_a单token分派/tiling|1.526 ms/步|同shape比较分组GEMV及现有TransposeBatchMatMul|中到高；当前tiny不能直接用已有单组2D开关|
|P1后续|单tokenMoE路由简化|TopK + InitRouting + Unpermute合计0.763 ms/步|保留路由语义，减少通用sort/permute准备|中到高；TP1/top2特化，迁移生产需重新设计|
|P1后续|tiny GMM1/GMM2小M路径|两者合计1.816 ms/步|检查GMM1四核分派、布局及选中expert的GEMV|高；tiny expert维度256与生产量化形状不同|
|P2|HC权重HF32预处理、finish/RMS相邻融合|新HC合计1.257 ms/步；邻接norm另计|先缓存静态权重HF32转换，再单独试finish+norm|中到高；需保持原舍入边界和全维归约|
|P2，先补证据|SparseFlashMla/Indexer控制开销|FlashMla 1.732，Indexer 0.340 ms/步|同shape逐核Default/MemoryDetail，定位等待和长尾|高；有较强生产相关性，尚未证明与HC同因|
|P2|metadata和图外小操作|五个AI_CPU metadata task合计0.271 ms/步|融合坐标准备，检查metadata生成与消费的等待|中到高；已有跨层共享，不能简单重复缓存旧内容|
|P3|wo_b/q_b/lm_head持续搬运|约3.624 ms/步|验证实际HBM速率、tiling、布局和分派|中到高；算术简化空间较少，不能假定低带宽|

覆盖时间并非收益上界的硬件实测，更不是可加的端到端提速预测。相邻融合和整条MoE优化可能覆盖同一段代码，累计评估时需去重。

## 3. P0：专家激活链是最明确的新机会

### 3.1 routed路径：原因已经对应到源码

当前 `ops/fused_moe/moe_mlp.py:91` 的默认SILU分支：

```python
gate, up = hidden_states.chunk(2, dim=-1)
gate.clamp_(max=swiglu_limit)
up.clamp_(min=-swiglu_limit, max=swiglu_limit)
hidden_states = torch_npu.npu_swiglu(hidden_states)
```

两个原地clamp作用于视图。已有trace中，**全部1600个ViewCopy都紧接 `Slice → ClipByValueV2`**。每层在GMM1和GMM2之间执行：

```text
GMM1
  → Slice → Clip → ViewCopy
  → Slice → Clip → ViewCopy
  → SwiGlu
  → GMM2
```

|组成|调用/步|累计 μs/步|
|---|---:|---:|
|Slice，输入 `[2,512]`|80|135.153|
|Clip，输入 `[2,256]`|80|191.125|
|ViewCopy|80|776.436|
|SwiGlu，输入 `[2,512]`|40|152.182|
|合计|280|**1254.896**|

ViewCopy只处理1024个BF16元素，却启动48个Vector block，Vec比例约0.27%。这更像视图写回、通用控制和任务固定开销；不应先通过增加UB buffer优化。

候选直接读取原始gate/up，以相同limit执行clamp和SwiGLU，输出 `[2,256]`，避免中间切片复制与写回。如果每层替换为一次调用，该段从7个task变为1个，减少240个task/步；kernel数量只是结构目标，采纳仍取决于无profiler墙钟。

必须验证：clamp正负边界、limit的BF16转换、极值、非恒定数据、输入视图的stride；还要确认原地输入修改没有其他活跃消费者，必要时保留相应副作用。生产路径另查pad slot、expert_map、量化输出与scale，不能用tiny未填充路径代替。

### 3.2 shared路径：单独处理alpha/beta和中间舍入

模型 `deepseek_v4/model.py:247` 选择 `SiluAndMulWithClamp`。当前图还有 `[1,256]` 的Clip、Muls、Sigmoid、Mul、Add链，合计约459.818 μs/步：

|组成|调用/步|累计 μs/步|
|---|---:|---:|
|Clip|80|175.517|
|Mul|80|119.870|
|Muls|40|50.269|
|Sigmoid|40|54.830|
|Add scalar|40|59.332|

通用定义为 `gate * sigmoid(alpha * gate) * (up + beta)`，前面分别clamp gate/up。`alpha=1, beta=0`只是默认值；在实际模型里读取配置并保留。当前trace保留了这些逐项操作，不能仅凭默认值删除路径。

对BF16，融合成全FP32计算再最终cast可能改变原逐操作舍入。第一次候选应明确复现必要的中间cast，验证后再判断是否有可接受的等价优化。routed与shared应独立A/B，分别确认收益，再组合。

## 4. P1：先用已有多流前处理做低成本实验

`attention/dsa_v41.py:319` 已实现 `multistream_preprocess`，forward在约589行按 `multistream_dsv4_dsa_overlap`选择路径。当前tiny配置为false。

代码刻意串行Cube matmul，并尝试使另一侧Vector工作重叠：

1. Q_a Cube matmul与独立KV量化重叠。
2. Q norm/量化与KV Cube matmul重叠。
3. Q_b Cube matmul与KV norm、RoPE、cache store重叠。

本次BF16 tiny未必能从量化重叠获益，主要观察后两阶段。源码已设置阶段事件和最后join，无需先重写一套前处理。

风险是单token下事件、排队开销超过重叠收益，或者两路争用MTE/L2；两个Cube matmul也不能假定独立满速并行。需要保留固定权重和请求，只切换配置重新捕获图，OFF/ON交错测量。图bank必须保存完整的事件/workspace/handles，不能只切换主图。

若该路径无收益，再考虑把同一输入的wq_a与wkv权重按输出维拼接。源码 `_project_q_kv` 中两者是独立投影，可共享一次输入读取和一次分派；但121次同shapeMatMul还包含shared expert等模块，不能把其全部1.149 ms计为Q/KV收益。拼接方案需保持q_norm、kv_norm和q_b的依赖以及原BF16输出边界。

## 5. P1：wo_a应按当前8组设计

当前 `wo_a` 的真实采集形状为 `[1,8,4096]×[8,4096,512]`，每步40次、单次约38.15 μs。

源码已有 `V41_O_PROJ_2D` 开关，但 `attention/dsa_v1.py:1835` 明确要求 `n_local_groups == 1`。本次TP1 tiny是8组，**打开该开关不会进入2D路径**。生产TP8可能有单组，但需在对应部署另行核验，不能由tiny实验代替。

本次可比较保持8组语义的分组GEMV、TransposeBatchMatMul tiling以及权重布局。按逻辑BF16权重计算约33.55 MB/task，38.15 μs对应约0.88 TB/s逻辑权重吞吐；这是估算，提示有搬运压力，不能直接称为物理HBM速率。

先检查GM→L1→L0的首发、tiling和prefetch。继续加UB双缓冲不对应这条Cube路径。若融合wo_a/wo_b，应保留wo_a输出的BF16舍入，不将两层权重直接代数相乘后当作等价实现。

## 6. P1后续：单tokenMoE路由与GMM

### 6.1 路由准备

当前TopK每步0.177 ms、InitRouting约0.461 ms、Unpermute约0.125 ms，合计0.763 ms。InitRouting为一个token/top2启动48个Vector block，Vec约0.33%；有小输入通用sort/permute的优化机会。

可在单token、TP1、top2且所有expert本地的条件下，直接准备两个路由slot、group list、反向映射和加权汇合。TopK必须保留bias、hash/token-ID策略、tie规则、权重归一化，不能替换成未经核验的普通topk。

`RmsNormCast`已提供BF16规范化输入与独立FP32路由输入。即使后续与router融合，也不能把BF16输出再转FP32来替代这条精度语义。

### 6.2 GMM小M

GMM1每步1.246 ms，只用4个Cube block；GMM2每步0.570 ms，用24个Cube block但Cube比例约4%。可检查tile分配是否受token/group大小限制，以及更多核、selected-expert GEMV或激活epilogue是否更合适。

权重报告格式为NCL，不能把逻辑contiguous当作物理ND。候选必须按真实物理布局或一次性转换后访问，并验证不同expert ID、零token group、非恒定expert权重。生产expert维度、W4A8 scale、EP通信和padding与tiny不同，单token特化首先作为tiny验证。

建议先完成激活融合，让GMM计时不再被中间链掩盖，再决定是否值得实现整条expert MLP融合；否则会同时改变多个因素，难以归因。

## 7. P2：HC继续优化的方向和限制

新的HC合计1.257 ms/步，当前已去除原CV标志9握手。剩余机会可按低侵入顺序探索：

- **静态fn的HF32转换预处理。** `project_hf32`每tile重复对权重做位级转换。推理权重稳定时可在编译前一次转换并缓存，输入x仍按当前方式转换。缓存必须随权重版本失效，并在非恒定权重审计中验证；不能改变本机已校准的舍入模式。
- **project的BK/PARTS分工。** 当前24个program、每个5个K tile，Vec和MTE2都已有实质占比。分K可能增加归约、launch和临时buffer，应做单变量实验。自动multibuffer之前已测无额外收益，排在后面。
- **finish与邻接RMSNorm/RmsNormCast融合。** 模型源码显示每层HC后紧接norm。但RMS需要完整5120维的归约，finish当前分5个block写输出，不能用每tile的RMS替代全维RMS；若为了融合引入全核同步，可能重现已消除的等待。候选还需保留先写BF16、再norm的数值边界，以及FFN的独立FP32输出。
- **HcPost与邻接操作。** HcPost每步0.589 ms且Vec约62.8%，没有证据表明它存在与原HcPre相同的同步问题。先核对内存访问和消费者，不能只因调用80次就重写。

减少Sinkhorn次数、使用较低精度权重或放宽容差不属于本轮语义保持的候选。

## 8. P2：SparseFlashMla、metadata与图外开销

### 8.1 SparseFlashMla值得采更细的控制流

40次FlashMla合计1.732 ms/步，主要shape的AIC Scalar约60%、Cube约1.6%、ICache miss约21%；AIV Scalar约56%、Vec约1.5%、MTE2约32%。QuantLightningIndexerV2另有8次、0.340 ms/步。

这提示小batch下调度、分支、同步或任务分工开销，但没有逐核等待数据证明其主因。下一步对本次真实shape采Default与MemoryDetail，检查最慢核、wait flag、工作量分配和cache访问，再决定调整核数/tiling、预取稀疏KV还是精简控制路径。不能直接把HC的wait_id9结论搬过来。

必须保留SWA window、compressed ratio、sink、mask、TopK indices和softmax语义。不能以减少参与注意力的token数作为等价性能优化。

### 8.2 metadata已有跨层共享

每步只有3次SparseFlashMlaMetadata、2次IndexerMetadata，合计0.271 ms，core类型为AI_CPU。源码 `_publish_task` 已按key在batch中共享metadata；不是每层40次重复生成。

metadata依赖当步seq_lens、C2 residual、query位置等，不能直接缓存上一步结果。可研究静态调度与动态长度字段分离、坐标准备的融合，以及任务发布到消费之间的等待。source中已复用rope和compressed lengths，需先保留已有机制再找冗余。

### 8.3 采集态空档不能作为收益承诺

both的20步CSV重新合并区间后，每步平均：kernel合计18.075 ms、并集18.045 ms、首末跨度30.580 ms、跨度内空档12.535 ms。数字包含AI_CPU task和采集开销，不是未采集态的关键路径分解。

现有trace的同步事件中，Event synchronize包含等待device完成的时间；该等待和device执行不能相加为host成本。下一轮如研究图外下发/调度，需独立比较无profiler墙钟、graph replay时序与必要事件，避免把这些12.5 ms全部归为Python或launch并承诺可回收。

## 9. 建议实验顺序与采纳门槛

1. **routed激活融合。** 先独立算子验证，再接入模型同进程A/B；检查原Slice/Clip/ViewCopy是否消失，实际路由与输出是否一致。
2. **shared激活融合。** 读取实际limit/alpha/beta，验证BF16中间舍入；与routed分别比较，再验证组合。
3. **Q/KV多流OFF/ON。** 使用已有实现，仅切换一项配置；确认有真实重叠且事件开销没有抵消收益。
4. **8组wo_a候选。** 对真实shape做布局与tiling单变量对照，再接入模型。
5. 根据新的热点排序，在MoE路由/GMM和SparseFlashMla逐核诊断之间确定下一项。

所有接入继续使用独立tiny、chip4，排除14–15；新增测试前核对设备占用，只停止本次自己的服务。每个候选保留回退路径、编译前非恒定权重验证和图捕获覆盖检查。

正式性能继续采用同模型、相同请求/KV/CPU绑定、正反交错、profiler与审计关闭的配对实验。收益须超过测量波动并在各组保持一致；kernel数量、单算子加速、采集态时间缩短均不单独作为采纳标准。

证据新增于 `results/opportunities/`：完整热点重聚合、7份当前源码快照和SHA256清单。未实施任何候选，因此本清单中的预期方向尚无新增实测提速。
