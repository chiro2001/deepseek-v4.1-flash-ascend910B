# Tiny 20 ms/step 优化机会总账

2026-10-10。最终metadata+多group启动组合：19.209 ms/step、A=1、52.058 token/s，紧邻十组全快且均≤20ms。旧21.991及19.970是不同会话，不相减。
本文件随实验更新；2026-10-09的`evidence/baseline/NEXT_OPTIMIZATION_OPPORTUNITIES.md`是历史筛选依据，不能当作当前待办或当前时延。

|机会|当前状态|证据与下一步|
|---|---|---|
|routed/shared clamp和激活融合|已采纳|保留原地clamp、alpha/beta及BF16舍入；40层实际覆盖|
|Q/KV多流|已采纳|完整事件/图bank，同进程交错对照|
|HC静态HF32、HcPost分块|已采纳组合|保留20次Sinkhorn，不将单项小收益跨会话累计|
|单token路由准备/汇合|已采纳|全部64个expert对独立校验及实际路由审计|
|GMM1 selected-expert Vector|已采纳|非恒定权重、专家覆盖，原生矩阵精度门槛保留|
|GMM1 + routed激活epilogue|已采纳|融合对被替换的两kernel路径0 ULP；紧邻十组全部更快|
|融合GMM multibuffer off/on|已试，无实用收益|18.501/18.591/18.476 us；不接入额外开关|
|8组wo_a Vector/Cube布局、tiling|当前候选全慢|六Vector、五Cube；不重做相同筛选|
|GMM2 Vector/Cube|当前候选全慢|五支持Cube配置，保留原生|
|shared上投影+激活|当前候选全慢|真实N-major布局七配置；不能引用旧KN布局成绩|
|shared下投影、q_b、wo_b候选|已有负结果|新的算法/布局才值得再试，不能重复原配置|
|HC finish + 全维RMS/RmsNormCast|精度门槛拒绝|固定FP32路由门槛未通过，未放宽|
|replay前同步跳过|已试，无稳定收益|三组路由/token/logprobs通过，一组更慢，不采纳|
|SWA单kernel Cube|编译失败|十二配置BiSheng生成失败，不是有效性能数据|
|SparseFlashMla逐核等待|已采有效Default/MemoryDetail|C128/C2仅1Cube、2Vector实质工作；C2 wait6最大10.807us|
|head分组|精度通过，计时隔离|晚期计时受别租户占用污染，尚未取得有效正式收益|
|metadata静态配置缓存|模型审计和正式计时通过|`mdstatic`，仅缓存不可变模型字段|
|group-local slot坐标/掩码融合|模型审计和正式计时通过|`mdslots`，224个边界用例逐位通过|
|C2 ring/源位置/RoPE gather融合|组合已通过|`mdall`，144个边界用例逐位通过；19.970ms/步，十组全快，1718→1536 task/步|
|已有多group原始slot mapping启动融合|已采纳|`mdfull`，三组verify/路由/token/logprobs通过；11个group、0回退；紧邻十组19.209ms/A1/52.058tok/s，1536→1526 task/步|

## 尚未完整尝试的新方向

1. **将多个group的2D slot准备合成一次启动。** 当前`mdslots`仍每个物理group调用一次Triton。可在同一步收集各group独立的输入/输出指针，并在metadata executor的inputs-ready事件前一次提交；不能共用不同物理group的输出buffer，也不能跨步复用旧slot值。
2. **减少Triton metadata启动的Python开销。** cProfile显示原始slot计算已有每步11次JIT调用。可对已编译核缓存launcher，使用当前stream；动态整数的specialization、指针对齐、dtype和设备必须进入缓存key或明确不specialize。不能固定metadata stream为模型主stream。
3. **将C2两个融合核再合成一个。** 当前ring counts和source/RoPE是两次启动。可在同一grid同时处理请求行和token行，保持padding、skip和buffer尾部语义，独立精度与墙钟确认。
4. **norm/RoPE/cache store的邻接融合。** 框架已有开关在Ascend会被禁用，需要真正的Ascend实现。必须保留BF16中间边界、cache布局、奇偶完成策略和多流依赖。
5. **Q_a与KV投影共享一次分派。** 拼接静态权重后仍分别物化原有BF16输出；需要与已采纳多流比较，不能假定减少一个MatMul一定更快。
6. **在空闲芯片重新测head分组。** 精度已通过但被污染的计时不能用；按C128/C2分别筛选，再判断整模型收益，保持native注意力参照。
7. **大投影/lm_head的其他分派与布局。** 现有部分Vector/Cube候选已慢；可研究不同原生tiling或预取。保留完整词表和BF16定义，不以缩小工作量冒充等价优化。
8. **注意力更细的指令/等待与Cache证据。** 现有Default/MemoryDetail仍不足以量化各级容量Occupancy和完整指令timeline；支持范围需按实际910版本核验，失败采集不计入证据。
9. **单token的slot核缩小block。** 本轮PipeUtilization显示当前slot融合核Vector约53.3%、Scalar约41.9%、MTE2约0.46%；B=128下处理一个有效token仍有不少计算。可独立筛选B=1/16等，严格验证mask和buffer尾部，不将此计算热点解释为搬运瓶颈。

## 精度语义修正

当前安装的vendor `RmsNormCast`源码与上板实验确认：FP32输出等于最终BF16输出再转FP32，max diff=0。
历史机会清单“独立未舍入FP32输出”的警告不适用于该实现。HC/RMS候选仍因固定门槛不通过而被拒绝，不能据此放宽精度。

所有收益以同模型进程正反交错、关闭profiler/审计的完整decode计时判定；三元组必须自洽。
TP1 tiny dummy的结果不能外推为真实checkpoint、生产TP8/W4A8、DSpark或CED-PD的收益。
