# GMM1与激活融合：稳定配对改善、1718 tasks/step和服务交付

日期：2026-10-10 · a3-21 chip4 · `feat/tiny-operator-opt-20261009`

**当前融合方案在紧邻十组正式对照中达到21.991 ms/step，A=1，45.474 token/s，十组全部更快。**
另一次三模式六组会话的融合中位为21.779 ms/step，A=1，45.915 token/s。
两套会话不直接相减；20 ms目标尚未达到，goal保持active。

## 1. 做了什么

本轮把已验证的Vector GMM1与routed clamp/SwiGLU合成一个kernel。
真实tiny形状为两个slot、5120输入、每expert 512个gate/up输出。每个program同时计算
16个gate及16个up，BK=512；两个slot共32个Vector program。

```text
原组合：Vector GMM1 → BF16写回 → clamp/SwiGLU kernel → GMM2
新组合：一个kernel计算GMM1 → 舍入BF16 → clamp/SwiGLU → GMM2
```

融合保持GMM1的BF16中间舍入、limit转BF16、gate单侧及up双侧clamp。
写回clamp后的中间输入，保留原地修改行为。专家从设备group list读取，不将路由ID固定为host常量。
GMM2、apply_act_quant及before-GMM2事件保持既有调用定义。

## 2. 采纳依据

关闭profiler、logprobs及路由回传；2048输入、48输出，seed轮换，prefix关闭、KV=4 GiB、
同CPU绑定、FULL_DECODE_ONLY `[1]`。每臂两轮预热，正反交错。

|十组紧邻、同一进程|ms/step|A|token/s|
|---|---:|---:|---:|
|已有HC/路由/激活/多流/static组合|22.276|1|44.892|
|增加GMM1/激活融合|**21.991**|**1**|**45.474**|

配对吞吐改善中位约1.31%，十组全部更快。
三模式六组另一次会话为原生GMM基线23.086/1/43.316、已有组合22.220/1/45.004、
融合21.779/1/45.915；融合对原生基线六组全快，对已有组合五组快、一组接近持平。
上述两个会话的绝对中位数不能混比或累计百分比。

40份旋转权重的独立筛选中，较优融合18.499 μs，已有两kernel27.440 μs，原生链56.254 μs。
单算子成绩用于筛选，正式收益由完整模型配对确认。

## 3. 精度参照与模型审计

未启用static和启用static各做三组非恒定权重审计，随机化gate、HC及专家权重后再编译。
实际83800个top2路由对、48个输出token一致，最大logprob差约9.537e-7/0。
每套图确认融合覆盖40层；融合输出对当前两kernel路径最大0 ULP。

初次新增“整链对原生GMM”的1 ULP检查，在极小值处报5 ULP、绝对差1.137e-12。
诊断发现已有两kernel基线对原生也有同样误差，而融合对已有基线逐位一致。
因此校正融合检查的参照为它实际替换的已验证路径，保留1 ULP门槛。
原有GMM对原生的全元素BF16容差、同输入激活精度、实际路由/token/logprobs检查均保留。
没有把已有近零归约误差掩饰为融合新增误差，也没有降低融合门槛。

GMM1中间输入会被clamp原地修改；审计参照同步应用实际limit，避免把clamp后buffer
与未clamp的原生GMM结果错误比较。prefill、量化、bias、LoRA及其他形状保留原路径。

## 4. 匹配profiling及微架构

同进程每臂连续20步，Device4、trace、CSV哈希及结构计数通过验收。

|每步|已有组合|融合|
|---|---:|---:|
|全部task|1758|1718|
|独立Vector GMM1|40|0|
|独立clamped激活（含shared）|80|40|
|新GMM1/activation kernel|0|40|
|全部kernel累计|14.881 ms|14.572 ms|

每步减少40个task。累计时间为采集态归因，不能直接当作可回收墙钟。

Default/MemoryDetail成功采到32个Vector核、1800 MHz。task约18.720/18.680 μs。
Default最慢核执行18.066 μs：Vector11.080、MTE2 5.719、MTE3 0.268、Scalar5.277 μs。
计数有重叠，不能相加。Vector约61%、MTE2约31%，说明该新kernel已有实质计算工作。

MemoryDetail核0 GM→UB约330.125 KiB、UB→GM0.375 KiB；路径利用率约8.97%/0.012%。
这是逐核口径，不能解释为整芯片HBM速率。未取得新的容量Occupancy或指令级TimelineDetail。
后续可对多K tile的buffer编排与分工做独立对照，不凭低带宽直接宣称dual-buffer收益。

## 5. 未采用的方案与运行时诊断

shared投影与激活按真实N-major布局筛选七种Vector配置，全部慢于原生。
原生约8.26 μs，较优候选约10.535 μs，拒绝接入；此前KN布局筛选的改善不能代替该实际布局结果。

chip6完整模型失败已定位到QuantLightningIndexerV2Metadata的AICPU任务/函数注册：
copy cpu so name失败、aicpu exception/timeout，507017。没有有效模型计时，不将该等待视为性能瓶颈收益。
未重置设备，未操作其他租户进程；继续在核验空闲的chip4工作，始终排除14–15。

## 6. 服务、证据及下一步

已恢复本次独立服务，默认`GOAL20_ARM=gmmact`、static启用。
模型id正确、health=200，32输入token→16输出token验收通过；地址为`http://172.17.0.4:18971`。
服务恢复不是20 ms目标完成声明。重测前只停止自己的API，并重新核验设备及容器归属。

代码与小型证据在`experiments/operator_optimization/`，主要目录：
`goal20_gmmact_model_audit_v3`、`goal20_gmmact_static_audit_v4`、`goal20_gmmact_perf_v1`、
`goal20_gmmact_paired_v2`、`goal20_gmmact_default_v1`及`goal20_gmmact_memory_v1`。
原始大trace/请求留在远程`/work/operator_opt/results/`。

下一步继续研究融合kernel的buffer/分工、metadata准备和图回放开销，以及norm/RoPE/cache邻接融合。
仍为TP1 tiny dummy BF16，生产TP8/W4A8、真实checkpoint、DSpark和CED-PD未验证。
