# Indexer K 后处理融合（A3）

入口：安装态 `vllm_ascend/models/deepseek_v41/indexer.py::DeepseekV41Indexer.update_keys`。保留 `wk` 投影输出的 BF16 边界，一个 Vector kernel 完成 RMSNorm → 尾64维 interleave RoPE → dynamic INT8 quant → K/FP16 scale 两份 cache 写入。

投影为 BF16 `[T,128]`，gamma 为 BF16 `[128]`，epsilon 由模块传入。Norm 在 FP32 归约和乘 gamma 后舍入 BF16；RoPE 用 FP32 Mul/Add，关闭 FP fusion，再舍入 BF16。量化按安装态 CANN dynamic_quant_single_row.h/multi_row.h 的真实顺序：`Div(127,max(abs(x))) → Mul(x,coefficient) → nearest-even → int8`。FP32 dequant scale 为 `max/127`，写 cache 前转换 FP16；全零行输出 scale=0、q=0。不能把除法重排成 `x/scale` 或 `x/max*127`，它们在半整数附近会改变 INT8 字节。

RN Div 的 127 分子采用一次性按设备缓存的 FP32 向量，规避该 Triton 工具链对 scalar/常量向量 RN Div 的 TypeRange 断言。RoPE mask 外的地址也保持有效（`col&63`），规避非连续 trig 的 masked negative-offset 寻址差异。

slots 是 builder 准备的 `[T,2]` `(block,row)` 坐标，支持 int32/int64；任一负坐标跳过。K/scale cache 可以是 layer-outermost parent 的 strided view，地址使用其真实 strides。独立验证比较整个 parent，检查 padding/其他层未被覆盖。

独立精度门槛是各阶段和两份完整缓存逐位相同；数值102case覆盖16seed×三幅度×T1/4以及zero/constant/half-tie。布局48case覆盖 FP32 trig、混合负坐标、strided X/cos/sin及int64坐标。原生低层 RoPE 的输入需要连续，任意stride的逻辑参考先连续化再执行原生。debug输出固定连续布局；正式图不输出中间张量。

模型接入只启用 tiny decode T=1、BF16/128/rope64、INT8 K/FP16 scale；prefill、空行和不适配布局保留原生。四K源层分别保存原生/fused图bank。审计在消费者点 clone 投影、坐标、trig和本次写入的缓存行，防止下一步更新可变buffer。47/48输出token交替覆盖C2完成和未完成状态。审计 RPC 使用 inference_mode；正式计时无这些快照或profiler。

已完成150case、随机整网审计与两轮各12组性能确认，保留融合及原生回退，详见INDEXER_RESULT.md。前两项已有完整结果与拒绝证据，不重复测试。
