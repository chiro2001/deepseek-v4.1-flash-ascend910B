# inverse RoPE / 八组 wo_a：当前候选均不采纳（A3 tiny）

完成三个保持 BF16 边界的方案验证：inverse RoPE与八组输入重排后使用 `torch.bmm`；只处理尾64维、融合负sin的Triton inplace RoPE并保留原生TransposeBatchMatMul；使用固定group counts的八组GMM替代wo_a。当前tiny均无收益，保留原生路径及可复现的候选。

32个随机用例覆盖T1/T4、FP32/BF16 trig、多seed、非恒定权重，三种方案的旋转/wo_a全部逐位一致。前两个方案的40层模型图bank审计，wo_a、wo_b、实际路由与logprobs也逐位一致，max delta0；审计在原始attention和各消费者点保存快照。未将wo_a/wo_b代数合并。

正式三臂同进程12组A/B，保持原生prefill、2K prompt、48输出、A=1，关闭审计和profiler；GPU独占锁及56次chip4采样均0。

|路径|`ms/step`|A|tok/s|对原生配对中位|更快组数|
|---|---:|---:|---:|---:|---:|
|原生|26.155935|1|38.232240|1|—|
|重排+torch.bmm|30.667795|1|32.607496|0.854201|0/12|
|尾部inplace RoPE+原生八组MM|27.222755|1|36.733975|0.964095|0/12|

分别增加配对时延中位4.465465ms和0.972245ms。少一个Neg task或少一次输入处理，没有转化为该工具链/该tiny的整体收益。

静态八组GMM在32个随机用例逐位通过后，以相同原生RoPE和相同权重作单算子设备图12组对照：原生25.188080us，GMM25.909870us，配对0.972292。两臂均包含相同的输入还原copy；该对照只改变wo_a算子，已在微观阶段无收益，未将GMM候选接入整网或默认启用。

实现：`scripts/epilogue_prepare.py`、`epilogue_patches.py`。baseline原生回退，其他arm为packed/inplace/gmm。证据：`evidence/final/epilogue_model_audit_v2`、`epilogue_model_perf_v1`、`epilogue_precision_v3.json`、`epilogue_gmm_micro_micro.json`。结论限定这些已测方案，不等同于950完整Cube/Vector融合在A3上全部不可行。
