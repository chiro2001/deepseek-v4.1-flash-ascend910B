# Attention 输出段：inverse RoPE 与八组 wo_a（A3）

当前 tiny decode 输入为 [1,64,512] BF16，尾64维 interleave inverse RoPE；wo_a 是八组 [4096,512] BF16，结果 [1,4096] BF16，然后 wo_b 原生线性层。TP1 不会进入已有 n_local_groups==1 的 2D 路径。

候选准备：一个Vector kernel把inverse RoPE与分组输入准备合并，直接输出 [8,T,4096]，在FP32执行旋转并保留BF16边界，融合负sin以减少额外Neg task。八组 wo_a 改用同形状 torch.bmm 专用入口，保留wo_a结果BF16，然后原生wo_b；不把两层权重合并。与950融合范围相比，此处先验证可落地的输入准备及BMM tiling，无需假设2201有相同UB→L1通路。

先对随机BF16 input/weight、FP32与BF16 trig、T1/T4、多seed逐段验证旋转及wo_a输出。native bmm若有浮点重排，必须记录差异并验证BF16误差界及随机模型的路由/logprobs，不放宽INT8 Indexer的byte门槛。只对T1、八组、BF16且无OTP/olora分支的目标路径启用；prefill和不适配形状保留原生。准备权重面板在加载/捕获前完成且跟随权重版本失效；不可逐步转置权重。

图bank审计在inverse RoPE之前保存raw attention及在wo_a、wo_b消费点保存输出，避免后续层改写可变buffer。正式计时关闭审计，与原生baseline同进程交错，等chip4闲时执行。三个实际候选已完成验证，当前均不采纳，详见EPILOGUE_RESULT.md。
