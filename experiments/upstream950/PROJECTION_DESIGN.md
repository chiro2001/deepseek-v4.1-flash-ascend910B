# Q/KV 与 q_b 权重面板（A3）

保持 tiny BF16 TP1 投影语义。Q/KV 从相同 [T,5120] 输入分别乘两份 [512,5120] 权重，q_b 将 [T,512] 映射到 [T,32768]。首个切口联合 Q_a/KV 权重，加载后一次 cat 成 [1024,5120]，一次原生线性投影后拆两份 BF16；后继 q_norm、kv_norm、RoPE 与 cache 写入保持原生。独立对照两个 projection 输出。

q_b 面板候选一次性将 [32768,512] 重排成 [256,4,128,128] 的 (N tile,K tile,K,N) 连续面板，按消费者顺序读取。A3 Cube tile 使用 M16×N128×K128，FP32 L0C 8KiB；两份BF16 B面板64KiB和两份A面板8KiB，只用72KiB L1，低于本机512KiB；不使用950的256KiB L0C/UB预算。T=1的其他M行零填充，最终BF16输出。

使用Triton标准dot和两阶段buffer，先核验工具链生成真实Cube代码与double buffer；不能仅凭num_stages参数称已经预取。分别对原生ND、一次NZ预排、连续panel候选进行单变量比较，先数据精度与资源合法性，再真实模型图bank/路由/logprobs和配对性能。权重预排需跟随storage/version失效；预热/捕获前完成，不在每一步复制权重。

四种切口已实现并验证，当前均不采纳，详见PROJECTION_RESULT.md。若Cube累加顺序导致BF16末位变化，按BF16精度约定评估与完整随机模型审计，不把Indexer byte相同门槛擅自套给矩阵乘，也不放宽已有INT8门槛。prefill、量化权重和不适配形状保持原生。
