Engram 消冗余与旁路重叠的当前判断：源码支持提前获得两层 lookup；真实输入的 8 片原生 BF16 投影恢复已验证，但 9/48 组消费者超过原 1 ULP 门限，因此候选被拒绝。TP8 通信与整网收益没有实测证据。

正式层 1、14 分别使用 `nn.Linear(6144,25600,bias=False,dtype=BF16)`。两层权重不同，不能跨层直接复用输出；重复计算来自每个 TP rank 都执行完整投影。`model.forward` 把投影输出按 `[20480,5120]` 分成 key/value，key 再变成 `[n,4,5120]`，随后调用原 `engram_gate`。gate 的 residual 依赖前序 backbone，投影本身只依赖该层 lookup。

按输出维切 8 片后，每 rank 原生 BF16 linear 权重为 `[3200,6144]`，K 和原始 BF16 规约保持。原权重每层 314,572,800 字节，单片 39,321,600 字节。M=1 时每 rank 输出 6,400 字节，按 rank 顺序 AllGather 恢复 51,200 字节后才能沿用原 key/value split 和 gate。第 6 片跨过 key/value 边界，不能分别把各片当完整 key/value 消费。AllGather 是输出搬运，不需要新增 FP32 reduction；原生 MatMul 因 N 变小可能改 tiling，仍须保留原 1 ULP 精度门。

`prepare_engram_inputs()` 在主图执行前运行，对两层同时算 hash/lookup，并复制进 `_engram_input_buffers` 的持久 lookup/mask。`EngramGraphCache` 捕获输入地址，重放时核对指针；主 `forward` 在各消费者层只读取 lookup 的有效前 n 行。因此从数据依赖看，两层 wkv 都可在 lookup 写完后启动，层 14 的投影可与前序 backbone 工作重叠。现有代码没有提前生成 kv，也没有跨步 kv 生命周期协议。

实现重叠至少需要三个事件关系：lookup 写入完成后旁路才能读；每层 gate 读取 kv 前等待对应投影完成；下一步覆盖同一 lookup/kv 槽位前，等待上一步旁路与 gate 消费结束。持久 kv 缓冲必须与 capture bucket 的 shape/地址一致，不能在重放时重新分配或更换字典张量。异步调度下不能依赖 Python 调用顺序或猜测上一轮 logits 拷回已释放这些缓冲。需要证明图内跨 stream event 能按每轮重放生效，并处理 profile/prefill/M>1 与 pointer mismatch 回退。

单片耗时只能回答 compute-only 的上界；单卡 8 片拼接只能验证本地恢复。真实 TP8 还须证明所有 rank 的 lookup/mask 相同，量化/格式与原生加载一致，测试原生 AllGather 的顺序、延迟与 graph 捕获，最后做原始 logits/routes/质量审计及客户端 `(ms/step,A,tok/s)` 配对。该 lane 不据 0.589ms 的 trace 总量宣称已经节省了相同整网时间。
