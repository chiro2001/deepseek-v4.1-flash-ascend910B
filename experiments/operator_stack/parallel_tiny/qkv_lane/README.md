# BF16 tiny QKV lane

本目录独立验证 `DeepseekV41EagerAttentionImpl.multistream_preprocess` 的 Q_a/KV 合并；保留原生浮点 qr、RMSNorm、Q_b、RoPE、SWA 写入、压缩器与流事件。只改两条原生 GEMM 调用，输出再切为两个 512 列张量。BF16 默认使用 ND(2) 合并权重，`QKV_LANE_PACK_FORMAT=29` 需要显式验证实际 NZ 格式；禁止把创建时回退 base format 的张量报告为 NZ。

模型口径为 a3-21 Ascend910C、physical6、TP1、40 层、hidden5120、8 专家/top2、q_lora512、BF16、dummy 权重、无 Engram/图片/推测。运行完整模型与原生缓存；它不是正式 W4A8 384 专家/top6/TP8 的收益证明，也不是算子 probe。

`qkv_worker.py` 约束 physical6，使用已存在的 privileged tiny 容器时仅在该次 `docker exec` 设置 `ASCEND_RT_VISIBLE_DEVICES=6`。源、缓存、输出目录均私有。`HCCL_DETERMINISTIC=strict` 必须保留。

图启动先用 single-stream 完成初始编译，再同实例启用 multistream，重新捕获两个原生银行和候选银行；这与正式入口阶段顺序一致。显式传 `--static-kernel` 验证 static 路径；省略时为各臂一致的 non-static FULL_DECODE 诊断，不能报告正式收益。初次 initial multistream=True 的 static 原生捕获曾在 `_fx_func_run` 的 stream synchronize 持续等待，编译输出目录为空，已保存 Python 栈和仅本任务终止记录。

执行 `bench_qkv_tiny.py --output <目录> --audit --pairs 2`，先在同实例建立两个独立原生图银行并完成 A/A；再捕获候选，比较实际 40 层、80 个 QKV 消费者输出，要求逐位一致或最多 1 BF16 ULP；全模型 token、真实路由及 top5 logprob 另行比较。精度运行成功后，才执行 `--accuracy-evidence <精度目录>/qkv_accuracy.json --pairs 6` 的无审计性能配对。性能运行会同时检查完整审计结果与候选源哈希。

正式审查发现：V4.1 模型经 `torch.ops.vllm.dsa_v41_forward` 调用 `attn.v41_impl.forward`，继而使用独立 `multistream_preprocess`。绑定 `AscendDSAImpl._mla_prolog_multistream` 能注册 40 个原始投影权重，却不会覆盖该生产链。零覆盖属于接线失败，尚未进入数值比较。

当前证据：chip2 启动被设备侧 HDC 文件空间拦住；chip6 SDK 通过，但完整 tiny 捕获受 AiCPU 507018/设备 OOM 拦住，未完成原生 A/A、真实消费者或模型精度门。短 BF16 projection-only probe 正常退出：24 输入的合并输出最大 8 ULP，超过 1 ULP 门，候选被拒绝，未做候选性能测试。该 probe 请求 NZ 时发生 base format 回退警告，旧输出未记录实际格式，因此它不能作为 NZ 证据。不得由此报告完整 tiny 或正式收益。
