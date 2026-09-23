# A3 双 TP8 CED-PD 实现日志

## 目标与基线

- 目标：BF16 KV 下，P 的长 prompt 主段只经过层 0–19，P 从 encoder 输出生成层 20 的全局 KV/Indexer K；D 重建 128-token SWA 尾部并用完整 40 层解码。
- 基线：`b529c05` 的 `scripts/serve_a3_pd.sh`，两侧各 TP8、全 40 层、`MooncakeHybridConnector`。2026-09-23 在 A3-21 实测约 144K 正确；BF16＋AscendStore 的 1M 首次回答正确，但 DRAM 取回未验收。
- 实验分支：`feat/ced-pd-a3`，独立工作树 `ced-pd-release`。在功能通过前保留发布分支的原有行为。

## 上游能力边界

- vLLM Core 支持通用 PD、混合缓存分组和外部 KV 连接器。Core 的 DeepSeek V4.1 `swa_bounded_replay` 只在 Model Runner V2 开启，用于**前缀缓存命中后**重建 encoder SWA，不能代替第一次 prefill 后的 decoder SWA 初始化。
- 当前 vLLM Ascend 官方 V4.1 指南只验收同置部署；模型前向对每步仍执行所有 40 层，Ascend V1 运行时没有 `replay_start` 路径。我们本地双 TP8 PD 功能通过属于额外实测。
- 通用 PD 只分开调度/传 KV；获得论文的 prefill 计算收益还需要 CED 专用模型路径。

## 最小实现顺序

1. **源投影数值门。** 层 19 产出的 `(hidden_states, pre_mix)` 是层 20 的输入。新增的 `DeepseekV41DecoderLayer.write_global_source_from_encoder()` 走与普通层 20 相同的 `hc_pre → input_layernorm → compressor → Indexer K/long KV` 路径，暂不跳层。实际 NPU 对照应在同一次 forward 中比较该方法写出的有效 cache 行与普通层 20 随后写出的行；主 KV、Indexer K 和 scale 均须一致。
2. **P 角色的主段截断。** 仅专用 P 实例对 prompt 主段停在层 19，之后调用上述投影。模型仍加载完整权重，先验证计算量与缓存正确性；避免在同一阶段改变权重布局。不能向普通客户端返回半模型 logits。
3. **D 的尾部重放。** 维持现有 `N−1` 交接边界，D 在 global KV 加载后对最后 128 token 重放，恢复各层 SWA/环状态；重放不得覆盖已收到的 global KV。先让 D 重算 lower+upper 尾部，正确后可传末尾 `H20` 只算 upper。
4. **协议与接口。** `MooncakeHybridConnector` 当前按 13 个缓存组交接并让 P 采样 1 个 token 完成 API 请求。P 跳上半层后需要显式“仅完成 KV 生产”的响应，或末 token 保持完整计算；不得用错误 logits 伪装正常模型响应。连接器要明确哪些组在 P 写入、哪些组由 D 重放补齐。
5. **验收。** 先短输入逐 token 对照，再做 144K/1M、流式、多轮、外部池取回。记录 P 各层实际处理 token 数、D replay token 数、P/D transfer 字节、prefill 时间、TTFT、TPOT、吞吐与错误指纹。只有 BF16 全链路通过后再叠 KV8、DRAM 分层和 P 权重裁剪。

## 当前进度

- 已在 `patches/files/model.py` 加入层 20 的源投影入口，默认不触发。它复用现有 `_write_compressed_source`，尚未在 A3 真实权重下对照。
- `compile()` 语法检查和 `git diff --check` 通过。没有声称 CED 运行时或性能已经实现。
