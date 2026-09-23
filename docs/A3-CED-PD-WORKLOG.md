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

- 已在 `patches/files/model.py` 加入层 20 的源投影入口，默认不触发。它复用现有 `_write_compressed_source`。2026-09-23 在 A3-21 以真实权重 `v41-flat-verify3`、TP8、BF16 KV、Engram host 路径跑了一次 14-token 请求。8 个 TP rank 均记录 `rows=14` 且无比较异常；原始日志为 [`evidence/ced_source_8ff6a4a/serve.log.gz`](../evidence/ced_source_8ff6a4a/serve.log.gz)，解压后的 SHA-256 为 `cce508a4a6bc814164b78ec325768b67400d004192ab6a227d1773141fef56e0`。该门只证明这一短请求的采样行相等，不能外推到分块或长上下文。
- 开发诊断开关 `V41_CED_SOURCE_COMPARE=1` 会在非图捕获的真实 forward 中，先调用源投影，再执行普通层 20，并对每块前 8 和后 8 个有效物理槽中的主 KV、Indexer K、scale 做逐张量精确比较。`V41_CED_SOURCE_COMPARE_CHUNKS=N` 设定每个 rank 最多比较的块数，默认 1；日志会列出块序号和 token 位置。不匹配立即报错。两个开关默认均不启用；长请求测试应设置足够大的 `N` 覆盖末块。
- **144K 分块数值门通过。** A3-21 的 `33a6024` 独立 P 实例以 `V41_CED_SOURCE_COMPARE_CHUNKS=20`、真实权重、TP8、BF16 KV、`BAT_TOKENS=8192` 跑 `bigprefill`。第一条请求实际 144,404 上下文 token；模型前向到位置 144,461，分 18 块（前 17 块各 8,168 token，末块 5,606 token）。8 个 rank 的每一块均精确匹配所采样的主 KV、Indexer K 与 scale，没有异常。第二条请求又覆盖了头两块，合计各 rank 20 条成功记录。证据：[`probe_144k.json`](../evidence/ced_chunks_33a6024/probe_144k.json)、[`serve.log.gz`](../evidence/ced_chunks_33a6024/serve.log.gz)；完整日志解压后 SHA-256 为 `2129a09e908e31639e00dbf408fecb5e4189b39d66f56afb4f2feaa68100e264`。探针请求将 `max_tokens` 设为 1，直接访问 P 角色服务；两条回答仅为 `Z`、`V`，不满足检索判据，因此这次**仅验源投影数值，不验回答质量**。测试容器已停止，0–7 卡无运行进程。
- 下一实验开关 `V41_CED_ROLE=prefill` 在模型主循环运行完层 19 后直接写层 20 的全局源，跳过层 20–39；当前要求 `SPEC=0`。P 端采样会被固定为内部传输标记 token 42，使 `MooncakeHybridConnector` 走 `FINISHED_LENGTH_CAPPED` 发布缓存。该 P 端点只能由 PD 代理内部调用，直连回答无语义；D 尾部重放和缓存组有效性协议尚未实现，不得用它搭配普通 D 对用户提供服务。此开关默认关闭，仍需在真实 A3 实例验证能完成长 prefill 与层 20 写入。

独立 P 计算实验的启动参数（端口和容器名须先确认空闲）：

```bash
MODEL="$HOME/models/out/v41-flat-verify3" \
DEVS="0 1 2 3 4 5 6 7" PORT=18770 KV_PORT=18870 \
NAME=dsv41-ced-p-cut RUN_ID=ced_p_cut \
SERVED_NAME=deepseek-v41-ced-prefill-only \
SPEC=0 STATIC_KERNEL=0 V41_CED_ROLE=prefill \
bash scripts/serve_a3_pd.sh prefill
```

服务就绪后，用 `python3 tools/ced_prefill_probe.py --base-url http://127.0.0.1:18770 --context-tokens 144000 --expect-masked-swa --out results/ced_p_144k.json` 检查内部 marker、12 组布局、上半层 SWA 屏蔽及 replay 标记。

**A3 真实权重验收（`302b586`）：** 按上面的独立 P 配置启动，`/health=200`，8 个 rank 都记录 `[CED-P] internal producer`，日志没有 decoder-layer 违规异常。短请求带 `do_remote_decode=true`，返回 `finish_reason=length`、标记 token ID 42，并带 `do_remote_prefill=true` 的 Mooncake 元数据。随后 [`ced_p_144k.json`](../evidence/ced_p_cut_302b586/ced_p_144k.json) 记录真实 prompt 143,963 token、17.07 秒、12 组 block ID、标记 token ID 42；完整日志为 [`serve.log.gz`](../evidence/ced_p_cut_302b586/serve.log.gz)，解压后 SHA-256 为 `02061375f211b754003e1c907588fbe74340414a967987a2b2806a5f1906c2dc`。该测试证明 P 端能运行长 prefill 并完成内部交接回执；其耗时不可直接与前一个开了数值探针和 DSpark 的 30.8 秒相减作为性能收益。容器已停止，0–7 卡无运行进程。

**P 侧组有效性协议已上 A3 验收。** `302b586` 的 12 组回执含 G7–G11，即上半层 SWA；P 跳层时这些组未写入，却会被普通连接器列入传输。`34fdf08` 的 [`experimental/ced/mooncake_hybrid_connector.py`](../experimental/ced/mooncake_hybrid_connector.py) 在 A3-21 实测将 G7–G11 block ID 置空，仍保留 G0 全局 KV、G1 环、G2–G6 低层 SWA；交接标记为 `ced_replay_tokens=128`、`ced_missing_swa_groups=[7,8,9,10,11]`、`ced_prefix_tokens=143962`。[`ced_p_mask_144k.json`](../evidence/ced_p_mask_34fdf08/ced_p_mask_144k.json) 记录真实 prompt 143,963 token、`finish_reason=length`、标记 token ID 42、组 block 数 `[1125,1,2,2,2,2,2,0,0,0,0,0]`；[`serve.log.gz`](../evidence/ced_p_mask_34fdf08/serve.log.gz) 解压后 SHA-256 为 `d0c6e8584ab56b3d7a3f3921333ff29f742418d3c584e6f1570e3072392a143e`。容器已停止，0–7 卡无运行进程。

`34fdf08` 的 D connector 对该 replay 标记直接拒绝，因此这里只验了 P 元数据屏蔽；**不能**作为 P→D 正确性或质量门。一次容器内单独导入连接器并调用拒绝分支，确实打印拒绝信息，但 Python 退出时发生 `corrupted size vs. prev_size`（退出码 134），不把它当作 D 端到端验收。后续工作树已加入尚未验证的 D 调度回退、缺失页清零和 replay 块内不回写全局 KV 原型；必须先在隔离实例验证，再做质量判断。下一步见 [`D_REPLAY_NOTES.md`](../experimental/ced/D_REPLAY_NOTES.md)。

## 8+8 D replay 首轮故障（`8b59d03`）

- A3-21 上 P=0–7、端口 18790，D=8–15、端口 18791，代理 18792；真实权重、BF16、`SPEC=0`。两侧均 `/health=200`，D 的 Core replay 补丁通过启动脚本现场应用，D 连接器识别到上层 SWA G7–G11。P 服务在本轮后仍运行，D 和代理已停止。
- 代理发最短数学请求后返回 HTTP 500。D 还在 KV 加载前的缺失页清零步骤就失败：`tensor.index_fill_(0, indices, 0)` 在当前 Ascend 共享缓存视图上申请 **7.36 GiB** 临时显存，而当时每卡只剩约 6.2–6.5 GiB。没有执行到 D replay 前向，因此不能判断调度或模型质量。[原始 D 日志](../evidence/ced_d_oom_8b59d03/serve.log.gz) 解压 SHA-256 为 `6d116cf5049ecd016aed7aae1f782a885134ac087cb11c456af7a3a6b08b5ec6`；[代理响应](../evidence/ced_d_oom_8b59d03/proxy_response.json)。
- 现已把清零改为对每个物理 block 用 `tensor.narrow(0, block_id, 1).zero_()` 原位写入，避免 `index_fill_` 的整视图临时申请。该修复**尚未真机复测**；下一步只重启 D 和代理，复用仍在运行的 P。
- `compile()` 语法检查和 `git diff --check` 通过。没有声称 CED 运行时或性能已经实现。
