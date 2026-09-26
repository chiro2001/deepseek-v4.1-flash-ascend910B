# D 侧 128-token replay：实现边界

2026-09-24，基于 A3 固定镜像的 vLLM Core 0.27.1、Ascend V1 模型路径和
`34fdf08` 的真实 P 侧交接回执。P 在 143,963-token prompt 上已产出 G0 全局缓存，
并标出 G7–G11 上半层 SWA 缺失；D 尚未实现，当前收到 `ced_replay_tokens` 会拒绝。

## 首个 D 原型：重放完整 40 层的最后 128 token

先让 D 从 token IDs 重跑尾部 encoder＋decoder，这样不依赖传输 `H20`。长 prompt
的主段仍只在 P 跑 20 层；D 额外的 20×128 层工作量相对 144K/1M 主段很小。
重放完成后 D 对新 token 仍运行完整 40 层。等正确性通过，再研究只让上半层
replay、从 P 传 `H20` 的优化。重放是论文允许的近似状态，不能用逐 token 完全
一致作为长上下文唯一门槛。

1. **调度。** 当前 D 接收 `N−1` 个 token 的 P 缓存，然后只重算末 token。
   vLLM Core 较新 `scheduler.py` 的 `_update_waiting_for_remote_kv()` 和
   `_mark_prefix_replay()` 提供了可回移的骨架：远端读完、缓存页已分配后，
   将 `num_computed_tokens` 从 `N−1` 回退到 `max(0,N−1−128)`，保存
   `replay_start/replay_end`。必须把 replay 与真正未缓存的末 token 分成两个
   forward；第一步最多 128 token，只恢复 SWA，不采样；第二步处理末 token、
   写全局缓存并采样。A3 镜像还没有上游的 `prefix_replay_tokens` 状态，需在
   本分支为其固定 Core 版本加补丁，并验证 1、127、128、129 token 边界。
2. **缓存写入。** P 的 G0 已覆盖 `N−1`。D replay 时层 2/8/14/20 的
   `_write_compressed_source()` 必须跳过，以免覆盖已传全局 KV；C2 的 FP32
   环也不应被旧 token 的重放回卷。末 token 是正常前向，仍须写全局 KV 和环。
   P 的 G1 环、G2–G6 低层 SWA 已传；G7–G11 上层 SWA 未传。D worker 在
   报告接收完成前须把 G7–G11 的对应本地页初始化，防止旧显存内容被读到。
3. **SWA 可见起点。** replay 位置 `s=N−1−128` 的 query 只能看
   `[max(s,i−127),i]`，不能看更旧的未定义 SWA。A3 当前
   `DeepseekV41EagerAttentionImpl._native_attention()` 固定传
   `ori_mask_mode=4, ori_win_left=127`；`npu_sparse_flash_mla` 没有
   `replay_start` 入参。上游 CUDA 的 `sparse_swa.py` 有该限制，但要求
   Model Runner V2。A3 算子实现还明确只在**纯 SWA 模式**支持
   `ori_sparse_indices`；CSA1/2 的混合全局＋SWA 模式不能直接用它补一个
   replay mask（`sparse_flash_mla_tiling.cpp::CheckSingleParaOriSparseIndices`）。
   因此必须验证可用的物理页/长度重映射，或为 replay 增加分开的局部注意力
   与全局注意力及稳定 LSE 合并；不能只把 scheduler 计数回退就声称重放正确。
4. **验收。** 先用短请求证明 P→D 交接、G7–G11 只在 replay 后有效、
   末 token 位点恰好是 `N−1`；然后 144K、1M 的检索、流式、多轮及复用。
   同时记录 P 运行层数、D replay 位置和层数、每组传输字节、prefill、TTFT、
   TPOT 与吞吐。对比基线必须固定同机、同请求、同 `SPEC=0`/缓存策略。

## 可直接借用与不能直接借用的上游代码

- 新 Core `vllm/v1/core/sched/scheduler.py` 的 `prefix_replay_tokens`、
  `Request.replay_start` 和 `NewRequestData.replay_start` 处理的是**前缀缓存命中**。
  它证明「保留已装载页、只回退计算游标」在调度层可行，但还需把 CED 的
  `N−1` 交接及缺失组标记接进去。
- 新 Core V4.1 `model_state.py`/`sparse_swa.py` 的 replay mask 依赖
  Model Runner V2 和 CUDA/Triton 注意力；A3 当前用 Ascend V1 与
  `npu_sparse_flash_mla`，不能直接启用该实现。
- 当前 `MooncakeHybridConnector` 按 12 组（无 DSpark）传输，G0 全局、G1
  C2 环、G2–G11 每组 4 层 SWA。P 的实测回执为
  `[1125,1,2,2,2,2,2,0,0,0,0,0]`。保留物理分组能先排除布局改动。

相关原始证据在 `../../evidence/ced_p_mask_34fdf08/`；设计的前提是 BF16 KV，
尚未纳入 AscendStore DRAM 池和 KV8。
