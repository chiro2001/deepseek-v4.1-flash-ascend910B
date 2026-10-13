# DBO（ubatching）第一次端到端跑通与 A/B 裁决（2026-10-06 晚）

> 承接 `docs/DBO-A3-PROGRESS-AND-BLOCKER-20261006.md`（错误链 1–12）。
> 本轮把 DBO 从"起不来"推进到**真机端到端跑通**（eager），并拿到第一份 A/B：
> **eager 下 conc=4/8 的总吞吐只有基线的 0.52×**。§1–§4 全部为【实测】。

## 0. TL;DR

| 结论 | 依据 |
|---|---|
| DBO 现在**真的会运行**（不再空转） | 运行期出现 `[DBO-CAT*]`（说明 `_run_ubatches` 被调用）；`should_ubatch` 在真机为 True |
| eager 端到端**能跑完** | conc=1/4/8 各 8/8 成功（修复后） |
| **但吞吐净亏 ~2×**（eager，conc=4/8） | §2 表：0.53× / 0.52× |
| 图模式 conc≥4 **挂死** | §4 |
| 根因：**全仓零个 `dbo_yield*`**，重叠从未发生 | §3 |

## 1. 错误链 12 → 16：本轮修掉的 4 个阻塞

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 13 | AICPU kernel `VllmQuantLightningIndexerMetadata` 抛 22007（`QuantLightningIndexerV2Metadata` 修复后换了个 kernel） | 上一轮把 **dspark draft** 的 device-metadata 也一起关了。draft 不在 ubatch 线程里跑，本来就是正常路径 | `dsa_v1` 的守卫改成可配（`V41_DBO_NODEVMD_SCOPE`，默认 0 = 恢复原行为），只对 v41 主模型关 |
| 14 | `KeyError: (DeviceMetadataStage.ATTENTION, id)` frontier | v41 改成 **eager 就地构建** metadata 后，消费侧仍无条件 `wait_for_device_metadata(...)`（该 frontier 从未 submit） | `_publish_task` 的 eager 分支登记 buffer id；三处 wait 改走 `_v41_wait_device_metadata()` 跳过 |
| 15 | engine 报 `Supported tasks: []` ⇒ **服务起来了但所有 generate 路由 404**（`/v1/completions`、`/v1/chat/completions`、`/v1/responses` 全 404，openapi 只剩 9 条） | Python 3.12 的 runtime_checkable Protocol `isinstance` 走 `getattr_static`，**不经过** `NPUUBatchWrapper.__getattr__` ⇒ `is_text_generation_model(wrapper)=False`。最小复现：plain=True / wrapped=False（proto attrs = `['compute_logits','embed_input_ids','forward']`） | `NPUModelRunner.get_supported_tasks()` 里临时把 `self.model` 解包到最内层 |
| 16 | `RuntimeError: Device metadata frontiers changed for an existing full-graph batch descriptor`（真机第一次 conc=4） | 上游只在 `data_parallel_size > 1` 时用 `coordinate_batch_across_dp` 决定 `should_ubatch`，**DP=1 恒为 False**；而我们的 capture/dummy 路径按阈值 ubatch ⇒ capture 注册 2 个 frontier、真机只提交 1 个 | DP=1 时补同一公式（`check_ubatch_thresholds`，与 `_dummy_run` 完全一致） |

> 第 16 条附带一个重要事实：**在这条修复之前，DP=1 下真机一帧都没有 ubatch 过**——
> 之前所有"DBO 起服成功"的验证，实际跑的都是全批路径（ubatching 只发生在 capture/dummy 里）。

另外给 `DeviceMetadataExecutor.submit` 加了诊断打印（首次登记 / 不一致时打印 submitted vs expected），
保留在 `fix_dm_dbg.py` 里，后续调 device-metadata 还会用到。

### 1.1 上游 PR #11273 的两个缺陷（本轮实测）

1. wrapper 遮 capability（第 15 条）——PR 的 `NPUUBatchWrapper` 同样会遮，它没有对应的 unwrap；
2. 它在 vllm-ascend main 上缺 `_build_attention_metadata` 的 ubatch 循环（上一轮已记录）。

## 2. A/B 实测（eager，tiny TP2，1024 prompt / 64 out，同一批 prompt）

命令（两侧完全一致）：

```bash
python3 tools/bench_concurrency.py --base-url http://127.0.0.1:19310 \
        --concurrency 1,4,8 --prompt-tokens 1024 --output-tokens 64
```

| conc | 基线 总吞吐 | DBO 总吞吐 | **比值** | 基线 单流 | DBO 单流 | 基线 TTFT | DBO TTFT |
|---|---|---|---|---|---|---|---|
| 1 | 6.7 tok/s | 6.7 tok/s | **1.00×** | 6.6 | 6.6 | 0.16 s | 0.32 s |
| 4 | 24.0 tok/s | 12.6 tok/s | **0.53×** | 6.1 | 3.2 | 0.55 s | 0.95 s |
| 8 | 45.6 tok/s | 23.7 tok/s | **0.52×** | 6.0 | 3.1 | 0.88 s | 1.57 s |

读法：

* conc=1 **不触发** ubatch（8 tok < 阈值 32）⇒ 持平。这也说明**非 ubatch 路径没有回归**。
* conc=4/8 触发 2 ubatch ⇒ 单流速度腰斩，TTFT 也近乎翻倍（prefill 同样被切成 2 段）。
* 同配置**图模式**（DP=1 修复前、conc=1 不 ubatch）= 37.3 tok/s；eager 只有 6.6 tok/s
  ⇒ eager 本身带约 5.6× 的 Python 侧开销。
  **因此本表只能证明"当前实现下 ubatching 是净亏"，不能直接量化图模式下的代价**——
  图模式 conc≥4 挂死（§4），暂时量不到。

数据：`~/tmp/base_eager_q.json`（基线）、`~/tmp/dbo_eager_q.json`（DBO）、`~/tmp/dbo_conc14b.json`（图模式 conc=1 参考）。

## 3. 根因：DBO 的重叠机制在我们的代码里不存在

上游协议（`vllm/v1/worker/ubatching.py`）里写得很直白：

> "It is critical for correctness that only one thread is running at a time."

即 **DBO 的"重叠"不来自两个线程真并行**，而是来自**在通信点主动让出**：
一个 ubatch 走到 all-to-all/MC2 时调用 `dbo_switch_to_comm` /
`dbo_yield_and_switch_from_compute_to_comm`，让另一个 ubatch 的**计算**填进这段通信时间。

实测（本轮）：

```bash
grep -rn "dbo_yield|dbo_switch_to_comm|dbo_switch_to_compute|dbo_register_recv_hook" \
     vllm_ascend/ --include=*.py     # ⇒ 0 命中（含全部 fused_moe/*.py）
```

⇒ 现在的实际行为 = **把一次前向拆成两次串行前向**：MoE all-to-all / 固定开销 ×2、重叠 0。
这与 §2 的 0.52× 完全吻合。

【推断】这也解释了为什么上游 PR #11273 长 prompt 只有 26% token-identical、
以及为什么"两个 ubatch 只是把 batch 切小"在吞吐上一定是负收益。

## 4. 图模式挂死（未定位到具体算子）

现象（FULL_DECODE_ONLY，conc=4 的首个 2-ubatch decode step）：

* 4 个请求全部 Running，`Avg generation throughput` 归零；
* EngineCore 每 60 s 打一次
  `No available shared memory broadcast block found in 60 seconds … processes are hanging`
  ⇒ **进程在等 NPU**，不是 Python 层死锁；
* 已排除：frontier 校验（本轮修复后 mismatch=0）、device-metadata AICPU、KV 容量。

【推断】capture 时 ubatch 的两个分支跑在**非捕获流**上：`_run_ubatches()` 走 Python 线程
（`torch.npu.set_stream`），而已经写好的 `_run_ubatches_graph()` **从未被调用**，
于是图里只留下不完整的双流结构，replay 时等一个永远不会来的事件。
要连线必须遵守 `tools/tiny_graph_ms.py` 验证过的规则：**fork event 必须 record 在捕获根流上**。

## 5. 给路线图的判断

1. **线 B（ubatching）在当前实现下不应作为吞吐方案推进**：收益符号是负的（0.52×），
   而且连"能不能算对"都还没验证（本轮的 2 个 ubatch 结果是 dummy 权重，只做了跑通性验证；
   四道验收门**一道都没过**）。
2. 要救它，必须先做下面 (a)，再考虑 (b)：
   * **(a) 插钩子**：在 Ascend MoE 的 all-to-all / MC2 前后调用
     `dbo_switch_to_comm` / `dbo_yield_and_switch_from_compute_to_comm`，
     然后用 profiler 验证 **`AIC ∩ COMM` 从 0.000 变成正值**——这是 DBO 能产生收益的**充要证据**；
   * **(b) 图模式**：把 ubatch 双流按已验证的 fork/join 规则捕进一张图。
3. 在 (a) 证明"重叠真的出现"之前，不建议再投入 (b) 和更多精度验证。
   与此同时，线 A 的暴露度排序（MoE w1/w3 3.107 ms、HcPre 1.987、通信 2.504 等）才是**已验证过暴露度**的收益来源。

## 6. 产物与复现

| 类别 | 路径 |
|---|---|
| 起服（图模式 DBO） | `~/tmp/launch_tiny_dbo.sh` |
| 起服（eager DBO / eager 静默 / eager 基线） | `launch_tiny_dbo_eager.sh` / `launch_tiny_eager_q.sh` / `launch_tiny_eager_base.sh` |
| 修复脚本（已进仓 `tools/`） | `dbo_fix_draft_devmd_scope.py`、`dbo_fix_v41_eager_wait.py`、`dbo_fix_caps_supported_tasks.py`、`dbo_fix_dp1_should_ubatch.py`、`dbo_dmdbg_diagnostics.py` |
| 数据 | `~/tmp/base_eager_q.json`、`~/tmp/dbo_eager_q.json`、`~/tmp/dbo_conc14b.json` |

恢复常规 tiny：`bash ~/tmp/launch_tiny_prof.sh`（本轮的常规态核验见 §7）。

## 7. 环境状态

* **tp8k5（chip8–15）全程未动**（本轮只重启 tiny 的容器进程）。
* tiny 结束时按 §6 的命令恢复常规配置（health=200、`Supported tasks: ['generate']`、
  KV cache size 3,403,198 tokens、inner 脚本里 `enable-dbo` 命中数 0）。
