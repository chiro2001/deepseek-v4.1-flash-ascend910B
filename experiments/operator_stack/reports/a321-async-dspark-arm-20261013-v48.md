# DSpark async-scheduling 候选臂：全门通过、短请求持平、长请求待判（v48）

2026-10-13 · a3-21 physical8–15 · 完整40层/hidden5120/384top6/W4A8/Engram int8/视觉/MTP3 · HCCL strict

本轮把 vLLM 的 **async scheduling** 接到 DSpark 正式栈（env 门控、默认关），
在 AA 审计 harness 上完成了从"能否跑"到"数值等价 + 计时"的全链条验证。
目标 ≤20ms/step **未达成**；本报告是候选臂的实测账。

## 1. 上游依据与门控

- `vllm/config/vllm.py` 对 `async_scheduling` 的硬校验**显式放行 `dspark`**
  （`method not in (...EAGLE..., "draft_model", "dspark")` 才拒绝）；`MultiprocExecutor.supports_async_scheduling()=True`。
- `vllm-ascend model_runner_v1.py` 实现 `use_async_spec_decode` 与 optimistic seq_lens 输入准备；
  `_bookkeeping_sync` 的 async 分支返回**空** token 列表（这是计数器原先拒绝 async 的原因）。
- 我方三处门控（全部默认关、同步路径字节不变）：
  `formal_dspark_counters.install_runner`（env `DSPARK_ASYNC_COUNTER=1` 才放行 async 并进入 async 模式）、
  `dspark_formal_prep/worker.py`（两个 env 同时置位才放行）、`contract.model_kwargs`
  （`async_scheduling` 由 `DSPARK_ASYNC_SCHEDULING` 决定）。API 桥保持冻结未改。

## 2. async 臂的计数语义（CPU 契约 8/8 通过）

- 轮次 = 每请求**实际执行的 target/verifier decode 轮**（含 async 特有的 overshoot 轮，见 §4）；
  `A = 客户端 tokens-1 / rounds`；窗口 = `[terminal-prefill bookkeeping, max(client_end, terminal fence)]`。
- 每轮 CPU token 不物化（stock async ABI），per-round 接受数显式记 `None`、`async_per_round_tokens_available=False`。
- 终止边界：优先在引擎为本请求发出的 **zero-token cleanup execute** 上做设备 fence（含 terminal 已提交 draft），
  否则 `finish()` 前做 post_client fence；随后执行与同步臂相同的 TP8 CPU Gloo join；晚于 client 50ms 以上 fail-closed。
- `test_async_counter_cpu.py`（容器内）：8/8（含占位符轮、overshoot 轮计数、fence 缺失/rank 分歧/eager draft 拒绝、sync 路径回归）。

## 3. AA 审计臂结果（47 token 请求，6 个计数请求）

`formal_dspark_native_graph_aa_v16_async`：exit0、`passed=True`、`async_reference_passed=True`；
**6/6 请求 token 与同步参考逐位一致、Top5 logprob 最大差 0.0**（async 语义正确）。

| 臂 | ms/step | A | tok/s | 说明 |
|---|---:|---:|---:|---|
| sync v4（同 harness） | 44.885 | 5.75 | 128.10 | 8 轮/请求 |
| sync v6（同 harness） | 46.603 | 5.75 | ~123.4 | 8 轮/请求 |
| **async v16** | **45.875** | **5.111** | **111.41** | 9 轮/请求（含 overshoot） |

## 4. 结构发现：稳态更快、但每请求多一轮

逐轮（rank0，宿主探针）：

- **稳态轮**：async ≈ 39.7ms（execute 12.3 + sample 26.7 + draft 2.8）；
  同步 ≈ 42.9ms（execute 11.3 + sample 28.2 + draft 2.9）⇒ **async 稳态快 ~3.2ms/轮**（host 重叠确实生效）。
- **每请求多一轮**：async 在客户端已收到输出后仍执行一轮已被丢弃的 target decode（overshoot），
  该轮是真实设备工作；本臂按"实际执行轮"计入分母（A 由 5.75→5.11），窗口延至其 fence。
- 结论：短请求（8–9 轮）下 overlap 收益（~26ms）小于 overshoot（~40ms）+ 首轮略慢，净持平略负；
  按稳态差线性外推，**2K/256（约86轮/请求）时 async 可能净赢 ~6–8%**。

## 5. 证据

- `evidence/formal_a321/dspark_native_graph_aa_v16_async/`：`launched.json`、`result.json`、
  `async_reference_comparison.json`（6/6 token_equal、delta 0.0）、`requests.json`、`pair-0-a_compact.json`。
- 源码与门控：`evidence/formal_a321/async_arm_prep/`（NOTE.md + counters/bench/contract/worker/launcher 冻结件 + SHA256SUMS）。
- 远端快照 `src_dspark_native_async_v1`（freeze 18 文件）。

## 6. 判定与下一步

- async 语义**已验证正确**（token/logprob 逐位等价），拥有可复用的计数器/启动器/门控基础设施。
- 短请求无收益；**判定其价值取决于 2K/256 长请求**：已启动同步 `sync256_v1` 与随后的 async @256 对照
  （同 harness、同快照、同提示，token 逐位门 + 三元组）。若长请求净赢稳定 ≥3%，再考虑为正式 API 桥
  重新冻结 async 通道；若不达，则按"已验证但无收益"归档，回到宿主/目标图优化主线。
