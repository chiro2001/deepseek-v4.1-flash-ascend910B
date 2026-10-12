# async 候选臂：上游依据、我方门控与 CPU 证据（2026-10-13）

## 为什么走 async

逐轮预算（AA rank0 探针，8 轮）：轮窗口 40.3ms = execute 11.58（prep 2.24 + spec-meta 0.42 +
目标图提交/前段 8.92）+ sample 28.57（D2H 1.11 + draft 编排 3.68 + 其余 23.78）+ gap 0.15。
设备侧剖析显示 pre_target 窗口（15.35ms 跨度）只有 2.84ms kernel，其余是主机胶水 ⇒ **launch-bound**。
vLLM async scheduling 的设计正是让"下一步输入准备"与"当前步设备执行"重叠；对无 DSpark 的 mask 线
曾带来 20.4→19.4 的收益，DSpark 线的主机暴露更大（~7ms/轮 + pre_target 空闲），预期收益更高。

## 上游依据（本栈 vLLM 0.27.1）

- `vllm/config/vllm.py`：`async_scheduling` 的硬校验名单**显式包含 `dspark`**：
  `... and self.speculative_config.method != "dspark")` 才拒绝 ⇒ 官方支持 async+DSpark。
- `vllm/v1/executor/multiproc_executor.py:526 supports_async_scheduling() -> True`（我们的后端是 `mp`）。
- `vllm-ascend worker/model_runner_v1.py`：`use_async_spec_decode = use_async_scheduling and num_spec_tokens`，
  且 `_prepare_inputs` 已有 async 推测解码的 optimistic seq_lens 路径（用有效采样计数在 CPU 修正）。
- `_bookkeeping_sync` 的 async 分支明确返回**空** `valid_sampled_token_ids`（token 留在设备），
  这是本仓计数器原先拒绝 async 的原因（"Async output placeholder ABI…"）。

## 我方三处门控（全部 env 门控、默认关、同步路径字节不变）

| 位置 | 原行为 | async 臂行为（`DSPARK_ASYNC_COUNTER=1` + `DSPARK_ASYNC_SCHEDULING=1`） |
|---|---|---|
| `formal_dspark_counters.install_runner` | async ⇒ 直接 raise | 允许；进入 async 模式（记 `async_mode`） |
| `parallel_tiny/dspark_formal_prep/worker.py` | `assert not async_scheduling` | 允许（两个 env 同时置位才放行） |
| `dspark_api_counter_bridge_cpu._guard_config` | async ⇒ raise | **未改**（API 桥保持冻结；async 先用 AA 路线验证） |

计数器 async 语义（CPU 契约 6/6 通过，`test_async_counter_cpu.py`）：

- 轮次计数不变：每次 `execute_begin`（有 scheduled tokens）为一个真实 target/verifier 轮；
  `A = 客户端输出 tokens-1 / rounds`；三元组自洽校验保留。
- 每轮 CPU token 不再物化（stock async ABI），per-round accepted 记为显式 `None`
  （`acceptance_zero_diagnostic_required=False` 并标注 `async_per_round_tokens_available=False`）。
- 终止边界：优先在引擎为本请求发出的 **zero-token cleanup execute** 上做设备 fence（含 terminal 已提交 draft），
  否则在 `finish()` 前做 post_client fence；随后执行与原同步臂相同的 TP8 CPU Gloo join。
  报告窗口 = `[decode_start, max(client_end, fence)]`，并记录 `terminal_fence_phase` 与
  `terminal_fence_delta_ms`；fence 晚于 client_end 超过 50ms 时 fail-closed。

## 交叉数值门

AA 臂以 `--sync-reference formal_dspark_native_graph_aa_v4/requests.json` 为参照：
同 tag/seed 的 **256×8 token 必须逐位一致**，Top5 logprob keys 必须一致、`max delta < 1e-3`，
否则该臂判失败（`async_reference_comparison.json`）。路由导出仍是同步审计臂（v4）的属性，
async 臂不声称 routes/接受前缀证据。

## CPU 证据

- `test_async_counter_cpu.py`：容器内 `ASYNC_COUNTER_CPU_CONTRACTS_PASSED 6/6`
  （含轮计数、窗口扩展、rank 分歧拒绝、缺 fence 拒绝、eager draft 拒绝、sync 路径仍拒绝空 token）。
- 容器内 stock `test_counters` 与本仓基线（hostprobe 快照）**失败集完全一致**（3 failures + 2 errors），
  即相对基线无回归；该 stock 测试文件本身在当前快照下即为陈旧（缺 terminal join fixture）。
