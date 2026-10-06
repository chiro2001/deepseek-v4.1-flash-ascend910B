# 通信结构解剖 + DBO 的"eager 不能当判据"（2026-10-06 晚）

> 承接 `docs/DBO-RUNTIME-VERDICT-20261006.md`（DBO 跑通、eager 0.52×）。
> 本轮把"通信到底是什么"查清，并把 yield 钩子接到唯一咽喉点做实验。全部为【实测】。

## 0. 一页纸

| 结论 | 证据 |
|---|---|
| 每步通信 = **91 次 allReduce**（tp8-class profile，72 步窗口） | `AivKernel` 与 `hcom_allReduce__503_*` 是**同一批事件**（union 重叠 4.114 = min） |
| 通信 **100% 暴露**：窗口内 AIC 忙 **0.000 ms** | `comm ∩ AIC = 0.000`、`comm ∩ AIV(纯计算核) = 0.025` |
| 全部通信**只有一个咽喉点** | 都经 `torch.ops.vllm.all_reduce` → `GroupCoordinator._all_reduce_out_place` |
| 把 ubatching 的 yield 钩子挂到该点**确实生效**（10.4 万次让出） | `[DBO-ALLYIELD] allreduce-yields=103600` |
| 但 **eager 下 DBO 仍然 0.52×**（挂钩子 0.47×） | §3 表 |
| **为什么：eager 是 Python 派发受限**，拆 2 个 ubatch ⇒ 派发次数×2 | graph conc=1 = 25.5 tok/s vs eager conc=1 = 6.6 tok/s（3.9×） |
| ⇒ DBO 的成败**只能在图模式下判定** | §4 |
| 顺带修掉一个影响面更大的坑：**NPU 的 set_stream 不更新 vLLM 的 current_stream 缓存** | §5 |

## 1. 通信 = 91 次 allReduce/步，且完全暴露

工具：`~/tmp/prof_comm_split.py`、`~/tmp/prof_comm_identity.py`（本轮新写，读 `kernel_details.csv`）。
样本：`armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_.../ASCEND_PROFILER_OUTPUT`（tp8-class，72 步窗口，STEP=40.02 ms）。

```
AivKernel union 4.114 | hcom union 4.215 | 二者重叠 4.114  ⇒ 同一批事件? YES
通信窗口里到底谁在忙：
  AIC 在 comm 窗口内的时长        : 0.000 ms/步
  AIV(真实计算核) 在窗口内        : 0.025 ms/步
  其它非计算核 kernel 在窗口内    : 4.215 ms/步
```

⇒ 之前 `prof_exposed_comm.py` 报的"通信暴露 2.504 ms（10.2%）"**是对的**，
不是统计口径问题：通信期间 AIC 真的一动不动。

**算子身份**：`AivKernel` 的 `Accelerator Core = COMMUNICATION`（6553 行全部如此），
它与 `hcom_allReduce__503_*` 成对出现（同 st、同 dur）⇒ 一个是 AIV 侧驱动 kernel，
一个是 HCCL 侧记录，**同一批 91 次/步的 allReduce**。

**发射点**（`operator_details.csv` 里这些 op 的名字是 `vllm::all_reduce` / `c10d::allreduce_` / `HcclAllreduce`）：

```
torch.ops.vllm.all_reduce(tensor, group_name)
  → vllm/distributed/parallel_state.py: all_reduce()
    → GroupCoordinator._all_reduce_out_place()
      → device_communicator.all_reduce()
```

调用者（`grep` 得到）：

| 位置 | 说明 |
|---|---|
| `vllm/model_executor/layers/linear.py:1767` | `RowParallelLinear`（attention o_proj / dense down_proj / MoE down_proj） |
| `vllm_ascend/ops/fused_moe/fused_moe.py:147 / 188 / 206` | MoE 的 shared / routed / final 三处归约 |
| `vllm_ascend/ops/vocab_parallel_embedding.py:257` | 词表并行 |

**另有两条"每步 1 次"的独立集合通信**（与上面 91 次不同）：
`allgatherAicpuKernel`（77 次/profile，中位 356 µs）、`allreduceAicpuKernel`（81 次，均值 490 µs）。

## 2. yield 钩子实验：挂上了，也确实触发了

`~ /tmp/fix_allyield.py` 在**唯一咽喉点** `all_reduce()` 里按 `V41_DBO_ALLYIELD=1` 插：

```python
dbo_yield_and_switch_from_compute_to_comm()
try:
    out = group._all_reduce_out_place(tensor)
finally:
    dbo_yield_and_switch_from_comm_to_compute()
```

（未开 ubatching 时这两个函数自带判空 ⇒ no-op，安全。）

实测：`[DBO-ALLYIELD] allreduce-yields=103600`（TP0/TP1 各一份）⇒ **机制真的在跑**，
即"每次 allreduce 都把计算流让给兄弟 ubatch"。

## 3. 三组 A/B（tiny TP2，DCP=1，prompt=1024，out=64，同一批 prompt）

命令：`python3 tools/bench_concurrency.py --base-url http://127.0.0.1:19310 --concurrency 1,4,8 --prompt-tokens 1024 --output-tokens 64`

| conc | **图模式基线** | eager 基线 | DBO eager（无钩子） | DBO eager（挂钩子） |
|---:|---:|---:|---:|---:|
| 1 | **25.9** | 6.7 | 6.7 | 6.6 |
| 4 | **66.0** | 24.0 | 12.6（0.53×） | 11.6（0.48×） |
| 8 | **84.5** | 45.6 | 23.7（0.52×） | 21.5（0.47×） |

（单位 tok/s 总吞吐；数据：`~/tmp/base_graph_q.json`、`~/tmp/base_eager_q.json`、
`~/tmp/dbo_eager_q.json`、`~/tmp/dbo_yield_conc.json`）

读法：

1. **同一 eager 体制内**：DBO 把吞吐砍到 0.52×。原因是拆成 2 个 ubatch 后，
   **算子派发次数×2**，而 eager 的每步成本正是被 Python 派发主导的。
2. **eager vs 图**：同一配置 conc=1，图 25.9 vs eager 6.6 ⇒ **3.9×**。
   也就是说 eager 下 84% 的时间花在 Python 侧；在那种体制里测"通信重叠"
   等于在噪声里找信号。
3. ⇒ **DBO 有没有前途，图模式之前的任何数字都不能作为判据。**

## 4. 下一步（唯一有意义的实验）

把 ubatch 双流分支**在捕获期内**走 stream 版（`_run_ubatches_graph`），让一张 ACL graph
完整捕获 fork/join（规则已由 `tools/tiny_graph_ms.py` 验证：fork event 必须 record 在捕获根流）。

本轮已写好 `~/tmp/fix_dbo_graph_capture.py`（按 `torch.npu.is_current_stream_capturing()`
选择路径），但**捕获检测没生效**：实测捕获期该 API 返回 False（在独立 `torch.npu.graph(g)`
上下文里返回 True）⇒ 捕获期仍走了 threading 版，随后捕获阶段报 aicore exception(507015)。

【推断】vLLM/Ascend 的 capture 走的不是 `torch.npu.graph()` 那层上下文，
所以要在**外层 wrapper**（`BreakableACLGraphWrapper` / `capture_model`）打标记，
而不是靠 stream 状态查询。这是下一步的第一件事。

## 5. 顺带修掉的坑：NPU `set_stream` 不更新 vLLM `current_stream()`

```
vllm/utils/torch_utils.py:662 current_stream()
  → 读线程本地 _current_stream_tls.value
  → 只有它 patch 过的 torch.cuda.set_stream 会更新这个值
实测：torch.npu.set_stream(s2) 之后
      torch.npu.current_stream() == s2  →  True
      vllm.utils.torch_utils.current_stream() == s2  →  False（返回旧流）
```

后果（本轮真实踩到）：ubatching 的
`assert current_stream() == self.comm_stream` **必然失败** ——
第一次 [DBO-ALLYIELD] 实验就是死在这条断言上（`AssertionError`，engine init 阶段）。

修法（`~/tmp/fix_npu_stream_tls.py`，只改我们自己的 `npu_ubatch_wrapper.py`）：
在 `NPUUBatchContext.update_stream()` 里同时写一次
`vllm.utils.torch_utils._current_stream_tls.value = stream`。

> 这个坑的影响面不止 ubatching：**任何在 NPU 上依赖 vLLM `current_stream()` 的逻辑
> 都会看到陈旧流。** 值得单独记住。

## 6. 环境状态

| 项 | 状态 |
|---|---|
| tiny | 已恢复常规配置：health=200、`Supported tasks: ['generate']`、KV 容量 3,403,198、`enable-dbo` 命中 0 |
| 容器内核补丁 | `parallel_state.py` 已回退（`DBO-ALLYIELD` 命中 0）；`npu_ubatch_wrapper.py` 保留（未开 DBO 时不被使用） |
| tp8k5（chip8–15） | **全程未动**，health=200 |

## 7. 复现

```bash
# 通信身份与暴露
ssh a3-21 'docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 prof_comm_identity.py /opt/dsv41/results/ab_mkc_1001_215615/prof_f"'
# yield 钩子 A/B
ssh a3-21 'docker exec dsv41-tinyspark python3 /tmp/fix_allyield.py && bash ~/tmp/launch_tiny_eager_yield.sh'
# 恢复常规 tiny
ssh a3-21 'bash ~/tmp/launch_tiny_prof.sh'
```
