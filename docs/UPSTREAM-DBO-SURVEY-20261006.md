# ★ 上游到底有没有人做「micro-batch 级并行」？—— 有，而且是一整条线

> 起因：用户问「看一下 vllm/vllm-ascend 真的没人做过这个吗，因为看起来收益是比较大的」。
> 结论：**有人做过，而且做了很久** —— 但那是 **DBO（通信 ‖ 计算）**，
> **不是**我们想的 **AIC ‖ AIV**。Ascend 侧有 PR，**开了 3 个月没人 review**。
> 全部为【实测·代码】/【实测·上游数据】，下附行号与 issue 号可复核。

## 0. 三句话

1. **vLLM 上游有 DBO（Dual Batch Overlap）**：`enable_dbo` + `ubatch_size`，
   **N 条 CPU 线程各驱动一个 micro-batch**，重叠的是 **MoE all-to-all 通信 ‖ 另一个 ubatch 的计算**。
   vLLM 自己的 **DeepSeek V4 / Kimi K3 实现都在用**。
2. **vllm-ascend 显式把它关掉**（`platform.py:901-912`），
   但有一个 **开着的 PR #11273**（2026-07-01 至今无人 review）要打开它，
   实测 **+3.5% ~ +9.9%（conc≥4）**、**−5.0%（conc=1）**，且 **Phase 1 是 eager-only**。
3. **我们要的「AIC ‖ AIV 跨 batch pingpong」上游没人做** ——
   但 vLLM 的 ubatching 骨架（`ubatching.py` + `*_ubatch_wrapper.py`）**正是能直接复用的地基**。

---

## 1. vLLM 上游：DBO 是一等公民

### 1.1 配置面（我们的容器 = vLLM **0.27.1**，全都已带）

```python
# vllm/config/parallel.py
enable_dbo: bool = False                    # :211
ubatch_size: int = Field(default=0, ge=0)   # :213   ← 通用 N 路，不限于 2
dbo_decode_token_threshold: int = 32        # :216
dbo_prefill_token_threshold: int = 512      # :221

@property
def use_ubatching(self) -> bool:            # :554
    return self.enable_dbo or self.ubatch_size > 1
@property
def num_ubatches(self) -> int:              # :558
    return 2 if self.enable_dbo else self.ubatch_size
```

容器内实测：

```
vllm: 0.27.1
/vllm-workspace/vllm/vllm/v1/worker/: gpu_ubatch_wrapper.py  ubatch_utils.py  ubatching.py   ← 全在
```

### 1.2 机制：**N 条 CPU 线程 + 每 ubatch 两条流**（`vllm/v1/worker/ubatching.py`，241 行）

这套机制的精髓就是"CPU OOO"的类比 —— **用线程把独立工作显式表达出来**：

```python
class UBatchContext:
    def __init__(self, id, comm_stream, compute_stream, ready_barrier,
                 cpu_wait_event, cpu_signal_event,
                 gpu_comm_done_event, gpu_compute_done_event, ...):
        self.comm_stream = comm_stream        # ← 每个 ubatch 有自己的通信流
        self.compute_stream = compute_stream  # ← 和计算流

    def _cpu_yield(self):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()

    def yield_and_switch_from_compute_to_comm(self):
        self._signal_compute_done(); self._cpu_yield()
        self.update_stream(self.comm_stream); self._wait_compute_done()
```

`gpu_ubatch_wrapper.py:262-294` 启动线程：

```python
for metadata in ubatch_metadata:
    thread = threading.Thread(target=_capture_ubatch_thread, args=(results, metadata))
    thread.start()
self.ready_barrier.wait()
with torch.cuda.graph(cudagraph, stream=compute_stream, pool=self.graph_pool):   # ← 图内也能跑
    ...
```

关键特征：

* **线程间协作式让渡**（同一时刻只有一条 CPU 线程在跑）—— 避免 host 侧争抢；
* **每 ubatch 两条 GPU 流**（comm + compute）；
* **支持 CUDA Graph 捕获**（`torch.cuda.graph(...)` 包住所有线程的 join）；
* `dbo_current_ubatch_id()` 让**模型代码**知道自己在哪个 ubatch（用于 per-ubatch 状态）。

### 1.3 ★ 它重叠的到底是什么：**MoE 通信 ‖ 另一个 ubatch 的计算**

全部 yield 点都在 `fused_moe` 里（`prepare_finalize/deepep_ht.py`、`modular_kernel.py`）：

```python
# deepep_ht.py:118-120
# We yield before launching the dispatch kernel since the dispatch
# kernel will block the CPU so we want to queue up all the compute
# for the other ubatch before the dispatch kernel starts.
dbo_yield_and_switch_from_compute_to_comm()
... self.buffer.dispatch(...) ...
# :374
dbo_yield_and_switch_from_compute_to_comm()
... self.buffer.combine(...) ...
```

**⇒ DBO 把「ubatch 0 的 MoE dispatch/combine（DeepEP all-to-all）」与「ubatch 1 的计算」重叠。**

**不是 AIC ‖ AIV。** 这一点必须分清，否则会把期望搞错（见 §4）。

### 1.4 vLLM 自己的 DeepSeek V4 就在用

```python
# vllm/models/deepseek_v4/nvidia/model.py:86
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
# :482  EPLB 需要 per-ubatch 的 token 计数
num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
```

（`kimi_k3/nvidia/model.py:114` 同样用法。）
⇒ **上游 DSV4 是"DBO-aware"的**，而我们的 `vllm_ascend` 路径完全没这条线。

---

## 2. vllm-ascend：显式禁用（有守卫代码）

```python
# vllm_ascend/platform.py:901-912
if getattr(vllm_config.parallel_config, "enable_dbo", False):
    logger.warning(
        "Parameter is currently ignored on Ascend. parameter=enable_dbo, action: resetting to False. "
    )
    vllm_config.parallel_config.enable_dbo = False

ubatch_size = getattr(vllm_config.parallel_config, "ubatch_size", 0)
if ubatch_size != 0:
    logger.warning(
        "Parameter is currently ignored on Ascend. parameter=ubatch_size, value=%d, action: resetting to 0. ",
        ubatch_size,
    )
```

这道守卫来自 **#8471 / #8507（2026-04-21，已合入）**。

**另外**：`grep limit_core_num vllm_ascend/` = **0 个文件** ⇒
官方 recipes 那套控核在 vllm-ascend 里同样**没有**。

---

## 3. ★ 有一个开着的 PR 要打开它：#11273

**`[Feature] Enable --enable-dbo on Ascend NPU (NPUUBatchWrapper + ascend_split_attn_metadata)`**

| 项 | 值 |
|---|---|
| 开于 | **2026-07-01** |
| 最后活动 | 2026-09-17 |
| 状态 | **open，从未有人类 review** |
| 作者 | `pop-aiminer`（反复 rebase，两次求 review） |
| 机器人 | 反复 "This pull request has conflicts"（作者已确认 mergeable） |

### 3.1 它做了什么

* `platform.py`：**移除强制 `enable_dbo=False`**；
* `worker/worker.py`：`num_ubatches = 2 if enable_dbo else 1`；
* **新增 `worker/npu_ubatch_wrapper.py`（266 行）**：上游 `gpu_ubatch_wrapper.py` 的 NPU 移植
  （`torch.cuda.*` → `torch.npu.*`；**跳过 CUDAGraph / SMControlContextManager**）；
* `model_runner_v1.py`：`check_ubatch_thresholds` + **`ascend_split_attn_metadata()`**
  （上游 `_make_metadata_with_slice` 硬编码 `CommonAttentionMetadata`，会丢掉
  `attn_state` / `prefill_context_parallel_metadata` / `kvcomp_metadata` 等 Ascend 专有字段）；
* `create_attn_groups`：`use_ubatching=True` 时建 `num_ubatches` 个 metadata builder；
* `load_model`：`use_ubatching` 时包 `NPUUBatchWrapper` 并**跳过 `ACLGraphWrapper`**。

### 3.2 ★ 实测数据（910B2C + Qwen1.5-MoE-A2.7B-Chat，bf16，eager）

| 场景 | DBO off (tok/s) | DBO on (tok/s) | 差异 |
|---|---:|---:|---:|
| 1 req, 64 tok | 46.26 | 43.96 | **−5.0%** |
| 4 req, 64 tok | 84.95 | 88.90 | **+4.6%** |
| 16 req, 64 tok | 392.52 | 406.23 | **+3.5%** |
| **4 req, 256 tok** | 65.82 | 72.33 | **+9.9%** |
| 16 req, 256 tok | 218.17 | 229.66 | **+5.3%** |

* **conc≥4 才正收益；conc=1 是 −5.0%**；
* 精度：greedy 短 prompt **18/18 token-identical**；
* ⚠️ **触发 ubatch 切分的 23 条长 prompt（prefill > 512）只有 26.1% token-identical** ——
  作者明说 "split path has numerical drift"。

### 3.3 已知限制（作者自己列的）

* **Phase 1 是 eager-only（无 ACL Graph）**；
* **PCP/DCP 不支持**（`split_attn_metadata` 不重建 `prefill_context_parallel_metadata`，
  被 `jiangkuaixue123` 在 #4894 里指出，作者承认要加 guard）；
* `max_model_len=2048` 的小模型；
* 长 prompt 有数值漂移。

---

## 4. ★ 与我们的想法对比：**不是同一件事**

| | **vLLM DBO（#11273）** | **我们的 AIC ‖ AIV pingpong** |
|---|---|---|
| 重叠的双方 | **MoE all-to-all 通信 ‖ 另一 ubatch 的计算** | **AIC（cube）‖ AIV（vector）** |
| ubatch 数 | 2（`ubatch_size` 可到 N） | 2+ |
| 机制 | 2 条 CPU 线程 + 每 ubatch 2 条 GPU 流 | 需要同样的线程/流骨架 |
| 被重叠的资源在 profile 里的占比 | 通信 **4.22 ms/步 = 11%** | AIC 空转 **17.97 ms/步 = 45%** |
| Ascend 现状 | **有 PR，未合入**，eager-only | **无人做** |

**这解释了两边的收益差**：

* PR 实测只有 **+3.5~9.9%** —— 因为它重叠的是**通信**，
  而我们 profile 里通信只占 11%，且它 **eager-only**（我们已测：无图时 host 开销吃掉一切，0.24×）；
* 我们的目标（AIC 55% → 提到更高）对应的**理论上界是 1.67×**（见 `docs/PINGPONG-AIC-AIV-VERDICT-20261006.md`）。

**⇒ 结论：上游做的是"通信 ‖ 计算"，我们想做的是"AIC ‖ AIV"，两者不冲突、也不重复。**

---

## 5. 相关 issue 全景（时间线）

| 编号 | 日期 | 状态 | 内容 |
|---|---|---|---|
| **#4894** | 2025-12-10 | **open**（40 条评论） | `[Performance]: adapt community's dbo to vllm-ascend`；作者明说 "only supported with DeepEP and DP+EP deployments, which cannot directly used in vllm-ascend" |
| **#8471 / #8507** | 2026-04-21 | closed | `[BugFix] Add compatibility guard for --enable-dbo` ← **就是那道禁用守卫** |
| **#11012** | 2026-06-26 | open+closed 同日 | `Enable initial DBO path on Ascend` |
| **#11273** | 2026-07-01 | **open（无人 review）** | `Enable --enable-dbo on Ascend NPU` ← 当前主力 |
| #5591 | — | open | `applying dual-batch overlap to improve Prefill performance` |
| #11767 | — | open | `support ubatch overlap to improve MoE model prefill throughput` |
| #15062 | — | open | `[Feat][MoE] Add DSA-CP MoE dual micro-batch overlap`（DSA-CP 方向） |
| **#17863** | 2026-10-01 | **open** | `[Performance][Attention] Reduce idle SFA cores and small-query scratch allocation` ← **和我们测到的"1 行 query 却发起 20 AIC/40 AIV"是同一个现象** |

> #17863 特别值得注意：它说的是 **A2/A3 的 NoPE SFA "一个 query group 却按全部硬件核发起并预留 scratch"**
> —— 这正是我们 profile 里"小算子远小于带宽/核规模"的**算子内部版本**。
> 它的修法（按 query rows 限核）与我们想的"AIV 预算按工作分"是同一思路。

---

## 6. 可执行结论

| # | 动作 | 依据 | 风险 |
|---|---|---|---|
| **1** | **直接 cherry-pick #11273**，把 DBO 在 Ascend 上打开 | PR 已实现 `NPUUBatchWrapper` + `ascend_split_attn_metadata`，正是我们缺的骨架；我们 vLLM 0.27.1 已带全部 ubatching 基础设施 | ⚠️ eager-only、PCP 不支持、长 prompt 数值漂移（26.1%）**—— 与我们刚修的精度问题同类，必须过精度关** |
| **2** | 借鉴它的**线程骨架**（`ubatching.py` 已是纯 Python，与 CUDA/NPU 无关） | `ubatching.py` 只依赖 `torch` + `forward_context`，`comm/compute_stream` 是构造参数 ⇒ **可直接复用** | 需要替 `compute_stream` 那路做"按核预算分配" |
| **3** | 我们的"**AIC ‖ AIV**"想法，改成在 **ubatch 骨架 + `limit_core_num`** 上实现 | §4：上游做的是 comm‖compute，没人做 AIC‖AIV；两个零件都在（骨架 + 控核 API） | 无人验证过，需自己测 |
| **4** | 顺带关注 **#17863**（SFA 空闲核） | 与我们"AIV 约 16 个核多余"的发现同源 | 与我们工作不冲突，可并行 |

### 6.1 一个重要的反向警示

#11273 里 **prefill > 512 token 的 token-identical 只有 26.1%**（数值漂移）。
我们刚花了很大代价定位并修复「静默答错」（32 位块回绕、BAT=2048 越界）。
**做 pingpong 之前必须先建立精度判据**（我们已有 `walk_blocks.py` 逐位比对 + 长文针 24/24），
否则很可能重蹈"性能上去、正确性悄悄坏掉"的覆辙。

---

## 7. 复现

```bash
# ① 上游 DBO 基础设施（我们的容器 vLLM 0.27.1 已带）
ssh a3-21 'docker exec dsv41-tp8k5 bash -lc "
  grep -n \"enable_dbo\\|ubatch_size\" /vllm-workspace/vllm/vllm/config/parallel.py
  ls /vllm-workspace/vllm/vllm/v1/worker/ | grep ubatch
  sed -n \"1,60p\" /vllm-workspace/vllm/vllm/v1/worker/ubatching.py"'

# ② Ascend 的禁用守卫
ssh a3-21 'docker exec dsv41-tp8k5 bash -lc \
  "sed -n \"898,914p\" /vllm-workspace/vllm-ascend/vllm_ascend/platform.py"'

# ③ 上游 issue / PR
curl -s https://api.github.com/repos/vllm-project/vllm-ascend/issues/11273      # 主力 PR
curl -s https://api.github.com/repos/vllm-project/vllm-ascend/issues/4894       # 40 条讨论
curl -s https://api.github.com/repos/vllm-project/vllm-ascend/issues/17863      # 空闲 SFA 核
curl -s https://api.github.com/repos/vllm-project/vllm-ascend/issues/8471       # 禁用守卫

# ④ 本地 clone 已更新到最新
cd ~/tmp/va-latest && git log -1 --oneline                 # cd96b9d 2026-10-06
cd ~/opensrc/cann-recipes-infer && git log -1 --oneline    # 2225cae
```
