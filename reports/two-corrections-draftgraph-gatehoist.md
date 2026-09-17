# 两处更正：`GATE_HOIST` 崩溃 + **DSpark draft 从来就不在图里**

> 2026-09-16 16:50 CST｜A3-node1｜由「把这些已验证补丁改成默认开启」触发

---

## 更正 1：`GATE_HOIST=1` 会让**真实权重起服在编译期崩溃**（已回退）

### 现象

我把 6 个已验证补丁改成默认开启后，真权重起服失败：

```
torch._dynamo.exc.UserError: Consider annotating your code using torch._check*().
  Could not extract specialized integer from data-dependent expression u0 (unhinted: u0)
  File ".../models/deepseek_v41/model.py", line 919, in forward
  File ".../models/deepseek_v41/model.py", line 31, in _engram_gate_channel_weight
```

### 根因

`probe_bneck/model.py.probe:21-23`（`engram-gate-hoist` 补丁加的缓存 key）：

```python
def _engram_gate_version_key(tensor):
    return (id(tensor), int(getattr(tensor, "_version", -1)),
            tuple(tensor.shape), tensor.dtype)     # ← tuple(tensor.shape)
```

它**被从编译区内的 `forward` 调用**（`model.py:919`），
而 `tuple(tensor.shape)` 在 torch.compile 追踪下会产生 **unbacked SymInt** ⇒ dynamo 无法特化 ⇒ 正是那个 `u0`。

**这与本项目早先的一次失败同类**：我当时用 `open()` 读运行时文件标志做开关，
同样因为「追踪期可见的运行时 Python 值」而炸（`Failed to trace builtin operator`）。

### 为什么之前的 dummy 验证没暴露

`tvB`/`vjA`/`vjB`/`vjC` 四臂**都开了 `GATE_HOIST=1` 且都起服成功** —— 但它们全是
`LOAD_FORMAT=dummy`。**真权重与 dummy 的编译路径在这个位置不同**（dummy 下该 gate 分支未进入编译区）。
⇒ **「dummy 起服成功」不能证明一个补丁在真权重下也能起服。**

### 处置

* **回退 `GATE_HOIST` 默认为 0**（`serve_a21.sh:51`），并注明原因与回退依据
* 该补丁的实测收益本只有 **24 次 dynamic cast ≈ 0.0001 ms/step**，**不值得修**
* 保留代码与开关（`GATE_HOIST=1` 仍可用于 dummy 实验），但**不进交付配置**

---

## 更正 2 ★★：**DSpark draft 从来就不在 ACLGraph 里 —— `SPEC_EAGER_OPT` 是 no-op**

### 起因

线 3 报告「eager 提交 748 次/pass，其中 draft 的 allreduce 慢 25×」，
并建议「把 draft 收进图」。我先前也据此在 `draft-graph-decision.md` 里
**把 `SPEC_EAGER_OPT=0`（draft 入图）记为"已采纳"**。

### 实测：draft 从未入图

| 证据 | 结果 |
|---|---|
| 起服日志里 `Wrapping draft model with ACLGraphWrapper` 这条 INFO | **一次都没有**（grep 全日志 = 0） |
| `api_statistic` 的 `aclmdlRIExecuteAsync` | **118 次**，而前向约 118 步 ⇒ **每步只有 1 次图重放（只有 target）** |
| draft 特征算子的 `OP State` | `FloorMod`/`FloorDiv`/`SelectV2`/`ArgMaxV2`/`MaskedFill` **全部 `dynamic`**（eager） |
| 图捕获日志 | 只有 **1 组** `Capturing CUDA graphs (decode, FULL): 0/1 → 1/1` |

### 根因（vendor 代码里写得很明白）

`vllm_ascend/spec_decode/dspark_proposer.py`：

```python
line 37:  super().__init__(vllm_config, device, runner=runner)   # 基类 line 218：
          #   self.use_cuda_graph = runner._use_aclgraph() and not enforce_eager  → True
line 75:  # DSpark runs eager only (Ascend cudagraph unsupported on this path).
          self.use_cuda_graph = False                            # ← 无条件覆盖为 False
```

而 `llm_base_proposer.py:604`：
```python
if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
    ... self._runnable = ACLGraphWrapper(...)      # ← 因为 use_cuda_graph=False，永不执行
```

**⇒ 无论 `--speculative-config` 里 `enforce_eager` 传 `true` 还是 `false`，
line 75 都会把它设成 `False` ⇒ DSpark draft 恒为 eager。**

### 影响（三处需要改口的结论）

| 原结论 | 更正 |
|---|---|
| `reports/draft-graph-decision.md`：「采纳 draft 入图，A2 弱 CPU 上收益更大」 | ❌ **void** —— draft 从未入图，开关无效 |
| `HANDOVER_A21_PERF.md` 把它列为"已采纳" | 需改为"**实测为 no-op**" |
| 线 3 的「把 draft 收进图可省 ≈2.5 ms」 | ❌ **不可通过配置实现** —— vendor 明确写 `Ascend cudagraph unsupported on this path`，需自行实现 DSpark 的 cudagraph 支持 |

### 那 `SPEC_EAGER_OPT` 到底有没有用？

`enforce_eager` 在整棵树里只有 4 个读取点：

| 位置 | 作用 | 对我们是否有效 |
|---|---|---|
| `vllm_ascend/.../llm_base_proposer.py:218` | 设 `use_cuda_graph` | ❌ 被 dspark line 75 覆盖 |
| `vllm/v1/spec_decode/llm_base_proposer.py:418` | `initialize_cudagraph_keys`（drafter） | ❌ `use_cuda_graph=False` 后该路径 moot |
| `vllm/v1/worker/gpu_model_runner.py:6195` | `_dummy_run` 里的 `use_cudagraphs` | ❌ 只在 warmup/profile 路径 |
| `vllm/v1/spec_decode/extract_hidden_states.py:254` | 另一条 drafter 路径 | ❌ 我们不使用 |

**⇒ 对 DSpark 配置，`SPEC_EAGER_OPT` 实际上没有可观测效果。**
这也**解释了** a22 子代理的 A/B/A/B2 为什么测出「−0.41 ms，但落在 +0.91 漂移内、不可分辨」——
**它测的是一个 no-op。**

### 真正的方向（若要减少 draft 的 eager host 开销）

必须**自己实现 DSpark 路径的 cudagraph 支持**（vendor 注释说明这是未支持项）。
这不是配置/算子级改动，而是一个独立的工程项目。
**在此之前，draft 的 ~718 次/步 eager 提交是架构性固定成本。**

---

## 附：这两次问题的共同教训

**「dummy 下起服成功」和「配置项被写进启动器」都不等于「生效」。**
判定一个开关是否真的生效，必须看**设备侧行为指纹**：

| 开关 | 应看的指纹 |
|---|---|
| `GATE_HOIST` | eager `[4,5120]` cast 计数（24→0） |
| `ENGRAM_JIT` | `[bneck]` 的 `hash` 相位（0.427→0.076） |
| `QLI_NOCAND` | QLI 每 op µs（99.3→50.3） |
| `ROPE_IDXSEL` | `Index`/`IndexCheck` 计数（−4104） |
| **`SPEC_EAGER_OPT`** | **`Wrapping draft model` 日志 + `aclmdlRIExecuteAsync` 次数** ← 本轮才发现它恒为 1/步 |

---

## 证据

| 内容 | 路径 |
|---|---|
| 崩溃栈 | `logs/perf/faA_serve.log`（`_engram_gate_channel_weight`） |
| vendor 注释 | `dspark_proposer.py:75`；`llm_base_proposer.py:604/218` |
| 图重放次数 | `logs/prof_vrB/.../api_statistic_*.csv` 的 `aclmdlRIExecuteAsync = 118` |
| draft 算子的 state | `/tmp/vrB_rank0.csv`（`FloorMod`/`SelectV2` 等全 `dynamic`） |
| 回退后启动器 | `serve_a21.sh:51`（`GATE_HOIST` 默认 0，带 `[REJECTED]` 注释） |
