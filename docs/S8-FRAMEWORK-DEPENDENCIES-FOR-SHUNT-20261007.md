# 评估：实现「多流并行分批次（Shunt）」需要动 vLLM / vllm-ascend 的哪些依赖（2026-10-07）

> 本文读的是**容器内实机源码**（vLLM **0.27.1** + 对应 vllm-ascend），不是文档转述。
> 结论：**框架侧几乎不用改，模型侧是主要工作量，但有一个真实的框架缺陷必须先打补丁。**

---

## 0. 一页纸

| 层 | 需要改什么 | 工作量 | 结论 |
|---|---|---|---|
| **vLLM 上游** | `current_stream()` 的 TLS 缓存（NPU 下失效） | **小补丁** | ⚠️ **必须打**，否则任何用它的代码看到陈旧流 |
| **vllm-ascend 框架** | 捕获根 / 流池 / 侧流 join **都已具备** | **0~3 天（确认）** | ✅ 基本不用改 |
| **vllm-ascend 风险点** | **PIECEWISE 图模式**会打断跨层流结构 | 待确认 | ⚠️ 我们交付用 `FULL_DECODE_ONLY` ⇒ 规避 |
| **模型层（我们的 patches）** | 算子→流的映射 + event 依赖 + 精度闸 | **1~2 周** | 🔴 **主要工作量在这里** |
| 流数量 | 实测 64 条可建；图内 2/3/4/6/8 流全部 OK | — | ✅ **不是瓶颈** |

---

## 1. 框架侧现有机制盘点（实机源码）

| 机制 | 位置 | 状态 |
|---|---|---|
| **单 capture root** | `acl_graph.py:189` `with torch.npu.graph(aclgraph, pool=self.graph_pool):` | ✅ 已有 |
| **侧流 join 的官方先例** | `acl_graph.py:192` `get_offloader().join_after_forward()` | ✅ **官方已在做**（注释明说 "Join offloader's copy stream after forward to avoid unjoined stream error"） |
| 流切换 helper | `utils.py:972` `npu_stream_switch(target_stream, enabled=)` | ✅ 已有（本质是 `torch.npu.stream()` 的包装） |
| 辅助流池 | `dsa_v1.py:112` `dsv4_dsa_overlap_stream()`（我们自己的 MULTISTREAM 在用） | ✅ 已跑通 |
| 流资源耗尽识别 | `acl_graph.py:30` 错误码 `207008` + `insufficient_stream_resources` | ✅ 已有专门处理 |

**⇒ 框架**已经具备**全部所需原语**：能开流、能 fork/join、能捕获、能识别资源耗尽。

---

## 2. 实测：流数量不是瓶颈

```
可创建流数（未捕获）: 64
图内并发流数  2 / 3 / 4 / 6 / 8 流 —— 全部捕获成功
耗时: 0.297 / 0.299 / 0.306 / 0.325 / 0.332 ms   （几乎不变）
```

⇒ 即便给每层开一对流，也远未触及 `207008` 的资源上限。**这一项可以划掉。**

---

## 3. ⚠️ 依赖 1：vLLM 的 `current_stream()` TLS 缓存（**必须打补丁**）

```python
# vllm/utils/torch_utils.py:662
def current_stream() -> torch.cuda.Stream:
    """
    ... here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.
    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
```

**问题**：这个 TLS 缓存**只被 `torch.cuda.set_stream` 的补丁更新**。
在 NPU 上 `torch.npu.set_stream()` **不会**更新它 ⇒ **`vllm.utils.current_stream()` 返回陈旧的流**。

**已在 DBO 线实测复现**：
```
torch.npu.set_stream(s2) 之后
  torch.npu.current_stream()      == s2  →  True
  vllm.utils.current_stream()     == s2  →  False（返回旧流）
```

**影响面**：**任何**在 NPU 上依赖 `vllm.utils.current_stream()` 的逻辑（不只是 ubatching）。
我们现有的 MULTISTREAM 补丁之所以没踩到，是因为它直接用 `torch.npu.current_stream()`。

**补丁**：已有 `tools/dbo_fix_npu_stream_tls.py`（在 `update_stream()` 里同步写 TLS）。
**成本：0.5 天。必须在 Shunt 之前打。**

---

## 4. ⚠️ 依赖 2：PIECEWISE 图模式会打断跨层流结构

`acl_graph.py` 里明确提到 PIECEWISE：

```python
# :158  # piecewise mode.
# :169  # during every model forward for piecewise aclgraph
# :199  # It is only safe to do this for the last graph in piecewise aclgraph mode
```

**风险**：piecewise 会**按层切成多张图**（注释："roughly one per layer"）。
而 Shunt 的流结构是**跨层**的（层 L 的 AIV 与层 L+1 的 AIC 重叠）
⇒ 切图会把 fork/join 拆到不同图里，**结构断裂**。

**我们的情况**：交付配置用 **`FULL_DECODE_ONLY`（单张整图）** ⇒ ✅ **天然规避**。
但若将来要支持 piecewise，**vllm-ascend 需要改**：让 fork/join 结构不跨 piece 边界，
或把 piece 边界也当作同步点。

**成本**：调查 1~3 天；若真要支持 piecewise，属于 vllm-ascend 侧的中等改动。

---

## 5. 🔴 模型层：主要工作量（不是框架问题）

**框架不提供「这个算子放哪条流」的机制。** 必须模型代码显式指定。

**现有模板**（我们自己的 `dsa_v1.py:322-373`，MULTISTREAM/DSA_OVERLAP）：

```python
main_stream = torch.npu.current_stream()
aux_stream  = dsv4_dsa_overlap_stream()
...
with npu_stream_switch(aux_stream, enabled=True):
    aux_stream.wait_event(q_quant_done)
    ...  aux_stream.record_event() ...
...
main_stream.wait_stream(aux_stream)
```

**要推广到主流全部算子，需要做的**：

| # | 工作 | 说明 |
|---:|---|---|
| 1 | **确定算子→流的映射表** | 依据是 `aicore_time`/`aiv_time`（纯 AIC → 流A，纯 AIV → 流B，MIX → 主/占大头那侧）。**这张表可以自动生成** |
| 2 | **在所有 fork 点插 event** | 遵守两条实测规则：① fork event **必须 record 在 capture root 上**；② **所有侧流必须在 `capture_end` 前 join 回根流** |
| 3 | **张量预分配** | 图捕获期**不能新建**会跨 replay 存活的中转张量（DBO 线已证：捕获期临时张量在 replay 时地址失效） |
| 4 | **共享状态复审** | V4.1 有 `shared_attention_state`、`_publish_task` 的**组级共享缓存**、RoPE buffer —— 跨流后读写时序要重审 |
| 5 | **精度闸** | 答案稳定性门 + 长文针（`walk_blocks` 逐位门已证不适用） |

**估算**：1~2 周（含精度调试）。

---

## 6. 一个必须先回答的前置问题

上面第 1 条的「算子→流的映射表」**依赖真实的数据依赖关系**。

**如果相邻的 AIC/AIV 多为真依赖**（S7 已证 `MatMulV2 → HcPost` 就是），
那这张表**填不满**——只有「输出不被紧邻消费者使用」的算子才能挪。

**⇒ 依赖审计是整个 Shunt 的前置条件**，不只是"顺便做一下"。

---

## 7. 结论与顺序

| 顺序 | 动作 | 依赖关系 |
|---:|---|---|
| 1 | **依赖审计** | 决定 Shunt 可不可做（前置） |
| 2 | 打 `current_stream` TLS 补丁 | 与 1 并行，0.5 天 |
| 3 | 确认交付用 FULL 而非 PIECEWISE | 与 1 并行 |
| 4 | 生成算子→流映射表 | 依赖 1 的产出 |
| 5 | 模型层改流 + 精度闸 | 依赖 4 |
| 6 | tiny 验证 → tp8 落地 | 依赖 5 |

**框架侧的总账**：
* vLLM：**1 个小补丁**（`current_stream` TLS）
* vllm-ascend：**基本不用改**（捕获根、侧流 join、流池、错误识别全都有）
* **主要工作在模型层**，且**依赖审计是前置条件**

---

## 8. 复现

```bash
# 源码位置核对
ssh a3-21 'docker exec dsv41-op-hcfuse sed -n "185,196p" /vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py'
ssh a3-21 'docker exec dsv41-op-hcfuse sed -n "972,981p" /vllm-workspace/vllm-ascend/vllm_ascend/utils.py'
ssh a3-21 'docker exec dsv41-op-hcfuse sed -n "662,690p" /vllm-workspace/vllm/vllm/utils/torch_utils.py'
ssh a3-21 'docker exec dsv41-op-hcfuse sed -n "320,375p" /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py'
# 流数量实验
ssh a3-21 'docker cp ~/tmp/streamlimit.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 streamlimit.py"'
```

工具：`tools/shunt_stream_limits.py`。
