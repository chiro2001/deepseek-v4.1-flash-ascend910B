# ★★★ DSpark × DCP 是**上游明确不支持**的组合（2026-10-01 实测定位）

> 用户要求「DSpark 要开，把开销逐个拆解」。**第一条结论：它在我们的 DCP8 拓扑上
> 连第一个请求都跑不完**。本文给出精确根因、证据链、以及三条出路。

---

## 1. 实测：开 DSpark 后第一个请求就崩

```
run_id      = dcpcap_1001_100010
配置        = SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1（其余与 32.58 基线逐项相同）
起服        = 成功，health=200，speculative_config=SpeculativeConfig(method='dspark')
容量        = 5,990,222 tokens（SPEC=0 时 6,082,458 ⇒ draft SWA 吃 ~1.5%）
第一个请求  = HTTP 500
之后        = Connection refused（8 个 worker 全挂）
```

worker 报错（8 个 rank 一致）：

```
File ".../vllm_ascend/worker/model_runner_v1.py", line 1952, in propose_draft_token_ids
    draft_token_ids = self.drafter._propose(...)
File ".../vllm_ascend/spec_decode/llm_base_proposer.py", line 1367, in _propose
    assert long_seq_args is not None
AssertionError
```

---

## 2. 根因（三段代码，逐段可验）

### 2.1 上游**已经**声明不支持 —— 但保护写错了架构名

`vllm/config/speculative.py:984-993`：

```python
if self.method in ("dflash", "dspark"):
    self.parallel_drafting = True                     # ← DSpark 强制并行草稿

if (
    self.method == "dspark"
    and "K3DSparkModel" in self.draft_model_config.architectures   # ← 只挡这一个名字
    and self.target_parallel_config.decode_context_parallel_size > 1
):
    raise ValueError(
        "MLA DSpark does not currently support decode context parallelism; "
        "set decode_context_parallel_size=1."
    )
```

**我们的 draft 架构名是 `DSparkDeepseekV41ForCausalLM`**（在镜像里
`vllm_ascend/models/deepseek_v41/dspark.py` 定义，实测枚举确认），
`"K3DSparkModel" in architectures` = **False**
⇒ **上游那条显式拒绝不触发**，于是我们绕过保护、撞上后面的隐式断言。

### 2.2 `needs_extra_input_slots` 必然为 True

`vllm/v1/spec_decode/llm_base_proposer.py:110-116`：

```python
self.extra_slots_per_request = 1 if not self.parallel_drafting else self.num_speculative_tokens
self.net_num_new_slots_per_request = self.extra_slots_per_request - (
    1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0)
self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0
```

代入 DSpark：`parallel_drafting=True` ⇒ `extra_slots = 7`；
`pass_hidden_states_to_model=True`（dspark）且 `method="dspark" != "dflash"`
⇒ `net = 7 - 1 = 6 > 0` ⇒ **`needs_extra_input_slots = True`**。

### 2.3 ★ 那个分支**完全没有 DCP 处理**，硬编码返回 `None`

`vllm_ascend/spec_decode/llm_base_proposer.py::set_inputs_first_pass`：

| 分支 | 条件 | DCP 处理 | 返回值 |
|---|---|---|---|
| **if** | `not needs_extra_input_slots` | ✅ 调 `dcp_manager.prepare_spec_decode_first_pass_inputs(...)`，从它拿 `long_seq_args` | `return ..., long_seq_args` |
| **else** | `needs_extra_input_slots`（**我们走这条**） | ❌ **完全没有** | `return total_num_output_tokens, token_indices_to_sample, new_cad, None` ← **硬编码 None** |

而调用方 `_propose`（同文件 1365-1368）：

```python
dcp_manager = getattr(self.runner, "dcp_manager", None)
if dcp_manager is not None:
    assert long_seq_args is not None          # ← 崩在这里
    _, ori_token_indices_to_sample = long_seq_args
```

**⇒ 只要 `use_dcp == True` 且走 parallel-drafting 分支，必崩。**

### 2.4 `long_seq_args` 为什么不能简单删掉

它被下游用来算 DCP 的 MTP 草稿槽位（同文件 1654）：

```python
dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs(
    ..., ori_token_indices_to_sample=ori_token_indices_to_sample, ...)
```
`dcp_utils.py::prepare_spec_decode_first_pass_inputs` 返回：
`long_seq_args = (decode_query_lens, original_sample_indices)`，
而 `_get_spec_decode_mtp_slot_inputs` 用它算
`num_reject_tokens = cu_num_tokens - original_sample_indices - 1` ⇒ 接受/拒绝数
⇒ `slot_indices` ⇒ DCP 的 MTP slot mapping。

**注意 `_propose` 只用了元组的第 2 项**（`_` 丢掉了 `decode_query_lens`）。

---

## 3. 影响面：DSpark 与**任何** DCP>1 互斥

崩点在 `_propose`（只在**有草稿**时执行）且条件是 `dcp_manager is not None`
（= `use_dcp`）。因此：

| 组合 | 能否跑 |
|---|---|
| `SPEC=0` + DCP8 | ✅（我们一直在跑，32.58 ms/step） |
| `SPEC=1` + DCP1 | ✅（历史 CED-PD，24.35 → 32.68 ms/step） |
| **`SPEC=1` + DCP>1** | ❌ **第一个请求必崩** |

这与历史一致：CED-PD 那条线的 DSpark 全部在 **DCP=1** 上测的
（`docs/CED-PD-DSPARK-ACCEPTANCE-20260926.md`），两个特性从未合并过。

---

## 4. 三条出路

### 出路 A：实现 DSpark 的 DCP 支持（真工程）
在 `else` 分支补上 DCP 准备：调 `dcp_manager.prepare_spec_decode_first_pass_inputs(...)`
拿到 `long_seq_args`（并处理它返回的 `num_tokens`/`positions`/`hidden_states`/`input_ids`
覆盖），或至少提供等价量。

**最小改法（待验证）**：`_propose` 只需要 `(_, ori_token_indices_to_sample)`，
而 else 分支的 `token_indices_to_sample` 就在手边 ⇒ 可给
`long_seq_args = (dcp_manager.query_lens_full.cpu[:num_decode_reqs], token_indices_to_sample.clone())`。

⚠️ **风险高**：parallel-drafting 的 token 布局与 `if` 分支完全不同
（走 `CopyAndExpandEagleInputs`，每请求多 `N` 个槽位），
**`ori_token_indices_to_sample` 的语义是否可比未经证实**。
本仓已有三次"看起来对、实际静默算错"的教训 ⇒ 必须有强判据（T=904 / 短问答 / 长针）。

### 出路 B：DSpark 与 DCP 二选一
- 选 DSpark ⇒ 回 DCP1（容量 1,242,687 vs 6,082,458，**丢 4.9× 容量**），
  换 ms/token 2.3×；
- 选 DCP8 ⇒ 保持 `SPEC=0`（当前生产口径）。

### 出路 C：向上游报 issue
上游已经有这条 ValueError，说明他们知道。可以确认
`DSparkDeepseekV41ForCausalLM` 是否也应该被那条保护覆盖（大概率是漏了），
以及 DCP 支持是否在路线图上。

---

## 5. 建议

**先做出路 C 的确认（便宜），同时在 1+1 或 DCP1 上把 DSpark 的开销拆解做完**
（拆解本身不依赖 DCP）—— 这样"开销逐个拆解"这个主任务不被 blocker 卡住。

出路 A 若要做，**必须先在小规模上验证语义**（比如 DCP2 + tiny），
不能直接上 8 卡 —— 因为失败模式是"崩"（好抓）或"静默算错"（难抓）。

---

# 6. ★★★ tiny-dspark 夹具建成（同一个 blocker，但迭代只要 3 分钟）

## 6.1 结论：**不需要"新权重"，只需要一个新 config**

tiny 是 `LOAD_FORMAT=dummy`（`launch_dcp2_tiny.sh:143`），**目录里根本没有权重文件**
（只有 `config.json` / `configuration.json` / `tokenizer*`）。
所以"DSpark 版 tiny"= **一个 config 变体**，代价接近于 0。

已创建 `~/models/out/v41-tiny-dspark`（独立目录，**不动 `v41-tiny`**，避免打乱既有基线）：

| `text_config` 字段 | v41-tiny | v41-tiny-dspark | 为什么 |
|---|---|---|---|
| `num_nextn_predict_layers` | 0 | **3** | tiny 转换时被清空（见 `l1_dummy_provenance.json`） |
| `dspark_target_layer_ids` | [] | **[37,38,39]** | 同上；draft 靠它取 aux hidden states |
| `dspark_n_routed_experts` | 128 | **8** | `patch_speculative_config.py:76` 用它覆盖 draft 的 `n_routed_experts` |
| `dspark_num_experts_per_tok` | 3 | **2** | 同上（对齐 tiny 规模） |

其余 dspark 结构参数（`block_size=5`、`markov_rank=256`、`noise_token_id`）**原本就在**。
派生记录写在 `v41-tiny-dspark/dspark_derivation.json`。

## 6.2 实测：起服成功【实测】

```
容器 dsv41-tinyspark，chip 2/3，端口 19310，TP=2 + DCP=2
speculative_config=SpeculativeConfig(method='dspark', num_spec_tokens=7)
load_format=dummy，health=200，0 error，启动 ~2 分钟
```

⇒ **夹具可用**：8 卡要 7–10 分钟/轮，tiny 只要 ~2–3 分钟/轮。

## 6.3 它复现了**同一个** blocker，并暴露**第二个**错误【实测】

第一个请求 = HTTP 500，两个**不同**的失败：

| rank | 错误 | 位置 |
|---|---|---|
| TP1 | `AssertionError: long_seq_args is not None` | `llm_base_proposer.py:1367`（同 8 卡那次的根因） |
| TP0 | `ValueError: Device tensor inputs are only supported for CP draft slot mapping` | `block_table.py:419` |

第二个错误的**完整调用链**：

```
model_runner_v1.py:1415  _prepare_input
dcp_utils.py:323         rebuild_async_spec_decode_...（device 侧重建路径）
block_table.py:865       MultiGroupBlockTable.compute_slot_mapping_draft
block_table.py:419       raise ValueError(...)
```

## 6.4 ★ 这证明 DSpark × DCP **是部分实现过的**

`dcp_utils.py:323` 那一段（`can_rebuild_on_device` 分支）：

```python
input_batch.block_table.compute_slot_mapping_draft(req_indices_mtp, positions_mtp)  # ← device 张量
slot_mapping = input_batch.block_table.block_tables[0].slot_mapping.gpu[:num_tokens_mtp]
self.mtp_slot_mapping = slot_mapping.clone()
```

**有人为「DCP + 推测解码」专门写过 device 侧重建路径** —— 它主动传 device 张量。

而 `BlockTable.compute_slot_mapping_draft`（→ 内部 `compute_slot_mapping`）的分支是：

```python
if self.effective_dcp_world_size > 1:
    self._compute_dcp_slot_mapping(req_indices, positions)      # ← device 张量 OK
else:
    if isinstance(req_indices, torch.Tensor) and req_indices.device.type != "cpu":
        raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
    ...（numpy 路径）
```

**⇒ 这两个错误的性质完全不同**：

| # | 错误 | 性质 |
|---|---|---|
| 1 | `assert long_seq_args is not None` | **分支遗漏**：parallel-drafting 分支没有 DCP 处理（上游只按 `K3DSparkModel` 名字挡，漏了我们的 `DSparkDeepseekV41ForCausalLM`） |
| 2 | `Device tensor inputs are only supported for CP draft slot mapping` | **守卫过窄**：device 侧重建路径 + **复制态组**（`effective_dcp=1`）这个组合没人实现 |

⇒ 修 #2 只需给"复制态组的 device 侧 draft slot mapping"补一条实现
（语义 = `block = table[req, pos // bs]; slot = block*bs + pos % bs`，与 numpy 路径逐字等价）；
修 #1 需要补 DCP 准备，或确认 parallel-drafting 的布局与 DCP 可兼容。

**两者都可以在 tiny 上以 3 分钟/轮 迭代验证。**
