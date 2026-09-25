# CED 图模式的启动前提：一个真开关与一个**死**开关（2026-09-25）

## 摘要

`docs/CED-PD-ACCEPTANCE.md` §2.4 列的"起服硬门"有三条，其中**第一条是过期的**：

| 硬门 | 状态 | 证据 |
|---|---|---|
| `[CED-META] inline metadata group=…` | **❌ 过期，已移除** | 当前分支上**没有任何代码打印这一行**；通过 21/21 验收的那台 D（`ced_prof_d_0925_165058`）日志里出现 **0** 次 |
| `[CED-GRAPH] one-token prompt tail forced eager` | ✅ 真判据 | 该 D 的日志里出现 **240** 次；由 `experimental/ced/core_model_runner_prompt_tail.patch` 打印，条件是 `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` |
| `Replaying aclgraph` | ✅ 真判据 | 生成步走图的证据 |

## 1. `V41_CED_METADATA_INLINE` 是死开关

* `grep -rn "V41_CED_METADATA_INLINE"` 在**代码里**命中 0 —— 只出现在
  launcher 脚本与文档里；`serve_a2.sh` 也**没有**把它透传进容器。
* 该功能的实际实现在 `experimental/ced/dsa_v41.py::enable_device_metadata()`：
  它**无条件**把 `self._device_metadata_enabled = True`
  （即 device-metadata 走主流已经是固定行为，不再是诊断臂）。
* 历史：这段代码曾经在 `fix/ced-metadata-inline` 分支上由 env 门控
  （提交 `d7953e5`，见 `experimental/ced/METADATA_INLINE.md`），
  但那个分支**不是** `feat/ced-pd-a3` 的祖先；当前分支上的版本没有这个门控。

⇒ 结论：**不要**再为它设环境变量，也不要把它当起服判据。

## 2. 真判据：`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`

`core_model_runner_prompt_tail.patch` 里唯一被读的 CED 图开关：

```python
ced_prompt_tail_eager = (
    os.environ.get("V41_CED_ROLE") == "decode"
    and os.environ.get("V41_CED_GRAPH_PROMPT_TAIL_EAGER") == "1"
    and max_num_scheduled_tokens == 1
    and any(int(...num_computed_tokens...) < int(...num_prompt_tokens...))
)
if ced_prompt_tail_eager:
    logger.info("[CED-GRAPH] one-token prompt tail forced eager")
```

**漏掉它的后果（本次实测）**：D 仍能起、health 200，但**输出静默乱码** ——
HTTP 200、`completion_tokens` 打满 `max_tokens`、无 `finish_reason`、含 `<|box|>`。
2026-09-25 23:23 那次 144K 多轮 **3/3 全错**（首轮 33 s，正常应约 11 s），
看起来像 CED 的缺陷，实际只是启动参数不全。
补上后同一批请求的短针立即 PASS，且 `[CED-GRAPH]` 标记出现 8 次。

## 3. 已经加的防线

`scripts/serve_a3_ced_pd.sh` 在 `CED_EXPERIMENTAL_GRAPH=1` 分支改成 **fail-closed**：

* 缺 `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` → **拒绝起服**（exit 2）；
* 确实要裸跑图模式做诊断 → 设 `V41_CED_ALLOW_BARE_GRAPH=1`，
  此时只 WARN，并明确"结果不可当正确性证据"；
* 不再要求 `V41_CED_METADATA_INLINE`。

自检（三种情况都验过）：`missing → FAIL-closed`、`ok → pass`、`bypass → WARN-pass`。

## 4. 教训

**"文档里的起服硬门"本身也会过期。** 这次差点把一个启动参数问题记成 CED 的功能缺陷，
靠的是把"通过验收的那台实例"的日志翻出来当参照 —— 也就是
**用已知good实例的日志去校准判据**，而不是照抄文档。
建议后续任何"硬门"都写成"在某次通过验收的实例上观察到 N 次"这种可回溯的形式。
