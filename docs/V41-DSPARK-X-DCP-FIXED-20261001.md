# ★ DSpark × DCP 打通实录（2026-10-01）：5 个独立缺陷、逐层推进

> 结论先行：**DSpark + DCP>1 现在能跑通第一个请求了**（tiny-dspark 实测
> `A=2.00`、0 error、200 OK）。过程中挖出 **5 个互相独立的缺陷**，其中 3 个是
> 「上游代码对 DSpark×DCP 这条路径本来就没写完」，2 个是「DSpark 覆写了基类方法
> 导致基类的准备逻辑从不执行」。全部修完后逐层通过，每一层都靠 tiny 夹具（~3 分钟/轮）
> 定位，**没有一次是靠猜的**。

---

## 0. 起点与判据

起点：`docs/V41-DSPARK-X-DCP-BLOCKER-20261001.md` 记录的前两个崩点
（`assert long_seq_args is not None`、`Device tensor inputs are only supported...`）。

夹具：`~/models/out/v41-tiny-dspark`（config 变体，无权重文件，`LOAD_FORMAT=dummy`）
+ `experimental/v41-dcp/launch_tiny_dspark.sh`，TP2 + DCP2 + SPEC=1 SP_TOKENS=7
DRAFT_GRAPH=1，chip 2/3，端口 19310，**起服 ~170 s**。

判据：① health=200；② 第一个请求不是 500；③ `Mean acceptance length > 1.0`
（= 草稿真的被采纳；`A≈1.0` 说明草稿算了但全被 verify 拒掉）。

---

## 1. 修复台账（按暴露顺序）

### fix #1 —— `AscendSpecDecodeBaseProposer.set_inputs_first_pass` 的 parallel-drafting 分支没有 DCP 处理

* **文件**：`patches/files/draft/llm_base_proposer.py`（第 2384 行起的 `else:` 分支）
* **症状**：`_propose:1367 assert long_seq_args is not None`
* **根因**：`needs_extra_input_slots=True`（DSpark 是 parallel drafting）走的分支
  末尾**硬编码 `return ..., None`**；只有 `not needs_extra_input_slots` 的 EAGLE 分支
  才调 `dcp_manager.prepare_spec_decode_first_pass_inputs`。
* **修法**：在那个分支里补一次同样的调用，但**只取 `long_seq_args`，丢弃它对
  token 布局的覆盖**（parallel drafting 的布局与 EAGLE 不同，不能被改写）；
  索引用**展开前**的 `token_indices_to_sample`。

### fix #2 —— `BlockTable.compute_slot_mapping_draft` 的复制态分支拒绝 device 张量

* **文件**：`~/dcpw/vllm_ascend/worker/block_table.py`
* **症状**：`ValueError: Device tensor inputs are only supported for CP draft slot mapping.`
* **根因**：`effective_dcp_world_size == 1`（复制态组：SWA / draft）分支里，遇到
  device 张量直接 raise；而 DCP 的推测解码 device 侧重建
  (`dcp_utils.rebuild_async_spec_decode_inputs:323`) 会**主动传 device 张量**。
* **修法**：新增 `_compute_replicated_slot_mapping()`，与 numpy 路径**逐字等价**
  （`slot = block_number * block_size + pos % block_size`，无 interleave mask）。
* **验证**：容器内单测 6 组形状 + 边界（pos=0/127/128/1023/8191/16383），
  **A（numpy）与 B（device）逐值相同**，`RESULT: ALL_SAME`。

### fix #3 —— `AscendDSparkProposer` **覆写了** `set_inputs_first_pass`，fix #1 对它无效

* **文件**：`patches/files/draft/dspark_proposer.py`（第 406 行）
* **症状**：fix #1/#2 都打了，**同一个 file:line 的 assert 仍然崩**。
* **根因**：DSpark 有**自己的** `set_inputs_first_pass` 实现（不调 super），
  第 528 行同样硬编码 `return num_query_total, token_indices_to_sample, cad, None`。
* **修法**：在该方法开头（`cad.query_start_loc` 被就地改写**之前**）抓取
  `ori_token_indices_to_sample`，末段补同样的 `prepare_spec_decode_first_pass_inputs`。
* **教训**：**改基类前先 grep 子类有没有 override**。这次因此多花了一轮 ~3 分钟。

### fix #4 —— `block_table_tensor_clone` 永不被创建（DSpark 覆写 `dummy_run`）

* **文件**：`patches/files/draft/llm_base_proposer.py`（`_propose` 第 1630 行）
* **症状**：`AssertionError: block_table_tensor_clone is not init`
* **根因**：该 buffer 只在**基类 `dummy_run`** 里创建（条件 `dcp_size>1 and
  use_cuda_graph and not is_profile`），而 `AscendDSparkProposer` 也覆写了 `dummy_run`
  ⇒ 那段从不执行。
* **修法（最终版）**：把 `_propose` 里那条 clone 分支**按 `parallel_drafting` 收窄**：
  `if self.dcp_size > 1 and self.use_cuda_graph and not self.parallel_drafting:`。
  *理由*：这条 clone 分支是为**多步草稿**准备的（draft_index 1..N 的合并图里同一个
  metadata 对象被就地改写，必须给第 1 步一份私有副本）；而紧随其后的
  `should_update_next_steps = not self.parallel_drafting and ...` 对 parallel
  drafting 恒为 False ⇒ 步进循环根本不执行 ⇒ 别名风险不存在 ⇒ 走 DCP=1 那条
  **已在 CED-PD 生产验证过**的 `.clone()` 路径即可。
  *顺带绕开的第二个缺陷*：clone 的宽度取自 `input_batch.block_table[0]`（实测 **256**），
  而本 metadata 的宽度是 **512** ⇒ 即使建出来也会
  `RuntimeError: The expanded size of the tensor (256) must match the existing size (512)`。
  （先写过「幂等 helper」版本，正是撞上这个 shape mismatch 才改成收窄方案。）

### fix #5 —— 不为 parallel drafting 计算 MTP 步进元数据

* **文件**：`patches/files/draft/llm_base_proposer.py`（`_propose` 第 1651 行）
* **症状**：`dcp_utils.py:236 assert seq_lens_cpu is not None`
* **根因**：`prepare_spec_decode_mtp_drafting_inputs` 的结果**只喂给**
  `should_update_next_steps` 的步进循环；parallel drafting 下该循环**永不执行**
  ⇒ 这个调用是纯死值。而 DSpark 的 attn_metadata 既没有 `seq_lens` 也没有
  `seq_lens_cpu`（不走 Ascend 那条 builder）⇒ 必崩。
* **修法**：加 `and not self.parallel_drafting` 门。非 parallel 路径**行为完全不变**。

---

## 2. 逐层推进的实测轨迹【实测】

| # | 修完的 fix | 起服 | 第一个请求 | 下一个崩点 |
|---|---|---|---|---|
| 0 | （起点） | ✅ 170s | ❌ 500 | `assert long_seq_args`（TP1）+ device 张量（TP0） |
| 1 | #1 + #2 | ✅ 170s | ❌ 500 | 同一个 `assert long_seq_args`（**在 dspark_proposer.py**） |
| 2 | +#3 | ✅ 170s | ❌ 500 | `block_table_tensor_clone is not init` |
| 3 | +#4（helper 版） | ✅ 170s | ❌ 500 | `expanded size (256) != (512)` |
| 4 | #4 改收窄版 | ✅ 170s | ❌ 500 | `assert seq_lens_cpu is not None` |
| 5 | +#5 | ✅ 170s | **✅ 200 OK** | — |

最终：`A = 2.00`（Mean acceptance length）、Accepted 8 / Drafted 56、
**serve.log ERROR 计数 = 0**。

> `A=2.00` 而不是更高，是因为 tiny 是 **dummy 权重**（无真实权重文件），
> 草稿质量本身没有意义。**关键判据是 `A > 1.0`** —— 它证明草稿被 verify 采纳，
> 链路（草稿槽位 → DCP slot mapping → 合并 → 验证）是**活的**，不是"算了但全被拒"。

---

## 3. 每个修复的"为什么不是巧合"

| fix | 独立性证据 |
|---|---|
| #1 | 崩点行号 `llm_base_proposer.py:1367`，与上游代码里那个硬编码 `None` 一一对应；补上后 TP1 的这条错误消失 |
| #2 | 6 组形状 + 边界逐值等价（numpy vs device），不是"跑过就算" |
| #3 | 补 #1 后**同一个错误文本、不同文件**（`dspark_proposer.py` 自己的 override）—— grep 确认该类确实覆写了该方法 |
| #4 | 不是"把 assert 删了"：改的是**分支条件**，并给出了"DSpark 是单次并行草稿 ⇒ 该分支要防的别名风险不存在"的代码依据（`should_update_next_steps` 那一行）；同时解释了 helper 版为何会 shape mismatch（256 vs 512） |
| #5 | 该调用的唯一消费者是 `if should_update_next_steps:` 的循环体，而该条件含 `not self.parallel_drafting` ⇒ 对 DSpark 是**可证明的死代码** |

---

## 4. 尚未验证 / 风险

1. **【未确认】8 卡 + DCP8**：以上全部在 TP2+DCP2 的 tiny 上验证。
   DCP8 的 slot mapping 分片、merge 合并、verify 路径必须**另跑**。
   ⚠️ 特别注意本次踩到的 #4：`input_batch.block_table[0]` 宽度在 DCP8 下是否也是
   不同宽度，要实测。
2. **【未确认】精度**：tiny 是 dummy 权重，`A=2.00` 只证明链路是活的，
   **不证明输出正确**。必须用真实权重跑回归：短问答 `17×23 → 391`、
   T=904 针 → `Q7`、长针 2000/8000/16000。
3. **【未确认】上游意图**：上游在 `vllm/config/speculative.py:988` 有一句
   `if method=="dspark" and "K3DSparkModel" in architectures and dcp>1: raise`。
   我们的 draft 架构名是 `DSparkDeepseekV41ForCausalLM` ⇒ 不触发。
   **我们等于绕过了一道上游显式设下的门**。所以：
   * 若 8 卡/精度回归暴露问题，第一嫌疑就是这里；
   * 应当向上游确认 `DSparkDeepseekV41ForCausalLM` 是否也该被那条保护覆盖。
4. **【未确认】性能**：本次只求打通，没测 `ms/step`。DSpark 的 ms/step 代价
   （历史 CED 口径 +34.2%）在 DCP8 下是多少，要等 A/B。

---

## 5. 复现命令

```bash
# 打补丁（幂等；锚点不唯一会报错，不会重复插入）
python3 ~/tmp/apply_dspark_dcp_fix.py    # fix #1 + #2
python3 ~/tmp/apply_dspark_dcp_fix2.py   # fix #3
python3 ~/tmp/apply_dspark_dcp_fix3b.py  # fix #4（收窄版）
python3 ~/tmp/apply_dspark_dcp_fix4.py   # fix #5

# 起服 + 发请求
bash ~/tmp/launch_tiny_dspark.sh
python3 ~/tmp/ask2.py 19310 16 "17乘23等于多少？只回答数字。"

# 判据
curl -s --noproxy '*' http://127.0.0.1:19310/metrics | grep spec_decode_num_accepted
grep -a "Mean acceptance length" <run>/serve.log | tail -1
```

**备份**（都在 a3-21，未删）：
`~/dcpw/vllm_ascend/worker/block_table.py.bak_dsparkdcp`、
`~/cedpd-repo/patches/files/draft/{llm_base_proposer,dspark_proposer}.py.bak_dsparkdcp`
