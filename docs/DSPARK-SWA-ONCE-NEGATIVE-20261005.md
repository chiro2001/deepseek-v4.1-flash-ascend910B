# ⛔ `DSPARK_SWA_ONCE`（draft SWA 索引去重）—— **假设被自己的上线前核对推翻**（2026-10-05）

> 状态：**已实现、已撤回**。补丁从未上卡（没有一条臂跑过），也没有进入任何发布件。
> 本文留档 **① 为什么一开始看起来是个便宜的大机会，② 核对时发现了什么，③ 后来者该怎么做**。
> 这是本仓"**先证明代码会被执行，再谈收益**"纪律的又一个实例（同类：`DUMB-KNOBS`、`KERNEL-CACHE-STALE`）。

## 1. 原来的假设（错在哪）

在 `llm_base_proposer.py` 里看到这样的循环：

```python
for draft_index in range(self.num_speculative_tokens):      # 1138 行
    ... 把同一份 slot_mapping / seq_lens / query_start_loc 复制进 *_group[draft_index]
    attn_metadata_eagle = builder.build_for_graph_capture(...)   # draft_index == 0
    # 或
    attn_metadata_eagle = builder.build_for_drafting(..., draft_index, ...)  # draft_index > 0
```

再结合三个事实：
1. `build_for_drafting()` 内部会调 `build_dspark_swa_indices()`（SWA 索引，"位置/槽位链"，
   正是 `TAIL-OP-COUNT` 里 s47 小算子海的一个来源）；
2. 该函数的入参（`block_table / seq_lens / query_start_loc / num_actual_tokens`）**不含 `draft_index`**；
3. device 分支把每次结果都写进**同一个** `dspark_swa_indices_buffer`；

⇒ 当时判断："**每步重复算 K（=5）次，其中 4 次纯浪费，去掉可省 0.24–0.4 ms/步（1.0–1.6%）**"。

## 2. 核对时发现（上线前自查第 2 条把它打掉了）

追调用链后，**`num_speculative_tokens` 那个循环属于 `dummy_run()`（捕获期），不是每步执行的路径**：

| 路径 | 函数 | 每步调用 `build_for_drafting` 的次数 |
|---|---|---|
| **捕获期** | `llm_base_proposer.dummy_run()`（`:1046`，循环在 `:1138`） | K=5 次（**只在捕获时发生一次**） |
| **捕获期（DSpark 实际用的）** | `dspark_proposer._build_capture_draft_attn_metadata()`（`:606`）→ `:709` | **每个 attn_group 1 次**（`draft_index=1`） |
| **replay（每步）** | `llm_base_proposer._propose()`（`:1290`）→ `build_draft_attn_metadata()`（`:3294`）→ `:3355` | **每个 attn_group 1 次**（`draft_index=1`） |

关键背景：**DSpark 走的是"并行 drafting"** —— 5 个 draft token 作为**一个 batch** 一次前向，
所以 `build_draft_attn_metadata()` 返回的 `multi_steps_attn_metadata` 只有 **1 个元素**；
而那个"逐步展开 K 步"的 `attn_update_stack_num_spec_norm()`（`:2597`，调用点 `:1720`）
被 `should_update_next_steps = not self.parallel_drafting and (...)`（`:1714`）**挡掉**，对 DSpark 恒不执行。

⇒ **前提"每步重复 K 次"不成立**。真实的 replay 次数 = `len(self.draft_attn_groups)`，
而 `draft_index` 在 replay 恒为 `1`。

## 3. 为什么当时的补丁会**危险**（庆幸没上卡）

我写的条件是"`draft_index > 0` 且 device 分支 ⇒ 跳过重算、复用 buffer"。但在 replay 路径上
`draft_index` **恒等于 1**（恒满足 `> 0`）⇒ **每一次调用都会跳过**，
包括那个**唯一负责把新索引写进 buffer** 的调用。

后果：`dspark_swa_indices_buffer` 在 replay 期**再也不被刷新**，draft attention 会一直读到
捕获期/上一步的旧索引 —— 正是本仓花了两天定位的那一类**静默错误**
（`_DSA_SWA_RESIDENT` 注释记录的 `pos0≈0.07`、`A≈1.07` 症状）。
即：一个"看起来省 1%"的改动，会直接毁掉 draft 的正确性。

## 4. 仍然开放的**真**问题（换正确的问题）

replay 期是"**每个 attn_group 构建一次**"。若 draft 跨越多个 kv-cache group
（`build_draft_attn_metadata` 的注释明说"every group gets its own `build_for_drafting` call"），
且这些 group 的 **per-group block table 内容相同**，那就存在 **2–3× 的真冗余**。

**上机核对方法（10 分钟，纯读日志/探针，不需要 A/B）**：

```python
# 在 build_draft_attn_metadata() 的 group 循环里临时插一行（诊断用，打完就删）
logger.warning("[swa-groups] gid=%s groups=%d bt_ptr=%s bt_row0=%s seq_lens_ptr=%s",
               gid, len(self.draft_attn_groups),
               common_attn_metadata.block_table_tensor.data_ptr(),
               common_attn_metadata.block_table_tensor[0, :4].tolist(),
               common_attn_metadata.seq_lens.data_ptr())
```

判据：
* `groups == 1` ⇒ **无冗余**，这条路到此为止（本文即为终局）；
* `groups > 1` 且各行 `bt_ptr` 互不相同 ⇒ 必须逐组构建（**不可去重**）；
* `groups > 1` 且 `bt_row0` 与 `seq_lens_ptr` **逐组相同** ⇒ 才存在可去重的结构，
  且此时**必须**在"同一步内"做键（例如按 `(bt_ptr, seq_lens_ptr, num_tokens)`），
  **绝不能再用 `draft_index` 当判据**（本次翻车的直接原因）。

## 5. 方法论（并入本仓纪律）

1. **看到"循环里做重复事"时，先问"这个循环在哪个函数里"** ——
   `dummy_run` / capture 路径的循环**与每步无关**；本仓的 profile 计数才是"每步几次"的判据。
2. **对任何 `draft_index`-类去重，必须找到 replay 路径上该参数的取值域** ——
   本次 replay 恒为 1，"`> 0` 就复用"必然全跳过。
3. **去重的判据必须是"输入是否相同"，不能是"循环变量是否相同"**。
4. 负结果同样留档：本文节省了后来者至少一次起服 + 一次静默错误排查。

## 6. 顺带定位：`TAIL-OP-COUNT` 的 **F2「位置/槽位链」真身**（2026-10-05 补充）

追 F2 时把源头定位到了：

| 项 | 结论 |
|---|---|
| 代码位置 | **`vllm_ascend/spec_decode/utils.py:130 SlidingWindowAdapter`**（`compute_sliding_window_block_table()` + `apply()`）—— **在上游文件里，不在我们的补丁集内** |
| 每步调用次数 | **= `len(self.draft_attn_groups)`**。`_propose:1613` 那次被 `self.method not in ("dspark","mtp")` **显式排除**（注释写明"DSpark 的窗口在 build_draft_attn_metadata 里应用"）⇒ dspark 只在 `:3353` 的 group 循环里调 |
| 链上算子 | `Sub/Add/Clamp/FloorDiv/Mul/Sub/FloorDiv/Range/BroadcastTo/Add/Clamp/Gather(+IndexCheck)/Add/FloorDiv/Clamp/Lt/Lt/And/Mul/Cast/ViewCopy`，两处 `needed/valid_mask` 分支各自成链 |
| **唯一的"廉价"去重** | `start_block_indices = ((x // b) * b) // b` 里的**第二个除法是冗余的**（`(m*b)//b == m`，且前面已 `clamp(min=0)`）⇒ 每次调用省 1 个 `FloorDiv`。按 1–3 次/步估 ≈ **2–6 µs/步（0.01–0.02%）** |
| 判决 | **不采纳**：收益低于噪声底，且要为一个上游文件新增挂载项（增加交付面）。真正的解法是把整条链**下沉成一个融合 kernel**（F2 本来的建议），那是独立项目 |
