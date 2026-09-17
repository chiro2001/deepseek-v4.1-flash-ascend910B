# DSpark draft 入图 `num_input_tokens` 缺陷修复（[DRAFT-GRAPH-NUMINPUT-FIX]）

> 2026-09-16 ｜ A3-node1 `probe_draft/` ｜ **纯离线**（未占卡、未启停任何容器、未改 `serve_a21.sh`）
> 修复对象：`probe_draft/dspark_proposer.py`（整文件 bind-mount）

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **draft 的 query 块是 5 还是 6** | **5**。`sample_from_anchor=True`（模型 `text_config` 无该键 ⇒ 默认 True）+ `num_speculative_tokens=5`（= 模型 `text_config.dspark_block_size=5`）⇒ `num_query_per_req = 5`。**6 是 target/runner 的 bucket 粒度**（`uniform_decode_query_len = 1 + num_speculative_tokens`），不是 draft 的块大小 |
| **图的 bucket 怎么定** | 保持「target/draft 共用一套 FULL-decode bucket」不变。这套 stack 里 **draft 拿不到 5 的倍数 bucket**（`adjust_cudagraph_sizes_for_spec_decode` 把每个尺寸向上取整到 6 的倍数；且 FULL key 由 runner 的 dispatcher 生成、`ACLGraphWrapper` 按 forward context 的 `BatchDescriptor` 建 entry）。正确做法 = **图仍按真实 query 数（5/10/15/20）捕获，bucket 多出来的行只在 replay 侧 metadata 里出现且必须惰性**（slot −1 / pos 0 / pdid） |
| **修复** | 单点一行 + 注释：`max_query_tokens = max(max_batch_size*num_query_per_req, max_batch_size*(1+num_speculative_tokens))`（= DFlash 父类自己的容量口径） |
| **为什么不会让 FIA 多算一行** | 捕获用的计数 = `min(batch*num_query_per_req, max_query_tokens)` = 真实块大小（修复不改它，因为 `5·batch ≤ 6·max_seqs` 恒成立）⇒ **图里只有真实行**；多出来的行只被 replay 侧 eager metadata 写进 buffer 的 `[real, bucket)`，而 DSA 层只消费 `[:num_actual_tokens] = [:real]` |
| **离线判据** | `preflight_numinput_fix.py`：**修复前 FAIL（原样复现 `Target sizes: [6, 2]. Tensor sizes: [5, 2]`，exit 1）**；**修复后 PASS 50/50（exit 0）** |
| **附带发现（必须一起做）** | ① 启动器 `DRAFT_GRAPH=1` 分支**没有设 `DSPARK_GRAPH_CAPTURE_METADATA=1`** ⇒ dgvB 那一轮 draft 图是在「无 attention metadata」的兜底分支上捕获的（`dsa_v1.py:1689-1698` 的 `attn_metadata is None` → `output.fill_(0)`），即便不崩，draft token 也全错（A 崩到 1.0 的形态）。上卡验证脚本已自动注入该 env，但**建议直接写进 `serve_a21.sh`**。② `_pad_query_start_loc_for_fia` 在 draft 上走 uniform 分支时会静默 no-op（`cu_seqlens_q[-1]=5 < num_input_tokens=6`）——这一点**只有**在"图按真实行捕获"时才是无害的，preflight 的 `CAPTURE` 项把这个不变量钉住了 |
| **上卡** | 一键脚本 `probe_draft/verify_numinput_fix_a21.sh` 已就绪（默认 `LOAD_FORMAT=dummy`、chips 8-15、8020、`dsv41-a21-perf`）；等主 Agent 给窗口后跑 |

---

## 1. Q1：draft 的 query 块到底应该是 5 还是 6

### 1.1 结论

**是 5。** 6 来自 target 侧（每步 1 个新 token + 5 个待验证 draft token）与 runner 的 bucket 粒度
（`uniform_decode_query_len = 1 + num_speculative_tokens`），draft 只是"搭了 target 的桶"。

### 1.2 逐行证据

行号 = 本次交付的 `probe_draft/dspark_proposer.py`（661 行）。

| # | 位置 | 内容 | 说明 |
|---|---|---|---|
| 1 | `dspark_proposer.py:96-99` | `sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)`；`num_query_per_req = num_speculative_tokens if sample_from_anchor else 1 + num_speculative_tokens` | 本模型 `text_config` **没有** `sample_from_anchor` 键（已读 `models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/config.json`）⇒ True ⇒ **5** |
| 2 | `models/.../config.json` → `text_config.dspark_block_size = 5` | draft 的原生 block size | 与 `serve_a21.sh` 的 `SP_TOKENS=5`（注释：原生 block size、A 最优）一致 |
| 3 | `dspark_proposer.py:330` | `num_query_total = batch_size * self.num_query_per_req`（`set_inputs_first_pass`） | **每步真实 query 行数** = 5·batch |
| 4 | `vendored: ops/triton/spec_decode/utils.py:125-131`（`copy_and_expand_dflash_and_dspark_inputs_kernel`） | 循环上界 `num_query_total = batch_size * num_query_per_req`；写 `out_query_positions_ptr/out_query_slot_mapping_ptr/out_input_ids_ptr` | 位置/槽位/输入只写这 5·batch 行 |
| 5 | 同上 `:169-175` | `SAMPLE_FROM_ANCHOR` 分支：`sample_out_idx = req_idx*num_speculative_tokens + q_idx`，`q_idx ∈ [0, num_query_per_req)` | 5 行 → 5 个 sample（索引 0..4）。**若块是 6，这里会写到索引 5，越界**（`token_indices_to_sample` 只有 5·batch 个元素） |
| 6 | `dspark_proposer.py:411` | `cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size` | 每个请求的 query 长度 = 5 |
| 7 | `dspark_proposer.py:415-419` | `cad.num_actual_tokens = num_query_total`；`cad.slot_mapping = ...[ :num_query_total]` | 真实 token 数 = 5·batch |
| 8 | `dspark_proposer.py:417-419` | `cad.seq_lens = effective_seq_lens + num_query_per_req` | 本步 draft KV 只新增 **5** 个位置（`[eff_seq_len, +4]`） |
| 9 | `llm_base_proposer.py:1514` | `logits = raw_logits.view(-1, self.num_speculative_tokens, V)`；`draft_token_ids[:, 0].copy_(seed)`，随后 `[:, idx+1] = argmax(logits[:, idx])` | block = 5 行 × 5 个 sample |
| 10 | `llm_base_proposer.py:1567` / `:1573` | `if ... self.parallel_drafting:` → `return draft_token_ids[:, 1:]` | 返回 **5** 个 draft token（DSpark `parallel_drafting=True`：`vllm/config/speculative.py:985`） |
| 11 | `llm_base_proposer.py:1421-1423` | `num_indices = batch_size * num_speculative_tokens`；`max_num_reqs_across_dp = (num_input_tokens // num_query_per_req) * num_speculative_tokens` | LMHead 侧的真实行数也是 5·batch |
| 12 | `llm_base_proposer.py:2717-2731`（`_pad_draft_buffers`） | `input_ids[num_actual_tokens:num_input_tokens] = parallel_drafting_token_id`；`positions[...] = 0`；`_slot_mapping_buffer[...] = -1`；每 group query slot mapping `= -1` | **代码自己把 `[5, 6)` 当作 padding 行**（slot −1 = 不写 KV）⇒ 5 是真实块、6 是桶 |
| 13 | `vendored: vllm_ascend/spec_decode/dflash_proposer.py:53` | DFlash 父类：`max_query_tokens = max_batch_size * (1 + num_speculative_tokens)` | 这是**同族容量口径**（DFlash 的块本来就 6/req，所以它的"容量"恰好 = 桶粒度，历史上不会踩这个坑） |

### 1.3 三个"行数"必须分开看（本缺陷的语义核心）

| 计数 | 值（MAX_SEQS=1, S=5） | 由谁决定 | 作用 |
|---|---|---|---|
| **真实 query 块** | 5·batch | 模型（`dspark_block_size`/`sample_from_anchor`） | 真正要算的行；写 KV 的行 |
| **图捕获计数** | `min(batch*num_query_per_req, max_query_tokens)` = 5·batch | `dspark_proposer.py:534`（dummy_run） | **决定图里的形状**（FIA 实际算几行） |
| **replay 计数 `num_input_tokens`** | bucket = 6·⌈…⌉（6/12/18/24） | `llm_base_proposer.py:1042/1056`（`cudagraph_dispatcher.dispatch(...)`） | 只用于**索引 eager 侧 buffer**；图重放不重跑 Python |

**eager 下为什么没事**：`use_cuda_graph=False` ⇒ `num_input_tokens = num_tokens = 5·batch` ⇒ 三者相等，
`_pad_draft_buffers` 直接 return（`llm_base_proposer.py:2723-2724`）。入图后第三个数被拉大到桶粒度，
而第 1、2 个数没变 ⇒ 只有"按 `num_input_tokens` 索引 buffer"的那条路会越界。

---

## 2. Q2：图的 bucket 应该怎么定

### 2.1 结构事实（三条，均可离线核对）

1. **FULL-decode bucket 只由 runner 生成一次**：`model_runner_v1.py:5507-5562` →
   `CompilationConfig.resolve_cudagraph_mode_and_sizes` → `adjust_cudagraph_sizes_for_spec_decode`
   （`vllm/config/compilation.py:1518-1563`）把 `cudagraph_capture_sizes` 里每个尺寸**向上取整到 6 的倍数**
   （`multiple_of = uniform_decode_query_len`，`enable_sp=False` 时；日志实测 `enable_sp=False`），
   再 `initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)` 只保留
   `6 ≤ x ≤ 6*max_num_seqs`（`cudagraph_dispatcher.py:212-232`）。
   本配置 launcher 给的是 `[1,2,3,4,6,8,12,16,20,24,32]` ⇒ 取整后 `[6,12,18,24]`，`max=32`；MAX_SEQS=1 时只剩 `[6]`
   —— 与 dgvB 日志 `Capturing CUDA graphs (decode, FULL): 1/1` 完全对上 ✅
2. **draft 的图 entry 就用 runner 的 `BatchDescriptor` 做 key**：`acl_graph.py:133-146`
   （`concrete_aclgraph_entries[batch_descriptor]`，`batch_descriptor = forward_context.batch_descriptor`），
   而捕获由 runner 的捕获循环驱动（`model_runner_v1.py:3921-3928` 调 `drafter.dummy_run(..., batch_descriptor=batch_desc,
   aclgraph_runtime_mode=FULL)`），replay 用 `dispatch()` 返回的同一个描述符（`llm_base_proposer.py:1036-1058`）。
   `BatchDescriptor` 是 `frozen dataclass`（`vllm/forward_context.py:29-48`）⇒ 值相等即同一个 entry。
3. **draft 的 dispatch 用的是 runner 的 dispatcher**（`self.runner.cudagraph_dispatcher`），
   且断言 `num_tokens_padded % uniform_decode_query_len == 0`（`cudagraph_dispatcher.py:143-146`）
   —— 任何 5 的倍数（5/10/15/20）在这个断言下都会被取整成 6 的倍数。

### 2.2 结论：**不要**给 draft 单独推导 5 的倍数 bucket

| 方案 | 可行性 | 评价 |
|---|---|---|
| (A) 共用 bucket，**图按真实行捕获 + 桶多的行惰性**（本次采纳） | ✅ 单文件、无需动 runner/启动器 | 图里行数 = 真实行（5/10/15/20），FIA 不多算；桶只影响 eager metadata 的索引范围 |
| (B) 给 draft 单独一套 5 倍数 bucket | ❌ 在这套 stack 里做不到：`adjust_cudagraph_sizes_for_spec_decode` 会把 5→6、10→12、15→18、20→24；FULL key 由 runner 的 `initialize_cudagraph_keys` 生成；`ACLGraphWrapper` 又按 forward context 的 descriptor 建 entry。要做就得改 runner（drafter 自带 dispatcher + 捕获循环按 draft 的 key 迭代），属于跨模块改动 | 而且**即使做到了**，为 2 个请求捕获一个 12 token 的图 = FIA 多算 2 行垃圾（真实只有 10 行）——正是要避免的 |
| (C) 把 capture 计数改成桶大小（`dummy_run` 用 `num_tokens`） | ⚠️ 可以，但要同时把 `query_start_loc` 按 target 的方式插 dummy request（`_pad_query_start_loc_for_fia` 的 mixed 分支），否则 TND 不变量（`hidden_states.dim0 == cu_seqlens_q[-1]`）不成立 | **明确不做**：会把"零垃圾行"变成"每步 1~4 行垃圾" |

**启动器侧建议（主 Agent 定）**：
* `MAX_SEQS=1` 时桶只有 6，draft 真实 5 ⇒ 不可避免 1 行 padding（惰性，安全）；`MAX_SEQS=4` 时桶 `{6,12,18,24}` vs 真实 `{5,10,15,20}` ⇒ 1/2/3/4 行 padding。
* 若要压掉 padding，可以**把 `cudagraph_capture_sizes` 收窄**（例如 `[6,12,18,24]`），但**不要**指望用它制造 5 的倍数桶：取整规则决定不可能。
* 真正必须加的是 `DSPARK_GRAPH_CAPTURE_METADATA=1`（见 §6.3）。

---

## 3. Q3：改哪几行（diff）

**只改 1 个文件、1 处赋值**（`probe_draft/dspark_proposer.py:136-162`，同时同步到生成源
`probe_draftmeta/dspark_proposer.py`，使 `make_draft_files.py` 重生成不会回退，见 §4.4）。

```diff
--- probe_draft/dspark_proposer.py   (改前 md5 77890932a8756221368dd8670fc1679f)
+++ probe_draft/dspark_proposer.py   (改后 md5 a740d2abb0a11826870b9dd07d1995cb)
@@ -134,7 +134,35 @@
             and not vllm_config.speculative_config.enforce_eager
         )
         # Max query tokens depend on whether sampling from anchor or not.
-        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
+        #
+        # [DRAFT-GRAPH-NUMINPUT-FIX] This attribute is a *capacity* for the
+        # per-query buffers (positions / query slot mappings), not the shape the
+        # draft graph computes: DSpark's real query block is
+        # `max_batch_size * num_query_per_req` (5 per request here, one row per
+        # speculative token — `copy_and_expand_dflash_and_dspark_inputs_kernel`
+        # loops `num_query_total = batch_size * num_query_per_req` and its
+        # SAMPLE_FROM_ANCHOR branch maps all of a request's rows onto that
+        # request's `num_speculative_tokens` samples), and `dummy_run` clamps the
+        # *capture* count to this capacity. Under ACLGraph, `_propose` however
+        # pads the draft to the runner's capture bucket
+        # (`cudagraph_dispatcher.dispatch(...).num_tokens`, always a multiple of
+        # `1 + num_speculative_tokens` — see
+        # `adjust_cudagraph_sizes_for_spec_decode`) and then indexes the per-query
+        # buffers with that padded count (`_pad_draft_buffers`,
+        # `build_draft_attn_metadata` -> `dsa_v1.build_for_drafting` ->
+        # `spec_slot_mapping[draft_index - 1][:num_input_tokens]`).
+        # A capacity below the bucket makes `_pad_draft_buffers` a silent no-op
+        # (its `buf[num_tokens:num_input_tokens].fill_(-1)` then covers nothing)
+        # while the padded slice still stays shorter than the assignment target —
+        # that is the `[6, 2] = [5, 2]` crash. Size it like DFlash, i.e. for the
+        # largest bucket the runner can dispatch a draft step to; the extra rows
+        # stay inert (slot -1 / position 0 / parallel-drafting token id) and are
+        # never read by the captured graph, which is captured with the *unpadded*
+        # count, so FIA never computes a junk row.
+        self.max_query_tokens = max(
+            self.max_batch_size * self.num_query_per_req,
+            self.max_batch_size * (1 + self.num_speculative_tokens),
+        )
         # Position ids for the draft query block [max_query_tokens].
         # Overrides dflash:49; v2 uses input_buffers.positions.
         self.positions = torch.zeros(
```

被这一行带动的 buffer（**全部由 `max_query_tokens` 派生**，无需另改）：

| buffer | 位置 | 容量变化（MAX_SEQS=1 / 4, S=5） |
|---|---|---|
| `self.positions` | `dspark_proposer.py:168` | 5→**6** / 20→**24** |
| `self._slot_mapping_buffer` | `:176` | 同上 |
| `_per_group_query_slot_mapping_buffers[gid]` | `:293-296` | 同上（**本次崩溃点**） |
| `dspark_swa_indices_buffer`（DSA builder） | `:279` → `dsa_v1.py:922-931` | 同上（只按 `[:num_actual_tokens]` 使用，本来就够，扩容无害） |

### 3.1 为什么这是安全的（"不会让 FIA 多算一行垃圾"的三条论证）

1. **图里的行数没变**。捕获计数 = `min(batch*num_query_per_req, max_query_tokens)`
   （`dspark_proposer.py:534`）。修复后 `max_query_tokens = 6*max_seqs ≥ 5*batch`（对任意 `batch ≤ max_seqs`），
   所以下界恒取真实值 ⇒ 捕获计数与修复前**逐 batch 相同**（5/10/15/20），
   与 `query_start_loc[-1] = batch*num_query_per_req` 一致（TND 不变量成立）⇒ **图里每一步只算真实行**。
   反过来，如果不扩容量而是想办法让捕获用桶大小，FIA 才会真的每步多算 1~4 行。
2. **桶多的行是惰性的**。`_pad_draft_buffers(num_tokens=real, num_input_tokens=bucket)` 本来就会写
   `input_ids/positions/_slot_mapping_buffer/per-group query slot mapping = pdid/0/-1/-1`
   （`llm_base_proposer.py:2717-2731`）；修复前之所以"没写上"，恰恰是因为 buffer 太短、切片退化成空
   （**静默 no-op**），这也是崩溃的另一半成因。修复后这些行真的被填成 −1。
3. **真正会被设备读的行数由捕获决定，与 replay 的 `num_input_tokens` 无关**。
   设备侧只消费 `spec_slot_mapping[:num_actual_tokens]`（`dsa_v1.py:1331`，`num_actual_tokens = real`）
   + 捕获时烘进图的 `cu_seqlens_q=[0,5,...]`；`spec_slot_mapping` 的 `[real, bucket)` 行虽然被写了
   `[-1,-1]`，但**从未被任何 kernel 读到**（`-1` 也是 `dsa_attn_kv_plan.py:111-120` 明确支持的"pad 行"约定：
   "SparseFlashMla's scatter path receives padded [-1, -1] rows directly"）。

**容量上界证明**：FULL-decode key 满足 `x ≤ uniform_decode_query_len * max_num_seqs`
（`cudagraph_dispatcher.py:212-232`），而 `max_batch_size == scheduler_config.max_num_seqs`
（`vllm/v1/spec_decode/llm_base_proposer.py:138`）、`uniform_decode_query_len == 1 + num_speculative_tokens`
⇒ `max_batch_size*(1+num_speculative_tokens)` 覆盖一切可达桶；并且 `_bs_to_padded_graph_size` 只会**向上取整**
（`cudagraph_dispatcher.py:69-88`）⇒ replay 的桶 ≥ 捕获计数，图永远不会被喂"比捕获更少"的行。

**可选加固（本次未做，避免扩散改动）**：在 `llm_base_proposer._pad_draft_buffers` 里对
`buf.shape[0] < num_input_tokens` 直接 raise，把"静默截断"变成响亮失败。离线 preflight 已经在测试层
钉住这个不变量（`REPLAY ... query-slot buffer covers num_input_tokens`）。

### 3.2 eager 臂（`DRAFT_GRAPH=0`）不受影响（判据④的基线可比性）

`max_query_tokens` 在朴素路径上只有两个读点：缓冲区容量，以及 `dummy_run` 的
`num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)`
（`dspark_proposer.py:534`）。
* `num_reqs > 0` 时（捕获循环、profile、warmup 都是这条）下界恒为 `num_query_total = 5·batch ≤ 5·max_seqs`，
  修复前后同值 ⇒ **dummy_run 的形状不变**；
* `num_reqs == 0` 的分支在当前 stack 不可达：`_dummy_run` 的三条分支都给
  `num_reqs = min(num_tokens, max_num_seqs) ≥ 1`（`model_runner_v1.py:3630-3641`）。
* eager 模式下设备侧只消费 `[:num_actual_tokens] = [:5·batch]`，多出来的容量不参与任何 kernel。

⇒ eager 臂的 `ms/step` 基线（判据④ 的对照）不会因为本修复变动。

---

## 4. Q4：离线可复现判据（含负控）

### 4.1 脚本

`probe_draft/preflight_numinput_fix.py`（纯 stdlib：无 torch、无 NPU、无容器）

它不是"重写一份逻辑"，而是用 **AST 从被 mount 的三个整文件里抽出真实语句**再执行
（`ast.unparse` + `exec`），因此断言不会与产物漂移：

| 抽取来源 | 抽取到的真实语句 | 用途 |
|---|---|---|
| `dspark_proposer.py` | `self.max_query_tokens = ...`（`__init__`） | 容量（**修复点**） |
| `dspark_proposer.py` | `self._per_group_query_slot_mapping_buffers = {...}`（`initialize_attn_backend`） | 崩溃 buffer 的真实容量 |
| `dspark_proposer.py` | `num_query_tokens = min(...)`（`dummy_run`） | 捕获计数 |
| `llm_base_proposer.py` | 整个 `_pad_draft_buffers` 函数体 | padding 行为 |
| `dsa_v1.py` | `self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = get_dsa_attn_kv_plan(...).format_dsa_slot_mapping(...)` | **崩溃行本体** |

fake tensor 精确复刻了本缺陷依赖的两条 torch 语义：
* `t[:n]`（`n > len`）**静默返回更短的视图**，不报错；
* `target[:n] = value` 在 `value` 更短时抛
  `The expanded size of the tensor (n) must match the existing size (m) at non-singleton dimension 0.  Target sizes: [...].  Tensor sizes: [...]`
  （与线上日志逐字同格式）。

同时镜像了 launcher 的 `CAPTURE_SIZES` 推导 + `adjust_cudagraph_sizes_for_spec_decode` 的取整 +
`initialize_cudagraph_keys` 的 FULL key 过滤 + `_bs_to_padded_graph_size`（`--enable-sp 0`，与日志实测一致）。

### 4.2 正控（修复后）

```console
$ cd ~/projects/dsv41/probe_draft
$ python3 preflight_numinput_fix.py            # 见 _numinput_fix/preflight.pass.log
[preflight] PASS=50 FAIL=0
RESULT: PASS
pos_exit=0
```

关键几行（`max_num_seqs ∈ {1,4}` 两种桶形）：

```
-- max_num_seqs=1: buckets=[6]
     [product] self.max_query_tokens = 6 (real block = 5)
  [PASS] REPLAY batch=1 query-slot buffer covers num_input_tokens :: capacity=6 num_input_tokens=6 (real=5)
  [PASS] REPLAY batch=1 spec_slot_mapping[:n] = format(slot_mapping) :: rows written=6
  [PASS] REPLAY batch=1 padded query-slot rows are -1 (no KV write) :: rows[5:6]=[-1]
  [PASS] REPLAY batch=1 real rows keep their (block, offset) :: row[4]=[7, 108]
  [PASS] CAPTURE batch=1 captured count == real query block :: captured=5 real=5 bucket=6
  [PASS] CAPTURE batch=1 TND invariant (hidden_states dim0 == cu_seqlens_q[-1]) :: hidden_states=5 cu_seqlens_q[-1]=5
  [PASS] SAMPLES batch=1 token_indices_to_sample ⊆ real rows :: max index=4 buffer=5
```

### 4.3 负控（改前文件，`probe_draft/_numinput_fix/dspark_proposer.prefix.py`，md5 `77890932...`）

```console
$ python3 preflight_numinput_fix.py --src-dspark _numinput_fix/dspark_proposer.prefix.py
                                                    # 见 _numinput_fix/preflight.prefix_fail.log
-- max_num_seqs=1: buckets=[6]
     [product] self.max_query_tokens = 5 (real block = 5)
  [FAIL] REPLAY batch=1 query-slot buffer covers num_input_tokens :: capacity=5 num_input_tokens=6 (real=5)
  [FAIL] REPLAY batch=1 spec_slot_mapping[:n] = format(slot_mapping) :: The expanded size of the tensor (6) must match
         the existing size (5) at non-singleton dimension 0.  Target sizes: [6, 2].  Tensor sizes: [5, 2]
...
[preflight] PASS=30 FAIL=4
RESULT: FAIL
neg_exit=1
```

**与线上日志逐字一致**（`logs/perf/dgvB_serve.log:1128`）：
`RuntimeError: The expanded size of the tensor (6) must match the existing size (5) ... Target sizes: [6, 2].  Tensor sizes: [5, 2]` ✅

负控在 MAX_SEQS=4 下还额外暴露 **batch=4** 的同类崩溃（容量 20 < 桶 24，`[24,2]` vs `[20,2]`）——
也就是说这个缺陷不只在 MAX_SEQS=1 上存在。

### 4.4 交付面自检（全部在 A3-node1 上执行，均未占卡）

| 检查 | 结果 |
|---|---|
| `python3 -m py_compile dspark_proposer.py llm_base_proposer.py dsa_v1.py preflight_numinput_fix.py` | PASS |
| `python3 make_draft_files.py --check-only`（线 3 的 8 项集成自检） | **集成成功**（8/8 PASS，含 `py_compile`、F3、0004/0005/0006/0002、无探针残留） |
| `python3 verify_equivalence.py` | PASS（"剥探针后与参考版逐字节相同" 仍然成立） |
| `python3 precheck_ast.py --probe-draft ~/projects/dsv41/probe_draft` | PASS（硬禁已解除等结论不变） |
| **重生成一致性**：`_numinput_fix/check_regen.py ../probe_draftmeta/dspark_proposer.py dspark_proposer.py 4` | `strip(src) md5 = a740d2ab... == ref md5`，`IDENTICAL = True` ⇒ **已同步改 `probe_draftmeta/dspark_proposer.py`**，`make_draft_files.py` 重生成不会回退本修复 |

---

## 5. Q5：上卡验证方案（一键脚本，等窗口）

脚本：`probe_draft/verify_numinput_fix_a21.sh`（`bash -n` 通过）。默认值即任务书要求：
`chips 8-15`（`serve_a21.sh` 的 `PRESET=back8`）、`PORT=8020`、`NAME=dsv41-a21-perf`、`LOAD_FORMAT=dummy`、
`MAX_SEQS=1`；只启停 `dsv41-a21-perf` 这一个容器。

```bash
# ① dummy：起服 + 判据 ①②③
MODE=start ARM=graph RUN_ID=nif_g1 bash ~/projects/dsv41/probe_draft/verify_numinput_fix_a21.sh

# ② 真权重（主 Agent 给窗口后）：graph 臂与 eager 臂各跑一轮
MODE=start ARM=graph LOAD_FORMAT= RUN_ID=nif_real_g MAXTOK=192 \
  bash ~/projects/dsv41/probe_draft/verify_numinput_fix_a21.sh
MODE=start ARM=eager LOAD_FORMAT= RUN_ID=nif_real_e MAXTOK=192 \
  bash ~/projects/dsv41/probe_draft/verify_numinput_fix_a21.sh

# ③ 判据④：用 eager 臂的客户端输出复算
REF_JSON=~/projects/dsv41/logs/perf/a21/verify_numinput_nif_real_e.log MODE=attach \
  bash ~/projects/dsv41/probe_draft/verify_numinput_fix_a21.sh
```

脚本内部（graph 臂）自动注入 `EXTRA_ENV="DSPARK_GRAPH_CAPTURE_METADATA=1"`（`serve_a21.sh` 的
`__EXTRAENV__` 会在容器内 `set -a` 导出），并使用独立 tmux 会话名（`RUN_ID_s`），不会误杀别人的 `a21p`。

### 判据与期望指纹

| # | 判据 | 观察点 | 期望 |
|---|---|---|---|
| ① | 起服 READY | `/health` 200 | ≤ READY_TIMEOUT_S（默认 900 s）内 200；容器不得"出现过又消失" |
| ②a | **不发请求也不崩** | 空转 `IDLE_S`（默认 90 s）后：`/health` 仍 200、容器仍在、`Traceback` 计数不增 | 全部成立 |
| ②b | 8K prompt 正常返回 | `delivery_20260914/assets/p42_t4_quote.sh`（`TOKENS=8192`） | 无 `ABORT/error`，打印 `ms_per_step / accept_length / ttft_s` |
| ③ | 静态内核未退化 | `grep -ac "static_kernel.py:650" <serve log>` | **0** |
| ④ | 真权重 A 不低于 eager | graph 臂 vs eager 臂 `ms_per_step` 中位数（同口径多点） | `graph ≤ eager + 0.5ms`（脚本打印 delta；**单点不可判**，需多点/多轮） |

**图真的进图了（三条指纹，脚本会打印）**：

* `Wrapping draft model with ACLGraphWrapper` = **8**（TP8；stock/eager 臂应为 0）
* `[dspark-graph-capture] capture #N descriptor=BatchDescriptor(num_tokens=6, num_reqs=1, uniform=True, ...)
   ... num_query_total=5 num_input_tokens=5` ← **本修复的关键指纹**：`num_input_tokens=5` 证明图按真实行捕获
  （若这里出现 6，说明有人把捕获改成桶大小，垃圾行风险回归）
* `Target sizes` 计数 = **0**（本次崩溃指纹）；`aclmdlRIExecuteAsync ÷ 步数 ≈ 2` 需 msprof（可选，见
  `exp_tools/verify_draftgraph.sh` 的做法）

### 上卡前必须先做的启动器改动（主 Agent 负责，我未改）

```diff
 # [DRAFT-GRAPH-ENABLE] ...
 if [ "$DRAFT_GRAPH" = "1" ]; then
   MOUNTS="$MOUNTS -v .../probe_draft/{dsa_v1,dspark_proposer,llm_base_proposer}.py ..."
   MOUNTS="$MOUNTS -e DSPARK_DRAFT_METADATA_MODE=sync"
+  # 必须：否则 draft 图在"无 attention metadata"的兜底分支上捕获（output.fill_(0)），replay 里没有 draft attention
+  MOUNTS="$MOUNTS -e DSPARK_GRAPH_CAPTURE_METADATA=1"
 fi
```
（等价做法：`EXTRA_ENV="DSPARK_GRAPH_CAPTURE_METADATA=1"`，就是本脚本自动做的事。）

---

## 6. 根因复核：与任务书对照（3 处更正/补充）

任务书的根因链 **成立**：`num_query_per_req=5`（`dspark_proposer.py:96-99`）⇒ `max_query_tokens=5`
（改前 `:137`，改后 `:162`）⇒ query slot mapping buffer 只有 5 行（改前 `:266`，改后 `:293`）⇒ runner 给的 `num_input_tokens=6`
（`llm_base_proposer.py:1042/1056`，bucket）⇒ `slot_mapping[:6]` 退化成 5 行（`llm_base_proposer.py:2665`）
⇒ `dsa_v1.py:1176` 赋值时 `[6,2] = [5,2]` 崩。三点补充：

### 6.1 更正：捕获计数**没有**被 clamp 到 5（我一开始也误判过）

改前 `max_query_tokens = max_batch_size * num_query_per_req`（= 5·MAX_SEQS），
而 `dummy_run` 的 clamp 上界正是它 ⇒ `min(batch*5, 5*max_seqs) = batch*5` **恒取真实值**。
也就是说"图按真实行捕获"这件事**在修复前后都成立**（这也是为什么 crash 只发生在 replay 侧 metadata 构建上、
捕获/起服是成功的）。真正的缺陷只有一条：**容量 < 桶**。

### 6.2 补充：TND 不变量与 `_pad_query_start_loc_for_fia` 的静默 no-op

draft 在 MAX_SEQS=1 下 `num_reqs==num_reqs_padded==1`，`6 == 1*uniform_decode_query_len(6)` ⇒
`_pad_query_start_loc_for_fia`（`model_runner_v1.py:931-990`）走 uniform 分支、切片为空 ⇒
`query_start_loc` 仍是 `[0,5]`，而 `num_input_tokens=6` ⇒ **eager metadata 本身不自洽**（`cu_seqlens_q[-1] < num_input_tokens`）。
它无害的**唯一理由**是：图是 5 行捕获的，`num_input_tokens=6` 只用于 eager 侧 buffer 索引，
设备侧从不看它。**这条不变量必须钉住**——preflight 的 `CAPTURE` 两项 + 上卡指纹
`num_input_tokens=5` 就是它的守卫。

### 6.3 补充（重要）：dgvB 那一轮的 draft 图**没有 attention**

`serve_a21.sh`（以及 `exp_tools/verify_draftgraph.sh`）都没有设 `DSPARK_GRAPH_CAPTURE_METADATA=1`，
而 `dspark_proposer.py` 里该 flag 默认 0（`_DSPARK_CAPTURE_METADATA`）⇒ 捕获期
`multi_steps_attn_metadata=[]` ⇒ `AscendDSAImpl.forward` 走 `attn_metadata is None` 兜底
（`dsa_v1.py:1689-1698`：`output.fill_(0)` / o_proj-on-zeros）⇒ **capture 里根本没有 draft attention**。
证据：`logs/perf/dgvB_serve.log` 里 `grep -c "dspark-graph"` = **0**（开了 flag 才会打
`[dspark-graph-capture] capture #...`）。这解释了"起服成功但第一个真实请求就崩"之外的第二层风险：
**即使只修崩溃，A 也会崩到 ~1.0**。上卡前必须把这个 env 打开。

### 6.4 补充：DFlash 为什么没这个 bug（对照实验的价值）

`dflash_proposer.py:53` 的 `max_query_tokens = max_batch_size * (1 + num_speculative_tokens)`，
而 DFlash 的真实块就是 `1 + num_speculative_tokens = 6`/req ⇒ 真实计数与桶粒度**同拍**（5·? 不存在），
所以它的 `spec_slot_mapping[:num_input_tokens]` 永远对得上。DSpark 的 `sample_from_anchor` 把每请求的块
从 6 缩到 5 ⇒ 与桶粒度错位 ⇒ 只有 DSpark 会踩。本修复本质上是"把 DSpark 的容量口径退回 DFlash 的口径"。

---

## 7. 交付物与 md5

| 路径（A3-node1 `~/projects/dsv41/`） | md5 | 说明 |
|---|---|---|
| `probe_draft/dspark_proposer.py` | `a740d2abb0a11826870b9dd07d1995cb` | **主交付**（改前 `77890932a8756221368dd8670fc1679f`） |
| `probe_draftmeta/dspark_proposer.py` | `e872ddcbddaa1663b48655106e819627` | 生成源同步（保证 `make_draft_files.py` 不回退） |
| `probe_draft/preflight_numinput_fix.py` | `d07468e25892cef7a5e9b67dac007486` | 离线 preflight（正/负控） |
| `probe_draft/verify_numinput_fix_a21.sh` | `cce3261c292871537f7d79ccaadb0b60` | 上卡一键脚本 |
| `probe_draft/_numinput_fix/dspark_proposer.prefix.py` | `77890932a8756221368dd8670fc1679f` | 负控基线（= 改前 `probe_draft/dspark_proposer.py`） |
| `probe_draft/_numinput_fix/dspark_proposer.draftmeta.prefix.py` | `2677db83fab8adf8882936b3fa9fb695` | 负控基线（改前 `probe_draftmeta`） |
| `probe_draft/_numinput_fix/check_regen.py` | `2dcd155ca83db25978d17121eb930768` | 重生成一致性检查 |
| `probe_draft/_numinput_fix/preflight.pass.log` / `preflight.prefix_fail.log` | — | 正/负控原始输出（含 exit code 行） |
| `reports/draft-graph-numinput-fix.md` | — | 本报告 |

未改动的既有产物（仅供对照）：`llm_base_proposer.py`（`1c79519cf7daca57f58105617145aaab`）、`dsa_v1.py`（`3c86de8250bd9cca7b886ff29180f8cd`）。

---

## 8. 未验证项与风险

1. **未上卡**：本修复的全部证据是"代码阅读 + AST 抽取真实语句的离线 shape/语义复算"。
   真正的 READY / 空转不崩 / 8K 返回 / `static_kernel.py:650=0` / 真权重 A 必须按 §5 在窗口内确认。
2. **判据④（A 不低于 eager）未测**，且它依赖 6.3 的 env 修复：若上卡时忘了
   `DSPARK_GRAPH_CAPTURE_METADATA=1`，会看到"不崩但 A≈1.0"，那不是本修复的问题（指纹：没有
   `[dspark-graph-capture]` 行）。
3. **多请求桶（MAX_SEQS>1）的图**：`MAX_SEQS=4` 时会捕获 4 个桶（6/12/18/24→4 个 batch），
   每个桶的图按真实行（5/10/15/20）捕获；离线 preflight 覆盖了 4 个 batch 的算术，但**没有真机验证**。
   上卡建议先 `MAX_SEQS=1`，再补一轮 `MAX_SEQS=4` + 2/4 并发请求。
4. **`_pad_query_start_loc_for_fia` 的静默 no-op**（6.2）是"靠不变量活着"的脆弱点。
   如果后续有人改捕获计数或 bucket 语义，preflight 的 `CAPTURE` 项会 FAIL，但那需要有人真的跑它——
   建议把 `python3 probe_draft/preflight_numinput_fix.py` 加进交付前的自检清单。
5. **`make_draft_files.py` 的再上游**：本次已同步 `probe_draftmeta/dspark_proposer.py`，
   但若有人重跑 A3-node2 的 `patches/draft_graph/build_wholefiles.py`（从 `fix2_files/` 重新生成
   `probe_draftmeta/`），本修复会被覆盖（0005/0006 也有同样的脆弱性）。建议把
   `[DRAFT-GRAPH-NUMINPUT-FIX]` 也搬进那条生成链（本轮刻意未动，避免跨机跨域改动）。
6. **显存**：扩容只有 `6*max_seqs` 行 int32（MAX_SEQS=4 时每 buffer 96B）+ DSA `dspark_swa_indices_buffer`
   多几行，量级可忽略；但仍建议对照 `GPU KV cache size`（脚本会打印一行）确认无变化。
7. **真权重判据④的口径**：与既有报告一致，`>16384` 上下文单会话输出本身非确定，**判据④必须用同口径多点
   （同 TOKENS/MAXTOK、graph 与 eager 各一轮以上）**，不能用"三次一致"。
