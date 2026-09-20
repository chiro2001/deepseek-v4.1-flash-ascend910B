# DSpark draft 入图 —— 排查记录（2026-09-20，含 6 个被否证的假设）

> 目标：把 draft 前向放进 ACLGraph，且 A（接受长度）不低于 eager。
> 现状：**未达成**。本文记录**已确证的事实**与**已否证的假设**，避免重复烧起服时间。

## 0. 一句话现状

`DRAFT_GRAPH=1` ⇒ **A ≈ 1.07 / 42 tok/s**（坏）；`DRAFT_GRAPH=0`（draft eager）⇒ **A ≈ 2.65 / 92 tok/s**（好）。
同口径（conc=1 跑 **8 条**取中位），差值稳定可复现。

## 0.1 ★★ 问题的性质已改变：**不是精度问题，是"draft 不被接受"的性能问题**

**2026-09-20 实测**（graph 臂、`DRAFT_GRAPH=1`）：10 条质量判据**全部正确**——

| # | 判据 | graph 臂实际输出 |
|---|---|---|
| 1 | 天空为什么蓝 | 瑞利散射，通顺 ✓ |
| 2 | 计算 17×23 | **391** ✓ |
| 3 | 中译英 | 通顺译出 ✓ |
| 4 | 写 Python 函数 | 正确实现（列表推导式） ✓ |
| 5 | TCP 三次握手 | 正确分步解释 ✓ |
| 6 | 红楼梦作者 | **曹雪芹 / 清朝** ✓ |
| 8 | 40×60%÷2 | 正确推理（女生 24 人…） ✓ |
| 10 | 反转字符串 | **`!dlrow ,olleH`** ✓ |

⇒ **target 模型没有坏，文本是对的**。低 A 只意味着 **draft 的建议没被接受**（白跑），
**不是**输出错误。这显著降低了这条线的风险等级：`DRAFT_GRAPH=1` 是一个**性能回退**，不是**正确性事故**。
（仍然不应发布：它比 `DRAFT_GRAPH=0` 慢 2.2×。）

**这个事实同时收紧了嫌疑范围**：既然 target 正确、且 draft 的**第 1 步** token 与 eager 臂逐位相同
（见 F2），低 A 就只能来自 **draft 从第 2 步起状态就坏了** ——
而此前的 token dump **只看了第 1 步**（每 rank 打一次），看不到后续退化。
⇒ 已新增**按步编号**的 dump（`[dspark-token] step=N ... rows=[该请求的 5 个 token]`），
下一步要在**同一次起服内**对比"图臂 vs eager 臂"的逐步 token（脚本：`/tmp/draftstep_ab.py`，
用 `DRAFT_FORCE_EAGER` 热切换 + 每臂唯一前缀绕开 prefix cache）。

## 1. ★ 口径纪律（两次翻车换来的）

1. **`bench_concurrency.py` 的 prompt 条数 = `--concurrency` 列表的最大值**。
   `--concurrency 1` **只发 1 条**请求，而 README §3.2 的基线是 **64 条中位数**；
   A 是**发放级抽签**（历史 163 发：steep 15% / flat 7% / shallow 77%），单发不可与中位比。
   实测差距：同一条臂，单发 A=2.04 / 67 tok/s，**8 发中位 A=2.65 / 92.0 tok/s**（判据从 FAIL 翻成 PASS）。
   ⇒ `tools/draft_graph_guard.sh` 与 `tools/draft_arm_probe.sh` 已固定用 `--concurrency 1,2,4,8` 并取 `conc=1` 行。

2. **端口会被别的租户抢**。实测 8020 上跑过**别人的 sglang**（`/home/models/DeepSeek-V4-Flash-W8A8`），
   而我用 `curl /health` 轮询，得到 `200` 却**不是自己的服务** ⇒ 后续所有轮询都必须用
   `/v1/models` 断言 `"id":"deepseek-v41"`（工具：`/tmp/mysvc.sh`）。

3. **探针不能进 capture 区间**。capture 期做 D2H（`.tolist()`/`.sum()`/`.item()`）会让图捕获失败：
   `Worker proc VllmWorker-N died unexpectedly` + `RuntimeError: cancelled`（引擎起不来）。
   且 capture 期日志量（每层 × 8 rank）本身会把 worker 拖垮。
   ⇒ 所有探针都要 `if get_forward_context().capturing: return`，并加**独立预算**。

4. **`_propose` 每步都跑（图之外）**，是唯一可靠的观测窗口；
   `dummy_run`/capture 只在起服时跑一次。
   但探针预算要按"真实 decode"过滤 —— 否则会被 **profile run**（同样几十行 × 8 rank）吃光。

## 2. 已确证的事实（实测）

| # | 事实 | 证据 |
|---|---|---|
| F1 | **draft 三个 patch 文件本身是好的** | 同口径下 draft eager = A 2.648 / 92.0 tok/s |
| F2 | **两臂的 draft 产出 token 逐位相同** | 均为 `[223, 5038, 10849, 271, 5038]` |
| F3 | **capture 与 replay 的契约张量地址完全相同** | `query_start_loc/seq_lens/slot_mapping/block_table/start_pos/sas_metadata` 六个 data_ptr 逐位一致 |
| F4 | **replay 时这些 buffer 的值是正确的** | `seq_lens=[1037]`、`start_pos=[1032]`、`slot_mapping=[[109,8]…]`（capture 期是 dummy 的 `[11]/[6]/全0`） |
| F5 | **capture 期 draft 层确实拿到了 attention metadata** | `[dsa-probe] HAS_METADATA md=dict n_keys=3 hit=~mtp.*` |
| F6 | **DSpark 走 `parallel_drafting=True` ⇒ `_run_merged_draft` early return** | 5 个 token 全来自 step0 一次前向；循环体是**死代码** |
| F7 | **形状契约一致** | `num_query_per_req=5 / net_new_slots=4 / max_query_len=5 / decode_token_per_req=5`，capture 与 replay 同 |
| F8 | **capture 期会把 dummy slot_mapping 写进真实 KV cache** | `[dsa-write] cmp_kv capturing=True` 与 replay 同一个 `cache_ptr`，capture 期 slot 全 0（= 物理 block0/offset0..4） |
| F9 | **capture 的 bucket 与 replay 不同** | 默认 capture `key=5x1`、replay `key=6x1`（见 §3 假设 3） |

## 3. 已否证的假设（每条都做了 A/B，均**无改善**）

| # | 假设 | 实验 | 结果 |
|---|---|---|---|
| 1 | **capture 期 KV 污染**（F8） | `DSPARK_CAPTURE_PAD_SLOTS=1`：capture 期把 slot_mapping 置 `-1`（pad，不写） | A=1.081 —— **无改善** |
| 2 | **`max_seqlen_kv` 标量被烘进图**（capture 时 `[11]`、replay 时 `[1037]`，而它是 metadata 算子的**标量参数**） | `DSPARK_CAPTURE_MAXSEQLEN=8192`：捕获时改用上界 | A=1.087 —— **无改善** |
| 3 | **capture/replay bucket 不一致**（F9；`_propose` 走两次 `cudagraph_dispatcher.dispatch`，而 `dummy_run` **从不** dispatch） | `DSPARK_CAPTURE_DISPATCH=1`：让 capture 也走相同 dispatch。**修复生效**（capture key 从 `5x1` 变 `6x1`，与 replay 一致） | A=1.081 —— **修复正确但 A 未变** |
| 4 | **`topk_indices_buffer` 被 target/draft 共享**（`_maybe_share_topk_indices` 把 target 的 buffer 直接赋给 draft 的每个模块 ⇒ draft 图 replay 会覆写 target 的 buffer，导致 verify 与 draft 不一致） | `DSPARK_NO_TOPK_SHARE=1`：跳过共享（draft 自算 topk）+ 配套把 `set_skip_topk(True)` 关掉（否则读到自己的旧值） | A=1.075 —— **无改善** |
| 5 | **`target_positions` 地址漂移** | 捕获/重放共享常驻缓冲（[TARGETPOS-FIX-v3]） | A=1.070（与未改前逐位相同）—— **无改善**，已回退 |
| 6 | **`start_pos_draft` 常驻化的表达式不等价** | 回退成 stock 的 `self.seq_lens[:num_reqs] - seq_lens_q` | 无变化 —— **已排除** |

## 4. 当前最大嫌疑（尚未验证）

## 4.0 ★★★ 竞态：**已拿到非确定性实证 + 定位到具体源码行**（2026-09-20 晚）

> ⚠️ 本节（竞态）后来被**否证**（见 §4.1），根因是 §4.2 的"捕获期固化"。

## 4.2 ★★★★ 根因：`build_model_inputs_first_pass` 在图内，依赖两个捕获期固化的量

**结论（2026-09-20 深夜，静态分析确认）**：
`build_model_inputs_first_pass` 被 `_run_merged_draft` 调用，而 `ACLGraphWrapper` 包的正是
`_run_merged_draft` ⇒ **它是图内代码**。它依赖两个量：

### 缺陷 ①（致命）：`_context_slot_mapping_buffers` 在捕获时是 `None`

它的**全部**赋值点（grep 过）：

| 位置 | 值 |
|---|---|
| `dspark_proposer.py:215`（`__init__`） | **`None`** |
| `dspark_proposer.py:357`（`set_inputs_first_pass` 内） | **`None`** |
| `dspark_proposer.py:406`（`set_inputs_first_pass` 内） | 真实 list |

而**捕获走的 `dummy_run` 从不调用 `set_inputs_first_pass`**（它只设 `_dflash_num_context`）
⇒ **捕获时该值仍是 `None`**。

下游 `models/deepseek_v4/dspark.py:245`：
```python
def precompute_and_store_context_kv(self, context_states, context_positions,
                                    context_slot_mapping=None) -> None:
    if context_states.numel() == 0 or context_slot_mapping is None:
        return                      # ★ 提前返回：一个 context KV 都不写
    for layer_idx, layer in enumerate(self.layers.values()):
        ...
        self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)
```

⇒ **捕获进图的那次调用直接走 `return`，图里没有任何"写 context KV"的算子。**
每次 replay 都跳过整个上下文 KV 写入 ⇒ draft 读到空/脏上下文 ⇒ **token 全错、A≈1.07**。

### 缺陷 ②（同链、次要但同样错）：`num_context` 是 Python int

`dflash_proposer.py:295` `num_context = self._dflash_num_context` 决定
`_dflash_hidden_states[:num_context]` / `_context_positions_buffer[:num_context]` /
`context_slots[:num_context]` 的 **slice 长度**，会被烘进图：
* 捕获（`dspark_proposer.py:767`）= `num_input_tokens`（桶对齐，单请求桶 **6**）
* 重放（`dspark_proposer.py:358`）= `int(cad.query_start_loc_cpu[batch_size])`（真实值，实测 32/256/1024）

### 这解释了**全部**已知现象

| 观察 | 解释 |
|---|---|
| **真实输入捕获 → replay == eager** ✅ | harness 在捕获前提供了真实的 `context_slot_mapping`（非 None）⇒ 图里有 KV 写入 |
| **dummy 捕获 → 0/5** ❌ | `None` ⇒ 图里没有 KV 写入 |
| **地址全同** ✅ | 传的是 `None`，根本没有地址可比 |
| **16 个标量全同** ✅ | 这个分支判断不在那 16 个里 |
| **shape 此前从未查过** ✅ | `None` 没有 shape |
| **每步都错、pos0 仅 0.047** ✅ | 上下文 KV 从来没写进去 |
| **此前 6 个修复全无效** ✅ | 它们都在改别的量，没碰这条路径 |
| **eager 稳定、graph 稳定但两者不同** ✅ | 两条路径执行的是**不同的代码分支**（一个 return、一个真写 KV） |

### 修法（已实现，门控 `DSPARK_HOIST_CONTEXT_KV`，默认 `1`）

**把该调用移出图边界**（与 metadata 的处理方式一致）：
1. `_propose` 内、`run_draft()` **之前**，图外执行一次
   `build_model_inputs_first_pass(num_input_tokens, self._context_slot_mapping_buffers)`
   —— 此时 `set_inputs_first_pass` 已填好真实 slots、`_dflash_num_context` 也是本步真实值；
2. `_run_merged_draft` 内**跳过**该调用（`method=="dspark"` 且开关开启时）。

⇒ 每步都用真实 slots 与真实 nctx，**两个缺陷一并消除**。
门控设 `0` 可回退到原行为做 A/B。

**验证状态**：修法已实现并同步到 A3-node1，**单 chip A/B 验证正在进行**（另一个子代理）。
未验证前**不要**把它当已成立的结论。

### ❌ 修法验证结果：**hoist 单独不够**（2026-09-20 实测，单 chip harness）

子代理跑了 4 个臂（`--capnone` = 捕获期 slots 为 None，即生产形态）：

| 臂 | `DSPARK_HOIST_CONTEXT_KV` | 捕获期 `precompute` 调用 | 捕获期 `slots_is_none` | replay == eager? |
|---|---:|---:|---|---|
| H0a / H0b | **0** | **1 次**（图内含 KV 写入） | **true** | ❌ 0/5（首分叉=0） |
| H1a | **1** | **0 次**（图内无 KV 写入） | true | ❌ **仍 0/5** |

探针原文（确认了 §4.2 的诊断）：
```
[hoist-probe] capture 期: {"capturing": true, "nctx": 6, "states_rows": 6,
                            "slots_is_none": true, "n_slot_tensors": null, ...}
[hoist]       replay-前:   slots_none=False slot_ptrs=['0x12d300660c00'×3] slot_elem0=[3458,3458,3458]
[hoist]       replay 期 precompute 调用=[{... "slots_is_none": false ...}]   ← HOIST-ON 时图外已写
[hoist]       replay run#0..2 = [[23950,201,15,19,16]]   （eager = [[18834,85,49016,25232,4373]]）
```

**⇒ 两个结论**：
1. **诊断被证实**：捕获期 `_context_slot_mapping_buffers` 确实是 `None`，图里确实没有 KV 写入算子。
2. **但"把 KV 写入搬到图外"不足以修复** —— KV 已在 replay 前写对，结果仍 0/5
   ⇒ **W（捕获期用真实 slots，✅）与 B'（图外写 KV，❌）之间还有别的差异**。

**处置**：`DSPARK_HOIST_CONTEXT_KV` 默认已改回 **`0`**（不改变生产行为，保留供实验）。
根因仍在追：已给两个子代理分别派了
① **W→B 差分二分**（从可用点出发逐个回退成生产形态，找第一个变坏的点）；
② **"图内算子清单"对比 + `_store_standard_swa_kv` 的副作用审计 + stream 有序性**。

### 两条必须记住的方法论事实（子代理穷举扫描得出）

## 4.3 ★★★★ 根因确定：**捕获期的 `seq_lens` 值域被固化**（单 chip 双向对照）

### 决定性证据（另一个子代理，`dsg-fixB`）

**① 双向对照表**（同 harness、同输入、同 metadata、同 kernel 直方图）：

| # | context KV 写入 | **捕获期 `runner.seq_lens`** | replay==eager |
|---|---|---|---|
| ① | 图内写 | 5（dummy） | ❌ |
| ②③ | 图外写(+sync) | 0（dummy） | ❌ |
| ④ | 图外写 | **1032（真实）** | **✅** |
| ⑤ | 图内写 | **1032（真实）** | **✅** |
| ⑥ | 图外写 | capture=0，**replay 前把 buffer 改成 1037** | ❌ |
| ⑦ | 图内写 | capture=1032，**replay 前把 buffer 改回 5** | ❌ |

**⑥⑦ 是双向对照** ⇒ **replay 期怎么改那个 buffer 都没用，只有 capture 期的值说了算**
（排除了"地址漂移"与"图读同一 buffer 内容"两种解释）⇒ 真·**值固化**。

**② 排除的三条**（都有数据）：
* **kernel 直方图两臂逐项相同**（50 种 / 2088 次；`SparseAttnSharedkv` ×24、
  `aclnnScatterNdUpdateSk_*` ×48、`InplacePartialRotaryMul` ×72 全一致）⇒ 不是"图里少了算子串"；
* **replay 期交给图的 metadata 逐项相同**（`seq_lens=[1037]`、`start_pos=[1032]`、
  `slot_mapping=[27,8..12]`、`sas_metadata` sum/非零数、`dspark_swa_indices` 的 **ptr 与内容**、
  `block_table`、`ori_win_left/right`）⇒ 分叉 100% 在"捕获期定型的东西"里；
* **把 replay 期新算的 indices 内容拷回 capture 地址** ⇒ 无效 ⇒ 不是"图读 capture 地址的 tensor 内容"。

**③ V×R 矩阵**（每格独立进程，eager(真实 R) vs capture(V) → replay(真实 R)×2）：

| V \ R | 6 | 1032 | 8192 | 65536 | 262144 |
|---|---|---|---|---|---|
| **6** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **1032** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **8192** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **65536** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **262144** | ❌ | ❌ | ❌ | ❌ | ✅ |

细扫（R=1032）：V=1032/1033/1040/1064 ✅，**1160/1290/1024/1016/1000/900 ❌**
⇒ 规律 **R ≤ V ≤ R+122**（122 ≈ `sliding_window(128) − num_context(6)`），不是对角线、不是同量级、也不是"V≥R"。

**④ ★ 打开"索引缓冲地址常驻"后，`cap_len` 变得与 R 无关**：
```
stable-idx enabled=True buf=(2048,1,256)
V=8192 R=6/8192/1032/65536/262144  → 全部 PASS（5/5）
V=1024 R=262144 PASS ; V=262144 R=1032/65536 PASS
V=6    R=1032/262144 FAIL                        ← V 太小仍失败
```
⇒ **存在与 R 无关的安全值 `cap_len = max_num_tokens`（生产 8192），但前提是索引缓冲地址常驻。**

机制：`dsa_v1.py:797` drafting 路径 `_device_metadata_enabled=False` ⇒ 某些分支
`build_dspark_swa_indices(*args)` **不带 buffer ⇒ 每次新分配** ⇒ 图里烘的是捕获期的指针。

### ⚠️ 一个致命顺序 bug（已修）

原实现把 (a)(b) 两段赋值放在 `_build_capture_draft_attn_metadata(...)` **之后**（`else` 分支里），
而该函数用 `runner.seq_lens` 构建捕获期 metadata ⇒ **修复完全不生效**。
实测（子代理）：同参数只改顺序，`--order early` ✅ / `--order late` ❌。
**已修**：赋值搬到构建之前（现在 `dspark_proposer.py:719`，构建在 `:745`）。

### 修法（三部分，门控 `DSPARK_CAPTURE_VALUE_FIX`，默认 **0**）

1. **(a) 代表性值域**：捕获前把 `runner.seq_lens[:n]` / `optimistic_seq_lens_cpu[:n]` 填成
   `DSPARK_CAPTURE_SEQ_LEN`（默认回落 `max_num_tokens`，生产 8192）；
2. **(b) 恢复 context KV 写入**：捕获前把 `_context_slot_mapping_buffers` 填成真实的
   per-group 缓冲 list（这些缓冲在 `initialize_attn_backend` 就建好、地址常驻）
   —— 实测"只做 (b) ❌"、"只做 (a)+图内写 ✅"。
3. **(c) 索引缓冲地址常驻**：让 drafting 路径用常驻 buffer（**待定** ——
   已核实 drafting 路径（`:1175`）**本来就传** `buffer=self.dspark_swa_indices_buffer`，
   而新分配来自 `build_req_metadata` 那条（`:1444` 的 else）。**正在验证生产捕获走哪条路径**，
   以决定 (c) 是否必需、以及用哪种最小改法）。

**验证状态**：顺序已修 + (a)(b) 已实现（默认关）；(c) 的取舍与 A/B 正在单 chip 上确认。

1. **重放时 Python 完全不执行** —— 扫描器给 `_runnable` 包计数器，整个 capture+replay 只有 `hooked call #1`
   （`ACLGraphWrapper` 重放路径只执行 `entry.aclgraph.replay()`）。
   ⇒ 能影响结果的**只有三类**：捕获时绑定的**地址**、**shape**、以及**图内被读的 Python 标量**。
2. 该扫描已把这三类逐一排除：地址漂移（收紧到 0 项）、shape 固化（**0 条**）、
   图内标量 `_dflash_num_context`（改掉无效）。
   ⇒ 若三类都排除，问题就落在**"同一算子在图内录制 vs 重放时行为不同"**或**图内外语义差异**上。

### 决定性实验（同一 prompt、同一次起服、跑 4 次：graph, graph, eager, eager）

每步 dump `[dspark-token] step=N use_graph=X start_pos=[seq_lens] rows=[5 个 draft token]`。

| 运行 | pos=37（第 1 步） | pos=39（第 2 步） |
|---|---|---|
| graph#1 | `[1757, 389, 77640, 19359, 9833]` | `[16783, 31325, 20871, 470, 303]` |
| graph#2 | **完全相同** | `[16783, 31325, 20871, 470, `**`97396`**`]` |
| eager#1 | **完全相同** | `[16783, 31325, 20871, `**`6881`**`, 303]` |
| eager#2 | **完全相同** | `[16783, 31325, 20871, `**`6881`**`, 303]` |

**三条推论**：
1. 第 1 步（pos=37）**四跑全同** ⇒ 起点确定、与模式无关。
2. 四次运行的步进都是 **37→39（都接受 2 个）** ⇒ **进入第 2 步时输入状态相同**（控制住了"位置/内容不同"这个混淆）。
3. 同输入下：**eager 两次完全可复现（6881）**；**graph 两次不同（第 5 个 token 303 vs 97396）**
   ⇒ **graph 模式下 draft 计算是非确定的** = **竞态**。

### 机制（源码级，已定位到具体行）

`vllm_ascend/compilation/acl_graph.py` 的 replay 路径：

```python
is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
if not self.enable_enpu and need_sync:
    torch.npu.current_stream().synchronize()      # ← DSpark 会跳过这里
entry.aclgraph.replay()
```

注释原文：「When FULL + **EAGLE draft** (merge path), replay does not need this barrier.」

**但 `use_eagle` 是个宽泛判定**（`vllm/config/speculative.py`）：

```python
def use_eagle(self) -> bool:
    # NOTE: This method is usually a stand-in for "speculative decoding using
    # target model hidden states"
    # TODO(ben): Refactor this so the naming is clearer
    return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")   # ← dspark 在内
```

⇒ DSpark 的 `is_draft_eagle=True` ⇒ **replay 前的屏障被跳过**。
而本配置又开了 `multistream_overlap_shared_expert` / `multistream_dsv4_dsa_overlap`
⇒ 跨 stream 的写/读缺少显式排序 ⇒ **draft replay 可能读到尚未写完的 metadata / KV**。

**这解释了为什么此前 6 个修复（全部针对 capture 期）都无效** —— 问题在 **replay 期的同步**，不在 capture。

### 判定实验（已实现，正在跑）
`DSPARK_DRAFT_SYNC_BEFORE=1`：在 draft replay **之前**插入一次 `current_stream().synchronize()`。
**若 A 从 1.07 回到 ~2.6，即坐实竞态。** 代价是每步一次同步（需实测多少 ms）。

### ❌ 判定结果：**竞态（同步）假设被否证，且该开关有害**（2026-09-20 实测）

| 配置 | A | 单流 tok/s | `hp`（ms/step） | 服务 |
|---|---:|---:|---:|---|
| 无 SYNC_BEFORE | 1.07 | **42.7** | 33.5 | 正常 |
| **`DSPARK_DRAFT_SYNC_BEFORE=1`** | **1.073（没变）** | **29.4（−31%）** | **41.4（+8 ms）** | **崩溃** |

崩溃签名（32 次断言）：
```
File ".../worker/model_runner_v1.py", line 960, in _pad_query_start_loc_for_fia
    assert num_reqs <= num_reqs_padded
AssertionError
```
⇒ **同步不解决问题，纯粹加成本，还会引崩**。该开关**不要启用**（默认已是 0）。
复核：`armZ`/`armAA`（有 `CAPTURE_DISPATCH`、无 `SYNC_BEFORE`）断言 = 0；只有带 `SYNC_BEFORE` 的这一次 = 32。

### ★★★ 新假设（当前最强）：**draft 在图中的 KV 写落到了错槽位，第 2 步起读到自己写的脏数据**

把各次实验的**序列位置推进**串起来（位置 = `common_attn_metadata.seq_lens[0]`）：

| 步 | graph 臂 | eager 臂 |
|---|---|---|
| 第 1 步（pos=37） | 37 → 39（接受 **1** 个） | 37 → 39（接受 **1** 个）← **两臂相同** |
| 第 2 步（pos=39） | 39 → 43（接受 3） | 39 → 45（接受 5） |
| 后续 | 逐渐退化到接受 ≈0 | 稳定保持 A ≈ 2.65 |

**关键：第 1 步两臂一样好，从第 2 步起 graph 才开始退化。**
而第 2 步正是 **draft 第一次读到自己上一步写入的 KV** 的时刻
（draft 每步把自己那 5 个位置的 KV 写进 cache：`dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)`）。

这解释了此前所有观察：
* 第 1 步永远正常（还没读自己的写）
* 运行间**不稳定**（脏数据取决于时序 / 脏槽位原有内容）
* 6 个 capture 侧修复全部无效（问题在 replay 的**写路径**）

**待验证（已交给单 chip 子代理）**：draft 写 KV → 下一步读回 → 比对是否自洽；并对照 eager。

**影子探针在 graph 臂内部做的"eager 重跑"复用同一批 buffer，因此它测不出"图把 buffer 写坏"**：

```
graph 臂：graph==eager 4/5，graph[0]=[223,5038,10849,271,5038] eager[0]=[...,531]
```

它与 F2（两臂 token 相同）合起来说明：**draft 前向的输出本身没问题，问题在它之后** ——
即"draft 图 replay 的动作"污染了**别的东西**，或"消费方读到的不是它写的那份"。

**下一步要做的判定实验（已实现，未跑完）**：**同一次起服内热切换 draft 用图 / 用 eager**
（`/tmp/v41_dspark_flags` 里写 `DRAFT_FORCE_EAGER=1`），这样同一进程、同一批请求、只变一个变量，
彻底排除跨起服的混杂（发放级抽签、前缀缓存、服务状态）。
脚本：`/tmp/hotswap_ab.sh <port>`（含服务身份断言 + 三臂切换：图 → eager → 图，第三臂验证可逆性）。

## 5. 相关工具与开关（本轮新增）

| 工具/开关 | 作用 |
|---|---|
| `tools/draft_arm_probe.sh <base_url> <run_dir> <tag>` | 8 条中位 A / tok-s / **逐位置接受率** / `[bneck] hp` |
| `DSPARK_CAPTURE_DISPATCH=1` | 让 capture 走与 replay 相同的 `cudagraph_dispatcher`（**修复 F9，建议长期保留**） |
| `DSPARK_CAPTURE_MAXSEQLEN=<n>` | 捕获期 `max_seqlen_kv` 用上界 |
| `DSPARK_CAPTURE_PAD_SLOTS=1` | 捕获期 slot_mapping 置 -1（不写 KV） |
| `DSPARK_NO_TOPK_SHARE=1` | 不共享 `topk_indices_buffer`（draft 自算 topk） |
| `DSPARK_DRAFT_FORCE_EAGER`（**运行时** flag） | 热切换：绕过 ACLGraphWrapper 直接 eager |
| `DSPARK_DRAFT_NO_ATTN=1` | 传 `draft_attn_metadatas=None` 走无 attention fallback（判定实验） |
| `DSPARK_TOKEN_DUMP=1` | 打 draft 产出的 token（跨臂对比用） |
| `DSPARK_ROW_DUMP=1` | 打 step0 的前几行输入（capture 期自动跳过 D2H） |
| `DSPARK_GRAPH_PTR_PROBE=1` | capture/replay 的**地址 + 值**快照对比 |
| `/tmp/mysvc.sh <port>` | 断言该端口是**我们自己的** deepseek-v41（不是别人的 sglang） |

## 6. 交付口径

**保持 `DRAFT_GRAPH=0`**（draft 永远 eager）。`DRAFT_GRAPH=1` 需要显式打开，
且 `serve_a2.sh` 的 DRAFT-GUARD 会拒绝"stock 文件 + DRAFT_GRAPH=1"这种**静默失效**组合。
**发布不受本问题影响。**
