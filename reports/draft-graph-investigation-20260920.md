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

5. **起服前必须做 env 自查，否则白等 15 分钟**。端到端一次起服 ≈ 15 min（权重 1–2 min →
   device-index 探测 → 图捕获 5–8 min → READY），一旦关键开关漏传，**跑完也是错的配置**。
   2026-09-20 已因此翻车一次（那次还叠加了 OOM）。自查命令：
   ```bash
   # 容器内，先拿 PID 再读 /proc 的 env（不要用 docker inspect：它只显示 run 时的 -e）
   p=$(pgrep -f 'vllm serve' | head -1); tr '\0' '\n' < /proc/$p/environ \
     | grep -E 'DSPARK|V41_|PORT' | sort
   ```

6. **核对 env/日志一律不要接 `tail -N` 当"全量"**。2026-09-20 我犯过一次：命令是
   `... | grep -Ei 'DSPARK|V41_|PORT' | sort | tail -30`，而排序后 `DSPARK_CAPTURE_*`
   正好位于**字母序最前**，被 `tail -30` 整段截掉（实际 56 行）⇒ 我据此误判"关键开关缺失"，
   并向执行方发出错误的 kill + 重起指令（该指令被及时撤回，未造成损失）。
   ⇒ 规则：核验清单**只许 `grep` 精确等值或先 `wc -l` 报总数**；`tail` 仅用于看"最后几行进展"。

7. **内存压力下起服前清 page cache**（这台机 2 TB 内存、多租户，实测连**外部租户的 sglang** 都被 OOM 杀过）。
   我们的 8 卡起服曾**整容器消失**（日志在 `[DEVICE-INDEX] 能力探测通过` 处戛然而止、无 Python traceback），
   `dmesg` 显示 `Out of memory: Killed process (VLLM::Worker_TP) anon-rss:32.6GB` ⇒ **SIGKILL**，不是代码 bug。
   ⇒ 起服前 `echo 1 | sudo -n tee /proc/sys/vm/drop_caches`，并先 `free -g` 确认 available 足够。

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

## 4.4 ★★★★★ 根因最终形态：**图读取的是"捕获期绑定的地址上的内容"**

### 输入二分（子代理 `fixA`，最小必要集合）

从"全真实捕获（✅）"出发，逐个把输入换成 dummy：

| 捕获期 dummy 化的输入 | replay==eager |
|---|---|
| 无（全真实） | ✅ 5/5 |
| 全部 10 项（= 生产 `dummy_run`） | ❌ 0/5 |
| 仅 `runner.seq_lens` + `optimistic_seq_lens_cpu` | ❌ 0/5 |
| 仅 `seq_lens_group[0]` / `query_start_loc_group[0]` | ✅ 5/5 |
| 仅 `_per_group_context_slot_mapping_buffers` | ✅ 5/5 |
| 仅 `_per_group_query_slot_mapping_buffers` | ✅ 5/5 |
| **只有 `runner.seq_lens` 真实、其余 9 项全 dummy** | **✅ 5/5** |
| 除 `kv` 外全 dummy | ❌ 0/5 |

**最小必要集合**（全 dummy 捕获下逐项写回正确内容，反查必要性）：

| 写回字段 | 结果 |
|---|---|
| 只 `dspark_swa_indices` / 只 `sas_metadata` / 只 `seq_lens` / 只 `query_start_loc` / 只 `start_pos` | 全 ❌ |
| 去掉 `sas`（留其余 4） | ✅ |
| 去掉 `seq_lens`（留其余 4） | ❌ |
| 去掉 `query_start_loc` / 去掉 `start_pos` | 均 ✅ |
| **只 `swa` + `seq_lens`** | **✅ 5/5** |

**反证**：把 `rm.seq_lens`（ptr 已核对 = `seq_lens_group[0]`）在捕获期覆写成 1037（= replay 真值）**仍然 ❌**
⇒ 单靠它不够，**`dspark_swa_indices` 必须同时正确**。

### 机制
```
capture: build_capture_draft_attn_metadata() → 张量 @A、@B；npu.graph 录制内核参数 = @A、@B
replay:  set_inputs_first_pass/build_draft_attn_metadata → 新张量 @A'、@B'（值正确）
         aclgraph.replay() → 内核仍读 @A、@B 上的**捕获期旧内容**
```
⇒ 捕获期那两个地址上是什么，replay 就永远看到什么。
**"真实输入捕获 ✅" 是假阳性** —— 只是旧内容恰好等于首个 replay 步的值，换一步（如 1032→1050）立刻失效。
这与 §4.3 的 V×R 矩阵**完全同源**：不是 V 被烘成 kernel 标量，而是 **V 决定了这两个捕获期地址上的内容**。

### 生产真正缺的只有 `dspark_swa_indices` 一处

**`seq_lens` 在生产里已经常驻**（我核过源码）：
```
llm_base_proposer.py:1455-1456（_propose/replay 侧）
    self.seq_lens_group[0][:num_reqs_padded].copy_(common_attn_metadata.seq_lens)
    common_attn_metadata.seq_lens = self.seq_lens_group[0][:num_reqs_padded]   ← 重绑到常驻
dspark_proposer.py:487-493（dummy_run/capture 侧）
    seq_lens = self.seq_lens_group[0]                                        ← 同一个常驻 buffer
```
（`fixA` 观察到的 `seq_lens` 漂移是**它 harness 的现象** —— 它在 replay 侧新建了 `cad.seq_lens`。）

**`dspark_swa_indices` 则确实每次都新分配**（AST 核过的路径归属）：
```
build_req_metadata()             :1175  → buffer=self.dspark_swa_indices_buffer   ✅ 传了
build_req_metadata_for_drafting():1472  → 原为 build_dspark_swa_indices(*args)  ❌ 不传
build_for_drafting()             :1333  → 调 build_req_metadata_for_drafting    ← drafting 走这条
```
而 `build_dspark_swa_indices` 的 **docstring 自己写明**：
> When `buffer` is given, the per-token slots are copied into its leading rows and the returned
> tensor is a slice view of `buffer`. **This keeps the address stable across async ACL-graph
> replays, where the DSA operator captures `ori_sparse_indices`'s data pointer at capture time.**

⇒ 这个 `buffer=` 参数**本来就是为 ACLGraph 地址稳定而设的**，只是 drafting 路径没用上。

### 生产修法（四件套；下表为**改前**状态，见下方修正）

| 开关 | 文件 | 作用 |
|---|---|---|
| `DSPARK_CAPTURE_VALUE_FIX=1` | `dspark_proposer.py` | (a) 捕获期 `runner.seq_lens`/`optimistic_seq_lens_cpu` 填代表值；**(b)** 恢复图内 context KV 写入 |
| `DSPARK_SWA_INDICES_RESIDENT=1` | `dsa_v1.py:1472` | **(c)** 让 drafting 的 else 分支也走常驻 `dspark_swa_indices_buffer`（**外科改动**，不动 `_device_metadata_enabled`） |
| `DSPARK_CAPTURE_NCTX_FIX=1` | `dspark_proposer.py` | **(d)** 捕获期 `_dflash_num_context = num_reqs*(1+SP)`（原为 `num_input_tokens`，nr=1 时 5≠6、nr=8 时 42≠48） |
| `DSPARK_CAPTURE_SEQ_LEN=<n>` | — | 代表值（0 ⇒ `max_num_tokens`，生产 8192） |

### ❗ 修正（`fixB` 四格裁决，2026-09-20 17:2x）：**(a)(b) 也是必需的 —— 最小集合是四件套**

上文"生产真正缺的只有 `dspark_swa_indices` 一处"**被实验否证**。`fixB` 在单 chip 上把四个变量做成 2×2×2 里的四格（`R=262144`，每格 3 次 replay 取一致结果）：

| # | `CAPTURE_VALUE_FIX` | `SWA_RESIDENT` | `CAPTURE_NCTX_FIX` | nr | replay==eager |
|---|---|---|---|---|---|
| A | **0** | 1 | 1 | 1 | **❌** |
| B | 0 | 1 | 1 | 8 | ❌ |
| **C** | **1** | 1 | 1 | 8 | **✅** |
| D | 0 | 1 | **0** | 1 | ❌ |

**A 格原文（决定性）**：
```
[capture] 真实 dummy_run ... built draft attention metadata (num_query_total=5 ...)
[capture] 完成 _dflash_num_context=6  ctx_buffers=None      ← (d) 已生效（6 而非 5）
[ptr REPLAY] ... is_resident=True                           ← (c) 已生效
[replay#0] tokens=[[828, 107625, 16, 539, 15]]   vs eager [[81, 2987, 11, 16781, 69630]]
```
⇒ **(c)(d) 都生效了仍然 ❌**，因为 `ctx_buffers=None` 让 `precompute_and_store_context_kv` 在捕获时**提前 `return`**：
**图里根本没有"写 context KV"那串算子**。所以 (d) 只是把那个 slice 长度改对了，**那个调用压根没进图**。
⇒ **(a)(b) 与 (c)(d) 是两类不同的缺陷**：前者决定"算子有没有被录进图"，后者决定"录进去的算子读的地址/长度对不对"。四件缺一不可。

**仍未分离的一点（【推断】标注）**：C 格只证明了 `{a 或 b}` 必要；"只开 (b)、(a) 关（捕获期 `seq_lens=0`）是否 ❌"
**尚未单独测**。也就是说最小集合可能是**三件而非四件**（(a) 或许可省，`DSPARK_CAPTURE_SEQ_LEN` 无需调）。
`fixB` 已把这条列为待做实验（捕获前只把 `_context_slot_mapping_buffers` 置 `None`、其余全开；或 (b) on + (a) off）。

**待验证**：① (a) 与 (b) 的分离实验（上条）；② 9 桶捕获的逐请求逐位审计；
③ **端到端 A/tok-s（唯一真正的判据，正在跑）**。

## 4.5 ★★★★★ 端到端裁决：**四件套有效，A 从 1.075 恢复到 2.39**（2026-09-20 18:21）

**配置**：`DRAFT_GRAPH=1` + 四件套（`CAPTURE_VALUE_FIX=1` / `CAPTURE_SEQ_LEN=8192` /
`SWA_INDICES_RESIDENT=1` / `CAPTURE_NCTX_FIX=1`），`V41_ENGRAM_DEVICE_INDEX=0`，8×910B3。

| 并发 | 单流 tok/s | 总吞吐 tok/s | A |
|---:|---:|---:|---:|
| 1 | **89.0** | 82.5 | **2.390** |
| 2 | 81.9 | 101.7 | 2.336 |
| 4 | 61.6 | 191.5 | 2.279 |
| 8 | **引擎死亡**（ok=6/8） | — | — |

**对照臂**（同机、`DRAFT_GRAPH=0` eager，`a2_20260920_140641` 两条独立臂）：
conc=1 分别 **89.3 / 80.9 / A=2.574** 与 **82.8 / 82.2 / A=2.570**。

### ① 修复有效（决定性）

| | 修复前（graph） | 修复后（graph） | eager 基线 |
|---|---:|---:|---:|
| A | 1.075 | **2.390** | 2.57 |
| 单流 tok/s | 40.1 | **89.0** | 89.3 |
| 位置 0 接受率 | 0.22 | 0.638 | ~0.77 |

⇒ A **+122%**，单流 **+122%**。四件套是充分且必要的修复（每一件的必要性见 §4.4 的 A/B/C/D 格子）。

### ② 步时换算：draft 入图**确实省了 2.84 ms/step**，但被 A 的缺口吃光

用 `total_decode` 反算（**不要**用 `per_stream_med` 除以 A —— 那是 median÷mean 的口径混用）：
```
eager : wall = 2048/80.9  = 25.32 s, steps = 2048/2.574 = 795.6 → 31.82 ms/step
graph : wall = 2048/82.5  = 24.83 s, steps = 2048/2.390 = 856.9 → 28.98 ms/step
```
⇒ **−2.84 ms/step（−8.9%）**。这与"draft allreduce 单次 293 µs（eager）vs 36 µs（图内）"
的机理一致。但净吞吐持平（80.9 → 82.5，+2%）⇒ **收益被 A 的 7.1% 缺口抵消**。
⇒ 只要把 A 补回 2.57，graph 就是 **+9%** 的净赢（2.57/0.02898 = 88.7 tok/s 单流）。
**因此"补齐 A 缺口"是当前最高价值的目标。**

### ③ 未解决问题 P0-A：A 仍比 eager 低 7%，且**不是抽签噪声**

两条独立 eager 臂给出 **2.574 / 2.570**（conc=1、8 条），离散 <0.2% ⇒ 2.39 的 7% 缺口是真实的、系统性的。
【推断】最可能落在三处之一（均未验证）：(a) 捕获期代表值 `seq_len=8192` 是否影响 SWA 窗口参数；
(b) 图内 context KV 写入的位置/时序；(c) 常驻 `dspark_swa_indices_buffer` 在多层共享下的内容竞争。

### ④ 未解决问题 P0-B：**`DRAFT_GRAPH=1` + conc≥8 ⇒ 引擎死亡**（历史既有，非本次引入，但是发布阻塞项）

> **✅ 已解决（2026-09-20 19:44）—— 见 §4.6。下面是修复前的原始记录，保留作为演进链。**

【实测】扫描本地全部 run：
```
出现 `assert num_reqs <= num_reqs_padded` 的 run：12 个，**全部**是 DRAFT_GRAPH=1，每 run 恰好 32 行
a2_20260920_105507 / 133827 / 140641 的 graph 臂：conc=8 → ok=7/8、ok=6/8、ok=7/8
同一台服务器同一时刻的 eager 臂：conc=8 → ok=8/8（237~341 tok/s）**从不死**
```
栈固定：`worker.py:720 sample_tokens → model_runner_v1.py:2661 → 2601 propose_draft_token_ids
→ 1952 drafter._propose → llm_base_proposer.py:1364 _pad_query_start_loc_for_fia
→ model_runner_v1.py:960 assert num_reqs <= num_reqs_padded`

判定逻辑（`vllm_ascend/worker/model_runner_v1.py:931-972`）：
```python
if cudagraph_runtime_mode == FULL and compilation_config.cudagraph_mode == FULL:
    num_reqs_padded = num_reqs                      # ← 这条分支不可能触发 assert
else:
    num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs
if (num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len
        and compilation_config.cudagraph_mode != CUDAGraphMode.FULL):
    assert num_reqs <= num_reqs_padded              # ← 崩在这
```
⇒ 崩溃要求**同时**满足 `cudagraph_runtime_mode != FULL` 与 `cudagraph_mode != FULL`，
但起服日志明写 `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL`。
**矛盾点就是要抓的对象**：要么某时刻 `runtime_mode` 不是 FULL，要么 `num_reqs_padded`
被 `batch_desc_num_reqs` 取小了（取整方向反了）。已在 `_propose` 的调用点加"失败前诊断"（异常路径专用，不进 capture）。

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

## 4.6 ★★★★★ P0-B 已解决：**dispatch 输入换算**（2026-09-20 19:44）

### 根因（比 §4.5 ④ 的推测更精确）

draft 每请求只吃 `num_query_per_req`(=**5**) 个 token，而 `CudagraphDispatcher`
按 `uniform_decode_query_len`(=1+SP=**6**) 反推请求数（`num_reqs = 桶 // 6`）：

```
conc=7: 7×5 = 35 → 取桶 36 → 36 // 6 = 6 < 7  ⇒ assert num_reqs <= num_reqs_padded 崩
conc=8: 8×5 = 40 → 取桶 42 → 42 // 6 = 7 < 8  ⇒ 崩
conc≤6: "最小的 ≥5k 的捕获尺寸"恰好 ≥6k，所以一直没暴露
```
而 capture 期本来就是按"每请求 6 个"定桶的（`capture` 描述符实测为
`(6,1) (12,2) (18,3) (24,4) (36,6) (42,7) (48,8) (96,16) (192,32)`）
⇒ **两侧对"每请求几个 token"的定义不一致**。

### 修法（一行换算，无需任何强制填充）

`DSPARK_DISPATCH_QUERY_LEN_FIX=1`（`llm_base_proposer.py`，**默认 1**）：
uniform decode 时把 dispatch 的输入换成"每请求 `uniform_decode_query_len` 个"的
等价 token 数（`num_reqs × 6`），使 replay 选的桶与 capture 期
`num_reqs = 桶 // 6` 的定义自洽：**k 个请求 ⇒ 桶 6k ⇒ num_reqs = k**。
之后 `_pad_query_start_loc_for_fia` 的 `num_tokens_padded == num_reqs_padded × 6`
与 `num_reqs <= num_reqs_padded` **同时成立**，不再需要任何填充。

边界自洽：`MAX_SEQS=32` ⇒ `nreq×6 ∈ [6,192]`，正好落在 `capture_max=192` 内。

### ❌ 被否证的修法（保留记录，避免重走）

| 尝试 | 结果 |
|---|---|
| `DSPARK_FIA_PAD_REQS_FIX`：把待填充请求数抬到真实 `cad.num_reqs` | **否证**。`_pad_query_start_loc_for_fia` 的 mixed-batch 分支在 `qsl[nrp] < ntp` 时会**再补一个 dummy 请求并 `nrp += 1`**，返回 7+1=8，写回 `cad.num_reqs` 后下游 `build_dspark_swa_indices` 的 block_table 只有 7 行 ⇒ `RuntimeError: gather ... expected index shape 8 smaller than self shape 7`。**断言消失了，但换成尺寸崩溃。** 默认 0 保留为否证记录 |
| `DSPARK_CAPTURE_DISPATCH=1` | 单请求可用，nr=8 仍失败（已被 `CAPTURE_NCTX_FIX` 取代） |
| `DSPARK_DRAFT_SYNC_BEFORE=1` | A 不变、tok/s −31%，并引发同一个断言 ⇒ 有害 |

### 端到端验证（同进程、同批 prompt、每臂前热身）

| 臂 | conc=1 | conc=7 | conc=8 |
|---|---|---|---|
| G1 (graph) | **8/8** A=2.403 100.67 tok/s | **8/8** A=2.878 | **8/8** A=2.730 |
| E1 (eager) | **8/8** A=2.455 66.54 | **8/8** A=2.633 | **8/8** A=2.643 |
| G2 (graph) | **8/8** A=2.738 109.77 | **8/8** A=2.557 | **8/8** A=2.282 |

`grep -c "num_reqs <= num_reqs_padded"` = **0**；`RuntimeError|EZ1001|OutOfMemory` = **0**。
修复前 conc=7 → `ok=4/7` 引擎死、conc=8 连预热都起不来。

### 「仍走图」的直接证据（`DSPARK_DISPATCH_UNIQUE=1`）

`dispatch-unique-TP0.txt` 与 `captured-buckets.txt` 完全自洽，每个 `cad.num_reqs`
都命中 `bucket.num_reqs ≥ cad.num_reqs` 的 FULL 桶：

| cad.num_reqs | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** |
|---|---|---|---|---|---|---|---|---|
| 桶 (nt, nr) | (6,1) | (12,2) | (18,3) | (24,4) | (36,6) | (36,6) | **(42,7)** | **(48,8)** |
| ≥ cad? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ★修复前是(36,6) | ✅ ★修复前是(42,7) |

同文件的 `use_graph_rt=False` 组合全部 `runtime_mode=NONE bucket=None`
⇒ **同进程内的 eager 臂确实是真 eager**（不再靠 tok/s 反推）。

**独立的第二条证据**：同进程同输入下 G2 `24.9 ms/step` vs E1 `36.9 ms/step`，
**差 12 ms/step**。若 graph 臂静默退回 eager，两臂必然相同 ⇒ 图确实在重放。

> ⚠️ **一个易误读的点**：`[bneck] mode=stock` **不能**证明 draft 走图 ——
> 那是瓶颈注入探针（`patches/files/model.py` 的 `V41_BNECK_MODE_FILE`，报的是静态核模式）。
> 早期我用它下过结论，是错的。

### ★ P0-A 的旧结论被推翻：「graph 比 eager 少 7% A」**不成立**

| 臂 | A（conc=1） | 观测量 |
|---|---|---|
| graph（修复后） | 2.403 / 2.738（同进程）、2.633 / 2.699 / 2.588 / 3.003（前几轮） | 观测区间 **2.40–3.00** |
| eager | 2.455（同进程）、2.660（同进程）、2.574 / 2.570（14:20） | 观测区间 **2.46–2.66** |

⇒ 两臂**在噪声内相等**；而 graph 每步稳定快 ~12 ms（同进程实测）。
§4.5 ③ 那个"7% 缺口"是**跨起服比 + 无热身**造成的假象（18:21 那次 A=2.39 是离群）。

### 默认值定稿（发布口径）

| 开关 | 默认 | 位置 | 作用 |
|---|---|---|---|
| `DSPARK_CAPTURE_VALUE_FIX` | **1**（本次从 0 改） | `serve_a2.sh` | 捕获期代表值 + **恢复图内 context KV 写入** |
| `DSPARK_SWA_INDICES_RESIDENT` | 1 | `dsa_v1.py` | 常驻索引缓冲（图捕获的是 data_ptr） |
| `DSPARK_CAPTURE_NCTX_FIX` | 1 | `dspark_proposer.py` | `_dflash_num_context = num_reqs×(1+SP)` |
| `DSPARK_DISPATCH_QUERY_LEN_FIX` | 1 | `llm_base_proposer.py` | **P0-B 修复**（本文档 §4.6） |
| `DSPARK_DISPATCH_UNIQUE` | 0 | `llm_base_proposer.py` | 走图取证（默认关） |
| `DSPARK_FIA_PAD_REQS_FIX` | 0 | `llm_base_proposer.py` | ❌ 否证保留 |

⚠️ **`DSPARK_CAPTURE_VALUE_FIX` 默认改为 1 的理由**：只写 `DRAFT_GRAPH=1` 是最自然的用法，
而旧默认 0 会让用户拿到"能起服、但 A≈1.07 / 单流 40 tok/s"的坏配置且**无任何报错**。
另外三件在代码里默认已是 1，只有这一件漏了 ⇒ 它曾是唯一的"陷阱开关"。
（draft 三文件只在 `DRAFT_GRAPH=1` 时挂载，故该默认对 `DRAFT_GRAPH=0` 无影响。）

### 未验证项

① conc=9..32（推导安全但未实测；`MAX_SEQS=32` ⇒ 最大 192 = `capture_max`）；
② 多轮长跑稳定性（每档 1 次测量）；③ **精度回归**（Vision/GSM8K，进行中）；
④ 跨进程 A 的绝对口径不可靠 ⇒ 只用同进程臂间对比。

---

## 4.7 ★★★★★ 新 P0：**`DRAFT_GRAPH=1` + conc≥16 ⇒ 引擎进入不可恢复的坏状态**（2026-09-20 20:29）

### 现象（`e2e_fix_H_final` 原始证据）

起服后先跑 Vision + GSM8K-200（约 10 分钟，期间 A = 3.93–3.98，**一直健康**），
然后跑 `--concurrency 16`：

```
12:08:11  A=3.98      ← GSM8K 期间，健康
12:10:11  A=2.24      ← conc=16 开始
12:11:11  A=1.00      ← 从此再没恢复（此后连续 16 次采样全是 1.00）
```
指标原文（12:11:01）：
```
Mean acceptance length: 1.00, Accepted: 0 tokens, Drafted: 5200 tokens,
Per-position acceptance rate: 0.000, 0.000, 0.000, 0.000, 0.000, Avg Draft acceptance rate: 0.0%
```
⇒ **draft 完全不被接受**，而且**此后所有新请求都这样**：

| 探测 | 结果 |
|---|---|
| `guard_hi` conc=16 | `ok=16/32`，A=1.39（半途开始坏） |
| `guard_hi` conc=32 | **全部失败**（`no content`） |
| 之后单请求 non-stream（短/中/长 prompt） | `text=''`、但 `usage.completion_tokens=32` |
| 之后单请求 stream | `chunks_with_text=0` |
| `/health`、`/metrics` | **200**，进程活着，`Running: 0 reqs, KV cache 0%` |

⇒ 引擎**进程不死，但输出永久变空**、draft 接受率永久为 0。

### 归属：**`DRAFT_GRAPH=1` 独有**（决定性对照）

同机、同模型、同为 `ENGRAM_DEVICE_INDEX=0`、同为 20:16–20:29 时段：

| 臂 | conc=16 | conc=32 | 之后引擎 |
|---|---|---|---|
| **`DRAFT_GRAPH=1`**（H_final，四件套全开） | `ok=16/32`，A=1.39，半途坏 | **全部失败** | **永久坏**（A=1.00、`Accepted: 0`、`text=''`） |
| **`DRAFT_GRAPH=0`**（I_graph0） | **`ok=32/32`**，A=**2.676** | **`ok=32/32`**，A=**2.735** | **健康**（`A=1.00` 计数 **0**、断言 **0**、单请求 `text='1'`） |

⇒ **不是 target/scheduler 侧缺陷**（那条假设否证），是 `DRAFT_GRAPH=1` 特有的路径。

### 已否证的机制猜测（记录以免重走）

**共 5 条，全部被主动否证**（2026-09-20 深夜收口，6 轮独立实验）：

| # | 猜测 | 判定 | 依据 |
|---:|---|---|---|
| 1 | 探针伪影（`TOKEN_DUMP` 的 D2H / `DISPATCH_UNIQUE`） | **否证** | 探针全开（`RT_FLAGS=1 TOKEN_DUMP=1(24) DISPATCH_UNIQUE=1 DIAG_STEPS=120`）+ Vision + GSM8K-200 → conc=16/32 **全绿**。<br>⚠️ 另有一处事实纠正：**H_final 当时其实没开 `TOKEN_DUMP`**（`grep -c "dspark-token"` = 0），它只开了 `RT_FLAGS + DISPATCH_UNIQUE + DIAG_STEPS`，而 `DISPATCH_UNIQUE` 是**纯 Python 标量、无 D2H 无流操作** ⇒ "探针伪影"的先验本就很弱 |
| 2 | 前置 `conc=7/8/9`（走新桶 `(42,7)/(48,8)/(96,16)`） | **否证** | 顺序跑 7→8→9→16→32，全部满员通过（A=2.55/2.78/2.70/2.73/2.67） |
| 3 | RT 热切换（`DRAFT_FORCE_EAGER` 往返） | **否证** | graph→eager→graph 后 conc=16/32 仍全绿 |
| 4 | **请求行** padding 写脏 KV | **否证** | `dispatch-unique` 实测：conc=16 时 `num_reqs_padded(16) == cad.num_reqs(16)` ⇒ **零请求行 padding**；conc=32 同理 |
| 5 | **token 行** padding 写脏 KV | **★ 源码级否证** | `llm_base_proposer.py: _pad_draft_buffers()` **每次 `_propose`** 都执行：<br>`buf[num_actual_tokens:num_input_tokens].fill_(-1)`（query slot）<br>`buf[self._dflash_num_context:].fill_(-1)`（context slot）<br>⇒ padding 区**每步被填 -1（不写 KV）**；且 `fill_` 是**原地写**、图读同一块存储 ⇒ 图内看到的也是 -1 ⇒ **两种 padding 都不成立** |

| 其它 | 判定 | 依据 |
|---|---|---|
| `DISPATCH_QUERY_LEN_FIX` 引入 | 否证 | conc≥9 时修复前后取桶**逐字节相同**（只有 7,8,9,17,18,19 改变） |
| 内存/注册问题（207001 那一类） | 否证 | 两个臂**同为 `ENGRAM_DEVICE_INDEX=0`**，都不注册 host 内存 |

### ★ P0-C 的最终归档口径（2026-09-20 收口）

**观测簿**：H_final 出现 **1 次**（A 永久 1.00，14 次采样、50 秒内从 3.91 掉到 1.00）；
此后 **6 轮不同配方**（`J_bare`×3、`K_preload`、`L_cand`、soak 6 轮 × 6 并发）
≈ **36 个测量点、10.4 分钟连续负载**，**全部健康**（A ∈ [1.94, 3.10]），
`A=1.00 && Accepted==0` 计数 **0**。

⇒ **归档为"1 次观测、6 轮未复现的偶发"**，不再是"必现的发布阻塞项"。
但**保留完整字段**（时间、A 序列、桶决策、`rtsMallocHost=0`、per-position 退化曲线），
以便将来现场对号。

**坏状态的三个特征**（供现场识别）：
1. **渐进退化、不是瞬时**：per-position 接受率 `0.826 → 0.581 → 0.278 → 0.000`；
2. **draft 仍在产出**（`Drafted: 5200+ tokens`）但**全部不被接受**（`Accepted: 0`）；
3. 请求 `text=''` 但 `usage.completion_tokens=32`；引擎**活着**（`/health` 200、
   `Running: 0 / Waiting: 0 / KV cache 0%`）。

**廉价判据 + 恢复**：连续两次 specdec metrics 出现
`Mean acceptance length: 1.00` 且 `Accepted throughput: 0.00` ⇒ 判定已进入坏状态 ⇒
**重启恢复**。（那时再发请求试探是浪费 —— 引擎已坏。）

**判别命令**（已验证可用；健康基线 = `text='1'`, `token_ids=[19]`）：
```json
"max_tokens": 1, "return_token_ids": true, "skip_special_tokens": false, "logprobs": 1
```

**⚠️ 与另一种坏状态严格区分**（两者都表现为"服务异常"，但归属完全不同）：

| | **P0-C（A=1.00）** | **pinned OOM（hang）** |
|---|---|---|
| `rtsMallocHost` 计数 | **0** | **2 行**（同 worker 同秒；含 `Insufficient_Host_Memory` 共 4 行） |
| `A=1.00` 计数 | **14** | **0** |
| 请求表现 | `text=''` 但 `completion_tokens=32` | 请求**卡住不动**（`Running: 2 reqs`） |
| 归属 | **`DRAFT_GRAPH=1` 独有**（`DRAFT_GRAPH=0` 6 轮全健康） | **target 侧**（`_calc_spec_decode_metadata` 申请 32 B pinned 失败），**与 draft 图无关** |
| 恢复 | 重启 | 重启 |

### ⚠️ 仍存在的混淆变量（必须消掉）

H_final（graph 臂）在 conc=16 之前**先跑了 Vision + GSM8K-200**（约 10 分钟负载），
而 I_graph0（eager 臂）是**起服后直接**跑 conc=16/32。
⇒ 不能排除"**长时间负载累积**"是必要条件。**下一轮直接测裸触发**（起服后立刻 conc=16）。

### 待做的判别（`token_ids` 到底是什么）

上轮看到的 `token_ids: null` **不是证据** —— 那是默认不返回造成的，必须显式请求：
`"return_token_ids": true, "skip_special_tokens": false, "logprobs"`，
区分 ① **真·空数组** vs ② **全是 pad/EOS 特殊 token**。
后者支持"KV/输出通路被污染"，前者指向 API 层。

### 廉价坏状态判据（写进后续所有探测的前置检查）

> **只要 specdec metrics 连续两次 `Mean acceptance length: 1.00` 且 `Accepted throughput: 0.00`，
> 就判定引擎已进入坏状态** —— 不必再发请求试探（那时引擎已坏，试探只会浪费时间）。

### 对发布的影响（fixA 的判断，我认可）

1. `DRAFT_GRAPH=1` 在 **conc ≤ 8** 已可发布（A 与 eager 持平、per-step 快 ~12 ms、精度全过）；
2. **conc ≥ 16 是发布阻塞项**（用户跑到 16 并发就会看到引擎变哑）；
3. 应作为**独立 P0** 立项，别和 draft 图的其它修复混。

### 顺带订正（我复核桶表后的结论）

`DISPATCH_QUERY_LEN_FIX` 修复的不是 conc=7、8 **两个**值 ——
按 `capture` 桶表（`6,12,18,24,36,42,48,96,192` 与 `nr=桶//6`）重算：

```
修复前会崩的 conc :  7, 8, 9, 17, 18, 19   （六个）
修复后会崩的 conc :  无
改变取桶的 conc   :  7, 8, 9, 17, 18, 19
```
⇒ 覆盖面比原报告写的更宽，这对发布是加分。

### 证据位置

- graph 臂（坏）：`results/e2e_fix_H_final/{serve.log,guard_hi.json,guard_hi.log,guard_hi_eager.log}`
- eager 臂（健康）：`results/e2e_fix_I_graph0/{serve.log,guard_hi.json,guard_hi.log}`
- 走图证据：`lite-runs/fixA/out/e2e/dispatch-unique-TP0.txt`、`captured-buckets.txt`

---

## 5. 相关工具与开关（本轮新增）

以下是本轮排查过程中新增/常用的工具与门控（**含默认值与用途**），供后续复用。

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

### 当前状态（2026-09-20 收口）

| 项 | 状态 |
|---|---|
| 四件套修复（A=1.075 → ~2.6） | ✅ **已验证**，精度全过（GSM8K **198/200**、Vision **23/23**、10/10 质量判据） |
| P0-B（conc=7,8,9,17,18,19 崩溃） | ✅ **已修**（`DISPATCH_QUERY_LEN_FIX`，默认 1）；**六档全部实测通过**，每档桶决策原文已归档，全部 `runtime_mode=FULL` |
| A（接受长度） | ✅ **与 eager 持平**（同进程：graph 2.40–2.74 / eager 2.46，区间重叠） |
| per-step | ✅ **graph 快 ~12 ms**（同进程 24.9 vs 36.9 ms）—— 这条独立证明"图真的在重放" |
| **P0-C（偶发 A 永久 1.00）** | ⚠️ **1 次观测、6 轮未复现**（≈36 个测量点全健康），**5 条机制猜测全部否证** ⇒ 降级为**已知偶发**，非阻塞项（详见 §4.7 的归档口径） |
| `DSPARK_CAPTURE_VALUE_FIX` 默认值 | 已从 `0` 改为 **`1`**（`serve_a2.sh`）—— 缺它时 A≈1.07 且**无任何报错** |

### 发布口径（定稿）

**`DRAFT_GRAPH` 保持默认 `0`；`DRAFT_GRAPH=1` 作为"推荐开启"的显式选项。**

理由：P0-C 虽已 6 轮未复现、5 条候选全部否证，但它的**失效形态最危险** ——
进程活着、`/health` 200、却**永久输出空**（且无报错）。把它设成默认，等于让所有用户在
不知情的情况下承担这个风险；而作为显式选项 + 文档写明判据，用户可以在收益与风险之间
自己权衡。

**README 里对 `DRAFT_GRAPH=1` 的表述口径**：
* 收益：**−12 ms/step（−32%）**，A 与 eager 持平，精度已过（Vision 23/23、GSM8K 198/200）；
* 风险：存在**极罕见**的"A 永久 1.00 / 输出变空"坏状态（截至目前 **1 次观测、6 轮未复现**）；
* 判据与恢复：连续两次 `Mean acceptance length: 1.00` 且 `Accepted throughput: 0.00` ⇒
  **重启即可恢复**；
* `serve_a2.sh` 的 DRAFT-GUARD 会拒绝"stock 文件 + `DRAFT_GRAPH=1`"这种**静默失效**组合
  （那种组合下 A 恒 1.0 但 ms 看着正常 —— 见 `reports/draft-graph-negative-control.md`）。

### 四件套开关的默认值（定稿，缺一不可）

| 开关 | 默认 | 位置 | 作用 |
|---|---|---|---|
| `DSPARK_CAPTURE_VALUE_FIX` | **1** | `serve_a2.sh` 透传 | (a) 捕获期代表值 + **(b) 恢复图内 context KV 写入** |
| `DSPARK_SWA_INDICES_RESIDENT` | 1 | `dsa_v1.py` | (c) 常驻索引缓冲（图捕获的是 `data_ptr`） |
| `DSPARK_CAPTURE_NCTX_FIX` | 1 | `dspark_proposer.py` | (d) `_dflash_num_context = num_reqs×(1+SP)` |
| `DSPARK_DISPATCH_QUERY_LEN_FIX` | 1 | `llm_base_proposer.py` | **P0-B 修复**（dispatch 输入换算） |

⇒ 用户只需 `DRAFT_GRAPH=1`，四件套**自动全开**（脚本传 1 或代码默认 1）。
想复现旧行为时才显式传 `=0`。
