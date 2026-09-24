# CED D 1M 偶发乱码：根因与修复方案（2026-09-24）

本文记录 2026-09-24 在 A3-21 真权重 8+8 上定位到的确定性根因，以及两个
候选修复。**尚未完成修复验证**；请在 1+1 tiny 或 8+8 上按本文第 4 节执行。

## 1. 现象与触发条件

固定同一 1M 请求（`prompt_tokens=1019847`），D 侧物理块分配形态是唯一变量：

| D 侧 g0 块形态 | 结果 |
|---|---|
| 连续（`descents=0`，如 first=1 last=7968） | 100% 正确，首 token `RB` logprob ≈ −0.0002 |
| 碎片/绕回（如 first=23968 last=6139 `descents=1851`） | 首 token 直接 `id=1`（EOS），logprob −2.10，`RB` 掉出 top-20 |

失败点与 KV 池相位严格吻合：1M（每请求 7989 块、池 30082）失败在第 3、7 次；
900K（每请求 6990 块）失败同样在第 3、7 次；144K（1127 块，8 次不绕回）8/8 全过；
22-token 短针 8/8 全过。

## 2. 已排除项（有原始证据，不要重复）

- **P→D 全局 KV 传输内容**：把快照点移进 replay 窗口（position 1019845）后，
  层 20 的 `long_kv`/`index_k`/`index_scale` 在「通过-失败」配对间 maxdiff
  = 0/0/0（逐位相同却结果不同）；含另一次通过的配对反而有差异
  （0.535/21.0/3.7e-4）。⇒ 传输是确定的，且与成败不相关。
- **Engram 4-gram 历史**：通过/失败请求的 `hash8` 逐位相同
  （`[6145177, 29200083, ...]`）。
- **metadata 独立流、DSA 辅助流、共享专家多流**：逐项消融都不消除失败。
- **D replay 位置**：每次都是 `positions=1019718..1019845`，无例外。
- **KV 传输完整性**：每次 8 rank 都有完整 transfer 日志。

## 3. 根因：replay 的第一个 query 会读到本请求未持有的 SWA 页

真实几何（`N=1019847`，`E=N-1=1019846`，`R=W=128`，`B=128`）：

```text
S = E - R                    = 1019718   # replay_start
full_visible_start = S - 127 = 1019591
full_view_page_start = floor(1019591/128) = 7965
full_view_page_end   = ceil (1019846/128) = 7968
⇒ 第一个 replay query 的 128 窗口需要逻辑页 7965, 7966, 7967（3 页）
```

但当前两端都只保留/分配 **2 页**：

- `experimental/ced/mooncake_hybrid_connector.py:1359`
  `num_swa_blocks = cdiv(sliding_window, block_size) + 1 = cdiv(128,128)+1 = 2`，
  该值只用于 P 侧 `get_sw_clipped_blocks`（唯一调用点 `:1567`）⇒ P 只传尾 2 页。
- D 侧 `[CED-BLOCKS]` 实测每个 SWA group 都是 `n=2`（`g1:(n=1) g2..g11:(n=2)`），
  即 D 也只持有 2 页；D 的块表行长度为 8192（`bt_shape=(4,8192)` 实测），
  有效项在行尾，逻辑页 7965 的位置是 **0**。
- `experimental/ced/dsa_v41.py:657` 把**完整** `metadata.swa.block_table[:num_reqs]`
  和完整 `seq_lens` 传给 `npu_sparse_flash_mla`（`:682`），`ori_mask_mode=4`、
  `ori_win_left=127`（`:697`），可见窗口没有裁到 `replay_start`。

⇒ 第一个 replay query 的窗口左沿回溯到逻辑页 7965，而该页在本请求的分配里是
0（无效）。连续分配时相邻物理页往往残留上一次同 prompt replay 写入的**正确**
数据（我们反复发同一请求），碎片分配时读到的是别的请求的残留 ⇒ 首 token EOS。

该结论与仓库既有离线原型一致：见
[`../evidence/ced_swa_window_view_proto_20260924/README.md`](../evidence/ced_swa_window_view_proto_20260924/README.md)
（"保留 255-token 跨度并传 3 页可静态覆盖"、"manager 只留8191；connector tail2
含空8190，缺8190"）。

### 逐层数值旁证（同一请求 4 次）

- 层 0 的 attention 输出在「通过」与「失败」之间**打印精度内完全一致**。
- 层 1 的**输入**已经分叉（in_sum 3.4038 vs 3.3953）。
- 层 2 的 SMLA 选中集合直接改变（`sel_sum` 98216 vs 91219）⇒ 离散 top-k 是放大器。
- 尾位置 SWA 快照：层 38/39 的「通过-失败」差 ≈1.53~1.83，是「通过-通过」差
  （0.22~0.37）的约 5 倍。
- 噪声底标定：**三次通过的请求之间 digest 也不同**（pass-pass max|Δattn_sum| ≈ 6~8，
  pass-fail ≈ 114~115）。⇒ 该路径本身对布局不是逐位可复现的；失败是偏差放大一个
  量级后触发 select 翻转，而不是"多了一点噪声"。

## 4. 修复方案

### 方案 A（推荐，改动自包含）：把 D 侧 attention 可见窗口裁到 replay 起点

只在 `experimental/ced/dsa_v41.py::_native_attention` 里改，不动 core。

在 CED decode 角色、且处于 replay 步（`metadata.swa.max_query_len > 1`）时：

```text
start_page  = positions[0] // block_size          # 1019718//128 = 7966
seqused_ori = seq_len - start_page * block_size   # 1019846-1019648 = 198
ori_block_table = metadata.swa.block_table[:num_reqs, start_page:start_page+ceil(seqused_ori/block_size)]
```

注意 `ori_block_table` 必须**窄化到从 start_page 开始**，这样 kernel 的行内下标
0 对应 kv 位置 `start_page*block_size`；配合 `ori_win_left=127`，query 的窗口
自然被 clamp 到 0，**永不读 start_page 以前的页**。

- replay 步：`seqused_ori=198`，需要 2 页（7966、7967）⇒ 正好是本请求持有的集合；
  第一个 query 的窗口由 128 缩短为 71（论文 bounded replay 接受的近似）。
- 末 token 步（单 query，position 1019846）：`start_page=7967`，`seqused_ori=71`，
  1 页；窗口 71 个全部真实 ⇒ **末 token 仍看到完整 71+ 真实历史**。

实现注意：
- `positions[0]` 是 device tensor，取 `start_page` 需要一次 D2H（`.item()`）。
  replay 步是 prefill 形状、**不走 FULL decode 图**（`FULL_DECODE_ONLY`），且
  `_maybe_snapshot_cache` 已在同类位置做 `.cpu().tolist()`；因此必须用
  `if not getattr(forward_context, "capturing", False):` 守卫，capture 期间走原路径。
- 末 token 那一步如果是图内执行，则 `start_page` 需用图外预计算的值（可从
  scheduler 已算好的 `request.ced_replay_start`/`ced_replay_end` 透传进
  metadata），或对该步沿用现有全量 block table（此时窗口只覆盖 71 个真实 kv，
  反而不会越界——需实测确认）。

### 方案 B（更精确，但要动 core）：让两端都保留 3 页

- P 侧：`num_swa_blocks` 的 `n_tokens` 从 `sliding_window` 改为
  `sliding_window - 1 + replay_tokens`（CED 角色下），即
  `cdiv(255,128)+1 = 3`。
- D 侧：需要 core 的 recycling-aware 块保留上限同步放宽（当前 D 只分配 2 页）。
  `vllm/v1/core/single_type_kv_cache_manager.py` 里有
  `_contiguous_blocks_for_hit` / recycling-aware admission cap 相关逻辑
  （`:884`、`:1862` 附近的注释提到 "runtime admission cap must match the
  recycling-aware bound"），**具体裁剪点尚未钉死**，需要继续定位。
- 但注意：P 的 lower-SWA 页在 P 自己的 manager 里已被回收，P 手里也没有第 3 页
  ⇒ 方案 B 同时要求 P 的 manager 保留更多窗口。改动面比方案 A 大。

因为方案 B 需要跨 core 改且 P 侧数据未必还在，**建议先做方案 A**。

## 5. 复现与验证环境

- 8+8：A3-21，P `dsv41-ced-trace-p-*`（chip0-7，18990），D
  `dsv41-ced-trace-d-*`（chip8-15，18991），proxy 18992。D 为
  `GRAPH=1 EAGER=0` + prompt-tail eager + metadata inline + `MULTISTREAM=0 DSA_OVERLAP=0`。
- 启动/探针：`tools/launch_ced_trace_a3.sh`、`tools/ced_layer_trace_sequence.sh`、
  `tools/ced_seq_probe.py`、`tools/patch_trace_env.py`。
- 判据：连续形态 ≥10 次全过；碎片形态（`descents>0`）≥10 次全过；两种不同碎片形态
  都通过；仍为 `GRAPH=1 EAGER=0`；连续路径相对既有基线不退化。
- 噪声底（必须先测）：同一布局重复两次、不同连续布局两次、连续 vs 碎片，三种配对。

## 5.5 修复前提已用算子自身文档核实（2026-09-24 21:5x，本地只读）

方案 A 依赖一个隐含假设：`ori_block_table` 的列号是**局部** KV 坐标
（`local_token // block_size`），而不是全局 token 坐标。这一点已在算子自身的
设计与源码里确认，不需要上机猜测：

- 上游算子源码在本仓库：
  `upstream-v41/vllm-ascend-upstream/csrc/attention/sparse_flash_mla/`。
- `docs/design.md` §2.4 原文：
  “原始侧右下对齐的 query 位置为 `p = Lori - Lq + i`。mode 4 的有限窗口保留
  `p - win_left <= j <= p + win_right`，**再与 `[0, Lori)` 相交**”。
  ⇒ query 的坐标是**相对** `Lori = seqused_ori_kv` 的局部量；窗口本身会在
  `0` 处被 clamp，这正是方案 A 想要的语义。
- 同文档 §2.3 原文：
  “`page = logical_token / block_size` … `physical_page = block_table[b, page]`”。
  ⇒ 列号就是上面那个局部 `logical_token // block_size`。
- Host 侧 checker 只校验 `ori_block_table` 的 dtype（INT32）、维数（2 维）、
  非空（`op_host/checkers/paged_attention_checker.cpp::CheckBlockTable`），
  **不要求块表宽度与 `seqused_ori_kv` 匹配** ⇒ 窄化后的块表不会被拒。

因此“窄化块表 + 相对 `seqused_ori`”是自洽的：kernel 用局部坐标 `p` 去索引窄表
的列 `0..num_pages-1`，而该窄表正是把原行表里 `base_page..` 的那几列 gather
过来的。CPU 索引仿真（`fix/ced-blockbug` 的 `tools/ced_swa_clip_sim.py`）在
N=100..2999 上复核：`clip` 视图越界样本 **0**，`legacy` 视图在真实 1M 例
（N=1019847）读到列 `7965,7966,7967`，其中 `7965` 不属于本请求持有的
`{7966, 7967}`。

触发条件也随之变精确：**N ≥ 256 且 N % 128 ≠ 0** 时 legacy 必然越界；
N % 128 == 0 或 N ≤ 255 时不越界（已用 2744 个长度扫过，无反例）。

### 为什么短上下文与"池未绕回时"仍然答对（当前最合理的解释）

行内被回收的列写的是 `0`（null block）⇒ kernel 会去读**物理块 0**。
KV 池没绕回时块 0 还没被分配给任何请求（内容为 0/未初始化），对 softmax 的
贡献近似为零，答案不被翻掉；一旦池绕回，块 0 已被分给**最早的那个请求**，
于是这 57 个 replay query 会把别人的 KV 以正常权重混进来 ⇒ 首 token 直接 EOS。
这与实测完全吻合：碎片（`descents>0`）正是"池已绕回"的代理指标，而 144K /
22-token 这些小池场景天然不会绕回。

> 待验证（下次允许上机时的一条只读探针即可）：在真实请求的 replay 步打印
> `metadata.swa.block_table[0]` 的第 `7965` 列，确认它是 `0`；再用
> `npu-smi`/调度日志确认块 0 在绕回后已分配给其他请求。

- `[CED-LAYER-TRACE]` 探针在 warmup/dummy 阶段会对 `positions` 取到未初始化张量，
  抛 `IndexError`（已 fail-open，只打日志）。它本身不影响模型，但会让启动日志变大、
  拖慢 warmup；做正式实验时应把 `V41_CED_LAYER_SNAPSHOT_POS` 置空。
- `[CED-BLOCKS]`/`[CED-SWA-TRACE]` 探针在调度进程里调用
  `get_tensor_model_parallel_rank()` 会 assert（TP 进程组未初始化）——已加 try/except，
  改动时务必保留。
- `/v1/chat/completions` 的 `token_ids` 字段被 reasoning parser 抑制，必须同时传
  `include_reasoning=true` 才会回传（`tools/ced_seq_probe.py` 已处理）。
## 6. 已知的观测陷阱

