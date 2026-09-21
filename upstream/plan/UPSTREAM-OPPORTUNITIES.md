# 上游 `origin/main` 优化机会清单（H 号，2026-09-21）

> **任务**：在 `vllm-ascend-upstream@origin/main`（`c173a64a4`）里找**新的、可独立提交**的优化目标，
> 只做**源码调研**（未占卡、未跑 NPU、未 ssh 单卡机、未改任何仓库文件——本文件是唯一产物）。
>
> **框架**：帮 RFC #16375（`[RFC]: DeepSeek V4.1 Roadmap`）打勾。下表每个候选在“RFC 关联”一列
> 给出它对应的条目号；我们的两个已提交分支（rope `index_select`、MoE expert-mask 范围比较）**不重复列出**，
> 只在“已被我们覆盖”一节做对照。
>
> **证据分级**：【实测】= 我们有同硬件（A3/910C）同代码形态的 profile 或配对 A/B 数字；
> 【推断】= 需要 profile 才能确认收益，代码路径本身可确证；【源码】= 仅能由代码判定存在性与调用点；
> 【未确认】= 是否活代码/是否 hot 没查清（按任务要求写“未确认”，不脑补）。

---

## 0. 口径与证据来源（先读，否则数字对不上）

### 0.1 代码基线

全部行号指向 **`origin/main` = `c173a64a44dec4ba97aaba6277b1dfc1562eda19`**，可用
`git show origin/main:<path> | awk 'NR>=A && NR<=B'` 复现。为可追溯，每个候选都附了取行命令。

### 0.2 【实测】数字的两处来源（都是我们自己的，不是上游的）

| 记号 | 来源 | 口径 |
|---|---|---|
| **P1** | `dsv41-release/reports/small-op-audit.md`（258,795 行 `op_summary.csv`，A3-node1、V4.1-Flash W4A8、8 token/步，40 层 target + 3 层 draft） | **每步算子次数**（锚点切步 = `wq_a` QBMV3，每 40 锚 1 步，主相位 68 步） |
| **P2** | 我们两个已合并分支的单卡复现（`logs/04`、`logs/05`、`pr/PR-rope-index-select.md` §3） | ACLGraph 口径 vs eager 口径**符号会相反**（见 `logs/04-…moe-mask-graph-vs-eager.md`） |

**P1 里逐条给出的关键账**（后面反复引用，原文 §2.2 / §3.1）：

| 每步次数 | 算子 | 调用点（P1 原文行号，指容器快照） |
|---:|---|---|
| **41** | `aclnnInplaceCopy_CastAiCore_Cast` `[8]` INT32→INT64 | `ops/fused_moe/router/fused_topk_router.py` 的 `input_ids = input_ids.to(torch.int64)` |
| **40 / 40** | `Index` + `IndexCheck` | `ops/fused_moe/token_dispatcher.py` 的 `expert_map[topk_ids]` ← **我们 0002 已覆盖** |
| **18 / 23** | `Index` / `IndexCheck` | `ops/rope_dsv4.py` 的 RoPE 表查询 ← **我们 0003 已覆盖** |
| 1 / 1 | `Index` + `IndexCheck` `[8,129280][1]→[1,129280]` | `sample/rejection_sampler.py` 取 last-token logits |
| 4 | `IndexCheck` | draft 侧 `expert_map[topk_ids]`（V4.1-Flash 的 dspark，128 专家 / topk 3） |
| 3 | `Index` `[8,5120][8]→[8,5120]` | draft 输入准备的 positions/ids 取行 |

> ⚠️ **P1 是“我们部署”的账，不是上游的账**；它给出的是**次数**（可迁移）与**形状**（可对齐源码），
> 绝对时间要按 P2 的 ACLGraph 口径另行测量。

### 0.3 在途冲突怎么查的（可复现）

```bash
# 1) 拉全部 open PR 标题（1933 个）
gh api --paginate 'repos/vllm-project/vllm-ascend/pulls?state=open&per_page=100' \
  --jq '.[] | "\(.number)\t\(.title)"' > /tmp/open_prs.txt
# 2) 拉「最近更新的 1000 个 open PR」各自改了哪些文件（GraphQL，search 上限 1000 条）
gh api graphql -f query='query { search(query:"repo:vllm-project/vllm-ascend is:pr is:open sort:updated-desc",type:ISSUE,first:100){pageInfo{hasNextPage endCursor} nodes{... on PullRequest{number files(first:100){nodes{path}}}}}}' \
  --jq '.data.search.nodes[] | .number as $n | .files.nodes[] | "\($n)\t\(.path)"' > /tmp/pr_files.tsv
# 3) 查某个文件被哪些 open PR 命中
grep -F "<path>" /tmp/pr_files.tsv | cut -f1 | sort -n | uniq
```

**这个方法的两条局限**（后面“在途冲突”一栏都按此口径，不要当完备结论）：

1. GitHub search 只返回**前 1000 条**，更老的 open PR（例如 #12663、#14858）不在 `pr_files.tsv` 里，
   只能靠标题检索（`grep -i` 于 `/tmp/open_prs.txt`）补；
2. `files(first:100)` 对**改动超过 100 个文件**的 PR 会截断——`#16993`（V4.1 framework 重贴 + DSpark 融合）
   就属于这一类，它的真实改动面比列出来的更大。

---

## 1. 摘要：12 个候选，按“值得做的程度”排序

排序权重 = **证据强度 × 每步可省算子/次数 × 独立性（能不能单独成 PR）÷ 在途冲突**。

| # | 候选 | 模式 | 每步规模（【实测】/【推断】） | RFC | 在途冲突 | 证据 |
|---:|---|---|---|---|---|---|
| **1** | `ops/causal_conv1d.py` 的 PyTorch 回退：逐请求 Python 循环 + `.item()` | **B** | 每次调用 O(batch) 次 `.item()` 同步 + O(batch) 个 conv kernel；**且代码注释自证会破坏 ACLGraph capture**；门控：`HAS_TRITON` 且 NPU Triton 内核 import 失败（`update` 分支） | [73][77] | 🟢 无（该文件 0 命中；`patch_triton.py` 有 0 命中） | 【源码】+ 注释自证 |
| **2** | `ops/fused_moe/router/fused_topk_router.py:162` 每层重复 `input_ids.to(torch.int64)` | 其他 | **41 次/步**【实测 P1】≈ 0.2 ms/步 | [73][91] | 🔴 `#16993`/`#16925`/`#16689` 都在改 | 【实测】+【源码】 |
| **3** | `models/glm5next/mtp.py:146` `topk_indices_buffer[slot_ids]` | **A** | 每次 MTP 提案 × MTP 层数，≥1 kernel | [77][88] | 🟢 **0 命中** | 【源码】 |
| **4** | `spec_decode/utils.py:30-32` 三个高级索引 | **A** | 3 kernel/步（async spec decode 步） | [77] | 🟢 仅 `#14995`/`#15893`（都在更早版本） | 【源码】 |
| **5** | `sample/rejection_sampler.py:202/219` 取 logits 行 | **A** | **1–3 次/步**【实测 P1 §3.1】 | [77][91] | 🟡 6 个 open PR 命中该文件 | 【实测】+【源码】 |
| **6** | `ops/fused_moe/routed_experts.py:595` `log2phy[topk_ids]` | **A** | MoE 层数 × 1 kernel/步（**仅 EPLB 开启时**） | [66] | 🟡 `#16871`/`#16800`/`#16899` | 【源码】 |
| **7** | `ops/rotary_embedding.py:102-103` `_cos_cache[positions]`（MLA/GLM-5.3） | **A** | ~2 次/步（每 cache group 一次）× 2 kernel | [87] | 🟡 同文件 12 个 PR，但**不同区域**（#16335 改 474+、#14858 改 494+） | 【源码】+【实测类比】 |
| **8** | `models/qwen3_dflash2.py:163,171` 多维索引（DFlash2 selector） | **A** | 2 kernel/草案步 | [88] | 🟢 **0 命中** | 【源码】 |
| **9** | `attention/dsa_v1.py:530-531,573` `build_vision_bidirectional_swa_indices` | **A** | 4 kernel/步（**仅视觉 prefill**），含 `block_table[req_ids]` 大 gather | [88] | 🔴 该文件 36 个 PR 命中 | 【源码】 |
| **10** | `attention/utils.py:342-349` `filter_chunked_req_indices` 逐个 `torch.arange` | **B** | N_req 个 Range kernel + 1 Cat / 步（**仅 PCP chunked prefill**） | [28][54] | 🔴 该文件 28 个 PR 命中（含 `#16915`） | 【源码】 |
| **11** | `worker/dcp_utils.py:480-495` `np.array_split`+逐请求 `np.append` | **B** | host 侧 O(num_reqs) 次重分配/步 | [28][29] | 🟡 4 个 PR（`#16790` 等） | 【源码】 |
| **12** | `ops/fused_moe/router/fused_topk_router.py:67` `tid2eid[lookup_ids]`（VL 路由） | **A** | VL 模型每 MoE 层 × 1–2 kernel/步 | [63][74] | 🔴 同 #2 那组 | 【源码】 |

---

## 2. 逐个候选详情

### 候选 1 ★ `ops/causal_conv1d.py`：纯 Python 回退里的逐请求循环 + `.item()`（模式 B）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/ops/causal_conv1d.py:132-146`（`causal_conv1d_fn`）、`:222-230` + `:251-297`（`causal_conv1d_update`） |
| 现状代码 | `causal_conv1d_update`：`idx = int(conv_state_indices[i].item())`（:224）、`for i in range(batch):`（:251/:265/:280）、`accepted = int(num_accepted_tokens[i].item())`（:257/:271/:290）、`start = int(query_start_loc[i].item())`（:281-282）<br>`causal_conv1d_fn`：`seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()`（:132）+ `for i, x_s in enumerate(splits):`（:136）、`cache_idx = int(cache_indices[i].item())`（:137） |
| 复现 | `git show origin/main:vllm_ascend/ops/causal_conv1d.py \| awk 'NR>=220 && NR<=300'` |

**模式**：B（Python 逐元素循环；而且是**最坏的一种**——循环里带设备同步）。

**为什么热（证据链，含两条必须交代的门控）**

1. 这个文件不是测试工具：`vllm_ascend/patch/worker/patch_triton.py:9-12` 把它 import 进来，`:57-58`
   把 vllm 的 `causal_conv1d_update` / `causal_conv1d_fn` 替换成这两个实现：
   ```python
   _cc1d.causal_conv1d_update = _npu_causal_conv1d_update
   _cc1d.causal_conv1d_fn = _npu_causal_conv1d_fn
   ```
   两条路径都只在**模块属性调用**（`causal_conv1d.causal_conv1d_update(...)`）时才被改写；
   对已经 `from ... import causal_conv1d_update` 的调用方无效 —— 这一点由 `:36-39` 的注释确认
   （“Models that live in vllm-ascend import the Ascend entry points directly and do not rely on this rebind”）。
2. **门控 A**：`patch/worker/__init__.py:22-24`
   ```python
   if HAS_TRITON:
       import vllm_ascend.patch.worker.patch_triton
       import vllm_ascend.patch.worker.patch_v2.patch_triton
   ```
   ⇒ 上面的替换**只在 `HAS_TRITON` 为真时发生**（不是无条件）。
3. **门控 B**：`patch_triton.py:321-337` 只尝试覆盖 `causal_conv1d_update`，**`causal_conv1d_fn` 没有任何 Triton 覆盖**
   （`grep -n 'causal_conv1d_fn' patch_triton.py` 只有 10/51/54/58 四处）⇒ prefill 版回退是**常驻绑定**。
   `update` 版则在 NPU Triton 内核 import 失败时退回 PyTorch，而这个 `except` 分支自己在日志里写明了后果：
   > `"NPU Triton causal_conv1d_update is unavailable (%s); falling back to the PyTorch implementation, which **syncs per request** and therefore **stalls ACL graph capture at decode-FULL**."`
   > —— `vllm_ascend/patch/worker/patch_triton.py:332-336`
4. 同一文件 `:36-39` 的注释再次确认：“CUDA causal_conv1d kernels use `tl.extra.cuda.gdc_wait` which Ascend Triton does not provide”。

⇒ **替换关系、门控条件、以及在 decode-FULL 下会打断捕获，这三条都由代码直接判定**（不是推断）。
**未确认（两条，PR 前必须先查）**：
* 在目标 CANN/Triton 组合上 `:325` 的 `causal_conv1d_update_npu` 是否总能 import 成功
  （失败才走 PyTorch 回退）；上游注释暗示**至少有人踩到过**；
* 实际有多少 vllm-core 的 Mamba/KDA 模型**走模块属性调用**（受替换影响）而不是 `from … import`（不受影响）——
  这决定本候选是“每步都疼”还是“特定模型才疼”。

**建议改法**（具体到代码）

把 `causal_conv1d_update` 改成无 host 同步的批量实现，保留现有签名与 `pad_slot_id` 语义：

1. `idx = conv_state_indices.to(torch.int64)`；`keep = idx != pad_slot_id`；用
   `safe_idx = torch.where(keep, idx, 0)` 后 `states = conv_state.index_select(0, safe_idx)`（**1 个 kernel，替掉 batch 次 `.item()`**）。
2. `accepted` 变成张量：`tokens = x` 按 `accepted` 掩码（`num_accepted_tokens` 已是设备张量），
   用 `masked_fill`/`where` 处理 `accepted<=0` 的行，而不是 `if accepted <= 0: continue`。
3. 用 `F.conv1d(..., groups=dim)` 一次算整批（现有 `causal_conv1d_ref` 已经是这么做的，只是被逐请求调用），
   最后用 `conv_state.index_copy_(0, safe_idx, final_states)` 写回（`gather_initial_states`/`scatter_states`
   在 `models/glm5next/ops/state_ops.py` 里已经是这种写法，可直接照抄风格）。
4. 全部控制在 `keep` 掩码内，`pad_slot_id` 行不读不写，避免 index_select 的越界语义差异。

**预期收益**
* 【源码】每次调用从 **O(batch) 次 `.item()`（每次一个 device→host 同步）** 降到 **0 次**；
  每步 batch=8~128 时，这是 host 侧的**串行同步**，量级远大于 1 个 kernel。
* 【推断】能让该路径**在 ACLGraph 里被捕获**（这正是注释里说的收益）——即从“掉出图、每步 host 停顿”变成图内。
* 参照我们的同类经验：`reports/engram-host-breakdown.md` 里把 host 侧 per-request 工作改成批量后，
  decode 步从 **3.379 ms → 0.058 ms**（RFC [77]，那是 Engram 路径，机制相同）。

**改动规模**：~80–120 行（重写 `causal_conv1d_update`；`causal_conv1d_fn` 可同 PR 做，~40 行）。

**风险**
* 语义等价性：`pad_slot_id` 跳过、`accepted<=0` 跳过、`seq_tokens[:accepted]` 截断、`out` 保持 `x` 的 dtype（原实现先 `x.to(conv_state.dtype)` 再 `out.clone()`）——这些都要逐条对齐。
* 该函数是**参考实现**：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py:6` 直接把它当 `ref` 对比 Triton 内核 ⇒ **改它等于改测试基准**，PR 里必须说明“ref 的数值语义未变”。
* 反向风险：如果上游把 NPU Triton 内核视为唯一生产路径，reviewer 可能认为回退不值得优化 ⇒ PR 描述要引用 `patch_triton.py:332-336` 的原文。

**在途冲突**：`grep -F "vllm_ascend/ops/causal_conv1d.py" /tmp/pr_files.tsv` → **0 命中**；
`patch_triton.py` 同样 0 命中。🟢 可安全并行。

**证据等级**：【源码】（调用点、替换关系、`syncs per request` 与 `stalls ACL graph capture` 都是仓库内自述）。

---

### 候选 2 ★ 每层重复的 `input_ids.to(torch.int64)`（“其他”：每层冗余算子）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/ops/fused_moe/router/fused_topk_router.py:162`（`input_ids = input_ids.to(torch.int64)`） |
| 建议落点 | `vllm_ascend/models/deepseek_v4/model.py:954-962`（层循环之前；`:939-940` 是 SP shard 分支） |
| 复现 | `git show origin/main:vllm_ascend/ops/fused_moe/router/fused_topk_router.py \| awk 'NR>=155 && NR<=172'` |

**模式**：其他（不是 A/B，但属于同一根因：**每层重复做同一件与层无关的事**）。

**为什么热（证据链）**

1. 调用链（全部 `origin/main`）：
   `AscendRoutedExperts.forward_impl`（`ops/fused_moe/routed_experts.py:670`）
   → `self._select_experts`（`:580`）
   → `self.router._select_experts(...)`（`:589`）
   → `AscendFusedTopKRouter._compute_routing`（`router/fused_topk_router.py:139`）
   → `:159-162`：`if self.tid2eid is not None or self.bias_vl is not None:` … `input_ids = input_ids.to(torch.int64)`。
   ⇒ **每个 hash 路由的 MoE 层每步各 1 次**（`models/deepseek_v4/model.py:328`：`self.hash = layer_idx < config.num_hash_layers`）。
2. **【实测 P1】** 我们的 V4.1-Flash 部署上，`aclnnInplaceCopy_CastAiCore_Cast [8] INT32→INT64`
   **41 次/步**（40 个 target 层 + 1 个 draft 层），来源就是这一行。
3. 该 cast 的输入是模型入参 `input_ids`（int32），在 40 层之间**从未改变** ⇒ 40 次完全重复。
4. 关键细节（决定了改法为什么安全）：这个 cast 发生在**集合通信之前**（`:164-168` 的
   `all_gather_input_ids` / `pad_and_split_input_ids`），所以把 cast 上提后，**通信拿到的 dtype 与今天完全一致**。

**建议改法**

在 `models/deepseek_v4/model.py` 的层循环前（`:954` 之前）加一行：

```python
if input_ids is not None:
    input_ids = input_ids.to(torch.int64)
```

`fused_topk_router.py:162` **不用改**：`.to()` 对已是 int64 的张量是 no-op（不产生 kernel），
留作对其它调用方的防御。这样 PR 的 diff 面收敛到 1 个文件、1 行。

**预期收益**：【实测】−41 算子/步（V4.1-Flash、A3、8 token/步）；
按我们报告里沿用的 5 µs/算子下发空隙常数 ≈ **−0.2 ms/步**。
其它模型按 `num_hash_layers` 等比缩小（非 hash 层本就不走这条分支）。

**改动规模**：1–3 行。

**风险**：低。
* 不能在**嵌入之前**cast（`models/deepseek_v4/model.py:921` 用 `input_ids` 做 embedding），落点必须在 `:954` 之后、层循环之前。
* PP 非首 rank 传 `None` ⇒ 必须保留 `is not None` 守卫（写法已含）。
* 若未来有人依赖“传给 `all_gather_input_ids` 的必须是 int32”，需在 PR 里说明今天传进去的**已经是 int64**（因为 cast 在调用之前）。

**在途冲突**：🔴 `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` 被
`#16993`、`#16925`、`#16689`、`#16192`、`#15740`、`#15363` 命中；
`vllm_ascend/models/deepseek_v4/model.py` 被 `#17016`、`#16960`、`#16958`、`#16621`、`#16548`、`#16355`、
`#16230`、`#16145`、`#16029`、`#15740`、`#15614`、`#15363`、`#15217`、`#15191`、`#15022`、`#15011` 命中。
**但本改法只动 model.py 的一行、不动 router**，实际合并风险远小于上面的 PR 数量所暗示的
（`#16993` 会新增 `models/deepseek_v41/`，届时同一行要移植过去）。

**证据等级**：【实测】（次数来自我们部署 profile）+【源码】（调用链与 dtype 流向）。

---

### 候选 3 ★ `models/glm5next/mtp.py:146`：`topk_indices_buffer[slot_ids]`（模式 A，**零冲突**）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/models/glm5next/mtp.py:140-146`（`compact_topk_indices`） |
| 现状代码 | `topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]` |
| 复现 | `git show origin/main:vllm_ascend/models/glm5next/mtp.py \| awk 'NR>=138 && NR<=150'` |

**模式**：A（高级索引；且是“读高级索引 + 切片写回”的复合形态）。

**为什么热（证据链）**

1. 调用点：`vllm_ascend/spec_decode/mtp.py:14-22` 的 `compact_mtp_topk_indices(draft_model, token_indices_to_sample, …)`
   → `:22 draft_model.compact_topk_indices(token_indices_to_sample)`；
   上层是 `vllm_ascend/spec_decode/llm_base_proposer.py:1391-1395`（MTP 提案的步 0 之后，**每个 decode 步都会走到**）。
2. `compact_topk_indices` 内部对 `self._mtp_mla_attns` 里**每个 MTP 层各做一次**（`:143-146`），
   MTP 层数 = `config.num_nextn_predict_layers`（`:110`）。
3. **同仓先例（重要）**：`vllm_ascend/spec_decode/mtp.py:37` 对**同一语义**已经写成
   `rows = buffer.index_select(0, gather_indices)`。
   ⇒ 这是“同一个仓里两种写法并存”的不一致，最适合做小型一致性 PR（reviewer 几乎无需判断新语义）。

**建议改法**

```python
rows = topk_indices_buffer.index_select(0, slot_ids.to(torch.int64))
topk_indices_buffer[:num_slots].copy_(rows)
```

必须**先出临时张量再写回**（不能 `index_select(..., out=...)` 直接写进前 `num_slots` 行）：
`slot_ids` 与写入区间重叠，原写法靠高级索引先拷贝整块才成立。

**预期收益**：【推断】每次调用省 1 个 kernel（`Cast`+`Index`+`IndexCheck`+`Copy` → `Cast`+`GatherV3`+`Copy`）；
乘 MTP 层数。**注意**：`slot_ids` 是 int32（`llm_base_proposer.py:267-269` 的 `self.token_indices_to_sample` 就是 int32），
两种写法都会带一个 index 提升 Cast，所以只能省下 `IndexCheck` 那一个。

**改动规模**：2–4 行。

**风险**：低。`slot_ids` 来自 `cad.query_start_loc[1:] - 1`（`llm_base_proposer.py:960` 或 `:1743` 同源），
非负且 < num_tokens；**未确认**是否存在 int32 溢出（num_tokens > 2^31 不可能）。

**在途冲突**：🟢 `grep -F "vllm_ascend/models/glm5next/mtp.py" /tmp/pr_files.tsv` → **0 命中**；
`vllm_ascend/spec_decode/mtp.py` 同样 **0 命中**。

**证据等级**：【源码】。

---

### 候选 4 `spec_decode/utils.py:30-32`：三个高级索引（模式 A）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/spec_decode/utils.py:28-32` |
| 现状代码 | `gather_indices = prev_positions.clamp(min=0)` / `valid_counts = valid_sampled_token_count[gather_indices]` / `prev_computed = num_computed_tokens[gather_indices]` / `prev_drafts = prev_num_draft_tokens[gather_indices]` |
| 复现 | `git show origin/main:vllm_ascend/spec_decode/utils.py \| awk 'NR>=25 && NR<=40'` |

**为什么热**

* 调用点 `vllm_ascend/worker/model_runner_v1.py:1446`，条件 `use_async_spec_decode and valid_sampled_token_count_gpu is not None and prev_req_id_to_index`
  ⇒ **async spec decode 的每个 decode 步 1 次**（另一个调用方是 `_310p/model_runner_310p.py:317`）。
* 三个索引都是 `[num_reqs]` 的整数张量，`gather_indices` 已 `clamp(min=0)` ⇒ 天然满足 `index_select` 的非负要求。

**建议改法**：三行各换成 `torch.index_select(src, 0, gather_indices)`。

**预期收益**：【推断】3 个 kernel/步（每处 `Index`+`IndexCheck` → 1 个 `GatherV3`）。
量级参考：我们 MoE mask 案的**实测**是“61 层 × 1 kernel ≈ −20 µs/步”，即单 kernel ≈0.3 µs，
所以这里是 **~1 µs/步**——**小**，只适合作为“同模式批量清扫”的一部分。

**改动规模**：3 行。

**风险**：低。**未确认**：`prev_positions` 的 dtype（若是 int32，`index_select` 需要 `.long()`，
会新增一个 Cast 把收益吃掉——高级索引今天也付同样的 Cast，所以最坏是不赚不亏）。

**在途冲突**：🟢 仅 `#14995`、`#15893` 命中该文件（都是较旧的 open PR）。

**证据等级**：【源码】。

---

### 候选 5 `sample/rejection_sampler.py:202/219`：按索引取 logits 行（模式 A）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/sample/rejection_sampler.py:202`、`:219` |
| 现状代码 | `bonus_logits = logits[bonus_logits_indices]`、`raw_target_logits = logits[target_logits_indices]` |
| 索引来源 | `vllm_ascend/worker/model_runner_v1.py:1725-1738`（`np.repeat` / `arange` 构造的 **numpy 数组**，`:1761` 还有 `draft_token_ids[target_logits_indices + 1]`） |
| 复现 | `git show origin/main:vllm_ascend/sample/rejection_sampler.py \| awk 'NR>=192 && NR<=222'` |

**为什么热**：spec decode（EAGLE/MTP/DSpark）每步 1 次拒绝采样 ⇒ 每步各 1 次索引。
且 `logits` 形状是 `[num_tokens, vocab]`，**是整步最大的单个张量**；【实测 P1 §3.1】给出
`Index`+`IndexCheck` 各 1 次/步，形状 `[8,129280][1]→[1,129280]` 与 `[7]→[7,129280]`。

**建议改法**
* 通用：`torch.index_select(logits, 0, torch.as_tensor(bonus_logits_indices, device=logits.device))`（省掉 `IndexCheck`）。
* 特例（decode 只有 1 行时）：`bonus_logits_indices` 恒为“每请求最后一行”，可以退化成
  `logits.narrow(0, i, 1)`/切片视图（**0 个 kernel**）——但那要改 metadata 的产生方式，属于更大改动。
* 注意注释 `:197-200` 依赖“索引结果是**新 storage**”这一语义（后续就地改 `target_logits` 不污染原 `logits`）：
  `index_select` 同样返回新张量，语义保持。

**预期收益**：【实测】1–3 算子/步（P1）；【推断】`index_select` 的访存模式对 129K 宽的行拷贝更友好，但需实测。

**改动规模**：2–6 行。

**风险**：低。索引来自 numpy 数组（int64），非负且 < num_tokens。

**在途冲突**：🟡 该文件被 6 个 open PR 命中：`#13755`、`#15011`、`#15235`、`#16363`、`#16430`、`#16653`、`#16665`。

**证据等级**：【实测】+【源码】。

---

### 候选 6 `ops/fused_moe/routed_experts.py:595`：`log2phy[topk_ids]`（模式 A，EPLB 路径）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/ops/fused_moe/routed_experts.py:594-595` |
| 现状代码 | `if self.log2phy is not None:` / `topk_ids = self.log2phy[topk_ids]` |
| 复现 | `git show origin/main:vllm_ascend/ops/fused_moe/routed_experts.py \| awk 'NR>=588 && NR<=598'` |

**为什么热（含一个重要的“不热”条件）**

* 位置在 `_select_experts`（`:580`）→ 由 `forward_impl`（`:670`）**每个 MoE 层每步调用一次**。
* **但**：`log2phy` 只有在 `init_eplb_config` 判定 `eplb_enable`（`dynamic_eplb` 或 `expert_map_path`）时才非 None
  （`vllm_ascend/eplb/core/eplb_utils.py:73-116`；`:83-84` `expert_map_path` 分支、`:93-96` 非 EPLB 分支直接 `return None`）。
  ⇒ **默认（非 EPLB）部署里这行根本不执行**。
* 因此它的受众是 RFC [66]（dynamic EPLB）与正在推进的 `#16871`（global expert pool EPLB）、`#16800`、`#16969`。
  对我们还有一层意义：它与我们 0002 的“**用 1 个 kernel 替掉 Index+IndexCheck**”是**同一手法**，可以复用同一份 benchmark 与 PR 模板。

**建议改法**

```python
flat = topk_ids.reshape(-1)
topk_ids = torch.index_select(self.log2phy, 0, flat).view_as(topk_ids)
```
（`topk_ids` 是 2-D `[tokens, topk]`，`index_select` 只接受 1-D 索引 ⇒ 拍平 + `view_as`，两者都是零成本 view。）

**预期收益**：【推断】MoE 层数 × 1 kernel/步（40–61 层）；按我们 0002 的实测量级折算 **−15 ~ −25 µs/步**。

**改动规模**：3–8 行（若需要处理 -1 与 dtype，见风险）。

**风险（三条，都必须先验证）**
1. **负索引语义**：高级索引里 `-1` 表示“最后一行”，`index_select` 对 `-1` 是**报错**。若 `topk_ids` 可能含 -1，必须先 mask。
2. **dtype**：`index_select` 要求 int64 索引；若 `topk_ids` 是 int32（很可能，见 `moe_gating_top_k` 的 `indices_type`），
   需要 `.long()` ⇒ 新增 1 个 Cast（与今天高级索引内部的隐式 Cast 抵消），**净收益只剩 IndexCheck**。
3. `log2phy` 经 `_promote_attr_to_buffer` 注册为 buffer（`:511`），改动不得破坏它的地址稳定性（不要换成新建张量再赋值给属性）。

**在途冲突**：🟡 该文件被 `#16800`、`#16871`、`#16899`、`#16743`、`#16555`、`#16448`、`#16447`、`#16384` 等命中；
`#16969` 改的是同目录 `ops/fused_moe/eplb.py` 与 `ops/triton/eplb.py`（**不同文件**）。

**证据等级**：【源码】（“仅 EPLB 生效”这一点也由源码判定）。

---

### 候选 7 `ops/rotary_embedding.py:102-103`：MLA/SFA 的 cos/sin 表查询（模式 A）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/ops/rotary_embedding.py:102-103`（`get_cos_and_sin_mla`） |
| 现状代码 | `cos = _cos_cache[positions].unsqueeze(1).unsqueeze(2)` / `sin = _sin_cache[positions].unsqueeze(1).unsqueeze(2)` |
| 复现 | `git show origin/main:vllm_ascend/ops/rotary_embedding.py \| awk 'NR>=96 && NR<=112'` |

**为什么热**

* 调用点（5 个，全部 `origin/main`）：`attention/mla_v1.py:639`（prefill 元数据）、`:726`（decode 元数据，`use_cache=True`）、
  `attention/sfa_v1.py:578`、`attention/indexer.py:1031`、`models/kimi_k3_dspark.py:204`。
* 频率：metadata builder 每步每 kv-cache group 调 1 次
  （MRV1：`worker/model_runner_v1.py:3486-3560`；MRV2：`worker/v2/attn_utils.py:342-348`）
  ⇒ decode 步通常 2 次（MLA group + indexer group）。
* 与我们的 0003 同源：**同一个“`table[positions]` 高级索引”形态**，只是表不同（这里是 `_cos_cache`，`rope_dsv4` 里是 `full_rope_cos`）。

**建议改法**

```python
cos = torch.index_select(_cos_cache, 0, positions).view(positions.size(0), 1, 1, -1)
sin = torch.index_select(_sin_cache, 0, positions).view(positions.size(0), 1, 1, -1)
```
（`.unsqueeze(1).unsqueeze(2)` 本来就是 view，保持返回形状 `[n,1,1,rope_dim]` 不变；
`use_cache=True` 分支里 `_cos_mla[:num_tokens, ...] = cos` 的拷贝**保留**，且必须仍写进**同一地址**的持久 buffer——
注释 `:106-113` 明确说 ACLGraph replay 依赖这个指针稳定。）

**预期收益**：【推断】每处 2 个 kernel（`Index`+`IndexCheck` → `GatherV3`）× 2 次/步 ≈ **4 kernel/步**；
【实测类比】我们在 `rope_dsv4` 上做同一替换时的账目是 **`Index −4104`、`IndexCheck −4104`、`GatherV3 +6624`**（A3，7912 锚点，`pr/PR-rope-index-select.md` §2）。

**改动规模**：4–8 行 + 单测（`tests/ut/ops/` 下已有同类测试文件可挂）。

**风险**：低。需确认 `positions` 非负、dtype 为 int64（高级索引允许 -1，`index_select` 不允许）——
调用点普遍是 `common_attn_metadata.positions[...].long()`，但没有逐处证明“无 -1 填充”；**这一条标【未确认】**。

**在途冲突**：🟡 同文件有 12 个 open PR 命中，其中最有威胁的是
`#16335`（改 `AscendMRotaryEmbedding.__init__` 与 `forward_oot`，**474 行以后**）与
`#14858`（改 **494 行以后**）；本候选在 **102-103 行**，区域不重叠，但同文件 rebase 要留意。

**证据等级**：【源码】+【实测类比】（机制与我们的 0003 完全一致）。

---

### 候选 8 `models/qwen3_dflash2.py:163,171`：DFlash2 候选选择器的多维索引（模式 A，**零冲突**）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/models/qwen3_dflash2.py:163`、`:171`（`_score_edges`） |
| 现状代码 | `successors = successor_table[candidate_ids]` / `predecessors = predecessor_table[predecessor_ids]` |
| 复现 | `git show origin/main:vllm_ascend/models/qwen3_dflash2.py \| awk 'NR>=154 && NR<=175'` |

**为什么热**

* `_score_edges` 由 `CandidateSelector.forward`（`:200-223`）唯一调用，`CandidateSelector` 是 DFlash2 草案头的一部分
  ⇒ **每个草案步 1 次**（和候选 3/4 同属 spec-decode 关键路径）。
* 两次索引都是**多维索引**（`candidate_ids` 是 `[B, L, K]`），这是模式 A 里最容易被忽略的一类：
  它不能直接换 `index_select`（索引不是 1-D），但**拍平后可以**。

**建议改法**（“多维索引 → 拍平 + `index_select` + view”）

```python
def _gather_rows(table, ids):
    return table.index_select(0, ids.reshape(-1).to(torch.int64)).view(*ids.shape, table.shape[-1])
```
`successors`/`predecessors` 两处各一行。注意 `predecessor_ids` 是由 `anchor_token_ids` 与 `candidate_ids` `cat` 出来的，
**也必须先拍平再索引**，否则形状会错。

**预期收益**：【推断】2 个 kernel/草案步。

**改动规模**：4–8 行。

**风险**：中低。
* **占位值**：若草案的 `candidate_ids` 用 `-1` 表示 padding，高级索引会“绕回最后一行”，而 `index_select` 会**报错**
  ⇒ **未确认**（需查 `vllm` 上游 DFlash2 speculator 的 `candidate_ids` 生成，或直接看 `qwen3_dflash2.py` 的调用方）。
* `ids.reshape(-1)` 对非连续张量会产生拷贝（此处 `candidate_ids` 来自上游 topk，通常是连续的）。

**在途冲突**：🟢 `grep -F "vllm_ascend/models/qwen3_dflash2.py" /tmp/pr_files.tsv` → **0 命中**。

**证据等级**：【源码】。

---

### 候选 9 `attention/dsa_v1.py:530-531,573`：视觉双向 SWA 的 paged 索引构造（模式 A）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/attention/dsa_v1.py:530`（`query_start_loc[req_ids]`）、`:531`（`seq_lens[req_ids]`、`query_lens[req_ids]`）、`:573`（`request_block_tables = block_table[req_ids]`） |
| 现状代码 | `token_offsets = torch.arange(num_tokens, device=...) - query_start_loc[req_ids]` / `positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets` / `request_block_tables = block_table[req_ids]` |
| 函数 | `build_vision_bidirectional_swa_indices`（`:499`） |
| 复现 | `git show origin/main:vllm_ascend/attention/dsa_v1.py \| awk 'NR>=519 && NR<=575'` |

**为什么热（以及为什么不是 decode）**

* 唯一调用点 `dsa_v1.py:1115`，条件 `if has_prefill and max_image_tokens > 0 and mm_ranges:`
  ⇒ **只在“带图 prefill + 使用 DSA 的 VL 模型”时执行**，每步 1 次（每组）。
* 但其中 `block_table[req_ids]` 的输出形状是 `[num_tokens, max_blocks_per_seq]`——**是本清单里数据量最大的单个高级索引**
  （8K 图像 token × 数百列），比“每步省 1 个 kernel”更值得关心的是它的访存。
* `req_ids` 由 `torch.repeat_interleave(..., dtype=torch.long)` 生成 ⇒ **已是 int64，天然满足 index_select**。

**建议改法**：四处改为 `torch.index_select(表, 0, req_ids)`；`block_table` 那一处输出形状与 `[req_ids]` 完全一致，无需 view。

**预期收益**：【推断】4 kernel/步（视觉 prefill 步），外加 `block_table` 大 gather 的访存路径可能改善（未测）。

**改动规模**：~6 行。

**风险**：低（`req_ids` 非负、int64；高级索引与 `index_select` 在 dim 0 上语义一致）。

**在途冲突**：🔴 该文件是**冲突最重**的一个：`#16929`、`#16993`、`#16925`、`#16421`、`#16377`、`#16371`、`#16346`、`#16339`、`#16338`、`#16332`、`#16285` 等 36 个 open PR 命中。
**结论：除非其它候选都被占了，否则不要先做这个。**

**证据等级**：【源码】。

---

### 候选 10 `attention/utils.py:342-349`：逐个请求 `torch.arange` 的 list-comp（模式 B）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/attention/utils.py:340-349`（`filter_chunked_req_indices`） |
| 现状代码 | `offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)` / `filtered_ranges = [torch.arange(offsets[i], offsets[i] + seq_len[i]) for i in range(len(mask_for_non_zero_chunk)) if mask_for_non_zero_chunk[i]]` / `return torch.cat(filtered_ranges)` |
| 复现 | `git show origin/main:vllm_ascend/attention/utils.py \| awk 'NR>=336 && NR<=352'` |

**为什么热**

* 唯一调用点 `vllm_ascend/attention/context_parallel/attention_cp.py:158`（`chunk_seq_mask_filtered_indices=`），
  即 **PCP（context parallel）的 chunked-prefill 元数据构建**，每步 1 次。
* 每个**真正做 chunk prefill 的请求**都会下发 1 个 `torch.arange`（独立 kernel），最后再 1 个 `torch.cat`
  ⇒ 成本随 `num_reqs` 线性增长，而**结果是一个纯前缀和可表达的区间拼接**，完全可以一次算完。

**建议改法**（保持顺序与 dtype 完全一致）

```python
keep = torch.tensor(mask_for_non_zero_chunk, device=seq_len.device)
starts = offsets[keep]
lens = seq_len[keep]
if lens.numel() == 0:
    return torch.empty(0, dtype=torch.long, device=seq_len.device)
total = int(lens.sum())
# 区间内偏移：每个请求从其 start 开始递增
rep_lens = lens.repeat_interleave(lens)          # 0 个 kernel 之外的 1 次 repeat_interleave
base = torch.repeat_interleave(starts, lens)
return base + (torch.arange(total, device=seq_len.device) - torch.repeat_interleave(
    torch.cumsum(lens, 0) - lens, lens))
```
（本质上等价于 `torch.cat([arange(s, s+l) for ...])`；关键是**把 N_req 次 kernel 降为常数次**。）

**预期收益**：【推断】`N_req − 1` 个 `Range` kernel/步（N_req 为真正 chunked-prefill 的请求数）。

**改动规模**：~10 行 + 单测（`tests/ut/attention/test_attention_utils.py:55-70` 已有该函数的两个用例，**直接可扩**）。

**风险**：中。需确认 `seq_len`/`offsets` 是否在设备上（若 `seq_len` 是 CPU 张量，那段代码本来就是 host 侧、优化意义变小）；
空集返回必须保持 `torch.empty(0, dtype=torch.long, ...)` 的 dtype/device（现有单测会验）。

**在途冲突**：🔴 `attention/utils.py` 有 28 个 open PR 命中，其中 `#16915` 明确改了该文件的其他函数。

**证据等级**：【源码】。

---

### 候选 11 `worker/dcp_utils.py:480-495`：`np.array_split` + 逐请求 `np.append`（模式 B，host）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/worker/dcp_utils.py:480-495`（`generate_dcp_mtp_input`） |
| 现状代码 | `req_indices_split = np.array_split(req_indices, cu_num_tokens)[: self.num_reqs]` / `for req_idx in range(self.num_reqs):` … `req_indices_split[req_idx] = np.append(req_indices_split[req_idx], np.repeat(req_indices_split[req_idx][-1], extra_tokens))` … `positions_split[req_idx] = np.append(positions_split[req_idx], np.arange(...))` |
| 复现 | `git show origin/main:vllm_ascend/worker/dcp_utils.py \| awk 'NR>=477 && NR<=496'` |

**为什么热**

* 由 `generate_dcp_mtp_input` 承担 **DCP + MTP 的输入构造**，在 `worker/model_runner_v1.py:1276-1290` 每步调用一次；
* 每个请求 2 次 `np.append` ⇒ **2×num_reqs 次 numpy 重分配 + 拷贝**，且是纯 host 串行（`num_reqs` 可达 128+）；
* 之后 `input_batch.block_table.compute_slot_mapping_draft(...)` 与 `pin_memory().to(device)` 还在同一段里 ⇒ 这里省下的 host 时间**直接落在“host 外露”**上（RFC [77] 关心的那类成本）。

**建议改法**：把两轮 `np.append` 换成一次性构造：
`np.repeat(req_indices, cu_diff)[...]` + `np.concatenate`；位置部分用 `starts = np.repeat(...)`、
`np.arange(total) - np.repeat(prefix_starts, lens)` 一次成型（与候选 10 是同一个数学形态）。

**预期收益**：【推断】host 侧 O(num_reqs) 次分配 → 常数次；量级 10–100 µs/步（取决于 num_reqs）。
对照我们 Engram 路径的经验：host 侧 per-request 工作批量化后 **3.379 ms → 0.058 ms/步**（同机制，不同处）。

**改动规模**：~15 行。

**风险**：中。必须逐位对齐“`cu_num_tokens` 分段 + 空段跳过（`:483-484`）+ 每段补 `extra_tokens` 个 token”的语义；
`positions_split[-1] + 1 ... + extra_tokens` 的取值也要逐点一致。

**在途冲突**：🟡 该文件被 `#16790`、`#16575`、`#16404`、`#14535` 命中（`#16790` 是 v0.27.1rc 的 cherry-pick 系）。
🟢 **补一条对我们有利的实测**：`reports/a2-draft-graph-20260920.md` §3.1 已经量过 `d2h 0.19–3.41 ms`、`route 1.34–2.89 ms` 的 host 账，可直接作为“为什么值得动 host 路径”的证据。

**证据等级**：【源码】。

---

### 候选 12 `ops/fused_moe/router/fused_topk_router.py:67`：VL 哈希路由的 `tid2eid[lookup_ids]`（模式 A）

**位置**

| 项 | 值 |
|---|---|
| 文件:行 | `vllm_ascend/ops/fused_moe/router/fused_topk_router.py:66-68`（`select_deepseek_v4_vision_experts`） |
| 现状代码 | `lookup_ids = torch.where(image_mask, 0, input_ids)` / `text_ids = tid2eid[lookup_ids].to(torch.int64)` / `topk_ids = torch.where(image_mask.unsqueeze(-1), dynamic_ids, text_ids)` |
| 复现 | `git show origin/main:vllm_ascend/ops/fused_moe/router/fused_topk_router.py \| awk 'NR>=60 && NR<=72'` |

**为什么热**

* 该函数在 `_compute_routing` 里由 `if self.bias_vl is not None and input_ids is not None:` 命中（`:182-195`），
  而 `bias_vl` 只要 **`config.vision_n_layers > 0` 就每层都有**（`models/deepseek_v4/model.py:329-331`）
  ⇒ **VL checkpoint 下每个 MoE 层每步 1 次**（text-only 模型完全不进这条分支，这是它比候选 2 受众窄的原因）。
* `tid2eid` 是 `[vocab, top_k]` 的二维表（`models/deepseek_v4/model.py:342-350`），索引是 1-D 的 `lookup_ids`
  ⇒ **可以直接 `index_select`**，是模式 A 的教科书形态。

**建议改法**

```python
text_ids = torch.index_select(tid2eid, 0, lookup_ids).to(torch.int64)
```
（`lookup_ids` 已由 `torch.where(image_mask, 0, input_ids)` 保证非负；`:162` 已保证 int64。
顺带可以省掉 `torch.where` 造 `lookup_ids` 的那一步：把 image 行换成 `clamp` 到 0 只是为了查表，
其查询结果随后被 `torch.where` 丢弃——但用 `index_select` 时仍需保证索引合法，`clamp`/`where` 至少要留一个。）

**预期收益**：【推断】1–2 kernel/层·步（仅 VL）。

**改动规模**：≤10 行。

**风险**：低。注意 `tid2eid` 在该函数入参里已被转成 int32（`:163` 的 `tid2eid_ones`），
`index_select` 的**索引**必须是 int64（`lookup_ids` 满足），**表**的 dtype 不限，无需额外处理。

**在途冲突**：🔴 与候选 2 同一组（`#16993`/`#16925`/`#16689`/`#16192`/`#15740`/`#15363`）。

**证据等级**：【源码】。

---

## 3. Top 3 推荐（为什么是这三个）

### 🥇 1. 候选 1：`ops/causal_conv1d.py` 的 Python 回退（模式 B）

理由（按重要性）：

1. **证据是仓库自述的，不依赖我们的 profile**：`patch/worker/patch_triton.py:332-336` 明写这段回退
   “**syncs per request** and therefore **stalls ACL graph capture at decode-FULL**”。
   这是本清单里唯一“危害由上游代码自己承认”的候选 —— 写 PR 时几乎不需要论证动机。
2. **正好落在 RFC 的 [73] / [77] 两项**（ACLGraph decode 支持 + eager/graph 边界、把 host 同步赶出捕获路径），
   而这两项是我们已经有可复现证据的方向（RFC-16375-CONTRIBUTION §2.4）。
3. **与任何 open PR 都不重叠**（该文件与 `patch_triton.py` 在这 1000 个 open PR 里 0 命中）。
4. 风险可控：它是**参考实现**，`tests/e2e/.../triton/test_causal_conv1d.py` 已有“ref vs kernel”的对照测试，
   等价性有现成判据。

**先做的两件事**（决定它是“硬伤”还是“兜底改进”）：
① 确认目标环境上 `:325` 的 `causal_conv1d_update_npu` 能否 import（`HAS_TRITON` 已由 `patch/worker/__init__.py:22` 保证为真）；
② 确认哪些模型走模块属性调用。若 ① 失败，`update` 路径是**每步打断图捕获**的硬伤；
若 ① 成功，仍有 `:58` 的 `causal_conv1d_fn`（prefill，永远没有 Triton 覆盖）这一条常驻回退 —— 那时本项定位为
“把 prefill 回退的 host 同步去掉”，收益从“图捕获”变成“prefill host 时间”。

### 🥈 2. 候选 2：每层重复的 `input_ids.to(torch.int64)`（【实测】41 次/步）

理由：

1. **收益已经实测过**：P1 逐条点名 41 次/步，是本清单里唯一有“每步次数”硬数字、且**与配置无关**
   （V4/V4.1 的 hash MoE 都走）的候选。
2. **改动极小**：`models/deepseek_v4/model.py` 加 1 行 + 守卫；router 侧不用动（`.to()` 变 no-op）。
3. 语义安全：cast 的位置**本来就在集合通信之前**，上提不改变任何通信 dtype。
4. 唯一缺点是在途 PR 密集（`#16993` 等 16 个 PR 命中 `models/deepseek_v4/model.py`）。
   ⇒ **战术**：等 `#16993`（V4.1 framework 重贴）状态明朗后，一次性给 `models/deepseek_v4/` 与
   `models/deepseek_v41/` 两处都补上，避免 rebase 两次。

### 🥉 3. 候选 3 + 4 + 8：`index_select` 的“零冲突清扫”（同一个 PR 里三处同模式）

理由：

1. 三个文件（`models/glm5next/mtp.py:146`、`spec_decode/utils.py:30-32`、`models/qwen3_dflash2.py:163/171`）
   在这 1000 个 open PR 里 **全部 0 命中** —— 在我们等待 `#16993`/`#16915` 落地期间，这三个可以立刻做完。
2. 它们是**同一个模式、同一句话能讲清**（“把高级索引换成 `index_select`”），reviewer 认知成本最低；
   而且 `spec_decode/mtp.py:37` 与 `models/glm5next/ops/state_ops.py:24` 里**上游自己已经这么写了**，
   我们可以把 PR 定位成“把仓内既有的写法补齐”，而不是引入新手法。
3. 单点收益都小（1–3 kernel/步），所以**必须合并成一个 PR**才有“值得 review”的量；
   如果分开提，很可能被当作噪音。

> **不推荐先做**：候选 9（`dsa_v1.py`，36 个在途 PR）、候选 10（`attention/utils.py`，28 个）、
> 候选 6（EPLB 受众窄、且 `#16871`/`#16800` 正在改附近代码）。它们不是“不值得改”，
> 而是**现在改的 rebase/协调成本高于收益**，建议等对应 PR 落地后再评估。

---

## 4. 已排除项（按任务要求：测试文件、默认关的实验、已被人改）

| 排除对象 | 原因 | 判定依据 |
|---|---|---|
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` 的 `expert_map[topk_ids] != -1` | **我们 0002 已覆盖**（`perf/moe-contiguous-expert-map`） | 分支 `3a0c48c0`；`logs/04-...moe-mask-graph-vs-eager.md` |
| `vllm_ascend/ops/rope_dsv4.py` 的 `full_rope_cos[pos_tensor]` / `expand+gather` | **我们 0003 已覆盖**（`perf/rope-fused-index-select`） | 分支 `d4167f52`；`pr/PR-rope-index-select.md` |
| `tests/**` 下所有 `xxx[ids]` 命中（`tests/ut/attention/test_dsa_v1.py:86` 等） | 测试文件 | 任务硬性要求 |
| `vllm_ascend/ops/triton/**` 里的 `for … in range(...)` | **Trition kernel 内部循环**，不是 host Python 循环 | 例如 `ops/triton/compressor/compressor_triton.py:584-640`、`worker/v2/spec_decode/*/speculator.py:66/99` |
| `vllm_ascend/ops/causal_conv1d.py` 里 `causal_conv1d_ref` 的 `F.conv1d` 参考实现 | 它是**算子的正确性基准**，不是热路径本身 | 只有 `test_causal_conv1d*.py` 把它当 ref |
| `vllm_ascend/_310p/**` 的 `cos[pos_ids]`（`_310p/ops/qwen3vl_310.py:33-34`） | 310P 专用分支；同一个形态在 `models/glm5next/multimodal.py:460-461` 有通用版本 | 若要一起改，只能作为**顺带**（`_310p` 命中 `#16814`） |
| `models/glm5next/multimodal.py:460-461`（mrope `cos[pos_ids]`） | 形态与候选 7/12 相同，但**同一行**正在被 `#16335` 改（`patch_qwen3vl.py:33` / `patch_qwen3_5.py:55` 是 #16335 的改动行） | `gh api .../pulls/16335/files` 实测 diff |
| `ops/gdn.py:612/626/643`、`ops/kimi_kda.py:470-486` 的 `ssm_state[prefill_state_indices]` | 属 **prefill** 路径，且 `#16887`/`#16811` 明确在改这两个文件（`#16887` 标题即 “Skip unused device metadata copies for fused prefill”） | `pr_files.tsv` 命中 `16887/16811/16895` |
| `_310p/fused_moe/token_dispatcher.py:55` 的 `expert_map[topk_ids]` | 310P 专用，且与候选 2 同源手法；可作“顺带” | `git grep` 命中 |
| 任何**已注册 env 门控且默认关**的代码 | 未在本次清单中出现（本清单 12 个候选**全部没有** env 开关，改的就是默认路径） | 逐项检查 `envs.py` 无相关变量 |
| 死代码检查结果 | `build_vision_bidirectional_swa_indices` **不是**死代码（`dsa_v1.py:1115` 有调用）；`filter_chunked_req_indices` 唯一调用点是 `attention_cp.py:158`；`ops/causal_conv1d.py` 的两个函数在 `HAS_TRITON` 下由 `patch_triton.py:57-58` 绑定（`causal_conv1d_fn` 无 Triton 覆盖，`causal_conv1d_update` 可能被 `:329` 覆盖） | `git grep -n` 见各候选“复现”栏；门控见 `patch/worker/__init__.py:22-24` |

---

## 5. 复现命令汇总（只读，全部可离线跑）

```bash
U=~/projects/dsv41/upstream-v41/vllm-ascend-upstream   # origin/main = c173a64a4

# A. 模式 A：高级索引
git -C $U grep -nE '\w+\[[a-z_]*(ids|indices|idx|positions|pos|tokens|topk|slots)[a-z_]*\]' \
  origin/main -- 'vllm_ascend/**/*.py' | grep -v '_310p/'

# B. 模式 B：Python 逐元素循环 / tolist
git -C $U grep -n '\.tolist()' origin/main -- 'vllm_ascend/**/*.py' | grep -v '_310p/'
git -C $U grep -nE '^\s+for [a-z_]+ in (range|enumerate|zip)\(' origin/main -- 'vllm_ascend/**/*.py' | grep -v '_310p/'

# C. 单个候选的上下文（示例：候选 1）
git -C $U show origin/main:vllm_ascend/ops/causal_conv1d.py | awk 'NR>=220 && NR<=300'
git -C $U show origin/main:vllm_ascend/patch/worker/patch_triton.py | awk 'NR>=320 && NR<=338'

# D. 在途冲突（见 §0.3）
grep -F "vllm_ascend/models/glm5next/mtp.py" /tmp/pr_files.tsv | cut -f1 | sort -n | uniq
```

## 6. 本清单的“不承诺”声明

* 所有【推断】收益都**没有**在本机或单卡机上实测（本任务不占卡）；
  它们只说明“能省掉什么算子/同步”，**不能**当作 PR 里的性能数字。
  给上游写 PR 时，必须按 `CI-ANALYSIS.md` §2.4 的要求，用**ACLGraph 口径**（`logs/04-…` 的教训：eager 与 graph 会给出相反符号）重新测量。
* §0.3 的冲突检查**不完备**（GitHub search 上限 1000 + `files(first:100)` 截断）；
  正式提交前应对目标文件重跑一次，尤其是 `#16993`（V4.1 framework 重贴）这类超大 PR。
* P1 的行号口径是**容器内快照**（`/vllm-workspace/vllm-ascend/...`），与本仓 `origin/main` 有 ±2 行的漂移
  （例如 P1 写 `fused_topk_router.py:164`，本仓是 `:162`）。本清单**一律以 `origin/main` 的行号为准**。
