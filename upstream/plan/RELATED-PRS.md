# 已有 PR 与我们工作的关系（2026-09-21 排查）

> 方法：按「我们改的文件 / 我们用的手法 / 我们解决的问题」三个维度搜上游在途 PR，
> 再用 git 把候选分支抓下来**逐个看 diff**（不只信标题）。

---

## 0. 一页总表

| 我们的改动 | 相关上游 PR | 关系 | 影响 |
|---|---|---|---|
| **0003** rope 取表融合 | **vllm-ascend #14428** ✅**已合入**(09-03) | **我们是它的增量**（同一函数的下一次优化） | 🟢 **有利**：明确了 PR 定位 |
| 0001 MoE AllGather | #15043（A2 W4A8→FUSED_MC2）、#14165（A5） | 同文件、**方向相反**、且都已过期 | 🔴 **需在 PR 里正面回应** |
| 0002 expert mask 范围比较 | #14933（同文件不同区域） | **无代码重叠** | 🟢 可安全并行 |
| **admission gate**（vllm core） | **vllm #56455** bounded prefill admission | **同一问题、同期、在途** | 🔴 **直接竞争** |
| Engram host 常驻 | vllm #54129（mmap PLE）、#16689（VMM） | 同类问题、不同解法 | 🟡 需对齐设计 |
| 调度类改动 | vllm-ascend #11352（ShortRequestFirst） | **已关闭未合并** | ⚠️ **先例：这类改动容易被关** |

---

## 1. 🟢 #14428 —— 我们 0003 的前置，**已合入**

**[Performance] Fuse index+copy into single `out=` gather in `get_cos_and_sin_dsa`**
作者 `ivyilike`｜**merged 2026-09-03**｜+20/−7｜1 文件

| 项 | #14428（已合） | **我们的 0003** |
|---|---|---|
| 手法 | 把"index-then-copy"融成 `torch.gather(..., out=...)` | 用 **`index_select`** 取代 `expand + cast + gather` |
| 声称 | **4 ops → 2 ops**，814 µs → 701 µs（−14%） | **6 kernel → 2**，−0.45 ~ −0.62 ms/pass |
| 是否消除 `expand` | **没有** —— 它仍构造 4-D `gather_idx`（`BroadcastTo`） | **是** —— 改成 1-D 索引，去掉 BroadcastTo 与 Cast |

**实测确认**：`2d5e56e06` 这个 commit **同时在我们基线和官方 main 里**（`merge-base --is-ancestor` 二者皆为真）。
两边代码里都还是：

```python
pos_tensor.to(torch.long).reshape(-1, 1, 1, 1).expand(num_tokens, 1, 1, full_rope_cos.size(-1))
torch.gather(full_rope_cos, 0, gather_idx, out=buf_cos[:num_tokens])
```

⇒ **我们是"在已合入 PR 之上再省两个 kernel"的天然 follow-up。**
这是最好写的一类 PR：有明确的前序、有清晰的增量、reviewer 熟悉上下文。

> **战术建议**：PR 描述里直接引 #14428，并说明"它保留了 `expand` 构造的 4-D 索引，
> 我们用 `index_select` 消掉它"。可以 @ivyilike 一起 review。

---

## 2. 🔴 #15043 —— 和我们**方向相反**的 A2+W4A8 PR

**Support W4A8 fused MoE operator (`dispatch_ffn_combine_w4_a8`) on Ascend 910B**
作者 `Wfd567`｜open｜创建 08-26、更新 09-03｜+79/−49｜6 文件

它的主张：**910B（A2）上 W4A8 应该走 `FUSED_MC2`**（此前 910B 上 W4A8 会退化成非融合路径）。
我们的 0001 的主张：**强制走 `ALLGATHER`**，因为 A3 上 MC2 把 8 个 token 切成每 rank 1 个、
标量开销摊不开。

**两者条件不同、结论相反** —— 这是我们必须主动交代的事。

**而且它已经严重过期**（实测）：

```
PR 分支落后 main：579 个提交
它改的函数 _select_a2_moe_comm_method —— 官方 main 里**已经不存在了**（git grep 实测为空）
官方 main 现在只有三个：_select_capacity_and_expert_density_moe_comm_method
                       _select_fused_or_capacity_moe_comm_method   ← 我们的 0001 改这个
                       _select_capacity_and_world_size_moe_comm_method
```

⇒ 它**不可能按现状合入**（要重写）。但它的**思路**还在（W4A8 on A2 值得融合），
所以我们写 0001 时要：

1. 说明我们的收益是在 **A3、TP=EP、8 token/batch** 这个特定条件下测的；
2. 说明我们的门控**默认关闭**、可随时关掉对比；
3. **不要**宣称"AllGather 普遍更好" —— 那会直接撞上 #15043 的主张。

---

## 3. 🟢 #14933 —— 同文件但不同区域，无冲突

**[Performance][MoE] Reuse MC2 expert scales for combine**｜作者 **GDzhu01**｜open｜+30/−1｜2 文件

抓下来看 diff：它改的是 `token_dispatcher.py` 的
**第 29 行（import）** 和 **`TokenDispatcherWithMC2` 内部（第 132/243 行）**；
我们的 0002 改的是 **`TokenDispatcherWithAllGather` 里的 expert_map 掩码（约第 442 行）**。

⇒ **同一文件、不同类、不同区域**，`git apply` 的 3 行上下文不会重叠。
唯一要注意的是：两边都在文件头部附近动 import，rebase 时留意一下顺序即可。

> 顺带一个信号：**#14933 的作者是 GDzhu01**，他也在做 `[Performance][MoE]` 方向的优化、
> 而且 PR 带了**单元测试**（`tests/ut/ops/a2/test_token_dispatcher.py`）。
> 我们写 0002 的测试时可以参照这个文件的结构。

---

## 4. 🔴 vllm #56455 —— 与我们的 admission gate **正面重叠**

**[Perf] Support bounded prefill admission for scheduler**｜作者 `shan-chen-feng`
open｜创建 **09-11**｜+493/−22｜9 文件

| | **#56455**（vllm core） | **我们的 admission gate** |
|---|---|---|
| 目标 | 提高 **decode-only batch 的占比**（走完整 CUDA graph） | **同一个 step 只放 prefill 或只放 decode**，防 prefill 饿死 decode |
| 机制 | **有界波浪**：decode 满则只跑 decode；decode 降下来再放**一批** prefill | **严格门控**：一步一个 prefill 或全是 decode |
| 触发场景 | **离线数据集服务 + 高并发 + 持续 backlog** | **长上下文 agent 场景**（实测 prefill-only step 连续上百步） |
| 默认 | 关（配置开启） | 关（env 开启） |
| 状态 | open，**无 conflicts**（09-17 还在更新） | 未提 |

**两者动机高度重合**（都引用 decode-only batch 的价值），但**策略不同**：
它是"有界波浪"，我们是"严格互斥"。**必须先读透 #56455 再决定我们的定位**：

* 若 #56455 能覆盖我们的场景 ⇒ 我们的 PR 应改为「**补充它没覆盖的长上下文饿死场景**」，
  甚至直接给 #56455 提改进；
* 若不能 ⇒ 在 PR 里显式对比两者（附我们的 prefill-only step 数百步的日志证据）。

> ⚠️ **绝不要**在不知道 #56455 存在的情况下提一个"新调度策略"PR —— 那是最容易被
> maintainer 一句 "see #56455" 关掉的情形。

---

## 5. 🟡 与 Engram/host 表相关的两个

| PR | 内容 | 与我们的关系 |
|---|---|---|
| **vllm #54129** "Support disk-backed (mmap) PLE table for Qwen3.8-Flash-Next" | 47.68 GiB FP8 PLE 表用 **read-only mmap** 读，靠 page cache 回收；**gather 后拷进 GPU 缓冲** | 同类问题（大表不常驻显存），但它走 **mmap + H2D 拷贝**，我们是 **host_register + 设备直读 + 入图**；且它针对 Qwen3.8 不是 V4.1。**可作对照** |
| **vllm-ascend #16689** VMM Engram | `aclrtHostRegisterV2` + 设备地址直读 | **同机制**，见 `PERF-COMPARE.md` §4 |

> 注意 vllm core 里 **"PLE"** 就是昇腾侧说的 **Engram** 那一类（per-layer embedding）表的通用叫法。
> 搜相关工作时两个词都要搜。

---

## 6. ⚠️ 一个先例：#11352 被关闭

**[Feature][Scheduler] Add ShortRequestFirst scheduling**｜作者 `immengzi`
**closed, 未合并**｜+1739/−7｜15 文件｜创建 07-02、关闭 07-07（**5 天**）

它解决的问题**和我们一样**（"long prefill at the front can delay shorter prefills"），
机制是队列策略（三条 lane + 年龄兜底），默认关、带完整配置与测试。

**但它在 5 天内被关掉了** —— 最后的可见动作是
`github-actions[bot]: This pull request has conflicts, please resolve those before we can evaluate`，
之后没有再更新就被关闭。

⇒ 对我们有两条直接教训：

1. **调度类改动在没有先建立共识时，很容易无声无息地死掉**（有冲突 + 无 reviewer 关注）；
2. **必须先开 issue 讨论**，让 maintainer 表态"这个方向我们要"，再提 PR。

---

## 7. 结论与调整建议

### 7.1 对原规划的修正

| 原规划 | 修正后 |
|---|---|
| A-2（0003 rope）"上游缺" | ❌ 措辞不准：**#14428 已合入**，我们是**它的 follow-up**（定位更有利） |
| A-3（0001 AllGather）"直接推" | ⚠️ 加一步：**先看 #15043 的思路**，PR 里限定条件、避免宣称普适 |
| C-1（admission gate）"先开 issue" | 🔴 **升级为最高优先**：先读透 **vllm #56455**，否则会被"see #56455"关掉 |
| 优先级排序 | **A-1（0002）仍第一**（同文件无重叠、上游仍缺）；**A-2（0003）升为并列第一**（有已合 PR 背书） |

### 7.2 新的建议顺序

```
1.  A-2（0003 rope）—— 有 #14428 背书，最好写；引 #14428、@ivyilike
1'. A-1（0002 expert mask）—— 上游仍缺、通用、无重叠
2.  读透 vllm #56455 —— 决定 admission gate 是「独立提」还是「并入它」
3.  A-3（0001 AllGather）—— 先与 #15043 的立场对齐，别宣称普适
4.  Engram 系列 —— 等 #16925
```

### 7.3 一个额外的判断

上游在这几个方向**都已经有人在动**了（MoE 通信、调度、大表 offload）。
这说明**我们的优化方向选得对**（都和社区关注点重合），
但也意味着**"先发优势"在收窄** —— 越早提，越容易占据"这个问题由我提出并解决"的位置。

---

## 附：抓取过的分支（本地已有，可离线查）

```
refs/remotes/pr/14933   43dde6380   [Performance][MoE] Reuse MC2 expert scales
refs/remotes/pr/15043   52103bde4   W4A8 fused MoE on 910B（落后 main 579 个提交）
refs/remotes/pr/14165   9956227c0   [WIP] all2all→allgather for a5（08-13 后未动）
refs/remotes/pr/16423   e67ab6495   V4.1 serving support（GDzhu01）
refs/remotes/pr/16925   382dc9289   V4.1 + Engram host offload（重新合入）
refs/remotes/gdz/main   6bb7aeecf   GDzhu01 开发仓 main
```
