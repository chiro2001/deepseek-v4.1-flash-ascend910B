# 回合（upstreaming）规划 —— 怎么把我们的优化提交到 vllm-ascend

> 2026-09-21 制定。**上游规范部分是实测**（读 `origin/main` 的文档、模板、最近 400 个提交），
> **拆分依据是我们自己补丁的实际可应用性测试**。

---

## 第 0 步：上游的规矩（实测，必须先满足）

> ⚠️ **先读 [`RELATED-PRS.md`](RELATED-PRS.md)** —— 上游已有 PR 与我们的重叠情况，
> 直接决定了两处定位修正：
> **0003 是已合入 PR #14428 的 follow-up**（对我们有利）；
> **vllm 的 admission gate 与在途 #56455 正面重叠**（必须先读透它）。

> **环境已就绪（2026-09-21）**：fork = `chiro2001/vllm-ascend`，
> 工作副本 = `upstream-v41/vllm-ascend-fork/`（`origin`=fork / `upstream`=官方），
> main 与上游同步在 `c173a64a`，已干跑验证可推送。详见 `README.md` 的「fork 与工作副本」。

### 0.1 硬性门槛

| 项 | 要求 | 出处 | 我们的现状 |
|---|---|---|---|
| **DCO** | 每个 commit 必须有 `Signed-off-by:`（`git commit -s`） | `docs/.../contribution/index.md`「DCO and Signed-off-by」 | ❌ **11 个补丁全部没有** |
| **标题** | `[Category]` / `[Category][SubCategory]` 前缀 | 同上「PR Title and Classification」 | ❌ 我们用的是 conventional commits（`perf(moe): …`） |
| **PR 描述** | 三段式模板：What / user-facing change / how tested | `.github/PULL_REQUEST_TEMPLATE.md` | ⚠️ 需按模板重写 |
| **测试** | bot 明确要求「Every PR should include unit tests and end-to-end tests」 | PR 自动回复 | ❌ **我们一个测试都没带** |
| **语言** | 英文 | 全仓惯例 | ❌ 补丁正文是中文（0002 有 20 行中文） |

### 0.2 标题前缀的实际用法（文档 vs 现实）

文档列了 12 个前缀（`[Attention] [Communicator] [ModelRunner] [Platform] [Worker] [Core] [Kernel] [BugFix] [Doc] [Test] [CI] [Misc]`），
**但最近 400 个已合并提交的实际分布是**：

```
 94 [BugFix]     78 [Feature]    55 [CI]       38 [Doc]
 36 [Performance]    ← 文档里没有，但实际高频使用
 29 [Test]       14 [Refactor]   11 [Bugfix]    9 [Misc]
  7 [Ops]         6 [Revert]      2 [Attention]
```

⇒ **`[Performance]` 是可以用的**（36 次），性能类 PR 就写 `[Performance][MoE] …` 这种。

### 0.3 粒度：上游偏爱**小 PR**

最近 30 个已合并 PR 的规模（实测）：

```
中位数 ≈ 4 文件 / 约 150 行
典型:   1 文件/1 行 · 2 文件/76 行 · 5 文件/98 行 · 8 文件/92 行
大块头（少数）: 104 文件（Main2Main 升级）· 96 文件（回退）· 48 文件（算子隔离）
```

bot 的原话：**「A PR should do only one thing, smaller PRs enable faster reviews.」**

⇒ 我们 **0007（+1284 行）**和 **0009（+1626 行）**明显超出常规，**需要在内部再拆**。

### 0.4 性能类 PR 的额外路径

仓库有专门的 issue 模板 `700-performance-discussion.yml`（标题 `[Performance]: `，标签 `performance`），
要求提供「detailed description of performance comparison」，并建议用
`vllm/benchmarks/` 里的脚本。

⇒ **性能优化建议先开 `[Performance]:` issue 建立共识，再提 PR**，这样 reviewer 有上下文、
也避免"这个优化我们不需要"的来回。

---

## 第 1 步：我们的 11 个补丁，按"能否独立推进"分三条轨道

### 依赖关系的实测结论

把每个补丁**单独**打到我们自己的基线 `46856f89e` 上（`git apply --check`）：

```
0001 ✅  0002 ✅  0003 ✅  0004 ✅  0005 ✅  0006 ✅  0007 ✅  0008 ✅
0009 ❌ 需要前置（= 0007）
0010 ❌ 需要前置（= 0009）
```

⇒ 只有 **0007 → 0009 → 0010** 是一条硬依赖链，其余 8 个两两独立。

### 轨道划分

| 轨道 | 含义 | 补丁 |
|---|---|---|
| **🟢 轨道 A：不依赖 V4.1 集成，可立即推** | 改的文件官方 main 就有，且上游**尚未**实现 | 0002、0003、0001、0005 |
| **🟡 轨道 B：需要 V4.1 先落地** | 改 `models/deepseek_v41/*`，官方仓没有该目录 | 0004、0006、0007、0008、0009、0010 |
| **🔵 轨道 C：独立仓** | 不在 vllm-ascend 里 | vllm 的 admission gate、msmodelslim 的两个 |

---

## 第 2 步：🟢 轨道 A —— 立即可推（建议从这里开始）

### A-1 `[Performance][MoE] Avoid Index+IndexCheck kernels when expert_map is a contiguous range` ← **首推**

| 项 | 内容 |
|---|---|
| 来源补丁 | `0002-perf-moe-range-compare-expert-mask` |
| 规模 | **1 文件 / +66 行**（改既有行 2 行）—— 完全符合上游粒度 |
| 对官方 main | ✅ **`git apply --check` 干净**（实测 `c173a64a`） |
| 上游现状 | **仍是旧写法** `mask = expert_map[topk_ids] != -1`（`token_dispatcher.py:394`）⇒ **没重复** |
| 收益 | **−0.51 ms**（省掉 aclnnIndex 的 Index + IndexCheck 两个大 kernel），GSM8K 100/100、Vision 23/23 |
| 为什么先推 | 通用 MoE 路径（**所有昇腾 MoE 模型受益**，不只 V4.1）+ 干净可应用 + 上游仍缺 + 改动小 |
| 需要补 | Signed-off-by、英文 commit message、**单元测试** |

> ⚠️ 这个补丁的**安全设计**是它的卖点，写 PR 时要突出：
> ① 只在 `expert_map` 确实是「本地专家连成一段」时才走快路径（**运行时内容校验**，不是假设）；
> ② **EPLB 开启时自动退回**原路径（EPLB 会重排 expert_map）；
> ③ 校验结果按 `(first, last, numel)` 缓存，**不进每步热路径**；
> ④ 掩码本身保留（`-1` 会让 unpermute 读到未写入的行，靠 0 权重压掉）。

### A-2 `[Performance][Attention] Fuse cos/sin table index selection in rope_dsv4`

| 项 | 内容 |
|---|---|
| 来源 | `0003-perf-rope-fuse-cos-sin-table-index-selection` |
| 规模 | **1 文件 / +52 行**（改既有行 7 行） |
| 对官方 main | ✅ **干净** |
| 收益 | **−0.45 ~ −0.62 ms/pass**，取表链 6 kernel → 2，**数值逐位等价** |
| 注意 | 改的是 `ops/rope_dsv4.py` ⇒ 影响 **DeepSeek V4/V4.1 家族**（官方 main 的 `models/deepseek_v4/` 也 import 它），标题用 `[Performance][Attention]` |

### A-3 `[Performance][MoE] Dispatch/combine over AllGather when TP=EP`

| 项 | 内容 |
|---|---|
| 来源 | `0001` |
| 规模 | **1 文件 / +8 行** —— 极小 |
| 对官方 main | ❌ 冲突（`ascend_forward_context.py` 已分叉） |
| 做法 | **不带 patch，按思路重写**（8 行而已）：在 `_select_fused_or_capacity_moe_comm_method` 里加一个门控早退 |
| 收益 | 128K **−4.25 ms**、32K −1.35、8K −1.23；**KV 池 3.39M → 4.16M（+22%）** |
| 注意 | 这个改的是**通用通信选择逻辑**，标题 `[Performance][Communicator]`；但收益数据是在 V4.1 上测的，PR 里要说明 |

### A-4 `[Performance][Attention] 2D wo_a matmul and dummy-shape guard`

| 项 | 内容 |
|---|---|
| 来源 | `0005` |
| 规模 | **1 文件 / +62 行**（改既有行 1 行——只是局部变量替换） |
| 对官方 main | ❌ 冲突（`dsa_v1.py` 已分叉） |
| 收益 | **−0.31 ~ −0.76 ms** |
| 做法 | 按思路重写，或 rebase 后重生成 patch |

---

## 第 3 步：🟡 轨道 B —— 依赖 V4.1 集成，**当前不要动**

### 为什么现在不能推

1. **落点不存在**：这 6 个补丁改 `vllm_ascend/models/deepseek_v41/*`，而官方 main **没有这个目录**
   （#16544 合入 7 小时后被 #16905 回退）；
2. **结构不匹配**：上游采纳的实现用的是 **`engram/` 子包**（`__init__.py` / `common.py` / `npu.py`），
   我们是**扁平三文件**（`engram_gate.py` / `engram_hash.py` / `engram_hbm.py`）⇒ **patch 套不上，必须改写**；
3. **机制也不同**：他们用 `aclrtHostRegisterV2(MAPPED|PINNED)` + `aclrtHostGetDevicePointer`，
   我们用 `acl.rt.host_register(ACL_HOST_REGISTER_MAPPED)`；
4. **PR 会撞车**：#16925（重新合入）与 #16689（VMM 后续）都在改同一片区域，现在插进去只会制造冲突。

### 触发条件（满足其一再启动）

- ✅ **#16925 合入 main** ⇒ 我们基于新结构改写
- ✅ **#16925 明确不会被合**（例如改走 #16423）⇒ 我们跟 #16423 的结构
- ⏳ 在等待期间：**只做"发评论提供证据"这件事**（见第 5 步）

### 到时候的拆分建议（仍按"小 PR"原则）

我们的 6 个补丁**不要合并成一个大 PR**，按下面拆（每个都对应上游 `engram/` 子包里的具体位置）：

| 建议 PR | 来源 | 内容 | 预估规模 |
|---|---|---|---|
| B-1 | 0006 | 分块 gate，去掉固定 2048 行 padding | ~200 行 |
| B-2 | 0007 **拆两半** | (a) host 常驻表加载；(b) local-owner 快路径（省 metadata all_gather + ids all_to_all） | 各 ~600 行 |
| B-3 | 0008 | hash/plan 的 numba JIT（含 sidecar `engram_jit_kernel.py`） | ~560 行 |
| B-4 | 0009 **拆三半** | (a) 能力探测 + host 映射表；(b) 每 batch shape 一张 ACLGraph；(c) model 接线 | 各 ~500-600 行 |
| B-5 | 0010 | A3 开 / A2 关的默认口径（`host_mem_pool` 判据） | ~160 行 |

> **优先级**：B-3（numba JIT，**上游完全没有**）和 B-5（**A2/A3 机型判据，上游也没有**）
> 是最有"上游没有、我们有"价值的两个。B-4 的 device-index 入图虽然最亮眼，
> 但和 #16689（VMM）是**竞争关系**，需要先对齐设计再推。

---

## 第 4 步：🔵 轨道 C —— 独立仓

### C-1 vLLM core：admission gate

| 项 | 内容 |
|---|---|
| 来源 | `patches/vllm/0001` |
| 规模 | **1 文件 / +282 行** |
| 上游现状 | `vllm/v1/core/sched/scheduler.py` **完全没有**类似机制（grep 实测无命中） |
| 为什么值得推 | 长上下文场景 **prefill 饿死 decode** 是通用问题（不只是 V4.1） |
| 前置 | vLLM core 的 PR 流程通常**要先开 issue/RFC**拿到 maintainer 认可；且要跑他们的 benchmark |
| 需要补 | Signed-off-by、英文、**调度器单测**、完整的防死锁说明（我们已有 `_gate_force_decode_steps` 这类兜底） |

### C-2 msmodelslim：W4A8 配方 + hiaux 变体

**建议先不动**。理由：RFC #16375 里 **A2/A3 的目标量化是 W8A8、A5 才是 W4A8**；
我们推 W4A8 配方会直接撞上他们的硬件矩阵规划（见 `PERF-COMPARE.md` §1、`README.md` §1.5.5）。
真要推，先发 `[Performance]:` issue 讨论"W4A8 要不要在 A2/A3 支持"。

---

## 第 5 步：等待期该做的事（不需要 rebase，收益最快）

### 5.1 去 #16925 / #16689 / #16828 下提供证据

我们手上有**别人没有的实测数据**，正好回答他们悬着的问题：

| 我们的证据 | 对应他们的问题 |
|---|---|
| **A2 上 `host_register` 返回 `207001`**，判据是驱动 `host_mem_pool` 能力（`PCI 19e5:d802` = 910B3 = PCIe ⇒ 0；`d803` = 910C = HCCS ⇒ 1） | #16828 问「two-node 的 offload 有没有支持边界」；#16925 **完全没有机型区分** |
| **A3 上 device-index 路径单流 29.5 → 28.4 ms/step**，同步 host 时间 3.379 → **0.058 ms** | #16828 的「decode 回退 −7~−30%」——我们的路径是**净收益** |
| **host 表必须整表注册**，分片注册要付 a2a+bcast ≈ 0.5 ms 的通信税，净收益上限只有 ~6% | #16689 说 VMM「bypasses owner-ID AllToAll, row-return AllToAll」——我们量过这笔账 |
| **pinned 池会耗尽**（我们踩过） | 他们在 #16544 里用的正是 pinned 暂存 |
| 206 GiB 表 + **C64 全成功**（无 507011） | #16828 的 C24–32 崩溃 |

> 发评论的**口径纪律**：只讲事实与数字，附复现条件（机型/驱动/CANN/表大小），
> **不要**说"我们的方案更好"——让数据说话。同时**不要**贴我们的发布包链接当广告。

### 5.2 同时准备轨道 A 的 PR

轨道 A 与 V4.1 大战**完全解耦**，可以并行推进。建议本周就做 A-1（0002）。

---

## 第 6 步：执行顺序（建议）

```
第 1 周  ├─ 发评论到 #16828 / #16925（提供 A2 失败证据 + host_mem_pool 判据）
        ├─ 开 [Performance]: issue（MoE expert mask 范围比较）
        └─ 把 0002 改造成 PR：英文 + Signed-off-by + 单测 + rebase 到官方 main

第 2 周  ├─ 提 A-1 PR（0002）
        ├─ 改造 A-2（0003）与 A-3（0001 重写）
        └─ 视 #16925 进展决定轨道 B 是否启动

第 3 周+ ├─ A-2 / A-3 / A-4
        └─ C-1（vllm admission gate）先开 issue

待定    └─ 轨道 B（等 #16925 落地）+ C-2（等 W4A8 路线对齐）
```

**为什么 A-1 排第一**：它是唯一同时满足「对官方 main 干净可应用 + 上游仍缺 + 通用（非 V4.1 专属）+
带安全校验设计 + 改动只有 66 行」的补丁。**用小 PR 建立 reviewer 信任，再推大的**，
这在开源协作里比一上来就提 1600 行的 Engram 补丁有效得多。

---

## 附：每个 PR 的 checklist（模板）

```markdown
标题:  [Performance][MoE] Avoid Index+IndexCheck when expert_map is a contiguous range

### What this PR does / why we need it?
<英文，说明改了什么、为什么>
Fixes #<对应的 [Performance]: issue 号>

### Does this PR introduce _any_ user-facing change?
否（新增 env 门控 V41_MOE_MASK_RANGE，默认关闭）

### How was this patch tested?
<单元测试：新增/已有的测试名>
<端到端：机型、CANN/驱动版本、模型、并发、收益数字>

Signed-off-by: <你的真实姓名> <邮箱>
```

**提交前自检**：
- [ ] `git commit -s` 带了 Signed-off-by
- [ ] 标题前缀符合 `[Category][SubCategory]`
- [ ] 正文英文，PR 描述按三段式模板
- [ ] 带了单元测试（`tests/ut/...`）
- [ ] 门控默认关闭（`os.environ.get(..., "0")`），开了才有新行为
- [ ] `bash format.sh` 通过（lint/pre-commit）
- [ ] 性能数字有对照（before/after、口径、机型）
