# 上游进度复查（第二轮）—— 2026-09-21 13:0x

> 触发：A3 已就绪、准备把材料做完善；顺手复查**我们引用的每个上游对象**是否发生了变化。
> 上一轮复查是 [`08-20260921-upstream-recheck.md`](08-20260921-upstream-recheck.md)（当天凌晨）。
> 结论用 `gh` 直读 API，命令见文末；所有时间戳为 GitHub 的 UTC 时间。

---

## 1. 一句话结论

**三个变化会影响我们的材料优先级**：

1. **#16925 现在 `MERGEABLE`**（73 文件 / +10554 −665，最新提交 09-21 02:41）—— 我们 §6 的 Engram 边界数据（针对 #16828）**正是他们合入后要面对的**，时机从"也许有用"变成"合入即需要"。
2. **#16285 拿到维护者第二句更硬的回复**（09-21 03:46）—— 我们撤回 draft-graph 主张的改向**被再次确认**。
3. **#16828 的官方方向已经写明是 "host-registered UVA path"** —— 与我们 [46]/[47] 的测量口径**同一条路**，可以直接对话。

---

## 2. 逐对象状态（2026-09-21 13:0x 实测）

| 对象 | 类型 | 状态 | 最后活动（UTC） | 对我们的意义 |
|---|---|---|---|---|
| **#16285** DSpark ACLGraph and DSA_CP | PR | **OPEN**，`CONFLICTING` | 09-21 03:46（维护者评论） | 与我们 `rope_dsv4.py` 撞同一函数；我们的合并分支 `perf/rope-index-select-on-16285` = `0121f203` 仍然有效 |
| **#16925** V4.1 framework support + Engram host offload | PR | **OPEN，`MERGEABLE`** | 09-21 02:41（新提交） | 我们 §3.3a 的 host-registration A/B、§6 的边界数据、[47] 五项测量**都挂在它这条线上** |
| **#16828** [Bug][A3] Engram host offload | issue | **OPEN**，2 评论 | 09-18 11:31 | 官方回复确认走 **host-registered UVA path**（原文见 §3） |
| **#16993** Reapply DeepSeek V4.1 framework support | PR | OPEN | 09-20 | 与 #16925 同题，注意别把两者混为一谈 |
| **#16689** host-backed VMM Engram | PR | OPEN | 09-16 | 另一条 host-offload 实现（VMM 路线），与 #16925 的 UVA 路线是**两条不同的路** |
| **#14428** Fuse index+copy into single out= gather | PR | **MERGED**（09-03） | 09-08 | 我们 rope PR 的**前置**，草稿里已按 follow-up 定位 |
| **#17004** Main2Main vLLM → v0.29.0 | PR | MERGED（09-20） | 09-20 | 我们 fork 的 `main` 基准；`perf/rope-fused-index-select` 基于 `c173a64a`，需要时再 rebase |
| **RFC #16375** DeepSeek V4.1 Roadmap | issue | **OPEN，0 评论** | 09-11 10:00 | **50 条目仍无人认领** —— 我们"帮他们打勾"的窗口仍然开着 |

---

## 3. 两条原文（引到时逐字用，别转述）

**#16285 / drslark / 2026-09-21 03:46**（维护者，第三次表态）：

> *"If DSpark graph support is required, please use v2. Alternatively, you may implement and maintain the necessary adaptations in your own private repository."*

⇒ 与 [`18-20260921-track-c-reaim.md`](18-20260921-track-c-reaim.md) 记录的"v1 不接受图模式"**结论一致、语气更硬**。
我们的应对不变：**撤回 draft-graph 主张，改交 sync 账**，并只在 [73]/[77] 名下谈"边界定义 + 我们的量化证据"。

**#16828 / QwertyJack / 2026-09-18 11:31**（官方对 offload 路线的说明）：

> *"Our current direction is the host-registered UVA path, chosen for overall performance and for staying closer to the upstream vLLM route. We are still working on that path. We believe the error disappears with PR #16544."*

⇒ 我们量的正是**同一条路**（`aclrtHostRegister` + `host_mem_pool` 判据 + 设备直接索引）。
注意他说的是 **#16544**（已被 revert、正由 #16925/#16993 重新合入），所以"错误是否消失"这件事**还没被验证** —— 我们的 §6 边界数据应当保持"可检验假设"的写法，不写成结论。

---

## 4. 对材料的具体影响（要不要改，怎么改）

| 材料 | 影响 | 动作 |
|---|---|---|
| `pr/PR-rope-index-select.md` | #16285 仍 OPEN 且 CONFLICTING；#14428 早已合入 | **不改定位**（follow-up + 与 #16285 的组合分支已备） |
| `pr/RFC-16375-CONTRIBUTION.md` §6 | 官方已表态走 UVA 路线、且认为 #16544 能修 | 保持"可检验假设 + 我们的数据"，**不要**写成"你们有 bug"；可在 §6.1 补一句"该路线已由维护者在 #16828 确认" |
| `pr/RFC-comment.md` [46]/[47] 段 | 与官方路线同向，可以更直白地说"我们做的是你们选的那条路" | 保留 `not measured` 的诚实清单 |
| `pr/issue-track-C.md` | 维护者对 v1 图模式的第三次表态 | 已改向；无需再动（但**不要**在 RFC 评论里再提 draft-graph 收益） |
| 整体节奏 | #16925 已 mergeable | RFC 评论里可加一句：*"#16925 is now mergeable; the boundary data in §6 is what we would attach to [47]"* —— **只在用户授权发帖后才用** |

---

## 5. 复查命令（可复跑）

```bash
cd ~/projects/dsv41/upstream-v41
for n in 16285 16925 16993 16689 14428 17004; do
  printf "%-7s " "#$n"
  gh pr view $n --repo vllm-project/vllm-ascend \
    --json number,title,state,updatedAt,mergedAt,author,mergeable \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['state'],d['mergeable'],d['updatedAt'][:10],d['title'][:60])"
done
gh issue view 16828   --repo vllm-project/vllm-ascend --json state,updatedAt,comments | head -c 400
gh issue view 16375   --repo vllm-project/vllm-ascend --json state,comments | head -c 200   # 期望仍是 0 评论
```

---

## 6. 还缺什么（诚实）

* 本复查只看了**状态与最近评论**，没有逐行读 #16925 新增的 10.5k 行 diff —— 若要把 §6 的假设写得更精确，需要有人在它合入前扫一遍 `engram/common.py` 与 host-register 调用点。
* `mergeable` 是 GitHub 的即时判断，会随 main 变动而变（#16925 在 09-20 还显示 CONFLICTING）。
* 没查 A5 / 其他模型线的 PR —— 与本批材料无关。
