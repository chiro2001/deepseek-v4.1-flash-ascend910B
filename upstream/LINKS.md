# 链接列表 —— vllm-ascend 上游合入材料（2026-09-21）

> 纯链接清单，供逐个点开审阅。背景与结论见 [README.md](README.md)。
> **全部是草稿，一份都没提交上游。**

---

## A. 三条"随时可发"的候选（等你授权）

| # | 材料 | 链接 |
|---|---|---|
| ① | **PR：RoPE `index_select` 融合**（正文可粘贴） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/PR-rope-index-select.md |
| ① | ↳ 对应分支（可直接开 PR） | https://github.com/chiro2001/vllm-ascend/compare/main...perf/rope-fused-index-select |
| ② | **PR：MoE mask 改范围比较**（正文可粘贴） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/PR-moe-mask-range.md |
| ② | ↳ 对应分支（可直接开 PR） | https://github.com/chiro2001/vllm-ascend/compare/main...perf/moe-contiguous-expert-map |
| ③ | **issue：上游死 import（cc1d）** | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/issue-draft-cc1d-dead-import.md |

---

## B. 大体量材料（RFC #16375 相关）

| 材料 | 链接 |
|---|---|
| **RFC #16375 贡献报告**（主文档，约 700 行） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/RFC-16375-CONTRIBUTION.md |
| RFC 评论草稿（拟发在 #16375 下） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/RFC-comment.md |
| Issue 草稿 A —— Engram host 路径 | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/issue-track-A.md |
| Issue 草稿 B —— MoE 通信与掩码 | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/issue-track-B.md |
| Issue 草稿 C —— 图执行边界（已按维护者表态改向） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/issue-track-C.md |
| Track D —— host 表支持矩阵（纯文档贡献） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/TRACK-D-engram-host-table-support.md |
| #16285 交互分析（与在途 PR 的合并方案） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/pr/PR16285-ROPE-OVERLAP-ANALYSIS.md |
| ↳ 与 #16285 的组合分支 | https://github.com/chiro2001/vllm-ascend/compare/main...perf/rope-index-select-on-16285 |

---

## C. 本轮补的数据（2026-09-21 下午，A3 空闲芯片上跑）

| # | 主题 | 链接 |
|---|---|---|
| 35 | RoPE 边界矩阵（RFC [91]）——32/32 + 5/5 | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/35-rope-edge-cases.md |
| 36 | Engram gate 的 control arm（RFC [97]） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/36-engram-gate-control-arm.md |
| 37 | ngram JIT + host-register 双 API A/B | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/37-ngram-and-hostreg-ab.md |
| 38 | host DRAM 带宽 / 并发 / NUMA（RFC [47]） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/38-host-dram-bandwidth.md |
| 39 | 上游进度复查（#16925 mergeable 等） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/39-upstream-recheck-2.md |

---

## D. 判断"数据够不够"用

| 材料 | 链接 |
|---|---|
| **数据缺口总表**（13 项逐条：补了 / 补不了 / 为何） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/DATA-GAPS.md |
| 审阅入口（背景 + 结论 + 诚实清单） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/README.md |
| 交接快照（含本轮补测小结 §3.5） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/HANDOVER.md |
| 日志索引（28 份） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/logs/README.md |

---

## E. 策略与规划（为什么这么写）

| 材料 | 链接 |
|---|---|
| 规划评审：怎么真正影响他们的开发计划 | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/plan/PLAN-REVIEW.md |
| 在途 PR 与我们的重叠（#16285 会撞） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/plan/RELATED-PRS.md |
| CI 剖析（功能自动、性能不自带） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/plan/CI-ANALYSIS.md |
| 回合规划（优先级与理由） | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/plan/UPLOAD-PLAN.md |
| 上游机会清单 | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/upstream/plan/UPSTREAM-OPPORTUNITIES.md |

---

## F. 上游对象（对照阅读）

| 对象 | 链接 | 与我们什么关系 |
|---|---|---|
| RFC #16375（他们的开发计划本体） | https://github.com/vllm-project/vllm-ascend/issues/16375 | 50 条目 / 0 完成 / 0 评论；我们逐条对 |
| PR #16925（V4.1 框架 + Engram host offload） | https://github.com/vllm-project/vllm-ascend/pull/16925 | 已 `MERGEABLE`；我们的 §6 边界数据挂在它这条线 |
| issue #16828（A3 host offload 报错） | https://github.com/vllm-project/vllm-ascend/issues/16828 | 官方确认走 host-registered UVA path ⇒ 与我们口径同路 |
| PR #16285（DSpark ACLGraph + DSA_CP） | https://github.com/vllm-project/vllm-ascend/pull/16285 | 改同一个函数；维护者三次否掉 v1 图模式 |
| PR #14428（我们 PR 的前置，已合入） | https://github.com/vllm-project/vllm-ascend/pull/14428 | 把 index+copy 融成 `out=` gather |
