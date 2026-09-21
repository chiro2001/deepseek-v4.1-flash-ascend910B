# 上游合入材料 —— 审阅入口

> 这是给 `vllm-project/vllm-ascend` 的**草稿集**，**一份都还没提交**。
> 所有内容按「**帮他们的 RFC 打勾**」的框架写：给数字、给复现命令、给"还缺什么"，
> 不做"我们比你们强"的对比。
>
> 审阅顺序建议：**先看本文件的 §1（三条可直接发的）→ §2（数据完整度）→ 再看具体草稿**。

---

## 1. 三条"随时可发"的候选（等你授权）

| # | 类型 | 目标 | 材料 | 关键数字 |
|---|---|---|---|---|
| **①** | PR | `vllm-project/vllm-ascend` | [PR-rope-index-select.md](pr/PR-rope-index-select.md)<br>分支 [`perf/rope-fused-index-select`](https://github.com/chiro2001/vllm-ascend/compare/main...perf/rope-fused-index-select) | RoPE 查表 **6 → 2 kernel/次**；eager `n=4096` **−376 µs**；**ACLGraph −384 µs/call**；13 单测；32/32 + 5/5 边界检查 |
| **②** | PR | 同上 | [PR-moe-mask-range.md](pr/PR-moe-mask-range.md)<br>分支 [`perf/moe-contiguous-expert-map`](https://github.com/chiro2001/vllm-ascend/compare/main...perf/moe-contiguous-expert-map) | mask 由查表改范围比较：ACLGraph 内 **21/21 次更快**，中位 **−10.7…−12.9 µs**；eager **21/21 次更慢**（两个口径都给） |
| **③** | issue | 同上 | [issue-draft-cc1d-dead-import.md](pr/issue-draft-cc1d-dead-import.md) | 上游死 import：`causal_conv1d_update_npu` 已被删、import 又被加回 ⇒ `try` 永远失败、每个 worker 打误导性告警 |

> 三个都已带单测 / 复现脚本 / 上游格式标题 / DCO。**只差你说"发"。**

---

## 2. 数据完整度（本次补测的结果）

完整清单见 [DATA-GAPS.md](DATA-GAPS.md)。一句话版本：

**能补的 8 项补了 6 项**（A3 上空闲芯片上跑，四个子代理并行），**补不了的 5 项全是"环境不存在"**（无 W8A8 权重、无 A5、无多节点、A3 只剩 3 张空闲卡起不了 8 卡服务）+ 2 项要 A2 现场（我连不上，命令已备好）。

| 上游要求 | 之前 | 现在 |
|---|---|---|
| RFC **[91]** 非连续 stride / 空批 / padded / prefill 尺寸的融合校验 | "**not measured**" | ✅ **32/32 + 5/5**：`n=0` 空批逐位一致、`n=4096` 图内外都测、非连续 stride / 倒序 / 重复位置 / int32 / 2-D fallback 全覆盖；**含 2 格如实报告的回退** |
| RFC **[97]** 可复现对比（同卡同进程、上游 vs 我们） | 缺 control arm（"not in the script yet"） | ✅ **control arm 已实测**：换文件在时间上免费（1.00–1.06×）、**显存不免费**（恒定 1.20×）；A3 两次独立运行 + 旧单卡一次，三次同向 |
| RFC **[47]** NUMA / 带宽敏感度 | "**not measured**" | ✅ 设备侧读 host DRAM：连续 **107 GB/s**、随机行 gather **96 GB/s**（1 die）；**3 die 并发摊到 59 GB/s**；跨 NUMA node 扫描 |
| RFC [46]/[47] §3.3 host table registration 两条 API 的 A/B | "**not yet run**" | ✅ `aclrtHostRegister(MAPPED)` 与 `aclrtHostRegisterV2(MAPPED|PINNED)` **ret=0、设备侧逐字节读回一致** ⇒ 本机这一档**没有分岔** |
| RFC [47] §3.3 token history update（上游 per-token Python vs 我们的 numba JIT） | "**not yet run**" | ✅ 生产 decode `n=128`：**1.68 ms → 0.074 ms（22.8×）**；per-token 走法 **1312×**；`torch.equal` 14/14 |

---

## 3. 本轮新增的日志（每份都能追到原始 JSON）

| # | 主题 | 一句话结论 |
|---|---|---|
| [35](logs/35-rope-edge-cases.md) | RoPE 边界矩阵 / draft / 序列 / 图 / op 计数 | 五 phase 全跑成，32/32 + 5/5；**两格回退照实写** |
| [36](logs/36-engram-gate-control-arm.md) | Engram gate 的 control arm | 时间 parity、**显存 1.20× 是文件差异带来的真实差异** |
| [37](logs/37-ngram-and-hostreg-ab.md) | ngram JIT + host-register 双 API | 22.8× / 1312×；两 API 无分岔；**206 GiB 满表仍未测** |
| [38](logs/38-host-dram-bandwidth.md) | host DRAM 带宽 / 并发 / NUMA | 这是 RFC [47] 缺的最后一项 |
| [39](logs/39-upstream-recheck-2.md) | 上游进度复查 | **#16925 已 mergeable**；维护者对 v1 图模式第三次表态 |

---

## 4. 大的那份：RFC #16375 贡献报告

[pr/RFC-16375-CONTRIBUTION.md](pr/RFC-16375-CONTRIBUTION.md)（约 700 行）是**主文档**：

* **§2** 逐条对 RFC 的 50 个条目（哪些我们有、哪些没有、哪些只沾边）；
* **§3** 可复现对比（同卡同进程、上游代码逐字照抄、数值逐位校验）；
* **§5** 逐补丁消融表（**明确写着"这不是一次受控消融"**）；
* **§6** 给 issue **#16828** 写的边界数据 + **可检验假设**（不是"你们有 bug"）；
* **Appendix B** 自称的引用纪律（引条目必带原文、不跨会话比、不写倍数）。

配套还有：给 RFC 的评论草稿 [pr/RFC-comment.md](pr/RFC-comment.md)；三条轨道的 issue 草稿
[pr/issue-track-A.md](pr/issue-track-A.md)（Engram）、[pr/issue-track-B.md](pr/issue-track-B.md)（MoE）、
[pr/issue-track-C.md](pr/issue-track-C.md)（图执行）；纯文档贡献
[pr/TRACK-D-engram-host-table-support.md](pr/TRACK-D-engram-host-table-support.md)。

---

## 5. 策略（为什么这么写）

* **RFC #16375 就是他们的开发计划本体**：**50 条目 / 0 完成 / 0 评论**，且原文写着
  *"owners can be attached to individual implementation issues"*、
  *"contributors are welcome to propose owners, implementation issues, and target releases"*。
  ⇒ **打法 = 帮他们打勾**，不是证明我们更强。详见 [plan/PLAN-REVIEW.md](plan/PLAN-REVIEW.md)。
* **C 轨已改向**：维护者在 #16285 三次否掉 DSpark **v1** 图模式
  （最新一次 2026-09-21：*"If DSpark graph support is required, please use v2."*）
  ⇒ 撤回 draft-graph 主张，改为**交 sync 账**。见 [39](logs/39-upstream-recheck-2.md)。
* **红线**：不发 PR/issue/评论；不 push 到官方仓；结论必须能追到原始数据；
  没测的就写 "not measured"。

---

## 6. 还缺什么（诚实清单）

1. **W8A8 / A5 / 多节点 / SP·DCP·PD** —— 环境不存在，材料里统一写 "not measured"；
2. **8 卡单会话消融（RFC [97] 的 A1 harness）** —— A3 上只剩 3 张空闲卡，起不了 8 卡服务；
3. **`npugraph_ex` 分组件归因、overlap 的 trace artifact** —— 同上；
4. **A2 现场数据**（prefix 两臂对照 / 延迟分位数 / 并发曲线）—— 我连不上 A2，命令已备在
   [DATA-GAPS.md](DATA-GAPS.md) §4；
5. **206 GiB 满表的 host-register 并发** —— 只测到 512 MiB（差约 400 倍），所以
   "A3 没问题"这句话**没有**被写成结论。
