# logs/ —— 日志类文档（按时间编号）

> 这里是**按时间累积的过程记录**：每次检查、裁决、实验报告都往这里放。
> **规格文档**（计划、分析、给上游的草稿）在**上一层**，不混放。

---

## 编号规则

```
logs/NN-YYYYMMDD-<主题>.md          ← 日志正文（NN 递增，不重用）
logs/raw/NN-<主题>-<日期>.<ext>     ← 对应的原始输出（与正文同 NN）
```

**为什么编号**：日志是**时间序列**，编号让人一眼看出先后；主题名让人不必打开就知道内容。

---

## 索引

> **接手先读这三个**：本文件 → [`../AGENTS.md`](../AGENTS.md)（规则/红线/环境）→
> [`../ONBOARDING.md`](../ONBOARDING.md)（命令级上手 + 排障表）。
> 想知道**当前在做什么/下一步**：[`01`](01-20260921-session-status.md)（状态看板）与
> [`../STATUS.md`](../STATUS.md)（若存在）。

## 一、上游材料与结论（**给上游看的**）

| # | 日期 | 文档 | 内容 |
|---|---|---|---|
| **14** | 2026-09-21 | [`14`](14-20260921-rfc-citation-audit.md) | **RFC #16375 引用审计** —— 抓原文快照，修掉 3 处行号错误（line 8/111/117 → 3/108/104），并纠正 2 个 PR 的 RFC 归属 |
| **18** | 2026-09-21 | [`18`](18-20260921-track-c-reaim.md) | **C 轨重估** —— 维护者明确否掉 v1 的 DSpark 图模式（含逐字原文）⇒ 撤回该主张，改为交出 sync 账 |
| **22** | 2026-09-21 | [`22`](22-20260921-causal-conv1d-dead-import.md) | **上游死代码** —— `causal_conv1d_update_npu` 已被 #14620 删除，#15127 又把 import 加回 ⇒ `try` 永远失败、每 worker 打误导性告警（有单卡复现） |
| **29** | 2026-09-21 | [`29`](29-20260921-host-register-18x-resolved.md) | ★ **18× 之谜解开** —— 真实 206 GiB 表实测 **119.4 s**；40 分钟是测试 VM 特性；顺带推翻自家"内存形态无关"（稀疏文件假象，差 65×） |
| **08** | 2026-09-21 | [`08`](08-20260921-upstream-recheck.md) | **上游进度复查 + 材料对齐** —— 8 个 PR/issue 状态、review 风向、逐文件改动清单 |
| **05** | 2026-09-21 | [`05`](05-20260921-rope-opcount-and-bitexact.md) | rope `index_select`：op 计数 45→15、逐位一致、eager/ACLGraph 双口径 |
| **04** | 2026-09-21 | [`04`](04-20260921-moe-mask-graph-vs-eager.md) | MoE mask：**eager 与 ACLGraph 符号相反** ⇒ 引收益必须用图口径 |
| **10** | 2026-09-21 | [`10`](10-20260921-moe-mask-variance.md) | MoE mask **run-to-run 方差**：符号 21/21 稳定，但幅度比原稿小 30–40% |
| **09** | 2026-09-21 | [`09`](09-20260921-index-sweep-verdicts.md) | `index_select` 清扫三候选：**两个否掉**（一处是 vLLM 上游副本、一处是 3-D 索引） |
| **02** | 2026-09-21 | [`02`](02-20260921-probe-verdict.md) | Engram host-mapped 探针裁决：同机两个相反结论，判定是探针 artifact |
| **03** | 2026-09-21 | [`03`](03-20260921-engram-gate-h2.md) | Engram gate 函数级对比：我们慢 1.12× 但显存 0.52×（**结论与预期相反，如实记录**） |
| **06** | 2026-09-21 | [`06`](06-20260921-host-register-cost-scaling.md) | `aclrtHostRegister` 成本曲线（**已被 29 修正**：file 那一行是稀疏文件假象） |
| **36** | 2026-09-21 | [`36`](36-20260921-engram-gate-control-arm.md) | **Engram gate control arm 实证 + A3 重跑** —— §3.1 的 TODO 系**文档陈旧**（脚本一直是 5 臂，control 在列）；A3 die 3 两次独立重跑：control 时间 1.00–1.06× 上游、峰值 **1.20×**（多出一份 FP32 `[n,4,5120]`）；显存比值三次一字不差，但 **n ≥ 2048 的时间符号翻转** ⇒ 文档改报 **parity**。`--verify-verbatim` PASS，RFC §3 已按新数据回填 |
| **35** | 2026-09-21 | [`35`](35-20260921-rope-edge-cases.md) | ★ **RoPE 边界矩阵**（RFC [91] 点名的 non-contiguous / empty / padded / prefill 尺寸）—— A3 die 6 上五 phase 全跑成：**32/32 + 5/5 checks**；`n=0` 空批**逐位一致**、`n=4096` eager **−376 µs** / **ACLGraph −384 µs/call**、每次取表 **6 → 2 kernel**、`draft_index=1..5` 10/10 逐位一致；**两格 int32 eager 回退（+20.7/+28.9 µs）如实写进正文** |
| **37** | 2026-09-21 | [`37`](37-20260921-ngram-and-hostreg-ab.md) | ★ **ngram JIT + host-register 双 API A/B**（补掉 RFC §3.3 两行 "not yet run"）—— 生产 decode `n=128`：上游 `update()` **1.68 ms → 0.074 ms（22.8×）**，per-token 走法 **1312×**，`torch.equal` **14/14**；`aclrtHostRegister(MAPPED)` 与 `aclrtHostRegisterV2(MAPPED\|PINNED)` **ret=0 + 设备侧逐字节读回** ⇒ **本机这一档没有分岔**。**206 GiB 满表仍标未测**（只到 512 MiB，差 ~400×） |
| **38** | 2026-09-21 | [`38`](38-20260921-host-dram-bandwidth.md) | ★ **host DRAM 带宽 / 并发 / NUMA**（补掉 RFC [47] 五项里最后一项）—— 1 die：连续读 **107 GB/s**、随机行 gather **96 GB/s**（HBM 对照 601/724 ⇒ host 便宜 5.6–7.6×）；**并发不是摊薄，而是按 CPU socket 封顶 ≈115 GB/s**（2 die 同 socket 各 57、跨 socket 各 107、3 die 三 socket 总 **321**、3 die 同 socket 各 38.8 ⇒ **最坏 2.8×**）。`numactl --membind` 被 seccomp 拒（`Operation not permitted`）⇒ 改用 cpunodebind + 首次触碰 + `/proc/self/numa_maps` 内核记账验证落点 |
| **39** | 2026-09-21 | [`39`](39-20260921-upstream-recheck-2.md) | **上游进度复查（第二轮）** —— **#16925 已 `MERGEABLE`**（73 文件 +10554/−665）；#16285 维护者 09-21 03:46 第三次表态（*"If DSpark graph support is required, please use v2."*）；#16828 官方回复确认走 **host-registered UVA path** ⇒ 与我们的测量口径同一条路；RFC #16375 仍 **0 评论**（窗口还开着）|

## 二、codex 接入 A3 本地模型（**本轮最大交付**）

| # | 日期 | 文档 | 内容 |
|---|---|---|---|
| **30** | 2026-09-21 | [`30`](30-20260921-codex-responses-incompat.md) | **问题定位**（⚠️ **标题的"不能直接用"已被 33 推翻**）—— 四个缺陷的根因、调用栈、以及"为什么修不到 HTTP 层"（239 条 pydantic 错误） |
| **33** | 2026-09-21 | [`33`](33-20260921-codex-on-a3-verified.md) | ★ **验证通过** —— 3 处修复；51+53 单测；HTTP 5/5；**真实 codex 5/5**（单轮/工具/图片/多轮/子代理） |
| **34** | 2026-09-21 | [`34`](34-20260921-subagent-semantics-verified.md) | ★ **子代理语义正确性** —— 抓包 + 用服务端同一编码器还原 prompt；载荷逐字到达、控制 token 已转义 |

## 三、运维事故与加固（**踩过的坑，别再踩**）

| # | 日期 | 文档 | 内容 |
|---|---|---|---|
| **20** | 2026-09-21 | [`20`](20-20260921-oom-postmortem.md) | **03:37 OOM 复盘** —— user-slice memcg 撞 26 GiB 上限、内核杀到 `systemd` ⇒ tmux + 全部 codex 一起消失 |
| **21** | 2026-09-21 | [`21`](21-20260921-oom-hardening-applied.md) | **加固执行** —— `MemorySwapMax 0→8G`、`/tmp 15G→8G`、启用 `systemd-oomd`、`~/tmp/<日期>/` 协议 |
| **32** | 2026-09-21 | [`32`](32-20260921-vllm-orphan-workers.md) | ★ **A3 起服连环失败真因** —— `VLLM::EngineCore`/`VLLM::Worker_*` 的进程名里**没有** `vllm serve` ⇒ `pkill` 杀不到 ⇒ 自我强化失败链 |
| **25** | 2026-09-21 | [`25`](25-20260921-single-card-handback.md) | **单卡机交回** —— 机器状态、文件存档（31 MB / 3217 文件）、会失去什么 |
| **28** | 2026-09-21 | [`28`](28-20260921-remote-teardown-checklist.md) | **回收前接回清单** —— VPN 脚本（tmpfs，必丢）+ OpenVPN 配置 + sshd 的重建步骤 |
| **26** | 2026-09-21 | [`26`](26-20260921-tmux-session-recovery.md) | **找回丢失的会话** —— 14 个 codex + 29 个 dsh；纯清单见 [`26-...-session-list.txt`](26-20260921-session-list.txt) |

## 四、A2 服务质量

| # | 日期 | 文档 | 内容 |
|---|---|---|---|
| **31** | 2026-09-21 | [`31`](31-20260921-a2-service-quality.md) | **A2 生产口径评估**（基于用户粘贴的 18 个采样）—— prefix 命中 **96.4%**（从 0% 修好）、A 中位 3.58、KV 占用 <10%、两种坏状态都未复现；反推 `SP_TOKENS=7` |

## 五、状态看板

| # | 日期 | 文档 | 内容 |
|---|---|---|---|
| **01** | 2026-09-21 | [`01`](01-20260921-session-status.md) | **主代理状态看板** —— Phase 0/1 进展、锁的三个 bug、子代理产出、待办、09:00 之后的进展 |

## 六、空号（**别去找，没有产出**）

`07` `11` `12` `13` `15` `16` `17` `19` `23` `24` `27` —— 这些编号是 **03:37 OOM 时被杀掉的子代理**预留的；
它们要么没来得及写日志，要么产出已并入相邻编号（例如 15/19/23/24 的原始证据在 `logs/raw/` 下同名目录里）。
**这是正常现象，不用补**。

---

## 非本目录的内容（**别放错**）

| 类型 | 放哪 |
|---|---|
| 计划 / 复盘 / 分析 | 上一层 `*.md`（`OVERNIGHT-PLAN.md` `PLAN-REVIEW.md` `CI-ANALYSIS.md` …） |
| 给上游的草稿（PR 描述 / issue / RFC 评论） | `../pr/` |
| 可执行脚本（探针 / bench） | `../pr/`（本地）＋ 单卡 `$W/bench/`（远端） |
| **过程日志 / 裁决 / 实验报告** | **本目录** |
