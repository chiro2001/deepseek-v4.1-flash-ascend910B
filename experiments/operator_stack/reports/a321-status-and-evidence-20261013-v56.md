# 正式状态与证据总览（a3-21 DeepSeek V4.1 + DSpark）2026-10-13 v56

一页纸回答三个问题：**做到哪了、还差什么、证据在哪**。

## 1. 目标验收清单（逐项）

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| 1 | 完整 40 层/hidden5120/384 专家 top6/W4A8_DYNAMIC/Engram int8/完整视觉/MTP3 真实权重 | ✅ | 质量收据绑定 `candidate_id=dsv41_mls_async_bi_v1` + checkpoint config sha `40ebd329…`（`quality_gate_pass_bi_v14/launched.json`） |
| 2 | DSpark 真实 ON、全部实际开销计入 | ✅ | `speculative_config={method:dspark,K:5,enforce_eager:false}`；每请求 actual verifier rounds 由 8 rank 收据统计（747 rounds / 8 请求） |
| 3 | 串行 ≥8 请求、固定 2K 输入/256 输出、审计/profiler 关 | ✅ | `acceptance_small.json`（8 samples、2048/255+1、audit/profiler off） |
| 4 | 按实际 target/verifier 轮次计 step；自洽三元组 | ✅ | **(26.5524 ms/step, A=2.7309, 102.85 tok/s)**，747 rounds/2040 tokens |
| 5 | clean-rate / 复读 / 空答 | ✅ | clean **8/8**、复读 **0**、空答 0 |
| 6 | HCCL_DETERMINISTIC=strict 与原数值门（A/A、路由/token/Top5/logprob、接受前缀/rollback） | ✅ | AA v4/v16 审计（routes 2094×40×6 逐位、logprob delta 0）；v51 分叉审计（含逐轮接受/回滚 trail） |
| 7 | 模型归属 | ✅ | `strict_quality_receipt.json`：candidate 指纹/checkpoint/source/8 rank 描述全绑定 |
| 8 | GSM8K 100/100 | ✅ | `gsm100.json`：n=100, correct=100, empty=0, errors=0（决定性题 gsm8k-12 = 13） |
| 9 | Vision 23/23 | ✅ | `vision23.raw.json` + `strict_quality_receipt.actual_Vision23_passed=True`（含 3 negative） |
| 10 | **包含 DSpark 的 ≤20ms 客户端验收** | ❌ **26.55 > 20**（缺口 6.55ms） | 见 §3 |
| 11 | 最优部署 | ⚠️ 已产出可复现套件（26.55 版），≤20ms 版待厂商内核 | `best_service_20261013/` |
| 12 | 交付：COS/links 发布 + 源码/证据提交 + MANIFEST + 自检 + 双推 | ✅（持续） | 见 §4 |

## 2. 最优服务的定义（可复现）

```
(26.5523755957162 ms/step, A=2.7309236947791167, 102.85044684362282 tok/s)
747 actual rounds / 2040 decoded tokens / clean 8/8 / 复读 0 / GSM100/100 / Vision23/23
快照：/work/src_dspark_native_graph_api_bi_v1（性能） + /work/src_dspark_general_quality_v11（质量）
关键开关：VLLM_BATCH_INVARIANT=1（质量关键）、async 调度、shared-expert 与 DSA 多流重叠
```

一键复现见 `evidence/formal_a321/best_service_20261013/BEST_SERVICE.md`（含逐字 env 与期望值）。

## 3. 缺口 6.55ms 的定量结论（本轮新增两条否定证据）

**新增证据（2026-10-13 晚）**

| 候选 | 结果 | 判定 |
|---|---|---|
| `HCCL_BUFFSIZE=200`（默认 1024） | 26.4313 vs 26.5524 ms/step（−0.12ms），但逐请求分解显示收益**只来自单个离群请求**（28.93→27.27），其余 7 个请求在 ±0.15ms 内 | **噪声级**，不采纳也不拒绝（无负效果）；记录为 `v15_bi_buf200` |
| `enable_sparse_sfa_c8` / `enable_sparse_li_c8` | 只读核对：本 checkpoint `config.json` **含 `compress_ratios`**（长度 43）⇒ `model_uses_sfa_sparse=False` ⇒ 厂商代码把两开关强制置 False（`ascend_config.py:643-644`） | **不可用**（确认 v54 判定） |

**已建立的量化结论**（详见 v54/v55 报告）

- 轮 ≈ 目标图 21.65 + 草稿 2.6–3.7 + 拒绝采样 ≈1–1.2；pre_target/Engram/slot-map 已被 async 隐藏。
- 1 行（关 DSpark 同栈）实测 19.23ms ⇒ 固定成本≈17.7ms、边际≈1.5ms/行；6 行要 ≤20ms 需固定成本 ≤12.7ms。
- 配置空间已穷尽（v50/v54/v55 三张表）：MC2 负、FUSED_MC2 配置门不满足、reduce_sample 与 DSpark 不兼容、
  kv_nz 需 PD、sp_by_pass 需 SP、slot-fused/ids64-hoist 空结果、gate_hoist 与 BI 冲突、
  upstream pass_config 未在 Ascend 注册、c8 系列不可用（本轮）、buf200 噪声级（本轮）。
- **厂商需求已提交**：R1 MoE 段长融合（保守 1.9–3.2ms）、R2 HcPre/HcPost 链（1.6–2.8）、
  R3 量化链三段一体（0.95–1.6，或扩展 dynamic pattern 先做）、R4 allReduce 图级成对合并（0.54–0.93）、
  R5 草稿/采样段、R6 pre_target 折叠；三类并集保守 **4.38–7.50ms**（并通信 4.9–8.4），覆盖缺口。
  文档 `reports/a321-vendor-kernel-fusion-request-20261013-v55.md`（含邻接级证据与"新增 vs 扩展"判定）。

## 4. 交付状态

- **报告**（COS + links-server，均回读校验 true）：
  v50（async+多流窗口）、v51（分叉归因）、v52（批不变四处缺陷）、v53（质量门通过+代价≈0）、
  v54（缺口定量）、v55（厂商需求）；回执 `publication_v46..v50.json`。
- **提交**：分支 `feat/operator-stack-tp8-20261010`（origin GitHub 与 a321 内网镜像同步至最新 HEAD）。
- **MANIFEST/自检**：每次提交从 HEAD 重生成 MANIFEST（当前 6051 项），对 HEAD 全量 `git archive` 逐项
  `sha256sum -c` 0 失败，`tools/selfcheck_pkg.sh` exit 0；回执 `selfcheck_v4{6,7,8}…v55_archive.json`。
- **最优服务套件**：`evidence/formal_a321/best_service_20261013/`（env_exports.sh + env_manifest.json + BEST_SERVICE.md）。

## 5. 结论与建议

- 质量、归属、数值门、交付路径**全部实际达成并有收据**；
- **唯一未达成项是 ≤20ms**：当前 26.55，缺口 6.55ms，其根源是该栈"每步固定成本 17.7ms（1 行实测 19.2）"
  加 5 个 verify 行的 7.3ms 边际——需要厂商按 R1–R6 落地内核融合；
- 建议：把 v55 作为厂商工单提交（附 `fusion_scan_20261013/` 的 CSV 级证据）；
  本仓侧在厂商内核可用前，持续维护"质量/交付/可复现"三条链，并对任何新内核按
  **A/A → 数值门 → 同实例配对 → 质量门 → 正式窗口** 的顺序复验。
