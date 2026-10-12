# 厂商内核融合需求（a3-21 DeepSeek V4.1 DSpark，含邻接级证据）2026-10-13 v55

本文件把"26.55 → 20 ms/step 还差 6.55ms"转成一份**可执行的厂商需求**：每一项都有 kernel 级
邻接证据、收益上界与保守区间、以及"是新增内核还是扩展已有路径"的判定。所有数字来自**离线只读**
剖析（未跑新 NPU 作业），**不是性能报价**，也不构成 ≤20ms 达成的声明。

## 0. 基线与口径

| 项 | 值 | 证据 |
|---|---|---|
| 当前最优正式服务 | **26.5524 ms/step, A=2.7309, 102.85 tok/s**，clean 8/8，GSM100/100 + Vision23/23 | `evidence/formal_a321/dspark_native_graph_api_v11_bi/`、`quality_gate_pass_bi_v14/` |
| target 图（40 层×6 行） | **1,906 kernel/步**（去重后）、sum 21.55ms/步、span 22.86–23.55ms | `fusion_scan_rank0.json` |
| 1 行（关 DSpark 同栈） | 19.23 ms/step ⇒ 固定成本≈17.7ms、边际≈1.5ms/行 | `a321-dspark-round-budget-202613-v47` / v54 |
| 关键路径 | 轮 ≈ 目标图 21.65 + 草稿 2.6–3.7 + 拒绝采样 1–1.2；pre_target/Engram/slot-map 已被 async 隐藏 | v54 报告 §1 |
| 融合收益口径 | 上界=参与算子 duration 之和/步；**保守=上界×0.35–0.6**（kernel 均值 11.32µs、p50 4.68、<5µs 小算子 991/步仅 2.63ms） | `fusion_scan_rank0.md` §1 |

去重规则（本 CSV 的关键结构）：`Stream N/A` 的通信行与具体 stream 行是同一 kernel 的两个视图，
按 `(Step, Model, Start, Duration)` 全等配对 **926/926、0 悬空**，配对后丢弃 N/A 行并把可读名回贴；
全 CSV 28,418 → 27,492 行，target 图 19,869 → 19,060。

## 1. 需求清单（按保守收益排序）

### R1｜MoE 段长融合：`MoeInitRoutingV3 → GroupedMatmulSwigluQuantV2 → GroupedMatmul → Cast`

- 邻接证据：`GMMSwigluQuantV2→GroupedMatmul` 40×/步、**上界 4.64ms、gap 中位 2.75µs**；
  `MoeInitRoutingV3→GMMSwigluQuantV2` 40×、上界 3.78ms、gap 2.75µs；`GroupedMatmul→Cast` 40×、上界 1.53ms、gap 1.0µs。
- **上界 5.37ms/步、保守 1.88–3.22ms/步**（实例并集去重后）。
- 现状：`GroupedMatmulSwigluQuantV2` 已是单算子（`fusion_ops_gmmswigluquant=True`），但**路由→GMM→聚合→Cast 仍是 4 段**。
- 需求：**(a)** 扩展既有 `fusion_ops_gmmswigluquant` 到"含 routing 前置与 Cast 收尾"的整段（优先，属扩展）；
  **(b)** 或提供 MoE 段长融合内核（新增）。
- 风险：MoE 段是 numerics 敏感区（topk 路由/量化缩放），需与现有 W4A8 累加顺序逐位一致的门（我们有现成 A/A 审计脚本）。

### R2｜HcPre/HcPost 链融合：`HcPost→HcPre` / `HcPre→RmsNorm(Cast)`

- 邻接证据：`HcPost→HcPre` **74×/步、上界 2.67ms、gap 中位 1.25µs**；`HcPre→RmsNorm` 40×、上界 1.32ms、gap 1.0µs；
  `HcPre→RmsNormCast` 40×、上界 1.31ms、gap 1.0µs。
- **整链上界 4.60ms/步、保守 1.61–2.76ms/步**。
- 现状：`HcPre`/`HcPost` 已是单算子（`hc_pre.py:191` / `hc_post.py:167`），但**层内 4 段（HcPre→norm→attn→HcPost→HcPre→norm_cast→MoE→HcPost）**未融合。
- 需求：把 `HcPre→RmsNorm(RmsNormCast)` 与 `HcPost→(下一)HcPre` 两组**扩展为融合内核**（属扩展已有算子 + 新组合）。

### R3｜量化链三段一体：`RmsNorm/DynamicQuant → QuantBatchMatmulV3 → Cast`

- 邻接证据：`DynamicQuant→QuantBatchMatmulV3` **168×/步、上界 2.14ms、gap 中位 1.25µs**；
  类型级：QBMM 208×2.00ms、RmsNorm 129×0.62ms、DynamicQuant 172×0.56ms、Cast 126×0.24ms。
- **core 链（RmsNorm/DynamicQuant/Cast/QBMM）上界 2.72ms、保守 0.95–1.63ms/步**；
  宽口径（含 GroupedMatmulSwigluQuantV2/GroupedMatmul/MatMulV3）上界 8.77ms、保守 3.07–5.26ms（与 R1 有重叠，勿相加）。
- 现状：厂商包内**已有** `rms_norm_dynamic_quant`、`rms_norm_cast`、`npu_add_rms_norm_quant`，
  但 `norm_quant_fusion_pass` 只覆盖**静态量化** pattern（`norm_quant_fusion_pass.py:30,80,143`）；
  本栈是 **W4A8_DYNAMIC**，动态量化未被覆盖。
- 需求：**(a)** 把现有 pass 扩展到 dynamic-quant pattern（先做，属扩展，成本最低）；
  **(b)** 提供 `quant→GEMM→cast` 单内核（新增内核）。

### R4｜集合通信成对合并（**口径修正**）

- 实测：`hcom_allReduce_→hcom_allReduce_` 相邻对 80×/步，但**同流相邻 gap 中位 234µs、≤10µs 的对为 0
  ⇒ 不存在"紧邻成对"形态**；此前 v54 的"合并 81→40"设想需修正为**图级成对合并**：
  合并后延迟 ~max(la,lb)、上界=Σ min(la,lb) = **1.55ms/步**（按观测顺序口径 1.00ms），**保守 0.54–0.93ms/步**。
- 需求：图级"同层两次 allReduce 合并为一次"（需要算子级 barrier 语义支持），或提供 allReduce+RmsNorm 融合
  （厂商已有 `fuse_allreduce_rms` 概念但**未在 Ascend 编译器注册** → 属新增）。

### R5｜采样/收尾与草稿路径（关键路径的第 2、3 段）

- 实测：`bookkeeping→sample_end` 3.80ms/轮**就是草稿提案**（3 层 MTP 图重放）；拒绝采样约 1–1.2ms；
  两者合计 ≈5ms 在关键路径上（**未被隐藏**）。
- 需求：**(a)** 草稿 3 层图重放合并为 1 次提交（减少 3→1 的 host/图入口）；**(b)** 拒绝采样与 target 图尾端重叠
  或并入图。

### R6（旁证，非 target 图）pre_target 405 小算子折叠

- 设备并集仅 2.1–4.4ms，但宿主侧 10.6ms 胶水（`storm_probe_20261013.json`）；async 已隐藏其大部分。
  若 R1–R3 使目标图缩短，该段会重新暴露 → 建议同时把"输入准备小算子折叠为元数据图 replay"列入路线图。

## 2. 整图收益与结论

- **R1+R2+R3 的实例并集（945 kernel/步 = target 图的 49.6%）：上界 12.50ms/步、保守 4.38–7.50ms/步；
  并入 R4 通信后 4.92–8.43ms/步。**
- 与缺口 6.55ms 的对照：**保守区间（4.9–8.4）覆盖缺口**，但需要厂商逐个内核落地；配置空间已被证明穷尽
  （v50/v54 两张总表：MC2 负、FUSED_MC2 配置门不满足、reduce_sample 与 DSpark 不兼容、kv_nz 需 PD、
  sp_by_pass 需 SP、slot/ids64 为空结果、gate_hoist 与 BI 冲突、upstream pass_config 未注册）。
- **"只需开开关"的增量路径几乎没有**：norm_quant/qknorm_rope/muls_add 已默认开且不覆盖 dynamic；
  `enable_static_kernel`+npugraph_ex 本栈已开；无 superkernel/combo 框架（vllm-ascend 0 命中）。

## 3. 复现与证据

- 邻接扫描（离线、可复现、两次运行 JSON 逐字节一致）：`evidence/formal_a321/fusion_scan_20261013/`
  （`fusion_scan_rank0.md` 15.3KB、`fusion_scan_rank0.json` 91KB、脚本 `fusion_scan_rank0.py` 32.5KB；
  容器内 `/work/tools/fusion_scan_rank0.py`；数据源 CSV sha256 `a25ceb5b…4952c52`）。
- 设备时间线、宿主预算、口径对照：`evidence/formal_a321/dspark_native_graph_profile_v1/`、
  `hostsplit_hspan/`、`storm_probe_20261013.json`。
- 服务与质量：`dspark_native_graph_api_v11_bi/`、`quality_gate_pass_bi_v14/`、
  以及可复现套件 `best_service_20261013/`。

## 4. 诚实边界

- 上界是理想化（假设融合后这些 kernel 成本全归零），**保守区间才是可用于规划的估计**；
  两者都不是性能报价，也不保证 ≤20ms 达成——需厂商实现后按我们的 A/A + 质量 + 三轮正式窗口流程复验。
- 本文件不改变任何已验证结论：质量门（GSM 100/100、Vision 23/23）已在当前最优服务上通过；
  ≤20ms 未达成。
