# 冻结剖析邻接融合扫描（rank0 / target 图 Model 48）— 离线证据 2026-10-13

角色：独立子代理并行任务 `fusion_scan`。全部结论来自**离线只读分析**：未启动任何 NPU 作业、
未重启/停止容器、未修改任何冻结文件。本报告不是性能报价，**不得据此声称"可达 20ms"**。

数据源（冻结，未改动）：`.../profile/dspark_rank0_1869601_20261012002259638_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
（9,476,527 B；28,418 数据行；sha256 `a25ceb5b015e506f30bcd5f1a6f7df7dcdcb69b3f5684ae936283d93d4952c52`，前缀 `/work/results/formal_dspark_native_graph_profile_v1`）

分析脚本（容器内可复现；两次运行 JSON 逐字节一致）：`/work/tools/fusion_scan_rank0.py`（本目录）；机读证据 `fusion_scan_rank0.json`。

## 1. 方法与口径

- **主对象**：Model ID 48（target 图），Step Id 5..14（10 步）。47=draft（3 层）、49=Engram、4294967295=图外 eager 仅作旁证。
- **去重/配对规则**（本 CSV 的关键结构）：`Stream ID = N/A` 的通信行与具体 stream 行是**同一设备 kernel 的两个视图**。
  以 `(Step, Model, Start, Duration)` 全等配对：**926/926 全配对、0 悬空**；配对后丢弃 N/A 行（保留带 stream 行，
  并把 N/A 行的可读名 `hcom_allReduce__503_i_k` 回贴），因此不会把重复行当成两个真 kernel。
  唯一不对称：step 5 stream 139 有 81 个 `AivKernel` 而 N/A 仅 80 行（采集起点截断），即 1 个 allReduce 无双行。
- **去重前后计数**：全 CSV 28,418 → 27,492 行；target 图 19,869 → 19,060（**1,986/1,987 → 1,906 kernel/步**）；
  draft 3,069 → 2,982；Engram 870 → 870；eager 4,610 → 4,580。无整行重复（exact-dup = 0）。
- **邻接**：同一 (Step, Stream) 内按 (Start, Duration, 行号) 排序后取紧邻对/三元组；
  `gap = 后一 Start − 前一 (Start+Duration)`（µs）。
- **收益口径**：`上界 = 参与融合算子的 Duration 之和/步`（理想化，假设融合后不再计这些 kernel 的成本）；
  `保守 = 上界 × [0.35, 0.6]`。依据：该图 kernel 均值 11.32 µs（p50 4.68、p90 29.2）、1,906 kernel/步、
  <5 µs 小算子 991/步（合计仅 2.63 ms/步），融合回收的主要是**固定成本+边界**，不可能全量回收。
- **整图收益**：按**算子实例并集**去重后再给区间（每个实例只计一次），不把 top 表简单相加。

## 2. 关键事实（rank0，10 步）

- target 图去重后 **1,906 kernel/步**（stream 140 计算 1,825 + stream 139 通信 81）；每步 sum 21.39–22.08 ms（均值 21.55），
  span 22.86–23.55 ms。stream 140：sum 18.24–18.35 ms / span ~23.09（busy 79%）；stream 139：sum 3.15–3.27 ms / span ~23 ms。
- stream 140 相邻边界 gap：p10 0.0 / **中位 0.75 µs** / p90 2.75 µs ⇒ 图是"发射/固定成本绑定"。
- 类型清单（cnt/步, ms/步）：hcom_allReduce_ 81×3.3245｜GroupedMatmulSwigluQuantV2 40×3.1604｜HcPre 80×2.2986｜
  MatMulV2 97×2.1513｜QuantBatchMatmulV3 208×2.0040｜GroupedMatmul 40×1.4839｜SparseFlashMla 40×1.3076｜
  MatMulV3 46×0.6791｜RmsNorm 129×0.6243｜MoeInitRoutingV3 40×0.6202｜HcPost 80×0.5825｜DynamicQuant 172×0.5591｜
  InplacePartialRotaryMul 136×0.4598｜QuantLightningIndexerV2 8×0.3943｜Cast 126×0.2363 …（与 v54 的 top 列表逐项吻合）

## 3. Top 30 邻接对（按 duration 之和排序；cnt/步、gap 中位 µs）

| # | pair (TypeA→TypeB) | cnt/步 | 上界 ms/步 | gap 中位 | 类别 |
|---|---|---:|---:|---:|---|
| 1 | hcom_allReduce_→hcom_allReduce_ | 80.0 | 6.3691 | 234.0 | 通信（见 §5） |
| 2 | GroupedMatmulSwigluQuantV2→GroupedMatmul | 40.0 | 4.6443 | 2.75 | MoE 段 |
| 3 | MoeInitRoutingV3→GroupedMatmulSwigluQuantV2 | 40.0 | 3.7805 | 2.75 | MoE 段 |
| 4 | HcPost→HcPre | 74.0 | 2.6706 | 1.25 | Hc 链 |
| 5 | DynamicQuant→QuantBatchMatmulV3 | 168.0 | 2.1423 | 1.25 | 量化链 |
| 6 | GroupedMatmul→Cast | 40.0 | 1.5315 | 1.0 | MoE 段 |
| 7 | MatMulV2→MatMulV2 | 40.0 | 1.4668 | 0.0 | 其它 |
| 8 | SparseFlashMla→Neg | 40.0 | 1.3554 | 1.0 | 注意力尾部 |
| 9 | HcPre→RmsNorm | 40.0 | 1.3209 | 1.0 | Hc 链 |
| 10 | HcPre→RmsNormCast | 40.0 | 1.3067 | 1.0 | Hc 链 |
| 11 | ScatterNdUpdateSk→SparseFlashMla | 32.0 | 1.2120 | 2.75 | 索引/注意力 |
| 12 | QuantBatchMatmulV3→RmsNorm | 80.0 | 1.1949 | 1.0 | 量化链 |
| 13 | InplacePartialRotaryMul→MatMulV2 | 48.0 | 1.1568 | 0.0 | QKV/rope |
| 14 | MatMulV2→HcPost | 40.0 | 0.8154 | 27.75 | Hc 链 |
| 15 | RmsNormCast→MatMulV3 | 40.0 | 0.7308 | 0.0 | 量化链 |
| 16 | MaskedFill→MoeInitRoutingV3 | 40.0 | 0.6763 | 1.25 | MoE 段 |
| 17 | MatMulV2→Cast | 10.0 | 0.6715 | 0.0 | 量化链 |
| 18 | QuantLightningIndexerV2→SparseFlashMla | 8.0 | 0.6676 | 14.75 | 索引 |
| 19 | QuantBatchMatmulV3→Add | 40.0 | 0.6317 | 1.0 | 量化链 |
| 20 | MatMulV3→Cast | 40.0 | 0.6296 | 0.0 | 量化链 |
| 21 | RmsNorm→DynamicQuant | 80.0 | 0.6229 | 0.0 | 量化链 |
| 22 | HcPost→MatMulV2 | 2.0 | 0.6200 | 0.0 | Hc 链 |
| 23 | QuantBatchMatmulV3→DynamicQuant | 40.0 | 0.5495 | 1.0 | 量化链 |
| 24 | DequantSwigluQuant→QuantBatchMatmulV3 | 40.0 | 0.5367 | 1.25 | 量化/MoE |
| 25 | Add→HcPost | 40.0 | 0.5072 | 48.625 | Hc 链 |
| 26 | QuantBatchMatmulV3→DequantSwigluQuant | 40.0 | 0.4363 | 0.0 | 量化/MoE |
| 27 | Cast→QuantLightningIndexerV2 | 8.0 | 0.4046 | 1.25 | 索引 |
| 28 | InplacePartialRotaryMul→ScatterNdUpdateSk | 44.0 | 0.3890 | 1.25 | QKV/索引 |
| 29 | RmsNorm→InplacePartialRotaryMul | 44.0 | 0.3733 | 0.0 | QKV |
| 30 | MoeTokenUnpermute→DynamicQuant | 40.0 | 0.2928 | 0.0 | MoE/量化 |

注：第 1 行为相邻对求和（中间项重复计数），仅用于排序，不用于收益口径（见 §5）。

## 4. Top 15 邻接三元组（上界 ms/步）

| # | 三元组 | cnt/步 | 上界 |
|---|---|---:|---:|
| 1 | hcom_allReduce_→hcom_allReduce_→hcom_allReduce_ | 79.0 | 9.3593（重复计数，仅排序） |
| 2 | MoeInitRoutingV3→GroupedMatmulSwigluQuantV2→GroupedMatmul | 40.0 | 5.2645 |
| 3 | GroupedMatmulSwigluQuantV2→GroupedMatmul→Cast | 40.0 | 4.6919 |
| 4 | MaskedFill→MoeInitRoutingV3→GroupedMatmulSwigluQuantV2 | 40.0 | 3.8366 |
| 5 | MatMulV2→HcPost→HcPre | 40.0 | 1.9731 |
| 6 | HcPre→RmsNormCast→MatMulV3 | 40.0 | 1.8885 |
| 7 | MatMulV2→MatMulV2→HcPost | 40.0 | 1.7614 |
| 8 | HcPost→HcPre→RmsNormCast | 40.0 | 1.6013 |
| 9 | InplacePartialRotaryMul→MatMulV2→MatMulV2 | 40.0 | 1.5980 |
| 10 | GroupedMatmul→Cast→Abs | 40.0 | 1.5819 |
| 11 | RmsNorm→DynamicQuant→QuantBatchMatmulV3 | 80.0 | 1.4875 |
| 12 | SparseFlashMla→Neg→InplacePartialRotaryMul | 40.0 | 1.4866 |
| 13 | DynamicQuant→QuantBatchMatmulV3→RmsNorm | 80.0 | 1.4810 |
| 14 | HcPre→RmsNorm→DynamicQuant | 40.0 | 1.4643 |
| 15 | Add→HcPost→HcPre | 34.0 | 1.4042 |

## 5. allReduce 成对出现位置（关键负结果）

- stream 139 上 81 个通信 op/步（全部 `hcom_allReduce_`/AivKernel，无 allGather），合计 **3.3245 ms/步**，平均 41.0 µs。
- **同流相邻 allReduce 的 gap：p10 189.5 / 中位 234.0 / p90 303.0 µs，gap ≤10 µs 的对数 = 0**。
  ⇒ 两个集合通信之间总隔着约 0.2–0.3 ms 的 stream-140 计算，**不存在"两个 allReduce 紧邻可融"的情形**；
  v54 第 2 项（80+1 次 → 合并）只能靠**图级/调度层成对合并**（把同层 attention 与 MoE 的 allReduce 合成一次调用），
  不是内核邻接融合。按"成对合并后延迟≈max(l_a,l_b)"口径，**上界 = Σ min = 1.5473 ms/步**
  （观测顺序配对为 1.0015 ms/步）⇒ 保守 **0.54–0.93 ms/步**（与 v54 的 ~1.6 ms 上界同量级、更保守）。

## 6. 逐候选保守收益（target 图内）

| 候选（实例并集去重） | kernel/步 | 上界 ms/步 | 保守 0.35–0.6× |
|---|---:|---:|---:|
| ①量化链 core（RmsNorm/RmsNormCast/Cast/DynamicQuant/QBMM 互邻） | 456 | 2.719 | 0.95–1.63 |
| ①'量化链 wide（并入 MatMulV2/GMM/Add/Mul 邻接） | 736 | 8.774 | 3.07–5.26 |
| ②通信成对合并（另计，不进并集） | — | 1.547 | 0.54–0.93 |
| ③HcPre/HcPost 链（含相邻 RmsNorm/Cast/MatMulV2） | 329 | 4.598 | 1.61–2.76 |
| MoE 段（MoeInitRoutingV3/GMMSwigluQuantV2/GMM/Cast） | 200 | 5.368 | 1.88–3.22 |
| **整图并集（①core ∪ ③ ∪ MoE）** | **945（49.6% 的 1,906）** | **12.505（58.0% 的 21.55）** | **4.38–7.50** |
| **整图并集 + 通信成对合并** | — | 14.052 | **4.92–8.43** |
| 交叉校验：贪心互斥对（排除通信，加 QKV/索引/rope 类） | — | 15.613 | 6.01–10.30（含通信） |

重叠处理：单价表按 pair 求和存在重复计数；本表按**实例并集**去重，且通信单独用 min-rule，不与计算候选混算。
两条独立路线（类并集 12.50 vs 贪心 15.61）给出的保守区间同量级，是稳健性交叉证据。

## 7. 厂商现有开关对照（容器内源码检索；VA=/vllm-workspace/vllm-ascend/vllm_ascend，V=/vllm-workspace/vllm/vllm）

版本：vllm-ascend HEAD `e43cf1e9f5d9bead076853aa6bcacb671465de94`（2026-09-11）；vllm HEAD `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`（2026-08-11）。

| 候选 | 已有/部分/没有 | 证据（file:line） |
|---|---|---|
| 量化链：norm+quant 图 pass | **部分** | `VA/ascend_config.py:70`（`fuse_norm_quant` 默认 True）；`VA/compilation/graph_fusion_pass_manager.py:58-63` 注册 `AddRMSNormQuantFusionPass`；其 pattern 仅覆盖 `npu_add_rms_norm_bias + 静态 quantize → npu.npu_add_rms_norm_quant`（`VA/compilation/passes/norm_quant_fusion_pass.py:30,80,143`）。本栈为 per-token **动态量化**，图上是 RmsNorm→DynamicQuant（80/步）两段 |
| 量化链：融合自定义算子 | **部分** | 厂商包已有 `rms_norm_dynamic_quant`（`VA/_cann_ops_custom/.../dynamic/rms_norm_dynamic_quant.py:185`）与 `rms_norm_cast`（同名目录 `rms_norm_cast.py`），模型已接线 `npu_rms_norm_cast`（`VA/models/deepseek_v4/model.py:732-734`）；但 **quant→GEMM→cast 三段一体**不存在 |
| 量化链：GMM+Swiglu+Quant | **已有** | `AscendFusionConfig.fusion_ops_gmmswigluquant=True`（`VA/ascend_config.py:94,102`）+ 自定义算子 `grouped_matmul_swiglu_quant_v2.py`（即图上 40×3.16 ms 的 GroupedMatmulSwigluQuantV2） |
| superkernel / combo kernel（通用内核融合框架） | **没有** | `VA` 全树 `superkernel/super_kernel/combo_kernel` **0 命中**；`V/vllm/config/compilation.py:978-986` 的 inductor `combo_kernels` 面向 GPU/inductor 路径，Ascend 走 npugraph_ex/torchair options（`VA/compilation/compiler_interface.py:88-122`），不消费该开关 |
| 集合通信合并（通用成对合并） | **没有** | 无任何"合并两个 collective"的开关；`VA` 内相邻 allReduce gap 中位 234 µs（本报告实测），邻接口径不成立 |
| MC2 家族（把通信并入 MoE 计算） | **部分（门不满足）** | `VA/ascend_config.py:419`（`enable_fused_mc2`）、`:403-478`（MegaMoe buffer/回退逻辑）；自定义算子 `dispatch_ffn_combine(_w4_a8).py`；v54 已证 `moe_intermediate_size=2304` 不满足 512 倍数门 |
| reduce-sample（采样段省 logits gather） | **部分（与 DSpark 不兼容）** | `enable_reduce_sample`（`VA/ascend_config.py:397`，兼容性门 `:668-681`）；v54 实测 DSpark 模型 `logits_processor` 缺失 |
| HcPre/HcPost | **已有单算子 / 链融合没有** | `hc_pre.py:191`、`hc_post.py:167` 注册为厂商自定义算子；无 Hc↔RmsNorm/量化/MatMul 的链式融合（图上 HcPost→HcPre 74/步、HcPre→RmsNorm|RmsNormCast 各 40/步，gap 1–1.25 µs） |
| 采样段（rejection/sampler） | **已有实现，但不在 target 图** | `VA/sample/rejection_sampler.py`、`VA/sample/sampler.py`（`_310p` 变体）；target 图内无采样 kernel（唯一匹配 MoeGatingTopKHash=路由 topk 40/步）⇒ 采样段收益需 draft/verify/尾部窗口证据 |
| pre_target 小算子折叠 | **已有机制且已在用** | `enable_static_kernel`（`VA/ascend_config.py:69`，默认 False；本冻结栈显式 True：`/work/src_dspark_native_graph_profile_v1/stack/parallel_tiny/dspark_formal_prep/contract.py:70`、`stack/serve_formal_tp8.py:144`）+ 编译接口 `VA/compilation/compiler_interface.py:88-122`；图外 eager 段 455 个计算 kernel/步、sum 3.55 ms/步（profiler Step 分桶对该段不对齐，span 45.9 ms）⇒ 进一步折叠=扩大捕获/改调度，属模型/调度改动 |

## 8. 分类结论：开已有路径 vs 需新内核

- **只需打开/接线已有能力**：① `fuse_norm_quant` 已在 True 但不匹配本图动态量化模式——把已有 `rms_norm_dynamic_quant` 算子接入图 pass 属于"接线+扩展 pattern"（厂商小改），能吃掉 RmsNorm→DynamicQuant（80/步）与 HcPre→RmsNorm/Cast 的一部分；
  ② `fusion_ops_gmmswigluquant` 已在用；③ `enable_static_kernel`+npugraph_ex 已在用。
- **需要新内核/新融合（本扫描的增量需求）**：
  1) **quant→GEMM→cast / DynamicQuant→QBMM→（RmsNorm/Add）** 一段式融合算子（QBMM 208/步、DynamicQuant 172/步、Cast 126/步；本候选 core 上界 2.72 ms/步）；
  2) **Hc 链融合**（HcPre 80×28.7 µs + HcPost 80×7.3 µs 与相邻 RmsNorm/Cast/MatMulV2 合成 1–2 个 superkernel；上界 4.60 ms/步）；
  3) **MoE 段长融合**：`MoeInitRoutingV3→GMMSwigluQuantV2→GroupedMatmul→Cast`（三元组上界 4.69 ms/步）——其中 GMMSwigluQuantV2 已是融合算子，新增点在路由/尾部 GMM+Cast；
  4) **集合通信成对合并**（图级；上界 1.55 ms/步，邻接口径为 0，见 §5）。
- **不在本次证据范围**：采样/拒绝采样段（图外）、pre_target 进一步折叠（图外 eager 段，需独立窗口证据）。

## 9. 诚实边界

- 所有数字是**离线估算**：上界口径理想化（假设融合后不计原 kernel 成本），0.35–0.6 是工程经验系数，未经上机校准；
  融合收益 ≠ 端到端 ms/step 收益（依赖链、流水重叠、宿主段会吸收一部分）。**不得表述为"可达 20ms"**。
- 仅 rank0；仅此一次 15 步诊断剖析（含 profiler 开销）；OC 与时钟状态未复核。
- 未跑 NPU 作业、未改冻结文件；脚本可复现（两次运行 JSON sha256 一致）。
- 与 v54 的 5 类清单：**方向一致**，两处细化/修正——(a) v54 第 2 项"allReduce 成对"在本剖析里**不存在邻接形态**（gap 中位 234 µs），只能做图级成对合并，上界按 min-rule 1.55 ms/步；(b) 量化链的"已有部分"比 v54 更强（`rms_norm_dynamic_quant`/`rms_norm_cast`/`npu_add_rms_norm_quant` 已在厂商包内），"接线+扩展 pattern"可先于"新 superkernel"落地。

## 10. 产物

| 文件 | 说明 | sha256 |
|---|---|---|
| `fusion_scan_rank0.json` | 机读证据（逐对/三元组 top、通信明细、候选并集、源码 file:line） | `5e325d0eb9de960698ecade7dcf17d0536baca3f91e5c2f06d3dd7e31475e6c5` |
| `fusion_scan_rank0.py` | 分析脚本（容器内 `/work/tools/fusion_scan_rank0.py`，read-only） | `b97fdd028582f65619047cfab6295ea8ea377c8f6bb2b0cf6615375eb128b45d` |
