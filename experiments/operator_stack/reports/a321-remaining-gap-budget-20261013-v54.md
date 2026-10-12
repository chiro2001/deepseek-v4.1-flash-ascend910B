# 剩余缺口定量：26.55 → 20ms 需要什么（2026-10-13 v54）

当前最优正式服务（含 DSpark/MTP、完整 40 层 W4A8 + Engram int8 + 视觉 + MTP3、strict HCCL）：

```
(26.5523755957162 ms/step, A=2.7309236947791167, 102.85044684362282 tok/s)
747 actual rounds / 2040 decoded tokens / clean 8/8 / 复读 0
GSM8K 100/100、Vision 23/23（同一快照、同一服务）
```

目标 ≤20ms/step ⇒ 还需 **−6.55ms（−24.7%）**。本报告给出这 6.55ms 的逐项归属与"什么才能做到"。

## 1. 每轮实测分解（v11_bi 臂的 8 请求、747 轮精确收据）

| 段 | 每轮中位 | 合计 | 说明 |
|---|---:|---:|---|
| execute → bookkeeping | 22.11ms | 16.52s | `_prepare_inputs`/Engram 准备（被隐藏）+ 目标图 21.65ms + `_sample`（拒绝采样）+ D2H |
| bookkeeping → sample_end | **3.80ms** | 2.84s | **草稿提案 propose_draft_token_ids ≈3.7ms**（3 层 MTP，图重放）+ 输出整理 |
| 轮间 gap | 0.11ms | 0.08s | 极小 |
| 轮总计 | **25.66ms**（均值 25.92） | 19.36s | 另有窗口 pre+post 约 0.47s（59ms/请求） |

⇒ 精化的关键路径模型：**轮 ≈ 目标图 21.65 + 草稿 ≈2.6–3.7 + 拒绝采样 ≈1–1.2**（其余宿主段与
Engram/slot-map 等设备小工作已被 async 完全隐藏——见下节的空结果实验证明）。

### 1.1 空结果实验（本轮两条新证据）

- **小算子捆绑完全无效（精确零差）**：`BI + V41_SLOT_MAP_FUSED + V41_IDS64_HOIST` 与 BI 基线
  逐位相同——`26.552488540829987` vs `26.5523755957162` ms/step，747 轮/A=2.7309 完全一致。
  ⇒ slot-mapping 与 int64-cast 都在**被隐藏的宿主段**，其设备收益不上关键路径。
- **`enable_reduce_sample` 与 DSpark 不兼容**：起服即报
  `'DeepseekV41DSparkModel' object has no attribute 'logits_processor'`。
- **`V41_ENGRAM_GATE_HOIST` 与批不变模式冲突**：其权重缓存键用 `id()/shape`，在 Dynamo 追踪下
  产生数据依赖表达式 `u0`，profile_run 直接失败（首测即排除）。

## 2. 设备预算与"墙"（决定性事实）

- 目标图（40 层 × 6 行）rank0 实测：span 22.96ms、union **21.65ms**，**1986 kernel/步**、平均 ~11µs/个。
- 该图内：allReduce 81× 3.32ms、GroupedMatmulSwigluQuantV2 40× 3.16、HcPre 80× 2.30、MatMulV2 97× 2.15、
  QuantBatchMatmulV3 208× 2.00、GroupedMatmul 40× 1.48、SparseFlashMla 40× 1.31，其余 ~8ms 由数百个
  1–10µs 小算子构成（Cast/DynamicQuant/RmsNorm/Add/Mask…）。
- **1 行（关闭 DSpark 的同一栈）实测 19.23 ms/step**（客户端 19.289956）。即：**即使只验证 1 行，
  这个 40 层栈本身就要 19.2ms/步**。
- 5 个额外 verify 行的边际成本 = 26.55 − 19.23 ≈ 7.3ms（≈1.5ms/行）。
- ⇒ 形式化：`T(6行) ≈ 固定成本 17.7 + 6×1.5`。要在 6 行下 ≤20ms，需要
  **固定成本 ≤12.7ms（现在 17.7）** 且边际 ≤1.2ms/行；或让 6 行与 1 行同价（算法不可能）。

## 3. 配置空间：已穷尽（本轮补齐最后两个）

| 候选 | 结果 |
|---|---|
| `VLLM_BATCH_INVARIANT` | **采纳**：质量门通过、ms/step 噪声级、clean 7/8→8/8 |
| `multistream_overlap_shared_expert` + `multistream_dsv4_dsa_overlap` | **采纳**：−1.9ms |
| `V41_SLOT_MAP_FUSED` / `V41_IDS64_HOIST` | 设备侧各 −1.2/−0.2ms 级，客户端两轮均噪声级（本轮在 BI 臂上复测） |
| `V41_MOE_COMM_ALLGATHER=0`（MC2） | 负（目标图 +3.1ms），排除 |
| `enable_fused_mc2=1`（FUSED_MC2/MegaMoe） | 厂商配置门不满足（`moe_intermediate_size=2304` 不是 512 倍数） |
| `enable_reduce_sample`（省 logits 全量 gather） | **与 DSpark 模型不兼容**：`'DeepseekV41DSparkModel' object has no attribute 'logits_processor'` |
| `enable_kv_nz` | 仅 PD 场景（D 节点），本服务非 PD |
| `enable_sp_by_pass` | 需 pass_config.enable_sp；SP 对 6 token/TP8 无意义 |
| `weight_nz_mode=2` | 只影响 bf16/fp16 权重（本栈绝大多数是 W4A8 量化），量级 ≪0.1ms |
| `draft_window_size` | 对 DSpark 会**降低接受长度**（厂商注释明示），净负 |
| `enable_sparse_sfa_c8/li_c8`、`c8_enable_reshape_optim` | 需稀疏层配置，本 checkpoint 无 |
| `enable_mc2_hierarchy_comm`、`mc2_comm_alg` | 仅 MC2 路径；本栈走 AllGather |
| upstream `pass_config`（fuse_gemm_comms / fuse_allreduce_rms / …） | Ascend 编译器未注册这些 pass（只注册 norm_quant/qknorm_rope/muls_add），实测不生效 |
| `V41_ENGRAM_GATE_HOIST` | **与 BI 冲突**：用 `id()/shape` 做缓存键，在 Dynamo 追踪下产生数据依赖表达式 `u0`，profile_run 直接失败（本轮首测） |

## 4. 要跨过 6.55ms，需要什么（厂商内核级）

按"预计可得收益 / 所需改动"排序（每一项都需要改 CANN 算子或 vLLM-Ascend 内核调度，不是配置）：

1. **量化链融合**（DynamicQuant 172× + QuantBatchMatmulV3 208× + Cast 126× 目前各自成 kernel）：
   把"quant→GEMM→cast"折成少数 SuperKernel，按小算子均值 ~11µs 估，可省 **1.5–2.5ms**。
2. **集合通信合并**：同层 attention AllReduce 与 MoE AllReduce 可合并为一次（延迟对齐），
   81× 3.32ms → 约 1.7ms，省 **~1.6ms**。
3. **HcPre/HcPost 链**（80+80 个小算子 2.9ms）：与相邻 RmsNorm/量化融合，省 **~1.0ms**。
4. **采样段瘦身**（每轮 3.8ms 的收尾 + 拒绝采样内核）：把采样并入上图或减同步点，省 **1–2ms**。
5. **pre_target 405 小算子折叠为元数据图**（设备并集 2.1–4.4ms）：省 **1–2ms**。

合计 **6.1–9.1ms**，与缺口同量级——但全部落在厂商内核域。**这不是配置或本仓 patch 可达的**。

## 5. 结论与建议

- 已交付部分：**质量门（GSM8K 100/100、Vision 23/23、模型归属）在同一最优服务上真实通过**；
  性能 57.58 → 26.55 ms/step（−53.9%），全部有原始 HTTP/轮次收据。
- ≤20ms 的剩余障碍是**该栈的每步固定成本（17.7ms，1 行实测 19.2ms）本身已接近目标**，
  加上 5 个 verify 行的 7.3ms 边际。要在 6 行下达 20ms，必须把固定成本砍到 ≤12.7ms，
  即上文第 4 节的厂商级融合。
- 建议路径（按代价）：① 向厂商提交上述 1–5 项内核需求（附本报告的 kernel 级证据与收益估算）；
  ② 若目标允许"最优服务"以不同几何验收（例如更短输出/更小 A），可另议；③ 本仓侧继续做
  数值/质量/交付硬化（已完成）。

## 6. 证据

- 三元组与逐轮分解：`evidence/formal_a321/dspark_native_graph_api_v11_bi/`（含 8 份原始 HTTP）；
  `/work/results/formal_dspark_native_graph_api_v11_bi/client/acceptance.json`（远端全量）。
- 质量：`evidence/formal_a321/quality_gate_pass_bi_v14/`。
- 设备时间线（目标图 kernel 明细、2026 kernel/步）：`evidence/formal_a321/dspark_native_graph_profile_v1/`、
  `.../dspark_native_graph_profile_mc2_v1/`。
- 1 行对照：`evidence/a321-formal-mask-client-pass-20261011-v32.md`（19.289956ms/step，A=1）。
