# ★ 近期探索出的优化点：按预期收益排序（含 tiny 验证方法）

> 口径：**真实服务步长 24.59 ms**（profile 步长 40.02 ms，换算因子 1.6275；本文所有
> "真实 ms"均已换算）。基线：单流 **91.0 tok/s**、conc=8 总吞吐 **344.7~374.7 tok/s**。
> 证据来源：`docs/DECODE-*.md`、`docs/PINGPONG-*.md`、`docs/OFFICIAL-LIMIT-CORE-*.md`、
> `docs/UPSTREAM-DBO-SURVEY-*.md`、`docs/CANN-RECIPES-CROSSCHECK-*.md`。
> 标注：【实测】有运行数据 /【推断】由实测推导 /【未确认】待测。
+
+## 0. 一页纸结论
+
+**根因（一句话）**：真实步长 24.59 ms 里，**AIC 只忙 13.55 ms（55%）**，
+闲着的 **11.04 ms（45%）** 在等 AIV（6.9 ms）、通信（2.59 ms）、AICPU（1.27 ms）。
+⇒ **所有优化点本质都是同一件事：把 AIC 的空转填掉。**
+
+**总上界**：全部资源完美重叠 = `max(AIC, AIV, 通信, AICPU) = 13.55 ms` ⇒ **1.81×**。
+
+**排序**（按预期收益，单流与吞吐分开看）：
+
+| 序 | 优化点 | 单流预期 | 总吞吐预期 | 精度风险 | 工作量 |
+|---:|---|---:|---:|---|---|
+| **1** | **pingpong：跨 micro-batch 并行** | 0 | **+25~35%** | ⚠️ 中 | 大（需 ubatch 骨架） |
+| **2** | **通信与计算重叠** | **+5~12%** | +5~12% | ✅ 无 | 中 |
+| **3** | **层内控核 `limit_core_num`** | **+4~9%** | +4~9% | ✅ 无 | **小** |
+| **4** | ScatterNdUpdate 换官方 AscendC 算子 | +1~3% | +1~3% | ⚠️ 低 | 小 |
+| **5** | metadata 提前到步首侧流 | +2~3% | +2~3% | ✅ 无 | 中 |
+| **6** | `RmsNorm + DynamicQuant` 融合 | +1~2% | +1~2% | ⚠️ 低 | 中 |
+| **7** | SwiGLU clip quant 融合 | +0.5~1% | +0.5~1% | ⚠️ 低 | 小 |
+| **8** | cherry-pick #11273（DBO 通信重叠） | #2 的超集 | 同 #2 | ⚠️ 中 | 小（有现成 PR） |
+
+**2~7 不叠加**（都从同一份 AIC 空转里取），叠加上界 = 24.59 → 17.87 ms = **1.38×**（单流）。
+**1 与 2~7 正交**（1 拿吞吐、2~7 拿单流延迟）。
+
+---
+
+## 1. 各项的证据与量化
+
+### #1 pingpong：跨 micro-batch 并行【实测，微基准】
+
+**机理**：vLLM 把所有并发请求塞进**同一个 batch、同一条图、同一次前向** ⇒
+8 个请求的 AIC 相位**完全重合**（不是错开）⇒ 空档依旧没被填。
+拆成 k 个 micro-batch、各走各的流后，队列**天然把相位错开一位**。
+
+**微基准实测**（图捕获、总工作量相同、AIC:AIV=2.16）：
+
+| 流数 | 1 | 2 | 4 | **6** | 8 |
+|---|---:|---:|---:|---:|---:|
+| 时间 | 3.082 ms | 2.727 | 2.406 | **2.242** | 2.450 |
+| vs 1 流 | 1.00× | 1.13× | 1.28× | **1.37×** | 1.26× |
+
+完美重叠上界 1.46× ⇒ **6 流达 94%**。
+
+**外推**：真实比例 AIC:AIV = 1.49 ⇒ 上界 **1.67×**；扣掉
+**MoE grouped GEMM（占 AIC 36%、已 683 GB/s = 58% 可达，压不动）**后 ⇒ **1.25~1.35×**。
+
+**硬前提**：① **必须在图里做**（Python 交错提交实测 **0.24×**）；
+② **需 conc≥2**（conc=1 没有第二个 batch）；③ 需 ubatch 骨架。
+
+**tiny 验证**：把一次前向拆 2 个 micro-batch × 2 流（同权重、不同 KV block），
+图内实现；判据 `[bneck] hp` + 聚合 tok/s + **walk_blocks 逐位一致**。
+
+---
+
+### #2 通信与计算重叠【实测】
+
+**证据**（重叠矩阵）：
+
+```
+通信 ∩ AIC   = 0.000 ms
+通信 ∩ AIV纯 = 0.028 ms
+```
+
+即 **2.59 ms（真实）的集合通信完全零重叠**。它是唯一"**既零重叠、又不受 token 批依赖限制**"
+的时间 —— allreduce 只需等**本层** GEMM，**下一层的 GEMM 完全可以同时跑**。
+
+**为什么以前失败**：已排除的 `FUSED_MC2`（−3.8%）、`enable_sp`（更慢）、
+custom allreduce（Ascend 无实现）**全在"算子融合"层面**，
+**没试过"多流 + 控核"**。官方对应 `CCU 展开 AllReduce` + 多流。
+
+**上限**：2.59 ms / 24.59 ms ⇒ **+11.8%**；现实 +5~12%。
+
+**顺带**：cherry-pick **#11273**（Ascend DBO PR，已实现 `NPUUBatchWrapper`）就是这条路，
+上游实测 **+3.5~9.9%（conc≥4）**、**−5.0%（conc=1）**，但 **Phase 1 eager-only**。
+⇒ 可作为起点，但**不能替代图内实现**。
+
+**tiny 验证**：把 allreduce 挂到已有 `aux_stream`，与下一层 GEMM 重叠；
+判据 `[bneck] hp` + 逐位一致。
+
+---
+
+### #3 层内控核 `limit_core_num`【实测·源码级】
+
+**官方在 A3 上用它做 2 组"层内互不依赖支路"的并行**（`limit_core_num` 全仓库 5 个使用点）：
+
+| 组 | 甲 | 乙 | 核预算 |
+|---|---|---|---|
+| 1 | `mla_stream`：`kv_norm` + kv RoPE | 主流：`wq_b` + `q_b_norm` + q RoPE | **12 + 8** |
+| 2 | `compressor_stream`：compressor | `indexer_stream`：Indexer 的 Q 支路 | **16 + 8** |
+
+官方注释写死因果：`# c4a supports compressor parallel only if it supports limit core num`。
+
+**我们的现状**：`patches/files/dsa_v1.py:1884-1893` 用 **`wait_event` 强制先后**代替控核：
+
+> `kv_matmul and q_b_matmul are both Cube ops. Ensure kv_matmul ... completes before q_b_matmul starts so they do not contend for the Cube units.`
+
+⇒ 官方**真并行**（各拿一部分 cube、各自变慢但重叠），我们**严格串行**。
+**整个 `vllm_ascend` 里 `limit_core_num` 零使用。**
+
+**核预算曲线**（实测，`2048×4096×4096` bf16 GEMM）：
+
+| AIC | 24 | 20 | 16 | 12 | 8 | 4 |
+|---|---:|---:|---:|---:|---:|---:|
+| 相对 | 1.00× | 1.16× | 1.42× | 1.92× | 2.81× | 5.59× |
+
+**★ 零风险头寸**（实测）：**cube 密集算子对 AIV 预算完全不敏感**
+（AIV = 48/32/16/8/4/2 六档，GEMM 全部 0.221 ms）⇒ 可以把主流的 vector 核让给向量流。
+
+**上限**：主流 AIV 6.31 profile ms（3.88 真实）若全藏进 AIC ⇒ **+8.6%**；
+把上表 12/8 的分法用满 ⇒ 现实 **+4~9%**。
+
+**tiny 验证**：先量我们**实际 shape** 下 kv_matmul / q_b_matmul 的核预算-时间曲线
+（**不要照搬 12/8**），再改 `dsa_v1.py` 的 `wait_event` → `limit_core_num`；
+判据 `[bneck] hp` + 逐位一致。
+
+---
+
+### #4 ScatterNdUpdate 换官方 AscendC 算子【实测·代码】
+
+我们的 profile：`aclnnScatterNdUpdateSk` **1.180 profile ms = 0.73 真实 ms**（58 个/步，中位 20.4 µs）。
+官方 `ops/ascendc/` 有 **`npu_scatter_nd_update_asc`**，**文档标注支持 A3**，我们未用。
+
+**上限** +3.0%；实际取决于算子差距（需实测）。**注意**：换算子会改变数值路径 ⇒ 需精度验证。
+
+---
+
+### #5 metadata 提前到步首侧流【实测】
+
+**我们的现状**（步内位置分布）：
+
+| 算子 | 个数/步 | 落在步的位置 |
+|---|---:|---|
+| `SparseFlashMlaMetadata` | 3 | **83% 在最后 10%** |
+| `SparseAttnSharedkvMetadata` | 2 | **72% 在 80–90%** |
+| `ArgMax`（采样） | 19 | 69% 在最后 10% |
+
+**官方的做法**（`modeling_deepseek.py:2853-2890`）：
+`generate_kernel_metadata()` 在**模型 forward 的最开头、每步调一次**，
+整步的 SFA/LI metadata 全部丢到 `metadata_stream`，**与 40 层主计算完全重叠**。
+
+⇒ **同一件事的两种排法**：官方"提前铺开"，我们"堆在末尾"。`ArgMax` 依赖 logits **不能提前**，
+但 metadata 只依赖 block table ⇒ **大概率可提前**。
+
+**上限** +3.2%（0.76 真实 ms）。**风险低**（纯重排，理论上逐位一致）。
+**tiny 验证**：先确认我们的 metadata 生成是否真只依赖 block table。
+
+---
+
+### #6 / #7 融合类【实测·代码】
+
+| 项 | 当前开销 | 说明 |
+|---|---:|---|
+| `RmsNorm` + `DynamicQuantV2` **分开** | 2.012 + 0.747 profile = **1.70 真实 ms** | 官方有 `npu_rms_norm_dynamic_quant`（A3 支持），但**只在 w8a8 分支被调用**（`_is_w8a8_dynamic` 门控）⇒ 我们这条路径没走 |
+| `DequantSwigluQuant` | 0.561 profile = **0.35 真实 ms** | 官方有 `npu_swiglu_clip_quant` |
+
+**上限**：#6 +2.1%、#7 +0.8%。**⚠️ 融合会改变累加顺序** ⇒ 必须过精度关。
+
+> **我们已用 12 个官方算子**（`npu_hc_pre_v2` / `npu_hc_post` / `inplace_partial_rotary_mul` /
+> `sparse_flash_mla` / `quant_lightning_indexer` / `mega_moe` …）。
+> **我此前建议的"融合 HcPre/HcPost"是错的** —— 它们**已经就是**官方融合算子。
+
+---
+
+### #8 #17863 SFA 空闲核【上游 open PR，2026-10-01】
+
+官方 PR 描述：**A2/A3 的 NoPE SFA "一个 query group 却按全部硬件核发起并预留 scratch"**
+（"A one-row TND query on the tested A2 launches 20 AIC / 40 AIV despite having one query group"）。
+
+这与我们实测的"**AIV 有约 16 个核是多余的**"（elementwise 48→32 无代价）**同源**。
+修法（按 query rows 限核）与我们想的"AIV 预算按工作分"是同一思路。
+
+**收益未量化**（需在 tiny 上量），但属于**低风险**（只改 launch/allocation）。
+
+---
+
+## 2. 建议的执行顺序（考虑依赖，不只按收益）
+
+用户要求"按收益排序"。但 #1 需要 ubatch 骨架（工作量大），
+而 #2/#3 **可以立刻做且精度风险为零**。所以建议**并行两线**：
+
+```
+线 A（单流，低风险，边做边验证）
+  A1 #3 层内控核        ← 小改动、零精度风险、有官方参考，先拿 +4~9%
+  A2 #2 通信重叠        ← 中等改动、零精度风险，再拿 +5~12%
+  A3 #5 metadata 提前   ← 低风险
+  A4 #4/#6/#7 换算子/融合 ← 有精度风险，逐项过 walk_blocks 关
+
+线 B（吞吐，大改动）
+  B1 复用 vLLM `ubatching.py` 骨架（纯 Python，可直接用）
+  B2 在 tiny 上把 2 micro-batch × 2 流跑通（同权重、不同 KV block）
+  B3 量 conc=1/4/8 的 hp 与聚合 tok/s；扫 k=2/4/6
+  B4 可选：cherry-pick #11273 作为通信重叠的起点
+```
+
+**为什么 A 先于 B**：A 的每一项都能独立验证、且不改 batch 结构（精度风险低）；
+B 会改变数值路径（#11273 实测长 prompt 只有 **26.1% token-identical**），
+需要在 A 建立的判据上再验。
+
+---
+
+## 3. 每一项都必须过的验收门（tiny）
+
+我们刚花很大代价修完"静默答错"（32 位块回绕 / BAT=2048 越界）。**性能优化不能重蹈覆辙。**
+
+| 门 | 工具 | 判据 |
+|---|---|---|
+| **逐位一致** | `walk_blocks.py` + `walk_cmp.py` | 同配置重复 ≥10 轮，`max\|Δ\| = 0` |
+| **长文针** | `ced_pd_acceptance.py --mode needle` | 60K/74K/150K 全过（我们已有 24/24 基线） |
+| **性能** | `[bneck] hp`（引擎侧 ms/step） | **不用聚合 tok/s**（它被接受长度主导） |
+| **带宽透视** | `hbm_bw_sample.py` | 确认没把 HBM 打爆（上限 1182 GB/s） |
+
+---
+
+## 4. 组合收益（单流）
+
+| 组合 | 累计节省 | 步长 | 单流 tok/s | 倍数 |
+|---|---:|---:|---:|---:|
+| 现状 | — | 24.59 ms | 91.0 | 1.00× |
+| + #3 层内控核 | 1.94 | 22.65 | 98.8 | 1.09× |
+| + #2 通信重叠 | 4.53 | 20.06 | 111.5 | 1.23× |
+| + #5 metadata | 5.29 | 19.30 | 116.0 | 1.27× |
+| + #4 Scatter | 6.02 | 18.57 | 120.5 | 1.32× |
+| + #6/#7 融合 | 6.72 | 17.87 | **125.3** | **1.38×** |
+
+（⚠️ 这是**上界**：假设各项互不冲突；实际会互相挤占同一份 AIC 空转。
+现实预估 **1.20~1.28×**。）
+
+**总吞吐**：在单流 1.38× 的基础上再叠加 #1 pingpong 的 **1.25~1.35×** ⇒
+conc=8 总吞吐 344.7 → **约 560~610 tok/s**（【推断】，需实测）。
+
+---
+
+## 5. 一句话给决策
+
+**先做 #3（`limit_core_num`）** —— 它是**改动最小（几十行）、精度风险为零、有官方完整参考实现**
+的一项，预期 **+4~9%** 单流；做完立刻能在 tiny 上用 `walk_blocks` 验完，再上 tp8。
+**并行推进线 B（pingpong）** —— 它是**吞吐最大的**一项（+25~35%），但需要 ubatch 骨架，
+且必须先在 tiny 建立精度判据。
+
+## 6. 复现（本文所有数字的来源）
+
+```bash
+# 资源账 / 步内位置分布 / 串行链
+python3 tools/prof_chain_core.py   <rank0>/ASCEND_PROFILER_OUTPUT
+python3 tools/prof_chain_blocks.py <rank0>/ASCEND_PROFILER_OUTPUT
+python3 tools/prof_overlap_where.py <rank0>/ASCEND_PROFILER_OUTPUT
+python3 tools/prof_overlap_who.py   <rank0>/ASCEND_PROFILER_OUTPUT
+# 控核能力边界（tiny）
+python3 tools/tiny_limit_verify.py   # 核预算→时间曲线
+python3 tools/tiny_limit_key.py      # cube 对 AIV 不敏感
+# pingpong（tiny）
+python3 tools/tiny_pingpong_v3.py    # 图捕获下 1/2/4 流对照
+python3 tools/tiny_pingpong_v4.py    # 流数扫描 1..8
+```
