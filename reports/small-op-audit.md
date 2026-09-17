# 小算子海审计：来源与消除机会（DeepSeek-V4.1-Flash W4A8 / A3-node1 / AllGather 形态）

> 纯离线代码分析 + 只读 profile 复核，未占卡、未起停容器、未改任何代码。
> **行号口径**：所有 `文件:行号` 都指**容器内实际生效的内容**（`/vllm-workspace/vllm-ascend/...`，
> 含 5 个 bind-mount 覆盖文件：`models/deepseek_v41/model.py`、`engram_hbm.py`、`engram_gate.py`、
> `engram_hash.py`、`ascend_forward_context.py`）。源码快照取自容器本体（tar 快照，
> `/tmp/smallop/src/`，快照时刻容器在跑），不是宿主机 probe 副本。vllm 侧为 `/vllm-workspace/vllm/`。

---

## 0. 测量口径（先看这节，否则数字对不上）

### 0.1 切步方法（与主 Agent 的 10 步窗口可对账）

| 项 | 值 |
|---|---|
| profile | `A3-node1:~/projects/dsv41/logs/prof_ag/.../mindstudio_profiler_output/op_summary_*.csv`（已抽到 `/tmp/ag_extract/rank0_op_summary.csv`，md5 `f76c29778b9bc77a8dcbb84bac18455c`，258,795 行） |
| 切步 | 以每层 attention `wq_a` 的 QBMV3（输入形状 `8,5120;40,320,16,32;1280;8`）为首锚点，每 **40 个锚点 = 1 步**，步区间 = `[锚_k, 锚_{k+1})` |
| 步数 | 74（全 profile）；**主相位 68 步**（第 25–27 s），前 6 步是暖机/target-only 段，本报告全部定量结论取主相位 68 步 |
| 每步内容 | 1 次 target aclgraph 重放（M=8，40 层）+ 1 次 draft pass（M=7，3 层）+ host 侧 Engram route / 采样 |
| 步墙钟 | 中位 **38.48 ms**（p10≈38.4 / p90≈39.6） |

### 0.2 每步算子总数（**不是 1400，是 2805**）

| stream | ops/step | 含义 |
|---|---|---|
| **140** | **1686.6** | target aclgraph 主流（图内，可被融合直接消掉） |
| 47 | 692.9 | eager 侧：draft 3 层 + Engram 回程解包 + 采样后处理 |
| 138 | 160.0 | multistream 的 KV 分支副流（`dsa_v41.py:323 aux_stream`） |
| 139 | 81.0 | AIV 伴随 kernel（`AivKernel`，与 hcom_allReduce 成对） |
| N/A | 95.9 | HCCL（hcom_*） |
| 35 | 61.5 | Engram route 元数据副流 |
| 38 | 12.9 | broadcast 副流 |
| 33 / 41 | 11.9 / 2.0 | draft KV / alltoallv 副流 |
| **合计** | **2804.6** | 均值 12.3 µs/算子 |

> 主 Agent 的“1400+”与**图内主流 1686.6** 同量级；差额来自把 draft / Engram / 采样一起算进来。
> 本报告后面所有“次/步”都按 **2804.6 这口径**统计（只在 ②③④⑤ 里给出分流明细）。

### 0.3 device 时间账（本窗口实测，供交叉验证）

| 指标 | 值 |
|---|---|
| `sum(算子 device 时长)` | **34.65 ms/步** |
| 区间并集（busy） | **30.09 ms/步** |
| 步墙钟 | 38.48 ms |
| busy/墙钟 | **78.2%** |

（主 Agent 在另一窗口测得 27.7/35.5 ms。两者不矛盾：窗口、步定义与是否含 draft 不同；
本报告只用**每步次数**做结论，收益估算一律沿用主 Agent 的 **5 µs/算子**下发空隙常数。）

---

## ① 结论速览（Top-5 + 合计）

| # | 候选 | 每步减少算子数 | 预计减少下发空隙<br>（5 µs/算子） | 实现成本 | 精度风险 | 需重捕获图 |
|---|---|---|---|---|---|---|
| **1** | **MoE 路由的 `expert_map` 掩码链**（`Index`+`IndexCheck`+`NeScalar`+`Mul_Cast`+`Mul` 各 40） | **200** | **≈1.00 ms** | 中（改 `token_dispatcher.py`） | **需验证**：掩码与 `active_expert_range` 是否语义等价 | 是 |
| **2** | **W4A8 解锁 `rms_norm_dynamic_quant`**（F2，A3-node2 在做，此处只做补充定位） | **80** | **≈0.40 ms** | 中（50–120 行） | 低（数值等价，需 1 次逐元素比对） | 是 |
| **3** | **RoPE cos/sin 表的 gather 链**（`BroadcastTo`19.8 + `Gather_Cast`21.8 + `GatherElementsV2`21.8 + `Index`18.0 + `IndexCheck`~23） | **≈104** | **≈0.52 ms** | 中（按 step 缓存 cos/sin，或整链下沉为 1 个 kernel） | 无 | 否（全在 eager 侧） |
| **4** | **`input_ids.to(torch.int64)` 每层重复 40 次** | **39** | **≈0.20 ms** | **极低（1 行）** | 无 | 是 |
| **5** | **Engram gate 的 dtype 往返 + fp32 逐元素链**（2 个 engram 层 × 8 个 Cast + ~13 个 fp32 数学算子） | **≈43** | **≈0.22 ms** | 高（要新自定义算子；可先做“零成本”的 hoist） | 低（融合需逐元素比对） | 是 |
| | **Top-5 合计** | **≈466** | **≈2.34 ms/步** | | | |

### 另有两个“低垂果实”（不算进 Top-5，但确定，建议顺手做）

| 候选 | 每步减少 | 预计收益 | 成本 | 风险 |
|---|---|---|---|---|
| **indexer `weights` 的 bf16→fp32→fp16 往返**（`indexer.py:143` + `:206`，8 层各 1 次） | 8 | 0.04 ms | 极低 | 低 |
| **indexer k-scale 的 `.to(torch.float16)`**（`indexer.py:114`） | 4 | 0.02 ms | 极低 | 无 |
| **`hc_collapse` 的 2 次 Cast**（`model.py:538`/`:854`，可用 `rms_norm_cast` 式融合） | 2 | 0.01 ms | 低 | 无 |

**一句话**：算子海的**可确证**来源是四类 ——
① MoE 路由掩码（每层 5 个小算子）；② 各类 dtype 往返（Cast 301.7/步，其中 205 个在图内）；
③ RoPE 表查询的 eager 链（Index/IndexCheck/Gather/Broadcast 一族）；
④ 索引/边界检查（IndexCheck 80.8/步，其中 40 个是 MoE 掩码、23 个是 RoPE 表查询，
其余是长度计算与采样，**没有纯调试用的 IndexCheck** —— 它们都是 aclnnIndex 的实现组成，
不是可关掉的 debug 开关）。

---

## ② A. `Cast` 明细（301.7 个/步，502.8 µs/步）

### 2.1 官方口径 vs 实测

用户表的 `Cast 277.5/步` 与本次 `301.7/步` 差 8%，来源是**每步口径不同**（用户表分母 98.9；
本次是纯主相位 68 步）。**没有任何 Cast 来自 HcPre/HcPost 的 fp32 混合**（见 2.4）。

把 `Op Name` 里带 `CastAiCore` 的所有行按 `(算子名, 输入形状, 输入→输出 dtype, stream)` 聚类：

### 2.2 图内（stream 140）—— 这些是真·可融合目标

| 次/步 | aclnn 算子名 | 输入形状 | dtype | 调用点（文件:行） | 能否消除 |
|---|---|---|---|---|---|
| **41** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8]` | **INT32→INT64** | `ops/fused_moe/router/fused_topk_router.py:164`<br>`input_ids = input_ids.to(torch.int64)` | **能**。每层重复 cast 同一个 `input_ids`；在 `model.py:828` 前把 id 转成 int64 一次即可（↓39） |
| 40 | `aclnnMul_CastAiCore_Cast` | `[8,6]` | **BOOL→FLOAT** | `ops/fused_moe/token_dispatcher.py:390`<br>`topk_weights = topk_weights * mask` | **能**（跟随候选 1；`torch.where(mask, w, 0)` 可省掉这次提升） |
| 40 | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,6]` | **FLOAT→BF16** | `ops/fused_moe/token_dispatcher.py:428`<br>`probs=combine_metadata.topk_weights.to(hidden_states.dtype)` | **能**：`MoeGatingTopKHash` 直接输出 bf16 权重，或 unpermute 接受 fp32（↓40） |
| 40 | `RmsNormCast` | `[8,5120]` | bf16→(bf16, fp32) | `models/deepseek_v41/model.py:589` `x, x_fp32 = self.rms_norm_cast(x)` | **已经是融合算子**（norm+cast），不属于“多余 Cast”；但见 C（它后面紧跟一次 DynamicQuant，可再融一次） |
| **8** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,32]` | BF16→FLOAT | `models/deepseek_v41/indexer.py:143`<br>`weights = weights.float() * self.weights_scale` | **能**：把 `weights_scale` 折进 `weights_proj` 权重，或直接用 fp16 常量乘（↓8） |
| **8** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,32]` | FLOAT→FP16 | `models/deepseek_v41/indexer.py:206`<br>`weights = weights.to(torch.float16)` | **不能直接省**（QLI 要 fp16 输入），但与上一行合并后可少 1 跳 |
| **7** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,4,5120]` | BF16→FLOAT | `engram_gate.py:69`（`hidden.float()`）、`:70`（`key.float()`）、`:77`（`hidden.float()` 第二次）指向 **2 个 engram 层**（`engram_layer_ids=[1,14]`）= 6；`model.py:538` `hc_collapse` = 1 | **部分能**：`:69` 与 `:77` 是**同一个张量的重复 cast**，源码里 hoist 成 1 个变量即可省 2 个/步；整链融合可省更多 |
| **5** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,5120]` | BF16→FLOAT | `engram_gate.py:77` `value.float()`（2 个 engram 层）+ `attention/dsa_v41.py:414` `hidden_states_fp32 = hidden_states.float()`（compressor ratio-2 层 = 2/8/14） | **能**（compressor 侧：把 `wkv/wgate` 换成 bf16 权重或做一次“fp32 输入 GEMM”融合） |
| 4 | `aclnnInplaceCopy_CastAiCore_Cast` | `[4,5120]` | BF16→FLOAT | `models/deepseek_v41/model.py:847`<br>`layer.engram.q_weight.float() * layer.engram.k_weight.float()` （每 engram 层 2 个） | **能**：`q_weight*k_weight` 与权重加载期无关？否 —— 它是**静态权重**，可在 `load_weights` 后预算成 fp32 buffer，**每步省 4 个** |
| **4** | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,1]` | FLOAT→FP16 | `models/deepseek_v41/indexer.py:114`<br>`scale.unsqueeze(-1).to(torch.float16)`（4 个 kv-source 层） | **能**：k-cache 的 scale 平面直接用 fp16 存（cache spec `scale_dtype=torch.float16` 已在 `indexer.py:84`，此处是重复转换） |
| 2 | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,4,5120]` | FLOAT→BF16 | `engram_gate.py:77` `.to(hidden.dtype)` | 随候选 5 融合 |
| 2 | `aclnnInplaceCopy_CastAiCore_Cast` | `[32,32]` | BF16→FLOAT | `engram_gate.py:69` `rotation_block.float()` | **能**：`self.engram_rotation` 是常量，预存 fp32 版本即可（↓2） |
| 1 | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,5120]` | FLOAT→BF16 | `model.py:538` `hc_collapse` `.to(x.dtype)` | 见候选 5 |
| ~6 | 若干 1/step 级小项 | `[8]`/`[1]`/… | 各种 | 采样与长度计算（见 2.3） | 多为 aclnn 实现内部带出 |

小计：**图内 ≈205 个/步**。

### 2.3 图外（stream 47/35/33）—— eager 尾巴，都是“host 逐条下发”的受害者

| 次/步 | 算子名 | 形状 | dtype | 归属 |
|---|---|---|---|---|
| 18.8 | `aclnnGeScalar_CastAiCore_Cast` | `[8]` | INT32→INT64 | 采样/长度比较链（`torch_npu` 逐元素 op 的 index 提升） |
| 15.8 ×2 | `aclnnDivMods_CastAiCore_Cast` | `[8]` | INT32↔INT64 | 同一链（`div`/`remainder` 的 aclnn 参数转换） |
| 13.8 + 6.0 | `aclnnGather_CastAiCore_Cast` | `[8,1,1,64]` / `[7,1,1,64]` | INT64→INT32 | **RoPE 表 gather**：`ops/rope_dsv4.py:131` `pos_tensor.to(torch.long).reshape(-1,1,1,1).expand(...)` → `torch.gather`（`:134`/`:135`） |
| 8.9 | `aclnnArgMax_CastAiCore_Cast` | `[1]`/`[7]` | INT32→INT64 | `rejection_greedy_sample_triton` 前后的 argmax 索引提升 |
| 6.0 | `aclnnInplaceCopy_CastAiCore_Cast` | `[8,1,1,64]` | FLOAT→BF16 | RoPE 表写回 `buf_cos/buf_sin` 后转 bf16 |
| 3.0 | `aclnnInplaceCopy_CastAiCore_Cast` | `[1,129280]`/`[7,129280]` | BF16→FLOAT | **logits→fp32**：`vllm/v1/sample/sampler.py:96` `logits = logits.to(torch.float32)`；`rejection_sampler.py:154/219/220` 同理 |
| 其它 | 1/step 级 ~15 项 | | | fill/lt/mul/sub/… 的 index 提升 |

小计：**图外 ≈97 个/步**。

### 2.4 特别检查项（任务点名）

| 点名对象 | 结论 | 依据 |
|---|---|---|
| **HcPre / HcPost 的 fp32 混合** | **不产生任何 Cast**。`hc_pre`/`hc_post` 是单算子（`npu_hc_pre_v2`/`npu_hc_post`），fp32 混合在 kernel 内部完成 | `models/deepseek_v41/model.py:541`、`:554`（每层 2 次 HcPre/2 次 HcPost，实测 80/80 per step，各 2589/659 µs）；`[8,4,5120]` 的 Cast 全部来自 engram gate 与 `hc_collapse`，与 HcPre/HcPost 无时间邻接（见 §5 的逐步序列） |
| **Engram gate** | **是 Cast 大户**：每 engram 层 8 个 Cast（`[8,4,5120]`×4、`[8,5120]`×1、`[4,5120]`×2、`[32,32]`×1），2 层共 **16/步**，另有 ~13 个 fp32 逐元素算子 | `models/deepseek_v41/engram_gate.py:69,70,73,77`；调用点 `models/deepseek_v41/model.py:843`（`hidden_states` 此刻是 `[n, hc_mult, hidden]` = `[8,4,5120]`，由 `:821` 的 `unsqueeze(1).repeat(1, hc_mult, 1)` 产生） |
| **attention o_proj** | **不产生 Cast**。`wo_a`（`TransposeBatchMatMul [8,1,4096]×[1,4096,1024]`）与 `wo_b`（`MatMulV2 [8,1024]×[5120,1024]`）全程 bf16 | `attention/dsa_v1.py:1476-1600`（`_forward_o_proj`），内核见表 2.2 无对应行 |
| **MoE shared expert 路径** | **不产生 Cast**；它贡献的是 `DynamicQuant [8,5120]`（40/步）而非 Cast | `ops/fused_moe/shared_experts.py:301` `torch_npu.npu_dynamic_quant(hidden_states)` |

---

## ③ B. `Index` / `IndexCheck` / `ViewCopy` 明细（80.8 / 157.7 / 30.6 个/步）

> 说明：`aclnnIndex_*` 是 `tensor[idx]` 的实现，**`IndexCheck` 是与它成对出现的 shape/边界校验 kernel**，
> 由 aclnn 自动下发 —— 它**不是**某个可关闭的 debug 开关，也不发出任何 warning；唯一的消除方式是
> **不写高级索引**（换成 `torch.gather/index_select/where` 或把它折进融合算子）。

### 3.1 逐项归因

| 次/步 | 算子 | 形状 | stream | 调用点（文件:行） | 性质 | 能否消除 |
|---|---|---|---|---|---|---|
| **40 / 40** | `Index` + **`IndexCheck`** | `expert_map[384][topk_ids[8,6]]→[8,6]` | 140 | `ops/fused_moe/token_dispatcher.py:389`<br>`mask = expert_map[topk_ids] != -1` | **算法必需（正确性）**，但实现方式可换 | **能**：`expert_map` 在标准 EP8 下是连续区间映射，等价判定可写成 `(topk_ids>=first)&(topk_ids<last)`（省掉 Index+IndexCheck+Ne 共 117 个）或直接交给 `npu_moe_init_routing` 的 `active_expert_range`（省 200 个，需验证） |
| 23 | `IndexCheck` | `[1;8]` | 47 | `ops/rope_dsv4.py:134/135` 的 `torch.gather(..., out=buf[:n])` 与 `:146/147` 的 `full_rope_cos[pos]` | 载体：RoPE 表查询 | **能**：见候选 3 |
| 18 | `Index` | `[1048576,1,1,64][idx]→[8,1,1,64]` | 47 | 同上（`full_rope_cos[pos_tensor]`，`rope_dsv4.py:146`） | 表大小 = `max_position_embeddings=1048576`，宽 = `qk_rope_head_dim=64` | **能**：decode 只需要 1 行的 cos/sin（8 token 各行不同，但可一次 gather 后按 layer 复用，而不是每个 config/每个 caller 各一次） |
| 20 | `GatherElementsV2` | `[1048576,1,1,64]` + `[8,1,1,64]` idx | 47 | `ops/rope_dsv4.py:130-135` | 与上一行同源（cos/sin 两个张量各 2 次） | 同上 |
| 20 / 20 | `BroadcastTo` + `Gather_Cast` | `[8,1,1,1]→[8,1,1,64]` | 47 | `rope_dsv4.py:131` `.expand(num_tokens,1,1,rope_dim)` + `.to(torch.long)` | | 同上 |
| **7** | `ViewCopy` | `[4096;3;3;1;8,1,64;…]→[4096]`、`[32]`、`[28]` | 47/140 | `models/deepseek_v41/model.py:787-789` `padded.zero_(); padded[:n].copy_(values)`（Engram lookup 的 padded 写回，`[4096]=max_num_batched_tokens`）；`[32]`/`[28]` 来自 `hc` 相关 reshape（`model.py:821`、`engram_gate.py:69`） | **Engram 静态缓冲的 padding** | **能**（若 lookup 直接写进静态 buffer）；`[32]/[28]` 属于 view 语义，可省 |
| 4 | `IndexCheck` | `[1;7,3]` 等 | 47 | draft 的 `expert_map[topk_ids]`（`dspark_n_routed_experts=128`, `dspark_num_experts_per_tok=3`） | 与候选 1 同源（draft 侧） | 能（同候选 1） |
| 2 / 2 | `IndexPutV2` + `IndexCheck` | `[192,256]` | 47 | Engram 回程解包（`engram_hbm.py:687/694` `torch.index_select` 的对偶写回） | 数据搬运用途 | 能（整链下沉） |
| 3 | `Index` | `[8,5120][8]→[8,5120]` | 47 | draft 的 `positions/ids` 取行（`copy_and_expand_dflash_and_dspark_inputs_kernel` 前后） | draft 输入准备 | 部分 |
| 1 / 1 | `Index` + `IndexCheck` | `[8,129280][1]→[1,129280]`、`[7]→[7,129280]` | 47 | 采样：取 last-token logits（`rejection_sampler.py` 的 `logits[logits_indices]`） | 采样必需 | 可改成 view（decode 只有 1 行） |
| 1 | `IndexFill` | `[1,8]` | 47 | `rejection_sampler` 屏蔽拒绝 token | 采样必需 | 否 |

### 3.2 结论

- **没有任何 IndexCheck 是“调试/安全开关”**，它们都是 aclnnIndex 的组成部分，成对出现（40+40、23+18、2+2…）。
- 可消除性排序：**MoE 掩码（80 个）> RoPE 表查询链（~104 个）> Engram padded 写回（7 个）> 采样取行（2 个）**。
- `ViewCopy` 只有 30.6 个/步，且 7 个是 Engram padded 写回、其余是 reshape/view 语义，
  **单独优化它性价比最低**（与 cannbot 报告里“stream 47 的 ViewCopy→Cast 链”是同一批）。

---

## ④ C. `RmsNorm` + `DynamicQuant`：还有哪些地方没被 F2 覆盖

（A3-node2 的子代理在做 F2 = 在 W4A8 下解锁 `rms_norm_dynamic_quant`。本节只做**覆盖度补全**，不改代码。）

### 4.1 现状全量清点

| 次/步 | 算子 | 形状 | stream | 说明 |
|---|---|---|---|---|
| 47 | `RmsNorm` | `[8,512]` | 138(40)/140(4)/47(3) | 40×`kv_norm` + 4×compressor norm + 3×draft |
| 42 | `RmsNorm` | `[8,5120]` | 140(41)/47(1) | 40×`input_layernorm`（`model.py:577`）+ 1×final norm（`model.py:855`，可在 §5 的尾部序列里看到它紧接 `hc_collapse`）+ 1×draft 侧（stream 47） |
| **80** | `HcPre` / **80** `HcPost`（对照） | `[8,4,5120]` | 140 | **不是 Cast、不是 norm/quant**：mHC 的单算子，每层各 2 次（`model.py:570/579/582/591`） |
| 40 | `RmsNorm` | `[8,1280]` | 140 | 40×`q_norm` |
| 4 | `RmsNorm` | `[8,128]` | 140 | 4×indexer `k_norm`（只在 `kv_source_layer_ids`） |
| 4+3 | `RmsNorm` | `[7,5120]`/`[7,512]` | 47/33 | draft |
| **83** | `DynamicQuant` | `[8,5120]` | 140 | **40×`wq_a/wkv` 共享量化 + 40×shared expert**（+3 draft） |
| **48** | `DynamicQuant` | `[8,1280]` | 140 | 40×`wq_b` + **8×indexer `wq_b`（Linear 内部隐式量化）** |
| 4 | `DynamicQuant` | `[8,128]` | 140 | 4×indexer k |
| 6 | `DynamicQuant` | `[7,5120]` | 47 | draft |
| **3** | `RmsNormDynamicQuant` | `[7,1280]` | 47 | **draft 已经在用融合算子** |

### 4.2 `RmsNorm → DynamicQuant` 相邻点（全量，逐条给出）

| # | norm 调用点 | 形状 | 紧接的 quant | 是否相邻 | 能否换 `rms_norm_dynamic_quant` | 是否被 F2 覆盖 |
|---|---|---|---|---|---|---|
| 1 | `models/deepseek_v41/model.py:577` `x = self.input_layernorm(x)` | `[8,5120]` | `attention/dsa_v41.py:331` `q_quant, q_scale = wq_a.quantize(hidden_states)` | **相邻（同 stream 140）** | ✔ 可（q_a/wkv 共用同一份激活量化） | **是**（F2 主目标，40/步） |
| 2 | `attention/dsa_v41.py:351` `qr = attn.q_norm(q_a)` | `[8,1280]` | `attention/dsa_v41.py:352` `q_b_quant, q_b_scale = wq_b.quantize(qr)` | **相邻** | ✔ 可 | **是**（F2 第二处，40/步） |
| 3 | `models/deepseek_v41/model.py:589` `x, x_fp32 = self.rms_norm_cast(x)`（**已是融合 norm+cast**） | `[8,5120]` | `ops/fused_moe/shared_experts.py:301` `quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)` | **相邻**（中间只隔 MoE gate 的并发事件，见 `shared_experts.py:299-301` 注释） | ✔ 可，但要注意 `x_fp32` 仍要产出（`hidden_states_fp32=x_fp32` 供下游 fp32 路径） | **否 —— 这是 F2 清单里没有的第三处（40/步）** |
| 4 | `models/deepseek_v41/indexer.py:97` `key = self.k_norm(self.wk(latent))` | `[8,1,128]` | `indexer.py:106` `torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)` | **不相邻**（中间夹 `indexer.py:98-104` 的 RoPE） | ✖ 单纯 norm+quant 融合不适用，需要 **norm+RoPE+quant** 三段融合 | **否（4/步）** |
| 5 | `attention/dsa_v41.py:359` `kv = attn.kv_norm(kv)` | `[8,512]` | 无量化（后接 RoPE + `scatter_cache_sk`） | — | 属 F6（norm+RoPE+scatter） | 否（40/步，但省的是别的算子） |
| 6 | `models/deepseek_v41/compressor.py:57` `npu_rms_norm(...)` | `[8,512]` | 无（ratio-2 的量化发生在 indexer k 分支） | — | — | 否（4/步） |
| 7 | draft：`attention/dsa_v1.py:1159` 路径下的 `_mla_prolog` | `[7,1280]` | **已经是 `RmsNormDynamicQuant`** | — | 已融合 | — |

### 4.3 给 A3-node2 / F2 的三条补充结论

1. **第 3 处（shared expert，40/步）没有被 F2 描述覆盖**。它前面的 norm 是 `rms_norm_cast`（`model.py:589`），
   后面紧跟 `npu_dynamic_quant`（`shared_experts.py:301`）。要省掉的是 **1 个 DynamicQuant（40/步）**，
   做法是把 `rms_norm_cast` 升级成 `rms_norm_dynamic_quant`（同时仍要产出 fp32 的 `x_fp32`，
   因为 `model.py:590` 把它传给了 MLP）。
   注意：`shared_experts.py:301` 的输入是 **hc_post 之后的 `hidden_states`**，与 `model.py:589` 的
   `rms_norm_cast` 输出**不是**同一个张量 —— 需要先确认二者是否真的共用（见 §6 第 4 条的不确定项）。
2. **融合算子在本部署已经跑通**：draft 路径 3 个/步的 `RmsNormDynamicQuant [7,1280]→INT8`（stream 47）
   证明该算子在 A3 + 本 CANN 包上可用；目标侧不用的**唯一原因**是 `attention/dsa_v1.py:121-127` 的
   `_is_w8a8_dynamic()` 类型门（只认 `AscendW8A8DynamicLinearMethod`，W4A8 的 method 类不匹配），
   以及 V4.1 自写的 multistream 路径（`dsa_v41.py:351-352`）没有接。
3. **不要把 `kv_norm`（47/步）算进 F2 的收益**：它后面没有量化，只有 RoPE + scatter（F6）。

---

## ⑤ D. `InplacePartialRotaryMul` 为什么是 134.7（实测 145）而不是 40 或 80

### 5.1 逐形状清点（主相位 68 步）

| 次/步 | 输入形状 | 涉及层 | 调用点（文件:行） |
|---|---|---|---|
| **80** | `[8,1,8,512]`（q：n_local_heads=8 × head_dim=512） | **全部 40 层，每层 2 次** | ① 前向 q 的 RoPE：`attention/dsa_v41.py:374-380`（multistream 生效路径）/ `:288-294`（单流）；② **attention 输出的逆旋转**：`attention/dsa_v41.py:602-608`（`-sin` 就是逆旋转，紧接着 `:609 _forward_o_proj`） |
| **44** | `[8,1,1,512]`（kv latent） | 40 层各 1 + 4 个 kv-source 层各 1 | `attention/dsa_v41.py:360-366`（multistream）/ `:296-302`；长 KV latent：`attention/dsa_v41.py:438-444`（`_write_compressed_source`，层 2/8/14/20） |
| **8** | `[8,1,32,128]`（indexer query: 32 head × 128） | `index_source_layer_ids=[2,8,14,20,24,28,32,36]` | `models/deepseek_v41/indexer.py:135-141` |
| **4** | `[8,1,1,128]`（indexer key） | `kv_source_layer_ids=[2,8,14,20]` | `models/deepseek_v41/indexer.py:98-104`（`update_keys`） |
| 6 + 3 | `[7,1,8,512]` / `[7,1,1,512]` | draft 3 层 | `attention/dsa_v1.py:675-690` 段（draft prolog） |
| **合计** | | | **145.0/步** |

### 5.2 结论

1. **重复调用不是 `rope_groups` 造成的**。V4.1 注册的是**单组**：`models/deepseek_v41/model.py:438`
   `rope_groups=["default"]`；带 `f"c{compress_ratio}"` 的第二组只存在于 **V4** 模型
   （`models/deepseek_v4/model.py:550` `rope_groups = ["default", f"c{self.compress_ratio}"]`），
   V4.1 用不同的 `base/original_seq_len` 来区分长上下文层（`model.py:434-437`）。
2. **每层 3 次**才是 134.7/145 的真相：**q-RoPE + kv-RoPE + 输出逆旋转**。第 3 次
   （`dsa_v41.py:602`）是 V4.1 特有的“先旋转 q 做 SparseFlashMla、再把 attention 输出转回原坐标系”的写法，
   形状与 q 完全相同（`[8,1,8,512]`），这就是“80 = 40×2”的来源。
3. 多出来的 8（indexer q）+ 4（indexer k）+ 4（长 KV latent）= 16，正好落在
   `index_source_layer_ids` / `kv_source_layer_ids` 两组层上，可逐层对齐验证
   （本次实测：`[8,1,32,128]` 出现在层 {2,8,14,20,24,28,32,36}；`[8,1,1,128]` 出现在 {2,8,14,20}）。
4. **可消除机会**：输出逆旋转（40/步）与 q-RoPE（40/步）作用在同一形状上，若把
   `nope_head_dim/head_dim` 的语义改成“在 SMLA 前不旋转、在 o_proj 前不逆旋转”，可省 40 个算子，
   但**必须同时改 `SparseFlashMla` 的 rope 约定**（高成本、高风险，不建议作为第一优先）。
   次优：把 q-RoPE 与 `kv_norm` 无关的事实利用起来，让 `[8,1,8,512]` 的两次调用走同一 stream
   以减少 event 等待（不改算子数，只改排布）。

---

## ⑥ 查不到 / 不能确定（不脑补）

1. **“每步 1400+” 与本次 2804.6 的口径差**：我按 `[wq_a 锚, 下一锚)` 定义步，得到 2804.6；
   只数 stream 140 是 1686.6。**未复现 1400 这个确切数**，可能主 Agent 统计时只算了某一段
   （例如排除 draft 与 Engram 回程）。本报告所有结论用 2804.6 口径，若要与 1400 对齐，需确认对方的切窗方式。
2. **`Index` 与 `GethersElementsV2` 在 RoPE 表查询里的分工没有 100% 拆清**：
   `rope_dsv4.py:130-147` 有两条路径（`use_cache=True` 用 `torch.gather(out=)`，否则用 `full[pos]`），
   两者在 profile 里都能看到（gather 19.8/步、index 18/步）。**哪一次调用落在哪个 caller（target 元数据
   `dsa_v41.py:852` / draft `dsa_v1.py:1159` / 其它）没有运行时打点确认**，只有“次数与形状对得上”的推断。
3. **`get_cos_and_sin_dsa` 的 config 循环次数未直接观测**：代码是“按 `registry_summary` 里每个 config_key ×
   每个 group”各做一次 gather，本部署至少 2 个 config_key（长上下文层用 `compress_rope_theta=160000`，
   SWA 层用 `rope_theta=10000`，见 `model.py:434`），所以单次调用至少 4 个 gather（cos/sin×2 config）。
   实际次数与观测值的对应关系未逐条验证。
4. **`shared_experts.py:301` 与 `model.py:589` 的 norm 是否是同一个张量未验证**：
   `model.py:589` 的 `x` 在 `hc_post`→`hc_pre` 之间被 MLP 消费，而 `shared_experts.py:301` 的输入
   是 MoE 分支里的 `hidden_states`。要在运行期打点（或读 `fused_moe.py` 的调用链）才能断定第 3 处
   norm→quant 相邻是否成立。**C.3 的 40/步收益因此是“上限值”。**
5. **候选 1（MoE 掩码链）的语义等价性未验证**：`expert_map[topk_ids]!=-1` 是否恒等价于
   `active_expert_range` 区间判定，取决于 EP8 下 `expert_map` 是否为连续区间映射
   （以及 `global_redundant_expert_num` 是否为 0）。**没有做运行时实验**，只做了代码推断。
6. **draft 侧的 18.8 个 `GeScalar_Cast`/15.8×2 `DivMods_Cast` 的宿主代码没定位到具体行**：
   它们在 `stream 47` 的采样/长度链里，性质是 aclnn 逐元素算子的 index 提升；推测来自
   `rejection_sampler.py` / `logits_processor` 的 Python 逐元素运算，**没有逐行确认**。
7. **本报告未审计**（不在任务范围，但量级不小）：`AivKernel` 95.9/步、`Fill` 48.5/步、
   `SWhere/SelectV2` 41.5/步、`AllReduce/AivKernel` 配对（stream 139 每层 2 次）。
   它们不在任务 A–E 的点名清单里，建议下一轮单独看。
8. **容器快照时点**：源码 tar 抓取时容器在运行（tar 成功、5913 个 .py、gzip 校验通过）。
   期间容器被重启过 1 次（第一次 tar 中断），本次使用第二份完整快照；若之后有人在容器内改了
   `token_dispatcher.py`/`indexer.py`，行号会漂移。

---

## 附：复现命令（只读）

```bash
# 1) 取源码快照（容器在跑时）
ssh A3-node1 'sudo -n docker exec dsv41-a21-perf bash -lc \
  "cd /vllm-workspace && find vllm-ascend vllm -name \"*.py\" -not -path \"*/.git/*\" -print0 | tar --null -T - -czf -"' > src.tar.gz
# 2) 取 profile（宿主机已有抽取件）
ssh A3-node1 'gzip -1 -c /tmp/ag_extract/rank0_op_summary.csv' > op_summary.csv.gz
# 3) 切步与统计：以 wq_a QBMV3 形状 '8,5120;40,' 为锚，每 40 个锚一步
```
