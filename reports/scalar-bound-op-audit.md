# 标量受限算子审计：调用来源、逐模块归因与可融合机会

**审计对象**：DeepSeek-V4.1-Flash W4A8，8 卡 TP8+EP8，decode 单流，aclgraph（NPUGRAPH_EX=1），MOE_AG=1（AllGather 路径），attention = MLA + SparseFlashMla + Lightning Indexer，draft = DSpark 3 层。

**数据来源**（只读）：

- profile：`A3-node1:/home/user/projects/dsv41/logs/prof_ag/`，rank0 = `dp0_pp0_tp0_dcp0_ep0_rank0_242_20260915224145200_ascend_pt/PROF_000001_20260915224145214_00000242READICRP/mindstudio_profiler_output/{op_statistic,op_summary,api_statistic}_20260916064651.csv`。**只有 rank0 被 msprof 分析出 CSV**，其余 7 个 rank 只有原始 `PROF_*` 目录（无 op_statistic），故全部定量结论均为 rank0。
- 代码：容器 `dsv41-a21-perf` 内 `/vllm-workspace/vllm-ascend`（`_version.py`: `0.1.dev5097+ge43cf1e9f`）。注意 5 个文件被宿主机 bind-mount 覆盖（见 §0.3），报告中的行号 = 实际生效文件的行号。

**纪律**：全程只读（`sudo -n docker exec ... cat/grep/sed/ls` 与宿主机 `cat/ls`），未占 NPU、未起停容器、未修改任何被服务文件。

---

## 0. 测量口径（不先看这节会误读 210 这个数）

### 0.1 一个 profile 里混了 4 种 M

| 分组 | 输入 M | 次数 | 含义 | 判定依据 |
|---|---|---|---|---|
| 目标解码步 | 8 | **74** | 1 (query) + 7 (SP_TOKENS) token 的 aclgraph 重放 | `api_statistic` 中 `aclmdlRIExecuteAsync` Count = **74**；QBMV3 行聚类（gap>2 ms）得 74 个簇，簇内 QBMV3 = 211（73 簇）/208（1 簇） |
| dspark draft 步 | 7 | ~92 | draft 3 层前向 | M=7 的 QBMV3 簇大小恒为 15（88 簇）× 3 层 × 5 op |
| prefill chunk | 2024 | 16 | 32K prompt 分块预填 | M=2024 行在时间上集中在 10.0–17.4 s |
| 其它 | 384 / 16 | 1 / 1 | 一次长 chunk + 一次 warmup | M=384 在 17.38–21.38 s、M=16 在 0–4.7 s |

**关键结论：每个前向 pass（无论 M=8 还是 M=2024）的 `QuantBatchMatmulV3` 数都恰好是 211**。M=2024 组共 3376 = 16 × 211（q_a 640=40/层、wkv 688=43/层、q_b 768=48/层、shared gate_up 640=40/层、shared down 640=40/层），与 M=8 组 15614 = 74 × 211（同样 40/43/48/40/40）完全同构。因此 211 是**结构性常量**，与 batch 大小无关。

### 0.2 与用户表数值的对账

| 算子 | profile 总次数 | 表里“每步” | 反推分母 | 纯解码步实测 |
|---|---|---|---|---|
| QuantBatchMatmulV3 | 20792 | 210.2 | 98.9 | **211**（M=8 组 15614 ÷ 74 = 211.0） |
| SparseFlashMla | 3680 | 37.2 | 98.9 | 40（20+18+2） |
| GroupedMatmul / GroupedMatmulSwigluQuantV2 | 3956 / 3956 | 40.0 / 40.0 | 98.9 | 40 / 40 |
| HcPre / HcPost | 7912 / 7912 | 80.0 / 80.0 | 98.9 | 80 / 80 |
| MoeInitRoutingV3 | 3956 | 40.0 | 98.9 | 40 |
| RmsNorm | 12880 | 133.1 | 96.8 | **133**（47+42+40+4） |
| ScatterNdUpdateSk | 5336 | 55.1 | 96.8 | **55**（41+3+3+3+1+1+3） |
| MatMulV3 | 4398 | 45.4 | 96.8 | 46（40+6） |
| QuantLightningIndexerV2 | 736 | 7.6 | 96.8 | 8（5+3） |

表里的数值 = `总次数 ÷ (96.8~98.9)`，即**把 prefill chunk 与 draft pass 一起摊进分母**。分母不统一的原因见 §⑤.1（无法复现，标为不确定）。下文一律用**纯解码步**口径，并在需要时给出摊薄值。

每步墙钟：74 步的步间隔中位数 **38.5 ms**（p10 38.3 / p90 40.4，仅一次 13 s 空闲被 prefill 打断）。本报告涉及算子的 device 时间合计 **≈19.6 ms/步**。

### 0.3 生效文件（bind-mount 覆盖，必须按这个读代码）

| 容器路径 | 实际来源（宿主机） |
|---|---|
| `vllm_ascend/models/deepseek_v41/model.py` | `/home/user/projects/dsv41/probe_bneck/model.py.probe`（ro） |
| `vllm_ascend/models/deepseek_v41/engram_hbm.py` | `probe_bneck/engram_host_ws_opt.localowner_v2.py`（rw） |
| `vllm_ascend/models/deepseek_v41/engram_gate.py` | `probe_gate/engram_gate_ws_opt.stock.py`（ro） |
| `vllm_ascend/models/deepseek_v41/engram_hash.py` | `probe_hash/engram_hash_ab.py`（rw） |
| `vllm_ascend/ascend_forward_context.py` | `probe_moe/ascend_forward_context.py`（ro） |

（来自 `sudo docker inspect dsv41-a21-perf --format '{{range .Mounts}}...'`）

---

## ① 结论速览

### Top-3 可融合 / 可精简机会

| # | 机会 | 每步减少 | 预估收益 | 置信度 | 一句话依据 |
|---|---|---|---|---|---|
| **1** | **共享专家并入 routed 专家路径**（进一步：启用 `dispatch_ffn_combine_w4_a8` 把 dispatch + 两个 grouped matmul + combine 收成 1 个 kernel） | 80 个 QBMV3 + 40 个 `DequantSwigluQuant` + ~40 个 `DynamicQuant`；整段融合再加 40 `MoeInitRoutingV3` + 40 `MoeGatingTopKHash` 与 combine | **0.8 ~ 1.1 ms**（保守）／**1.3 ~ 1.6 ms**（整段融合） | 保守收益 **中高**；整段融合 **中低** | 共享专家当前 = 2 次独立 `npu_quant_matmul`（`shared_experts.py:305`/`:350`）+ `npu_dequant_swiglu_quant`（:331）+ `npu_dynamic_quant`（:301），实测 1059 us/步；`dispatch_ffn_combine_w4_a8` 已随包下发但**仓库内零调用点** |
| **2** | **把 `rms_norm_dynamic_quant`（norm+量化融合）解锁到 W4A8** | 80 个 `RmsNorm`（input_layernorm 40 + q_norm 40），并省掉两次 kernel 往返 | **0.25 ~ 0.40 ms** | **中高** | 融合算子现成且**本仓库已在 W8A8 路径使用**（`attention/dsa_v1.py:1688`/`:1815`）；W4A8 因 `_is_w8a8_dynamic(...)` 判断（`dsa_v1.py:1677`）与 V4.1 自写的 multistream 路径（`dsa_v41.py:352` 单独 `quantize`）而没有接上 |
| **3** | **o_lora 两跳（wo_a / wo_b）精简或量化** | 最多 40 个 `TransposeBatchMatMul` + 40 个 `MatMulV2`（量化后合并则 80→40） | **1.0 ~ 2.0 ms**（量化）；**≈1.1 ms**（仅消除退化 batch 维） | **中低**（精度/显存需实测） | `wo_a` 实测 **47.26 us/op × 40 = 1890 us/步**，是单算子最贵的非 MoE 算子；形状 `[8,1,4096]×[1,4096,1024]`（`n_local_groups=1` 的退化 batch matmul），cube 利用率 13.9%、mac 0.109 |

### 另两个规模小但确定性高的机会

| # | 机会 | 每步减少 | 预估收益 | 置信度 |
|---|---|---|---|---|
| 4 | **q_a 与 wkv 合并为一次量化 matmul**（输入同为 `hidden_states`，同一 per-token scale） | 40 个 QBMV3（21.83 us/对） | **0.25 ~ 0.30 ms** | 中 |
| 5 | **attention 的 wq_b 与 indexer 的 wq_b 合并**（输入同为 `qr`，形状同为 1280→4096） | 8 个 QBMV3 | **0.05 ~ 0.08 ms** | 高（收益小） |

### 明确的否定结论（避免走弯路）

- **q_a / q_b 不能合并成一次 matmul**：中间夹着 `q_norm`（1280 维 RMSNorm）+ 逐 token 动态量化，非线性不可折叠（`dsa_v41.py:284-286`、`models/deepseek_v4/model.py:507-518`）。
- **gate/up 已经合并**：共享专家 `gate_up_proj` 输出 576 = 2×2304/8，已经是“gate+up 一次 matmul”（`ops/fused_moe/shared_experts.py:305`）；routed 专家侧由 `grouped_matmul_swiglu_quant_v2` 一次完成 gmm1+swiglu+quant（`quantization/methods/w4a8/w4a8.py:492`）。
- **`MoeGatingTopK*` 不能把 5120→384 的 gate matmul 吃进去**：`moe_gating_top_k` 入参是 `(x, bias, y, expert_idx, out, k, ...)`，**没有权重入参**（`_cann_ops_custom/.../dynamic/moe_gating_top_k.py:141`），只做 logits→topk。router gate 仍是独立 `MatMulV3`（实测 15.42 us × 40 = 617 us/步）。

---

## ② `QuantBatchMatmulV3` 210 次/步的逐模块明细

### 2.1 分布明细（rank0，纯解码步 M=8，74 步，均值/步）

| # | 模块（生产者） | 次数/步 | 权重 shape（原始） | 单卡 shape（TP8 生效） | op 类型 / 实测 | 代码位置（文件:行） |
|---|---|---|---|---|---|---|
| 1 | **attention `wq_a`**（q 下投影） | **40** | 5120→1280 | `[8,5120]×[5120,1280]→[8,1280]` | MIX_AIC 13.46 us, aic_scalar 0.281, mac 0.061 | `models/deepseek_v4/model.py:501-506`；调用 `attention/dsa_v41.py:284`（单流）/ `:331,341`（multistream） |
| 2 | **attention `wkv`**（kv 投影，512 = kv_lora_rank） | **40** | 5120→512 | `[8,5120]×[5120,512]→[8,512]` | MIX_AIC 8.37 us, aic_scalar 0.378, mac 0.098 | `model.py:521-526`；调用 `dsa_v41.py:287` / `:339,349` |
| 3 | **attention `wq_b`**（q 上投影） | **40** | 1280→32768 | `[8,1280]×[1280,4096]→[8,4096]` | MIX_AIC 10.34 us, aic_scalar **0.487**, mac 0.083 | `model.py:511-518`（ColumnParallel）；调用 `dsa_v41.py:286` / `:352,372` |
| 4 | **indexer `wq_b`**（Lightning Indexer 的 q 投影） | **8** | 1280→4096（32 head × 128，全 rank 复制） | `[8,1280]×[1280,4096]→[8,4096]` | 与 #3 同形状、同 op 名 | `models/deepseek_v41/indexer.py:50-57`；调用 `indexer.py:134`（`select()`，由 `dsa_v41.py:471` 触发）；仅存在于 `index_source_layer_ids` = 层 **2/8/14/20/24/28/32/36** |
| 5 | **共享专家 `gate_up_proj`** | **40** | 5120→576（576 = 2×2304/8） | `[8,5120]×[5120,576]→[8,576]` | **AI_CORE** 8.09 us, aic_scalar 0.316, mac 0.099 | 权重定义 `models/deepseek_v4/model.py:227-234`（`MergedColumnParallelLinear(hidden, [intermediate]×2)`）；调用 `ops/fused_moe/shared_experts.py:305` |
| 6 | **共享专家 `down_proj`** | **40** | 288→5120（288 = 2304/8） | `[8,288]×[288,5120]→[8,5120]` | MIX_AIC 11.32 us, aic_scalar **0.659**（本 profile 最高）, mac 0.021 | 权重定义 `model.py:235-242`（`RowParallelLinear`）；调用 `ops/fused_moe/shared_experts.py:350` |
| 7 | **draft 的 context-KV 预投影**（dspark） | **3** | 5120→512 | `[8,5120]×[5120,512]→[8,512]` | 与 #2 同形状 | `models/deepseek_v4/dspark.py:208-215`（`_project_shared_kv` 内 `attn.wkv`）← `dspark.py:245-259` ← `spec_decode/dspark_proposer.py:386` |
| | **合计** | **211** | | M=8 组 device 时间 **2171 us/步** | | |

时间换算（rank0，M=8 解码步）：`40×13.46 + 43×8.37 + 48×10.34 + 40×8.09 + 40×11.32 ≈ 2171 us/步`；再加 draft 的 `15 × ~17.6 us × 1.24 pass ≈ 327 us/步` ⇒ **QBMV3 合计 ≈ 2.50 ms/步**。

### 2.2 推导过程（可复现）

1. 从 `op_summary` 抽出全部 20792 行 `QuantBatchMatmulV3`，按 `Task Start Time` 排序，以“输入形状首个张量以 `8,5120;40,` 开头”的行作为每层起点（即 `wq_a`）。
2. 相邻起点间隔 > 2 ms 判定为跨步；得到 74 个簇，簇大小 {211: 73, 208: 1, 3: 1}（208 那次是首簇、图捕获边界；3 那次是残段）。74 与 `aclmdlRIExecuteAsync` = 74 互相印证。
3. 簇内按时间排序后，每层固定出现 `A B D C E` 五个 op：
   - `A` = `[8,5120]→[8,1280]`（wq_a）
   - `B` = `[8,5120]→[8,512]`（wkv）
   - `D` = `[8,1280]→[8,4096]`（wq_b；序列中排在 `C` 之前是 cube 队列下发顺序，不是数据依赖顺序）
   - `C` = `[8,5120]→[8,576]`（AI_CORE，AIV 占用为 0 → 共享专家 gate_up）
   - `E` = `[8,288]→[8,5120]`（共享专家 down）
4. 40 层中偏离 `ABDCE` 的位置恰好两类：
   - **层 2/8/14/20/24/28/32/36 各多 1 个 `D`**（共 8 个）→ 与 config `index_source_layer_ids` **完全一致**（`/home/user/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/config.json`），即 indexer `wq_b`。
   - **最后一层尾部多 3 个 `B`**（比层循环晚 ~1.0–1.3 ms，间距 ~150 us）→ 与 draft 层数 3 相同，且与 `precompute_and_store_context_kv` 对 3 个 draft 层各调一次 `attn.wkv` 相符。**归因为 draft context-KV 预投影；属形状 + 代码推断（运行时打点未做，见 §⑤.4）**。
5. 交叉验证：`op_statistic` 中 `QuantBatchMatmulV3` = `MIX_AIC` 16836 + `AI_CORE` 3956 = 20792，其中 `AI_CORE` 3956 = 40/层/步 × 98.9 且**没有额外项**，正是共享专家 gate_up（唯一跑在纯 cube 上的 QBMV3）。

### 2.3 为什么 attention 的 QKV 是多个小 matmul 而不是 1 个大的

1. **checkpoint 本来就是 fused 的，是运行时主动拆开的**。权重加载里存在 `fused_qkv_a_proj` 的名字映射（`models/deepseek_v4/model.py:1218`），但运行期用两个独立 Linear：`wq_a`（5120→1280，`ReplicatedLinear`）与 `wkv`（5120→512，`ReplicatedLinear`），见 `model.py:501-506`、`521-526`。拆开的原因是二者标签不同：`dsa_v1.py:1775-1807` 与 `dsa_v41.py:325-341` 显式判断“wq_a 与 wkv 的 quant_method 类型 / `_has_communication` 是否一致”，一致才复用同一份激活量化；DSA 的 KV 分支还要接 `kv_norm` + `inplace_partial_rotary_mul` + scatter 写 cache（`dsa_v41.py:349-380`），与 q 分支的 `q_norm`+量化不同。
2. **`wq_b` 与 `wq_a` 之间夹着非线性**：`qr = q_norm(q_a)`（`model.py:507-510`、`dsa_v41.py:285`），然后 `wq_b(qr)`。RMSNorm 不可折叠，所以 1280→4096 必须单独一次 matmul。
3. **`wq_b` 在 TP8 下每卡只有 4096 列**：`n_heads × head_dim = 64 × 512 = 32768`，`ColumnParallelLinear` 切 8 份 = 4096/rank（`model.py:511-518`；`dsa_v41.py:286` 的 `.unflatten(-1, (attn.n_local_heads, attn.head_dim))` 印证 n_local_heads = 8）。所以“1 个大的 QKV matmul”在 TP8 下物理上也不存在。
4. **indexer 复用了 q_lora 隐空间**：`DeepseekV41Indexer.wq_b` 的输入也是 `qr`（1280→32×128 = 4096，全 rank 复制，`indexer.py:50-57,134`），因此 8 个 index source 层各多 1 个同形状 matmul（= `aic_scalar 0.487` 那类小 matmul 的主要来源）。
5. **o_proj 不在 QBMV3 里**：MLA 输出走 o_lora 低秩两跳且**未量化**——`wo_a` = `TransposeBatchMatMul [8,1,4096]×[1,4096,1024]`（47.26 us），`wo_b` = `MatMulV2 [8,1024]×[5120,1024]`（14.55 us），见 `attention/dsa_v1.py:1464-1558`（`:1557` 调 `wo_a`、`:1558` 调 `wo_b`）。SparseFlashMla 的入参里出现 `1,8192`（8192 = n_groups × o_lora_rank）与尾部 `8;1024`，但 o_proj 的两次 matmul 仍以独立 kernel 出现。

---

## ③ 融合候选逐条

### F1（Top-1）共享专家并入 routed 专家路径 / 整段 MoE 融合

**现状**（每层，共 40 层）：

| 步骤 | 生产者 / op | 代码位置 | 实测 |
|---|---|---|---|
| router gate | `MatMulV3` 5120→384（FLOAT 权重） | `models/deepseek_v4/model.py:286-288`（`ReplicatedLinear(..., quant_config=None)` + `precast_fp32_weight = True`，解释了实测 dtype = FLOAT） | 15.42 us × 40 = 617 us |
| topk | `MoeGatingTopKHash` | `ops/fused_moe/router/fused_topk_router.py:186`；`device/device_op.py:145` | 4.23 us × 40 = 169 us |
| dispatch | `MoeInitRoutingV3` | `ops/fused_moe/token_dispatcher.py:397`（`DeviceOperator.npu_moe_init_routing` → `device_op.py:101/114`） | 13.93 us × 40 = 557 us |
| expert gmm1 + swiglu + quant | `GroupedMatmulSwigluQuantV2` | `quantization/methods/w4a8/w4a8.py:492` | **78.57 us × 40 = 3143 us** |
| expert gmm2 | `GroupedMatmul` | `ops/fused_moe/routed_experts.py:272` | 37.23 us × 40 = 1489 us |
| **共享专家** | 2× `QuantBatchMatmulV3` + `DequantSwigluQuant` + `DynamicQuant` | `shared_experts.py:301,305,331,350` | 26.5 us × 40 = **1059 us** |
| combine | `MoeTokenUnpermute` + `Mul`/`Add` 等 | — | ~0.5 ms（未逐项拆） |

**候选做法**

1. *最小改动*：把共享专家两个 matmul 折进 grouped 路径（把共享专家权重当作第 N+1 个专家，或直接用 `grouped_matmul_swiglu_quant_v2` 的 tensor-list 变体），省掉 80 个 QBMV3 + 1 个 `DequantSwigluQuant`（`shared_experts.py:331`）+ 1 个 `npu_dynamic_quant`（`:301`）。
   - 现成算子：`grouped_matmul_swiglu_quant_weight_nz_tensor_list`、`grouped_matmul_swiglu_quant_v2_apt`、`dequant_swiglu_quant`（同目录；`:331` 已在用）。
   - 收益 **0.8~1.1 ms/步**；改动量估计 150~300 行（共享专家权重的 tensor-list 打包 + MoE 侧入口）；**需重捕获 aclgraph**；精度：共享与路由的 combine 顺序改变 ⇒ 需逐 token 比对（数学等价，预期相对误差 ≤1e-3）。
2. *最大收益*：启用 `dispatch_ffn_combine_w4_a8`——签名 `(a, w1, w2, expertIdx, scale1, scale2, bias1, bias2, probs, xActiveMask, out, expert_token_nums, group, M, transB, weightNz, swigluLimit)`（`_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/dispatch_ffn_combine_w4_a8.py:141`），一次 kernel 完成 dispatch + gmm1 + swiglu + gmm2 + combine，正好是 W4A8。
   - 收益：在上面基础上再加省 `MoeInitRoutingV3`（557 us）与后续 combine 若干 ⇒ 上限 **1.3~1.6 ms/步**。
   - **风险/未知**：该算子在仓库内**没有任何调用点**（全仓 grep 无命中）；是否支持 MOE_AG=1（AllGather）路径、是否支持 EP8 + shared expert、是否要求 `expert_token_nums` 预知，都需实测。`token_dispatcher.py:60` 的注释提示 `grouped_matmul_swiglu_quant_v2` 需要 per-expert counts，接口约束较强。置信度 **中低**，成本高（>500 行）。

### F2（Top-2）RMSNorm + DynamicQuant 融合（`rms_norm_dynamic_quant`）

**现状**（每步 133 个 `RmsNorm`，其中 80 个紧跟一次 `DynamicQuant`）：

- `input_layernorm`（5120，40/步，4.80 us，aiv_scalar 0.425）→ 需要给 `wq_a`/`wkv` 提供 int8 激活 ⇒ `DynamicQuant [8,5120]`（3.71 us）。代码：`models/deepseek_v4/model.py:717`（定义）、`:770`（调用）、`dsa_v41.py:331`（`wq_a.quantize(hidden_states)`，共享给 wkv）。
- `q_norm`（1280，40/步，5.94 us，aiv_scalar **0.401**）→ 给 `wq_b` / indexer `wq_b` 提供 int8 ⇒ `DynamicQuant [8,1280]`（2.88 us，48/步 = 40+8）。代码：`model.py:507-510`、`dsa_v41.py:285,352`。
- **现成算子已在同一仓库用于 W8A8**：`torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, weight, epsilon=...)`（`attention/dsa_v1.py:1688`、`:1815`；draft 路径实测 `RmsNormDynamicQuant` 3.69/步），但被 `if _is_w8a8_dynamic(self.wq_b)` 包住（`dsa_v1.py:1677`）；V4.1 自写的 multistream 路径完全没接（`dsa_v41.py:352` 单独 `wq_b.quantize(qr)`）。

**收益**：−80 个 kernel（40 RmsNorm + 40 DynamicQuant）≈ **0.25~0.40 ms/步**；算子现成、语义等价（W4A8 激活也是 int8 per-token dynamic），成本估计 50~120 行（去掉 gate / 加 W4A8 分支），**需重捕获**，精度中性（需 1 次逐元素比对确认融合量化与 `npu_dynamic_quant` 一致）。置信度 **中高**。

### F3（Top-3）o_lora 两跳（wo_a / wo_b）精简或量化

- 代码：`attention/dsa_v1.py:1464`（`_forward_o_proj`）→ `:1557`（`wo_a`）、`:1558`（`wo_b`）；权重定义 `models/deepseek_v4/model.py:530-544`。
- 实测：`wo_a` = `TransposeBatchMatMul [8,1,4096]×[1,4096,1024]` **47.26 us/op（min 44.9 / p50 47.2 / max 51.1，无长尾）× 40 = 1890 us/步**，cube 利用率 13.9%、mac 0.109、aic_scalar 0.183；`wo_b` = `MatMulV2 [8,1024]×[5120,1024]` 14.55 us × 40 = 582 us/步（cube 84%、mac 0.083）。
- 候选：
  1. **消除退化 batch 维**（`n_local_groups=1` 时 `[T,1,4096]×[1,4096,1024]` 退化为普通 2D matmul）：零精度风险，估 `47 → ~18 us`，省 **≈1.1 ms/步**（置信度中：需先做一次 kernel 级 A/B 确认 47 us 来自 batch 语义而非固定开销）。
  2. **把 wo_a / wo_b 也做 W4A8 量化**（当前是 bf16，是唯一没量化的注意力投影）：按 QBMV3 同形状的 10 us 量级，省 **≈1.7 ms/步**（置信度中低：需确认 W4A8 支持 4096→1024 / 1024→5120，且 o_lora 低秩对量化更敏感）。
  3. **折叠两跳为一跳**（`wo_b∘wo_a`，中间无非线性）：省 1 个 kernel，但权重从 9.4 M 变 21 M/层（bf16 全网 +1.0 GB），**不建议**，除非同时量化。
- 均需重捕获；精度需 o_lora 专项验证。

### F4 `wq_a` + `wkv` 合并（同输入、同量化 scale）

- 代码：`dsa_v41.py:284`（`q_a = attn.wq_a(hidden_states)`）与 `:287`（`kv = attn.kv_norm(attn.wkv(hidden_states))`）；multistream 版 `:331/339/341/349`；权重定义 `model.py:501-506`、`521-526`。
- 做法：按输出维拼接成一次 `[8,5120]×[5120,1792]`，输出切片后分别接 `q_norm` / `kv_norm`。数学**严格等价**（同一输入、同一 per-token int8 scale、两组 per-channel 权重 scale 分别拼接）。
- 收益：−40 个 kernel（21.83 us/对）⇒ 估省 **0.25~0.30 ms/步**（省掉一次激活搬运与一次 launch/标量开销，cube 工作量不变）。成本 100~200 行（权重打包 + 切片 + 复用 `share_hs_quant` 分支；checkpoint 里本来就有 `fused_qkv_a_proj` 可直接用），需重捕获，精度中性。置信度 **中**。

### F5 attention `wq_b` + indexer `wq_b` 合并（同输入 `qr`）

- 代码：`dsa_v41.py:286` / `indexer.py:134`，同在 8 个 index source 层；输入量化已被共用（`DynamicQuant [8,1280]` 实测 48/步 = 40+8）。
- 做法：拼接权重为 `[1280,8192]` 做一次 matmul，输出切片。
- 收益：−8 个 kernel ≈ **0.05~0.08 ms/步**；成本 60~120 行（indexer 权重拼接 + 切片），需重捕获，精度中性（同输入同量化，严格等价）。置信度 **高**，但收益小，建议与 F4 一起做（同一封装）。

### F6 `kv_norm` + RoPE + scatter 三段融合（低优先）

- 现状：`kv_norm`（40/步，7.32 us）之后紧接 `inplace_partial_rotary_mul`（全 profile 13340 次，6.3 us）与 `npu_scatter_nd_update_sk`（40/步，6.2 us），三步在同一 stream 串行。
- 同目录有 `store_kv_block`（`_build_args(keyIn, keyCacheIn, groupLen, groupKeyIdx, groupKeyCacheIdx, blockSize)`）这类“直接写 KV 块”的算子，若能把 norm+rope+scatter 收成一次，可省 ~80 个 kernel ≈ 0.5 ms/步；但需确认该算子支持的 cache 布局与 `33998,128,512` 平面一致。**置信度低，建议先不动**。

---

## ④ 归因：`RmsNorm` 133 次/步 与 `ScatterNdUpdateSk` 55 次/步

### 4.1 RmsNorm = 133（rank0，M=8 解码步，与用户表 133.1 一致）

逐层对齐方法：以每层 `wq_a` 的时间戳作为层锚点，把层锚点之间的窗口内各 op 归类到层。

| 形状 | 次数/步 | 归属（逐层对齐验证） | 代码位置 | 为什么没融合 |
|---|---|---|---|---|
| `[8,5120]` | **42** | 40 × 每层 `input_layernorm`（L0–L37 各 1）+ ~2 × draft `main_norm`（出现在层循环之后的尾窗口） | `models/deepseek_v4/model.py:717`、`:770`（V4.1 生效版本 `probe_bneck/model.py.probe:577`）；draft `main_norm`：`models/deepseek_v4/dspark.py:206` | 紧跟着 `DynamicQuant`（给 wq_a/wkv）；融合算子存在但没接到 W4A8（见 F2） |
| `[8,1280]` | **40** | 40 × 每层 `q_norm(q_a)`（L0–L39 各 1，逐层 mean = 1.00） | `model.py:507-510`；`dsa_v41.py:285` | 后面接着 `wq_b.quantize`（`dsa_v41.py:352`），可融合未融合 |
| `[8,512]` | **47** | 40 × `kv_norm`（每层 1，逐层 mean = 1.00）+ **4 × compressor norm**（层 2/8/14/20，正是 `kv_source_layer_ids`）+ 3 × draft context-KV 的 `kv_norm` | `model.py:527-529`、`dsa_v41.py:287`；compressor：`models/deepseek_v41/compressor.py:69`（定义）、`:122`/`:130`（调用，ratio-2 走 `pool_projected`）；draft：`dspark.py:214` | `kv_norm` 后接 RoPE + scatter，属“norm+rope+写 cache”三段（见 F6） |
| `[8,128]` | **4** | 4 × indexer `k_norm`，只在 owns_k 的层 2/8/14/20 | `models/deepseek_v41/indexer.py:73`（定义）、`:97`（调用） | 后面是 RoPE + `npu_dynamic_quant` + scatter（`indexer.py:98-115`） |
| **合计** | **133** | | | |

**已融合的对照用法**（说明剩下的确实是漏网的）：

- `rms_norm_cast`（`npu_rms_norm_cast`，`model.py:732-735`）：MLP 之前把 RMSNorm + fp32 cast 融合，实测 **40/步**（`RmsNormCast [8,5120]` 4.08 us）——本模型确实在用融合 norm。
- `npu_hc_pre_v2` / `npu_hc_post`（`model.py:741`/`:755`；V4.1 覆盖实现 `probe_bneck/model.py.probe:541`/`:554`）：mHC 已把 pre-mix / post-mix 全融合进大 kernel（实测 `HcPre` 80/步 × 29.52 us、`HcPost` 80/步 × 7.63 us），**这两个不需要再融合**；问题在 kernel 内部标量开销（HcPre aic_scalar 0.329、aiv_scalar 0.201，cube 利用率 56.5%，合计 2.36 ms/步）。
- `add_rms_norm_bias` / `dequant_swiglu_quant` / `rms_norm_dynamic_quant` 三个自定义算子在包内存在（`_cann_ops_custom/.../dynamic/`）；本模型未用 `add_rms_norm_bias`，因为 residual add 被 mHC 的 `hc_post` 取代。

### 4.2 ScatterNdUpdateSk = 55（rank0，与用户表 55.1 一致）

**唯一来源：KV cache 写入**，通过 `torch.ops._C_ascend.npu_scatter_nd_update_sk`：

- 封装函数 `scatter_cache_sk`：`attention/dsa_v41.py:207-226`（第 226 行发算子；`cache.squeeze(-2)` 解释了为什么同一物理 cache 有时以 3D、有时以 4D 出现）。
- 调用点：`dsa_v41.py:308`（`preprocess` 写 SWA cache）、**`:367`**（`multistream_preprocess`，本部署 MULTISTREAM=1 走这里）、`:445`（`_write_compressed_source` 写 long KV）；`models/deepseek_v41/indexer.py:110`（int8 k cache）、`:111`（k scale cache）。
- 其它路径（本部署未走，作对照）：`attention/dsa_attn_kv_plan.py:125`（BF16 SparseFlashMla 分支，且该分支发的是**不带 Sk 后缀**的 `torch_npu.npu_scatter_nd_update_`）、`device/device_op.py:530/556/557/570/580/593/594`、`models/minimax_m3/minimax_m3.py:191`。

**按 cache 平面拆分（逐层对齐 + 时间排序验证）**：

| 输入形状 | 次数/步 | 归属 | 依据 |
|---|---|---|---|
| `33998,128,512;8,2;8,512` (BF16) | **41** | 40 × 每层 SWA cache（`dsa_v41.py:367`）+ **1 × 层 20 的 long-KV**（`dsa_v41.py:445`，该 cache 物理 4D `[pages,128,1,512]`，被 `squeeze(-2)` 成 3D） | 逐层验证：L0–L19 / L21–L39 各 1，**L20 出现 2** |
| `33998,64,128;8,2;8,128` (INT8) | 3 | indexer `k_cache`（int8，宽 128） | 仅在 L2/L8/L14；`indexer.py:110` |
| `33998,64,1;8,2;8,1` (FP16) | 3 | indexer k scale cache | 同上；`indexer.py:111` |
| `33998,64,512;8,2;8,512` (BF16) | 3 | 同层的第三个 64-block 平面（**归属未定论**：`indexer.py` 只解释 2 个，可能是 compressor ring / ratio-2 latent 平面） | 见 §⑤.5 |
| `33998,128,128;8,2;8,128` (INT8) | 1 | L20 的 indexer `k_cache`（block 128） | 仅 L20 |
| `33998,128,1;8,2;8,1` (FP16) | 1 | L20 的 indexer k scale | 仅 L20 |
| `33998,128,1,512;8,2;8,1,512` (BF16, 4D) | 3 | draft 3 层的 context-KV 写入（M=8，紧跟最后一层之后 ~25.5–25.8 ms 处） | `models/deepseek_v4/dspark.py:228-243`（`_store_standard_swa_kv`）← `precompute_and_store_context_kv` |
| **合计** | **55** | | 每步 device 时间 ≈ 345 us（单算子 4.9~7.9 us） |

**是否 Engram 相关？——不是。** `npu_scatter_nd_update_sk` 在 engram 相关文件（`engram_hbm.py` = `probe_bneck/engram_host_ws_opt.localowner_v2.py`、`engram_hash.py`、`engram_gate.py`）里**没有任何调用点**；这些文件里出现的 "scatter" 只是注释中描述 all_to_all 的 scatter 阶段（例如 `probe_bneck/engram_host_ws_opt.localowner_v2.py:203`、`:729`、`:844`、`:916`、`:989`）。ScatterNdUpdateSk 全部是 KV cache 写入。

---

## ⑤ 查不到 / 不能确定（不脑补）

1. **用户表“每步次数”的分母无法复现**：rank0 上是两种分母——QBMV3/SparseFlashMla/GroupedMatmul*/HcPre/HcPost/MoeInitRoutingV3 用 98.9，RmsNorm/ScatterNdUpdateSk/MatMulV3/QuantLightningIndexerV2 用 96.8。cannbot 内部口径未公开；其余 7 个 rank 只有原始 `PROF_*` 目录（未被 msprof 分析出 op_statistic），无法核对跨 rank 平均。**纯解码步口径可直接引用本报告 §0.2 的“纯解码步实测”列。**
2. **92 次 draft pass ÷ 74 个 target 步 = 1.24 的原因未确认**：推测与投机接受率、draft 重跑或 profiler 窗口边界有关，没有直接证据。
3. **draft（M=7）是否与 target 在同一张 aclgraph 内未确认**：M=7 与 M=8 的 op 在同一时间窗内交织，但没有拿到 graph 边界信息。
4. **“尾部 3 个 wkv 属于 draft context-KV 预投影”是形状 + 代码推断**：数量、形状、时间位置、代码路径都对得上，但未做运行时打点（例如给 `precompute_and_store_context_kv` 加 marker）确认。
5. **L2/L8/L14 的第三个 64-block scatter（`33998,64,512`，BF16）对应的 cache 平面没有定论**：`indexer.py:110-115` 只解释 2 个（k int8 + scale fp16），第三个 512 宽的平面可能属于 compressor ring / ratio-2 latent 写入，未找到确凿代码行。
6. **单算子 avg 时间与用户表不一致**：例如 QBMV3 在本 profile 的 `op_statistic` avg = 21.61 us，用户表写 10.9 us；RmsNorm avg 10.97 vs 表 6.2。原因是摊薄口径不同（本报告 §2.1 用的是 M=8 解码步切片的 avg）。未逐项追溯用户表的取值方式。
7. **`dispatch_ffn_combine_w4_a8` 的可用性未验证**：仓库内零调用点，是否支持 MOE_AG=1（AllGather）、EP8、shared expert、`expert_token_nums` 静态化，全部需要实测（F1 整段融合方案成立与否取决于此）。
8. **未审计项**：`allreduceAicpuKernel`（1296 次，avg 2392 us，占 profile 31.9%）、`MatMulV2` 的 engram 路径（`8,6144;25600,6144` 307 us × 2/步）、`QuantLightningIndexerV2`（1789 us/步，aic_scalar 0.247 / mac 0.139 / cube 13.5%）未做深入归因——它们不是本任务列出的算子，但按收益排序值得下一轮单独看。

---

## 附：≤30 行摘要

1. **210 这个数不是“每层 5.25 次”**：纯解码步（74 次 aclgraph 重放，M=8）的 QBMV3 恰好 **211 次/步**；表里 210.2 = 20792 ÷ 98.9，分母把 prefill chunk 和 draft pass 一起摊进去了。
2. 211 = 40 `wq_a` + 40 `wkv` + 40 `wq_b` + 8 indexer `wq_b` + 40 共享专家 `gate_up` + 40 共享专家 `down` + 3 draft context-KV `wkv`。
3. 每层固定 5 个：attention 的 q_a / wkv / q_b 三个（5120→1280 / 5120→512 / 1280→4096）+ 共享专家两跳（5120→576 / 288→5120）。
4. 多出来的 8 个 = `index_source_layer_ids`（2/8/14/20/24/28/32/36）上的 indexer q 投影（1280→4096，与 attention wq_b 同形状）。
5. 多出来的 3 个 = dspark 3 层 draft 的 context-KV 预投影（`dspark.py:245-259`，由 `dspark_proposer.py:386` 触发），发生在 40 层循环结束之后 ~1 ms。
6. 为什么 QKV 不是一个大 matmul：checkpoint 有 `fused_qkv_a_proj`（`model.py:1218`）但运行时拆开（`model.py:501/521`），因为 kv 分支要走 kv_norm+RoPE+scatter、且 wq_a/wkv 的 quant/通信标签需各自判断；q_a 与 q_b 之间夹着 `q_norm`（不可折叠）；wq_b 在 TP8 下每卡只有 4096 列。
7. o_proj 不在 QBMV3 里：o_lora 两跳未量化（`wo_a` = TransposeBatchMatMul 47.26 us × 40 = **1.89 ms/步**，`wo_b` = MatMulV2 14.55 us × 40 = 0.58 ms/步）。
8. QBMV3 device 时间合计 ≈2.17 ms/步（M=8）+ 0.33 ms/步（draft）；单步墙钟 ≈38.5 ms。
9. **draft 贡献**：3 层 × 5 = 15 个 QBMV3/次 draft，92 次 draft ÷ 74 步 = 1.24 次/步 ⇒ **≈18.6 个 QBMV3/步（占 QBMV3 的 8.1%）**，device 时间 ≈0.33 ms/步；draft 全量 op 约 ~1.0 ms/步。
10. **Top-1 融合机会**：共享专家并入 routed 专家 grouped 路径（省 80 QBMV3 + dequant + quant ≈ **0.8~1.1 ms/步**，中高置信度）；进一步启用现成的 `dispatch_ffn_combine_w4_a8` 把整段 MoE 收成 1 kernel（**1.3~1.6 ms/步**，中低置信度，需验证 AG/EP8/shared 支持）。
11. **Top-2**：W4A8 解锁 `rms_norm_dynamic_quant`（融合算子已在 W8A8 路径使用：`dsa_v1.py:1688`），省 80 个 RmsNorm ≈ **0.25~0.40 ms/步**（中高置信度）。
12. **Top-3**：o_lora 的 `wo_a` 退化 batch matmul 精简（≈1.1 ms）或 wo_a/wo_b 量化（≈1.7 ms），中低置信度，需精度验证。
13. 次优：`wq_a`+`wkv` 合并（省 40 QBMV3 ≈0.25~0.30 ms，中置信度）；attention `wq_b` + indexer `wq_b` 合并（省 8 个 ≈0.05~0.08 ms，高置信度）。
14. **否定结论**：q_a/q_b 不能合并（中间有 q_norm）；gate/up 已经合并（576 = 2×2304/8）；`moe_gating_top_k` 无权重入参，吃不下 5120→384 的 gate matmul。
15. **RmsNorm 133 次/步** = `[8,5120]` 42（40 input_layernorm + ~2 draft main_norm）+ `[8,1280]` 40（q_norm）+ `[8,512]` 47（40 kv_norm + 4 compressor norm + 3 draft kv_norm）+ `[8,128]` 4（indexer k_norm）。未融合的是 norm→动态量化、norm→RoPE→scatter 两类组合；`rms_norm_cast`(40/步) 与 mHC 已是融合用法。
16. **ScatterNdUpdateSk 55 次/步** = 41（SWA/long-KV 平面）+ 3+3+3（L2/L8/L14 的 indexer k、k-scale、第三个 64-block 平面）+ 1+1（L20 的 indexer k/k-scale）+ 3（draft context-KV 4D 写入）。
17. **ScatterNdUpdateSk 与 Engram 无关**：engram 三个文件里没有 `npu_scatter_nd_update_sk` 调用，只有注释里的 all_to_all scatter；全部 55 次都是 KV cache 写入（`dsa_v41.py:226`/`367`/`445`、`indexer.py:110-111`）。
18. 未确认项：表中“每步”分母（96.8 vs 98.9 两种）、draft 1.24 pass/步的原因、尾部 3 个 wkv 的运行时打点、第三个 64-block scatter 的平面归属、`dispatch_ffn_combine_w4_a8` 的路径支持性（详见 §⑤）。
