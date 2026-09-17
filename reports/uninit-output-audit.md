# 「未初始化 / 未完全写入输出」审计（DSV4.1-Flash 128K 运行间非确定性的同类源）

> 审计 Agent：`/root/uninit_audit`｜2026-09-16 CST
> 纪律：**未占卡、未起停任何容器、未改任何被服务文件**。结论来自
> ①镜像源码快照（`A3-node1:/tmp/uninit_src`，由已退出的 `pypto-x-a3` 容器 `docker cp` 得到，
> 与运行中容器 `dsv41-a21-perf` 的 stock 文件 md5 一致）、②运行中容器**只读** inspect/env、
> ③线上报告与探针文件。代码基线 `vllm_ascend 0.1.dev5097+ge43cf1e9f`（`SOC_VERSION=ascend910_9391`）。

## 0. TL;DR（给主 Agent）

1. **当前生产配置下，"未完全写入"只剩一条被证实的通道家族：MoE dispatch 的未写入行**
   （`npu_moe_init_routing_v2` 的 `expanded_row_idx == -1` 行及其派生物）。线 3 已证内核本身确定、
   残留随分配历史变；主 Agent 的清零实验（A 3.0→3.4）证实它**确实进有效计算**。
   我把它做成 **env 门控 + 整文件 bind-mount**（`probe_uninit/token_dispatcher.py`，默认 0 = stock），
   并附一个**设备侧 NaN 计数探针**（`V41_UNINIT_NANWATCH=1`），用于在真 128K 服务里直接回答
   "残留到底有没有进计算"。
2. **QLI（Lightning Indexer）这条线在 A3 上可代码级排除**：`supportFd_` 只在 `ASCEND950` 置真
   ⇒ `isNeedLD` 恒 false ⇒ `ProcessLD` 永不执行 ⇒ `sparse_indices`/`candidate_topk_index`
   每行都由"行末直出"或 `CleanInvalidOutput(-1)` 写满（`cuRealAcSeq <= 0` 也有专门分支）。
   与线 1 的单算子 20 次逐位一致互相印证。**但 950/分支变化时 LD 路径的候选输出缺失是定时炸弹。**
3. **两条此前未列入清单、值得继续追的点**（A 级表 A2/A3）：
   `dynamic_scale`（`quant_mode=-1` 时**整张量不写**，线 3 实测 `min=-2.24e27`）与
   `npu_grouped_matmul` 的输出行（行数 = `active_num`，含垃圾行；`unpermute` 按 `-1` 索引读回）。
   二者与 A1 同一个补丁即可覆盖，但**单独测 MoE 零化时应把三者一起 log**。

**最需要的窗口**：A3-node2 chip7 单卡 harness（10–20 分钟）跑 `bench_pool_residue.py`
——正控（MoE，已知通道）+ 待查（QLI / HcPre）。脚本已写好，**未跑**（等主 Agent 给窗口）。

## 1. 判定前提：当前生产配置（决定"可达性"这一列）

来自 `docker inspect dsv41-a21-perf`（只读）与 `logs/perf/inner_a21.sh`：

| 项 | 值 | 对本次审计的意义 |
|---|---|---|
| `V41_QLI_NO_CANDIDATE` | **1** | 候选机制（mode 1/2）**关闭** ⇒ `candidate_topk_index` 输出为空张量，`candidates` 缓冲不被消费 |
| `V41_MOE_MASK_RANGE` | 1 | 掩码走范围比较；**未初始化行仍存在** |
| `V41_FORCE_CAND_MODE` | 0 | 无诊断覆盖 |
| MoE 通信 | `V41_MOE_COMM_ALLGATHER=1`（`MOE_AG=1`） | 走 `TokenDispatcherWithAllGather` ⇒ `active_expert_range` 压缩布局 ⇒ **大片 `-1` 行** |
| `FUSED_MC2` / `MC2` | 0 / 0 | MC2/All2All 路径**不激活**（其 `torch.empty` 输出判为 C 级） |
| `MULTISTREAM=1` `DSA_OVERLAP=1` | 是 | aux 流 + `_c2_*` ring metadata 路径**激活** |
| `SP_TOKENS=5`、`MAX_SEQS=1`、`BAT_TOKENS=2048`、`BLOCK=128` | – | 每 decode 步 6 token；prefill chunk 2048 |
| 上下文 | 128K = 11 chunk，layer20 的 S2 = 20440+2048 | 第 10 个 chunk 首次跨过 `candidate_topk_blocks×block_size = 16384` |
| layer 拓扑 | `index_source=[2,8,14,20,24,28,32,36]`，`kv_source=[2,8,14,20]`，`candidate_source=20`；`compress_ratios`：0–1 层=0，2–19 层=2，**20–39 层=1** | layer 20 是**第一个 `cmp_ratio=1` 的 index/kv source** ⇒ 同 chunk 下它的 S2 是 ratio-2 层的 2 倍（22488 vs 11244） |

**判据（父 Agent 指定的关键问题）**：某未写入区是否**可能被有效计算读到**。我把"被读到"定义为
下列任一：进入 all-reduce / RMSNorm / **带共享 scale 的量化** / 广播 / softmax 分母 / top-k 选择 /
按行 scale 的 GMM / 任何 gather-index 路径。只进 GMM `group_list` 死区、或只进被后续 mask
完全覆盖的输出行 ⇒ 记为"不可达"。

## 2. 分级命中表

### A 级：当前配置下**可达**有效计算

| ID | 位置（文件:行） | 张量 & 形状 | 写入覆盖条件 | 是否被有效计算读到 | 置信度 | 消除方案 | 性能代价 |
|---|---|---|---|---|---|---|---|
| **A1** | `ops/fused_moe/token_dispatcher.py:461`（AG dispatch）→ `device/device_op.py:101`（`npu_moe_init_routing_v2`） | `sorted_hidden_states [M*topk, 5120]`（bf16 或 int8） | **只写 `expanded_row_idx >= 0` 的行**；`-1` 行（非本 rank 专家）从不写 | ✅ `npu_moe_token_unpermute` 以 `expanded_row_idx`（含 `-1`）为 gather 索引（`token_dispatcher.py:488`）；`-1` → 越界/环绕读，靠 `topk_weights*mask=0` 压制 ⇒ **残留若为 NaN/Inf，`0×NaN=NaN` 直接进 token 输出**（线 3 实测 `max_abs=nan`） | **高**（线 3 单算子 + 主 Agent 清零实验 A 3.0→3.4） | `probe_uninit/token_dispatcher.py` + `V41_MOE_ZERO_UNINIT=1`（或 `=decode`） | `=decode`：仅 M·topk ≤ 4096 行时生效（decode 36 行 ≈ 0）；`=1`：prefill 12288 行 × 5120 bf16 ≈ 126 MB 写/层 ≈ **0.3–0.6 ms/层**（prefill 专属，不占稳态 ms/step） |
| **A2** | 同上（第 4 个返回值） | `dynamic_scale` | `quant_mode=-1` 时**整张量不写**；`quant_mode=1` 时 `-1` 行不写 | ⚠️ 条件可达（**与 A3 同一判据**：只有 GMM 按 block 读 scale 时 `-1` 行的 scale 才进计算；若 GMM 严格按 `group_list` 逐行读，则只有“不融合分支”下的整张量残留可达）：AG 路径 `quant_mode = 1 if (with_quant and dynamic_scale is None)`（`token_dispatcher.py:383`）⇒ **默认走融合量化**，`-1` 行的 scale 仍是残留；它作为 `per_token_scale` 全量传给 `npu_grouped_matmul`（`w4a8.py:520`、`base.py:302 _quant_hidden_states`）。若走"不融合"分支（`dynamic_scale` 非 None）⇒ 整张量残留、`min=-2.24e27` 级 | **中高** | 同 A1（补丁同时清 `sorted_hidden_states` 与 `dynamic_scale` 的 `-1` 行） | 同 A1（`dynamic_scale` 只有 M·topk 个元素，可忽略） |
| **A3** | `quantization/methods/w4a8/w4a8.py:508-530`（gmm1）/ `:532+`（gmm2）→ `torch_npu.npu_grouped_matmul` | GMM 输出 `[active_num, N]` | 行数 = `active_num`（**含 A1 的垃圾行**）；GMM 对这些行也写（用垃圾输入算） | ⚠️ 见下（取决于 GMM tiling 是否按 block 读行） | **中** | 同 A1（治本：让垃圾行为 0）；若 A1 已开而问题仍在，再查 GMM tiling | 0（无额外算子） |
| **A4** | `ops/fused_moe/token_dispatcher.py:488-495` | `npu_moe_token_unpermute(permuted_tokens, sorted_indices=expanded_row_idx, probs)` | 消费端：按 `sorted_indices == -1` 读 | ✅ 同一根因的消费侧；**"掩码置 0"并不能拦住 NaN**（`0×NaN=NaN`） | 高 | 同 A1 | 0 |

> A3 的机制细节：`group_list` 是 **count 模式**，第 g 组的行是 `[cumsum[g-1], cumsum[g])`。
> 线 3 实测 `expanded_row_idx = [2,7,13,35 有效，其余 32 个 -1]` 是**交错**的 ⇒
> 若 GMM 的 tiling 按 8/16 行 block 取 x（Ascend GMM 常见做法），block 内会混入 `-1` 行；
> 且 `per_token_scale` 逐行对齐，同一 block 内也会读到 `-1` 行的 scale。
> **必须在真机用"清零 A1/A2 后再测"来分离**（`bench_pool_residue.py --case moe` 已覆盖）。

### B 级：条件可达 / 需要额外条件才进有效计算

| ID | 位置 | 张量 & 形状 | 写入覆盖条件 | 可达性 | 置信度 | 消除方案 | 性能代价 |
|---|---|---|---|---|---|---|---|
| **B1** | `attention/dsa_v41.py:478-480`（`_select_sparse_indices`） | `shared.candidates [2048,1,2048]` int32（`models/deepseek_v41/model.py:446` 以 `torch.full(-1)` 分配） | `copy_` 只覆盖 `candidates.shape[0] = 本 forward 行数` 行 | 当前**不可达**（`QLI_NO_CANDIDATE=1` ⇒ mode=3，`candidates` 是自身切片、`copy_` 为空操作）。**stock（候选开）时**行数总是 = 本 forward 行数 ⇒ 未覆盖行只落在 `max_tokens` 尾部、不被消费；真正缺口是 kernel 侧 `ProcessInvalid`（整批无任务）不写候选输出（C3） | 中 | `probe_uninit/dsa_v41.py` + `V41_CAND_FILL=1`（写前把待写行填 -1） | prefill 16 MB fill ≈ 0.03–0.05 ms/层（**仅候选打开时有意义**） |
| **B2** | `csrc/.../quant_lightning_indexer.cpp:153,238`（op 输出 `at::empty`） | `sparseIndicesOut [S1,1,512]` int32；`candidateTopkIndexOut [S1,1,2048]` int32 | 行级：`needCopyOutGm && !isNeedLD` 直出（`service_vector_arch22.h:891/914`）；`cuRealAcSeq<=0` → `CleanInvalidOutput` 填 -1（`:940-947`）；LD 路径由 `ProcessLD` 写（`:1120`） | 当前**不可达**（A3 上 `isNeedLD == false`，见 §3.1）。候选输出在 **LD 路径完全缺失** ⇒ 950 上必然错 | 高（代码级） | 无需（防 950：`dsa_v41.py` 侧对 `candidates` 全行 fill -1，即 B1 补丁） | 0（当前路径） |
| **B3** | `models/deepseek_v41/engram_hbm.py:199/320-500`（**挂载版** `probe_bneck/engram_host_ws_opt.localowner_v2.py:476-1261`） | `torch.empty((ids.numel(), width))` 等十余处；host workspace + alltoallv | 按 `q.size`、`total_recv`、`maxc` 等**条件**分配/写 | 主 Agent 已用 `ENGRAM nohost` 排除主线；但**设备侧表行 + host 返回 buffer** 的残留仍在 stock 可达。engram layer = {1, 14}（**不含 20**） | 中低 | 建议保留 `nohost` 作为交付默认 | 0 |
| **B4** | `worker/model_runner_v1.py:422`（`valid_sampled_token_count_cpu`）、`:567`（`sampled_token_ids_pinned_cpu`）、上游 `vllm/v1/spec_decode/llm_base_proposer.py:883/890/1079-1080` | pinned CPU / device 小张量 | 只写 `[:num_reqs]` / `[:num_actual_reqs]`；padded draft 槽位不写 | 消费者只读有效行 ⇒ **不可达**；但它**直接决定 A**，A 的 run 间抖动与此类"未写槽位"在语义上等价（`-1` = rejected） | 中低 | 无（不建议改） | 0 |
| **B5** | `attention/dsa_v41.py:626-628`（`_c2_ring_metadata`/`_c2_complete_mask`/`_c2_source_positions`）+ `:945-985` | `[5*max_reqs]` int32 / `[max_tokens]` bool / int64 | `torch.zeros` 一次性初始化；每 forward 只写 `[:num_input_tokens]` | 消费端全部 `[:num_input_tokens]` 切片（`:1014-1016`）⇒ **不可达**；`torch.zeros` 已给确定初值 | 高（不可达） | 无需 | 0 |
| **B6** | `attention/dsa_v41.py:619-620` `_slot_mapping` / `_slot_mapping_2d` | `[max_tokens]` int64 / `[max_tokens,2]` int32 | `torch.full(-1)`；仅写 `[:num_input_tokens]`；padding/dummy 行 `valid=False` ⇒ 写 `(-1,-1)`（`:804-830`） | 消费端切片一致；`-1` 行被 store 侧跳过 ⇒ **不可达** | 高 | 无需 | 0 |

### C 级：当前配置下**不可达**（列出以避免重复排查）

| ID | 位置 | 为什么现在不可达 | 什么条件会让它变可达 |
|---|---|---|---|
| C1 | QLI `ProcessLD`（`service_vector_arch22.h:993-1120`） | `isNeedLD` 需要 `s2Start>0 \|\| s2LoopEnd<s2BlockNum`，而 S2 跨核切分（`AssignByBlock`）只在 `supportFd_==true`（`..._metadata_aicpu.cpp:328-336`，仅 `ASCEND950` 且 `maxS2Size > 2*2048`）时执行 | 换 950 / 分支变化 ⇒ 候选输出永不写出（**定时炸弹**） |
| C2 | MC2 / All2All / Fused MC2 的 `torch.empty`：`ops/fused_moe/moe_utils.py:57/115/119`、`moe_comm_method.py:539`（`dispatch_ffn_combine` 的 `out=`）、`prepare_finalize.py:217`、`token_dispatcher.py:296` | `FUSED_MC2=0`、`MC2=0`、`use_sequence_parallel_moe=False`（DP=1） | 打开 FUSED_MC2 / MC2 / DP>1 |
| C3 | QLI `ProcessInvalid`（`kernel_arch22.h:670-674, 682-700`）：整批无任务时**只写 `sparse_indices`，完全不写候选输出** | 只有"整批 0 行"才会进（本部署不会发生） | 空 batch / 全 dummy 步 |
| C4 | `npu_hc_pre_v2` / `npu_hc_post` / `npu_scatter_nd_update_sk` | 线 3 已 30/30 逐位一致（含跨进程 3 次）；三者语义上"按索引/全量写" | 仅当输入含未初始化区时 |
| C5 | `npu_quant_lightning_indexer_v2_metadata` 输出（`at::empty(1024)`） | kernel 用 `*metadataPtr = {}` 把**整个结构体（864 个 int32）**清零后再逐项写（`..._metadata_aicpu.cpp:915-953`）；尾部 160 个 int32 从不写但也从不读（消费索引 < 864） | 若 kernel 侧改读尾部 |

## 3. 逐条证据

### 3.1 QLI 在 A3 上确实"写满"（本节最费时的结论）

1. **输出分配**：`quant_lightning_indexer.cpp:153` `sparseIndicesOut = at::empty(outputSize, ...)`；
   `:238` `candidateTopkIndexOut = at::empty(candSize, ...)`（均为未初始化分配）。
2. **行级写入路径**（arch22 kernel）：
   * 正常行：`service_vector_arch22.h:888-920`，`needCopyOutGm = (blockS2StartIdx_==0) && isS2End`，
     随后 `CopyOut(indiceOutGm[offset + cuS1Idx*sparseCount], ..., sparseCount)` ⇒ **整行 512 槽**；
     槽内不足 topk 时由 `InitSortOutBuf(-inf,-1)` 与 `MergeSort` 的 `ifExhaustedSuspension` 填 -1。
   * `cuRealAcSeq <= 0`：`:940-947` → `CleanInvalidOutput()` → `Duplicate(-1, sparseCount)`。
   * 整批 0 行：`kernel_arch22.h:670-674` → `ProcessInvalid()`（**只写 sparse_indices**，C3）。
   * LD 路径：`:1120`（**候选输出缺失**，C1）。
3. **LD 在本机不激活**：`supportFd_` 默认 false（`..._metadata_aicpu.h:285`），只在
   `ProcessSocVersion()==ASCEND950` 且 `maxS2Size > fdToleranceRatio*s2BaseSize_` 时置真；
   容器 env `SOC_VERSION=ascend910_9391` ⇒ `ASCEND910B`。且 `AssignBlockToCore` 只有
   `supportFd_` 才调 `AssignByBlock`（`:772-806`），`curS2Idx` 不变 ⇒ 每核都拿"整行"、
   `isNeedLD` 恒 false。
4. **单算子实测佐证**（线 1）：真实几何下 mode=1/mode=3 各 20 次逐位相同。
   ⚠️ 该 harness 对"未写入区被读到"**天然不敏感**（同一调用序列的残留内容稳定），
   所以"20 次相同"只排除了"内核写入部分非确定"。
5. 内核自带的 3 处历史缺陷注释（`service_vector_arch22.h:541/557/590/686`、
   `quant_lightning_indexer_v2_vector.h:189`）**全部位于 mode=1（`ProcessCandBlockTopk`）路径**，
   当前 `QLI_NO_CANDIDATE=1` 已把它关掉 ⇒ 与"cand3 臂仍非确定"一致。

### 3.2 MoE 通道的完整链路（A1→A4）

```
AG dispatch (token_dispatcher.py:461)
  npu_moe_init_routing_v2(hidden, topk_ids(全局专家号), active_expert_range=[rank*48,(rank+1)*48],
                          quant_mode=1 融合量化)
    ├─ sorted_hidden_states : 只有本地专家的行被写；其余行 = 内存池残留     ← A1
    ├─ dynamic_scale        : 无效行 scale 不写（quant_mode=-1 时整张量不写）← A2
    └─ expanded_row_idx     : 无效位置 = -1
  → _quant_hidden_states(base.py:302)/apply_gmm1(w4a8.py:508)             ← A3
      dynamic_scale is None ⇒ 对 sorted_hidden_states 全张量 npu_dynamic_quant
      （残留含 Inf/NaN ⇒ 该行 scale 也含 ⇒ per_token_scale 进 GMM）
  → npu_grouped_matmul(..., group_list=count 模式, per_token_scale=全量)
  → npu_moe_token_unpermute(sorted_indices=expanded_row_idx 含 -1, probs=masked)← A4
```

关键判据：`0 × 残留` 只有在残留有限时才等于 0；**NaN/Inf 会穿透掩码**。
线 3 在污染实验里实测到过 `max_abs = nan`（去掩码版），主 Agent 的清零实验改善了 A 的下限/上限。

### 3.3 为什么"类 MoE"的其它点没被列成 A 级

* `npu_moe_token_permute` / `npu_moe_finalize_routing`：本配置（AG）不用前者做 dispatch
  （只在 All2All 路径 `token_dispatcher.py:749` 用），后者本仓刻意回避
  （`moe_comm_method.py:249-255` 注释：有精度问题）。
* `npu_grouped_matmul` 的 out 是 op 内分配（无 Python `out=`），行数 = `x.shape[0]`、**每行都写**，
  只是"写的内容来自垃圾输入" ⇒ 归到 A3，而不是独立点。
* `dequant_swiglu_quant` / `rms_norm_cast` / `npu_swiglu` / `npu_dynamic_quant`：逐元素 ⇒ 全写。
* `npu_scatter_nd_update_sk` / `store_kv_block`：语义就是"按索引写"，未覆盖位置是**设计上的旧值**
  （KV cache 历史内容），消费者按 `seq_lens` 裁剪 ⇒ 不属于本类。

## 4. 微基准：把"内存池残留会不会改变有效输出"做成可复现实验

**脚本**：`A3-node1:~/projects/dsv41/probe_uninit/bench_pool_residue.py`（同步到
`A3-node2:~/projects/dsv41-workspaces/wt-graph/layer_bench/probe_uninit/`；**尚未运行**）。

### 4.1 设计要点（三处必须做对，否则假阴性）

1. **同输入重复调用测不出残留**：同一调用序列下所有未写入区的内容**稳定**
   （正是线 1 "QLI 20 次逐位相同"的成因）。必须在两次调用之间**改变内存池状态**。
2. **污染必须命中同一 size class**：先 `torch.empty(与被测 op 输出同形)` → 填 pattern（NaN/±Inf/0x5A/…）
   → 释放 → **立刻**调 op，使 op 的 `at::empty` 复用刚释放的 block（NPU caching allocator LIFO 复用）。
3. **判据必须是 `torch.equal`（bitwise，NaN==NaN 视作相等）**，并打印 `n_nonfinite`。
   用 `a != b` 会把"同一个 NaN"误报成不同（线 3 已踩过）。

### 4.2 case 与期望

| case | 内容 | 期望 | 意义 |
|---|---|---|---|
| `moe` | 真实 EP 形状（E=384、active range=[48,96)、K=6、M=6）→ init_routing → dynamic_quant → GMM/scale → unpermute | **DIFF**（正控） | 校准脚本灵敏度；DIFF 才说明"污染确实能进有效输出" |
| `moe_nomask` | 同上但去掉 mask | DIFF（更大，可能 NaN） | 复现线 3 的 `max_abs=nan` |
| `qli` | layer20 几何：S1=2048、S2=22488、cmp_ratio=1、topk=512、mode=3、PA_BBND int8 | 预期 same | 若 DIFF ⇒ **QLI 输出有未写入槽被读**（推翻 §3.1 的代码结论） |
| `hcpre` | M=6、`x[6,4,5120]`、`hc_fn[24,20480]` | 预期 same | 与线 3 的 30/30 一致互为印证 |
| `--scan` | 线性扫污染元素数（1 → 2^22） | 给出**最小触发体积** | 回答"最小污染体积" |

### 4.3 成本与运行方式

```bash
# A3-node2 宿主机（需先起 chip7 harness 容器；脚本含 5 次 warmup，单 case 约 30–60 s）
bash /home/user/projects/dsv41-workspaces/wt-graph/layer_bench/start_container.sh
bash /home/user/projects/dsv41-workspaces/wt-graph/layer_bench/probe_uninit/run_pool_residue.sh moe
bash /home/user/projects/dsv41-workspaces/wt-graph/layer_bench/probe_uninit/run_pool_residue.sh qli
bash /home/user/projects/dsv41-workspaces/wt-graph/layer_bench/probe_uninit/run_pool_residue.sh all
bash /home/user/projects/dsv41-workspaces/wt-graph/layer_bench/stop_container.sh   # 释放 chip7
```

全流程 < 10 分钟（含容器起停）；`qli`/`hcpre` case 用随机输入，**不需要权重**。

### 4.4 结果怎么读

* `moe` DIFF + `qli`/`hcpre` same ⇒ 通道**只存在于 MoE dispatch**，应集中火力清零 A1/A2，
  并在 8 卡上做 `V41_MOE_ZERO_UNINIT=decode` 的 A 中位 A/B。
* `qli` DIFF ⇒ 立即重跑线 1 的 QLI 单算子测试，**并在两次调用之间加污染**（他们的 harness 缺这一步）。
* 全部 same ⇒ 本 harness 的分配序列不足以再现（**不等于生产不可达**）；改用
  `V41_UNINIT_NANWATCH=1` 在真服务里看非有限计数。

## 5. 消除方案（3 个，全部默认关闭、env 门控、整文件 bind-mount）

| # | 文件 | env | 语义 | 性能代价 | 风险 |
|---|---|---|---|---|---|
| **P1** | `probe_uninit/token_dispatcher.py`（基于当前挂载的 `probe_moe_mask/token_dispatcher.py`） | `V41_MOE_ZERO_UNINIT=1`（全部）／`=decode`（仅 M·topk ≤ `V41_MOE_ZERO_UNINIT_MAX_ROWS`，默认 4096） | AG dispatch 之后把 `expanded_row_idx < 0` 的行（`sorted_hidden_states` 与 `dynamic_scale`）显式置 0，**去掉"任意残留"**；`=0`（默认）与 stock 逐位一致 | `=decode`：36 行 ≈ 0；`=1`：prefill 每层约 126 MB 写（0.3–0.6 ms/层，仅影响 TTFT，不影响稳态 ms/step） | 新增一次 `index_put_`（无 host 同步）；int8 融合路径上置 0 = 零激活，本身不参与输出 |
| **P2** | `probe_uninit/dsa_v41.py`（基于镜像 stock） | `V41_CAND_FILL=1` | 写 `shared.candidates` 前先 `fill_(-1)`，把"未覆盖行"从任意残留变成合法语义值；默认 0 不变 | prefill 16 MB fill ≈ 0.03–0.05 ms/层；decode 可忽略 | **仅当 `V41_QLI_NO_CANDIDATE=0`（候选开）时有意义**；当前默认下是 no-op |
| **P3** | 同 P1 文件 | `V41_UNINIT_NANWATCH=1`（+`V41_UNINIT_NANWATCH_EVERY=N`，默认 200） | **设备侧计数**：统计 `-1` 行里的非有限值个数，累计后每 N 次调用打印一次（含 1 次 `.item()`） | 每次 dispatch 一次 `isfinite().sum()`：decode 36×5120 ≈ 20 µs/层（诊断用，默认关） | 纯诊断，不改数值 |

挂载命令（整文件，一行一条）：

```bash
# P1 + P3（MoE 未写入行清零 / NaN 计数）—— 与现有 MOE_MASK 补丁同文件（本文件已含该补丁）
-v $P/probe_uninit/token_dispatcher.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:ro
# P2（候选缓冲 fill）—— 该文件当前未被挂载，新增一条
-v $P/probe_uninit/dsa_v41.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:ro
```

与既有的挂载共存（`serve_a21.sh` 风格）：

```bash
UNINIT_ZERO=${UNINIT_ZERO:-0}   # 0 | 1 | decode
EXTRA_ENV="V41_MOE_ZERO_UNINIT=$UNINIT_ZERO"
```

**注意**：`token_dispatcher.py` 已有一条来自 `probe_moe_mask/` 的挂载（MOE_MASK 补丁）。
P1 文件是**在 MOE_MASK 补丁之上**生成的 ⇒ 只挂 P1 这一份即可（`V41_MOE_MASK_RANGE=1` 仍生效）。
上游若更新了 `probe_moe_mask/token_dispatcher.py`，重跑 `bash probe_uninit/make_patches.sh`
（幂等、锚点唯一自检）即可重新生成。

与数值语义的关系：P1/P2 只改**未写入区**的内容，不改任何被写过的元素 ⇒ 无残留污染时与 stock
**逐位相同**（可用 `=0` 与 `=1` 两臂各自连发 N 次逐位比对来验证）；不引入任何 host 同步；
`index_put_`/`fill_` 是图内合法算子（算子数变化 ⇒ 需重捕获一次）。

## 6. 未验证项 / 风险

1. `bench_pool_residue.py` **未运行**（纪律：先要窗口）。`qli` case 的入参（dtype/scale 形状、
   metadata 属性组合）按 `indexer.py:226-270` 与 `dsa_v41.py:910-935` 反推，真机可能需要
   按报错修正 1–2 处（例如 `query_scale` 的 dtype 需 fp32/fp16 二选一）。
2. **A3（GMM 是否 block 粒度读越界行）没有代码级证据**：`torch_npu.npu_grouped_matmul` 实现
   不在本仓（在 torch_npu/CANN 侧），需 `--case moe` 的 pair 结果 + 真机 dump 判定。
3. **未覆盖** `npu_sparse_flash_mla` 的输出覆盖（线 3 构造 metadata 失败 EZ1008，我也未能在
   预算内反推合法 metadata）。建议从真实调用点（`dsa_v41.py:543`）dump metadata 张量再喂单算子。
4. **未覆盖** draft（DSpark）3 层自己的 MoE dispatch。它是**同一份代码**
   （`TokenDispatcherWithAllGather`）⇒ P1 自动覆盖；但"draft 残留是否比 target 更影响 A"需单独统计
   （`V41_UNINIT_NANWATCH=1` 的 `calls` 计数可把 target（40/步）与 draft（3/步）分开看）。
5. **未覆盖** `hc_pre/hc_post` 的 out 参数变体（若 Python 侧有 `out=` 传入路径，未写区会落在调用方
   buffer 上）；线 3 的 30/30 只覆盖 harness 形态。
6. 与"layer 20 / chunk 10"形态的关系：**我没有找到能解释该形态的未初始化点**
   （QLI 已排除、MoE 是全局性的）。若 P3 在真 128K 上返回 `nonfinite_total == 0`，
   应把注意力转回"同步/竞态"类，而不是继续找未写入区。

## 7. 附录：数据来源与复现命令

```bash
# 1) 镜像源码快照（无需起容器；用已退出的同镜像容器 docker cp，只读）
sudo -n docker cp pypto-x-a3:/vllm-workspace/vllm-ascend/vllm_ascend /tmp/uninit_src/vllm_ascend
sudo -n docker cp pypto-x-a3:/vllm-workspace/vllm/vllm            /tmp/uninit_src/vllm
sudo -n docker cp pypto-x-a3:/vllm-workspace/vllm-ascend/csrc     /tmp/uninit_src/qli_csrc
# 一致性自检：running 容器 vs 快照（stock 文件）
sudo -n docker exec dsv41-a21-perf md5sum /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py
md5sum /tmp/uninit_src/vllm_ascend/attention/dsa_v41.py     # 两份都应为 6e60fc4d40e486e75a3bac5588c3b77f

# 2) 关键行定位
rg -n "npu_moe_init_routing_v2|expanded_row_idx" vllm_ascend/ops/fused_moe/token_dispatcher.py
rg -n "isNeedLD|supportFd_|CleanInvalidOutput|ProcessInvalid|needCopyOutGm" \
   csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/*.h \
   csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/*.cpp
rg -n "at::empty" csrc/attention/quant_lightning_indexer_v2/torch_extension/csrc/quant_lightning_indexer.cpp

# 3) 生成/复核补丁（幂等）
bash probe_uninit/make_patches.sh
diff -u probe_moe_mask/token_dispatcher.py probe_uninit/token_dispatcher.py
```

引用的既有证据（未重复验证）：`reports/nondeterminism-rootcause.md`、
`reports/scalar-bound-op-audit.md`、`reports/three-line-plan.md`（MoE 掩码机制）、
`A3-node2:wt-op/op_line/reports/a-third-source-init-routing-uninit.md`（线 3 单算子）、
`A3-node2:wt-graph/reports/correctness-line.md`（HCCL 门 + 单算子排除表）。
