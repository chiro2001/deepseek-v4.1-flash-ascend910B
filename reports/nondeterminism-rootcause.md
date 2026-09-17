# DeepSeek-V4.1-Flash W4A8 长上下文输出非确定 — 离线代码根因调查

> 调查者：子 Agent `/root/nondet_rootcause`｜2026-09-16 CST
> 约束：**未占卡、未起停容器、未改任何代码**。所有代码证据均取自运行中的容器 `dsv41-a21-perf`
> （只读 `cat`）与 A3-node1 宿主机上的探针副本；实测事实（≤16384 确定 / >16384 非确定）沿用主 Agent 结论，未推翻。
> 代码版本：容器内 `/vllm-workspace/vllm-ascend` @ `e43cf1e9f`；挂载覆盖见 §7。

---

## ① 结论速览（按证据强度排序）

| # | 假设 | 置信度 | 一句话依据 | 反证/待验证 |
|---|---|---|---|---|
| **H1** | **mode=1（源层 layer 20）块级 top-k 的"块选择结果"不是本 forward 数据的纯函数**：块级路径大量依赖 UB 临时区（`tmpBuf_` 多路复用）、块级 sort/merge 与位置级 sort 并行、以及"上轮残留"位型；这些缺陷在 **总块数 ≤ candidate_topk_blocks 时完全不可见**（此时候选集合=全部块，过滤器恒等 ⇒ 任何块级错误都不改变结果），只有 >2048 块时才改变被选中的块集合 | **0.45** | kernel 自身注释记录过 3 次同类实测缺陷（R6 `candBuf` 跨行污染 / R10 尾 tile stale 块分数挤掉真实块 / P3 `MergeSortVecCopy`），修复均为"点杀"；mode=1/2 的块级路径**没有 LD 变体实现**，说明该路径验证矩阵不完整（`CopyOutCandTopkIndex` 只在 `!isNeedLD` 分支） | 未见"同一输入、同一次内核调用内"的非确定证据；要成为根因，残留内容必须随调用历史变化（UB 残留来自**上一次内核调用**的数据 ⇒ 天然随请求历史变化，见 §4.3 论证） |
| **H2** | **消费层（mode=2，layer 24/28/32/36）读到了不属于本 forward/本行的候选行**（写覆盖缺口 / 行错位 / 图模式地址），因为 `reset()` 是 no-op、候选输出用 `at::empty`（未初始化）、消费端只有 **shape 校验**没有任何值域/有效性校验 | **0.30** | `model.py:257-260` reset 为 no-op；`model.py:613-619` 只用 `-1` 初始化一次；`quant_lightning_indexer.cpp:228-238` 输出是 `at::empty`；`indexer.py:193-194` 只校验 shape/dtype；`indexer.py:175` 注释把"仅在同一 forward 内有效"写成了**契约**，但代码里**没有任何强制** | 我逐条追了 4 条写路径，**单请求、无 padding、非 LD** 的实测形态下每个真实行都被写（含 `CleanInvalidOutput` 写 -1），**未发现缺口** ⇒ H2 需要 padding/空批次/其它边缘形态才能触发（唯一确证缺口见 §2.5 `ProcessInvalid`） |
| **H3** | 挂载的 **非 stock 补丁**（`engram_hbm.py=localowner_v2`、`probe_dsa/dsa_v1.py`、`probe_moe/ascend_forward_context.py`、`sitecustomize.py`）引入 host 侧/时序非确定，被候选门"放大"成可见输出差异 | **0.15** | `docker inspect` 显示这些文件是主 Agent 的探针文件直接挂载进容器（§7）；Engram host 路径含跨 rank all-to-all 与共享 host workspace | 阈值精确等于 `2048×8+1` 太"巧"，纯 host 竞态很难解释；且 ≤16384 完全确定（8~9 chunk 差异不足以让竞态开关翻转）⇒ 更可能是**放大器**而非**主因** |
| **H4** | 上游数值离散化放大：`quantize_indexer_query`（fp16 scale + `nearbyint` 四舍六入五成双）与 `prepare_indexer_indices`（bitcast 排序）把极小 fp 差异放大成 int8 离散差异 | **0.10** | `prepare_indexer_indices.py:38-58`、`quantize_indexer_query.py:26-33` | 单独不足以解释"首 token 从 `'# '` 变 `<ds_s`"这种量级的变化；只能作为放大环节 |

**⚠️ 一个决定性的逻辑约束（请主 Agent 优先接受这条）**：
块级过滤器在语义上是 **"上界保持"**的——`blkScore = 块内 amax ≥ 块内任意位置分数`（`service_vector_arch22.h:546` 的 `BlockReduceMax`），
所以"边界处并列/差一个块"这类**轻微扰动不可能把高分位置过滤掉**。
⇒ 观测到的"首 token 从正常 `'# '` 直接退化成特殊 token `<ds_s`"**无法**用 tie-break/数值噪声解释，
**必须有"整行/大范围错误的候选行"进入 mode=2**（未初始化内存、错行、或大量 `-1` 被 stale 高分占据）。
这直接把根因收敛到 **H1（块级 top-k 输出大范围错误）** 与 **H2（读到非本 forward 的候选行）** 两条。

---

## ② 逐条证据（文件:行号 + 原文）

### Q1 候选 buffer 的写入覆盖范围

**写点只有一处，写的是 op 输出张量的行数，不是整个 buffer：**

`vllm_ascend/attention/dsa_v41.py:451-481`
```python
    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            return shared.topk_indices[: hidden_states.shape[0]]
        ...
        selected, candidates = attn.indexer.select(
            hidden_states, qr, positions, cos, sin,
            source_layer.kv_cache[0], metadata.indexer.cache,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidates=shared.candidates[: hidden_states.shape[0]],      # 476
        )
        shared.topk_indices[: selected.shape[0]].copy_(selected)          # 478
        if self.role.is_candidate_source:                                 # 479
            shared.candidates[: candidates.shape[0]].copy_(candidates)    # 480
        return shared.topk_indices[: selected.shape[0]]                   # 481
```

* 写入范围 = **`candidates.shape[0]`**（op 输出行数）= `query.size(0)`（见 §Q2 的 wrapper）。
  不是整个 2048 行 buffer ⇒ **未覆盖行保留上一轮/任意内容**（buffer 是持久张量，见下）。
* buffer 是**持久分配 + 一次性初始化**：
  `models/deepseek_v41/model.py:610-627`
  ```python
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens     # 612
        candidate_buffer = torch.full(                                     # 613
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer                    # 619
        self.shared_attention_state = DeepseekV41SharedAttentionState(      # 620
            self.topk_indices_buffer,
            candidate_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state  # 626
  ```
  实测配置 `--max-num-batched-tokens 2048` ⇒ buffer = `(2048, 1, 2048)` int32 = 16 MiB（§7 命令行）。

**chunked prefill 每个 chunk 的 `query.shape[0]`：**

* 服务器配置 `--max-num-batched-tokens 2048`、`--max-num-seqs 4`、`--block-size 128`、`--max-model-len 1048576`
  ⇒ 长 prompt 每 step 最多 2048 token；与 decode（`num_speculative_tokens=5` ⇒ 6 token/请求）混批时
  为 `2024+24` 这类组合（主 Agent 观测一致）。
* `hidden_states.shape[0]` = 本次 forward 的 **padded token 数** `num_tokens_padded`
  （runner：`worker/model_runner_v1.py:2414-2426` 用 `num_tokens=num_tokens_padded` 建 forward context）。
* ⚠️ **killer 细节（对 Q1 关键）**：`cudagraph_mode=FULL_DECODE_ONLY` ⇒ **prefill chunk 走 eager**，此时
  `num_tokens_padded == 真实 token 数`；decode 走 FULL graph，padding 通过**插入 dummy request** 实现：
  `worker/model_runner_v1.py:968-975`
  ```python
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded
            # Do not insert if the last value already equals the num_tokens
            if query_start_loc.np[num_reqs_padded] < num_tokens_padded:
                # Insert a dummy request instead of change the last value directly
                query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded = num_reqs_padded + 1
  ```
  且 `dsa_v41.py:714-716`（`_build_batch_metadata`）把 dummy request 的 seq_lens 清零：
  ```python
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
  ```
  ⇒ dummy 行在 QLI kernel 里走 `actS2Size == 0` 分支 → **`CleanInvalidOutput` 会把候选行填 -1**
  （见 Q3），所以 **padded 行也是"被写过"的**。

**内核侧的行覆盖（这是 Q1 的核心，逐条核过）：**

| 路径 | 文件:行 | 行为 | 候选行是否被写 |
|---|---|---|---|
| 正常行 | `service_vector_arch22.h:888-891`：`bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq; bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End; if (needCopyOutGm && !info.isNeedLD) {` | 行末直出 sparse_indices，随后 `CopyOutCandTopkIndex`（`service_vector_arch22.h:916-920`） | ✅ 写满 `candidateTopkBlocks` 个槽 |
| 行有效长度为 0（含 padding/dummy request） | `kernel_arch22.h:568-573`（`curActSeqLenIsZero`）→ `kernel_arch22.h:735-737`（`DealActSeqLenIsZero`）→ `kernel_arch22.h:421-448`（TND 分支）→ `CleanInvalidOutput`（`service_vector_arch22.h:320-341`） | 填 -1 | ✅ 写 -1（见 Q3 代码） |
| 整批无任务（`coreZeroEnable == 0`，即 0 核无任务） | `kernel_arch22.h:670-674` `if (coreZeroEnable == 0) { ProcessInvalid(); return; }` + `kernel_arch22.h:682-700`（`ProcessInvalid` 体） | **只** `InitGlobalMemory(indiceOutGm[…], …)`（sparse_indices），**完全没有候选输出** | ❌ **不写**（唯一确证缺口；该形态下消费端同样无有效行，理论上不读，但污染已经被 `copy_` 写进持久 buffer） |
| LD（S2 跨核切分）路径 | `service_vector_arch22.h:891` 的 `&& !info.isNeedLD`；`ProcessLD`（`service_vector_arch22.h:993-1120`）只搬出 `indiceOutGm` | **候选输出在 LD 路径完全缺失** | ❌ 不写（本机不激活，见 §3 排除项 E2；一旦 950/分支变化即成为正确性定时炸弹） |

**结论（Q1）**：
1. 写入**不是**整 buffer，只写 `candidates.shape[0]` 行；未覆盖行 = 上一轮/任意内容 ⇒ **残留可读**。
2. 但**对于一个请求、无 padding、非 LD 的实测形态，所有真实行与 padded/dummy 行都被写了**（真实行由行末直出，dummy 行由 `CleanInvalidOutput` 填 -1）
   ⇒ **"consumer 读到上一轮残留"在实测形态下我无法用代码证明会触发**（这是与主 Agent 假设 2 的差异，必须靠实验判定，见 §5 E2）。
3. `ProcessInvalid` 是**唯一确证的"未写候选"路径**，它同时说明：**"候选输出未被写"这件事会静默污染持久 buffer**（因为 `dsa_v41.py:480` 无条件全行 copy）。

### Q2 consumer 层的读取范围

**读的也是行切片，与源层写范围同形：**

* `dsa_v41.py:476` `candidates=shared.candidates[: hidden_states.shape[0]]`（传给 op，作为 **mode=2 的输入**）。
* 消费端形状校验（唯一的校验）：
  `models/deepseek_v41/indexer.py:193-195`
  ```python
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidates.shape != candidate_shape or candidates.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
  ```
  ⇒ **要求 `candidates.shape[0] == query.shape[0]`**，否则直接抛错（不会静默错位）。
* 内核对候选输入的寻址（源=消费同一公式）：
  `quant_lightning_indexer_v2_kernel_arch22.h:645-646`
  ```cpp
        candidateOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.candidateTopkBlocks +
                                 runInfo.n2Idx * constInfo.candidateTopkBlocks;
  ```
  消费端读取 `service_vector_arch22.h:466-472`
  ```cpp
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset +
                                                    cuS1Idx * constInfo.candidateTopkBlocks],
                             copyInParams, padParams);
  ```
  ⇒ 行号 = `cu_seqlens_q[b] + 行内偏移`，与源层写入用**同一个公式**；TND 布局下 token 连续打包，两者一致。
  **chunked prefill 下是否一致：一致**（同一 forward 内 `n`、`cu_seqlens`、行序完全相同；跨 chunk 各行都由本 chunk 重写）。

### Q3 `-1` 的语义与有效性

* 定义：`quant_lightning_indexer_v2_common_arch22.h:89` `static constexpr int INVALID_IDX = -1;`
  `quant_lightning_indexer_v2_proto.h` 文档：值域 `[0, numBlocks)` 或 `-1`（无效槽）。
* **mode=1 写 -1 的路径**（两条）：
  `service_vector_arch22.h:320-341`（`CleanInvalidOutput`）
  ```cpp
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {                    // 333
        uint64_t candOffset = static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount *
                              constInfo_.candidateTopkBlocks;                    // 334-335
        ... Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks); // 338
        QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
    }
  ```
  以及 pad 块（`service_vector_arch22.h:585-620`）：`isPad` 纯 int32 向量链把 pad 块 idx 置 -1、score 置 `-inf` 位型；
  pin 最新块置 `+inf`（`623-660`）。
* **mode=2 如何"处理" -1**：候选行被 `Cast` 成 fp32 后与槽号一起降序排序（`service_vector_arch22.h:466-486`），
  再用 `CountGE` 二分找出落在本 tile 块号窗口 `[tileBlockBase, tileBlockBase+tileBlkNum)` 的区间
  （`489-513`）；`-1` 排在最尾部、永远不落入窗口 ⇒ **不会产生错误命中**。
  候选块外位置不是置 `-inf`，而是降级到 `NEG_HUGE=-1e30`（`quant_lightning_indexer_v2_vector.h:26`；
  使用点 `service_vector_arch22.h:844-853`），语义 = "不足 topk 时可达的候选外位置作为填充入选"（R11 leak，对齐参考实现）。
* **"initial prefill 第一个 chunk 就来读（还没写过）"**：
  `indexer.py:197-203`
  ```python
        if query.shape[0] == 0:
            selected = torch.full((0, topk), -1, dtype=torch.int32, device=query.device)
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates
  ```
  这只是空行保护。**"读到全 -1 行"在 kernel 里没有断言/校验**：若整行 `-1`，`BuildCandidateMask` 的
  `CountGE(...)` 给出 `lo=hi=0` ⇒ `acc` 保持 `POS_INF` ⇒ `isOut` 全 1 ⇒ 整行分数降级为 `NEG_HUGE`，
  最终 topk 里塞进"候选外可达位置"（R11 leak）⇒ **输出退化为规范化错误而不是崩溃**。
  代码里**没有任何 runtime 校验**能发现"候选行非法/未初始化"（只有 shape/dtype 校验，见 Q2）。

### Q4 `shared_attention_state.reset()` 的调用时机与跨 forward 残留

* 调用点：`models/deepseek_v41/model.py:820`（`DeepseekV41Model.forward` 开头）`self.shared_attention_state.reset()`。
* 实现是 no-op（stock 与挂载探针版一致）：
  `models/deepseek_v41/model.py:253-260`
  ```python
    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None
  ```
* 契约靠注释与设计约束，**没有任何强制**：
  - `indexer.py:173-176`（docstring）
    ```
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.
        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. ...
    ```
  - 设计文档 `csrc/attention/quant_lightning_indexer_v2/docs/qli_v2_two_level_topk_design.md:52`
    > 7. **候选块跨层共享**，同一 forward 内 KV 状态不变 → 块划分一致；**跨 step 复用无效**（语义约束）。
* **跨 forward 残留是否可能被读到**（逐形态回答）：
  - **长请求结束、短请求开始**：短请求的每次 forward 都会重写自己的 `[:n]` 行（含 dummy 行 -1）；
    只有"未覆盖行"才会留下长请求的内容，而消费端只读自己 forward 的行 ⇒ **在本机实测形态下不可达**。
  - **decode（6 token/step）与 prefill chunk（2024/2048）交替**：行数不同，但每次 forward 的行都自己重写；
    `shared.candidates` 行 `[0, n)` 与 `shared.topk_indices[0, n)` 都是本 forward 的语义行 ⇒ **一致**。
  - **真正风险点**：`at::empty` 输出 + no-op reset + `copy_` 全行拷贝 ⇒ **任何"未覆盖行"都会被静默注入持久 buffer**，
    且下一轮若该行仍未被覆盖会**继续存活**。这就是 H2 的结构性隐患（唯一确证触发形态见 Q1 的 `ProcessInvalid`）。

### Q5 qli/smla metadata 的 device 同步与 per-chunk 正确性

* metadata buffer 在 builder 构造时**一次性分配**，每个 `build()`（= 每个 forward）**重算并 copy_ 进同一 buffer**：
  `dsa_v41.py:620-628`
  ```python
        self._smla_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
  ```
  `dsa_v41.py:910-935`（QLI metadata 构建，**每 forward 重算**）
  ```python
            def build_qli_metadata() -> None:
                value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                    int(_config_value(text_config, "index_n_heads")), 1,
                    int(_config_value(text_config, "index_head_dim")), index_topk, 2,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),   # 917 行级前缀（每 forward 变）
                    seqused_k=coordinates["cache_seq_lens"],                     # 918 每 forward 变
                    cmp_residual_k=residual,
                    batch_size=num_reqs, max_seqlen_q=..., max_seqlen_k=...,
                    layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=ratio)  # 925 mask_mode=3
                self._qli_metadata.copy_(value)
  ```
* 提交/等待链（**不是** Python 时序，而是事件）：
  `worker/device_metadata.py:68-133`
  ```python
        self._inputs_ready.record(torch.npu.current_stream())   # 95 输入(host→device 拷贝)完成后才放行 metadata 算子
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self._inputs_ready)
            if self._has_reuse_fence:
                self.stream.wait_event(self._buffer_reusable)   # 99 上一 forward 用完才复用 buffer
            ... task.run(); self._stage_ready[frontier].record(self.stream)   # 108
    def wait(self, stage, group_id):                            # 113
            stream.wait_event(self._stage_ready[frontier])      # 120
    def release(self):
        self._buffer_reusable.record(torch.npu.current_stream())# 130
  ```
  消费点：`indexer.py:225`（`wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))`）、
  `dsa_v41.py:540-542`（ATTENTION 阶段）、`dsa_v41.py:413`（COMPRESSOR 阶段）。
  runner 侧：`worker/model_runner_v1.py:2414`（attach）、`2465-2466`（release）、`3567-3570`（submit，FULL graph 时带 `batch_descriptor` 用 `ExternalEvent`）。
* **结论（Q5）**：metadata **不是** per-chunk 复用的；正常路径下不会读到上一 chunk 的 metadata。
  唯一"共享"是 `_publish_task` 的 dedupe（`dsa_v41.py:691-708`）：同 ratio 的多个 cache group 共享**同一个 buffer**——
  本模型里所有 ratio-1 index-K 层几何一致（32 head / 128 dim / topk 512 / cmp_ratio 1），因此是**有意共享且安全**。
  ⚠️ 残余风险：`group_id` 用 `id(buffer)`（Python 对象 id）做键；若某天 buffer 被重建而旧 task 仍在飞，等待会失配（现在不会发生）。

### Q6 `npu_quant_lightning_indexer_v2` 的 mode=1/2/3 语义与非确定要素

* mode 常量与语义（kernel 侧）：
  `quant_lightning_indexer_v2_common_arch22.h:21-23`
  ```cpp
  inline constexpr uint32_t CANDIDATE_MODE_SOURCE = 1;   // is_candidate_source: 输出候选块索引
  inline constexpr uint32_t CANDIDATE_MODE_CONSUMER = 2; // use_candidate: 输入候选块索引, 候选内选topk
  inline constexpr uint32_t CANDIDATE_MODE_OFF = 3;      // 关闭candidate功能(默认)
  ```
  模型侧选择：`models/deepseek_v41/indexer.py:226-238`
  ```python
        mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
        selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            quantized_query, key, weights, query_scale, key_scale, topk, 2,
            block_table=source_metadata.block_table, metadata=op_metadata,
            candidate_topk_index=candidates if uses_candidate_filter else None,
            candidate_mode=mode, candidate_topk_blocks=..., candidate_block_size=..., **common)
  ```
* mode=1 做什么（`service_vector_arch22.h:524-691` `ProcessCandBlockTopk`）：
  块内 8:1 `BlockReduceMax`（`546` `brmRepeat = CeilDiv(blkLen, 64)`）、pad 块 -inf/-1（纯 int32 位型链 `585-620`）、
  pin 最新块 `+inf`（`623-660`）、块级 `SortAll` + `MergeSortVecCopy` 进 2048 宽块级累加器（`683-690`）。
* mode=2 做什么（`service_vector_arch22.h:449-522` `BuildCandidateMask` + `844-853`）：
  候选行排序 → `CountGE` 窗口 → 每 8 位置 `Brcb` 展开距离 → `isOut`；候选外分数 `-1e30` 降级（不是 `-inf`）。
* **是否存在非确定操作**：
  - **没有原子操作**（`rg 'Atomic'` 无命中）；没有 `SetAtomicAdd` 之类累加。
  - 跨核同步只有 AIC↔AIV 的 `CrossCoreSetFlag/WaitFlag<FIA_SYNC_MODE2>` 与 kernel 尾部 `SyncAll()`（`kernel_arch22.h:710-715, 771-815, 827`）。
  - LD 跨核归约（`ProcessLD`）在**本机不激活**（§3 E2）。
  - ⇒ mode=1/2 本身**没有设计上的非确定源**；风险在**共享 UB 临时区的残留/竞态**（H1）——kernel 自己的注释
    （`service_vector_arch22.h:585-620`、`683-690`；`MergeSortVecCopy` 的 P3 说明）证明这类缺陷在该算子里**真实存在过 3 次**，
    且**每次都只在 numBlocks > candidate_topk_blocks 时可见**（R10 注释原文："小 shape 总块数≤2048 时候选集合=全部块，stale 分数不改变集合故漏检"）。

---

## ③ 阈值归属：为什么恰好是 16385（严格推导）

模型 config（`models/DeepSeek-V4.1-Flash/inference/config.json`）：
```
compress_ratios      = [0,0, 2×18(layer2..19), 1×20(layer20..39), 0,0,0]
kv_source_layers     = [2, 8, 14, 20]
index_source_layers  = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer = 20          # ratio=1
candidate_topk_blocks  = 2048
candidate_block_size   = 8
index_topk             = 512
```
* `uses_candidate_filter = index_source > 20`（`model.py:349`）⇒ **mode=2 生效层 = 24/28/32/36**（及其非 index-source 邻居 25-27、29-31、33-35、37-39 复用其 topk）。
* layer 20 是 **ratio=1** 的 KV/index source ⇒ 其"压缩后长度" = 原始 token 长度。
* 候选块数 `numBlocks = ceil(actS2Size / block_size)`；预算 `candidate_topk_blocks = 2048`。
  ⇒ `numBlocks ≤ 2048 ⟺ actS2Size ≤ 16384`：**候选集合 = 全部可达块 ⇒ mode=2 与 mode=3 严格等价**（过滤器恒等），
  任何块级/候选 buffer 缺陷在此区间**不可见**。
  ⇒ `actS2Size = 16385` 起 `numBlocks = 2049 > 2048`，**开始真的丢块** ⇒ 缺陷可见。
* 与实测表（8192/16384 全同；16385 起非确定）**精确吻合**，且 16385 的 `2/3`（首次不同、后两次相同）与
  "状态在连续同 prompt 请求之间演化/收敛" 的历史依赖特征一致（128K 的 `4/4` 不同 = 不收敛）。

---

## ④ 已排除 / 需修正的假设（含反证）

| # | 假设 | 判定 | 证据 |
|---|---|---|---|
| E1 | "candidate_indices_buffer 在 chunked prefill 下某 chunk 行数 < 上一轮 ⇒ 消费端读到上一轮残留" | **实测形态下未发现缺口（但不能排除边缘形态）** | TND 下 padding 用 dummy request（`worker/model_runner_v1.py:968-975`），dummy 的 `seq_lens=0` ⇒ kernel 走 `actS2Size==0` ⇒ `CleanInvalidOutput` 对候选行写 -1（`service_vector_arch22.h:333-341`）。唯一确证未写路径是 `ProcessInvalid`（整批无任务，`kernel_arch22.h:670-674, 682-700`）。 |
| E2 | "LD（S2 跨核切分）路径不写候选 ⇒ 长上下文必错" | **本机不成立（潜在隐患）** | `supportFd_` 仅在 `ProcessSocVersion()==ASCEND950` 分支可能置 true：`..._metadata_aicpu.cpp:222-229`（只匹配字符串 `"Ascend950"`）、`328-336`；`supportFd_` 默认 false（`aicpu.h:285`），`fdToleranceRatio=5`（`aicpu.h:311`）只在该分支使用。A3 = 910_93 ⇒ 判定 ASCEND910B，`AssignByBlock` 不执行（`..._aicpu.cpp:772-806`）。⚠️ 若换 950 或分支变化，`service_vector_arch22.h:891` 的 `!info.isNeedLD` 会让候选行**永不写出**。 |
| E3 | "-1 被 kernel 当成越界块号读坏数据" | **排除** | `-1` 在 mode=2 里作为 fp32 值参与降序排序，永远排在窗口之外（`service_vector_arch22.h:466-513`）；mode=1 显式写 -1（`333-341`）。真正的风险是"整行 -1 ⇒ 全行降级 -1e30 ⇒ R11 leak 填充"（退化但不崩）。 |
| E4 | "metadata 用了上一 chunk 的" | **排除（正常路径）** | 每 forward 重建 + `copy_` 进持久 buffer（`dsa_v41.py:910-935`），事件栅栏 `_inputs_ready`/`_stage_ready`/`_buffer_reusable`（`device_metadata.py:95-133`），模型侧 `wait_for_device_metadata`（`indexer.py:225`、`dsa_v41.py:540`）。 |
| E5 | "prefix caching 造成跨请求复用" | **排除** | 启动参数 `--no-enable-prefix-caching`；builder 对非零 prefix 直接 `raise NotImplementedError("V4.1 prefix caching is not implemented")`（`dsa_v41.py:746-747`）。 |
| E6 | "多流 overlap 导致跨层竞态（候选 copy 与消费读不同流）" | **排除** | `dsa_v41.py:315-385`：aux 流只用于 Q/KV 投影，层末 `main_stream.wait_stream(aux_stream)`；QLI、候选 `copy_`、attention 全在主流。 |
| E7 | "chunk 边界（16384 vs 16385）本身改变语义（如 mask_mode 分支）" | **排除为主因** | QLI metadata 恒用 `mask_mode=3`（`dsa_v41.py:925`），行级因果长度 `cuRealAcSeq=(actS2SizeOrig-actS1Size+行内偏移+1)/cmpRatio` 对首块/中间块都成立（`service_vector_arch22.h:805-817`），16384→16385 不改变任何分支；变化的只有 numBlocks 越界丢块。 |

---

## ⑤ 最小验证实验设计（不改模型语义 / 可在 A3-node2 chips 8-15 跑）

> 所有方案都给：改哪一行 + env 门控名 + 预期结果与判别逻辑 + 代价。**建议按 E1 → E2 → E4 → E3 的顺序做**（E1/E2 是判别力最强的两步）。

### E1（最强判别：把 mode 逐段降级；不需重编译，需重启服务）
**改点**：`models/deepseek_v41/indexer.py:226` 之前插入 env 门控（该文件不在挂载列表里，需要绑定挂载一份副本）：
```python
_forced = int(os.environ.get("V41_FORCE_CAND_MODE", "0"))   # 0=off, 3=全部关闭, 4=仅消费端关闭
if _forced == 3:
    mode = 3
elif _forced == 4 and mode == 2:
    mode = 3
```
**三个变体与判别逻辑**：
| 变体 | 语义 | 预期 | 若结果的判别 |
|---|---|---|---|
| `V41_FORCE_CAND_MODE=3` | 源层也不产候选、消费层不过滤（全 mode=3） | 若 131072 全部 3/3 相同 ⇒ 非确定性**确定在候选路径内**（正控）；若仍非确定 ⇒ 候选路径只是"放大器"，要查上游（转 E5） | 最强正控 |
| `V41_FORCE_CAND_MODE=4` | 源层仍 mode=1（照写候选），消费层降到 mode=3（不读候选） | 若**变确定** ⇒ 问题在"读到的候选内容"（H2/写覆盖）；若**仍非确定** ⇒ 问题在 mode=1 的写路径/块级 top-k 自身（H1） | 直接区分 H1/H2 |
| `V41_FORCE_CAND_MODE=5`（附加） | 源层强制 mode=3（不写候选），消费层保持 mode=2（读旧候选） | 若"更乱/更不稳定" ⇒ 反证"候选行确被读取且能读到非本 forward 内容"（H2 正控） | H2 反证 |
**代价**：需重启服务（配置/挂载变化），无重编译；`mode=3` 下**不重捕获**（mode 是 Python 侧属性，但会改变算子属性 ⇒ 首次运行触发一次编译/图捕获，之后稳定）。
**注意**：mode=3 会改变语义（质量下降），只用于定位，不可作为交付态。

### E2（零语义改动：哨兵检测"未覆盖行"；只需一次重启 + 短扫描）
**改点**：`dsa_v41.py:479-480` 前后加 env 门控（仅 eager prefill 生效，避免污染图）：
```python
if os.environ.get("V41_CAND_SENTINEL") == "1" and not torch.npu.is_current_stream_capturing():
    sentinel = -7
    shared.candidates[:n].fill_(sentinel)        # 写前哨兵（在 source 写之前）
    # 写后再统计仍有哨兵的行
```
消费端（`dsa_v41.py:476` 之前）统计 `(shared.candidates[:n] == -7).any(dim=-1)` 的行数并与 `n` 比较。
**预期/判别**：出现 `>0` 行 ⇒ **直接证实"消费端读到本 forward 未写的行"（H2 成立）**；恒为 0 ⇒ H2 的"行未覆盖"分支被排除，转向 H1。
**代价**：仅 eager 生效；多一次 16 MiB fill（~0.1 ms 量级），**不要在图内打开**（会改变图并被捕获，必须重捕获）。

### E3（阈值身份正控：把候选预算改小；需重启，不需重编译）
**改点**：模型 config（或 `build_layer_plan` 读取处）把 `candidate_topk_blocks: 2048 → 64`。
host 校验允许 `(0,2048]` 内 64 的倍数（设计文档 §1.3 / §7 R-O2 更新），kernel 侧 `BASE_TOPK=2048` 结构不变、
`blockNumPad` 走 `blockNum<64 ⇒ 64` 分支（`service_vector_arch22.h:535`）。
**预期/判别**：非确定阈值应随预算移动 ⇒ 从 16385 降到 `64×8+1 = 513`（即 1024/2048 token 的 prompt 就开始非确定）。
* 阈值移动到 513 ⇒ **强证实"候选预算越界"就是阈值来源**（H1/H2 二选一由 E1 定）。
* 阈值仍停在 16385 ⇒ 候选预算不是唯一门限，需要重查（例如 S2 tile/其它门限）。
**代价**：重启服务；因 config 变化会重新实例化模型 + 重捕获（一次性，几秒）；质量下降（仅定位用）。

### E4（单算子双跑，单卡可跑，不需服务）：判"内核自身是否非确定"
**做法**：用 `tests/e2e/nightly/single_node/ops/singlecard_ops/test_deepseek_v41_qli.py` 作模板构造**真实几何**：
`g=32, index_topk=512, mask_mode=3, cmp_ratio=1, layout_q=TND, layout_k=PA_BBND, S1=2048, S2=131072, block_size=128, candidate_topk_blocks=2048, block_size=8`，
`candidate_mode=1` 调 **100 次**，逐位比较 `candidate_topk_index_out`；再用 mode=2 比较 `sparse_indices`。
**关键要求（决定这个实验有没有判别力）**：
1. **不要用均匀随机 int8** 造 Q/K——随机数据下块 amax 分散、无 tie，历史上正是这类 harness 漏掉了 R10。
   建议用 *量化后* 的真实/类真实分布（例如把线上 index-K cache dump 下来，或构造大量"整块同分/零分"的块，使 2048 名边界出现并列）。
2. 每次调用前**不清 workspace**（模拟线上连续 forward），并在同一进程内交替跑不同 S2 长度（模拟 chunk 交替）。
**判别**：出现任意一次候选输出不同 ⇒ **H1 成立（内核自身非确定）**，与模型侧无关；全程一致 ⇒ 内核在真实数据下确定，根因在模型侧数据流（H2/H3）。
**代价**：单卡、分钟级；无需服务/权重（随机或 dump 数据即可）。

### E5（排除挂载补丁与上游放大器）
**做法**：同样做 ctx 扫描，但分别跑：
1. `--additional-config '{"enable_engram":false}'`（或把 `probe_bne/engram_hbm.py`、`probe_hash/engram_hash_ab.py`、`probe_dsa/dsa_v1.py`、`probe_moe/ascend_forward_context.py` 换回 stock 文件挂载）；
2. `--no-enable-prefix-caching` 保持；`MULTISTREAM`/`MULTISTREAM_DSV4_DSA_OVERLAP=0`。
**判别**：任一配置下 131072 变确定 ⇒ 该组件是主因（H3）；都不变 ⇒ H3 排除，聚焦 H1/H2。
**代价**：每次需重启（~1-2 min），扫描一次 128K×3 约 70 s。

### E6（定位"大范围错误"发生在哪一环；仅在需要时做）
在**决定首 token 的最后一个 prefill chunk**打开 dump（`.cpu()` 同步，仅单请求、短输出）：
(a) layer20 的候选行（mode=1 输出）、(b) layer24 的候选输入、(c) layer24 的 topk 输出（统计 -1 个数）、
(d) `quantize_indexer_query` 的 `quantized/scale` 哈希。
**判别**：若 (a) 两次运行不同 ⇒ H1（源层块级 top-k 不稳定）；若 (a) 相同而 (b) 不同 ⇒ H2（buffer 生命周期）。
**代价**：需要在 NPU 上 dump 并同步，步频会掉；只在定位阶段用。

---

## ⑥ 修复方案草图（不实现，仅思路与风险）

| # | 方案 | 思路 | 风险/代价 |
|---|---|---|---|
| F1 | **写覆盖显式化（最保守，先做）** | 在 `dsa_v41.py:479-480` 写候选前 **显式 `shared.candidates[:n].fill_(-1)`**，再 `copy_` 有效行；把"未覆盖行"从"任意旧内存"变成"确定 -1" | 多一次 16 MiB fill（prefill 每 step 一次，~0.1 ms 量级）；图模式下 fill 是合法 op、地址不变，但会**增加图内的算子**（若 `static_kernel` 对算子数敏感需复测）；**不修语义**（-1 行本来就表示"无候选"），但不解决 H1（内核内部错误） |
| F2 | **恢复 reset 语义（可选替代 F1）** | 让 `DeepseekV41SharedAttentionState.reset()` 真正 `candidates.fill_(-1)`（仍是固定地址，兼容 ACL Graph） | 同 F1 成本；注意 `topk_indices` 不要一起清（多一次大 fill） |
| F3 | **kernel 侧补齐（根治候选输出覆盖）** | ① `ProcessInvalid`（`kernel_arch22.h:682-700`）增加候选输出填 -1；② `ProcessLD`/`isNeedLD` 分支补 `CopyOutCandTopkIndex`（否则 950 上必然错） | 需重编译自定义算子（910_93 需重装 `custom_transformer` 包）；改动在算子仓库，跨版本维护成本 |
| F4 | **消除块级路径对 UB 残留的依赖（H1 根治方向）** | 把 `ProcessCandBlockTopk` 用的 `tmp[6144,16384)` 区域改为**专用 UB buffer**（`pipe->InitBuffer` 独立分配）或在使用前显式初始化；把 `MergeSortVecCopy` 的 "必须 int32 域回拷" 契约写成单测（`service_vector_arch22.h:683-690`、P3） | 会改变 UB 预算（mode=1 已 176/192 KB，见 `service_vector_arch22.h:224-236` 注释），可能超限需重新规划；需重编译 |
| F5 | **消费端合法性校验（观测/防御）** | 在 `indexer.py` mode=2 前加 env 门控的 device 端校验：块号 ∈ [0,numBlocks) 或 -1；整行 -1 记一次计数/告警（甚至临时 fallback 到 mode=3） | 一次 device reduce + host 同步会破坏流水（只在 debug 打开）；**不要**把语义改成 `masked_fill(-inf)`——会与 R11（对齐参考实现的 leak 语义）冲突 |
| F6 | **测试矩阵补洞（防止再犯）** | 把 E4 的"真实分布 + 连续调用 + 交替长度"用例加入 op UT；把"numBlocks>topkBlocks 且边界大量并列"作为门禁用例；harness 增加 `force_rows` 风格的全量行检查（设计文档 §6.6 已有机制） | 仅测试成本 |

---

## ⑦ 环境事实与挂载覆盖（影响结论适用范围）

* 容器启动命令（`ps` 实测）关键项：`--max-num-batched-tokens 2048 --max-num-seqs 4 --block-size 128 --max-model-len 1048576`、
  `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,20,24,32]}`、
  `--speculative-config {"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}`、`--no-enable-prefix-caching`、
  `additional-config: {"enable_engram":true,"multistream_dsv4_dsa_overlap":true,"enable_static_kernel":true,"enable_npugraph_ex":true,"enable_fused_mc2":1}`
  ⇒ **prefill 走 eager，decode 走 FULL graph**；decode 批 = 6 token/请求。
* `docker inspect` 挂载（**非 stock 文件，重要**）：
  ```
  probe_dsa/dsa_v1.py            -> vllm_ascend/attention/dsa_v1.py                 (ro)
  probe_moe/ascend_forward_context.py -> vllm_ascend/ascend_forward_context.py      (ro)
  probe_bneck/engram_host_ws_opt.localowner_v2.py -> models/deepseek_v41/engram_hbm.py (rw)
  probe_hash/engram_hash_ab.py   -> models/deepseek_v41/engram_hash.py             (rw)
  probe_gate/engram_gate_ws_opt.stock.py -> models/deepseek_v41/engram_gate.py     (ro)
  probe_bneck/model.py.probe     -> models/deepseek_v41/model.py                   (ro)
  draft_hot_sp/sitecustomize.py  -> site-packages/sitecustomize.py                 (ro)
  ```
  ⇒ 候选相关逻辑（`model.py`/`indexer.py`/`dsa_v41.py` 的 candidate 部分）**与 stock 一致**（已 diff：
  `/home/user/projects/dsv41/f24_work/stock_model_v41.py` 与挂载版候选段落逐行相同），
  但 **Engram host 路径与 dsa_v1 是实验补丁**（H3 的对象）。
* kernel 源码来源（只读）：
  `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascend910_93/.../quant_lightning_indexer_v2/{,arch22/*.h}`
  与 `csrc/attention/quant_lightning_indexer_v2/op_host/...`（host tiling）、
  `csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/...`（AICPU 分核）、
  `csrc/attention/quant_lightning_indexer_v2/docs/qli_v2_two_level_topk_design.md`（设计/风险记录）。
  ⚠️ 注意设计文档 §9 P5：`build/binary/ascend910b/src/` 下的源码副本与编译产物可能不同步——本次只读源码，
  **不保证部署 .o 与 .h 完全一致**；若实验结果与源码推断不符，应优先核对 `build/binary/ascend910_93/` 下的源码副本与 `.o` mtime。

---

## ⑧ 查不到 / 不能确定（明确列出，不脑补）

1. **未能在离线条件下证明"哪一次调用出现未覆盖的候选行"**。我逐条核了 kernel 的 4 条写路径，在"单请求 + 无 padding + 非 LD"的实测形态下**全部覆盖**；H2 的"行未覆盖"分支需要实验（E2）判定。
2. **`ProcessInvalid` 的填充范围存在越界嫌疑，未能定性**：`kernel_arch22.h:684-686` 用
   `totalOutputSize = batchSize * qSeqSize * kHeadNum * sparseCount`。TND 布局下 `qSeqSize` 的具体取值（最大 S1 还是 token 总数）
   我没能在离线条件下确认；若它等于最大 S1，则该 fill 的 `[Σ 行 … batchSize×qSeqSize)` 会**写出输出张量范围之外**，污染相邻显存。
   这是**独立于候选路径**的潜在污染点，值得单独验证（可作为 E2 的附加 dump：看 `sparse_indices` 张量之后的 16 MB 内存是否被改）。
3. **`supportFd_` 的 SoC 判定是推断**：`ProcessSocVersion()` 只匹配字符串 `"Ascend950"`（`..._aicpu.cpp:222-229`），
   我据此推断 A3(910_93) → ASCEND910B；但**没能在容器内直接打印 `socVersion_`**（AICPU 内部状态）。
   建议用 `npu-smi info` / `ASCEND_SOC_VERSION` 佐证后，再把 E2 排除项确定为结论。
4. **没跑任何 NPU 实验**（遵守"不占卡/不起停容器"），
   因此所有"内核自身非确定"的判断都来自：(a) kernel 自身注释记录的 3 次同类实测缺陷、
   (b) `at::empty`+no-op reset 的生命周期事实、(c) 阈值与候选预算的精确吻合。**H1 与 H2 的最终裁决需要 E1/E2/E4。**
5. **上游 `ops-transformer` 的 950 分支 / arch35 代码未审**（容器内只有 arch22 与 arch35 源码，后者与本机无关）；
   如果同一模型将来跑 950，候选路径在 LD 下**必然**缺写（§Q1 表），这是一个独立的、尚未修复的正确性缺陷。
6. **`index_topk=512` 与候选泄漏（R11）的交互未量化**：当候选集有效位置少于 512 时，R11 语义会把候选外可达位置填进 topk。
   在 128K 场景（候选位置 ≈ 16382 ≫ 512）不会触发，但在**短上下文 + 消费层**（例如 4K 上下文、候选块预算被 pin/尾块吃掉）可能触发，
   本次未展开验证。

---

## ⑨ 一页速查（给主 Agent 的最小行动清单）

1. **先做 E1（mode 逐段降级）**：`V41_FORCE_CAND_MODE=3` 与 `=4` 两次扫描即可把 H1/H2/上游 三选一。
2. **同时准备 E4（单算子双跑）**：这是唯一能在不重启的情况下判定"内核自身非确定"的实验，且用真实/类真实量化数据（避免随机数据漏检，历史教训）。
3. **E2（哨兵）** 是 H2 的直接证据，代价最低但要注意只在 eager 打开。
4. **E3（预算 64）** 作为阈值身份的正控；预期阈值移动到 513。
5. 若 E1 指向 H2：优先 F1/F2（写覆盖显式化）+ F5（消费端校验）；若指向 H1：优先 F4（UB 区隔离）+ F3（补 LD/Invalid 路径），并重编译算子。
