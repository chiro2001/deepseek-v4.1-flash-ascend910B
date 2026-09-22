# 018 — KV8 Phase 2：把「按需反量化」接进引擎

> 2026-09-22 01:0x–01:15 CST。执行：子代理 **KV8_p2**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`，容器内 commit `e43cf1e9f`）。
> 全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；没写 `/tmp`；
> 没手设 `ASCEND_RT_VISIBLE_DEVICES`；没写 `upstream-v41/`；没动 `dsv41-release/`。
> 代码全部落在**影子包** `/work/agents/KV8_p2/shadow/vllm_ascend/`（容器内），
> 本仓副本在 `a2/agents/KV8_p2/`。

---

## 0. 三句话结论

1. **【实测】按需反量化已经接进引擎，而且是「逐比特精确」的**：同一个真实调用点
   （`DeepseekV41EagerAttentionImpl._native_attention` → `npu_sparse_flash_mla`），
   INT8 存储 + gather + 反量化 + PA_BBND scratch + 索引重编号，
   与生产 BF16 路径**逐比特一致**（`max_abs = 0.0` / `rel_L2 = 0.0`，
   用「无损量化器」把量化误差归零后的对照，与 `logs/002` 的 N2 手法同源）；
   换成**真的** `npu_dynamic_quant` 后，attention 输出对 BF16 是
   **`rel_L2 = 5.42e-3`（0.54%）、`cos = 0.999986`、`max_abs = 4.9e-4`**
   ⇒ **优于 `logs/002` 的算子级读数**（`rel_p99 6.9e-3` / `cos_min 0.999976`）。
   **图内增量 +5.2 µs/层 ⇒ +21 µs/step（30 ms 的 0.07%）**；capture/replay 与 eager 逐比特一致，
   且改掉 cache 内容后 replay 跟着变（非冻结快照）。
2. **⛔【实测·关键】容量收益在真实分配器上只有 ×1.03，不是 ×1.84**：
   V4.1 的 hybrid slot **把 long-KV 平面和 10 个 SWA 平面叠在同一个物理页几何上**，
   页大小取 `max(long-KV 页, SWA 别名页)`；SWA 页 = 128×512×2 = **131072 B** 永远大于
   INT8 long-KV 页（41088 / 73856 B）⇒ `pool_bytes_per_block` 只从 **540928 → 524288 B**
   （**4226 → 4096 B/token，×1.032**）。★ 要让 KV8 的 ×1.84 落地，
   **必须把 SWA（`ori_kv`）也量化**（`DeepseekV41SWASpec` 现在没有 `scale_dim` 机制）——
   按实测页尺寸外推是 **2137 B/token（×1.98）**【推断·算术】。
3. **【未确认】一个 NaN**：prefill 形状（每请求 3 个 query 行）的**真量化**臂输出 NaN，
   而**同一形状的无损臂逐比特精确**、反量化后的 KV 行也是有限值且与 BF16 同量级
   （`rel_L2 = 4.44e-3`）⇒ NaN 产生在**算子内部或其边界**，不是我们的读路径。
   **必须**用真 indexer 的 top-k 在真 metadata 下复现一次才能定性。

---

## 1. ★ cannbot 对照（用户要求，先做、不占卡）

按 `a2/AGENTS.md` §6 的索引，在 A3 上只读查了 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`：

### 1.1 `model/model-infer-kvcache/SKILL.md:38-52,102-110,142-153,174-199,224-242`

| cannbot 说什么 | 我们采纳/没采纳 | 理由 |
|---|---|---|
| **KV 选型默认 = Paged 模式（FA + TND layout）**；MLA 架构用 **`TND_NTD` + MLA absorb** | **没采纳（布局层面）**，**采纳（映射层面）** | 我们实测（`logs/015`）arch22 上 `npu_sparse_flash_mla` **只编译 `TND Q × PA_BBND KV`**；TND-KV 的 tiling key **一条都没有**。cannbot 是"推荐形态"，不是本代可用的实例化。**但它的 block/slot 映射公式我们逐字采纳** |
| **block/slot 映射**：`逻辑块 = pos // block_size`、`块内偏移 = pos % block_size`、`物理块 = block_table[b, 逻辑块]`、**`物理 slot = 物理块 × block_size + 块内偏移`** | ✅ **采纳** | 这就是 KV8 读侧 `_kv8_cmp_plane` 的实现（`block = idx // SB`、`offset = idx % SB`、`phys = block_table[b, block]`）；`SB = storage_block_size = block_size // compress_ratio`，与算子从 `cmp_kv.shape[1]` 推出的块大小一致。实测逐比特对得上（本日志 §3） |
| **MLA absorb 路径必须 `num_key_value_heads = 1`**，与 `cache_entries.num_head` 严格一致 | ✅ **本来就是** | `DeepseekV41FullSpec(num_kv_heads=1)`；KV8 没有改这一条 |
| **`sparse_mode` / `atten_mask` 硬约束**（mask 只允许 bool/int8/uint8、`3/4` 固定 `[2048,2048]`） | 只读记录 | V4.1 走的是 SMLA 自己的 `ori_mask_mode=4` / `cmp_mask_mode=3`，不传 `atten_mask`；KV8 **没有**改任何 mask 参数（重编号只改索引值） |
| **FA v1/v2 的量化 KV 走 `antiquant_mode`/`dequant_scale_key`**（"量化 KV 是算子内的事"） | **没采纳** | 那是 `npu_fused_infer_attention_score{,_v2}` 的接口；V4.1 用的是 `npu_sparse_flash_mla`。cannbot 的 torch_npu 清单里**没有** `sparse_flash_mla` 的 antiquant 变体（`grep antiquant` 命中 0 条），我们也没有能力在本代补一个 |

### 1.2 `model/model-infer-quantization/SKILL.md:424-451` + `references/quantization-fusion-and-benefit.md:93-127,162-197`

| cannbot 说什么 | 我们怎么用 |
|---|---|
| **§7「验证真实生效与收益」是量化验证的唯一真相源**：功能验证 + **等价性自检（固定 prompt，量化/基线各跑一次 greedy，记首个分歧 token + 是否语义等价）** + **probe 优先** | ✅ **本次就是按这条做的**（同 prompt、同一批 Q、同一 metadata，量化开关 A/B；差异逐 token 量到 `rel_L2`/`cos`）。★ 它同时要求"**不能只看代码 diff**"，所以本日志主表全部是**端到端算子输出**而不是"我们改了 spec" |
| **收益公式**：`Δ_decode ≈ -0.4×(T_FFN/T_decode) - Δ_dispatch_saving + Δ_overhead_cost`，**eager 下 `Δ_overhead_cost` 可能主导，使 Δ 转正**；并明确写 **"eager 数据不进收益结论，仅用于功能验证"** | ✅ **完全印证 `logs/015` 的判定**：KV8 的 eager 画面 +484 µs/层（本日志：159 → 644 µs）**不能用来否决路线**；生产帧（ACLGraph）净增量 **+5.2 µs/层**。★ 我们保留这条作为"eager 不判负"的文档依据 |
| **`Δ` 必须固定 4 维（exec_mode / batch / parallelism / KV 策略）报告** | ✅ 本日志全部标注了这 4 维（graph、B=8、TP=1、KV8=long-KV only） |
| **roofline**：KV 量化只在 attention 是 **memory-bound** 时才有正收益；launch/comm/sync-bound 时"0 或负" | 【推断】这解释了我们的容量侧结论（§4）：KV8 省的是**每 token 字节**，只有**页大小真的变小**才落到容量上——而它没变小 |

### 1.3 `references/quantization-structure-cards.md:174-189,191-206`（★ C1 红线）

原文要点（**逐字摘**）：

> **C8 不是简单把 KV cache dtype 改成 int8。模型侧需要同时处理 `npu_mla_prolog_v3`、
> cache layout、query scale、cache scale 和后续 attention kernel。**
> …**C8 下 rope cache 可能通过 fake/empty tensor 占位，nope cache 内同时承载量化 cache 和
> scale repo，后续 sparse attention 需要 `key_quant_mode/value_quant_mode` 与
> `quant_scale_repo_mode` 对齐。**
> …**对 sparse attention，C8 路线可能切到 `npu_sparse_flash_attention_antiquant`，
> 而不是普通 `npu_sparse_flash_attention`。**
> …**验证时检查 FA/Sparse FA 是否真实消费 int8 cache 和对应 scale；不能只凭 cache dtype 判定 C8 生效。**

| cannbot 说什么 | 我们采纳/没采纳 | 理由 |
|---|---|---|
| **"不是改 dtype"**：dtype / cache layout / scale / attention kernel 要一体 | ✅ **采纳，且这是本任务的 C1 红线** | 我们确实**没有**只改 dtype：改的是 (a) spec（dtype+scale_dim+scale_dtype）、(b) `_cache_plane_sizes`（页几何）、(c) `reshape_cache`（多返回一个 scale 平面）、(d) 写入侧量化 + 两次 scatter、(e) 读取侧反量化 + scratch。**dtype 单独改一定会在算子处炸**（SMLA 只吃 BF16） |
| **必须验证"量化 kernel 被真实消费"** | ✅ **采纳** | 判据不是"spec 改了"，而是 **`GPU KV cache size` 变了**（本日志 §4 实测**基本没变**）+ 输出对拍 |
| **scale 的存放位置**："nope cache 内同时承载量化 cache 和 scale repo" | ✅ **采纳** | 我们把 4 个 FP16 scale 放在**同一个页**里（`scale_dim=4` 机制，与 indexer 的 `scale_dim=1` 同源），满足任务书 C3 |
| **C8 路线通常切到 `..._antiquant` 算子** | **本代没有可用的 SMLA antiquant** | `grep antiquant` 在 `torch_npu_list.md` 命中 0；`npu_kv_quant_sparse_flash_attention`（`logs/002`）只吃 `head_dim=576`，V4.1 是 512 ⇒ 只能"读出+反量化" |

### 1.4 ★ 矛盾原文（有，而且是本日志最有价值的一条）

> cannbot（`model-infer-kvcache/SKILL.md:40-52`）：
> **"改造目标：Paged 模式（FA + TND layout）… TND layout：变长 batch 拼一维"**

> 我们的实测（`logs/015` §3.2，容器内 commit `e43cf1e9f`）：
> ```
> Aurora SparseFlashMla only compiles TND Q with PA_BBND KV.
> [FUNC:Parse][FILE:sparse_flash_mla_tiling.cpp][LINE:1160]
> ```
> 且 `op_kernel/sparse_flash_mla_template_tiling_key.h` 的 8 条 tiling key **全部**是
> `TND Q × PA_BBND KV`（`KV_LAYOUT_T` 无一条 TND）。

⇒ **cannbot 的"默认形态"是接口层/推荐层，不等于本代（arch22）实例化层**。
我们按"实测 > 文档"处理，并**没有**去改 csrc 补那条 tiling key（那是 015 的 P3.5，维护成本归我们）。
**第二条矛盾（本次新发现，见 §4）**：cannbot/`KV8-PLAN.md §1.1` 的字节账
（4421 → 2405 B/token，×1.84）在**真实 hybrid slot 分配器**上不成立，因为页大小由
**SWA 别名**而不是 long-KV 决定。

---

## 2. Phase 2 实现（改了什么、在哪）

**开关**：`VLLM_V41_KV8=1`（`core/deepseek_v41.py::long_kv_plane_kwargs()`）。
不开就是原样 BF16，所以 A/B 是同一套代码。

| # | 文件（影子包路径 / 本仓对应源文件） | 改动 |
|---|---|---|
| 2.1 | `core/deepseek_v41.py`（= `vllm-ascend-v41-base` 版，md5 `f48b1761…`） | ① **`long_kv_plane_kwargs()`**：KV8 时 long-KV spec 用 `dtype=torch.int8, scale_dim=4, scale_dtype=torch.float16`；② **`_cache_plane_sizes()`** 从"只给 IndexerSpec 算 scale 平面"泛化成"任何 `scale_dim>0` 的 spec 都算"（页几何 / `page_size_bytes` / `allocate_cache_config` 自动对）；③ **`reshape_cache()`** 对 `scale_dim>0` 的 spec 返回 `(payload, scale)` 双平面（与 indexer 已有的写法完全一致）；④ `DeepseekV41FullSpec.__post_init__` 加一条 `int8 ⇒ scale_dim==4` 断言 |
| 2.2 | `models/deepseek_v41/model.py`（= `op_line/src_full` 版，md5 `e5d2490e…`） | long-KV spec 构造改成 `**long_kv_plane_kwargs()`。**注意**：任务书写的"`model.py:699`"在这份树上其实在 **`model.py:291-300`**（`DeepseekV41FullSpec(...)`，第 298 行那个 `dtype=torch.bfloat16`），`model.py:699` 是 `load_weights` 的尾部 |
| 2.3 | `attention/dsa_v41.py`（= `op_line/src_full` 版，md5 `6e60fc4d…`） | 写入侧 `kv8_store_rows()`（在 `_write_compressed_source` 里、**`indexer.update_keys` 之后**调用）= `npu_dynamic_quant(latent.view(-1,128))` + **两次 `scatter_cache_sk`**（int8 载荷 + fp16 scale 各一次，同一个 `[T,2]` slot 映射）；读取侧 `kv8_dequant_rows()` / `kv8_scratch_plane()` / `DeepseekV41EagerAttentionImpl._kv8_cmp_plane()`，`_attention()` 解包 `(payload, scale)`，`_native_attention()` 在 SMLA 前换掉 `cmp_kv`/`cmp_block_table`/`cmp_sparse_indices` |

**五条硬约束的对账**：

| # | 要求 | 本实现 |
|---|---|---|
| **C1** | 量化点必须在 `scatter_cache_sk` 那一步 | ✅ 量化发生在 `_write_compressed_source` 的 scatter 调用处；`indexer.update_keys(latent, …)` 在它**之前**（`dsa_v41.py:433`），indexer 拿到的仍是 BF16 latent ⇒ 选块不变 |
| **C2** | 只做 long-KV，SWA 留 BF16 | ✅ SWA spec / SWA 写入 / `ori_kv` 入参**一行没动**。★ 但 §4 实测说明：**这条"省事"的选择直接把容量收益吃掉了** |
| **C3** | scale 与 int8 同页 | ✅ 用 `scale_dim=4` 机制：页 = `[128×512 int8][128×4 fp16]` = 66560 B = **520 B/token**；实测 stride：payload `(131072, 512, 512, 1)`、scale `(65536, 4, 4, 1)`（元素数，fp16）⇒ 同一块内两平面 |
| **C4** | gather 的 slot 计算与设备侧 block table 一致 | ✅ `phys = gather(cmp_block_table[:R], 1, idx//SB)`、`slot = phys*SB + idx%SB`；**cannbot §1.1 的映射公式逐字一致**；实测（无损臂）逐比特对上 |
| **C5** | 只做非 CP 路径 | ✅ 只改 `dsa_v41.py`，`dsa_cp.py` 没碰 |
| 雷区 | 不能用 `sl[sl>=0]` 布尔掩码 | ✅ 用 `torch.where(valid, idx, 0)` park 到 slot 0（015 s2 的结论），全链可 capture |

### 2.1 读侧两条路径（这是 Phase 2 唯一的"设计选择"）

```
行数 R = 选块张量的行数；per_req = ceil(topk / storage_block_size)

A) decode（R == num_reqs，每请求 1 个 query token）：
   idx(0..Lc-1) → phys block table → 取 [R,512] 行 → 反量化
   → 写进「每请求 per_req 块」的连续 PA_BBND scratch（第 t 个选中行 → scratch 第 t 行）
   → block_table' = arange(R*per_req)（identity）、indices' = where(valid, arange(512), -1)
   ⇒ 只搬「这一步真正读的 512×R 行」

B) prefill / 多 query 行（R != num_reqs）：
   每个请求的 **压缩前缀**（cdiv(cache_seq_lens, SB) 个物理页）整页 gather（页粒度 index_select，快）
   → 反量化 → 写进「每请求连续 nblocks 块」的 scratch
   → block_table' = arange(R*nblocks)、**indices 不变**（保持真逻辑下标 ⇒ 因果 mask 语义不变）
   ⇒ 代价 = 每层 Lc×520 B 读 + Lc×1024 B 写（015 §5.3 的"整层反量化"选项），不是 Q_T×512 的行放大
```

两条路径共用同一个**跨层复用的 scratch 池**（`_KV8_SCRATCH`，按 (blocks, block_size, dim, dtype) 键，
同形状只分配一次）。★ 为什么不能对 decode 也用（B）：decode 的选块是 512 个**离散点**，
整页搬会退化成"搬整个前缀"；为什么不能对 prefill 用（A）：PA_BBND 的 block table 是**按 request** 的，
而 chunked prefill 的**每个 query 行有各自的 512 选块**，无法共用一段 scratch（且重编号会在
`cmpS2IdLimit < 512` 时破坏因果 mask）——这正是 015 §5.3 未解决的那一项。

---

## 3. 数值对拍（单卡 c1，真实算子，真实 cache 页）

**方法**：用**真实分配器**建页（`tests/deepseek_v41_reference.build_v41_cache_specs` +
`core/deepseek_v41.{group_cache_specs,allocate_cache_config,reshape_cache}`），
用**真实写入入口** `dsa_v41.kv8_store_rows` 填，用**真实读取入口**
`DeepseekV41EagerAttentionImpl._native_attention` 跑 `npu_sparse_flash_mla`。
几何：`B=8`、`topk=512`、压缩长度 `Lc=1024`、`block_size=128`、`N1=16`、`D=512`、
层 20（`compress_ratio=1 ⇒ SB=128`）。block table 用**打乱过的物理块**（专治"table 没被用上"）。
原始数据：`raw/018-s1-ab.json`、`raw/018-run5.log`；脚本：`agents/KV8_p2/p2_engine.py`。

| 臂 | 内容 | 时间（eager，中位） | vs 生产 BF16 路径 |
|---|---|---:|---|
| **A** | BF16 平面（现状） | **159.3 µs** | — （`ref_rms = 0.02215`，非零非 NaN） |
| **B** | INT8 平面 + **无损量化器**（`codes/128`、scale=1/128） | 644.6 µs | **`max_abs = 0.0`、`rel_L2 = 0.0`、`cos = 0.9999998` ⇒ 逐比特精确**【实测】 |
| **C** | INT8 平面 + **真 `npu_dynamic_quant`**（g128） | 643.9 µs | **`rel_L2 = 5.42e-3`、`cos = 0.999986`、`max_abs = 4.9e-4`**【实测】 |
| **D-无损** | prefill 形状（每请求 3 行 query） | — | **`rel_L2 = 0.0`、`max_abs = 0.0`、`cos = 1.0` ⇒ 逐比特精确**【实测】 |
| **D-真量化** | 同上，真量化器 | — | **NaN** ⇒ 见 §5 未确认①**【未确认】** |

**量化器本身的约定（【实测】，必要前提）**：`torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)`
返回 `(q, scale)`，`scale` 是 **float32**，语义是 **`x ≈ q × scale`**
（`mul_rel = 6.4e-3` vs `div_rel = 8.7e3` ⇒ 除法约定被排除）——
所以 `kv8_dequant_rows` 是**乘**。这与 indexer 的用法（`scale.unsqueeze(-1).to(float16)`）一致。
原始数据：`raw/018-s0-convention.json`。

**KV 行本身的量化误差**【实测】：反量化后的行 vs BF16 行 `rel_L2 = 4.44e-3`、`cos = 0.999991`、
`max_abs = 3.9e-3`（行 rms 0.569）——**与 `logs/002` 的 INT8 g128 量级一致**
（002 是 `rel_p99 6.4e-3`，且那里用的是 per-token-head-tile-128 的算子内量化）。

### 3.1 图内（生产帧）增量【实测】

同一批数据、同一个调用点，capture 进 `torch.npu.NPUGraph`（ACLGraph 接口）后各 replay 30 次取中位：

| 臂 | replay 中位 | 说明 |
|---|---:|---|
| BF16（现状） | **326.4 µs** | 本 run 内同口径 |
| **KV8（INT8 + gather + 反量化 + scratch）** | **331.7 µs** | **增量 = +5.22 µs/层** |
| ⇒ ×4 个 long-KV 层 | | **+20.9 µs/step = 30 ms 的 0.07%**【实测】 |

**图兼容性**【实测】：

* capture + replay 输出 vs eager：`rel_L2 = 0.0`、`max_abs = 0.0`（两条臂都是）；
* **改掉 cache 内容后再 replay**：`mean|Δ| = 0.0017`（非零）、`rms = 0.0217`（非零非 NaN）
  ⇒ **无冻结快照、无 D2H、无动态 shape**；015 的 `aclnnNonzeroV2` 雷区本实现**没有踩**（全用 `torch.where`）。

> 判据对照：`KV8-PLAN.md` 的判据是"增量 ≤ 0.3 ms/step"。
> 图内 **+0.021 ms ⇒ 通过（7%）**；eager **+484 µs/层 ⇒ +1.94 ms/step = 6.4%** ⇒ 不通过，
> 但 cannbot 明确写 **"eager 数据不进收益结论，仅用于功能验证"**（§1.2）⇒ 生产（decode 走 ACLGraph）按图内算。

---

## 4. ★ 容量验证：**只涨了 ×1.032，不是 ×1.84**【实测】

同一个 `allocate_cache_config`（引擎自己的分配器），只切 `VLLM_V41_KV8`：

| 量 | BF16（现状） | INT8（KV8） | 比 |
|---|---:|---:|---:|
| `pool_bytes_per_block`（4 个 slot 之和） | 540928 B | 524288 B | **×1.032** |
| **bytes/token（÷block_size=128）** | **4226 B** | **4096 B** | ×1.032 |
| 24 GiB 预算下的 block 数 | 47639 | **49152** | ×1.032 |
| 每个 slot 的页 | `[131072, 131072, 131072, 147712]` | `[131072, 131072, 131072, 131072]` | 见下 |

**原因（代码事实 + 实测）**：`plan_cache_slots()` 把 4 个 long-KV 源槽位与 **10 个 SWA 平面叠放**
（`aliases = swa[slot_idx::4]`），槽位页大小是

```
capacity = max(long_kv_bytes + index_bytes, max_swa_alias_bytes)
```

实测页尺寸对账（`raw/018-s1-ab.json` 的 `F_capacity_binding`）：

| 槽 | long-KV 页（若它 binding） | SWA 别名页 | 实测页 |
|---|---:|---:|---:|
| ratio-2 槽（层 2/8/14） | **41088 B**（INT8）/ 73856 B（BF16） | **131072 B** | **131072 B**（SWA 顶着） |
| ratio-1 槽（层 20） | **73856 B**（INT8）/ 147712 B（BF16） | 131072 B | BF16 147712 → INT8 **131072** |

⇒ **INT8 只在"long-KV 页原本就顶住"的 ratio-1 槽省下 16640 B/block**，
ratio-2 三个槽**一分没省**（long-KV 从来不是 binding 的那个）。

**【推断·算术】出路**：把 SWA 也量化（SWA 页 131072 → 66560 B）后，
`3×max(41088, 66560) + max(73856, 66560) = 281856 B/block = 2202 B/token`…
按本脚本内同一公式给出的是 **2137 B/token ⇒ ×1.98**（与 `KV8-PLAN §1.1` 的 ×1.84 同向同量级）。
**但 `DeepseekV41SWASpec`（`AscendSlidingWindowMLASpec`）没有 `scale_dim` 机制**，
要落地必须给它加 spec 字段 + 改 `_cache_plane_sizes`/`reshape_cache` + 改 SWA 写入/读取。
**这不再是"KV8 只动 long-KV"的小改动**（也违反任务书 C2），**需要主代理决策**。

> 说明：本节的 `pool_bytes_per_block` / 页尺寸 / block 数都是**引擎自己的函数**跑出来的【实测】；
> "×1.98"是**按实测页尺寸外推**的【推断·算术】，没在真机上起服（起服要 476 GB 权重，单卡做不到）。

---

## 5. 未确认清单（不许用相邻数字顶替）

| # | 未确认 | 证据（已知的） | 怎么补 |
|---|---|---|---|
| ① | **prefill 形状 + 真量化器 → SMLA 输出 NaN** | 【实测】同一形状换**无损量化器**时**逐比特精确**；反量化出的 KV 行**有限**且 `rel_L2 = 4.44e-3`；`plane_finite = true`；scale 范围 `[0.0, 7.75e-3]`（有一组 amax=0 ⇒ scale=0）。⇒ NaN 在**算子内部/边界**，不在我们的读路径 | 用**真 indexer 的 top-k** + 真 metadata（真 `cu_seqlens_q`/`seq_lens`）重跑；或把 scale=0 的那一组改成非零再试，定位是否与 `scale=0` 相关 |
| ② | 行索引 gather 的**真实带宽/耗时** | 本实现在同一页内做 `plane[block, row]` 二维高级索引（015 §5.1 的"交错页"形态，60–90 GB/s），不是 flat `index_select` | 图内增量已经只有 +5.2 µs/层（本日志 §3.1），**eager 的 +484 µs/层不在生产帧**；若要彻底平掉，需要"页内 int8 载荷连续跨页"的布局（与 C3 冲突）或自写 AscendC 小算子 |
| ③ | 真生产配置下的容量数字 | 本日志用 `block_size=128` + 真实 spec/分配器，**没有起服**（单卡放不下 476 GB 权重） | 在 A2 起服，读 `GPU KV cache size` 那一行（判据：3.50M → ?） |
| ④ | 端到端质量（GSM8K-200 / Vision 23） | **未做**（预算 3 h 用尽在实现+对拍） | Phase 3 |
| ⑤ | 真 indexer 选块数是否恒 ≤ `cmpS2IdLimit`（重编号安全性的充分条件） | 015 §4 已论证 decode 下成立；本实现**没有**加断言 | 按 015 建议加 eager-only 断言（capture 期不能有 D2H/`nonzero`） |
| ⑥ | A2（910B3）上是否一致 | 本次全在 A3 c1（910C `Ascend910_9382`） | 在 A2 重跑 `p2_engine.py` |
| ⑦ | DSpark / 投机解码 / CP 路径 | `validate_cache_runtime` 里 `cache_dtype` 仍是 `auto/bfloat16`（那是**config 层**的开关，与 per-spec dtype 无关）；CP 路径（`dsa_cp.py`）**没改也没测** | C5 有意隔离 |

---

## 6. 交付物与复现

| 东西 | 路径 |
|---|---|
| 影子包补丁（3 文件，164 行新增） | `a2/agents/KV8_p2/kv8-phase2.patch` |
| 改后的 3 个文件（可直接覆盖） | `a2/agents/KV8_p2/shadow/vllm_ascend/{core/deepseek_v41.py,attention/dsa_v41.py,models/deepseek_v41/model.py}` |
| 测试脚本 | `a2/agents/KV8_p2/p2_engine.py` |
| 原始数据 | `a2/logs/raw/018-s1-ab.json`（主表）、`018-s0-convention.json`（量化器约定）、`018-run4.log` / `018-run5.log`（完整 stdout） |
| A3 上的副本 | `~/projects/dsv41-upstream-pr/agents/KV8_p2/`（容器内 `/work/agents/KV8_p2/`） |

```bash
# 上传（本机 → COS → A3），再在容器里把补丁叠到影子包
bash a2/scripts/cos-xfer.sh put a2/agents/KV8_p2/p2_engine.py kv8_p2/p2_engine.py
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr/agents/KV8_p2 && \
  bash ~/projects/dsv41-upstream-pr/tools/cos-xfer.sh get kv8_p2/p2_engine.py ./p2_engine.py'

# 跑（只用 c1；退出码 75 = 没抢到锁）
ssh A3-node1 'source ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --name kv8p2 --timeout 700 -- \
  bash -c "cd /work/agents/KV8_p2 && export TMPDIR=/work/agents/KV8_p2/tmp \
    PYTHONPATH=/work/agents/KV8_p2/shadow KV8_RAW=/work/agents/KV8_p2/raw && python3 p2_engine.py"'

# 影子包怎么来的（容器内，一次性）
cp -r /vllm-workspace/vllm-ascend/vllm_ascend /work/agents/KV8_p2/shadow/vllm_ascend
cp /work/agents/KV8_p2/{dsa_v41.py} /work/agents/KV8_p2/shadow/vllm_ascend/attention/
cp /work/agents/KV8_p2/deepseek_v41.py /work/agents/KV8_p2/shadow/vllm_ascend/core/
cp /work/agents/KV8_p2/model.py /work/agents/KV8_p2/shadow/vllm_ascend/models/deepseek_v41/
```

> **harness 自身的两个坑（记录，避免下次重踩）**：
> ① 这棵树**直接 import `vllm_ascend.attention.dsa_v41` 会撞一个既有循环 import**
> （`dsa_v41 → dsa_v1 → attention_v1 → device_op → ops.triton… → vllm_ascend.ops.__init__ → fused_moe → device_op`），
> **原树也一样**；先 `import vllm_ascend.ops` 再导子模块即可（脚本里已这么做）。
> ② 本代 torch_npu **没有 `torch.npu.CUDAGraph`**，要用 `torch.npu.NPUGraph()` + `torch.npu.graph(g)`。

---

## 7. 给主代理的决策项

| 优先 | 决策 | 依据 |
|---|---|---|
| **P0** | **KV8 的容量账要重写**：`KV8-PLAN §1.1` 的 4421 → 2405 B/token（×1.84）**在真实分配器上不成立**，实测 ×1.032。要么把 SWA 一起量化（→ 约 2137 B/token，×1.98，但破坏 C2 的"SWA 不动"），要么承认 KV8 在当前形态下**不带来容量收益** | 本日志 §4【实测】 |
| **P0** | 把 §4 的结论同步进 `docs/KV8-PLAN.md`（§1.1 / §3 / §5 Phase 2.5） | 同上 |
| P1 | 用真权重/真 indexer 复现 §5-① 的 NaN（否则 prefill 侧不能判"可用"） | 本日志 §5① |
| P1 | 若 KV8 继续做：把 decode 读侧从"页内二维索引"换成**页粒度 staging + flat `index_select`**（015 §5.1 的 15× 带宽差），或者干脆接 §4 的 SWA 量化一起做 | 015 §5.1 / 本日志 §5② |
| P2 | 端到端（GSM8K/Vision/step 时间）：**未做** | 本日志 §5④ |

**纪律**：只用 c1（die 6）；没写 `/tmp`；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；
没写 `upstream-v41/`；没动 `dsv41-release/`；跨机传输全走 coscli；容器/影子包改动都在 `agents/KV8_p2/`。
