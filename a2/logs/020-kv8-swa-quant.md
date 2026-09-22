# 020 — KV8-SWA：把 `ori_kv`（SWA 窗口）也量化

> 2026-09-22 01:2x–01:4x CST。执行：子代理 **KV8_swa**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`）。全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` /
> Phy-ID 8–15；没用 `/tmp`；没手设 `ASCEND_RT_VISIBLE_DEVICES`；没写 `upstream-v41/`；没动 `dsv41-release/`。
> 代码在**影子包** `agents/KV8_swa/shadow/vllm_ascend/`（= KV8_p2 影子包 + 本次 4 文件增量），
> 本仓副本 `a2/agents/KV8_swa/`。跨机传输全走 coscli。

---

## 0. 三句话结论（先给主代理）

1. **【实测】`scale_dim` 加上了，SWA 页真的变小了**（`131072 → 66560 B`），
   **但容量只涨到 ×1.135（540928 → 476416 B/block，4226 → 3722 B/token），不是 ×1.98。**
   ⛔ **原因不是 SWA**：4 个 hybrid slot 里有 3 个（ratio-2 槽）的页大小被
   **FP32 compressor state ring（32 行 × 1024 dim × 4 B = 131072 B/block，逐字等于）** 顶住，
   把 SWA 量化到 66560 之后**一分钱没省**（§2 的分槽对账是逐项列出来的）。
   要拿 ×1.98 得**同时**缩小 state ring（不在 KV8 范围，需要主代理决策）。
2. **【实测】SWA 量化在数值上是干净的**：解码形状 **逐比特精确**（`max_abs = 0` / `rel_L2 = 0`，
   与 018 的 long-KV 同款无损对照）；真量化 `rel_L2 = 5.46e-3`、`cos = 0.9999856`
   （018 只有 long-KV 时是 5.42e-3 ⇒ **SWA 加进来几乎不额外损失精度**）；
   prefill 形状在**因果安全**的 top-k 下同样逐比特精确（§4、§5）。
   018 §5① 的 NaN **没有复现**：同一形状/同一数据，A vs A 自身就有 7.6e-3 的非确定性，
   换因果安全 top-k 后误差归零 ⇒ **NaN 是 harness 的假象，不是 KV8 读路径的问题**【实测+推断，§5】。
3. ⛔ **【实测·关键】图内性能不达标，而且比 018 测到的严重得多**：
   用**生产形状的 40 层整图**（不是 018 的单层 replay）测：
   SWA 层 **+169.4 µs/层**、带 long-KV 的整层 **+436.1 µs/层**（两者都远超 ≤0.2% 判据）；
   而**把 rebuild 预置好、只留算子**时增量是 **+0.16 µs/层** ⇒ **成本 100% 在 rebuild 的 gather/dequant，
   不在 `npu_sparse_flash_mla`**。⇒ 直接推算 **≈ +7.8 ms/step（30 ms 的 +26%）**。
   ★ 同时说明 **018 的「图内 +5.2 µs/层」是被单层 replay 的 host 开销掩盖后的欠估**（§6）。

---

## 1. ★ cannbot 对照（先做，不占卡）

在 A3-node1 只读查了 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`。

### 1.1 `model/model-infer-kvcache/SKILL.md`（本任务最相关）

| cannbot 说什么 | 我们采纳/没采纳 | 理由 |
|---|---|---|
| **滑窗 + 长序列的硬约束**（原文）：*"滑窗 Decode 必须保留窗口约束：`sparse_mode=4` + `pre_tokens={sliding_window}` + `next_tokens=0`…**长序列 `KV_len > sliding_window` 的正确性必须靠模型层保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，不是 op 层负责**"* | ✅ **直接采纳，并成为本次读侧的设计依据** | 我们**没有**改 `seqused_ori_kv`（它是 cmp 因果 mask 的输入），而是按"模型层保证"的思路：**按窗口重建被读的那些行**，`ori_mask_mode=4` / `ori_win_left=127` 一个字没动 |
| block/slot 映射：`逻辑块 = pos // block_size`、`物理 slot = block × block_size + 偏移` | ✅ 采纳（018 已采纳） | SWA 侧同样成立（见 §3 的算子源码对照） |
| 量化 KV 的 FA 走 `antiquant_mode` / `dequant_scale_key` | 不适用 | 那是 FA v1/v2 的接口；V4.1 用 `npu_sparse_flash_mla`，本代**没有**它的 antiquant 变体 |
| **`model-infer-kvcache` 全文没有任何"滑窗场景下 KV 量化"的专门约束或推荐形态** | 记录 | `grep -rn "滑动窗口\|滑窗\|sliding_window\|SWA" model-infer-quantization/` **命中 0 条** ⇒ 量化 skill 里没有 SWA 专门条款；我们只能按通用 C8 卡片 + 本次实测走 |

### 1.2 `model/model-infer-quantization/SKILL.md:424-451` + `references/quantization-fusion-and-benefit.md`

* **收益口径（decode 四段）**：`T_decode ≈ T_attn + T_FFN_GEMM + T_dispatch + T_overhead`，
   其中 `T_attn`（MLA/softmax/**KV 读写**）"**通常不量化 → 0**"，
   且明确写 **"eager 数据不进收益结论，仅用于功能验证"**。
   ⇒ **正好解释本次结论**：KV 量化（含 SWA）**不省 attention 时间**，它的收益是**容量/显存**；
   我们实测的图内增量全部落在 `T_overhead`（额外的 gather/dequant/scratch）上。
   ★ 按这个口径报告：本次收益栏只能填**显存/容量**（×1.135），**时延栏是负的**（§6）。
* **等价性自检 / "不能只看代码 diff"**：✅ 本次主表全部是**端到端算子输出**对拍 + 一个**独立 fp32 golden**。
* `references/quantization-structure-cards.md:174-206`（C1/C8 卡片）：**"C8 不是改 dtype"**、
   **"验证时检查 FA 是否真实消费 int8 cache 和对应 scale"** ⇒ ✅ 我们的判据就是
   **`pool_bytes_per_block` / 页尺寸是否真的变了** + 算子输出对拍，不是"spec 改了"。

---

## 2. ★ 容量：SWA 页确实砍半了，但收益被 state ring 吃掉【实测】

判据（任务书）：`pool_bytes_per_block` 应从 524288/540928 降到 ~266000（≈×1.98）。

| 组合（long-KV / SWA） | pool B/block | B/token | 24 GiB 下 block 数 | 容量比 | slot 页 |
|---|---:|---:|---:|---:|---|
| bf16 / bf16（现状） | 540928 | 4226 | 47639 | ×1.000 | `[131072, 131072, 131072, 147712]` |
| **int8 / bf16**（= 018 p2 现状） | 524288 | 4096 | 49152 | ×1.032 | `[131072, 131072, 131072, 131072]` |
| bf16 / int8 | 540928 | 4226 | 47639 | ×1.000 | `[131072, 131072, 131072, 147712]` |
| **int8 / int8（本次）** | **476416** | **3722** | **54090** | **×1.135** | `[131072, 131072, 131072, 83200]` |

**★ 逐项对账（`F2_binding`，逐字复算 `plan_cache_slots` 的 `max(...)`）**：

| slot | 层 | long-KV+index | state ring | 每个 SWA 别名页 | capacity | **binding** |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2（ratio2） | 41600 | **131072** | **66560**（int8，原来是 131072） | 131072 | **compressor_state** |
| 1 | 8（ratio2） | 41600 | **131072** | 66560 | 131072 | **compressor_state** |
| 2 | 14（ratio2） | 41600 | **131072** | 66560 | 131072 | **compressor_state** |
| 3 | 20（ratio1） | 83200 | —（ratio1 无 state） | 66560 | 83200 | **long_kv+index** |

* SWA int8 页 = `128 × (512×1 + 4×2) = 66560 B` **确实是一半**【实测，`reshape_cache` 的 stride：
  payload `(320,128,1,512)` stride `131072`（= 槽页）、scale `(320,128,1,4)` stride `65536`；
  §2 表里的 66560 是 `_cache_plane_sizes` 的和】。
* 但 ratio-2 三个槽的页是 **`DeepseekV41CompressorStateSpec`**：
  `storage_block_size 32 × 1 head × (head_size 2×512=1024) × 4 B(FP32) = 131072 B`，
  **恰好和 SWA 别名页同值** ⇒ 018 把这个 131072 记成了"SWA 顶着"，
  本次把 SWA 降到 66560 后页**没变**，**证明 binder 是 state ring，不是 SWA**【实测】。
* **【推断·算术】若要 ×1.98**：把 state ring 也变小（或把它从 KV slot 里去别名）后，
  pool = `3×66560 + 83200 = 282880 B/block = 2210 B/token`（**×1.91**，与 018 的外推同向）。
  ⇒ **×1.98 不是"SWA 量化"一个问题，是 SWA + state ring 两个问题**；后者不在 KV8 范围。

原始数据：`raw/020-s0-capacity.json`（含 F 与 F2）。

---

## 3. 实现：SWA spec 的 `scale_dim` + 读侧重建（4 文件，+175 行）

补丁：`agents/KV8_swa/kv8-swa.patch`。开关：`VLLM_V41_KV8_SWA`（默认继承 `VLLM_V41_KV8`）。

| # | 文件 | 改动 |
|---|---|---|
| 3.1 | `core/kv_cache_interface.py` | `AscendSlidingWindowMLASpec` 加 `scale_dim: int = 0` / `scale_dtype`，`real_page_size_bytes` 计入 scale 平面，**`merge()` 带上这两个字段**（不加就会在 `_uniform` 合并时丢掉 ⇒ 页几何与视图不一致） |
| 3.2 | `core/deepseek_v41.py` | `kv8_swa_enabled()` / `swa_plane_kwargs()`（`int8 + scale_dim=4 + fp16`）；`DeepseekV41SWASpec.__post_init__` 加 `int8 ⇒ scale_dim==KV8_SCALE_DIM` 断言 |
| 3.3 | `models/deepseek_v41/model.py` | `AscendDeepseekV41SWACache.get_kv_cache_spec` 用 `swa_plane_kwargs()` 构造 SWA spec |
| 3.4 | `attention/dsa_v41.py` | ① 写入侧：两处 SWA scatter（`preprocess` / `multistream_preprocess`）改走 `kv8_swa_store` ⇒ **C1 红线**：量化就在 scatter 那一步（`dsa_v41.py:371/430`），索引/下游只见过 BF16 行。② 读取侧：`kv8_ori_plane()` + `_native_attention` 里 `ori_kv` 为 tuple 时重建 |

**★ C1 红线对账**：SWA 行的量化点 = 它自己的 scatter 调用点（`kv8_store_rows(swa_cache_layer.kv_cache[0], swa_metadata.slot_mapping, kv)`），
索引器在此之前已经用全精度 `latent` 选完块，量化后的行**不回流**到任何选择逻辑。C2 的"只做 long-KV"是**本次要打破的目标**（任务书要求），
C3（scale 与 int8 同页）用同一个 `scale_dim` 机制满足，C4（slot 映射与设备侧 block table 一致）见下，C5（不碰 `dsa_cp.py`）保持。

### 3.1 ★ 读侧设计：swa 的 slot 映射与 long-KV **不同**，而且不能改 `seqused_ori_kv`

【代码事实】`sparse_flash_mla_swa_block_vector.h::GetOriSparseKeyGmOffset`（容器内
`/tmp/uninit_src/qli_csrc/build/binary/ascend910_93/src/sparse_flash_mla/op_kernel/arch22/`）：

```cpp
int32_t oriLenLimit = actualSeqLengthsKVGm.GetValue(runInfo.bIdx);
if (logicalIdx < 0 || logicalIdx >= oriLenLimit) return -1;
blockTableIdx     = logicalIdx / constInfo.paOriBlockSize;   // 逻辑块
idInBlockTable    = oriBlockTableGm_.GetValue(bIdx * oriMaxBlockNumPerBatch + blockTableIdx);
keyOffset         = idInBlockTable * oriKvStride0 + n2*headDim*paOriBlockSize + inBlockIdx*headDim;
```

而 band mask（`sparse_flash_mla_swa_kernel.h::CalcParams`）是：

```
oriMaskRight = actOriS2Size - actS1Size + s1EndIdx + oriWinRight      // 再 clamp 到 actOriS2Size-1
oriMaskLeft  = max(actOriS2Size - actS1Size + s1StartIdx - oriWinLeft, 0)
s2StartPoint = oriMaskLeft
```

⇒ 请求 `b` 的第 `i` 个 query 行读的逻辑位置是 `[seq_len - q_len + i - 127, seq_len - q_len + i]`，
**寻址是"绝对逻辑 token 坐标"**，与 `tests/deepseek_v41_reference.py::small_op_attention`
（`local_start = max(0, position - window_size + 1)`；`keys = local[local_start:position+1]`）
逐个位置一致 —— 这是**我们对拍用的 golden 的同一套语义**。

**⚠️ 不能把 `seqused_ori_kv` 截断到 128**（cannbot 给的另一条路）：同一个算子调用的
**cmp 因果 mask** 是 `cmpMaskS2Size = GetCmpMaskS2Size(bIdx, actOriS2Size, actCmpS2Size)`
—— 它由 `actOriS2Size`（= `seqused_ori_kv`）推出，截断会静默改掉长 KV 的因果边界。

**采纳的做法（`kv8_ori_plane`）**：

```
① 每个请求取窗口覆盖的**整页**（逻辑块 first..first+span-1，解码 ≤2 页）
② 页粒度 index_select 取出 int8/scale 页 → 反量化 → 直接 copy_ 进 PA_BBND scratch
   （整页拷贝 ⇒ 每行保留原来的页内偏移 ⇒ 一张页对应一张 scratch 页）
③ scratch block table：`col==first+k → base+k`，其余填 0（mask 永远读不到）
④ `seqused_ori_kv` / `ori_win_*` / `ori_mask_mode` **一个都不改**
```

* 优点：① 只搬"这一步真读的行"（解码 2 页/请求，不是整个前缀）；
  ② **整页丢失的是行级 scatter**，写入退化成一次连续 `copy_`（015 §5.1 的"flat/整页才快"）；
  ③ 全链无 D2H、无动态 shape ⇒ 可 capture。
* 边界：`positions` 越界行被 park 到 0 号位置（mask 不读它们）；prefill 分支用 `span.max().item()`
  （**只在 eager 的 prefill 走**，解码分支零同步）。

---

## 4. 数值对拍【实测】（B=8、窗口 128、topk 512、序列长 4000/3863/…/3041、真实分配器建页）

方法：真实 spec + 真实分配器（`group_cache_specs → allocate_cache_config → allocate_cache_views`）
建页；真实写入入口 `kv8_store_rows` 填；真实读取入口 `DeepseekV4.1EagerAttentionImpl._native_attention`
跑 `npu_sparse_flash_mla`。block table 用**打乱过的池**（专治"table 没被用上"）。
SWA 表按**绝对逻辑块**建（`b_lo/b_hi` → 两页），只映射窗口那两页（与生产一致）。

| 臂 | 内容 | vs A（BF16） |
|---|---|---|
| **B** | int8 long-KV + 无损量化器 | **`max_abs = 0.0`、`rel_L2 = 0.0`、`cos = 1.0000001`** ⇒ **逐比特** |
| **C** | **int8 long-KV + int8 SWA**（都无损） | **`max_abs = 0.0`、`rel_L2 = 0.0`、`cos = 1.0000001`** ⇒ **逐比特** |
| **C\*** | 都换**真 `npu_dynamic_quant`** | **`rel_L2 = 5.4616e-3`、`cos = 0.9999856`、`max_abs = 7.3e-4`** |
| C\* 的 SWA 窗口行本身 | 反量化行 vs BF16 行 | `rel_L2 = 4.43e-3`、`cos = 0.9999905`（018 的 long-KV 行是 4.44e-3） |

**独立 fp32 golden（`G_golden`）**：A vs golden `rel_L2 = 2.26e-3`（= 算子噪声地板），
C 与 A **完全相同**，C\* vs golden `5.02e-3` ⇒ 三条臂都落在噪声+量化该有的位置。

★ 判据对照：**容量：×1.135（不达标，原因见 §2）**；**逐比特：✅**；**真量化 rel_L2 ≤ 1e-2：✅ 5.46e-3**。

---

## 5. prefill 形状 + 018 那个 NaN【实测 + 推断】

**先发现一个 harness 假象**（值得写进下次的踩坑清单）：

| 对照 | 结果 |
|---|---|
| **A vs A（同一条 BF16 臂跑两遍）** | **`rel_L2 = 7.64e-3`** ⇒ prefill 形状下**算子自身不确定**（tile 会读到自己 mask 之外的行；018 的 harness 把那些行留成了 0/垃圾，读到的内容取决于内存状态） |
| int8（无损）vs A，**top-k 用随机下标**（每行选自己看不见的位置） | `rel_L2 = 8.38e-3` ← **就是上面那个非确定性，不是量化误差** |
| int8（无损）vs A，**top-k 改成因果安全**（`< p-127`，即 mask 一定会放行的行） | **`max_abs = 0.0`、`rel_L2 = 0.0`** ⇒ **逐比特精确**【实测】 |
| int8（真量化）vs A（同形状） | `rel_L2 = 9.99e-3`、`cos = 0.99995`、`nan_total = 0`、`inf_total = 0` |

**018 §5① 的 NaN 定性**：

| 证据 | 结论 |
|---|---|
| 本次同形状（每请求 3 个 query 行）+ 真量化器：**`nan_total = 0`**（`D_prefill_real`） | **未复现** |
| 把真量化的**行值**写进 BF16 布局、走生产读路径（`H_nan_probe`） | 有限、`nan_total = 0` ⇒ **值本身不会造 NaN** |
| 用 018 那种**不一致的 metadata**（`seqused_ori_kv=128` 而 cmp 跨 4000）重跑 | **仍然有限**（lossless 与 real 都不 NaN）；只是输出整体跑偏（`rel_L2 = 0.42`） |
| A vs A 就有 7.6e-3 非确定性 | prefill 形状下算子的**读范围**本身不干净，018 的 NaN 更像是"随机读到了未初始化行" |
| 真量化器：`zero_scale_groups = 0`、`all_zero_rows = 0`、scale ∈ [7.2e-3, 7.75e-3] | 018 观察到的"scale_min = 0"是**整块页**（含从未写过的页）的最小值，不是被读行的性质 |

⇒ **【推断】018 的 NaN 是 harness 假象**（prefill 形状 + 未初始化/越界行 + 该 harness 的 metadata 不自洽），
**不是 KV8 读路径或量化器的问题**；要彻底钉死需要"真 indexer top-k + 真 metadata"复现，
本次是**等价构造**（同形状、同数据、真量化器）下**未复现**，故标 **【未确认：未复现】** 而非【已否决】。

---

## 6. ⛔ 图内性能：换"生产形状整图"测后，结论比 018 严重

**方法学问题（本次最大的一个坑）**：单层 replay 的 wall time 里，
**图 replay 的 host 开销约 320 µs**（空图 replay 也要 ~16 µs/次，见 `I_swa_rebuild_micro.parts_graph_us.replay_overhead`），
它随图内节点数漂移 ⇒ **018 的"+5.2 µs/层"是欠估**。本次改用**一个图装 40 层**（= 生产 decode 的形态，一次 replay 40 层），
再除以 40：

| 图（40 层一图） | BF16 | 全 int8 | 增量 |
|---|---:|---:|---:|
| **SWA-only 层**（`has_cmp_kv=False`，40 层重复） | 24.42 µs/层 | 193.81 µs/层 | **+169.39 µs/层** |
| **整层**（SWA + long-KV/cmp 读） | 44.38 µs/层 | 480.47 µs/层 | **+436.09 µs/层** |
| ⇒ 差出来的 cmp 读路径 | | | **+266.70 µs/层** |

**归因【实测】**：

| 对照 | 增量 |
|---|---|
| **把两个 rebuild 都替换成"预先建好的 scratch"**（只留算子） | **+0.16 µs/层** |
| SWA rebuild 单独 micro（同图内重复 20 次） | **164 µs/次**：page_select 49.2 / dequant 18.2 / table_build 17.8 / copy 16.3 / 空图 15.8 |

⇒ ① **算子本身没问题**（它读 scratch 与读真页等价且几乎零成本）；
② **成本全在 rebuild 的 gather/dequant**，有效带宽只有 ~18–22 GB/s（搬到 3 MB 用了 ~170 µs），
与 015 §5.1 的"交错页 2D 高级索引只有 60–90 GB/s、flat `index_select` 才 1.2 TB/s"同一个病；
③ 本次虽然把行级 scatter 换成了整页 `copy_` 与页粒度 gather，**`kv_i8[phys]` 这种二维高级索引依然极慢**
（16 页 = 8 KB 花了 49 µs ⇒ **不是带宽瓶颈，是 kernel 效率**）。

**⇒ 生产步时间推算（40 SWA 层 + 4 个 long-KV 源层）**【推断·按实测外推】：
`40 × 169 µs + 4 × 267 µs ≈ 7.8 ms/step` = **30 ms 的 +26%** ⇒ **判据（≤+0.2%）严重不达标**。
（018 的 long-KV-only 若按同一方法重测，也应是 4 × 267 ≈ +1.07 ms/step，而不是 +0.021 ms。）

**下一步的建议（都是【推断】，未验证）**：
1. 用**专用 gather 算子**替换二维高级索引：`torch.index_select` / `torch.take` / 或把页索引拍平成一维后
   `index_select`（015 §5.1 的快路径 1.2 TB/s）；或
2. **自写 AscendC 小算子**（一次 kernel 完成 slot→读→反量化→写 scratch；按 `AGENTS.md` §6
   应走 `ops/ops-profiling/` 的四文件模板）；或
3. 认清"KV 量化换容量、不换时延"（cannbot 的 `T_attn` 栏本来就是 0）：**如果容量收益因 state ring 只到 ×1.135，
   而代价是 +26% step，那这条路线在当前形态下是负收益** —— 需要主代理先决策分母（state ring）怎么处理。

---

## 7. 五条判据总表【实测】

| # | 判据 | 期望 | 本次实测 | 结论 |
|---|---|---|---|---|
| 1 | 容量 `pool_bytes_per_block` | ~266000（×1.98） | **476416 B/block（×1.135）**，4226 → 3722 B/token | ⛔ **不达标**；原因 = 3 个 ratio-2 槽被 **FP32 state ring 131072 B** 顶住（SWA 已砍到 66560，白砍） |
| 2 | 逐比特（无损量化器） | `max_abs=0 / rel_L2=0` | **解码 `0.0 / 0.0`；prefill（因果安全 top-k）`0.0 / 0.0`** | ✅ |
| 3 | 真量化精度 | `rel_L2 ≤ 1e-2` | **5.46e-3（cos 0.9999856）**；SWA 行本身 4.43e-3 | ✅ |
| 4 | 图内性能 | ≤ +0.2% | **+169 µs/层（SWA 层）、+436 µs/层（整层）⇒ ~+7.8 ms/step = +26%** | ⛔ 不达标（且 018 的 +5.2 µs/层是方法学欠估） |
| 5 | 端到端 GSM8K-200 | 197–199 不退化 | **未做**（预算用尽；起服要 476 GB 权重，单卡做不到） | 【未做】 |

---

## 8. 交付物与复现

| 东西 | 路径 |
|---|---|
| 本次增量补丁（vs KV8_p2 影子包，4 文件 175 行新增） | `a2/agents/KV8_swa/kv8-swa.patch` |
| 改后的 4 个文件（可直接覆盖影子包） | `a2/agents/KV8_swa/shadow/vllm_ascend/{core/deepseek_v41.py,core/kv_cache_interface.py,attention/dsa_v41.py,models/deepseek_v41/model.py}` |
| Harness（18 个臂：容量/容量分解/AB/真量化/golden/prefill/NaN 探针/微基准/整图） | `a2/agents/KV8_swa/p3_swa.py` |
| 原始数据 | `a2/logs/raw/020-s0-capacity.json`（F + F2）、`020-s2-arms.json`（最终主表）、`020-run.log`（最终 stdout）；过程稿 `020-run2..6.log`、`020-s1-arms*.json` |
| A3 副本 | `~/projects/dsv41-upstream-pr/agents/KV8_swa/`（容器内 `/work/agents/KV8_swa/`，影子包在 `shadow/vllm_ascend/`，由 `KV8_p2` 影子包 + 本次 4 文件叠加而成） |

```bash
# 跑（只用 c1；退出码 75 = 没抢到锁）
ssh A3-node1 'source ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --name kv8swa --timeout 1100 -- \
  bash -c "cd /work/agents/KV8_swa && export TMPDIR=/work/agents/KV8_swa/tmp \
    PYTHONPATH=/work/agents/KV8_swa/shadow KV8_RAW=/work/agents/KV8_swa/raw && python3 p3_swa.py"'
```

**harness 自身的三个坑（记录，避免重踩）**：
① 单层 replay 的 host 开销 ~320 µs，**必须用 40 层整图**才量得到真实每层增量；
② prefill 形状下算子 tile 会读到 mask 之外的行 ⇒ **A vs A 都不逐比特**，对拍必须用**因果安全 top-k** 或只比解码；
③ `allocate_cache_views` 出来的视图是 4D `[pages, block, 1, head]`，二维索引后要 `.squeeze(-2)` 才能喂 `kv8_dequant_rows`。

**纪律**：只用 c1（die 6）；没写 `/tmp`；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；
没写 `upstream-v41/`；没动 `dsv41-release/`；跨机全走 coscli。

---

## 9. 给主代理的决策项

| 优先 | 决策 | 依据 |
|---|---|---|
| **P0** | **KV8 的容量账要按 state ring 重写**：只做 SWA 量化拿不到 ×1.98（实测 ×1.135）；要么动 `DeepseekV41CompressorStateSpec`（FP32 32×1024 = 131072 B/block，缩它才到 ×1.91），要么承认 KV8 当前形态无容量收益 | §2【实测】 |
| **P0** | **图内性能口径要重测**：018 的 +5.2 µs/层是单层 replay 的欠估；按生产形状整图，KV8 读侧是 **+169 µs/层（SWA）/ +267 µs/层（cmp）** ⇒ 整步 +7.8 ms（+26%）。在 gather 换专用算子（或自写 kernel）之前，**KV8 + SWA 的时延是负收益** | §6【实测】 |
| P1 | 018 的 NaN 可以**降级**：本次同形状真量化器未复现，且 A vs A 自身就有 7.6e-3 非确定性 ⇒ 先按 harness 假象处理；要结案需真 indexer top-k + 真 metadata | §5 |
| P1 | 若继续 SWA 量化：先做 §6 的优化 1（`index_select`/`take` 替二维高级索引）再看能不能进 ≤ +0.2% | §6 |
