# 015 — KV8（线 2）Phase 0 + Phase 1：TND/np_sparse_flash_mla 的可行性判定

> 2026-09-22 00:19–00:35 CST。执行：子代理 **KV8_p0**。机器：**A3（A3-node1）槽位 c2 = die 7**
> （`Ascend910_9382`，容器 `prbench-c2`，容器内源码 commit `e43cf1e9f`）。
> 全程只用 c2；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；没写 `/tmp`；没手设 `ASCEND_RT_VISIBLE_DEVICES`。
>
> **本文的靶子**是 `docs/KV8-PLAN.md` §2 的核心假设：
> *"只要把 gather 出来的 KV 拼成连续的 `[K,1,kv_d]`（TND），配 `layout_kv="TND"`，就完全不需要碰 block table"*。

---

## 0. ★ 三句话结论（先看这段）

1. **【实测·否决】KV8-PLAN §2 的 TND 路线在 A2/A3 上不存在。**
   `npu_sparse_flash_mla` 在 arch22（A2/A3）**只编译 `TND Q × PA_BBND KV`**：
   ```
   Aurora SparseFlashMla only compiles TND Q with PA_BBND KV.
   [FUNC:Parse][FILE:sparse_flash_mla_tiling.cpp][LINE:1160]
   ```
   设备选择表（`op_kernel/sparse_flash_mla_template_tiling_key.h`）里 **每一个** tiling key 的
   `KV_LAYOUT_T` 都是 `PA_BBND` —— 3 张表（host / A5 / A2A3）共 8 条，**没有一条是 TND KV**。
   README 里 "TND/TND" 那一行是**接口级**说法；这台机器上的算子实现没给它 kernel。
   ⇒ **Phase 0.3 = 不通过**，且它出现在 Phase 1.2/1.3 之前，是**设计级**拦路虎。
2. **【实测·通过】替代设计成立，而且数值上是"逐比特精确"的**：
   `PA_BBND scratch 平面（identity block table）+ 索引重编号 0..511` 与生产路径
   （真 block table + 真实逻辑下标）**输出完全相同**（`max_abs = 0.0`、`rel_L2 = 0.0`、`cos = 1.0000001`，
   两者对 fp32 golden 都是 `rel_L2 = 2.77e-3 / cos = 0.999996`，即算子自身的噪声地板）。
   **【实测】`-1` 填充也工作**（300/512 有效 vs 局部 golden：`rel_L2 = 2.84e-3 / cos = 0.999996`）。
3. **【实测】性能与图兼容都过关（在 ACLGraph 内），但 chunked prefill 那一侧有硬伤**：
   decode（B=8，4 层）图内增量 **+17 µs/层 ⇒ +0.068 ms/step = 0.23%**（判据 0.3 ms ⇒ **通过**）；
   可 eager 帧内这条链是 **+650~970 µs/层**（host 派发主导，判据按 step 计 ⇒ 若走 eager 会被否决）；
   **prefill（Q_T×topk 行）是个独立的坑**：2048-token chunk ⇒ 每层 537 MB int8 + 1.07 GB bf16，
   实测 **11.2 ms（flat）/ 26.2 ms（strided）每层** ⇒ **4 层 45–105 ms/step**，**这条路在 prefill 上不可用**。

---

## 1. Phase 0.1 — gather 量到底多大（含被 §1.3 漏掉的 prefill 那一项）

### 1.1 口径（全部来自代码事实，不是估的）

| 量 | 值 | 出处 |
|---|---|---|
| SWA 窗口 `window_size` | **128** | `sliding_window`（`dsa_v41.py:819`），SWA 侧 `ori_win_left = 127`、`ori_win_right = 0` |
| 稀疏选块数 `index_topk` | **512** | `config.index_topk`（`dsa_v41.py:826`），`cmp_topk = index_topk` |
| 拥有全上下文 KV 的层 | **4**（layer 2/8/14/20） | `kv_source_layer_ids` |
| `head_dim` / plane 宽度 | **512**（nope 448 + rope 64 内嵌） | `model.py:361-373` |
| INT8 g128 每 token 每平面 | **520 B** = 512 int8 + 4×fp16 scale | `AscendMLAAttentionSpec.real_page_size_bytes`（`kv_cache_interface.py:129-136`） |

### 1.2 每步要"取出来"的 token 数（★ 关键：是 **per query token**，不是 per request）

> 下面这张表是 **【推断·算术】**：所有输入（128 / 512 / 4 层 / 520 B / head_dim）都是代码常量，
> 乘法本身没在设备上跑过；只有 §5 的耗时是 **【实测】**。

```
rows = Σ_requests ( ori: 128 + cmp: index_topk )        # 每个 query token 一份选择
与 §1.3 的差别：§1.3 只算了 decode（每请求 1 个 query token）；
chunked prefill 的 Q_T 个 query token 各有各的 512 个选块 ⇒ 行数 × Q_T。
```

| 场景 | 行数（4 层合计） | int8 读 | bf16 scratch 写 | scratch 再读 |
|---|---:|---:|---:|---:|
| **decode，B=8** | 8×640×4 = 20,480 | **10.6 MB** | 21.5 MB | 21.5 MB |
| decode，B=32 | 81,920 | 42.6 MB | 85.9 MB | 85.9 MB |
| **chunked prefill，Q_T=2048（1 请求）** | 2048×512×4 = 4.19 M | **2.18 GB** | 4.40 GB | 4.40 GB |
| prefill Q_T=2048 × 8 请求 | 33.6 M | 17.5 GB | 35.2 GB | 35.2 GB |

⇒ **decode 侧确实"≪ 全上下文"（§1.3 的判断成立）**；
⇒ **prefill 侧完全不成立**：它随 Q_T 线性放大（Q_T 个 query token 各挑 512 个，
同一批选块在算子内部被反复读）。这是 Phase 1 实测里唯一真正"结构上"的问题（§5）。

> 补充：**当前 BF16 算子自己也是这么读的**（PA 路径按 per-row 下标 gather），
> 所以 prefill 的读放大**本来就存在**；KV8 的增量是"写一遍 BF16 scratch 再读回来"，
> 也就是 prefill 侧**多付一倍读 + 一倍写**。

---

## 2. Phase 0.2 — `layout_kv="TND"` 下 `cmp_kv` / `cmp_sparse_indices` 的确切形状与语义

### 2.1 接口约束（README 原文，`csrc/attention/sparse_flash_mla/README.md`）

| 项 | TND 约束 |
|---|---|
| 组合 | 仅 `BSND/BSND`、**`TND/TND`**、`BSND/PA_BBND`、`TND/PA_BBND`；非 PA 场景 **`layout_q` 必须 = `layout_kv`** |
| `q` | `[Q_T, Q_N, D]` |
| `cmp_kv` | `[CMP_KV_S, KV_N, D]`（TND 惯例：跨 batch 拼接的总长） |
| `cmp_sparse_indices` | **`[Q_T, KV_N, K2]`**，`K2` = 每个 query token 从 `cmp_kv` 离散选出的 token 数；**A2/A3 上 K2 只支持 512 或 1024** |
| `cu_seqlens_q` | **必传**，`[B+1]`，前缀和、首元素 0 |
| `cu_seqlens_cmp_kv` | **有 `cmp_kv` 时必传**（TND 才能传 KV 侧 cu_seqlens） |
| `ori_sparse_indices` / `ori_topk_length` | 只在 "SWA 稀疏 ori_kv" 场景需要；**CSA 场景不用**（`ori_mask_mode=4` + `ori_win_left=127`） |
| 其它 | `cmp_mask_mode` 仅支持 3；`cmp_topk_length` 仍是**预留**入参（不可依赖） |

### 2.2 索引语义（arch22 kernel 源码，两个来源对账）

`op_kernel/arch22/sparse_flash_mla_csa_kernel.h:368-380`（`GetActualSeqLenCmpKV`）：

```cpp
} else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
    actualSeqCmpKVPrefixSum = actualSeqLengthsCmpKVGm.GetValue(bIdx);      // cu_seqlens_cmp_kv[b]
    actualSeqCmpKVNextSum   = actualSeqLengthsCmpKVGm.GetValue(bIdx + 1);
    return actualSeqCmpKVNextSum - actualSeqCmpKVPrefixSum;                // 本 batch 段长
}
```

`..._csa_block_vector.h:545-566`（`GetKeyGmOffset`）：

```cpp
realKeyGmOffset = runInfo.tensorCmpBOffset                       // = cu_seqlens_cmp_kv[b]
                + realS2Idx * kvHeadNum * headDim + n2Idx * headDim;
```

⇒ **`cmp_sparse_indices` 里的值是"batch 内局部下标"**（0 = 本 batch 段首），
地址 = `cu_seqlens_cmp_kv[b] + idx`。有效性判定是
`0 ≤ idx < cmpS2IdLimit`，`cmpS2IdLimit = clamp((p+1)/cmp_ratio, 0, Lc_valid)`；
**`idx < 0` 一律无效**（`GetKeyGmOffset` 里 `realS2Idx < 0 || >= s2IdLimit → -1`），
所以 `pad_sparse_indices(..., value=-1)`（`dsa_v41.py:225-229`）在 TND 下同样成立。

**【实测】metadata 算子接受 TND**：`npu_sparse_flash_mla_metadata(..., layout_kv="TND", cu_seqlens_ori_kv=…, cu_seqlens_cmp_kv=…)` 调用成功（`015-s0-layout.json` 的 `B_tnd_metadata = "ok"`）；
**【实测】主算子不接受** —— 见下。

---

## 3. ★ Phase 0.3 — TND 分支在 A2/A3 上**没有 kernel**（否决）

### 3.1 静态证据（容器内 e43cf1e9f 的真实源码）

`csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_template_tiling_key.h`：

```
// Aurora always uses TND Q and PA_BBND KV: C0 -> SWA, C1/C2 -> CSA.
#if !defined(__CCE_AICORE__)          // host validation：14 keys，KV_LAYOUT_T 全是 PA_BBND
    ... SMLA_LAYOUT_TND ... SMLA_LAYOUT_PA_BBND ...   (× 4 条)
#elif (__CCE_AICORE__ == 310)         // A5
    ... SMLA_LAYOUT_TND ... SMLA_LAYOUT_PA_BBND ...   (× 2 条)
#else                                 // A2/A3
    // A2/A3: keep the CSA single-head kernel; never compile A5-only flags.
    ... SMLA_LAYOUT_TND ... SMLA_LAYOUT_PA_BBND ...   (× 2 条)
#endif
```

⇒ 8 条 tiling key **全部**是 `TND Q × PA_BBND KV`。**没有 TND KV 的实例化**。
（arch22 的 `csa_kernel.h` / `csa_block_vector.h` / `csa_block_cube.h` 里**确实有**
`if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND)` 的分支代码 —— 但那段是**死代码**：
选择表里没有对应实例，编不出 kernel。**"看得到代码" ≠ "有 kernel"**，这正是本次要确认的东西。）

【实测·容器内 e43cf1e9f 逐文件计数】`grep -c "KV_LAYOUT_T == SMLA_LAYOUT::TND" op_kernel/arch22/*.h`
→ `csa_kernel.h:4`、`csa_block_cube.h:2`、`csa_block_vector.h:1`、`swa_block_cube.h:6`、`swa_kernel.h:4`、
`arch22_metadata.h:0`、`common_arch22.h:0` —— **代码在、实例不在**。

### 3.2 运行期证据（决定性）

同一份数据、同一批 Q、同样的 metadata 配置，只改 `layout_kv`：

| 臂 | layout_kv | 结果 |
|---|---|---|
| A `pa_real` | `PA_BBND` | ✅ 195.9 µs（中位）；vs golden `rel_L2=2.77e-3 / cos=0.999996` |
| B `tnd` | **`TND`** | ❌ `Execution_Error(EZ1008)`，tiling 阶段直接失败：<br>`Aurora SparseFlashMla only compiles TND Q with PA_BBND KV.[FUNC:Parse][FILE:sparse_flash_mla_tiling.cpp][LINE:1160]` |
| C `pa_scratch` | `PA_BBND`（identity table） | ✅ 197.4 µs；**与 A 输出逐比特相同** |
| D `pa_scratch_partial` | `PA_BBND`（300/512 + `-1` 填充） | ✅ 与局部 golden 一致（`rel_L2=2.84e-3 / cos=0.999996`） |

原始数据：`raw/015-s0-layout.json`。脚本：`agents/KV8_p0/s0_layout_kernel.py`。

### 3.3 这一条对 `KV8-PLAN.md` 的影响（要改文档）

* §2「为什么"按需"是可行的：算子支持 `layout_kv="TND"`」→ **前提在 A2/A3 不成立**；
* §2.1 三步流水里的 `②/③`（拼成 `[K,1,kv_d]` TND + 直接喂）→ **要走 PA_BBND scratch**；
* §1.3 的字节账**仍然有效**（gather 量确实小），只是**落地的载体**换了；
* §5 Phase 0.3 的判据（"确认 TND 分支在 arch22 里存在"）→ **判否**；
* §6 风险表 R6（"TND 布局的 SMLA 在 A2/A3 上没特化"）→ **从"中"升为已发生**。

---

## 4. ★ 替代设计（已用四臂对拍验证）：PA_BBND scratch + identity block table

```
① slot:   slot = cmp_block_table[req, idx // SB] * SB + (idx % SB)     （idx = -1 的行 park 到 slot 0）
② gather: 从 int8 面取 [rows, 512] int8、从 scale 面取 [rows, 4] fp16
③ dequant: k_bf16 = k_i8.to(bf16) * scale.repeat_interleave(128)（= view(rows,4,128) * scale[...,None]）
④ 布局:   取出来的行按"每请求 512 连续 token"排成一页对齐的 scratch 平面
          scratch 形状 [B*ceil(512/SB), SB, 1, 512]，block_table' = arange（identity）
          cmp_sparse_indices = arange(512)（+ 不足 512 时 `-1` 填充）
⑤ 调用:   npu_sparse_flash_mla(..., layout_q="TND", layout_kv="PA_BBND")   ← 原封不动
```

**判据对账（`015-s0-layout.json`）**：

* `C_pa_scratch` vs `A_pa_real`：`max_abs = 0.0`、`rel_L2 = 0.0`、`cos = 1.0000001`
  ⇒ **换布局不改数值**（同一批 KV、只换寻址），所以 KV8 的精度账**仍然只由量化决定**（K4：2.21%）。
* C 与 A 对 golden 的偏差相同（`2.77e-3 / 0.999996`）= 算子自身噪声地板（与 `logs/002` §2.3 的 N2′ 一致）。
* `-1` 填充（D 臂）也正确 ⇒ `pad_sparse_indices` 的语义在这条路上不用改。
* **重编号的安全性**：`idx` 变成"选择序的第几个"，而 mask 判据是 `idx < cmpS2IdLimit`；
  由于 indexer 只可能选出**可见**的压缩 token（`n_selected ≤ cmpS2IdLimit`），
  `0..n-1` 全部落在可见区间内 ⇒ decode/正常 prefill 下语义等价。
  **⚠️ 边界（【推断】，未实测）**：如果某一行的选块数 > 该行可见压缩长度（例如 indexer 没做因果过滤），
  重编号会把不可见的 token 变成"可见"。**Phase 2 必须加一条断言：每个 query 行的有效选块数 ≤ cmpS2IdLimit。**

---

## 5. Phase 1.2 — 性能（gather + dequant）：**图内过关，eager 不过关，prefill 有硬伤**

### 5.1 两种取数布局的差别（这是本次最有工程价值的发现）

| 布局 | 取数写法 | 实测带宽 |
|---|---|---|
| **hybrid page（520 B/token 交错）** | `k_view[page, off]`（2D 高级索引，行跨步 520 B） | **60–90 GB/s** |
| **flat 连续平面**（Engram device-index 的写法） | `torch.index_select(k_flat, 0, slot)` | **~1.2 TB/s**（4.29 GB / 7.17 ms，`015-s4`） |

⇒ **"scale 与 int8 同页"（C3）如果用 520 B 交错页，读侧会被 2D 高级索引拖慢 ~15×**；
`scale_dim` 的**页大小账**必须保持，但**页内排布**最好让 int8 载荷按平面连续
（这样 `flat[slot]` 才有出处）。`raw/015-s4-verify-bw.json` 还给了标定：
plain `copy_` **1161 GB/s**、fp32 `mul` **1098 GB/s**、连续 `index_select` **~1.2 TB/s** —— 这台 die 本身不慢，
慢的是**访问模式**。

### 5.2 decode 尺寸（B×512 行，4 层）—— 按帧分两种结论

| 帧 | 全链（gather+dequant+copy+SMLA） | SMLA only | 增量 | ×4 层 | 占 30 ms |
|---|---:|---:|---:|---:|---:|
| **ACLGraph（生产帧）** flat | 335.7 µs | 318.7 µs | **+17.0 µs/层** | **+0.068 ms** | **0.23%** ✅ |
| ACLGraph，strided 交错页 | 322.6 µs | 322.5 µs | **+0.17 µs/层** | +0.001 ms | 0.002% ✅（见注） |
| eager（非图帧） strided 全链 | 967.9 µs | 316.0 µs | **+652 µs/层** | +2.6 ms | **8.7%** ❌ |
| eager，flat（无 slot 计算） | 314.4 µs | 316.0 µs* | ~0（被 SMLA 的 host 派发盖住） | ~0 | ~0 |

* **判据（> 0.3 ms ⇒ 否决）**：图内 **通过**（0.068 ms）；eager 的 strided 写法 **不通过**（2.6 ms）。
  生产 decode 走 ACLGraph ⇒ **1.2 判"图内通过、eager 不通过"**。
* 注：strided 的图内 +0.17 µs 与 flat 的 +17 µs 之差，说明**图内这条链几乎完全被 SMLA 掩盖**
  （SMLA 本身 ~320 µs 且没吃满机器）；两个数都远在 0.3 ms 之下，但**别把 0.17 µs 当能力**。
* eager 那一行是**host 派发瓶颈**（B=4/8/32 三种规模耗时几乎一样：656/671/843 µs），
  所以"eager 慢"不等于"设备慢"——但 prefill 走 eager，这条要按下面 §5.3 单独算。
* `*` SMLA-only 的 eager 读数取自 **s2 的同机型 run**（316.0 µs），不是与 s5 同一次 run；
  s4 里 SMLA 的 eager 读数在同一量级（`015-s4` 的 `PS1_eager` 是端到端输出而非计时，未用于本表）。

### 5.3 ★ prefill（chunked）—— 真正的硬伤

`raw/015-s3-scale.json`（strided）与 `raw/015-s5-flat.json`（flat）按行数标定：

| rows | strided 全链 | flat 全链 | 说明 |
|---:|---:|---:|---|
| 4,096（= decode B=8） | 274 µs | 121 µs | 小尺寸：host/派发主导 |
| 16,384 | 456 µs | — | |
| 65,536 | 1.47 ms | — | |
| 262,144 | 6.38 ms | — | |
| **1,048,576**（= Q_T 2048 的**一层**） | **26.2 ms** | **11.2 ms** | 1.62 GB 流量 |

⇒ 一个 2048-token chunk、4 个 long-KV 层：
**flat 45 ms / strided 105 ms 每 step**。A2 的 prefill 吞吐是 4.3–5.6k tok/s（≈400–480 ms/2048 token 一步），
所以这是 **+10%~+26%** 的 prefill 开销 —— **不可接受**。

三条出路（**【未确认】**：本机都没实测，只给方向）：
1. **按请求去重**成"选块并集"（union）：行数从 `Q_T×512` 降到 `min(Lc, …)`；
   4K/32K 上下文（ratio 2）分别是 2K/16K 行 ⇒ 约 0.02/0.17 ms·层。
   代价：要算"每行下标 → 并集位置"的重映射，**图内做 unique/sort 有风险**（见 §6 的 Nonzero 案例）。
2. **prefill 期间退回全平面反量化**：每层额外 `Lc×520 B` 读 + `Lc×1024 B` 写（32K 上下文 ≈ 25 MB ⇒ 约 30 µs），
   **内存上是"int8 面 + 一份 BF16 scratch"**（峰值比现状更差，只在 prefill 期间存在）。
3. **prefill 不进 KV8**（保持 BF16 存储则容量收益归零 ⇒ 不成立）；
   **或**让 KV8 只在 decode 生效、prefill 阶段用"整层反量化"过渡（即 2）。

### 5.4 写入侧（顺带）

`torch_npu.npu_dynamic_quant(x.view(T*4, 128), dst_type=torch.int8)` 实现 g128 量化
（= `logs/007` §3.1 的推荐写法），B=8 时 **74 µs/层（eager，host 主导）**；
decode 每步只写 1 token/请求 ⇒ **可忽略**（与 KV8-PLAN §3.2 一致）。

---

## 6. Phase 1.3 — 图兼容：**可以进 ACLGraph（判过）**，但有一个必须避开的写法

### 6.1 逐级 capture（`raw/015-s2-graph.json` 的 `A_stages`）

| # | 级 | capture |
|---|---|---|
| 1 | slot 计算（block table → flat slot） | ✅ ok |
| 2 | **布尔掩码版**（`sl[sl >= 0]`） | ❌ **失败**：`aclnnNonzeroV2` → `rtStreamSynchronize ... stream is captured` |
| 3 | **park 版**（`torch.where(sl>=0, sl, 0)`，行数固定） | ✅ ok |
| 4 | + int8 gather | ✅ ok |
| 5 | + scale gather | ✅ ok |
| 6 | + dequant（cast × broadcast mul） | ✅ ok |
| 7 | + 写 scratch | ✅ ok |
| 8 | SMLA only | ✅ ok |
| 9 | **全链（slot→gather→dequant→scratch→SMLA）** | ✅ **ok** |

⇒ **只要不用数据相关的形状操作（`nonzero`/布尔索引/`unique`），这条链完全可 capture。**
这也是 KV8 读侧的一条硬性实现约束（等价于 KV8-PLAN §6 R2 的"按 Engram 手法"）。

### 6.2 replay 正确性（`raw/015-s4-verify-bw.json`，用真实量级的 fp16 scale 重做一遍）

| 检查 | 结果 |
|---|---|
| 图输出 vs eager 输出 | `rel_L2 = 0.0`、`max_abs = 0.0`、`cos = 0.99999994` ⇒ **逐比特一致** |
| **原地改输入后再 replay**（cache 内容整块换掉） | `rel_L2 = 0.0`（仍与新的 eager 结果一致） |
| 改输入后输出确实变了 | `mean|Δ| = 0.0199`（非零）⇒ 上一条不是"两边都空"的假阳性 |
| 量级自检 | 图输出 `rms = 0.0209`（与 eager 参考同值，非 0/非 NaN） |

⇒ **无 D2H 同步、无冻结快照、无动态 shape**；`npu_sparse_flash_mla` 的 README 也写着
"该接口支持 aclgraph 模式"。**Phase 1.3 = 通过。**

> ⚠️ **记录一次 harness 自身的坑**（不要当成算子的锅）：`s2_graph.py` 第一版的 replay 对拍
> 打印出 `ref_rms = 0.0`（参考臂全零）——那是脚本里 fp16 scale 用了**随机比特**（含 NaN/Inf）
> + 比较函数对全零输入的退化行为造成的。`s4` 用**真实量级**的 scale 重做后得到上面那张表。
> 结论用 `s4`，`s2` 只用于 §6.1 的 capture 逐级表与 §5.2 的耗时。

---

## 7. Phase 1.4（只记录）— TND 与 PA_BBND 在算子层面差多少

**做不了对照**：TND KV 在本机根本没有 kernel（§3），所以"同数据下 TND vs PA_BBND 的算子差异"
**无法在本代产品上测量**。能记录的只有接口级差异：

| 维度 | PA_BBND | TND |
|---|---|---|
| 额外入参 | `ori_block_table` / `cmp_block_table` | `cu_seqlens_ori_kv` / `cu_seqlens_cmp_kv` |
| 索引语义 | 逻辑 token 号（经 block table 映射到物理页） | **batch 内局部下标**（`+cu_seqlens[b]` 得地址） |
| A2/A3 kernel | ✅ 有（SWA + CSA 两模板） | ❌ **无**（仅 A5 的 host 表里也仍是 PA_BBND） |
| 稀疏索引宽度 K2 | 512 / 1024 | 同（接口层） |

⇒ 结论：**"省掉 block table" 这件事在 A2/A3 上换不来收益，反而撞墙**；
PA_BBND + identity table 已经把"逻辑连续"这件事表达出来了（代价是一张 B×4 的 int32 表 + 索引重编号）。

---

## 8. 判定汇总（对照任务书的判据）

| # | 判据 | 结果 |
|---|---|---|
| 0.1 | gather 总量 ≪ 全上下文 | decode ✅ 10.6 MB/step（B=8）；**prefill ❌**（Q_T 线性放大）|
| 0.2 | TND 下 `cmp_kv`/`cmp_sparse_indices` 形状与语义确定 | ✅ 已定（`[T,1,D]` / `[Q_T,1,512]`；索引 = batch 内局部下标；`-1` 无效）|
| **0.3** | **arch22 有 TND 稀疏分支的 kernel** | ❌ **没有**（静态表 + 运行期报错双证）⇒ **设计级否决**|
| **1.2** | 性能 ≤ 0.3 ms/step | **图内 ✅ +0.068 ms（flat）**；eager ❌ +2.6 ms；**prefill ❌ 45–105 ms/step** |
| **1.3** | 能否进 ACLGraph | ✅ 能（全链 capture + 改输入 replay 逐比特一致）；**禁用布尔掩码/动态形状** |
| 1.4 | TND vs PA_BBND 算子差异 | 无法测（TND 无 kernel），仅记录接口差异 |

**一句话**：**KV8-PLAN §2 的 TND 形态在 A2 上不可行**（0.3 判否），
但**"gather+反量化+原封不动喂 SMLA"这件事本身是可行的**——换成
**PA_BBND scratch + identity block table + 索引重编号**，数值逐比特等价、图内开销 0.23%；
**剩下的真问题是 chunked prefill 的读放大**（那一项要么做并集去重、要么 prefill 单独走全平面反量化）。

---

## 9. 替代方向（按性价比，给主代理决策）

| 优先 | 方向 | 依据 | 成本 |
|---|---|---|---|
| **P0** | **改写 KV8-PLAN §2 / §5**：去掉 TND 假设，改成 PA_BBND scratch 形态 | 本日志 §3/§4 | 30 min，纯文档 |
| **P0** | **只量化 `cmp_kv`，`ori_kv`(SWA) 完全不动** | 与 §4 的 scratch 形态天然一致（ori 直接用真 PA 缓存，不需要 gather/重编号） | 设计上已经这样 |
| **P1** | 读侧用 **flat 平面 + `index_select`**（Engram device-index 手法，1.2 TB/s）而不是交错页 2D 索引（60–90 GB/s） | 本日志 §5.1 | Phase 2 的设计约束 |
| **P1** | prefill：**按请求并集去重** 或 **整层反量化** | 本日志 §5.3 | 需实测，且要注意图内不能出现 `unique`/动态形状 |
| **P2** | 若并集/整层都不划算 ⇒ **KV8 只服务 decode**，prefill 侧另想 | —— | —— |
| **P3** | 自写一个 `flat[slot]` gather+dequant 小算子（AscendC） | `logs/007` §3 的 Level 1；本日志 §5.1 给出动机 | 中，先不做 |
| **P3.5** | 给 `sparse_flash_mla` 的 TPL_SEL **补一条 TND-KV 实例**，让 arch22 能编出 TND 分支 | 【推断】arch22 的 TND 分支代码**已经在**（`csa_kernel.h` 等，见 §3.1），接口层也允许；缺的只是选择表那一行 | 一行 + 一轮回归；**但那是从未被编译/测试过的死代码**，风险归我们，且改的是 vllm-ascend 的 `csrc/`（不是 CANN），升级要 rebase |

> 关于 P3.5 的定位：**这不是硬件限制**——kernel 里有 TND 分支、README 也允许 TND/TND；
> 挡路的是**编译期的 template selection 表没实例化**。所以"TND 路线"不是永久死，
> 但**不能把它当作 KV8 的既定前提**（要做到就得我们自己去动 csrc 并承担维护）。

---

## 10. 未确认清单（不许用相邻数字顶替）

| # | 未确认 | 为什么 | 怎么补 |
|---|---|---|---|
| ① | **A2（910B3）上是否完全一样** | 本次全在 A3 c2（910C，`Ascend910_9382`）；A2 本机 ssh 不可达 | 在 A2 上重跑 `s0_layout_kernel.py`（5 分钟），看 tiling 报错是否同一条 |
| ② | 图内 +0.17 µs 那一格是否真被 SMLA 掩盖 | 只测了 50~100 次的中位数，没有 profiler 级拆分 | 用 torch_npu profiler 看 kernel 时间线 |
| ③ | 重编号在"选块数 > 可见长度"时会不会错 | 需要真 indexer 输出才能构造 | Phase 2 加断言 + 用真权重跑一次 |
| ④ | prefill 的并集去重能不能进图 | 本机没测（`unique` 类算子有 capture 风险） | 单卡 capture 试验 |
| ⑤ | 实际生产 N1（query head 数） | 本次用 N1=16（省时间）；A2 的 TP 配置决定真值 | 从 A2 配置读 `num_attention_heads / TP` |
| ⑥ | `page_size`/`block_size` 真实取值对 scratch 的页数影响 | 本次用 SB=128；生产 `cache_config.block_size` 未在本机确认 | A2 起服日志里的 `block_size` |
| ⑦ | 端到端 step 时间 | 本次只测了"这一步流水"的孤立开销 | Phase 3（起服 A/B） |

---

## 11. 复现（脚本 / 原始数据 / 命令）

| 东西 | 路径 |
|---|---|
| 脚本（本仓） | `a2/agents/KV8_p0/{kv8_lib,s0_layout_kernel,s1_perf,s2_graph,s3_scale,s4_verify_bw,s5_flat}.py` |
| 脚本（A3 上） | `~/projects/dsv41-upstream-pr/agents/KV8_p0/`（= 容器内 `/work/agents/KV8_p0/`） |
| 原始数据 | `a2/logs/raw/015-s0-layout.json`（layout/kernel 四臂）、`015-s1-perf.json`（eager 分解，B=4/8/32）、`015-s2-graph.json`（逐级 capture + 耗时）、`015-s3-scale.json`（行数标定 strided）、`015-s4-verify-bw.json`（replay 正确性 + 带宽标定）、`015-s5-flat.json`（flat 变体 + 图内增量） |

```bash
# 取数（本地 → COS → A3）
bash a2/scripts/cos-xfer.sh put a2/agents/KV8_p0/s0_layout_kernel.py kv8_p0/s0_layout_kernel.py
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr/agents/KV8_p0 && \
  bash ~/projects/dsv41-upstream-pr/tools/cos-xfer.sh get kv8_p0/s0_layout_kernel.py ./s0_layout_kernel.py'

# 跑（只用 c2；退出码 75 = 没抢到锁，重试）
ssh A3-node1 'bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c2 --name kv8-s0 --timeout 850 -- \
  bash -c "cd /work/agents/KV8_p0 && export TMPDIR=/work/agents/KV8_p0/tmp && python3 s0_layout_kernel.py"'
```

**纪律**：只用 c2（die 7）；没写 `/tmp`；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；
没写 `upstream-v41/`；没手设 `ASCEND_RT_VISIBLE_DEVICES`。
