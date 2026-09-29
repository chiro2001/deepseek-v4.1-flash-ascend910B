# V4.1 DCP8 进展与容量模型（2026-09-29）

**目标**：8 chip 模拟 A2 上跑 `TP8 + --decode-context-parallel-size 8`，
KV 池容量接近 8×、正确性达标、decode 性能不退化。

**状态**：容量这条线已经**跑通并逐位对齐真机**；attention 执行路径**尚未接入**
（当前 DCP8 的服务输出一定是错的，只用于测容量）。

**约定**：【实测】= 本机跑出来的；【推断】= 由代码/公式推出来的；【未确认】= 没验。

---

## 1. 三道门（全部实测）

DCP8 不是"设个参数"就能开的，实测撞了三道 fail-fast 门，每道都拿到原文：

| # | 门 | 位置 | 报错原文 | 处置 |
|---|---|---|---|---|
| ① | PP=DCP=PCP=1 | `core/deepseek_v41.py::validate_cache_runtime` | `NotImplementedError: V4.1 initial runtime requires PP=DCP=PCP=1` | overlay 里按 `V41_DCP` / `V41_DCP_ALLOW_CAPACITY_PROBE` 放开 |
| ② | Engram 共享表 | `models/deepseek_v41/engram_hbm.py::EngramQueryGroup.from_vllm` | `ValueError: Engram HBM sharing requires EP and PP=PCP=DCP=1` | DCP **不新增 rank**（复用 TP rank），Engram 组仍按 hostname 分组 ⇒ 同一节点内语义不变，允许 |
| ③ | 滑窗 DCP | `vllm/v1/kv_cache_interface.py::SlidingWindowSpec.max_memory_usage_bytes` | `AssertionError: DCP not support sliding window.` | 去掉这条保守断言（公式不动），见 §3 |
| ④ | interleave 上界 | `vllm/config/vllm.py::validate_block_size` | `AssertionError: Block_size(32) should be greater than or equal to and divisible by cp_kv_cache_interleave_size (128).` | interleave 取 `gcd(block_size, STATE_RING_ROWS) = 32`，见 §4.1 |
| ⑤ | 我自己的补丁静默失效 | `vllm_ascend/platform.py` 里的宽 `except` | 日志只有一条 `WARNING [V41-DCP] interleave/cap patch setup skipped: ImportError("cannot import name 'set_dcp_size' ...")`，**服务照起、health 200，容量腰斩**（子代理实测 DCP2 从 1.977× 掉到 1.165×） | 去掉宽 except，改 fail-fast + 显式断言 |
| ⑥ | **请求时**才炸：滑窗 DCP 断言（在 vllm-ascend 自己的补丁里） | `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:472` | `AssertionError: DCP not support sliding window attn now.`（起服完全正常、health 200，**第一条请求**才死，EngineCore 直接退出） | 该补丁把 `dcp_world_size=self.dcp_world_size` 传给了**所有** group；上游 `kv_cache_coordinator.py:774` 本来就写成 `if isinstance(spec, FullAttentionSpec) else 1`，这个 vllm-ascend 补丁漏了。overlay 该文件并补上同一规则 |
| ⑦ | `ori_kv` 是 op 层硬必填 | `sparse_flash_mla_tiling.cpp:270` | `RuntimeError: call aclnnSparseFlashMla failed ... Parameter ori_kv of SparseFlashMla is invalid. Reason: The tensor of ori_kv is nullptr.` | 见 §5.2：改用 `seqused_ori_kv=0` 表达"本 rank 不贡献 ori" |

三道门都在**容量核算之前**，所以不存在"KV 缩了 8 倍但 attention 读全序列"的
静默算错组合 —— 它在第一道门就死了。【实测】

### 1.1 第 ④ 道门为什么存在（【实测】+【推断】）

`EngineCore._initialize_kv_caches` 在 `validate_block_size()` 之前，把

```python
vllm_config.cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups)
```

写成 = **min(group block_size)**。V4.1 里最小的是 compressor state ring 的
**32**（`STATE_RING_ROWS`），于是"interleave ≤ block_size 且可整除"这条通用断言
在 interleave=128 时直接炸。

取 `interleave = gcd(128, 32) = 32` 同时满足两件事：

1. 32 整除 128（也整除 32）⇒ 断言通过；
2. `compress_ratio(=2)` 整除 32 ⇒ **ratio=2 的压缩组不跨 rank** ——
   这是 `compress_ratios` 路径唯一的硬约束（推导见 §3）。

⇒ `interleave=32` 而不是 `=block_size`（SFA 的做法）。SFA 没有 ratio=2 压缩平面，
所以它可以要求 `interleave == block_size`；V4.1 不行。

---

## 2. 容量模型（已与真机逐位对齐）

离线复算器：`a2sim-ref/v41_pool_sim.py`（在容器内跑，不需要权重/设备）。

### 2.1 池子的四个 slot（【实测】）

`[V41-DCP-DIAG] pool_bytes_per_block=540928 (slots=[131072,131072,131072,147712])`

| slot | 对应 | page 字节 | 谁决定 |
|---|---|---:|---|
| 0/1/2 | layer 2/8/14（`compress_ratio=2`）+ index + state + 10 层 swa 别名 | 131072 | **swa 平面**（128 token × 512 × 2 B） |
| 3 | layer 20（`ratio=1`）+ index + 10 层 swa 别名 | 147712 | layer 20 的 kv+index |

合计 540928 B/块（128 全局 token）—— 与 `scripts/serve_a2.sh` 里
CED-POOL-GUARD 用的 540928 完全一致，互为交叉验证。

### 2.2 每请求占多少块（【实测】）

```
full 组（4×long_kv + 4×indexer.k_cache，8 层）   → 1024 块
state 组（3×compressor state ring）              → 1 块
swa 组 × 10（每组 4 层，一列一层）                → 130 块 / 组 ⇒ 1300 块
                                             合计 2325 块
num_blocks = 5 GiB ÷ 540928              = 9924 块
max_concurrency = 9924 ÷ 2325            = 4.27
GPU KV cache size = 4.27 × 1048576       = 4,475,277 token   ← 真机逐字一致
```

### 2.3 复算器与真机的一致性自检

| 配置 | 复算器 | 真机日志 | 一致 |
|---|---:|---:|---|
| DCP1, BAT=8192 | 1,096,072 | 1,096,072 | ✅ |
| DCP8, BAT=8192 | 4,475,277 | 4,475,277 | ✅ |

⇒ 这个模型可以当**秒级设计工具**用（真机起一次 ~12 分钟）。

---

## 3. ★ 中途被推翻的假设：滑窗**不能**分片，必须复制（【实测】×2 条独立证据线）

### 3.1 我最初的错误判断

我逐行读 `_compute_slot_mapping_kernel` 的 DCP 分支后，曾判断滑窗可以分片：

```
owner(pos) = (pos // I) % dcp
slot(pos)  = block_table[pos // (block_size*dcp)] * block_size + pos % block_size
```

推理是「物理 block 仍只装 `block_size` 个 token ⇒ 单块内存不涨；
各 rank 算窗口内不重叠的一段，LSE 合并后等于全局」。**这个推理只对了一半**：
物理布局确实如此，但**算子无法表达"看哪些 key"**。

### 3.2 证据线 A：算子表达能力（子代理 `ori_sparse_probe`，【实测】）

设备的 `ori`（未压缩）路径在 A3 上被**硬绑**：

| 参数 | 约束 | 证据 |
|---|---|---|
| `ori_sparse_indices`（显式可见键集合） | **A5-only** | `ori_sparse_indices is only supported on A5` `sparse_flash_mla_tiling.cpp:1246`（`[2,1,128]`/`[2,1,32]`/`[2,1,28]`/`[2,128]` 四种形状、带/不带 `-1` 全部同一错误） |
| `ori_mask_mode` | 必须 4 | metadata 报 EZ0024 |
| `ori_win_left` | 必须 127（band 也不能收窄） | metadata EZ0027；attention 也 tiling 拒绝 |
| `ori_topk_length` | A2/A3 保留参数，非空即拒 | `tiling.cpp:854/859` |

**替代路线也被堵死**：cmp 路径带显式索引在 A3 上只接受 `cmp_ratio=4`
（`cmpRatio should be 4 ... when cmp_sparse_indices is provided`, line 1389），
且实测**索引被静默忽略**（给 8/16 个有效索引，输出与"仅 ori 窗口"逐位一致，
`max|d_lse|=4.77e-7`，而与"并集"参考差 0.12）；ratio=4 下索引还是 4:1 压缩 token 号，
即使生效也表达不了 1:1 的窗口。

### 3.3 证据线 B：数值可表达性（子代理 `dcp_equiv_harness`，【实测】）

用**真算子 + rank 私有 pool + 真 PA block table** 做 14 组分片用例：

| 场景 | 结果 |
|---|---|
| 窗口 `[p-127, p]` 完全落在拥有该 query 的 rank 内 | 11/14 用例 `relout=0.0`、`Δlse=0.0` ✅ |
| `p ≡ 127 (mod 128)`（窗口恰为一个完整 128-chunk） | 精确 ✅ |
| **反例**：`T=129/130/201`、`dcp=2` | `relout` **8.4% / 12.4% / 62%**，`Δlse` 7.9e-3 / 1.6e-2 / 0.45 ❌ |

规则 14/14 预测=实测：**精确 ⟺ 窗口不跨 rank，或窗左沿块对齐**。
负控有效（不跳过越窗 rank → `relout` 0.63~0.98）。

### 3.4 结论与处置

⇒ A3 上滑窗**必须整份复制**：`SlidingWindowSpec` 的 manager/block table
按 **DCP=1 语义**寻址（`patch_v41_dcp.py::resolve_group_dcp` 把非 full 组的
`dcp_world_size` 归 1；`worker/block_table.py` 关掉 rank 过滤）。

### 3.5 ★ 附带的重要发现：**复制滑窗不贵，反而最优**

滑窗是「最近 128 token + 在飞 token」的**滚动窗口**，成本与序列长度**无关**：

```
swa_cap = cdiv(window-1 + in_flight, block_size) + 1     # 与 dcp 无关
```

| 方案 | 1M 上下文下每 SWA 组（block_size=128） |
|---|---:|
| **复制**（滚动窗口） | `cdiv(127+2048,128)+1` = **18 块** |
| 分片（若可行） | `L/dcp/128` = 1024 块 + halo ⇒ **1024~2048 块** |

⇒ 分片看起来省内存，实际因为"窗口是固定大小、分片却要存满整个 1/dcp 序列"
而贵 8~16 倍。**复制既唯一可行，也长期最优。**

---

## 4. 真正的容量杠杆：`in_flight = max_concurrent_batches × BAT`

复制态 SWA 的 cap 只看**在飞 token 数**：

```
in_flight = max_concurrent_batches × max_num_batched_tokens
request_blocks = full_blocks(dcp) + state(1) + 10 × (cdiv(127 + in_flight, 128) + 1)
full_blocks(dcp) = 8192 / dcp          # 8 个 full 平面 @1M、block_size=128
```

`max_concurrent_batches`（`vllm/config/vllm.py:540`）在 `pp=1` 时：

| async_scheduling | max_concurrent_batches |
|---|---:|
| 开（默认） | **2** |
| 关（`--no-async-scheduling`） | 1 |

所以合法的容量杠杆是 **关 async 和/或调小 BAT**，而不是去改 SWA 的 cap 公式。

### 效果（解析模型，已被两次真机测量逐位验证）

| DCP | BAT | async | in_flight | SWA cap | request_blocks | KV tokens | vs DCP1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 开 | 16384 | 130 | 9493 | **1,096,072** ← 真机一致 ✅ | 1.00× |
| **8** | 8192 | 开 | 16384 | 130 | 2325 | **4,475,277** ← 真机一致 ✅ | 4.08× |
| 8 | 8192 | 关 | 8192 | 66 | 1685 | 6,175,085 | 5.63× |
| 8 | 4096 | 关 | 4096 | 34 | 1365 | 7,622,725 | 6.96× |
| **8** | **2048** | **关** | **2048** | **18** | **1205** | **8,634,871** | **7.88×** |
| 8 | 1024 | 关 | 1024 | 10 | 1125 | 9,248,906 | 8.44× |

复算工具：`a2sim-ref/v41_capacity_sweep.py`（纯解析、秒级、无需容器）。

### 安全性

没有改任何 cap 公式 ⇒ 上游「startup pool sizing 与 runtime admission cap 必须同源」
的约定完好无损（两者都调用同一个未被修改的
`SlidingWindowSpec.max_admission_blocks_per_request`）。
`in_flight` 只是通过公开配置项调小，`max_in_flight_tokens` 的语义与上游一致。

### 代价（需求方需知情）

关 async scheduling 会牺牲一部分 GPU 利用率（上游注释：async scheduling
"helps to avoid gaps in GPU utilization, leading to better latency and throughput"）；
把 BAT 从 8192 调到 2048 会降低单步 prefill 吞吐。二者都是**吞吐换容量**的显式取舍，
不是免费的。7.88× 是 (BAT=2048, sync) 下的数字。

> 对照：若滑窗可分片，同样参数下 `request_blocks` 会是 1065（SWA cap=3），
> 容量更高。但 §3 已证明 A3 上不可行 ⇒ 该数字**不可交付**，仅作理论上限记录。
> 真机实测 `request_blocks=1205` 与**复制**模型一致（分片模型会给 1065），
> 这是"复制已生效"的直接证据。

---

## 4.1 ★ 压缩平面（cmp）的正确性前提：未解决，且**不能**靠复制解决

长上下文稀疏注意力分两步：

1. **indexer 选 top-k**（512 个压缩 token）；
2. **`npu_sparse_flash_mla` 用 `cmp_sparse_indices` 对这些键做稀疏注意力**。

A3 上 `ori_sparse_indices` 是 A5-only（§3.2），所以滑窗只能复制；
而 **cmp 路径的索引通道是可用的** ⇒ 只要 top-k 集合正确、索引能重映射到本 rank
的本地坐标，长上下文路径就能精确 LSE 合并。映射推导见
`attention/context_parallel/v41_dcp.py` 的模块 docstring，三条不变量已离线逐位验证：

| 不变量 | 验证 |
|---|---|
| 写侧 `compressed_slot_mapping` 与读侧 `compressed_local` 互为同一映射 | 20000 项 0 mismatch |
| rank 拥有的压缩 token 的块列 == 超块号 `g // (dcp·B')` | 20000 项 0 mismatch |
| 每 (rank, 超块) 恰好 `B'`=64 个 token、本地列空间连续无空洞 | N=204800 零违规 |

### 障碍：全局 top-k 需要每个 rank 看到全量 indexer K

top-k 必须在 8 个 rank 上**逐位一致**，否则各 rank 的 partial 覆盖不同键集，
LSE 合并出的不是全局 softmax。A3 上没有分布式 top-k 通道，SFA 的做法是
**把 indexer K cache 物理复制 dcp 份**（零通信）。

### 实测：复制路线暂时不可交付（**两个**障碍）

真机 run `dcpcap_0929_132019`（打开 `V41_DCP_REPLICATE_INDEXER=1`）：

```
[V41-DCP-DIAG] pool_bytes_per_block=660480 (slots=[132096,132096,132096,264192])
[V41-DCP-DIAG] request_blocks(total)=1205
ValueError: Aurora circular state must fill its slot with 32 contiguous FP32 rows
```

| # | 障碍 | 说明 |
|---|---|---|
| 1 | **容量 −22%** | pool 540928 → **660480**，容量 7.88× → ~6.45×。机制：index 面 ×8 ⇒ slot 0/1/2 的 `kv+index` 73856→132096（超过 SWA 别名 131072），slot 3 154752→264192 |
| 2 | **结构性**：state ring 必须等长连续 | `reshape_cache` 断言 state 必须填满槽位，且 `compressor_triton.py:665` 要求 `state_cache.is_contiguous()`；槽位涨到 132096 后 32×1024×4=131072 填不满 ⇒ 直接报错。**要让复制可用，必须先改 ring 内核的 stride 假设或把 state 迁出该槽位** |

### 正确且不涨内存的替代路线：分布式 top-k（下一步）

indexer 保持分片，各 rank 在自己分片上算分数、取**本地 top-k（k = 全局 k = 512）**，
all-gather `8×[T,512]` 的 (index, score)，本地归并取全局 top-512，再走已实现的 remap。

* **正确性**：全局 top-512 的元素在其本 rank 内排名必 ≤512 ⇒ 本地保留 512 足够（标准结论）。
* **代价**：每 indexer 层每步 `T×8×512×8B = T×32KB`；8 层 ⇒ `T×256KB`/步。
  T=128（decode）→ 32 MB/步；T=8192（prefill）→ 2 GB/chunk，prefill 需另想办法
  （可复用现有的 `candidate_topk_blocks` 两段式筛选先把候选压到 2048 块）。

⇒ **当前可交付口径仍是 7.88× 容量 + 滑窗复制已通**；cmp 路径的正确性尚未接入，
因此 DCP8 服务的输出**仍不正确**（与 §6「还没做的」一致）。

---

## 5. 开发基础设施（可复用）

### 5.1 overlay 挂载：改文件 → 重启服务

`scripts/serve_a2.sh` 新增 `V41_DCP_MOUNT=<dir>`：把一棵「容器路径布局」的目录
整树 `-v` 进去，免去每次改一行就重打镜像。

**三重保险**（都是被坑出来的）：

1. 起服前打印挂载清单；
2. 起服后**逐文件 md5 比对**容器内 vs 宿主，不一致直接 die
   （`-v SRC:DST` 在 DST 写错时会静默变成目录，服务照样起）；
3. 与 `patches/files/*` 的**重复目的地**自动去重（overlay 优先）—— docker 对重复
   目的地直接 `Duplicate mount point` 拒绝起容器；且去重必须让 overlay **最后追加**，
   否则会把 overlay 自己删掉。

### 5.2 容量复算器

```
docker exec -i <container> sh -c "cat > /root/v41_pool_sim.py" < a2sim-ref/v41_pool_sim.py
docker exec <container> python /root/v41_pool_sim.py --sweep [--dcp-aware-cap]
```

### 5.3 起服脚本

- `a2sim-ref/dcp_stage_capacity.sh`：自动挑 8 张空闲设备 + 起 DCP8 + 抓容量行。
- `a2sim-ref/dcp_sync.sh`：本地 overlay → `a3-21:~/dcpw`，**镜像式**同步
  （远端多余文件会删掉，避免旧补丁残留叠加）。

---

## 5.2 输出合并：数学已验证，落地受内核表达能力限制（【实测】）

### 5.2.0 ★ 关键实现 bug：压缩平面的可见长度必须是**本 rank 的本地长度**

**症状**（8-chip 真权重，DCP8，算子零报错但输出乱）：
```
"计算 17*23 的值，只输出数字。" → '根据您提供的文本内容，我无法确定您。如果您想了解关于"的英文表达…'
"请原样重复这句话：山高路远坑深。" → '很抱歉，libcurl 的英文怎么说？"山高"这个词在中文里是什么意思？…'
"1+1等于几？只回答数字。" → '1+1'
```

**根因**：builder 把 `coordinates["cache_seq_lens"] = seq_lens`（**全局**长度）
传给了算子。而本 rank 的 `long_kv` / `index_k` 物理块只装 **1/dcp** 的序列 ⇒
算子会去读本 rank 块表里**不存在的行** ⇒ 读到 null/邻块 ⇒ 返回一个**有限的**
LSE ⇒ 在 `Σ e^{L_r}·O_r / Σ e^{L_r}` 里按错误权重参与合并。

**极端且已复现的情形**：prompt 只有 **17 个 token**（`I=32, dcp=8`）时
`owner(pos) = (pos//32) % 8` ⇒ **只有 rank 0 拥有 KV**，rank 1-7 的本地长度是 **0**。
全局长度会告诉 rank 1-7「你有 8 行」，于是它们读空块、报有限 LSE、把结果搞乱。

**修复**：新增 `v41_dcp.local_compressed_len()`，与写侧 `compressed_slot_mapping`
**完全同源**的推导：

```
本 rank 拥有的压缩 token = { g : owner(g·ratio + ratio - 1) == rank }
owner(pos) = (pos // I) % dcp  ⇒  以 Is = I/ratio 为块、块号 % dcp 即归属
```

已用 **600+ 个长度 × 8 个 rank** 与「逐 g 模拟写侧」对拍，**零 mismatch**。

| L | 每 rank 本地压缩行数 |
|---:|---|
| 17 | `[8, 0, 0, 0, 0, 0, 0, 0]` ← 只有 rank 0 |
| 4096 | `[256, 256, 256, 256, 256, 256, 256, 256]` |

**推论**：对**短 prompt**（`L < I·dcp = 256`）只有 rank 0 有 KV ⇒ 全局 top-k
退化为 rank 0 的本地 top-k ⇒ 这一版应当就能给出正确输出。这是检验该修复的
最灵敏用例。

合并式 `O = Σ_r w_r·O_r / Σ_r w_r, w_r = exp(L_r)`：只要各 rank 的可见键集
**互不重叠地覆盖全局集**，结果就精确等于全局 softmax。子代理用**真算子**做了
离线验证（`/home/chiro/tmp/dcp_merge_probe/`、`/home/chiro/tmp/ori_rank0_probe/`）：

| 验证 | 结果 |
|---|---|
| 8 rank 分段合并 vs 全局 fp64 参考 | `Δlse=1.8e-7`、`relout=2.94e-3`（= bf16 输出舍入地板 2.93e-3，**合并没有引入额外误差**） |
| 负控：8 rank **都**携带共享窗口 | `relout=1.192`、`Δlse=1.085`（差 3 个数量级，判据有效） |
| sink 只计一次（只 rank0 传真值，其余 -1e4） | `Δlse=2.8e-7` |
| 负控：所有 rank 都传真 sink | `relout=0.056`、`Δlse=0.055` |

### ★ 落地阻碍：A3 无法表达"某 rank 只带 cmp、不带 ori"

真机实测（run `dcpcap_0929_133659`）：
```
RuntimeError: call aclnnSparseFlashMla failed, detail:[PID: 2142] ... Invalid_Argument(EZ0037):
Parameter ori_kv of SparseFlashMla is invalid. Reason: The tensor of ori_kv is nullptr.
```

子代理进一步定位（设备 9 单卡短跑）：

1. `has_ori_kv=False` 只在 **metadata 层**可用；**op 层 `ori_kv` 是硬必填**
   （`sparse_flash_mla_tiling.cpp:270`）。
2. 更硬的约束：**镜像内核只编了 SWA / CSA 两个模板**。
   「ori + cmp 但无 cmp_sparse_indices」= **HCA 模板，A3 没有内核**
   （`Cannot find tilingKey[578] in kernel json file`）。

⇒ 已把默认方案从 `rank0`（摘掉 ori_kv）改成 **`seqused0`**：
所有 rank 都挂 `ori_kv`/`ori_block_table`（保持与生产路径**同一个 tiling key**，
即已验证可用的 CSA 模板），只把 rank>0 的 `seqused_ori_kv` 设 0，
同时 metadata 仍声明 `has_ori_kv=True`。

`V41_DCP_ORI_OWNER` 可在 `seqused0`（默认）/ `rank0` 之间切换以便真机二分。

---

## 6. 还没做的（下一步）

1. **attention 执行路径**：`dsa_v41.py` 的 DCP impl/builder —— Q all-gather、
   LSE 打开（`return_softmax_lse`）、输出 all-to-all 合并、top-k 的分布式化。
2. **压缩域 top-k**：`compress_ratio=2` 的压缩槽与 rank 边界对齐（推导已闭合：
   强制 `I = block_size` 后 `g // (I/ratio)` 即组号，remap 用 `I//ratio`）。
3. **正确性**：144K/1M 针 + 短问答对照 DCP1。
4. **性能**：`(ms/step, A, tok/s)` 三元组，DCP1 vs DCP8。
5. **`cp_kv_cache_interleave_size`**：必须在**每个进程**（含 EngineCore）强制成
   `block_size`。v1 把它写在 worker 的 `validate_cache_runtime`，
   结果 EngineCore 侧 DIAG 打出 `interleave=1` —— 两个进程各有一份 VllmConfig。
   现已移到 `platform.py::_validate_parallel_config`。

---

## 7. 相关文档

| 文档 | 内容 |
|---|---|
| `docs/DSA-DCP-FEASIBILITY-20260929.md` | 可行性调研：V3.2(SFA) 与 V4.1(DSA) 的代码级分界线是 `compress_ratios` |
| `docs/DCP-OPERATOR-INVENTORY-20260929.md` | 算子级清单：SFA 的全局 top-k 靠 **indexer cache 物理复制**，不是分布式 top-k |
| `docs/SFA-DCP-PORTING-MANUAL-20260929.md` | 照抄手册：双视图 metadata、地址公式、LSE 合并、与 V4.1 的差距清单 |
