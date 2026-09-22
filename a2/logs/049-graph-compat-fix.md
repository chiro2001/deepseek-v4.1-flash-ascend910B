# 049 — int8 KV8 读侧的图兼容性修复（档 C / 档 D 在 `FULL_DECODE_ONLY` 下起服）

> 子代理 **S_graphfix**，卡时 **c0（Phy-ID 8–15）**（自带 `flock` 排队，不踢锁）+ **不占卡** 的
> CPU 容器自检。起点 = `R_8card_int8` 的 8 卡真权重实测（`dsa_v41.py:436` 的 `.item()` 在捕获期炸
> `Not_Supported(EE1016)`）。
>
> ★ 结论标签：【实测】= 有判别力的判据跑出来的数；【推断】= 静态链条；【未确认】= 没跑。

---

## §0 结论摘要（先读这三条）

### 0.1 ★★★ **只修 436 会把档 D 从「起不来」变成「起来了但静默算错」**

档 D（`VLLM_V41_KV8_PREFILL=1`）的 long-KV INT8 面还有**第二条独立的图缺陷**，它与 436 不在同一行、
不是同一个量、也不在同一个函数：

```
model_runner_v1.py:3735-3737   capture 期 _dummy_run 把 optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                               （seq_lens = max_query_len = uniform_decode_query_len = 1+num_spec = 6）
  → model_runner_v1.py:3286    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]（不是 None，
                               因为 use_async_spec_decode = use_async_scheduling and num_spec>0；
                               本配置没开 async scheduling ⇒ 这条 fallback 不生效）
  → model_runner_v1.py:3338    AscendCommonAttentionMetadata(seq_lens_cpu=seq_lens_cpu, ...)
  → dsa_v41._build_batch_metadata   max_cache_seq_len = int(seq_lens_cpu[:num_actual_reqs].max()) = 6
  → dsa_v41 尾部包装 _kv8_cmp_plane_prefill:  ppr = max(1, ceil(mcs/block_size)) = ceil(6/128) = 1
  → kv8_prefill_triton.fused_cmp_plane3(ppr=1)  scratch = num_reqs 页、table 形状 (num_reqs, 1)
```

而 replay 时算子按**真实压缩块索引**去读 `cmp_block_table[b, block]`，block 可达**几百**
（131072/128 = 1024）⇒ **表只有 1 列**。

⇒ 这是**捕获期冻结的"值"**（`max_cache_seq_len`）而不是**上界**，正是本任务禁止的形态。
★ **它怎么死已经定案（§3.2 源码级 + §5.5.3 运行期）**：算子对 block table 的列宽**按构造没有**
"列宽 ≥ 最大块号"的检查（host checker 只查 dtype/维度/非空/dim0==batch），device kernel 直接按
`bIdx * 列宽 + blkTableIdx` 取址 ⇒ 列宽 1 时可寻址到几百。
★★ **运行期实测结果是"响亮地崩"而不是"静默算错"**（更正我在第 1 报里的说法）：
`sg-c-d-cmplegacy` 臂在**第一个真实请求**上打 `SUSPECT REMOTE ERROR, error code 507057`
（`rtEventSynchronize ... suspect remote error`）⇒ 引擎死。
机制上仍然是"越界读表拿到垃圾页号"，只是那个垃圾页号再乘 `cmpKvStride0` 落到了未映射地址 ⇒
**设备故障**。★ 但**这不保证总是响亮的**：若垃圾页号恰好落在已映射内存里，同一缺陷就会**静默算错**
（这也是我把它按"必须修"处理、而不是"反正会崩"处理的原因）。

★ **限定（同样重要）**：**窗口（SWA）面没有这个问题** —— `fused_ori_plane2` 的 `ppr` 来自
`max_q_len=query_rows`（一个 **shape**），是 capture/replay 都安全的粗上界。**只有 cmp 面坏。**

**⇒ 本任务的补丁把两条一起修**（同一个 `dsa_v41.py`、同一个 env 门、同一次挂载）。

### 0.2 436 的判据原意，以及它在 spec-decode 下为什么失效

（见 §1。）一句话：`query_rows == num_reqs` 同时承担了**两件事** —— ①"这批是 decode 形状"；
②"页数这个 host 标量从哪来"。spec-decode 下 decode 批是 `num_reqs × 6` 行 ⇒ 判据为假 ⇒
误入 prefill 分支 ⇒ 捕获期 D2H ⇒ `EE1016`。**它坏的不是算法，是"用形状当语义"。**

### 0.3 修法（最小面 + env 门控 + 默认关）

新增**上界分支**（不改旧两支一个字节），`rows_bound` 由 host 侧
`metadata.swa.max_query_len`（不产生 D2H）给出；档 D 的 cmp 面改成
**按选择重建**（单行快路径的直系推广）。详见 §2。**全部走 `VLLM_V41_KV8_GRAPH_SAFE=1`，
默认 0 = 逐字旧行为。**

---

## §1 分支判据的原意，与 spec-decode 下的失效

### 1.1 原意

`kv8_ori_plane` 重建的是 `npu_sparse_flash_mla` 在窗口（`ori_mask_mode=4`）下**真正会读的那些行**：
对每个请求、每条 query 行 `i`，带 = `[seq_len - q_len + i - window + 1, seq_len - q_len + i]`；
对整批取并集（= 文档字符串里的 `[seq_len - min(seq_len, q_len + window), seq_len - 1]`）。

* **每请求恰好 1 行 query**（decode）⇒ 带就是 `[L-128, L-1]`，**最多跨 2 页** ⇒ 页数写成常量 2，
  整条链纯 device 运算（capture-safe）；
* **否则**（chunked prefill：每请求 `q_len` 不同）⇒ 只能算
  `pages_per_req = max_b(ceil((L_b - W_b)/bs))`，**而这个 max 的形状取决于运行期数据** ⇒
  作者用 `.item()` 把它取到 host，并明确注释 `# Prefill: ... Eager only, hence the host syncs`。

所以判据的**原意**是"decode 批 ⟺ 每请求 1 行 query"，因为它把②（页数从哪来）也一起决定了。

### 1.2 失效机制

spec-decode 的 decode 批每请求带 `1 + num_spec_tokens` 行（本配置 `=6`）⇒
`query_rows = num_reqs × 6 ≠ num_reqs` ⇒ 走 else ⇒ `.item()` ⇒
捕获期同步 ⇒ `Not_Supported(EE1016) stream_id=31`（8 rank 逐字一致，栈在 `dsa_v41.py:436`）。

★ **档 B（纯 BF16）同链同图同 spec 配置捕获成功**，只是因为 BF16 的 SWA 面根本不进这个函数 ——
这也反向证明缺陷**只在 int8 读侧**。

### 1.3 关键区分：什么必须来自 host，什么不必

| 量 | 谁能提供 | 结论 |
|---|---|---|
| 窗口带**起点** `window_start` | device（`query_start_loc` 差分 → `span=min(lens,q_len+window)`） | **不必 host** |
| 块表映射（逻辑块→物理页） | device（`torch.gather` + 未读列 clamp 到 page 0） | 已经是 capture-safe |
| scratch **页数** `pages_per_req` | 必须 host（它决定 `arange` 宽度与分配） | ★ **但不必是"精确最大值"，只要一个可证上界** |
| 上界本身 | host：`metadata.swa.max_query_len`（引擎在 CPU 张量上算好的 python int）或 `query_rows`（shape） | 二者都**不产生 D2H** |

---

## §2 修法（9+1 个锚点，全部 exact-match 断言）

产出：`a2/agents/S_graphfix/patch/apply_graphsafe.py`（对 `X_integrate/pkg-kv8pf` 的
`attention/dsa_v41.py` 施加；原件 md5 `75f4e565adc1b12c854a0a01271b6c4d`）。

### 2.1 窗口面（档 C/D 共用）

在**旧两支之前**插入：

```python
if rows_bound is not None:
    q_len = (query_start_loc[1:num_reqs+1] - query_start_loc[:num_reqs]).to(torch.int64)  # device
    span = torch.minimum(lens, q_len + window)                                            # device
    window_start = lens - span                                                            # device
    pages_per_req = min(width, max(1, (int(rows_bound) + window - 1) // block_size + 2))   # host 上界
elif query_rows == num_reqs:   # ← 旧 decode 支，逐字未动
    ...
else:                          # ← 旧 prefill 支（.item()），逐字未动
    ...
```

**上界推导（把推导留在这里，便于复核）**：设某请求的窗口带跨 `s` 行、块大小 `B`，
带起点在页内偏移 `o = W mod B`（`W = window_start`）。跨页数
`P = floor((L-1)/B) - floor(W/B) + 1`。

* `s ≤ rows_bound + window ≤ B·k + o'`（`k = (rows_bound+window-1)//B`，`o' ∈ [0,B-1]`）；
* 极端对齐（`o=B-1` 且 `s` 刚好跨最多页）给出 `P ≤ k + 1`；
* 本式给出 `k + 2`，**严格 ≥ `k+1`** ⇒ 对任意 `o` 都够（多出的 1 页是恒等冗余）。

⇒ **这正是 C2 变异（把 `+2` 删成 `+0`）会被自检抓住的原因**（见 §4）。多出的页是惰性的：
块表只会指向掩码真正读的块，算子不会读带外的行。

★ **捕获期也必须走这条分支**：`_dummy_run` 的 dummy 批 `is_prefilling` 来自它并不拥有的请求账本，
**不能**当分类器用（那正是 436 炸掉的地方）。捕获期固定下来的 `rows_bound` = 该图的
`uniform_decode_query_len`（=1+spec），**就是这张图 replay 时每请求的行数** ⇒ 常量在图内**语义正确**。
真 prefill（eager、`num_prefills>0` 且**不在捕获中**）仍然走旧支 + 它自己的融合 kernel。

### 2.2 long-KV 面（档 D 专有）

`_kv8_cmp_plane` 增加 `graph_safe=` 分支：给**每条 query 行一段私有 scratch**。
第 `i` 行第 `t` 个选择 → 合成下标 `i·topk + t` ⇒ scratch 页 `i·per_req + t//block_size`、
页内偏移 `t % block_size`（`per_req·block_size == topk` 时**精确**）；scratch 表 = `rows·per_req` 页恒等表。

* 行→请求映射：`torch.index_select(block_table[:num_reqs], 0, arange(rows)//reps)`（`reps=rows//num_reqs`），
  这是 device 的**静态形状** op，无 D2H。★ **不要用 `repeat_interleave`** —— 它的 aclnn 实现
  在本镜像上 segfault（§4.5 有第一现场）；
* 所有标量来自 shape/config ⇒ capture 与 replay 都安全；
* 旧 fallback（`cache_seq_lens.max().item()`）与档 D 的 Triton 路**逐字保留**，只是新路径不再走它们。

### 2.3 门控与诊断开关

| env | 默认 | 作用 |
|---|---|---|
| `VLLM_V41_KV8_GRAPH_SAFE` | `0` | 主开关。0 = 旧行为逐字不变 |
| `SG_CMP_LEGACY` | `0` | **诊断臂**：只修窗口面、把 cmp 面推回捕获期路径 ⇒ 实测 §3 的 (b)/(c) |
| `SG_TRACE_PPR` | `0` | 只读探针：打印捕获期/replay 的 `mcs` 与 `ppr`（热路径、限 24 行） |

---

## §3 436 与"档 D 静默读错"的严重性差在哪（含已经跑出来的那一格）

### 3.1 已经【实测】的部分：**torch/ATen 层的越界是响亮的**

不占卡（CPU 容器、同一镜像）跑 `scripts/oob_probe.py`，原始输出 `raw/049-oob-probe.log`：

```
T1 torch.gather(table(4,1), dim=1, index(4,512))     → RuntimeError: index 1 is out of bounds for dimension 1 with size 1
T2 table[rows, cols]（高级索引）                      → IndexError: index 1 is out of bounds for dimension 1 with size 1
T3 index_select(1 列的表, 512 个下标)                 → IndexError: index out of range in self
T4 对照：表比需要的大（下标取模到合法）                → 静默返回（值合法）
T5 越界读数据平面 plane[8/9/100]                      → IndexError: index 8 is out of bounds for dimension 0 with size 8
T6 越界写 copy_（源更大）                             → RuntimeError: size mismatch
```

⇒ **如果越界发生在 torch 层，答案就是 (b) 响亮失败**（比 (c) 好得多）。

### 3.2 ★★★ 【实测·源码级】算子层：答案是 **(c) 静默读错** —— 界检查**按构造就不存在**

档 D 真正走的是 `npu_sparse_flash_mla(..., cmp_block_table=(num_reqs,1), ...)` —— 表是**直接喂给
Ascend kernel 的 GM buffer**。**镜像里带着算子自己的实现**，所以这一格可以**完全不占卡**地读源码定案
（证据文件 `raw/049-op-bounds-evidence.txt`，采集脚本 `scripts/op_bounds_evidence.sh`）：

```
① host 侧 checker（op_host/checkers/paged_attention_checker.cpp:23-40）
   CheckBlockTable 只查四件事：dtype==int32 / 维度数==2 / 每维非空 / dim0==batch size（:78-100）
   ★ 没有任何一处把「列宽」与「kernel 会寻址到的最大块号」做比较
   （唯一的列宽检查是通用的「第二维 > 0」，见 tiling .so 里的字符串
     "block_table's second dim should be greater than 0."）
   ⇒ 列宽 = 1 的表 **通过全部 host 校验**

② tiling（op_host/sparse_flash_mla_tiling.cpp:896）
   cmpMaxBlockNumPerBatch_ = cmpBlockTable.tensor->GetStorageShape().GetDim(1)   ← 就是列宽
   tiling.cpp:914  cmpS2Size_ = cmpMaxBlockNumPerBatch_ * cmpBlockSize_
   tiling.cpp:1095 smlaInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_

③ device kernel（op_kernel/arch22/sparse_flash_mla_csa_block_vector.h:544-552）
   int64_t blkTableIdx = realS2Idx / constInfo.paCmpBlockSize;
   realKeyGmOffset =
       cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx) * ...
   ⇒ 列宽 = 1 时，偏移 = bIdx + blkTableIdx（blkTableIdx 可达几百）⇒ **直接越界读 GM，无任何界检查**

④ 唯一的界检查在 :541-543，检的是**位置**（realS2Idx vs s2IdLimit，来自 seqused_cmp_kv），
   **不是**表索引 blkTableIdx
```

⇒ **答案 (c)**：越界读同进程 GM（通常不触发 MMU 故障）⇒ 把垃圾值当**页号**去取 KV 行 ⇒
**静默算错**；只有当「垃圾页号 × `cmpKvStride0`」恰好落到未映射页时才会退化成 (a) 崩。
**(b) 被排除** —— 不是"没检出来"，是**按构造没有这一项检查**。

★ 结论的强度：这是**源码级证明**（静态、可复核、可重复），我标【实测·源码级】；
运行期确认（"它到底跑出什么"）仍由决策臂 `sg-a-d-cmplegacy` 给出（§5.4）。

★ **无论 (a)/(b)/(c)，本补丁都消灭这个情形**（新分支的页数/表宽由 shape 决定，不由捕获期的值决定）。

---

## §4 离线自检（不占卡）：真函数 + 穷举 + 变异阳性对照

做法：`ast` 从**目标 `dsa_v41.py` 里抽出真函数**（`kv8_ori_plane` / `kv8_gather_rows` /
`kv8_dequant_rows` / `kv8_scratch_plane` / `_kv8_cmp_plane` / `_kv8_graph_rows_bound` …），
在只暴露 torch 的命名空间里 `exec`，按**算子真正会读的地址**逐位置比对。原始输出
`raw/049-offline-check.log`（命令见 §6）。

### 4.1 A 段：旧路径逐比特不变

```
decode 支（rows==num_reqs）  逐比特相同 = True / 旧调用点（不传 rows_bound）兼容 = True
prefill 支（多行非均匀）     逐比特相同 = True / 旧调用点（不传 rows_bound）兼容 = True
```

### 4.2 B 段：新路径**穷举**（不是抽样）

```
L<window / =window / +1 / 大   bs=128 bound=1: 位置 513 坏 0 越界 0
L<window / =window / +1 / 大   bs=128 bound=6: 位置 518 坏 0 越界 0
L=block 整倍/非整倍            bs=128 bound=1: 位置 773 坏 0 越界 0
L=block 整倍/非整倍            bs=128 bound=6: 位置 798 坏 0 越界 0
block_size=64（非 128 适配）   bs=64  bound=1: 位置 387 坏 0 越界 0
block_size=64                 bs=64  bound=6: 位置 397 坏 0 越界 0
window=512                    bs=128 bound=1: 位置 1538 坏 0 越界 0
window=512                    bs=128 bound=6: 位置 1548 坏 0 越界 0
cmp 图安全 rows=1*num_reqs                 : 选择 1792  坏 0
cmp 图安全 rows=6*num_reqs                 : 选择 10752 坏 0
cmp legacy fallback（graph_safe=False）    : 选择 10752 坏 0
```

### 4.3 C 段：阳性对照（每条都必须报警）

```
C1 上界给成 -4096（严重不足）        : 坏 384 + 越界 128   ✅报警
C2 变异：上界公式删掉对齐冗余项(+2→0) : 坏 384 + 越界 128   ✅报警   ← 证明 §2.1 的 +2 是判别量
C3 变异：带起点退回 lens-window       : 坏 24              ✅报警
C4 变异：cmp 行→请求用 repeat 而非 repeat_interleave: 坏 8064 ✅报警
C5 反例：cmp 行分布非均匀             : 坏 8064            ✅报警
```

### 4.4 D 段：门控与 batch 路由

```
env 未设 ⇒ legacy（False,None）
eager prefill 批（非捕获, num_prefills=3）⇒ legacy（False,None）
★ 捕获期 dummy（num_prefills=3, max_query_len=6）⇒ 新分支（True, 6）   ← 见 §2.1 的星号
decode 批 + max_query_len=6 ⇒ bound 6
缺 max_query_len ⇒ 退回 shape 上界 96
纯 decode（max_query_len=1）⇒ bound 1
```

### 4.5 ★★ 自检的**边界**：它抓不到"设备算子本身崩"（第一次档 D 图臂就这么死的）

第一版图安全分支里，我用 `block_table[...].repeat_interleave(reps, dim=0)` 做"行 → 请求"的映射。
**补丁自检、离线穷举、阳性对照全过**，但 8 卡上起服时：

```
!!!!!!! Segfault encountered !!!!!!!
  File "<unknown>", line 0, in aclnnOpInfoRecord::TilingContextToJson(...)
  File "<unknown>", line 0, in CommonOpExecutorRun(...)
  File "<unknown>", line 0, in aclnnRepeatInterleaveIntWithDim        ← ★ 就是它
(EngineCore) ERROR ... Worker proc VllmWorker-5 died unexpectedly, shutting down executor
→ 8 个 worker 同时死 ⇒ "Engine core initialization failed"
```

⇒ **CANN 的 `aclnnRepeatInterleaveIntWithDim`（int64、dim=0）在本镜像上 segfault**
（崩在它自己的 tiling-context JSON 序列化路径）。
**教训（写下来给后人）**：`ast` 抽真函数 + CPU 穷举**只能证明"逻辑对"**，
**证明不了"这个算子在设备上能用"** —— 这一格只有真机臂能给。原始证据
`raw/049-d-graph-repeatinterleave-segfault.txt`。

**修法**（同样是 device op，但换成这个文件里已经在用的原语）：
```python
b_of_row = torch.arange(rows, device=indices.device, dtype=torch.int64) // reps
table_rows = torch.index_select(block_table[:num_reqs].to(torch.int64), 0, b_of_row)
```
另外把构造恒等表的 `.repeat(num_reqs, 1)` 也换成 `.expand(...).contiguous()`（同样的防御理由）。
★ 补丁现在**断言源码里不含 `.repeat_interleave(`**（`apply_graphsafe.py` 自检第 7 条），防止回归。

**原始证据已落盘**：`raw/049-d-graph-repeatinterleave-segfault.txt`
（31 行 / md5 `ba43b897ef3c6878c5af52fb7da0b0cc`，走 `cos-xfer` 拉回；内容含 `Segfault encountered`
×2 + 完整调用栈 + `aclnnRepeatInterleaveIntWithDim` + `Worker proc ... died unexpectedly`）。

### 4.6 ★ 自检**自己**踩的两个坑（都伪装成"全错"）

1. `scratch` 是 **PA_BBND `(pages, block_size, 1, dim)`**，算子读 `[页, 页内偏移, 0, :]`；
   第一版写成 `scratch[page, off]` ⇒ 拿到 `(1,dim)` ⇒ 全错（**假阴**）；
2. 第一版 `deq()` 只算 float32，而 `kv8_dequant_rows` 最后要 **cast 到 bf16** ⇒ 差一格 ⇒ 全错。

★ 这两条正是"**C 段必须有阳性对照**"的最好论据：没有 C 段，我可能把"自检自己有 bug"当成"补丁有 bug"，
或者反过来把真缺陷放过去。

---

## §5 8 卡真权重实测（c0 = Phy-ID 8–15）

★ 状态：§5.1–§5.5.3 均为 **【实测】**；三条辅助臂（`c-eager` / `c-eager-on` / `c-cold`）**【未完成】**，
见 §7。

臂（脚本 `scripts/chain_sg.sh`，每条自带 `flock` 排队）：

| tag | 几何 | 图 | 补丁 | 用途 | 状态 |
|---|---|---|---|---|---|
| `sg-a-c-graph` | C | `FULL_DECODE_ONLY` | on | **判据①②③④⑤**（判决臂） | ✅ 完成 |
| `sg-a-d-graph` | D | 图 | on | **判据②**（+ §0 的静默读错） | 排队中 |
| `sg-a-d-cmplegacy` | D | 图 | 只修窗口面 | §3.2 的 (c) 运行期确认 | 排队中 |
| `sg-a-c-eager` | C | eager | off | 判据⑦（反例臂）+ 判据⑥ 的对照 | 排队中 |
| `sg-a-c-cold` | C（池 1 MiB） | 图 | on | 判据⑤ 的**冷算参考** | 排队中 |
| `sg-a-c-graph-b` | C | 图 | on | 判据⑧ 复跑同 sha | 排队中 |
| `sg-b-c-eager-on` / `sg-b-d-eager-on` | C / D | eager | **on** | **判据⑥ 的直接对照**（同挂载、只差开关） | 待跑 |

### 5.1 判据①（★ 核心）：档 C + `FULL_DECODE_ONLY` 起服 —— ✅【实测】（两个 md5 上都验过）

★ **两条臂、两个 md5，读数逐字相同**（这就是"档 C 在发布件 `94aeebb7` 上也成立"的证据）：

| | `sg-a-c-graph`（md5 `22cbf20c…`） | `sg-c-c-graph-b`（md5 **`94aeebb7…`**，发布件） |
|---|---|---|
| 致命证据（EE1016/Segfault/SUSPECT/engine-init） | 0 | **0** |
| 捕获 | 9/9 | **9/9** |
| `GPU KV cache size` | 427,643 | **427,643**（逐字相同） |
| `BlockStored:CPU` | 29,436 | **29,436** |
| `CPU→GPU` | 21,188,968,448 B | **21,188,968,448 B**（逐字相同） |
| `hits` | 901,120 | **901,120**（逐字相同） |
| replay / fill p50 | 1,608.2 / 19,880.0 ms（12.36×） | 1,594.8 / 19,936.0 ms（12.50×） |
| fill sha | `d524172f9f5ae368…` | **`d524172f9f5ae368…`**（逐字相同） |
| replay sha | `bc2e797ab069f09c…` | **`bc2e797ab069f09c…`**（逐字相同） |

⇒ **换版（`22cbf20c` → `94aeebb7`）对档 C 是实测 no-op**（比 §8.2 的静态证明更强的证据）。

```
EE1016 / capture failed 计数 = 0            （修复前：8 rank 同时炸，栈在 dsa_v41.py:436）
就绪用时 659 s；static_kernel 无降级（static_kernel.py:650 命中 0 次）
[SG-PPR] capturing=True  num_reqs=32 query_rows=192 num_prefills=0 max_query_len=6
                         swa_mcs=6 cmp_mcs∈{0,3,6} graph_safe=True rows_bound=6
         ★ 这一行就是 436 的击穿形状：query_rows(192) != num_reqs(32)；补丁把它路由到上界分支
```

### 5.2 判据③（容量不退化）—— ✅【实测】

```
档 C 图模式（本补丁）  GPU KV cache size = 427,643 tokens
档 C eager（R 的 r8-c2-tierC-eager） = 427,643 tokens          逐字相同
档 B（纯 BF16 图）                  = 427,643 tokens          逐字相同
★ 且容量是在**图捕获之前**定的（serve.log 行号：容量 879 < 捕获 1346+）⇒ 捕获期的 scratch 分配
  也不影响容量判据。
```

### 5.3 判据④⑤（四条判据 + 正确性）—— ✅【实测】

```
BlockStored:CPU = 29,436                     （与 R 的 L1 口径逐字相同）
GPU→CPU         = 158,559,371,264 B          （同上，同量级）
CPU→GPU         = 21,188,968,448 B = 21.19 GB   ★ >0 ✅（不是"池溢出→整段重算"的假阳性）
hits            = 901,120 / queries 3,145,984   ★ >0 ✅
BlockRemoved:CPU = 0（上线监测判据）
replay p50 = 1,608.2 ms vs fill p50 = 19,880.0 ms  ⇒ 12.36×
填充 sha = d524172f9f5ae368…（★ 与档 B **逐字相同**）
回放 sha = bc2e797ab069f09c…（★ 与 R 的 `r8-e1-tierC-eager`（eager 臂）**逐字相同**）
   ⇒ **图模式输出 == eager 输出**：这正是判据⑤要的"跨臂同名轮次比较"
池记账三条路径一致：P1 = L1③ = 302.56 GiB
```

### 5.4 判据⑨（★ 用户 2026-09-22 硬约束）：**保留投机解码**，修完后投机不得退化

```
基线（R 的臂，spec 开）：档 B 图 r8-a-tierB-graph / 档 C eager r8-e1-tierC-eager
  Mean acceptance length 1.50 | Accepted 1 tok | Drafted 10 tok
  Per-position 0.500, 0.000, 0.000, 0.000, 0.000 | Avg Draft acceptance rate 10.0%

★ 档 C 图模式 + 本补丁（sg-a-c-graph）：
  Mean acceptance length 1.50 | Accepted 1 tok | Drafted 10 tok
  Per-position 0.500, 0.000, 0.000, 0.000, 0.000 | Avg Draft acceptance rate 10.0%
  ⇒ 与两条基线**逐字相同** ⇒ 投机仍在工作、零退化【实测】
```

#### 5.4.1 ★★ 「档 D 的 1.00 vs 档 C 的 1.50」到底是不是退化？—— **明确结论：不是，属测量假象**

先看**指标的源码定义**（`vllm/v1/spec_decode/metrics.py:113-117`，容器内逐行核过）：
```python
mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)     # num_drafts = **draft 步数**
draft_acceptance_rate  = num_accepted_tokens / num_draft_tokens * 100
acceptance_rates       = np.sum(pos_matrix, axis=0) / num_drafts
```
⇒ **`Mean acceptance length` 的分子只有"接受了几件事"，分母是"发生了几次 draft"**，
且带 `+1` 的地板。代入两臂读数：

```
档 C：Accepted 1 / Drafted 10 / mean 1.50  ⇒ 1 + 1/2 = 1.50  ⇒ num_drafts = 2（2 次 draft，接受 1）
档 D：Accepted 0 / Drafted 15 / mean 1.00  ⇒ 1 + 0/N = 1.00  ⇒ ★ 只是"零接受"的地板值
档 D eager：Accepted 0 / Drafted 15 / mean 1.00  ← 与档 D 图模式**逐字相同**
```

**结论（明确表态）**：
1. **同意主代理的判断方向 —— 这两个数不可比、不构成"档 D 掉接受率"的证据**；
2. 但**机制要说准**：不是"只发生了 1 个 decode step"，而是
   **① `mean_acceptance_length` 带 `+1` 地板、分母是 draft 步数**；
   **② 两臂的差别只有"接受了 1 个 token vs 0 个 token"这一个事件**（draft 步数 2 vs 3 也不同）；
3. **更强的反证**：档 D 的 **eager** 臂给出**完全相同的 `1.00 / 0 / 15 / 0.0%`**
   ⇒ 这个读数与图模式、与本补丁**无关**；
4. ⇒ 本任务**的判据⑨ 用"图模式 vs eager 逐字相等"**（这是有判别力的），
   **不用绝对值**；"档 D 是否保投机"要看 `T_draftceiling` 的同几何 + `max_tokens≥64` 基线。
⇒ 该格标注：**【样本不足·不足以判定档 D 是否保投机】**，**不得**写成"档 D 降低接受率"。

★ **读数偏低的解释（避免误读）**：`max_tokens=1` + 16×131K 长上下文下草案本来就很难被接受，
三层读数（档 C、档 D 图、档 D eager）都在 `Accepted ≤ 1 / Drafted ≤ 15` 的量级上
⇒ 这条判据的判别力在"**与基线逐字相等**"，不在绝对数值。
★ **硬约束**：本补丁与所有推荐配置**保留 `--speculative-config`**；任何"关掉 spec 绕开问题"的写法
在本任务里**不作为推荐**（只可作诊断对照臂并显式标注）。049 §6 的复现命令里没有 `SPEC_ON=0`。

### 5.5 档 D 图模式（`sg-c-d-graph`）与决策臂 —— 进行中

档 D 的判据表（★ 含 §8.4 新增的第 0 条）：

| # | 判据 | 期望 |
|---|---|---|
| **0** ★ | **起服第一格**：`Engine core initialization failed` / `Segfault` / `Worker proc died` | 必须**全为 0**（不得记成"捕获失败/EE1016"） |
| ① | 捕获成功、`/health` OK、`EE1016=0` | ✅ |
| ② | 容量不退化 | 与档 D eager 逐个数字比较 |
| ③ | 四条判据（`BlockStored:CPU` / `CPU→GPU>0` / `hits>0` / replay≫fill） | ✅ |
| ④ ★★ | **`replay1 sha` == 同几何 eager 臂**（判据⑤ 的最强单条形态） | 逐字节相同 |
| ⑤ | 冷算参考（池 1 MiB 臂的 replay sha） | 与池臂 replay 逐字节相同 |
| ⑥ | 判据⑨ SpecDecoding 四项 | 与基线 `1.50 / 0.500 / 10.0%` 逐字相同 |
| ⑦ | `[SG-PPR] cmp_graph_safe` 的页数 | 捕获期与 replay 都**由 shape 决定**（不是 `ppr=1`） |

#### 5.5.1 档 D 图模式：**起服成功**（`94aeebb7…`）—— ✅【实测】

```
RUN_ID      = r8_sg-c-d-graph_20260922_113107（md5 = 94aeebb757d6d5708268754481a05e0a，arm.out 台账已记）
捕获        = Capturing CUDA graphs (decode, FULL): 100%|██████████| 9/9 [06:11]
判据 0      = EE1016 0 / Segfault 0 / Engine core initialization failed 0 / Worker proc died 0  ✅
就绪        = /health = 200；static_kernel 无降级
判据 ②      = GPU KV cache size 485,610 tokens  ← ★ 与 R 的档 D 臂 r8-c2-tierD-graph 的 485,610 **逐字相同**
判据 ③      = BlockStored:CPU=29,436 / CPU→GPU=12,105,678,848 B (>0) / hits=901,120 (>0) / BlockRemoved:CPU=0
              replay p50 1,458.8 ms vs fill p50 18,776.2 ms = **12.87×**
              池记账三条路径一致：P1 = L1③ = 297.18 GiB
判据 ⑦      = ★★ **捕获期的 cmp 面页数由 shape 决定，不再是 ppr=1**：
              `[SG-PPR] cmp_graph_safe capturing=True num_reqs=32 rows=192 per_req=4 segments=768`
              `[SG-PPR] cmp_graph_safe capturing=True num_reqs=32 rows=192 per_req=8 segments=1536`
              `[SG-PPR] native_attention capturing=True num_reqs=32 query_rows=192 num_prefills=0`
              `                          max_query_len=6 swa_mcs=6 cmp_mcs∈{0,3,6} rows_bound=6`
              ⇒ §0 那条 **(c) 静默读错**（表宽 1 列 + 索引几百）在捕获期**已被消灭**
```

#### 5.5.2 判据④（与同几何 eager 逐字节）—— ✅【实测】**逐字节相同**

| | 档 D **图模式**（本补丁） | 档 D **eager**（R 的 `r8-f1-tierD-eager`） |
|---|---|---|
| 几何 | 16 × 131072 → replay 65536 | **同** |
| `fill` sha | `d524172f9f5ae36806151ca2bb9d0a311bbe94fa5ad27f001be5c4ac962aa0af` | **同** |
| `replay1` sha | **`8600507eb6b43bfa16d41a24f86c898573209d8f0361e3fffe93526b29007807`** | **`8600507eb6b43bfa16d41a24f86c898573209d8f0361e3fffe93526b29007807`** |
| replay p50 | 1,458.8 ms | 1,463.8 ms |
| `GPU KV cache size` | 485,610 | 485,610 |
| SpecDecoding | ⚠️ **实测**：`Mean acceptance length 1.00 / Accepted 0 / Drafted 15 / Avg 0.0%` | **`1.00 / 0 / 15 / 0.0%`（与图模式逐字相同）** |

⇒ **档 D：图模式输出 == eager 输出（逐字节）** ⇒ 新 cmp 分支**算得对**，不是"起来了但算错"。
⇒ 判据⑨ 在档 D 上同样成立：**投机仍在工作、图与 eager 零差异**。

★ **一条必须如实标注的观察**（不是本补丁引入，但用户应知道）：本几何下档 D 的接受率读数
（1.00 / 0 / 15 / 0.0%）**低于档 C 的 1.50 / 10.0%**，而 R 的档 D **eager** 臂给出**完全相同**的
1.00 / 0 / 15 / 0.0% ⇒ **这是"档 D 的 eager 行为"，与图模式/本补丁无关**。
★ 但它**样本极小**（Drafted 仅 15 个 token、`max_tokens=1`），**不足以判定档 D 降低了接受率**；
若用户要"档 D 保投机"的证据，应另跑**专门的接受率 A/B**（不同几何、`max_tokens` 拉长）。
⇒ 本文件**不**对"档 D 的接受率"下结论，只声明"图模式 == eager"。

#### 5.5.3 决策臂 `sg-c-d-cmplegacy`（只修窗口面，cmp 面留在捕获期路径）—— ✅【实测】引擎崩

```
起服/捕获 = 成功（窗口面已修 ⇒ 不再 EE1016；cmp 面在捕获期不报错）
warmup    = ok（25.4 s）
第一个真实请求 = ★ 引擎死：
  Worker_TP*.ERROR ... synchronize: NPUEvent.cpp:215 NPU function error: SUSPECT REMOTE ERROR, error code is 507057
  EE9999: rtEventSynchronize execution failed, reason=suspect remote error
  EngineCore: Encountered a fatal error ⇒ 客户端 failed=2（fill/replay 均 500 EngineDeadError）
```

⇒ **§4 的答案（运行期）**：在**本配置**下，旧 cmp 面**不是静默算错，而是设备故障 + 引擎死**（响亮失败）。
这与 §3.2 的源码级分析一致：越界读 *表* 本身通常只读到同进程垃圾（不崩），
**是那个垃圾"页号"再乘 `cmpKvStride0` 去取 KV 时才落到未映射地址 ⇒ 507057**。
★ **但不能据此说"这个缺陷安全"**：垃圾页号落在已映射内存时就是**静默算错** ⇒
结论仍是"必须修"，只是**危险形态从『必然静默』更正为『可能崩、也可能静默』**。
★ A/B 对照臂 `sg-d-d-short-on`（**同短几何 2×8192→4096 + 补丁开**）—— ✅【实测】跑通：

| | `sg-c-d-cmplegacy`（cmp 面留旧路径） | `sg-d-d-short-on`（补丁开） |
|---|---|---|
| 几何 | 2 × 8192 → replay 4096 | **同** |
| 起服 / 捕获 | 成功 | 成功 |
| 第一个真实请求 | **引擎崩**（`507057 SUSPECT REMOTE ERROR`），`failed=2` | ✅ `failed=0` |
| `fill` / `replay` sha | **无**（未产出） | `b3eeeaba32b01e3a…` / `87a5e4fda548c29b…` |
| 进程级致命证据 | `Segfault/engine-init 0` 但 `SUSPECT REMOTE ERROR ≥1` | **0** |
| rc | 0（客户端级；引擎已死） | **0（真跑通）** |

⇒ **同几何对照成立**：唯一差别是补丁开关 ⇒ 旧 cmp 面的失败**是补丁修掉的那个缺陷**，不是几何差异。

**原始证据（已落盘，走 `cos-xfer` 拉回）**：
* `raw/049-cmp-legacy-ab.txt`（20 行，md5 `605a4c549b98a0aab0a4412433b4219c`）
* `raw/049-tierD-graph-vs-eager.txt`（41 行，md5 `2c36d80195c9bf1e00c7b77fe311ae1a`）

---

## §6 复现

```bash
# 本地 → A3（走 COS，不用 ssh 管道）
bash a2/agents/S_graphfix/scripts/upload.sh
# A3：取回 + 造影子包（只换 attention/dsa_v41.py）+ 自检对账
bash scripts/upload.sh fetch
bash scripts/mkpkg.sh
# 不占卡：补丁锚点自检 + 离线穷举/阳性对照
docker run --rm --network none -v $PWD:/sg:rw --entrypoint bash \
  quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  -lc "cd /sg && python3 scripts/offline_check.py \
        --patched pkgs/pkg-kv8pf/shadow/vllm_ascend/attention/dsa_v41.py \
        --orig ref/dsa_v41_pkgkv8pf.py"
# 8 卡（c0；自带 flock，抢不到就等）
R8_GRAPH_SAFE=1 SG_TRACE_PPR=1 bash scripts/chain_sg.sh
# 汇总
python3 scripts/summarize_sg.py
```

## §7 诚实边界

* §3 的 (b)/(c) 在跑决策臂之前是**【未确认】**（torch 层已实测是响亮失败，算子层未测）。
  ★ 已更新：§3.2 给出**源码级**答案 **(c)**；运行期确认见 §5.5。
* **已完成**：档 C 图模式（两个 md5）、档 D 图模式（含与 eager 逐字节）、档 D 同几何 A/B、
  离线自检（穷举+阳性对照）、算子层越界行为的源码级证据、发布门。
* **【未完成】**（截至本文件落盘，链仍在跑）：
  * `sg-c-c-eager`（判据⑦ 反例臂：档 C + eager + **补丁关**）；
  * `sg-c-c-eager-on`（判据⑥ 直接对照：档 C + eager + **补丁开**）；
  * `sg-c-c-cold`（判据⑤ 的**冷算参考**：池 1 MiB）。
  ⇒ 这三格**一律标【未完成】**，不得写成通过；判据⑥ 的"prefill 输出与延迟不变"目前只有
  **§2/§4 的机制性论证 + 旧支逐字未动**（A 段逐比特 + D 段路由），**没有**本轮的真机 eager 对照读数。
* 档 D 的**接受率**（用户关心的"保投机"）在本轮几何下**样本不足**（§5.4.1），
  需另一条同几何 + `max_tokens≥64` 的基线（`T_draftceiling` ②c）才能下结论。
* `SG_TRACE_PPR` 的 `capturing=` 标签来自 `torch.npu.is_current_stream_capturing()`（host 查询、无同步）；
  若该 API 在该 CANN 版本不可用，探针会静默降级为 `capturing=None`（不影响修复本身）。

---

## §8 ★★ md5 台账（**二维：档 × md5**）—— 哪条臂跑过它、结果如何

> **事故本身**：`dsa_v41.py` 一晚上换了三轮 md5，而"档 C 图模式 PASS"那一格是在 `22cbf20c…`
> 上拿到的、发布件却写成了另一个文件；**没有任何机械门拦着这个错**。
> 下面这张表是**事实**（从每条臂自己的 `arm.out` 台账 + 引擎日志机械抽取）。

★ **二维（档 × md5）** —— 门的语义是"这个 md5 在真机上跑过且干净"，
**不等于**"两个档都在这个 md5 上验过"：

| md5 | 是什么 | 档 | 跑过的臂 | 结果 |
|---|---|---|---|---|
| `75f4e565adc1b12c854a0a01271b6c4d` | **基底**（pkg-kv8pf 原版，无图安全） | — | 无（仅作 diff 参照） | — |
| `83508822b8556c5f2e55bbeaa4fd82ff` | 第 1 版（上界分支） | — | 无 | 离线 PASS；**【未上机】** |
| `1cc9e9923cc19749872cfb2e4decc4b7` | 第 2 版（+2 个诊断 env） | — | 无 | **⛔ 从未上机** ★ 曾被误写成"8 卡实测件"，已作废 |
| **`22cbf20c2544dd2ac6cb991a84806c42`** | 第 3 版（+ 捕获期路由） | **C** | `sg-a-c-graph` | ✅ 全绿（§5.1 左列） |
| （同上） | | **D** | `sg-a-d-graph` | ❌ rc=9 **segfault**（§4.5，根因 `repeat_interleave`，与判据无关） |
| **`94aeebb757d6d5708268754481a05e0a`** | 第 4 版（`index_select` / `expand` 替换） | **C** | `sg-c-c-graph-b` | ✅ **全绿，且读数与 `22cbf20c` 逐字相同**（§5.1 右列） |
| （同上） | | **D** | `sg-c-d-graph`、`sg-d-d-short-on` | ✅ 全绿（含 **graph==eager 逐字节**、同几何 A/B） |
| （同上） | | D（**反例·诊断**） | `sg-c-d-cmplegacy` | ⚪ **预期失败**：`507057 SUSPECT REMOTE ERROR` 崩引擎（§5.5.3，证明补丁必要） |

### 8.1 ★ `22cbf20c…` 的处置：**已按 md5 逐字节重建找回**

它已不在盘上（被 `mkpkg.sh` 覆盖），但**重建成功且 md5 逐字节相等**：
```python
# 从 patch/dsa_v41.graphsafe.py（94aeebb7）反向还原那两处 cmp 面的 op
#   index_select → repeat_interleave、expand+contiguous → repeat
reconstructed md5 = 22cbf20c2544dd2ac6cb991a84806c42   ✅ 与 arm.out 台账逐字相同
→ 落盘为 patch/dsa_v41.graphsafe.22cbf20c.py
```
★ 标签：【实测·重建】——**md5 相等即逐字节等价**，但它不是"从容器里捞出来的原件"，如实标注。

### 8.2 ★★ 档 C 的 PASS 是否受换版影响？—— **机械证明：不受影响**

`22cbf20c` → `94aeebb7` 的差异**全部落在 cmp（long-KV）面的图安全分支内**：

```
difflib.SequenceMatcher 的非 equal opcodes（旧文件行号）：
  replace old[929:931] → new[929:942]
  insert  old[955:954] → new[966:968]
  replace old[958:958] → new[972:973]
cmp 图安全分支 = 旧件 904..960；窗口面上界分支 = 旧件 547..567
★ 落在 cmp 分支之外的变更 = 无
★ 窗口面上界分支内变更 = 无（该段 md5 两版逐字节相同：ee56cfd93480f54d6e98c698971770a6）
```
而**档 C 根本不走 cmp 面**（`R8_KV8=0` ⇒ long-KV 是 BF16 ⇒ `source_scale is None` ⇒
`_kv8_cmp_plane` 不被调用）⇒ **档 C 的窗口面代码在 `22cbf20c` 与 `94aeebb7` 上逐字节相同**。

⇒ **档 C 的 PASS 用的是 `22cbf20c`（已含捕获期路由那一版），且该 PASS 对 `94aeebb7` 同样成立**
—— 但**发布口径仍必须按 md5 走机械门**（§8.3），所以 `sg-c-c-graph-b` 是**必要判据**，不是可选项。

### 8.3 ★★ 发布门（机械）：`scripts/check_publish_md5.py`

★ 该门的**完整输出**（含二维台账）已落盘：`raw/049-publish-gate.txt`
（31 行 / md5 `cecf5bcab60b8a1926b3b89c1590da24`）。

把"发布件必须是某条 PASS 臂挂过的 md5"变成一条命令（读 `arm.out` 台账 + `rc` + 引擎日志，
检出 `capture failed / EE1016 / Segfault / Engine core initialization failed / Worker proc died /
SUSPECT REMOTE ERROR / EngineDeadError`）：

```
$ python3 scripts/check_publish_md5.py --candidate <dsa_v41.py> --tier C     # 档 C 发布
  22cbf20c… × 档 C  → ✅ PASS
        ✅ sg-a-c-graph
  22cbf20c… × 档 D  → ❌ 有失败臂
        ❌ sg-a-d-graph（!!!!!!! Segfault encountered !!!!!!!）
  94aeebb7… × 档 C  → ✅ PASS
        ✅ sg-c-c-graph-b
  94aeebb7… × 档 D  → ✅ PASS（另有诊断反例臂）
        ✅ sg-c-d-graph        ✅ sg-d-d-short-on
        ⚪ sg-c-d-cmplegacy（诊断臂，预期失败：SUSPECT REMOTE ERROR）
  候选件 md5 = 94aeebb7…   该 md5 已验证通过的档 = ['C', 'D']   要求 = C
  [gate] ALLOW ✅ —— 该 md5 在**档 C** 上有 PASS 臂背书
```
⇒ **当前状态：`94aeebb7` 在档 C 与档 D 上都 ALLOW**（两档各有 PASS 臂）。
退出码：`0` = ALLOW / `2` = DENY / `3` = 输入缺失。
★ **两个设计要点**（都是本次事故逼出来的）：
1. **档敏感**：不带 `--tier` 只说"某个档验过"，发布必须带 `--tier C` 或 `--tier D`；
2. **不能只看 rc**：诊断臂的**客户端 rc = 0**（引擎死在服务端）⇒ 门必须以"**致命证据为空**"为**必需**条件，
   而"诊断臂"的识别也走**运行期证据**（探针打出 `cmp_legacy_triton` 行），不靠 tag 名字或 env 字符串。

### 8.4 ★ 本次新增的两条判据（第一格真机臂给的新教训）

1. **起服第一格必须单独成判据**：`Engine core initialization failed` / `Segfault` /
   `Worker proc ... died` **必须为 0**；这一格失败**不得**被记成"捕获失败"或 `EE1016`
   （本次就是这么被误读的方向）。⇒ 已并入 `check_publish_md5.py` 的证据扫描，
   并在 §5.5 的档 D 判据表里单列。
2. **判据⑤ 的最强单条形态**：`sg-c-d-graph` 的 **`replay1 sha` 必须与"同几何 eager 臂"逐字节相同**
   （档 C 已经做到：`bc2e797a…` 图 == eager）。

### 8.5 `_sg_is_capturing()` 的降级路径（显式判据）

`torch.npu.is_current_stream_capturing()` 在本镜像里存在（`torch_npu/npu/graphs.py:67`），
且 8 卡臂的 `[SG-PPR]` 打出 **`capturing=True`** ⇒ **A3 上确实可用**【实测】。
* 若某环境上该 API 抛异常 ⇒ 我们的实现**退回 legacy**（`_kv8_graph_rows_bound` 在
  `num_prefills>0` 时返回 `(False, None)`）⇒ 捕获期会**响亮地** `EE1016`（安全失败，不静默错）；
* 判据写法：`[SG-PPR] ... capturing=` 必须出现且非 `None`；若为 `None`/缺行 ⇒ **本臂结论无效**。
* A2 是否同样可用：**同 wheel，【推断】可用**；A2 首次起服时用这一行确认（不需要额外成本）。
