# `kv8-graphsafe` —— 让 int8（档 C / 档 D）在 `FULL_DECODE_ONLY` 下可用的图安全补丁

> **来源**：`S_graphfix`（`logs/049`）。**状态**：档 C 已实测通过（8 卡真权重 + 生产图模式）。
> 本目录是**自包含的交付件**：一个替换文件 + 一个可重放的生成器 + 一个接入脚本。

---

## 1. 为什么需要它（两个问题，一份补丁）

```
① dsa_v41.py:436 的 .item()（宿主同步）在 spec-decode 下必被走到：
   if query_rows == num_reqs:   ← decode 分支（device-side、capture-safe）
   else:                        ← prefill 分支（.max().item() ⇒ 捕获期炸 EE1016）
   捕获时是 spec-decode 的 decode 批（num_spec_tokens=5 ⇒ 每请求 6 行 query）
   ⇒ query_rows = 6 × num_reqs ≠ num_reqs ⇒ 误走 prefill 分支
   ★ 判据：speculative-config 里 num_speculative_tokens=5；纯 BF16（档 B）同一条链能捕获成功
     （因为 BF16 的 SWA 面根本不进这个函数）

② ★★ 档 D 的 long-KV INT8 面（`_kv8_cmp_plane`）在 spec-decode 下同样 rows != num_reqs
   ⇒ 走"整段压缩前缀重建"，页数来自 cache_seq_lens.max().item()（D2H）；
   而档 D 的 VLLM_V41_KV8_PREFILL=1 会把它换成 fused_cmp_plane3(ppr=ceil(max_cache_seq_len/bs))，
   这个 mcs 是【捕获期 _dummy_run 的 seq_lens=max_query_len=6】算出来的 ⇒ ppr≈1 页，
   而 replay 的压缩前缀有几百块 ⇒ ★ 不炸但静默读错
   ★ [SG-PPR] 探针实测（8 rank 一致）：capturing=True 时 cmp_mcs=6 ⇒ ppr = ceil(6/128) = 1
```

---

## 2. 两条改动（同一份 `attention/dsa_v41.py`，全部 env 门控、默认关）

| # | 改动 | 关键洞察 |
|---|---|---|
| **①** | `kv8_ori_plane(..., rows_bound=)` 新增**上界分支** | ★ **"必须 host 的不是【精确最大值】，而是【可证上界】"** —— 窗口带的起点仍在 device 上按真实 `q_len` 算（`query_start_loc` 差分），只有"每请求最多几页"这个 host 标量改成 `(rows_bound + window - 1) // block_size + 2`，其中 `rows_bound` 来自 `metadata.swa.max_query_len`（引擎在 CPU 张量上算好的 Python int，**不产生 D2H**）。★ `.item()` 那条 **prefill 分支原样保留**（eager only，是本任务的"不许动"项） |
| **②** | `_kv8_cmp_plane(..., graph_safe=)` 新增**按 query 行私有一段**的 selection-based 分支 | 第 `i` 行的第 `t` 个选择 → 合成下标 `i * topk + t` ⇒ scratch 页 `i * per_req + t // block_size`、页内偏移 `t % block_size`，scratch 表取 `rows * per_req` 页的恒等表。★ **全部标量来自 shape/config**（不再用捕获期冻结的 `max_cache_seq_len`） |

**与档 D 的 `VLLM_V41_KV8_PREFILL=1` 尾部包装共存**：`kv8_ori_plane` 包装在 `rows_bound` 非 None 时转交原函数；
`_kv8_cmp_plane_prefill` 包装在 `graph_safe` 且行数整除时转交类方法。

---

## 3. 文件与 md5

| 文件 | md5 | 说明 |
|---|---|---|
| `dsa_v41.py` | **`94aeebb757d6d5708268754481a05e0a`** | ★ **成品**（档 C 已 8 卡实测通过；**档 D 的候选**，其图臂在验） |
| ~~`dsa_v41.py`（旧件，2026-09-22 11:2x 前）~~ | ~~`1cc9e9923cc19749872cfb2e4decc4b7`~~ | ⛔ **作废**（1749 行）：它含 `repeat_interleave` ⇒ **档 D 图臂 8 卡同时 segfault**（见下 §3.1） |
| （基底）`X_integrate/pkg-kv8pf/…/dsa_v41.py` | `75f4e565adc1b12c854a0a01271b6c4d` | 镜像是这个版本；本成品 = 它 **+301/−3** 行（1499 → **1797** 行，`diff` 实测） |
| （参考）`pkg-ring/…/dsa_v41.py` | `9db97849aaa5c7284a46f49ea2e81681` | 不含 prefill triton ⇒ **档 D 不要用这份** |
| `apply_graphsafe.py` | — | **可重放的生成器**（幂等、9 个 exact-match 锚点）：`python3 apply_graphsafe.py --src <基底> --out <目标>` |
| `patch_serve_sg.sh` | — | 容器接入（幂等，只加 3 行 export） |

**★ 替换清单：`attention/dsa_v41.py` 一个文件**（其余 6~7 个 int8 挂载件不用动）。

### 3.0 ★★ 「每条实测结论对应哪个 md5」（**md5 一换，"已过"就要重标**）

本次踩过一次：`1cc9e992 → 94aeebb7` 换了 md5，而新件**不只改 cmp 面**，还改了
**`_kv8_graph_rows_bound`（窗口面 = 档 C 走的那条路）** ⇒ **档 C 的图模式结论必须重新对号**。

| 结论 | 在哪个 md5 上测的 | 新 md5 上是否还有效 |
|---|---|---|
| 档 C `FULL_DECODE_ONLY` 起服 + `EE1016=0` + `replay sha == eager` | ⏳ 待 `S_graphfix` 确认（`1cc9e992`？） | ⚠️ **待 `sg-c-c-graph-b` 复跑**（因为窗口面判据改了） |
| 档 D 图模式四条判据 | ⛔ 在 `1cc9e992` 上**失败**（起服就 segfault，见 §3.1） | ⏳ `sg-c-d-graph` 在跑 |
| 档 D 的 cmp 面 `[SG-PPR] ppr=1` 捕获期读数 | `1cc9e992` | 【推断】仍成立（那段没改） |
| 离线自检 7/7 + 锚点 10/9 | — | ★ **已在 `94aeebb7` 上重跑 PASS**（主代理复算，逐字节可重放） |

> ★ **纪律（写进流程）**：以后任何 patch 件换 md5，**`logs/` 里每条"已过"的判据都要标上它的 md5**；
> 没标的按【未确认】处理。这条是本次 segfault 事故的直接教训之一。

### 3.1 ⛔ 为什么旧 md5 作废（`049` §4，【实测】）

```
!!!!!!! Segfault encountered !!!!!!!
  aclnnOpInfoRecord::TilingContextToJson(...)
  CommonOpExecutorRun(...)
  aclnnRepeatInterleaveIntWithDim          ← ★ 就是它（int64、dim=0）
(EngineCore) ERROR Worker proc VllmWorker-5 died unexpectedly → 8 个 worker 同时死
→ "Engine core initialization failed"      （起服就挂，不是捕获失败、不是 EE1016）
```
**修法**：把那条"行 → 请求"的映射从 `repeat_interleave` 换成**本文件里已经在用的原语**：
```python
b_of_row   = torch.arange(rows, device=indices.device, dtype=torch.int64) // reps
table_rows = torch.index_select(block_table[:num_reqs].to(torch.int64), 0, b_of_row)
```
恒等表也从 `.repeat(num_reqs, 1)` 换成 `.expand(...).contiguous()`（同样的防御理由）。
`apply_graphsafe.py` 现在有**第 7 条自检**：`need(".repeat_interleave(" not in text, ...)`
⇒ 生成器本身就能拦住回归。

> ★ **教训**：`ast` 抽真函数 + CPU 穷举**只能证明"逻辑对"**，**证明不了"这个算子在设备上能用"**。
> 这一格只有真机臂能给 —— 补丁自检 / 离线穷举 / 阳性对照当时**全过**，真机上 8 卡一起 segfault。

---

## 4. env（全部默认关 ⇒ 不设就是逐字旧行为）

| env | 默认 | 作用 |
|---|---|---|
| `VLLM_V41_KV8_GRAPH_SAFE` | **0** | ★ **主开关**（档 C/D 上线必须打开 = 1） |
| `SG_CMP_LEGACY` | 0 | 诊断臂专用：只修窗口面、cmp 面留在捕获期路径（发布**不用**） |
| `SG_TRACE_PPR` | 0 | 只读探针：打印捕获期/replay 的 `mcs` 与 `ppr` |

---

## 5. 档 C 的实测判据（`logs/049`，8 卡真权重 + `FULL_DECODE_ONLY`）

| 判据 | 读数 |
|---|---|
| 起服 | ★ `EE1016 = 0`、`capture failed = 0`、就绪 **659 s**、`static_kernel` 无降级 |
| `GPU KV cache size` | **427,643**（与档 B / 档 C-eager **逐字相同** ⇒ 容量零退化） |
| `BlockStored:CPU` | **29,436**（逐字相同） |
| `CPU→GPU` | **21,188,968,448 B = 21.19 GB**（> 0 ⇒ 真命中） |
| `hits` / `queries` | **901,120 / 3,145,984**（> 0） |
| replay vs fill | **1,608.2 vs 19,880.0 ms = 12.36×** |
| `BlockRemoved:CPU` | **0**（上线监测判据） |
| `fill sha` | `d524172f9f5ae368…`（与档 B **逐字相同**） |
| ★★ `replay1 sha` | `bc2e797ab069f09ced…`（**与档 C-eager 逐字相同** ⇒ **图模式 = eager 输出**） |

★ **为什么"sha 逐字相同"是最强判据**：唯一变量就是"图 vs eager"。
而 `logs/037` 证明该服务在 `temperature=0` 下**同臂内都会抖**（32K 2/4、512 短 prompt 3/4 不同）——
⇒ **在这样抖的环境里跨模式逐字相同**，说明两条路径在这条 workload 上**数值等价**。

---

## 6. ⚠️ 两个必须知道的边界

1. ★★★ **`[SG-PPR]` 证明档 D 的 cmp 面在捕获期 `ppr=1`**（实测），**且"越界读会怎样"已定案 = (c) 静默错**
   —— **源码级证据，零占卡**（`S_graphfix` 的 `raw/049-op-bounds-evidence.txt`；主代理已独立核实内核那一段）：

   ```
   ① host 侧 checker（op_host/checkers/paged_attention_checker.cpp:23-40）
      CheckBlockTable 只查四件事：dtype==int32 / 维度数==2 / 每维非空 / dim0==batch size
      ★ 没有任何一处把「列宽」和「kernel 会寻址到的最大块号」做比较
      ⇒ 列宽 = 1 的表【通过全部 host 校验】⇒ (b) 响亮失败被排除

   ② tiling（op_host/sparse_flash_mla_tiling.cpp:896）
      cmpMaxBlockNumPerBatch_ = cmpBlockTable.tensor->GetStorageShape().GetDim(1)   ← 就是列宽

   ③ ★★ device kernel（op_kernel/arch22/sparse_flash_mla_csa_block_vector.h:544-552）—— 主代理逐行核实：
      if (realS2Idx < 0 || realS2Idx >= s2IdLimit) { return -1; }   ← 唯一的界检查：查【位置】
      int64_t blkTableIdx = realS2Idx / constInfo.paCmpBlockSize;
      realKeyGmOffset =
          cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx) * ...
      //                          ↑ 行偏移（= bIdx × 列宽）        ↑ 列偏移（可达几百）
      //  ★ 没有任何一处检查 blkTableIdx < cmpMaxBlockNumPerBatch
      ⇒ 列宽=1 而 blkTableIdx=300 时索引 = bIdx + 300 ⇒ 直接越界读 GM，无检查
      ⇒ 读到同进程 GM（通常不触发 MMU 故障）⇒ 垃圾值被当成"页号"去取 KV 行 ⇒ ★ 静默算错
      （只有"垃圾页号 × cmpKvStride0"恰好落到未映射页时才退化成崩溃）
   ```
   ⇒ ★ **标签：【实测·源码级】**（静态、可复核、可重复）。
   ⇒ **决策臂 `sg-a-d-cmplegacy` 的价值从"分 (b)/(c)"降级为"运行期复现 (c)"**
     （预期：起服成功 + 输出错；若它居然输出正确，说明那条路没被走到 ⇒ 当**空判据**如实报告，而不是"没问题"）。
2. ★ **调度器侧的 trace 只证明"调度器没冻结"** —— 它打在 EngineCore（图**外**），
   **不能**证明"图内部的值没被冻结"。图内部那一格由 `[SG-PPR]` 回答（答案：**`ppr=1`，被抓死**）。
