# 009 · DSV4.1 八卡 DRAM KV 卸载：把"能存不能取"修到"真的取回"（**已定位根因并修复**）

**日期**：2026-09-21 23:0x – 2026-09-22 00:2x（A3 本地时钟；共 4 条臂）　**执行**：子代理 `D2_offload`　
**机器**：A3（A3-node1），Phy-ID 8–15　**上游**：vLLM `0.27.1` + vllm-ascend `e43cf1e9f`（镜像
`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`）
**接续**：[`001`](001-dsv41-dram-offload-8card.md)（前任 `D_off8`，结论是"未达标"）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/日志推出来但没直接测；【未确认】= 没跑到。

---

## 0. 一句话结论

**【实测·已修复】两条独立根因，缺一不可**：

1. **命中判定被"结构性不可能命中的组"否决**：DSV4.1 的 `state` 组
   （`prefix_cacheable=False`、每请求只有 1 页环形 scratch）在**取回路径**上依然被当作
   full-attention 组参与命中判定，而它在**存储路径**上永远存不出 chunk
   （`len(block_ids)//blocks_per_chunk = 1//8 = 0`）⇒ 它的 key 必然 MISS ⇒
   上游 `_lookup()` 里那句 `if num_hit_chunks == 0: return 0` 把**整轮请求**
   （含 full 组与 40 个 SWA 资源）的命中全部判死；
2. **池子比"一轮工作集"小 6.4 倍**：32 GiB 的 CPU 池只有 ~963 个 chunk 位（算法见 §3.1），
   而 16 请求 × 32768 token 一轮就要 6,144 个 chunk（12 个参与组 × 32 chunk × 16 请求）
   ⇒ LRU 从**插入序前端**淘汰 ⇒ 每个请求的**前缀链头部最先被挤掉**，
   而 `_maximal_prefix_lookup` 要求从第 0 块起连续命中 ⇒ 整轮 0 命中。

> 一句话版：**存的组比查的组少一个（state 永远查不中，一票否决）；
> 而且池子装不下一轮，LRU 把前缀链的头先吃掉（全有或全无）。**

把两处都修掉之后，**四条判据全部转正**（臂 `d2-dram32-2p`，§5）：
`CPU_to_GPU = 1.56 GB`（原 0）、`external_prefix_cache_hits = 63,488`（原 0）、
**replay TTFT p50 298.2 ms vs fill 4424.6 ms（14.8×，原 +0.6%）**。

| 判据 | 修复前（`001`/臂 D，16 请求） | 修复后（本次臂 `d2-dram32-2p`，2 请求） |
|---|---|---|
| ① `BlockStored(medium="CPU") > 0` | ✓ 12,288 | **✓ 768**（= 2 请求 × 12 组 × 32 chunk，精确吻合） |
| ② `kv_offload_total_bytes_total{CPU_to_GPU} > 0` | ✗ **0.0** | **✓ 1,555,480,576 B（1.56 GB）** |
| ③ `external_prefix_cache_hits > 0` | ✗ **0** | **✓ 63,488**（queries 131,328） |
| ④ replay TTFT ≪ fill TTFT | ✗ 4167 vs 4192 ms（+0.6%） | **✓ 298.2 vs 4424.6 ms（14.8×）** |

> ⚠️ 口径说明（不许用相邻数字顶替缺的那格）：**判据 ④ 的请求数从 16 缩到 2**。
> 原因是 §3.2 的"pinned 单次分配上限"把池子卡在 32 GiB，而 32 GiB 只装得下 ~963 个 chunk，
> 16 请求一轮要 6,144 个 chunk。**同一份代码、同一套探针、同样 32768-token 的请求**，
> 只把请求数和池子配平。**"16 请求 + 32 GiB 零取回"这个现象本轮仍然【实测】复现了**
> （臂 `d2-dram32`：`CPU_to_GPU=0`、`hits=0`），只不过它现在**有完整解释**而不是谜。

---

## 1. 环境与复现入口

| 项 | 值 |
|---|---|
| 机器 / 卡 | A3-node1，Phy-ID **8–15**（起跑前 `npu-smi` 确认 8–15 无进程；0/1/2 有别人的进程，未动） |
| 模型 | `~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（真权重 DSV4.1-Flash W4A8） |
| 臂参数（四条臂共用） | `ENGRAM=0 PREFIX_MATCH_UNIT=32 BLOCKS_PER_CHUNK=8 OFFLOAD_SCHED_PATCH=1`、`MAX_LEN=40960`、`KV_MEM_BYTES=1 GiB`（`GPU KV cache size: 46,387 tokens`）、每请求 **32768 token**、2 轮、轮间 `POST /reset_prefix_cache`、`max_tokens=1`、`concurrency=1` |
| 各臂只差两处 | ① `PROMPTS`（请求数）② `OFFLOAD_GB`（CPU 池）：`d2-dram32`=16/**32**、`d2-dram32-2p`=**2**/32、`d2-dram64-4p`=4/64、`d2-dram128-6p`=6/128 |
| 起服 / 压测入口 | `agents/D2_offload/scripts/run_arm_d2.sh`（拿 c0 锁）→ `agents/D_off8/scripts/run_arm_8card.sh`（前任的臂运行器，**未改**） |
| 补丁入口 | `agents/D2_offload/scripts/make_offload_scheduler_patch_d2.py` → 挂到 `shadow-pkg/patches/files/offload_dsv41/scheduler.py`（`OFFLOAD_SCHED_PATCH=1` 时 `serve_a2.sh` 把它挂进容器） |
| 与前任的差别 | **只换了 `offloading/scheduler.py` 这一个文件**；臂参数、workload、探针口径与 `001` 臂 D 完全一致（便于对比） |

复现（A3 上，一条臂 ≈15 min，其中起服 ≈11 min：6 min 权重 + KV 初始化）：

```bash
cd ~/projects/dsv41-upstream-pr/agents/D2_offload
TAG=d2-dram32-2p ENGRAM=0 OFFLOAD_GB=32 PREFIX_MATCH_UNIT=32 OFFLOAD_SCHED_PATCH=1 PROMPTS=2 \
  bash scripts/run_arm_d2.sh        # 内部先 flock c0.lock，抢不到 = 退出码 75
```

> 注：池子的选择**不是随手填的** —— 按 §3.1 的算法，`OFFLOAD_GB=32` ⇒ ~963 个 chunk 位，
> 而 2 请求 × 12 组 × 32 chunk = 768 个 chunk（占用 80%，留 20% 余量）。
> 请求数再多（4 / 6 / 16）就必须把池子抬到 64 / 96 / 256 GiB，而那会撞上 §3.2 的 pinned 上限。

---

## 2. ★ 根因链（逐行，全部来自镜像内源码）

### 2.1 `_lookup()` 的"一票否决"语义

`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`（镜像内原版行号）：

```python
self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups   # __init__

for group_idx in groups_iter:                     # _lookup()
    ...
    if sliding_window_size_in_chunks is None:
        num_hit_chunks = self._maximal_prefix_lookup(offload_keys, ...)
    else:
        num_hit_chunks = self._sliding_window_lookup(offload_keys, required_window, ...)
    if num_hit_chunks == 0:
        return 0                                  # ★★ 任何一组 0 命中 ⇒ 整轮 0 命中
```

而"是不是 full-attention 组"的唯一判据是
`get_sliding_window_size_in_chunks(kv_spec, ...) is None`，该函数对**任何** `AttentionSpec`
都 `return None`（滑窗/ Mamba 之外全都算 full attention）。

### 2.2 `state` 组：结构上永远存不出 chunk

| | 事实 | 出处 |
|---|---|---|
| spec | `DeepseekV41CompressorStateSpec`（→ `AscendCircularBufferSpec(AttentionSpec)`，`prefix_cacheable = False`，`block_size = 32`，FP32 32 行环） | `vllm_ascend/core/deepseek_v41.py:68`、`core/circular_buffer.py:40` |
| manager | `AscendCircularBufferManager`：**每请求只给 1 页**（`get_num_blocks_to_allocate()` 恒返回 `1` 或 `0`；`max_num_blocks_per_req = 1`） | `core/circular_buffer.py:59-90` |
| 它的 GPU 命中 | `find_longest_cache_hit()` **恒** `return tuple([] ...), 0` —— 设计上就"不参与前缀缓存" | `core/circular_buffer.py:92-107` |
| store 侧 | `storable_chunks() = min(num_chunks_by_tokens, len(block_ids)//blocks_per_chunk)` = `1 // 8` = **0** ⇒ 它的 offload key 一个都不会被存 | `offloading/scheduler.py` `RequestOffloadState.storable_chunks` |

⇒ **它不是"存了没取回"，而是"从来没存过，却每次都要查"**。

### 2.3 算术闭环：12,288 这个数字本身就证明了 state 组没被存【实测】

臂 D（`001` §2）里 `BlockStored(medium="CPU") = 12,288`。按模型结构算：

```
每个请求 32768 token；tokens_per_hash = 32（--prefix-match-unit 32）
  * full 组（block 128，blocks_per_chunk 8）  : chunk = 1024 token ⇒ 32 chunk
  * 10 个 SWA 组（block 128）                : 每组 32 chunk ⇒ 320 chunk
  * dspark 组（block 128）                    : 32 chunk
  * state 组（block 32，每请求 1 页）          : 0 chunk   ← ★
  ------------------------------------------------
  每组每请求 384 chunk；16 请求 × 2 轮 = 16×2×384 = 12,288   ← 与实测**精确相等**
```

也就是说：**12 个可缓存组全存了（含 40 个 SWA 资源的 10 个组），唯独 state 组一个 chunk 都没存**；
而 `_lookup()` 每次都去查 state 组的 key ⇒ 第一次查就 `return 0`。

### 2.4 为什么 GPU 侧前缀缓存（生产路径，命中率 96%）不受这个坑影响

`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py::verify_and_split_kv_cache_groups()`：

```python
for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
    if not prefix_cacheable(g.kv_cache_spec):
        continue                       # ★ GPU 侧命中计算里，state 组被显式跳过
```

⇒ **"不参与前缀缓存的组不进命中判定"这条规则，GPU 侧有、卸载层没有。**
本次修复就是把这半条规则补上去（见 §4）。

### 2.5 对 `state` 组"不取回"的数值影响：与 GPU 路径同语义【推断】

GPU 前缀缓存命中时，state 组同样是每请求新分配 1 页、内容由模型自己算
（`AscendCircularBufferManager.find_longest_cache_hit` 恒返回 0、`allocate_external_computed_blocks`
只是 claim 一页）。卸载层跳过它之后，行为与 GPU 命中路径**逐字一致**：
**不是"少取回了一份必需数据"，而是"这个组本来就不属于可复用的前缀数据"。**
（⚠️ 本次**没有**做数值正确性/精度验证，只跑了 TTFT 与命中判据。标【推断】。）

---

## 3. 前任（`001` §4.4）三个候选根因的逐条判定

| # | 候选 | 判定 | 依据 |
|---|---|---|---|
| ① | **`group_idx` 错位**（调度侧与 worker 侧遍历序不同） | **排除**【推断，代码级】 | 调度侧 `SchedulerOffloadConfig.from_spec` 用 `enumerate(spec.tokens_per_block)` 并以 `kv_cache_config.kv_cache_groups[idx].kv_cache_spec` 取 spec；`spec.tokens_per_block` 本身由 `build_offloading_config` 用**同一个** `kv_cache_config.kv_cache_groups` 元组按序构造。worker 侧 `offloading_connector.py:300-305` 也是 `for group in kv_cache_config.kv_cache_groups: ... group_data_refs.append(group_refs)`。**两边的顺序依据是同一个列表、同一个枚举序 ⇒ 对位**（且这是进程间传的对象，不是两处独立推导的 dict 遍历序）。 |
| ② | **state 组把整轮命中判死** | ★ **确认（就是根因）**【推断，代码级 + 算术闭环】 | 见 §2。判死机制**不是**"要求同组内所有 group 都有 key"，而是 `_maximal_prefix_lookup` 返回 0 后 `if num_hit_chunks == 0: return 0`。 |
| ③ | **hash 粒度 32 vs chunk 粒度 1024 不对齐** | **不成立（但 `--prefix-match-unit 32` 仍必需）**【推断 + 实测】 | `hashes_per_chunk = tokens_per_chunk // tokens_per_hash`：full/SWA/dspark = `1024//32 = 32`，state = `256//32 = 8`；**存与查用的是同一个公式**（`update_offload_keys` 的 `islice` 步长 / `_lookup` 的 `len(offload_keys)*tokens_per_chunk`），没有算错。而且实测 chunk 数（32/组/请求）与 12288 精确吻合 ⇒ 粒度对齐无问题。`--prefix-match-unit 32` 仍然**必需**：否则 `build_offloading_config()` 里 `tokens_per_block % tokens_per_hash`（state 组 `32 % 128`）先断言失败（`001` §4.1）。 |

**同时纠正一条主代理的推断**（`2026-09-21 23:4x` 的插入消息）：
`alignment_chunk_count=None` **不是**"SWA 组判不中"，而是"不做可达性裁剪（全存）"——
它只有两处用法，store 侧 `is_store_reachable_swa_chunk()` 的**首行**就是
`if alignment_chunk_count is None: return True`；命中路径完全不读这个字段。
它确实会退化成 `None`（`full_attn_tokens_per_chunk = {1024, 256}` 因为 state 组被算成 full attention），
但后果只是**多存了一些永远用不上的 SWA chunk**，对命中没有影响。
（修掉 state 组之后，该集合自然变成单值 `{1024}`。）

### 3.1 ★ 第二个独立根因：**池子比"一轮工作集"小 6.4 倍 ⇒ LRU 把前缀链头部挤掉**

只修 state 组的臂（`d2-dram32`，**唯一改动 = 换 scheduler 补丁**）跑完后，
判据 ②③ 仍然是 0（`CPU_to_GPU=0.0`、`external hits=0`），但**诊断日志把原因指出来了**：

| 观测（`d2-dram32`）【实测】 | 值 | 含义 |
|---|---|---|
| `[D2_offload] lookup-summary ... group=0 hit=0` | 每个请求都打一行、**group=0** | 否决来**自 full 组本身**（不是 state 组——state 组在我这一版里已经不进命中判定了） |
| `tally` 逐请求 +32（组 0）、一路涨到 992 | **31/32 个请求都执行了 store** | ⇒ **replay 轮的 lookup 全部未命中**（命中就不会再 store） |
| `BlockStored:CPU` / `BlockRemoved:CPU` | 12,288 / **8,192** | 池子只留得住约 1/3 |
| `kv_offload_cpu_allocation_size_{count,sum}` | 160 次 / **12,288 个 CPU 块** | 池子的**块计数**口径：一个 chunk = 一个 CPU 块 |

**池子到底多大（【实测·算术闭合】）** —— `vllm/v1/kv_offload/cpu/spec.py:86-110`：

```python
num_copies = world_size                      # Ascend 不是 cuda_alike ⇒ replicated_layout=False ⇒ 8 份
kv_bytes_per_block  = worker_kv_bytes_per_block * num_copies
kv_bytes_per_chunk  = kv_bytes_per_block * blocks_per_chunk
self.num_blocks     = cpu_bytes_to_use // round_up(kv_bytes_per_chunk, BLOCK_SIZE_ALIGNMENT)
```

代入本臂的实测值（`kv_cache_memory_bytes=1 GiB`、`num_gpu_blocks=1984` ⇒
`worker_kv_bytes_per_block = 1 GiB/1984 = 541,198 B`）：

```
kv_bytes_per_chunk = 541,198 × 8（块/chunk） × 8（rank 副本） = 34.6 MB → 对齐后 ≈ 35.7 MB
OFFLOAD_GB=32  ⇒ num_blocks ≈  963 个 chunk
OFFLOAD_GB=64  ⇒ num_blocks ≈ 1927
OFFLOAD_GB=128 ⇒ num_blocks ≈ 3855
```

而本臂（16 请求 × 32768 token，chunk=1024 token）**每轮的 chunk 需求**是：

```
每请求：12 个参与组 × (32768/1024 = 32 chunk) = 384 chunk
一轮 16 请求 = 6,144 chunk        ← ★ 是 32 GiB 池（963 块）的 6.4 倍
```

⇒ **池子只装得下一轮工作集的 16%**，而 `cpu/policies/lru.py` 的淘汰是
**从插入顺序的前端开始**（`evict()` 按 `evictable_blocks` 顺序取）⇒
**每个请求最先生成的 chunk（= 前缀链的头部）最先被淘汰**；而
`_maximal_prefix_lookup` 要求**从第 0 个 chunk 起连续命中**，首块一 MISS 就 `break` ⇒ 整轮 0 命中。
这解释了两件事：`BlockRemoved:CPU=8,192`（1,536 个无法安置的块被挤掉 —— 口径见下）
以及"31/32 个请求都 store 了"（replay 轮全部退回重算）。

> 为什么 `001` §4.4 的"32 GiB 池已能覆盖 18.5 GiB 工作集"这条推理不成立：
> 18.5 GiB 是**按唯一 KV 字节**算的（512 个唯一 chunk 位置 × 35.7 MB ≈ 18.3 GB，与
> `4421 B/token × 524288 token × 8 rank = 18.5 GB` 对得上）；但卸载层是**按 (group, 位置) 计费**的
> —— 同样这 512 个位置，在 12 个组里各占一个 chunk ⇒ **6,144 块**。
> 换句话说：**DSV4.1 的 12 个参与组把"有效容量"除以了 12**。

**★ 对 A2 容量规划的直接影响**【推断，依据本机【实测】的 12× 倍率】：
`docs/KV-CACHE-ACCOUNTING.md` 的 `tokens = cpu_bytes_to_use / (4421 × 8)` 只在
**单组模型**（如 Qwen3，`logs/45` 那次）成立；DSV4.1 有 12 个参与组 ⇒
**同样的池子能覆盖的"会话历史 token 数"要再除以 ~12**。
（实测倍率：每请求每 rank 实际搬运 **1.54 GB**，而该请求的唯一 KV 只有
`4421 B × 32768 = 145 MB` ⇒ **10.6×**；池子块数口径是 **12×**。两者同源。）

### 3.2 ★ 第三个坑：**池子不能无限放大 —— 16 GiB/worker 的 pinned 分配上限（`207001`）**

为了直接检验 3.1 的"池子不够"结论，我把池子从 32 GiB 抬到 **128 GiB**（其余参数不变），
结果**起服直接失败**，而且是在 `ENGRAM=0`（关掉 Engram）的情况下【实测】：

```
[cpu_npu.py:277] Allocating 16 CPU tensors...          ← 8 个 worker 都走到这一步
torch.OutOfMemoryError: allocate_host_memory_slowpath:...
  NPU function error: aclrtMallocHostWithCfg, error code is 207001
  Resource_Error_Insufficient_Host_Memory(EL0018):
    Failed to allocate 17179869184 bytes host memory     ← 16 GiB = 128 GiB / 8 rank
  rtsMallocHost execution failed, reason=driver error:out of memory
（失败序列：TP4 先要 16 GiB；TP0 随后要 2 GiB(=16 GiB/8) 也失败；  8/8 worker 全中）
```

**这条推翻了 `001` §4.2 的因果结论**："`ENGRAM=1` 必挂、`ENGRAM=0` 必通"不成立 ——
真正被卡住的是"**单个 pinned 分配太大**"：

| `OFFLOAD_GB` | 每 worker 要 pin | 结果 |
|---|---|---|
| 4 GiB | 0.5 GiB | 【实测·前任】起服成功 |
| 32 GiB | 4 GiB | 【实测】起服成功（本轮 `d2-dram32`、`d2-dram32-2p` 两条臂 + 一次被打断的重跑） |
| **64 GiB** | **8 GiB** | **【实测】起服失败：`207001`（`ENGRAM=0`）；8/8 worker 全中** |
| **128 GiB** | **16 GiB** | **【实测】起服失败：`207001`（`ENGRAM=0`）；8/8 worker 全中** |

⇒ 【实测·推断】本机这条路径的 pinned 上限落在 **(4, 8] GiB / worker**，
即**服务级池子上限落在 (32, 64] GiB**（本次可用的是一个 **32 GiB** 的池子）。

同时宿主 `MemAvailable` 起跑前 = **859 GiB**、容器 cgroup 无限制 ⇒ **不是"宿主没内存"**，
与 `001` §4.2 的"是驱动侧 pinned 池"结论方向一致，但**触发条件被重新定位为分配粒度/单次大小**，
而不是"Engram 抢资源"。（为什么 D_off8 的独立探针能在单进程里连续 pin 128 GiB：
那些探针是**逐 1 GiB 申请**，而 real 路径是**一次要 16 GiB** —— 这解释了"探针全通过、真机必挂"的悖论。）

**对 A2 的直接含义**【推断】：池子大小的上限**不是**宿主内存，而是**驱动侧单次 pinned 分配**；
要更大的池子必须换分配形态（例如 `logs/45` 里 Engram 用的 `mmap + aclrtHostRegister(MAPPED)`，
或把 `cpu_page_size_per_worker` 切小），**不能只把 `cpu_bytes_to_use` 往上调**。
（A2 的 `host_mem_pool=0` 让这条更硬：A2 上很可能连 4 GiB 都要先验证。）

---

## 4. 修复（最小改动，只动一个文件）

`agents/D2_offload/scripts/make_offload_scheduler_patch_d2.py` 生成
`agents/D2_offload/patches/offload_dsv41/scheduler.py`（挂载替换镜像内同路径文件）。
**核心概念**：给 `GroupOffloadConfig` 加一个 `offload_participating` 位
= "该 group 是否参与前缀缓存"（递归 unwrap `UniformTypeKVCacheSpecs` 后看
`prefix_cacheable` / `participates_in_prefix_caching`，与 vllm-ascend 的 `prefix_cacheable()` 同口径），
然后**在存与查两侧都排除它**：

| # | 位置 | 改动 |
|---|---|---|
| 1 | `get_sliding_window_size_in_chunks()` | 入口 unwrap `UniformTypeKVCacheSpecs`；assert 从 `FullAttentionSpec` 放宽到 `AttentionSpec`（DSV4.1 的 13 个组全是包装类型） |
| 2 | `SchedulerOffloadConfig.from_spec()` | 逐组建 `offload_participating`；alignment 统计**只算参与组**（否则 `{1024,256}` ⇒ `alignment_tokens=None`，退化成"全存 SWA"） |
| 3 | `OffloadingConnectorScheduler.__init__` | `_lookup_groups` / `_sliding_window_groups` **只收参与组**（★ 核心修复） |
| 4 | `RequestOffloadState.update_offload_keys()` | 非参与组**不生成 key**（从源头消除"永远查不中"的 key） |
| 5 | `update_state_after_alloc()` | 非参与组 `group_sizes=0 / block_indices=0` 作为**占位**保留（worker 侧断言 `len(group_sizes) == 组数`；`cpu_npu.py` 对 `group_size == 0` 直接 `continue`，src/dst 偏移对齐不受影响） |
| 6 | `_build_store_jobs()` | 非参与组不参与 store；第二段循环里同样 `group_sizes=0/block_indices=0` 占位（保持与 13 个组一一对应） |
| 7 | 诊断 | group 清单 / 参与组清单 / 逐组 store 记账 / 每次 lookup 的逐组命中结果 / load job 明细（全部 `[D2_offload]` 前缀，进 `serve.log`） |

**为什么"跳过 state 组"是正确语义而不是打补丁**：见 §2.4/§2.5 —— GPU 侧前缀缓存路径
本来就跳它（`verify_and_split_kv_cache_groups`），把它排除在卸载命中判定之外，
只是让卸载层与 GPU 层**语义一致**。

**回滚方式**：`shadow-pkg/patches/files/offload_dsv41/scheduler.py` 的上一版已备份为
`scheduler.py.D_off8.bak`；或把 `OFFLOAD_SCHED_PATCH=0`（默认值）即可完全不挂这个补丁。

---

## 5. 修复后实测

三条臂，**探针口径、prompt 生成方式、HBM 池（1 GiB / 46,387 token）、每请求 32768 token
全部一致**，只差请求数与池子：

| 观测 | 修复前（`001` 臂 D，16 请求） | 只修 state 组（`d2-dram32`，16 请求） | **两条都修（`d2-dram32-2p`，2 请求）** |
|---|---|---|---|
| `BlockStored(medium="CPU")` | 12,288 | 12,288 | **768** |
| `BlockRemoved(medium="CPU")` | 8,192 | 8,192 | **0**（一轮就装下了 ⇒ 无淘汰） |
| `kv_offload_total_bytes_total{GPU_to_CPU}` | 393,568,321,536 B | 393,568,321,536 B | 24,598,020,096 B |
| **`kv_offload_total_bytes_total{CPU_to_GPU}`** | **0.0** | **0.0** | **1,555,480,576 B** |
| `kv_offload_cpu_allocation_size_sum`（CPU 块/chunk） | 12,288 | 12,288 | 768 |
| `kv_offload_allocation_failure_total` | 0 | 0 | **0** |
| `external_prefix_cache_queries_total` | 1,048,832 | 1,048,832 | 131,328 |
| **`external_prefix_cache_hits_total`** | **0** | **0** | **63,488** |
| fill TTFT p50 | 4192.5 ms | 4201.6 ms | **4424.6 ms** |
| **replay TTFT p50** | **4167.4 ms**（全量重算） | **4181.0 ms**（全量重算） | **298.2 ms（14.8×）** |
| replay 轮 wall | —（33.8 s/请求） | 66.96 s（16 请求） | **0.598 s（2 请求）** |

**命中率的两种算法（都不含糊）**：全 bench 口径 `63,488/131,328 = 48.3%`
（fill 轮本来就该 0 命中，占了一半 query）；**replay 轮口径 `63,488/65,536 = 96.9%`**
——缺的 2,048 token = 2 请求 × 1 chunk，正是 MTP/draft 组"尾 chunk 必须重算"
（`is_eagle_group` 的 `num_hit_chunks -= 1` 语义）。

**"取回真的发生了"的独立证据**（不只看计数器）：

```
[D2_offload] load job req=cmpl-b9c3c71bb85e63d3-0-8f328dc0 keys=42
              group_sizes=[248, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
              src_blocks=42 dst_blocks=259        ← replay 轮真的发出 2 个 load job（2 请求各一）
[D2_offload] miss-scan req=... group=0 scanned=32 present=0 head_key=4989c9330d8d569f
                                                  ← 只有 fill 轮的 2 次 lookup 未命中（本来就不该命中）
```

### 5.1 三条副产品证据（各自独立成立）

1. `group_sizes[1] = 0`：**state 组不再参与取回**，但在 load spec 里保留占位 ⇒
   worker 侧 `len(group_sizes) == 13` 的断言满足、`if group_size == 0: continue` 跳过
   ⇒ src/dst 偏移不错位（`src_blocks=42 / dst_blocks=259` 与 `group_sizes` 自洽）；
2. `group_sizes[2..12] = 1`：SWA 组的 `sliding_window=128` token ⇒
   `sliding_window_size_in_chunks = cdiv(128,1024) = 1` ⇒ 只要 1 个连续命中
   （这部分**本来就工作**，与 `001` §4.3 的判断一致）；
3. `store-key` 日志显示 **group 0 与 group 2 的 chunk_0 哈希完全相同**
   （都是 `4989c9330d8d569f`）⇒ 上游确实是"同一个 block_hash 复用给每个组、
   只用 4 字节 `group_idx` 后缀区分"（`make_offload_key`）——
   这正是 state 组能"一票否决"的机制基础。

### 5.2 调度侧第一手清单（`serve.log` 的 `[D2_offload]` 行）

```
[D2_offload] KV 卸载 group 清单 n=13: [(0,'DeepseekV41FullSpec',128,8,...,True),
   (1,'DeepseekV41CompressorStateSpec',32,3,...,False), (2..11,'DeepseekV41SWASpec',128,4,...,True),
   (12,'DeepseekV41DraftSWASpec',128,3,...,True)]
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
```

末位 `True/False` 是本补丁新加的 `offload_participating`：**只有 group 1（state）是 False**，
与 §2 的推断完全一致。

### 5.3 失败的中间臂（不算通过，但对判据有价值）

| 臂 | 配置 | 结果 |
|---|---|---|
| `d2-dram32` | 16 请求 + 32 GiB 池 | 【实测】**仍然零取回**（`CPU_to_GPU=0`、`hits=0`）⇒ 定位到 §3.1 的池子/LRU 根因 |
| `d2-dram128-6p` | 6 请求 + **128 GiB 池** | 【实测】**起服失败**：`aclrtMallocHostWithCfg 207001`，单次请求 **16 GiB** pinned（§3.2） |
| `d2-dram64-4p` | 4 请求 + **64 GiB 池** | 【实测】**起服失败**：同样 207001，单次请求 **8 GiB** pinned ⇒ 上限落在 (4, 8] GiB/worker |

---

## 6. 未确认 / 风险 / 下一步

| 项 | 状态 |
|---|---|
| `state` 组不参与卸载的**数值正确性** | 【推断】与 GPU 前缀缓存路径同语义（§2.4）；**本次没跑精度/一致性验证**（GSM8K 或 logprob 对比），只跑了 TTFT/命中 |
| 池容量 | **已确认是关键变量**（§3.1 + §5）：池子 ≥ 一轮工作集 ⇒ 命中 96.9%；池子 < 一轮 ⇒ **0%**（全有或全无）。**32 GiB 只装得下 ~963 个 chunk ≈ 2.5 个 32768-token 请求** |
| pinned 上限的**归属** | 【未确认】：(4, 8] GiB/worker 这个窗口是在**多租户共用**的宿主上量的（NPU 0/1/2 上有别人的 `VLLMEngineCor`，还有 `dsv4-offload-serve-*` 常驻容器）；到底是"进程内 pinned 预算""驱动池全局上限"还是"被邻居吃掉一部分"**没有分离**。要分离只需一条：把 8 卡空着的窗口里，用单进程逐步 pin 到失败（`pin_host_probe.py`），但本轮没做 |
| `--prefix-match-unit 32` 的精度/性能代价 | 【未确认】：沿用前任结论（`001` §4.1 的绕过），未量化 |
| 上游修法（建议） | ① `SchedulerOffloadConfig.from_spec` 里就把**非前缀缓存组**排除在命中判定与 store 之外（本补丁的做法）；② 更省事的上游形态：`_lookup()` 里对"不可缓存组"不要返回 0（返回 `None`/跳过），别让它一票否决；③ `build_offloading_config()` 的整除断言应只作用于可缓存组（否则用户仍必须手写 `--prefix-match-unit`）；④ **容量**：CPU 池的 `num_blocks` 按 (group, chunk) 计费，对多组模型等于把池子除以"参与组数"——建议按**唯一 chunk 位置**计费，或至少在启动日志里打印"本配置能覆盖多少 token" |
| A2（`host_mem_pool=0`） | 本次**没有**碰 A2。本修复解决的是"取回"，**不解决** pinned 上限；而 §3.2 说明该上限**就是 A2 的头号风险**（A2 连 `host_mem_pool` 都没有）。A2 的探测（`001` §5.1 的 ①②③④）仍待用户粘贴执行 |
| 16 请求 + 32 GiB 的"生产口径"通过臂 | 【未确认/未做】：要让 16 请求一轮（6,144 chunk ≈ 220 GB 池）跑通，必须先解决 §3.2 的 pinned 上限（换分配形态：`mmap + aclrtHostRegister(MAPPED)`，或分片申请），本轮时间不够 |

---

## 7. 产物清单（本机 `a2/`）

| 文件 | 作用 |
|---|---|
| `a2/agents/D2_offload/scripts/make_offload_scheduler_patch_d2.py` | 补丁生成器（带锚点唯一性校验 + `compile()` 自检 + 逐处改动清单） |
| `a2/agents/D2_offload/scripts/run_arm_d2.sh` | 臂运行器包装（c0 锁 + 产物落 D2 目录） |
| `a2/agents/D2_offload/patches/offload_dsv41/scheduler.py` | 生成的补丁文件（A3 的 `shadow-pkg/patches/files/offload_dsv41/scheduler.py` 是同一份，md5 一致） |
| `a2/logs/raw/009-d2-dram32-2p.*` | ★ **通过臂**的原始数据（client.json / metrics_before+after / kv_events / meta / `D2_lines.txt` = `[D2_offload]` 全部日志行） |
| `a2/logs/raw/009-d2-dram32.*` | 只修 state 组、16 请求、仍然零取回的臂（用于定位 §3.1） |
| `a2/logs/raw/009-arm-d2-{dram32,dram32-2p,dram64-4p,dram128-6p}.out` | 4 条臂的完整运行日志（含 64/128 GiB 的起服失败全过程） |
| `a2/logs/raw/009-d2-dram128-6p.keylines.txt` | `207001` 失败的关键行（`Failed to allocate 17179869184 bytes` / `2147483648 bytes`） |
| `a2/logs/raw/001-off8-*` | 前任 23 个原始文件的副本（按任务书要求复制，**原件未动**） |

---

## 8. 红线遵守

* 新产物只落 `a2/`（本机）与 `agents/D2_offload/`（A3）；**没有写 `upstream-v41/`**（只读，复制了 001 的原始数据到 `a2/logs/raw/`）；
* 不用 `/tmp`（本机 `source a2/scripts/tmpdir.sh d2_offload` → `~/tmp/20260921/d2_offload`；A3 侧只读地做了 `docker create/cp` 抽取源码到 `~/tmp/20260921/d2_offload/`）；
* **没有**手设 `ASCEND_RT_VISIBLE_DEVICES`；**没有**碰 `dsv41-a3`（保持 `Exited`）/ `mooncake-master` / 别人的容器；
* 8 卡臂全程持 `locks/c0.lock`（`flock`，owner 文件已写/已清）；起跑前 `npu-smi` 确认 8–15 无进程；
* ssh 一律 `-o ControlPath=none`；跨机传文件走 `cos-xfer.sh`（不走 ssh 管道）；
* 结论全部标了【实测】/【推断】/【未确认】；没有用相邻数字顶替缺的那格。

**收尾状态（【实测】）**：`docker ps -a` 里**无 `abl-off-*` 残留**；`npu-smi` 显示
**Phy-ID 8–15 无进程**；`locks/*.owner` **无残留**；`dsv41-a3` = `Exited`（未动）；
宿主 `MemAvailable` = **858 GiB**（安全线 150 GB 之上）。

**一处需要报备的写入（A3 侧，不在 `a2/` 内）**：为了让 `OFFLOAD_SCHED_PATCH=1` 走到修复版，
我把生成的补丁覆盖到了 `~/projects/dsv41-upstream-pr/shadow-pkg/patches/files/offload_dsv41/scheduler.py`
（**D_off8 那一版已先备份为同目录的 `scheduler.py.D_off8.bak`**，可随时回滚；
`OFFLOAD_SCHED_PATCH=0`（默认）时该文件根本不会被挂载）。
这属于"跑实验必须占用的运行环境"，不是往 `upstream-v41/` 写产物。
