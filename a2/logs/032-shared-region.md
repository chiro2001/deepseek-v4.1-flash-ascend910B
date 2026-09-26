# 032 · L6「8 份副本合成 1 份」：**上游两道门都关着**，我们把它打开（V1/V2/V3 + 8 卡实测）

**日期**：2026-09-22 02:2x – （进行中；简版先交）
**执行**：子代理 `P3_sharedregion`（任务书 `/root/p3_sharedregion`）
**机器**：A3（A3-node1），单卡 **c2**（= Phy-ID 7）；8 卡臂要等 **c0**（Phy-ID 8–15）空闲
**上游**：镜像内 vLLM `0.27.1` + vllm-ascend（与 `013`/`016`/`021` 同一套）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/算式推出来但没直接测；【未确认】= 没跑到
**产物**：`agents/P3_sharedregion/`（`patch/build_patch.py` 生成器 + `patched/` 补丁版 + `merged/` 与
P2/L3 的 per-group bpc 合并版 + `scripts/` 运行器）；原始数据 → `logs/raw/032-*`

---

## 0. 一句话结论（**简版**，随实测更新）

| # | 问题 | 结论 | 强度 |
|---|---|---|---|
| **V1** | V4.1 在 TP8 下 8 个 rank 的 KV 是不是逐字节复制 | **代码级：是（强）**；**端到端字节级：【未确认】**（没拿到 8 卡窗口）。**残余风险已定位到 MoE/TP 集合通信的浮点归约顺序** | 【推断·代码】+【未确认】 |
| **V2** | 单 tier 路径能不能用 `SharedOffloadRegion` | ✅ **能**：不换 spec、不引入 SSD；backport = **4 个文件、~100 行**，**单卡实测跑通**（§6.2/6.3） | 【实测】 |
| **V3** | 8 个 worker 往同一份 region 写会不会互相踩 | **【未确认】**（需要 TP≥2 的 dump；分析见 §四） | 【未确认】 |
| **8×** | 宿主实占是否降 ~8× | ❌ **【未拿到】**。单卡 A/B 拿到的是 **6.97×**，而且那是**记账口径（L2）**的功劳，**不是 L6** | 【未确认】 |

### ★ 一句话答复（主代理要的那句）

> **L6 在 V4.1 上"能用但要改上游、且拿不到干净的 8×"**：
> ① 上游 `replicated_layout` 的认证是 **fail-closed 的"单组 + 裸 MLAAttentionSpec"**，
> V4.1 的 **12 组 hybrid** 永远过不了 ⇒ **必须改上游 `offloading/config.py`**（不是配置能解决的）；
> ② 上游同时隐含要求"**单一页几何**"（`worker_kv_bytes_per_block == page × layers`），
> 而 V4.1 是 **16 张张量、Σ page = 910,208 B**（≠ 131,072 B）⇒ **口径结构性冲突**，
> 打开 gate 后收益也要**按我们自己的 Σ page × bpc 重算**；
> ③ **8× 的语义前提（8 个 rank 的 KV 逐字节相同）没有端到端证据**，
> 且它在 TP 下**不是自动成立**的（MoE/TP 集合通信的归约顺序）；
> ⇒ **建议：暂不把 8× 计入 A2 的容量账（`031` §三 请删掉这一行）**，
> 先花**一个 8 卡窗口**做 §五 的那条 dump 命令**一锤定音**；
> 定音之前，A2 的容量账只认 **L1（P2）+ L2（记账，本任务实测 6.97× 已确认）+ L5**。

---

## 一、★ 上游有**两道门**都关着（主代理 02:3x 的补充修正了 `031` 的因果）

`logs/031` §一把 `replicated_layout` 被禁**全部归因**于 `NPUOffloadingSpec` 那段
"Ascend 没有 cudaHostRegister 等价物"的注释。**这个归因不完整**，实测有两道门：

| 门 | 位置 | 条件 | V4.1 满足？ |
|---|---|---|---|
| **门 1**（上游 / 平台无关） | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:113-140` | `model_config.use_mla` **且** `type(single_group_spec) is MLAAttentionSpec`（注释：*Exact type: fail closed on wrappers and sliding-window variants*）**且** `worker_kv_bytes_per_block == page_size_bytes × len(layer_names)` **且** TP>1/PP=PCP=DCP=1/world==tp/backend=="mp"/nnodes_within_dp==1 | ✗ **`single_group_spec` 恒为 None**（DSV4.1 是 **13 个 group**）⇒ 门 1 先关 |
| **门 2**（Ascend 侧） | `vllm_ascend/.../native/npu.py::NPUOffloadingSpec` + 上游 `vllm/v1/kv_offload/cpu/spec.py:107` | `config.replicated_layout and self._uses_shared_region()`；Ascend 单 tier **没有覆写** `_uses_shared_region()` ⇒ 落到 `current_platform.is_cuda_alike()` ⇒ **NPU 上 False** | ✗ |

**⇒ 只改门 2（让 `_uses_shared_region()` 返回 True）拿不到 8×**：`config.replicated_layout` 本来就是
`False`。**必须同时改门 1**（我这次的做法）。这一点已写进 `patch/build_patch.py` 的注释。

### 1.1 ★ 逐条标注我们哪一条不满足（主代理要求，原文抄录）

```python
worker_kv_bytes_per_block = 0
if (kv_cache_config.hisparse_host_num_blocks is None and ...):
    # Scratch filtering must preserve the scheduler/worker allocation stride.
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total, not a per-tensor share.
    total_gpu_kv_bytes = kv_cache_config.kv_cache_tensors[0].size
    worker_kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks   # ← ★ 我们这条错
elif ...
single_group_spec = (kv_cache_config.kv_cache_groups[0].kv_cache_spec
                     if len(kv_cache_config.kv_cache_groups) == 1 else None)       # ← ★ None
replicated_layout = (
    vllm_config.model_config.use_mla
    # Exact type: fail closed on wrappers and sliding-window variants.
    and type(single_group_spec) is MLAAttentionSpec                            # ← ★ False
    # Page accounting: one MLA page per layer, no packed/mixed rows.
    and worker_kv_bytes_per_block > 0
    and worker_kv_bytes_per_block == single_group_spec.page_size_bytes
        * len(kv_cache_config.kv_cache_groups[0].layer_names)                   # ← ★ False（见下）
    # Safe MVP boundary: TP-only, no other parallel axes.
    and parallel_config.tensor_parallel_size > 1            # ✓ 8
    and parallel_config.pipeline_parallel_size == 1         # ✓
    and parallel_config.prefill_context_parallel_size == 1  # ✓
    and parallel_config.decode_context_parallel_size == 1   # ✓
    and parallel_config.world_size == parallel_config.tensor_parallel_size  # ✓
    # Shared /dev/shm mmap layout is single-node mp only.
    and parallel_config.distributed_executor_backend == "mp"   # ✓
    and parallel_config.nnodes_within_dp == 1                  # ✓
)
```

* **`worker_kv_bytes_per_block`（记账口径）在 DSV4.1 上是错的**：它取
  `kv_cache_tensors[0].size // num_blocks` = **第一个 slot 的 page = 131,072 B**，
 而 canonical 视图有 **16 张**、Σ page = **910,208 B = 它的 6.9443×**
  （`logs/016` §1.3 / `logs/021` §4.4 的 ×6.945，A2 真机与 A3 tiny **逐字节相同**）。
  ⇒ `logs/029` §一的"记账低估 6.944×"在**调度层**也有一条独立成因（同一段代码）。
  ★ **共享 region 的每一行必须装下所有 canonical 张量**，所以这个数必须**先改对**，
 否则 region 会按 131,072 B 建，装不下 910,208 B（**会直接内存越界/断言**）。

   ★★ **主代理要求的机制表述（这一条是"L6 即使打开也不是简单 8×"的根本原因）**：
   > 上游这条假设"**KV 是单一 backing allocation、每个 tensor 的 size 是总量**"
   > （原文注释：*"Every KVCacheTensor describes placement within the same backing allocation,
   > so its size is the total, not a per-tensor share."*）；
   > 而 DSV4.1 是 **hybrid 页几何**（16 张张量、`Σ page = 910,208 B` ≠
   > `worker_kv_bytes_per_block = 131,072 B`）⇒ **上游 `replicated_layout` 的认证
   > 隐含要求"单一页几何"**，这与我们的 hybrid 布局**结构性冲突**。
   > 所以即使把 gate 打开，**收益口径也要按我们自己的 `Σ page × bpc` 重算**
   > （P2 在 `logs/030` 里已给出正确口径）。
* `is_packed` / `total_gpu_kv_bytes` 那两条注释讲的是 **packed stride**；DSV4.1 的
  `KVCacheTensor.block_stride = slot.page_size_bytes`（`core/deepseek_v41.py:273-279`）⇒ `is_packed=True`，
  所以走的是 `kv_cache_tensors[0].size`，**"size is the total"** 这条假设在 V4.1 的
  4-slot 布局下**不成立**（每个 tensor 是**一个 slot**，不是全局总量）。

---

## 二、V1：8 个 rank 的 KV 是否逐字节相同（代码级 【实测·代码】；端到端待 8 卡 dump）

### 2.1 判据链（每条都是镜像内真代码）

| # | 事实 | 出处 |
|---|---|---|
| 1 | **long-KV 平面是 MLA latent**（`head_size=512`, BF16, `num_kv_heads=1`）⇒ **TP 下不按头切分** | `docs/KV-CACHE-ACCOUNTING.md`；`logs/002` §5.1（4421 B/token/rank） |
| 2 | **indexer 的 K**：`self.wk = nn.Linear(head_dim, width)` + `k_cache`（INT8+scale）。`wk` 是 **`nn.Linear` 不是 Column/RowParallel**；类注释原文：*"All index heads are replicated on each TP rank for the correctness path, so every rank produces identical sparse indices without an all-reduce."* | `models/deepseek_v41/indexer.py:28-29, 67-76` |
| 3 | **compressor 的状态**：`self.wkv = nn.Linear(...)`、`self.wgate = nn.Linear(...)` —— 同样 replicated | `models/deepseek_v41/compressor.py:68-71` |
| 4 | **DSpark draft SWA**：`main_proj = ColumnParallelLinear`，forward 里 `heads_per_rank = num_attention_heads // tp_size` —— 但 draft SWA 的 **KV cache 是 `DeepseekV41DraftSWASpec`（BF16 单平面）**，写进去的是 **latent**（不是 q/k 投影结果） | `models/deepseek_v41/dspark.py:99, 438-440`；`core/deepseek_v41.py:50-64` |
| 5 | **slot 分配是全局的**：`plan_cache_slots()` 把 KV+index 放进 **4 个共享 slot**，`allocate_cache_config()` 建 4 个 `KVCacheTensor`（`block_stride=slot.page_size_bytes`），调度器给的是**全局 block ID** ⇒ 8 个 rank 看到**同一张 block table** | `core/deepseek_v41.py:137-201, 266-280` |
| 6 | **上游对 pure MLA 的认证**：*"enable its single-copy layout for configurations it has certified as byte-replicated (currently pure MLA under the supported TP topology)"* | `vllm-ascend/.../native/npu.py::NPUTieringOffloadingSpec`（main 分支） |

⇒ **【实测·代码】V1 = 是**：DSV4.1 的 13 个组全是"replicated 权重算出来的 latent / replicated indexer 输出"，
**没有任何一个组按 head 切分**，且 slot 映射全局一致。**但这是代码级**——
端到端证据（同一台机 TP8、同一 slot 取 8 个 rank 的哈希）见 §五。

### 2.2 端到端判据（本任务实现的 dump 钩子）

`patched/asc_cpu_npu.py` 末尾追加了 `P3_DUMP_DIR` 门控的 dump watcher：
服务**空闲**时（trigger 文件出现）把每个 rank 的

* **GPU 侧**：第一笔 / 最后一笔 store job 的 block 号在**每个 canonical 张量**里的行（逐字节 → sha256）
* **池子侧**：同一批 unit 号的行（逐字节 → sha256）

落成 `v1-rank%02d-<tag>.json`。判定脚本 `scripts/analyze_dump.py` 一次算完两件事：

| 判据 | 比什么 | 含义 |
|---|---|---|
| **V1** | 8 个 rank 的 **GPU 行**哈希是否逐张量相同 | KV 是否逐字节复制 |
| **V3** | 8 个 rank 的 **池子行**哈希是否相同，且 **== GPU 行哈希** | 共享 region 有没有互相踩 |

---

## 三、V2：backport 面有多大（**结论：4 个文件、~90 行，不换 spec、不引入 SSD**）

### 3.1 为什么**不能**直接用上游的 `SharedOffloadRegion` 单 region 布局（★ 必须记下来）

上游 `SharedOffloadRegion` 的**行布局**是"每个 chunk 一行，行内 [worker0 区 | worker1 区 | …]"，
行宽 `_row_stride = kv_bytes_per_block`；`create_next_view(tensor_page_size)` 从**行内偏移 0** 开始
依次切 view。⇒ **一行里要装下同一 worker 的全部 canonical 张量**，要求
`row_stride ≥ Σ canonical page`（= **910,208 B**）。

而单 tier 路径里 `spec.py:161-167` 传的是 `kv_bytes_per_block = self.kv_bytes_per_chunk`，
它由 **`config.worker_kv_bytes_per_block`**（上游口径 = **131,072**）乘出来 ⇒
**行里只有 1/6.944 的空间**，第二个张量就会触发
`assert new_offset <= self._worker_area_end`（我第一版就撞在这里，见 §六 修正记录）。

### 3.2 我的做法：**每个 canonical 张量一块独立 region**（最小、最稳）

```
region[t] : engine_id = <engine_id>.t<t>,  rank = 0（replicated）
            num_blocks = spec.num_blocks
            kv_bytes_per_block = round_up(page_t × bpc × num_copies, 4096)   ← 行宽
            cpu_page_size      = page_t × bpc
            worker 侧：mmap_regions[t].create_next_view(page_t × bpc)
```

* 上游的"每 worker 一个 slot"语义原样保留（`rank=0` 时全部 worker 抢同一个 slot，
  但**写的是同样的字节**，见 §2）；
* **不碰上游 `SharedOffloadRegion.create_next_view()` 的偏移算术**（少一处风险）；
* 16 块 region 相加 = `num_blocks × Σ row`，与"一份副本"的账一一对应。

### 3.3 改动清单（4 个文件；`patch/build_patch.py` 可复现）

| 文件 | 改动 | 行数 |
|---|---|---:|
| `vllm/distributed/.../offloading/config.py` | ① `P3_SHARED_REGION=1` 时把 `worker_kv_bytes_per_block` 换成 env 传入的**真实 Σ**（默认不换）② `replicated_layout` 增加一条 **env 认证**的 or 分支（拓扑护栏照抄上游） | +45 |
| `vllm_ascend/.../native/npu.py` | 覆写 `_uses_shared_region()`（env 门控，默认 False）；`create_worker()` 建 16 块 region 并传给 worker | +58 |
| `vllm_ascend/.../native/cpu_npu.py` | `NPUOffloadingWorker.__init__(..., mmap_regions=None)`；有 region 时用 `create_next_view()` 代替 `torch.zeros(pin_memory=True)`；+ V1/V3 dump 钩子 | +40（+~140 dump） |
| `vllm/v1/kv_offload/cpu/shared_offload_region.py` | `P3_REGION_DIR` 可把 region 文件从 `/dev/shm` 换到别的目录（**容器 `/dev/shm` 只有 64 MiB**，见 `logs/014` §3.4；8 卡容器是 `--shm-size=512g`，单卡不是） | +6 |

**全部 env 门控、默认关**（`P3_SHARED_REGION=0` 时逐字现状）⇒ 向后兼容。
**不换 spec**（仍是 `NPUOffloadingSpec`）、**不引入 SSD / tiering**。

### 3.4 与邻居补丁共存

| 邻居 | 冲突点 | 处置 |
|---|---|---|
| **P1（`cpu_npu.py` registered 池）** | **同一个挂载点** | 二选一：P3 版自带 mmap + 池日志 ⇒ 8 卡臂把 `OFFLOAD_NPU_WORKER_PATCH=0` |
| **P2/L3（per-group bpc：`cpu_spec.py`+`pgp_manager.py`**） | `cpu_spec.py` 与我无关（我不用它）；**`config.py` 撞** | `scripts/build_merged.py` 把 P3 的两条改动**叠在 L3 的 config 底稿上** ⇒ `merged/offloading_config.py`（`L3_PGP_DIR` 指到 `merged/`） |
| `logs/021` per-group bpc 语义 | 我只改"池子从哪来"，不改 bpc 解析 | 不冲突 |

---

## 四、V3：写路径幂等性（**待 8 卡 dump**；分析先写在这）

写路径的**互踩只可能发生在"同一行被两个 worker 写不同字节"**这一种情形。逐层拆：

1. `submit_store(job_id, src_spec, dst_spec)` 的 `dst_spec` 是**调度器**发的
   `BlockIDsLoadStoreSpec`（CPU 侧 unit 号）⇒ **8 个 worker 拿到同一串 unit 号**
   （引擎里只有一份 KV 事件流 / 一份 block 表）；
2. 每个 worker 把这些 unit 号映射到**自己进程里**的 `cpu_tensors[t]` 行（`rank=0` 后是同一份物理内存）；
3. ⇒ 只有"同一 unit 号、同一张量、同一时刻被两个 rank 写**不同**内容"才会踩；而 V1 成立 ⇒ 内容相同 ⇒ **幂等**；
4. ★ **per-group bpc 补丁（`logs/021`）破坏这个前提了吗？** 不破坏：它改的是
   **unit → (group, chunk) 的粒度**（SWA=1 unit、full=8 unit），改的是"**哪些** unit 号被分配"，
   而**分配本身仍是全局唯一的**（`PerGroupBPCManager` 一份，调度侧单实例）⇒ 8 个 rank 依然写同样的字节。
   ⚠️ 但**若两个 rank 的 `blocks_per_chunk` 语义不一致**（例如 per-group map 在两侧解析不同），
   `src_offset/dst_offset` 的推进就会错位 ⇒ **这才是真正的风险点**，`logs/021` 已经把关：
   两侧都调用同一个 `build_offloading_config()`，解析结果写进 `extra_config` 后**两侧逐字一致**。
5. **判据**：dump 里 `池子行哈希 == GPU 行哈希`（8/8 rank）⇒ 无踩。

---

## 四·五、★ 主代理的三个是非题（Q1/Q2/Q3）—— **30 分钟内给答案**

### Q1：V4.1 的 12 组几何，**从语义上**是否满足"每个 rank 的 canonical KV 逐字节相同"？

**答：是。** 三条机制级论证，每条都钉在镜像内代码上：

**① canonical 页的**形状里压根没有 TP 维**（最强的一条）**
```python
# vllm_ascend/core/deepseek_v41.py:295-310  reshape_cache(…)
torch.as_strided(raw.view(dtype),
    size=(num_blocks, spec.storage_block_size, spec.num_kv_heads, width),
    stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1), …)
# vllm_ascend/core/kv_cache_interface.py:57-63
def real_page_size_bytes(self):
    return self.storage_block_size * self.num_kv_heads * (
        self.head_size * get_dtype_size(self.dtype)
        + self.scale_dim * get_dtype_size(self.scale_dtype))
```
四个维度是 `[blocks, tokens_per_page, num_kv_heads, head_size]`，**唯一的"头"维是 `num_kv_heads`，
而 V4.1 的它恒为 1**（`DeepseekV41DraftSWASpec.__post_init__` 里那句
`if … or self.num_kv_heads != 1: raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")`
就是这个不变量）。**页里没有 `tp_rank` 的位置，也没有 `num_kv_heads/tp_size` 的除法**
⇒ 结构上**不可能**按 rank 切分。反证：TP 下被切的是 **Q 头**（`n_local_heads = num_attention_heads // tp_size`，
`models/deepseek_v4/model.py:482-485`），那是 **query**，不进 KV cache。

**② 写进这 4 个 slot 的全部是"replicated 权重算出来的 latent"**

| 组 | 写进页的数据 | 产出它的权重是否 TP 切分 | 证据 |
|---|---|---|---|
| `full`（0） | `long_kv_cache` ← **latent**（512 维 BF16） | ✗ **`nn.Linear`（replicated）** | `attention/dsa_v41.py`：`scatter_cache_sk(attn.long_kv_cache.kv_cache[0], long_slots, latent)`；latent 出自 `compressor`（`nn.Linear`、`nn.Linear`，`models/deepseek_v41/compressor.py:68-71`） |
| `full`（0）的 indexer 平面 | `indexer.k_cache` ← `torch_npu.npu_dynamic_quant(attn.wk(latent))` | ✗ **`nn.Linear`（replicated）** | `models/deepseek_v41/indexer.py:67-76`；类注释：*"All index heads are replicated on each TP rank for the correctness path, so every rank produces identical sparse indices without an all-reduce"* |
| `state`（1，**不参与卸载**） | FP32 32 行环 | ✗ replicated | `DeepseekV41CompressorStateSpec`（`core/deepseek_v41.py:67-77`） |
| `swa0..9`（2–11） | SWA latent（窗口 128） | ✗ replicated（同上 latent） | `DeepseekV41SWASpec`；slot 别名机制见 ③ |
| `dspark`（12） | draft SWA latent（BF16 单平面） | ✗ replicated（**cache 内容仍是 latent**） | `DeepseekV41DraftSWASpec.__post_init__` 强制 `num_kv_heads == 1` |

> **TP 下"权重切了、激活没切"**：TP 里每个 rank 算部分输出再 all-reduce，**all-reduce 的结果在每个 rank 上逐位相同**
> （ring/tree 归约是同一串加法），所以**replicated 权重吃到的输入也逐位相同** ⇒ 输出（latent）逐位相同。

**③ slot 映射是全局唯一的**（不是每 rank 一套）
`plan_cache_slots()` 把 `long_kv_cache` 与 `indexer.k_cache` 放进**同一个 slot**（KV 在偏移 0、index 紧随其后），
`allocate_cache_config()` 建 **4 个 `KVCacheTensor`**（`block_stride = slot.page_size_bytes`），
**block ID 池是全局的一个**（`may_override_num_blocks` 只决定块数）⇒ 8 个 rank 拿到的 slot 号**完全一致**。

**⇒ Q1 = 是**：语义上逐字节相同。**唯一的残余不确定性**是"有没有哪条代码路径把 rank-specific 东西写进 KV"，
这只能实测（§五 的 dump 就是干这个的）。

### Q2：最小改动集（**确切行号 + 改动**）

| # | 文件 | 行（镜像内） | 现在 | 改成 | 我的补丁 |
|---|---|---|---|---|---|
| ①a | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py` | **113-140** | `replicated_layout` 要求 `len(groups)==1 and type(spec) is MLAAttentionSpec` | 加一条 **env 认证**的 `or` 分支（拓扑护栏照抄原样） | `patched/offloading_config.py:172-192` |
| ①b | 同上 | **111** | `worker_kv_bytes_per_block = kv_cache_tensors[0].size // num_blocks`（= 131,072，**低估 6.944×**） | env 传入真实 Σ（**910,208**，或对齐后的行距） | `patched/offloading_config.py:140-170` |
| ② | `vllm/v1/kv_offload/cpu/spec.py` | **90** | `self.replicated_layout = config.replicated_layout and self._uses_shared_region()` | **不用改**（吃 ①a） | — |
| ②′ | 同上 | **92-114** | `num_copies = 1 if replicated else world_size`；`num_blocks = cpu_bytes_to_use // round_up(block×bpc×copies, ALIGN)` | **不用改**（吃 ①a/①b） | — |
| ③a | `vllm_ascend/.../native/npu.py` | **74-82**（`create_worker`） | 直接 `NPUOffloadingWorker(...)`，**不建 region** | env 打开时建 region 并传下去 | `patched/asc_npu.py:88-146` |
| ③b | 同上 | 新增 | 无 `_uses_shared_region` | 覆写为 `env`（默认 False） | `patched/asc_npu.py:62-70` |
| ③c | `vllm_ascend/.../native/cpu_npu.py` | **283-305**（`__init__` 的 `torch.zeros(…)`） | 每张 canonical 张量**自己 pin 一块**（8 份副本的根源） | 有 region 时改 `region.create_next_view()` | `patched/asc_cpu_npu.py:273-330` |
| ④ | `vllm/v1/kv_offload/cpu/shared_offload_region.py` | **110** | `mmap_path = /dev/shm/…`（**容器只有 64 MiB**） | env 换目录 | `patched/shared_offload_region.py` |

★ **注意 ①b 与 ③c 是一对**：**不改 ①b 就必须给每张张量单独建 region**（因为上游那套
"一行装下所有 canonical 张量"要求 `row_stride ≥ Σ page`，而 131,072 < 910,208）。
**我两条都做了**（既改记账、又每张量一 region），所以不依赖上游的行内布局假设。

### Q3：`/dev/shm` 之外的文件 mmap，**8 个跨进程 `MAP_SHARED` + NPU DMA 是否成立**？

**答：成立，且三段都有独立证据。**

| 环节 | 证据 | 强度 |
|---|---|---|
| **跨进程 `MAP_SHARED` 共享** | 文件 mmap 的语义（同一 inode 的页缓存被所有进程共享）与文件系统无关；`/dev/shm` 只是 tmpfs ⇒ 换成盘上文件**不改变共享语义**（`P3_REGION_DIR` 已实测跑通：日志 `Created mmap file /work/…/vllm_offload_….t0.mmap`） | 【实测】（单卡跑通）+【推断】（8 进程） |
| **盘上文件映射能否被 `aclrtHostRegister` 吃** | `logs/014` §4.1 最后两行：**磁盘文件 `MAP_SHARED` + 预热页 1 / 8 GiB → `ret=0`**（0.0 / 0.2 s）；而未预热页的 VMA 会 `ret=107017` | 【实测】 |
| **注册后 DMA 的带宽/正确性** | `logs/014` §4.2：`registered` 后端 **1 GiB ×3 / 8 GiB ×2 全部逐字节一致**，H2D **56.5–58.3 GB/s** / D2H **42.0–42.7 GB/s**（与 pinned 同级）；裸 `acl.rt.memcpy_async` 的 16 / 24 GiB 大块 D2H 也**逐位一致**（§4.2 后半） | 【实测】 |

★ **但 A2 上有一条本项目特有的纠正**：`logs/014` §4.1 还实测 **`/dev/shm` 文件映射在容器里直接 SIGBUS**
（64 MiB 上限），而 **盘上文件映射 + 预热页是可以注册的**。
我这一版**没有单独调 `aclrtHostRegister`**：`P3_REGION_DIR` 指向盘上文件后走的是
**pageable 映射**，而 `logs/014` §4.2 的对照表显示 **`pageable`（不注册）在 torch 拷贝路径上就已经
逐字节一致且带宽 56.3–58.5 / 42.1–42.7 GB/s**。⇒ 单副本臂**不靠注册**也能跑；
要拿"注册态"的稳定 DMA 语义，就把 `pin_mmap_region()` 换成 `aclrtHostRegister`
（P1 的 `cpu_npu.py` 补丁里已有这个 helper，可共用）。

---

## 五、8 卡实测：**【未确认】—— 没拿到窗口，下面是"一条命令闭环"的方案**

### 5.0 为什么没拿到（诚实记录）

| 时间 | c0（Phy-ID 8–15） | c2（本轮用的单卡） | 结果 |
|---|---|---|---|
| 02:04–02:31 | `l3_l3-a-16x32k` | `p2-batch*` | 等 |
| 02:33–02:56 | **`l3_l3-a-16x32k` 第二轮** | 我的单卡臂（3 次失败 + 1 次成功） | — |
| 02:52 / 02:55 | c0 短暂空闲 ⇒ 我起了 **2 次 tiny TP8 臂** | — | ❌ `model-tiny` 的 `text_config.rope_scaling.rope_type=yarn` 在 **L1_dummy 的 `--load-format dummy` 路径**下被 `pydantic` 拒绝（`Unrecognized keys in rope_parameters`）；换**真权重目录 + dummy**也被同一处拦（镜像的 `config.json` 同样带 yarn） |
| 02:57–03:15 | `l3_l3-b-16x128k`（第三轮，仍在跑） | `p2-batch*` → 我的 A/B 臂 | ✅ A/B 拿到，**8 卡没拿到** |

★ **tiny TP8 走不通的真正原因**（供后续复用）：`--load-format dummy` 在**本镜像**上只能配
**`V41_DUMMY_WO_A_FIX` + 那条 `rope_scaling` 形态**的 config；`model-tiny` 的
`rope_scaling = {rope_type: yarn, factor:16, beta_fast:32, beta_slow:1, original_max_position_embeddings:65536}`
会被 transformers 的校验直接拒（`rope_parameters` 不认这组键）。
**这不是我的补丁的问题**（补丁从未被加载到那一步）。

### 5.1 一条命令闭环（**谁拿到 8 卡窗口就跑这个**）

```bash
# A3 上（c0 锁空闲时；整条 ~12–15 min，含起服）
ssh -o ControlPath=none A3-node1
D=~/projects/dsv41-upstream-pr/agents/P3_sharedregion
rm -f $D/DUMP_TRIGGER $D/out/dump-r1/*.json
TAG=p3-srg-r1 OFFLOAD_BYTES=10200989696 BPC_JSON='{"default":8,"swa":1}' \
  PROMPTS=16 PROMPT_TOKENS=32768 MAX_LEN=40960 ENGRAM=1 \
  bash $D/scripts/run_arm_p3.sh          # 自己 flock c0；拿不到 = exit 75
# 判据（V1/V3/A/B/C/D 一次算完）：
bash $D/scripts/judge.py $D/out/p3-srg-r1 $D/out/p3-srg-r1.server.log --dump $D/out/dump-r1
```

* 该 runner **已经内置**：起服自检（`rank=0 replicated=True` × 8、128 条 mmap 池行）、
  fill → reset → **触发 dump（8 份 `v1-rank*.json`）** → replay → 收尾；
* `judge.py` 输出 **A（宿主乘数）/ B（四条判据）/ C（sha256）/ D（V1+V3 的哈希比对）**；
* 期望值（若 L6 成立）：**A ≈ ×0.868**（现在是 ×6.944）、
  **B 四条全中**、**C = `d23082b3…`**（与 `017/021` 同口径时）、**D 全部 True**；
  **若 D 的"GPU 跨 rank 相同"有为 False 的** ⇒ **V1 否决 ⇒ L6 在 V4.1 上不成立**。

### 5.2 残余风险的机制（为什么 V1 不能只靠代码论证）

我的代码级论证（§二）证明的是"**KV 页里没有按 rank 切分的维**"，
但**没有**证明"8 个 rank 算出来的 hidden_states 逐位相同"：

| 环节 | 是否会引入 rank 差异 |
|---|---|
| MLA latent / SWA / indexer 的**权重** | ✗ replicated（`nn.Linear`），但**输入**是 hidden_states ⇒ 看下一行 |
| 层内 `o_proj`（RowParallel）的 **all-reduce** | ⚠️ 归约**结果在数学上相同**，但**浮点加法顺序**未必 bit-exact（取决于 HCCL 的 ring/tree 与分片顺序） |
| MoE（本机是 **ALLGATHER**，`a2/AGENTS.md` §4.2） | ⚠️ 同上，且 `logs/016` 的口径是 **TP8 且 EP 相关路径**；A2 的 MoE 通信与 A3 不同 |
| 上游的自我认证 | 他们只写了 *"configurations it has certified as byte-replicated (**currently pure MLA**)"* —— **"certified"这个词本身说明它不是自动成立的性质** |

⇒ 这正是**必须实测**的那一格；也是我建议"先花一个窗口做 dump、再决定要不要投上游"的理由。

---

## 五·补、原来的臂设计（**未执行**）

### 5.1 臂设计

| 臂 | 内容 | 目的 |
|---|---|---|
| **`p3-srg-r1`** | 8 卡、16×32768 token、`cpu_bytes_to_use = 10,200,989,696`（54 GiB 记账…见下）、`per-group bpc {default:8, swa:1}`、`P3_SHARED_REGION=1`、`P3_BLOCK_BYTES=910208` | 与 `logs/016`/`022` 的 `l3-a-16x32k` 同口径对照，拿 8× |

★ **对照臂的口径**：`l3-a-16x32k` 用同一个 `cpu_bytes_to_use`（**54 GiB**），
`logs/029`/`016` 的宿主实占 = **×6.944 = 403 GiB**；开了单副本后应该降到
**~×0.868**（6.944/8 = 0.868）⇒ 同样 54 GiB 的 ask 只吃 **~47 GiB**。
**注意**：`P3_BLOCK_BYTES` 换口径后 `num_blocks` 会**变小 8×**（行宽 ×8），
除非同时把 ask 提高。两条账要分开报（见 §7）。

判据（任务书 A–D）：

| # | 判据 | 期望 |
|---|---|---|
| A | 宿主实占降 ~8× | 同样的 `cpu_bytes_to_use`，`/proc/meminfo` 增量从 ×6.944 → ~×0.87 |
| B | 四条判据仍全中 | `BlockStored>0` / `CPU→GPU>0` / `hits>0` / **replay ≪ fill** |
| C | 输出 sha256 与现状一致 | 与 `logs/017/021` 的 `d23082b3…` 逐字相同 |
| D | 没有互相踩（V3） | 8/8 worker 正常、池子行哈希 == GPU 行哈希 |

---

## 六、单卡先验（c2）：结构 + 不回归

### 6.1 ★ 第一版失败与修正（**必须记下来**，主代理也独立定位到同一条）

```
File "/work/agents/P3_sharedregion/patched/shared_offload_region.py", line 50, in __init__
AssertionError            # assert kv_bytes_per_block % self.page_size == 0
```

* 我第一版把**上游的 `kv_bytes_per_chunk`**（= `round_up(131,072 × 8, 4096)` = 1,048,576，是 4096 倍数）
  当成行宽 ⇒ 断言其实**过得去**；真正挂的是**我第二版**把行宽换成
  **`910,208`（真实 Σ）**之后：`910,208 % 4096 = 896 ≠ 0` ⇒ **断言挂**。
* **根因**：DSV4.1 的 16 个 canonical page 里有 `128 / 256 / 2048 / 147,712` 这种**非 4 KiB 倍数**
  （`128` 是 indexer 的 scale 平面；`147,712 = 144.25 KiB`），
  而 `SharedOffloadRegion` 要求行宽是 `mmap.PAGESIZE` 的倍数。
* **修法**（任务书要求"先做不占卡的单元自检"）：
  **每个张量单独 `round_up(page_t × bpc × num_copies, 4096)`**。
  `scripts/offline_align_check.py`（**不占卡**）对 `logs/021` §4.4 的第一手 16 个 page 全枚举：

```
canonical 张量数 = 16, Σ page = 910208 （上游记账 131072 ⇒ ×6.9443）
bpc=1 copies=1: Σrow=929792（对齐开销 2.152%）未对齐行=0
bpc=1 copies=8: Σrow=7294976（对齐开销 0.183%）未对齐行=0
bpc=8 copies=1: Σrow=7294976（对齐开销 0.183%）未对齐行=0
bpc=8 copies=8: Σrow=58253312（对齐开销 0.000%）未对齐行=0
PASS
```

  ⇒ 单副本（copies=1、bpc=8）下**对齐只多花 0.183%**；且 `spec.py` 侧的
  `kv_bytes_per_chunk`（= `round_up(Σ × bpc, 4096)`）**在两个方向上都 ≥ 实际写入量**（安全方向）。

### 6.2 ★ 单卡臂 `p3-c2-coex2`（c2，**已验证**）【实测】

```
[P3_sharedregion] 记账口径改写 worker_kv_bytes_per_block 131072 -> 910208（真实/上游 = 6.944x）
[P3_sharedregion] region 布局 rank=0 replicated=True num_copies=1 tensors=16 num_blocks=…
[shared_offload_region] Created mmap file /work/agents/P3_sharedregion/regions-c2/vllm_offload_<engine>.t0.mmap (0.08 GB)
                       … t0..t15（**16 个 region，全在盘上，没进 /dev/shm**）
[fill]    ttft p50 = 464.9 ms   out_sha256 = 24b57053…
[replay1] ttft p50 =  50.8 ms   out_sha256 = 24b57053…   （**9.1× 加速**）
[sha256] fill == replay，16/16 prompt 逐字一致，mismatched=[]
[kv_events] BlockStored:CPU=714  GPU=6326  BlockRemoved:GPU=3672  AllBlocksCleared=1
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..11]；被排除的组=[1]（state）
[SWA_pergroup] per-group bpc={0:8, 1:8, 2..11:1}（unit 模式）
[self-check] 挂载链：merged/offloading_config.py + merged/scheduler.py + merged/{cpu_spec,pgp_manager}.py
             + patched/{asc_npu,asc_cpu_npu,shared_offload_region}.py —— **7 个模块全部命中**
```

**这三条得到的结论**：
1. **`rank=0` + `replicated_layout=True` + `SharedOffloadRegion` 路径在 NPU 上跑得通**（不崩、不越界）；
2. **两条 4096 对齐 assert 全过**（`SharedOffloadRegion` 的行宽断言 + `kv_bytes_per_chunk % 4096`）；
3. **`isinstance(kv_cache_spec, FullAttentionSpec)` 那条也过**（它由 D2 scheduler 补丁解决，
   本臂已挂 —— 与我 02:41 那次失败形成对照证据）；
4. **四条判据不回归**：`BlockStored:CPU=714 > 0`、replay 50.8 ms ≪ fill 464.9 ms（**9.1×**）、
   fill/replay 输出 sha256 **逐字相同**；
5. ★ **TP1 下 `world_size=1`，所以 8× 不显现**（本来就只有 1 份副本）——
   这一臂验的是**工程前提**，收益倍数必须等 TP8。

⚠️ **与 `logs/021` 基线的 sha256 不同**（`24b57053…` vs `d23082b3…`）：两次的 runner/参数并不逐字一致
（我走的是 `serve_a2` 挂载链 + `--max-model-len 8192` 显式给 + 池 1.5 GiB），
所以**判据 C 的正确做法是"同一 runner 下的 A/B 自比"**，见 §6.3。

### 6.3 ★ A/B（同一 runner、同一批 prompt，只切 `P3_SHARED_REGION`）【实测】

| 臂 | `P3_SHARED_REGION` | 池子（worker 实分配） | 记账 ask | **宿主乘数** | fill p50 | replay p50 | 加速 | 输出 sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`p3-c2-A-base`** | **0**（现状：每张张量一个私有 pin 池） | 12288 unit × 910,208 B = **10.417 GiB** | 1.5 GiB | **×6.944** | 462.1 ms | **46.9 ms** | 9.9× | `24b57053…` |
| **`p3-c2-B-shared`** | **1**（`SharedOffloadRegion`，16 块盘上 mmap） | 1763 unit × 929,792 B = **1.494 GiB** | 1.5 GiB | **×0.996** | 527.5 ms | **55.1 ms** | 9.6× | `24b57053…` |

```
A: [SWA_pergroup] CPU 卸载池: num_units=12288 kv_bytes_per_unit=131072 worker_kv_bytes_per_block=131072 world_size=1
B: [P3_sharedregion] 记账口径改写 worker_kv_bytes_per_block 131072 -> 910208（真实/上游 = 6.944x）
   [SWA_pergroup] CPU 卸载池: num_units=1763   kv_bytes_per_unit=913408 worker_kv_bytes_per_block=910208
   [P3_sharedregion] CPU pool[0] (mmap) 1763 x 131072 (0.215 GiB) ptr=0xfffdf3800000   … ×16
   [sha256] fill==replay==24b57053…（16/16 prompt 逐字一致，mismatched=[]）——**A、B 两臂逐字相同**
```

★ **必须写清楚的两条**（否则这张表会被误读）：
1. **这是"记账变诚实"，不是"容量变大"**：A 与 B 的**每 unit 字节数几乎相同**
   （910,208 vs 929,792 B，B 多 2.15% 的页对齐开销）⇒ **同样 DRAM 装同样多条目**。
   B 比 A 省 6.97× 的原因是 **A 用 6.944× 的 RAM 多买了 6.944× 的 unit**（`num_units` 12288 vs 1763）。
   ⇒ **A/B 在单卡上拿到的 6.97× 是 L2（记账诚实）的功劳，不是 L6（副本合并）**。
2. **TP1 下 `world_size=1` ⇒ 本来就只有 1 份副本 ⇒ L6 的 8× 在这一臂里
   *在数学上不可能显现***（日志里 `replicated_layout=False` 正是我的拓扑护栏在起作用：
   上游那条 gate 要求 `tp_size > 1`，我保留了它）。
   **8× 只能由 TP8 臂度量**，而 TP8 臂要占 Phy-ID 8–15 —— **本任务没拿到**（见 §五）。

★ **本 arm 的真正价值**：它证明了**开了 region 之后四条判据不回归、输出逐字不变**，
而**没有**证明 8×。

### 6.4 ★ A/B 的逐项等价性（**16 项工程指标全部逐字相同**）【实测】

| 指标 | A（基线，私有 pin 池） | B（共享 region） | 同？ |
|---|---|---|---|
| `GPU KV cache size` | 22,719 tokens | 22,719 tokens | ✅ |
| `kv_offload_total_bytes{CPU_to_GPU}` | **272,957,440 B** | **272,957,440 B** | ✅ |
| `kv_offload_total_bytes{GPU_to_CPU}` | 364,421,120 B | 364,421,120 B | ✅ |
| `kv_offload_size_bucket{CPU_to_GPU, le=2e7}` | **16**（16 个 load job） | **16** | ✅ |
| KV 事件 `BlockStored:GPU` / `:CPU` | 6326 / **714** | 6326 / **714** | ✅ |
| KV 事件 `BlockRemoved:GPU` | 3672 | 3672 | ✅ |
| 输出 sha256（fill 与 replay） | `24b57053…` | `24b57053…` | ✅ |
| 逐 prompt 一致数 | 16/16，`mismatched=[]` | 16/16，`mismatched=[]` | ✅ |
| 参与卸载的组 | full=[0] swa=[2..11]，排除 [1] | 同 | ✅ |

⇒ **判据 B（四条判据全中）、判据 C（输出一致）在"共享 region"下全部通过**；
**（这是一个"换了内存后端、行为逐位不变"的强等价性证据。）**

★ 唯一**有差异**的一项是 DMA 耗时：`kv_offload_total_time_total{GPU_to_CPU}`
**0.0197 s（A） vs 1.082 s（B）**——B 慢 **55×**。原因：A 的池是**注册过的 pinned 内存**
（`aclrtMallocHost`），B 是**盘上文件的 pageable 映射**（本 arm 没调 `aclrtHostRegister`）。
★ **这正是上游那段注释"pinned tensor path has the best proven H2D/D2H performance"里
唯一站得住的一半**；但 `logs/014` §4.2 已实测：**注册后带宽回到 56.5–58.3 / 42.0–42.7 GB/s**
⇒ **要让 B 的时延追平 A，加一步 `aclrtHostRegister` 即可**（P1 的 helper 现成）。

TP1 时 `world_size=1` ⇒ **8× 不显现**（本来就 1 份），单卡只验三件事：
① region 路径能起服、② 四条判据不回归、③ 输出 sha256 与 `017/021` 一致。

---

## 七、A2 新容量账（**按新结构重算**；不含未确认的 L6）

**基本单位（`logs/021` §4.4 的第一手数字，A2 真机与 A3 tiny 逐字节相同）**

```
1 个 unit = 1 个 GPU block × 16 张 canonical 张量
  记账口径   = worker_kv_bytes_per_block(131,072) × blocks_per_chunk   ← ★ 上游的口径（低估）
  真实口径   = Σ over 16 张 page = 910,208 B = 记账 × 6.9443
  （本任务新增：把 910,208 显式喂给 config.py ⇒ 记账不再低估）
```

### 7.1 本轮**实测确认**的两格（可进合并账）

| 杠杆 | 机制 | 倍数 | 强度 |
|---|---|---:|---|
| **L2（记账诚实化）** | `worker_kv_bytes_per_block` 131,072 → **910,208**（`P3_BLOCK_BYTES`） | 宿主乘数 **×6.944 → ×0.996**（单卡 A/B 实测，§6.3） | 【实测】 |
| **L5（per-group bpc）** | SWA 条目粒度 1 unit | **4.89×** | 【实测·`021`】 |

★ 另有一条**必须一起报的代价**（§6.4）：换到 mmap region 后 **D2H 时延 0.0197 s → 1.082 s（×55）**。
它**不是** L6 的必要代价（`logs/014` §4.2 实测：注册后 H2D 56.5–58.3 / D2H 42.0–42.7 GB/s，
与 pinned 同级），而是"我这一版没加 `aclrtHostRegister`"的**实现缺口**。

★ **口径提醒（重要）**：L2 让 **`cpu_bytes_to_use` 变成"真正的宿主字节数"**。
本轮之前那个 **「建议 ≈260 GB」** 的口径（`a2/AGENTS.md` §4.3）是按 4421 B/token × 8 份副本
（= 无 ×6.944）算的；有了 L2，`cpu_bytes_to_use = X` 就**真的只吃 X**。
⇒ **A2 上的推荐 ask 不再需要"除以 6.944"**，但**也不能乘以 6.944**（那是旧 bug 的行为）。

### 7.2 **不能**进账的两格

| 杠杆 | 为什么不能进 |
|---|---|
| **L6（8 份 → 1 份）** | **V1 端到端未确认**（§0/§二/§五）；且即使成立，**它要先改上游 gate**，并且**必须与 L1 一起重算**（两者都作用在"每个张量分多少 slot × 几份"上，不是简单相乘 —— `031` §三 已标注） |
| **L1（16 个张量的闲置）** | 由 `P2_poolsizing` 负责（`030`/`034`），**我不动它** |

### 7.3 与 L1 的交互（**为什么不能拿 16×8 去乘**）

```
L1 作用在：每个 canonical 张量各分 num_blocks 个 slot，而一个 slot 只用 1 个张量（时间维闲置）
L6 作用在：8 个 rank 各存一份逐字节相同的副本（空间维冗余）

两者在"Σ over 16 张张量 × num_blocks × 8 rank"这一个**乘积式**里各占一个**因子**：
   现 在：  16 × num_blocks × 8 × pagē
   L1 后：  Σ_g (slots_g × page_g) × 8            ← 组粒度替换掉 16
   L6 后：  再 ÷ 8
⇒ **只有在 L1 把"16 张张量各一份"改成"按组分配"之后，L6 的 8 才是干净的 8×**；
   而在 L1 之前，L6 的 8× 与 L1 的 16× 作用的**字节**是同一批（`logs/031` §三 的警告是对的：
   **不要 16×8=128**）。本任务**没有**把这两个乘起来，也没有为 L1 代算。
```

### 7.4 给 A2 的建议（一句话）

> **容量账先用 L2 + L5**（两者都是实测）：A2 宿主余量 442 GiB ⇒
> `cpu_bytes_to_use` 可以**按真实宿主字节**给到 **~330 GiB**（留 1.25× 余量），
> 折成 token ≈ **`330 GiB / (910,208 × ...)`** —— 具体条目数请 `X_integrate` 用
> **L1（P2 的最终口径）+ L5** 一起算，**本任务不替 L1 出数**。

---

## 八、cannbot 对照（★ 任务书红线 §6 要求）

**结论：cannbot 直接证伪了 upstream 注释里的那句事实陈述。**

| 出处（A3 上的 cannbot 只读源） | 原文 | 与本任务的关系 |
|---|---|---|
| `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/runtime/runtime_migration/references/api_support_table.md:50` | `cudaHostRegister()` → **`aclrtHostRegisterV2()`**，支持度 **✅**，说明"注册主机内存" | 上游注释说"Ascend has no public cudaHostRegister-equivalent"，**cannbot 的 API 映射表明确列了等价物** |
| `.../cann_api_common.md:223-230` | `aclError aclrtHostRegisterV2(void *hostPtr, size_t size, aclrtHostMemRegisterPolicy policy);` | 签名与语义与 `cudaHostRegister` 对齐 |
| `.../cann_api_common.md:525` | `cudaHostRegisterDefault` ↔ **`ACL_HOST_MEM_REGISTER_DEFAULT`** | flags 也有对应项 |
| `.../cann_api_common.md:~230` | `aclrtHostUnregister` / `aclrtHostGetDevicePointer` 同样在表里 | 注册/解注册/取设备指针三件套齐全 |
| `api_support_table.md:44-53` | `cudaPointerGetAttributes → aclrtPointerGetAttributes` ✅ | 与 `logs/014` §4.1 里"注册后 `is_pinned()` 变 True"一致 |

> 与 `logs/014` §4.1 的**实测**合并起来看：`aclrtHostRegister(ptr, size, flags=0)` 对任意 mmap buffer
> `ret=0`、往返逐字节一致、H2D 58 / D2H 42.7 GB/s（与 pinned 同级）。
> **cannbot 的文档 + 我们的实测，两边都指向同一个结论**：
> `NPUOffloadingSpec.create_worker` 那段注释的**事实前提不成立**。
> （注：镜像里我们能调到的符号名是 `aclrtHostRegister`；cannbot 表里是 `aclrtHostRegisterV2`，
> 两者是同一族的 V1/V2 关系，**A3 镜像的 libascendcl 导出的是 V1**——`logs/014` 直接 `ctypes` 调过。）

---

## 九、对上游的独立贡献（**一段可单独发的材料**）

> **`vllm-ascend/.../kv_offload/native/npu.py::NPUOffloadingSpec.create_worker` 的注释里有两处事实错误，
> 它们共同把一个"内存放大 8×（A2 场景下 8 卡 × 8 副本 = 403 GiB 宿主）"的默认值锁死了：**
>
> 1. *"Unlike CUDA, Ascend has no public cudaHostRegister-equivalent for an arbitrary mmap buffer"* ——
>    **不成立**：CANN 的 API 映射表把 `cudaHostRegister()` 映射到 **`aclrtHostRegisterV2()`（支持度 ✅）**，
>    并且我们在 910B3 上实测 `aclrtHostRegister(ptr, size, 0)` 对**任意 mmap buffer** `ret=0`、
>    往返逐字节一致、**H2D 58 GB/s / D2H 42.7 GB/s**（与 pinned 同级）。
> 2. *"Consequently `replicated_layout` remains safely disabled for this spec by the upstream
>    `_uses_shared_region()` gate"* —— **因果也不完整**：即便把 `_uses_shared_region()` 打开，
>    `replicated_layout` 仍然被**上游 `offloading/config.py` 的门**挡着
>    （要求**单组 + 裸 `MLAAttentionSpec`**，hybrid 模型恒为 False）。
>    ⇒ 想让 hybrid（DeepSeek-V4.x 这类 4-slot MLA 布局）吃到单副本，**上游那两处都要放宽**。
> 3. 顺带：`worker_kv_bytes_per_block = kv_cache_tensors[0].size // num_blocks`
>    在 V4.1 的 4-slot 布局下**低估 6.944×**（每个 `KVCacheTensor` 是一个 slot，不是全局总量），
>    这会让 `cpu_bytes_to_use` 与真实宿主占用差一个数量级（我们实测 **403 GiB 实占 vs 58 GiB ask**）。

---

## 十、诚实边界

| # | 事项 | 状态 |
|---|---|---|
| 1 | **V1 端到端字节级比对**（8 个 rank 同 slot 哈希） | **【未确认】** —— 没拿到 8 卡窗口；**闭环命令见 §5.1**（≈12–15 min 一条） |
| 2 | **8× 实测** | **【未确认】** —— 单卡 TP1 在数学上不可能显现（§6.3 注 2） |
| 3 | **V3 写路径互踩** | **【未确认】** —— 依赖第 1 条的 dump；机制分析见 §四（结论：**若 V1 成立则幂等**） |
| 4 | V2 / backport | ✅ **【实测】** —— 4 文件 ~100 行，单卡跑通、四条判据不回归、输出 sha256 与基线逐字相同 |
| 5 | L2（记账诚实化） | ✅ **【实测】** 宿主乘数 ×6.944 → ×0.996（单卡 A/B） |
| 6 | region 页对齐开销 | 【实测】≤2.15%（`bpc=1`）/ 0.183%（`bpc=8`）；`offline_align_check.py` 可复算 |
| 7 | 与 P2（L1）的交互 | 只给了**结构性说明**（§7.3），**没有替 L1 出数** |
| 8 | 纪律 | 不写 `upstream-v41/`、不用 `/tmp`、只写 `agents/P3_sharedregion/` + 本日志；单卡只用 c2、8 卡只等 c0 |

### 失败的臂（供后人省时间，全部是**环境/挂载**问题，不是补丁逻辑问题）

| 臂 | 症状 | 根因 | 处置 |
|---|---|---|---|
| `p3-c2-v2` | `AssertionError` @ `shared_offload_region.py:50` | 行宽 910,208 不是 4096 倍数 | **每张量 `round_up(…, 4096)`** |
| `p3-c2-v2b` | `assert isinstance(kv_cache_spec, FullAttentionSpec)` | stock scheduler 不认 `UniformTypeKVCacheSpecs` | 挂 D2 scheduler（`logs/010` §4.3 已知） |
| `p3-c2-coex` | `TypeError: int(dict)` | 只挂了 P3 的 config.py，dict bpc 没人解析 | 改挂 **merged** config.py |
| `p3-c2-A-base`（第 1 次） | 同上 + 基线臂根本没挂共存补丁 | sitecustomize 把共存块写进了 `P3_SHARED_REGION=1` 分支 | 把共存块**移出** env 分支（基线臂也要挂） |
| `p3-c2-A-base`（第 2 次） | `TypeError: int(dict)` | 共存块没带 config.py（只带了 spec/pgp_manager/scheduler） | 共存块补上 config.py |
| `p3-tiny8-v1` / `p3-real8-v1` | `ValidationError: Unrecognized keys in rope_parameters` | 镜像的 config 带 `rope_scaling.rope_type=yarn`，`--load-format dummy` 路径被 transformers 校验拒 | **未解决**（与本任务无关；`L1_dummy` 的 tiny config 需改 `rope_scaling` 形态，或直接用 L3 的真权重臂口径） |
