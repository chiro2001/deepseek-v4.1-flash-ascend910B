# 021 · per-group `blocks_per_chunk`：SWA 池条目**真的变小了（4.89×）**，而且**变长前缀安全**

**日期**：2026-09-22 01:33 – 01:56（A3 本地时钟；**11 条判据臂**，单臂 1.0–1.4 min）
　**执行**：子代理 `SWA_pergroup`
**机器**：A3（A3-node1），**只用 c2 槽位**（`a3_up.sh` 的 c2 = Phy-ID 7，容器 `prbench-c2`，TP1）
**上游/镜像**：与 `013`/`017` 同一套：vLLM `0.27.1` + vllm-ascend `e43cf1e9f`；模型 `agents/L1_dummy/models/model-tiny`
（dummy 权重，保留 40 层 KV 结构；**16 张 canonical KV 张量**，与 A2 真机同构，见 §4.4）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/算式推出来但没直接测；【未确认】= 没跑到。
**原始数据**：`logs/raw/021-swa-pergroup/`（**67 个文件**：11 臂的 meta/client/kv_events/metrics + 三份 batch 日志 + 探针输出 + `021-summary.md`）

---

## 0. 一句话结论（先给判断）

| 任务书判据 | 结果 |
|---|---|
| **① 总字节数降 4.9×？** | **【实测·是】按记账口径 704 MiB → 144 MiB（**4.89×**）**，且同样的 16×4096 工作负载下 **1.000× 的池子四条判据全中**（臂 `pgp-pg144m-8`）。按宿主物理口径 305 MiB → 62.5 MiB/请求，**同一个 4.89×**（§4.4/§6）。 |
| **② 四条判据仍全中？** | **【实测·是】**`BlockStored:CPU=714>0`、`CPU→GPU=273.0 MB>0`、`hits=65,520>0`、**replay 49.9 ms vs fill 464.5 ms（9.3×）**（§4.2）。 |
| **③ ★ 变长前缀安全（核心）？** | **【实测·是·已修复】**同池同代码、4096 填充 → **2048 回放**：`hits=32,752`、`CPU→GPU=178.4 MB`、replay **46.1 ms**（10.1×）；而**同一 per-group 代码**下把存侧规则换成 `SWA_TRIM=window`（017 那条）**仍然是 0/0/240.6 ms**（§5）。 |
| **④ 文本 sha256 一致？** | **【实测·一致】**4096→4096：`d23082b3…`（**5 条**同 workload 臂逐字相同，两条 4196 边界臂也是同一个值，且与 `017` 基线**逐字节相同**）；4096→2048：`38c32f99…` = 2048 **冷算**参考（§4.3）。 |
| **⑤ 每请求/每层实际字节？** | **【实测】**单位（1 个 manager block × 16 张张量）= **131,072 B 记账**、**910,208 B 物理**（×6.945，与 `016` 的 A2 真机数字**逐字节相同**）；full chunk = 8 单位 = 1 MiB、**SWA chunk = 1 单位 = 128 KiB** ⇒ 每请求 9 MiB（旧 44 MiB）（§4.4）。 |

> **一句话**：把 `blocks_per_chunk` 做成 **per-group**（SWA=1、full=8）、池的记账单位改成"**1 个 GPU block**"之后，
> 上游那条**本来就有、在 DSV4.1 上空转**的 `is_store_reachable_swa_chunk()` 立刻生效（`alignment_chunk_count` 从 `None` 变成 **8**），
> 于是**每 1024 token 的每个对齐段只留最后 1 个 128 KiB 的 SWA 条目**——池子 **4.89× 变小**，而"短前缀回放"这个 017 的反例**不再归零**。

**第一手一行证据**（`scripts/summarize.py` 的 `SWA acc` 列，`align_tokens` 两臂都是 1024）：

```
| 臂              | ... | align_tokens | SWA acc        |
| pgp-base4g-8    | ... | 1024         | None(全存)      |  ← 标量 bpc=8：钩子空转
| pgp-pg144m-8    | ... | 1024         | 8 (bpc=1)      |  ← per-group：每 8 个 chunk 只留 1 个
| pgp-pg144m-mixed| ... | 1024         | 8 (bpc=1)      |  ← 同一规则，短前缀回放 hits=32,752
```

---

## 1. 环境与做法（11 条臂，全在 c2）

| 项 | 值 |
|---|---|
| 卡 | **只 c2**（`tools/a3_chip.sh c2` 锁；**没碰** c0/c1、Phy-ID 8–15、`dsv41-a3`、`mooncake-master`） |
| 模型 / 起服 | 与 `013`/`017` 逐字一致：TP1、`--load-format dummy`、`--block-size 128`、`--enable-prefix-caching`、`--prefix-match-unit 32`、`--kv-cache-memory-bytes 1 GiB`、`ENGRAM=0` |
| 补丁入口 | `agents/SWA_pergroup/patch/sitecustomize.py`：`sys.meta_path` **整文件替换** `offloading/scheduler.py`（= D2 副本 + SWA_trim 4 处 + 本轮的 per-group bpc/unit 展开），外加**三个函数级钩子**（`build_offloading_config` / `CPUOffloadingSpec.get_manager` / connector 自检）与两段只读日志（池子 + worker 物理池）。**不写镜像**。 |
| 每臂流程 | 起服 ≈75 s → `/health` → ZMQ KV 事件探针 → fill 轮 → `POST /reset_prefix_cache` → replay 轮 → 收 `/metrics` → 停服 |
| 固定量 | `PROMPT_SALT=20260922`（11 条臂 prompt 逐 token 相同 ⇒ 可跨臂比 TTFT/sha256） |
| `blocks_per_chunk` 传法 | `--kv-transfer-config` 的 `kv_connector_extra_config.blocks_per_chunk`：**标量**（旧口径）或 **JSON dict**（新口径，如 `{"default":8,"swa":1}`） |

### 1.1 臂清单（一行一臂，全部【实测】）

`池（记账）`= `num_units × 128 KiB`（unit 模式下）/`num_blocks × 1 MiB`（标量模式）；`池（物理）`= `num_units × 910,208 B`（§4.4）。

| 臂 | bpc | 池（记账） | 池（物理） | token：fill→replay | 条目/请求 | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` (MB) | `hits`/`queries` | fill p50 | **replay p50** | 加速 | 输出 sha256 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pgp-base4g-8`（**标量 8**，向后兼容对照） | 8 | 4096 MiB | 27.8 GiB | 4096→4096 | **44** | 704 | — | 273.0 | 65,520/131,328 | 462.3 | **47.3** | 9.8× | `d23082b3…` ✅ |
| `pgp-pg4g-8`（per-group，大池） | PG | 4096 MiB | 27.8 GiB | 4096→4096 | 44.6 | 714 | — | 273.0 | 65,520/131,328 | 463.0 | **47.9** | 9.7× | `d23082b3…` ✅ |
| `pgp-pg144m-8` ★（**1.000×**） | PG | **144 MiB** | 1.0 GiB | 4096→4096 | 44.6 | 714 | **10** | 273.0 | 65,520/131,328 | 464.5 | **49.9** | 9.3× | `d23082b3…` ✅ |
| `pgp-pg144m-8b`（同上**复跑**） | PG | 144 MiB | 1.0 GiB | 4096→4096 | 44.6 | 714 | 10 | 273.0 | 65,520/131,328 | 463.2 | **50.0** | 9.3× | `d23082b3…` ✅ |
| `pgp-pg176m-8` ★（1.22×） | PG | 176 MiB | 1.19 GiB | 4096→4096 | 44.6 | 714 | — | 273.0 | 65,520/131,328 | 463.7 | **49.3** | 9.4× | `d23082b3…` ✅ |
| **`pgp-pg144m-mixed` ★★** | PG | **144 MiB** | 1.0 GiB | 4096→**2048** | 44.6 | 714 | 10 | **178.4** | **32,752**/98,560 | 465.4 | **46.1** | **10.1×** | `38c32f99…` ✅ |
| `pgp-pg4g-mixed` | PG | 4096 MiB | 27.8 GiB | 4096→2048 | 44.6 | 714 | — | 178.4 | 32,752/98,560 | 464.4 | 44.2 | 10.5× | `38c32f99…` ✅ |
| **`pgp-win144m-mixed` ✗**（017 那条规则） | window | 144 MiB | 1.0 GiB | 4096→**2048** | 24.6 | 394 | — | **0** | **0**/98,560 | 462.6 | **240.6** | **1.9×** | `38c32f99…`（**重算出来的**） |
| `pgp-ref2048-4g`（2048 **冷算**参考） | PG | 4096 MiB | 27.8 GiB | 2048→2048 | 22.6 | 362 | — | 178.4 | 32,752/65,792 | 243.3 | 42.9 | 5.7× | `38c32f99…` ✅ |
| `pgp-pg176m-4196`（4096+100 边界） | PG | 176 MiB | 1.19 GiB | 4196→4196 | 44.6 | 714 | — | 273.0 | 65,536/134,528 | 465.8 | 150.5 | 3.1× | ✅ |
| `pgp-base4g-4196`（同上，标量 8） | 8 | 4096 MiB | 27.8 GiB | 4196→4196 | 44.0 | 704 | — | 273.0 | 65,536/134,528 | 466.3 | 155.2 | 3.0× | ✅ |

> 读表要点：
> * **标量臂逐字复现 017 基线**（44 条/请求、704 条、47.3 ms、`SWA acc=None`）⇒ §2 的"向后兼容"成立；
> * per-group 臂 **44.6 条/请求**（714 = 16×44 + **10** 条尾部碎片）——**条目数没降**（这是设计使然，见 §3.3），
>   降的是**每条的大小**：SWA 条目 1 MiB → **128 KiB**；
> * `hits`/`CPU→GPU` 与基线**完全相同**（65,520 / 273.0 MB）⇒ 取回路径一点没变，只是池子更小；
> * `pgp-pg144m-8` 在 **1.000×** 处已经有 10 次淘汰（多出来的那 10 条尾部碎片），四条判据仍然全中。

---

## 2. ★ 第 1 步：`blocks_per_chunk` 的**完整**来源链（任务书的"关键问题"）

镜像内（vLLM `0.27.1`）**这一个值被 5 处共用，全部是"全局一个标量"**：

| # | 位置 | 用法 | 全局？ |
|---|---|---|---|
| 1 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py::build_offloading_config()` | 读 `kv_connector_extra_config["blocks_per_chunk"]` → `OffloadingCacheConfig(tokens_per_hash, blocks_per_chunk)` | **是**（`int(...)`，**给 dict 会 `TypeError`**） |
| 2 | 同上 | `worker_kv_bytes_per_block = total_gpu_kv_bytes // num_blocks`（packed 用 `kv_cache_tensors[0].size`，否则求和） | 全局（与 group 无关） |
| 3 | `vllm/v1/kv_offload/base.py::OffloadingSpec.__init__` | `self.blocks_per_chunk = config.cache.blocks_per_chunk` | **是** |
| 4 | `vllm/v1/kv_offload/cpu/spec.py::CPUOffloadingSpec.__init__` | `kv_bytes_per_chunk = worker_kv_bytes_per_block × num_copies × blocks_per_chunk`；`num_blocks = cpu_bytes_to_use // round_up(kv_bytes_per_chunk, 4096)` | **是（整个池只有一个容量）** |
| 5 | `vllm_ascend/.../native/npu.py::NPUOffloadingSpec.create_worker()` → `native/cpu_npu.py::NPUOffloadingWorker` | `blocks_per_chunk=self.blocks_per_chunk`、`num_cpu_blocks=self.num_blocks` ⇒ 每张 canonical 张量 `torch.zeros((num_cpu_blocks, page_size × blocks_per_chunk))` | **是** |
| 6 | `offloading/scheduler.py::SchedulerOffloadConfig.from_spec()` | `tokens_per_chunk = tokens_per_block × spec.blocks_per_chunk`（`spec.blocks_per_chunk` 就是 #3） | **是** |

**⇒ 回答"`spec.blocks_per_chunk` 从哪来"**：`extra_config` → `OffloadingCacheConfig.blocks_per_chunk` → `OffloadingSpec.blocks_per_chunk`（#1→#3），
调度侧（#6）与 worker 侧（#5）**读的是同一个字段**。

**⇒ 回答"`cpu_bytes_to_use` 是全局一个池、per-group 变小后怎么记账号"**：
把它拆成两层——

1. **池的"格子"改名成 unit**：让 `cache.blocks_per_chunk = 1`，于是 #4 算出的 `num_blocks` 的含义从
   "1 MiB 的 chunk 位"变成 **"1 个 GPU block 的位"（= 131,072 B，本机 TP1）**；`cpu_bytes_to_use` 仍然是**一个池**、**一个 LRU**（不切分容量）。
2. **一个 chunk 占几个 unit 由"组"决定**：`bpc_g`（full=8、SWA=1）只在 **manager 的分配**与**调度侧的展开**里出现（§3），
   池的容量参数（`num_blocks`）**仍然是全局一个数**——这正好是任务书里"按块数加权"的最简形式。

---

## 3. 第 2 步：实现（`agents/SWA_pergroup/`，4 个文件 + 1 个新 manager）

### 3.1 配置面（向后兼容）

`kv_connector_extra_config.blocks_per_chunk` 现在接受两种形态：

```jsonc
"blocks_per_chunk": 8                        // ← 老口径：**标量**，行为逐字不变（A/B 对照臂）
"blocks_per_chunk": {"default": 8, "swa": 1}  // ← 新口径：per-group（键还支持整数 group id / "full"）
```

解析结果（`resolve_per_group_bpc()`，本机 tiny 12 个组）：
`{0: 8, 1: 8, 2: 1, …, 11: 1}`（`kinds=['full','full','swa'×10]`），并把 `cache.blocks_per_chunk` 设成 **1**。
**标量 ⇒ 一个字段都不改、manager 也不换**（臂 `pgp-base4g-8` 就是这条路径）。

### 3.2 池：unit + "一个 key 占 `bpc_g` 个 unit"

* `PerGroupBPCManager(CPUOffloadingManager)`（`patch/pgp_manager.py`）：`_allocate_blocks()` 给每个 key 分 `bpc_g` 个 unit，
  记 `block_id(首 unit) → [units]`；`_get_load_store_spec()` 返回**按 chunk 内 block 顺序展开的 unit id 平铺表**；
  `prepare_store()` 的**淘汰按 unit 记账**（镜像内原版按 key 数，在单位池下会少淘汰）。
* 池容量 = `cpu_bytes_to_use // 131,072`（本机 TP1）：4 GiB → **32768 unit**、144 MiB → **1152 unit**（实测日志）。

### 3.3 调度侧：两个 job 各把 unit id 展开成"一个 GPU block 一个 id"

worker 侧 `blocks_per_chunk=1` ⇒ 两侧 **1:1**（`skip` 恒为 0），所以调度侧必须给出**与 GPU block 列表逐项对齐**的 CPU id 表：

* **store**：`chunk 的第 i 个 block ↔ 该 key 的第 i 个 unit`（`i = 0..bpc_g-1`；`block_id==0` 的块照旧跳过、不占 id）；
* **load**：pending 的第一个 block 在组内序号 `pos` ⇒ `key 序号 = pos // bpc_g - start_chunk_idx`、`unit 序号 = pos % bpc_g`；
* 两处都有 `assert len(cpu_ids) == len(gpu_blocks)`（**对不齐就 fail-fast，不会静默搬错数据**）。

**取回侧的第一手证据**（`pgp-pg144m-8b`，4096→4096 的回放轮，与 `017` 的同一条日志对照）：

```
# 017（标量 bpc=8）：CPU 侧一个 id = 一个 chunk
[D2_offload] load job req=… keys=14 group_sizes=[32,0,1,1,…] src_blocks=14 dst_blocks=42
# 本轮（per-group，CPU 侧一个 id = 一个 unit/block）：
[D2_offload] load job req=… keys=14 group_sizes=[32,0,1,1,…] src_blocks=42 dst_blocks=42
```

`group_sizes[0]=32` = full 组 4 个 chunk × 8 block；后面 10 个 `1` = 10 个 SWA 组各 1 个 block（窗口 128 token）；
**`src_blocks` 14 → 42** 正是"CPU 侧改成按 unit（=1 block）记账"的直接体现——搬的字节数没变（273.0 MB），但每个条目只占 1 个 unit。

**⇒ 为什么"条目数不变、字节数变小"**：`bpc_swa=1` 让 `alignment_chunk_count = alignment_tokens(1024) / tokens_per_chunk(128) = 8 > sw_chunks(1)`，
上游 `is_store_reachable_swa_chunk()` **每 8 个 chunk 只留最后 1 个** ⇒ 每 SWA 组每请求仍是 **`ceil(L/1024)=4` 条**（与 full 组同阶），
但每条从 **8 unit 变成 1 unit**。这是**"细粒度 + 上游原有规则"**的乘积效应，不是新写的裁剪规则。

### 3.4 离线探针（不占卡，`scripts/probe_unit_mode.py`）

```
== 1) 谓词：storable=32 alignment=8 sw=1 -> kept=[7, 15, 23, 31]（4/32）
   旧口径（alignment_chunk_count=None）storable=4 -> kept=4（全存）
== 2b) 任何 H=k*1024 要的 SWA chunk（idx=H/128-1）都在 kept 里：fill=8192 也成立
== 3) full 组 1 条占 8 个 unit，SWA 组 1 条占 1 个 unit；load 与 store 的 unit id 逐字一致；
   淘汰路径：capacity=22 时 evicted=4 stored=4 free_units=0
== 4) resolve_per_group_bpc()：8 -> (8, None)；{"default":8,"swa":1} -> (1, {0:8,1:8,2:1,...,11:1})
```

**⇒ 判据 ③ 的"算术前提"先在这里被验证**：只要 H 是 1024 的整数倍（前缀缓存命中长度**必然**是，见 017 §2.1 的推导），
需要的 SWA chunk 就是"那个对齐段的最后一个"，而它**一定**被上游规则留下。

---

## 4. 第 3 步：五条判据的实测

### 4.1 判据 ①：总字节数

| 口径 | 旧（标量 8） | 新（per-group） | 比值 |
|---|---:|---:|---:|
| **每请求记账** | 44 条 × 1 MiB = **44 MiB** | 4×8 unit + 40×1 unit = 72 unit × 128 KiB = **9 MiB** | **4.89×** |
| **池子（实测刚好放得下 16 请求）** | 704 MiB（`013`/`017`） | **144 MiB**（`pgp-pg144m-8`，四条判据全中） | **4.89×** |
| 每请求物理（宿主） | 305 MiB | **62.5 MiB** | **4.89×** |

> "条目**数**"没有降（44 条 ⇒ 44.6 条）——**降的是每条的大小**。任务书 §1 的提醒在这里兑现：
> SWA 组**可用** chunk 数从 `ceil(L/1024)=4` 变成 `ceil(L/128)=32`，但上游规则只留其中 **4 条**，每条 **1/8 大**。

### 4.2 判据 ②：四条判据（`pgp-pg144m-8`，池 = 1.000× 工作集）

| 判据 | 值 | 门槛 | 结果 |
|---|---|---|:--:|
| `BlockStored{medium="CPU"} > 0` | **714** | >0 | ✅ |
| `kv_offload_total_bytes{CPU_to_GPU} > 0` | **273.0 MB**（16 个 load job） | >0 | ✅ |
| `external_prefix_cache_hits > 0` | **65,520 / 131,328** | >0 | ✅ |
| replay ≪ fill | **49.9 ms vs 464.5 ms** | ≪ | ✅ **9.3×** |

同一条臂里 `kv_offload_cpu_cache_usage_perc = 1.0`（池子正好被占满）、`kv_offload_cpu_allocation_size_sum = 714`（= 条目数）【实测】。

### 4.3 判据 ④：输出文本 sha256

| 臂 | fill 轮 | replay 轮 | 与谁相同 |
|---|---|---|---|
| `pgp-base4g-8` / `pgp-pg4g-8` / `pgp-pg144m-8` / `pgp-pg144m-8b` / `pgp-pg176m-8` / `pgp-pg176m-4196` / `pgp-base4g-4196` | `d23082b36fc1146e…` | 同左（16/16 逐 prompt） | 与 `017` 的基线**逐字节相同** |
| **`pgp-pg144m-mixed`**（4096→2048） | `d23082b3…` | **`38c32f99ff8b4ffa…`** | = `pgp-ref2048-4g`（2048 **冷算**）**逐字节相同** |
| `pgp-win144m-mixed`（017 的窗口规则） | `d23082b3…` | `38c32f99…` | 输出**也对**，但它是**重算**出来的（`hits=0`）——"输出对"≠"取回成功" |

### 4.4 判据 ⑤：每请求 × 每层的实际字节（★ 这里把 `016` 的 ×6.945 **复现到字节**）

臂 `pgp-pg144m-8b` 的第一手日志（worker 侧打印 canonical 张量的真实 page 尺寸）：

```
[SWA_pergroup] CPU 卸载池: num_units=1152 kv_bytes_per_unit=131072 cpu_page_size_per_worker=131072
               replicated_layout=False blocks_per_chunk=1 per_group={0: 8, 1: 8, 2..11: 1}
               cpu_bytes_to_use=150994944 worker_kv_bytes_per_block=131072 world_size=1
[SWA_pergroup] worker 物理池: tensors=16 bpc=1 units=1152 sum_page_bytes=910208 host_bytes=1048559616
               pages=[65536, 8192, 128, 65536, 8192, 128, 65536, 8192, 128,
                      131072, 16384, 256, 131072, 131072, 131072, 147712]
```

| 量 | 记账 | 物理（宿主） | 倍数 |
|---|---:|---:|---:|
| **1 个 unit（= 1 个 manager block × 16 张 canonical 张量）** | **131,072 B**（= `worker_kv_bytes_per_block × num_copies`） | **910,208 B**（= Σ 每张量 page） | **×6.945** |
| full 组 1 条（8 unit） | 1 MiB | 7.28 MB | ×6.945 |
| **SWA 组 1 条（1 unit）** | **128 KiB** | **910 KB** | ×6.945 |
| 每请求（4 full + 40 SWA） | **9 MiB** | **62.5 MiB** | ×6.945 |
| 池子 1152 unit（16 请求、1.000×） | 144 MiB | **1,048,559,616 B = 0.977 GiB** | ×6.945 |

> ★ **`016` §1.3 的 ×6.945 不是 A2 特有**：本机 tiny 的 **Σ page = 910,208 B** 与 `016` 反解出来的
> A2 真机数字**逐字节相同**（`7,281,664 / 8 = 910,208`）⇒ 它是**这 16 张 canonical 张量的几何**决定的，
> 与 `blocks_per_chunk` 无关 ⇒ **新口径下这个 6.945 倍**照旧**要乘（§6 的 A2 账已经乘了）。

---

## 5. ★★ 判据 ③：变长前缀安全（本任务的核心）

### 5.1 三臂对照（同一 workload：16 请求填 4096 → 回放 **2048** 前缀）

| | `pgp-pg144m-mixed`<br>（per-group + **上游规则**） | `pgp-pg4g-mixed`<br>（同上，大池） | `pgp-win144m-mixed`<br>（**017 的窗口规则**） |
|---|---|---|---|
| 池 | **144 MiB（1.000×）** | 4 GiB | 144 MiB |
| `hits` | **32,752** ✅ | 32,752 ✅ | **0** ⛔ |
| `CPU→GPU` | **178.4 MB**（16 job） | 178.4 MB | **0** ⛔ |
| replay p50 | **46.1 ms** | 44.2 ms | **240.6 ms** ⛔（≈ 冷算 243.3） |
| fill p50 | 465.4 ms | 464.4 ms | 462.6 ms |
| 加速 | **10.1×** | 10.5× | **1.9×** |
| 输出 | `38c32f99…`（正确） | `38c32f99…` | `38c32f99…`（**重算**出来的） |

**⇒ 017 §6 那个反例（"池里有数据也一条不取"）在本方案下消失了**：短前缀要的那一块（`idx = 2048/128 - 1 = 15`）
正是"第 2 个对齐段的最后一个 chunk"，而它**一定**会被上游规则留下（§3.4 的探针 + 这里的实测）。
同时，**同一条 per-group 代码**下把规则换成 `window` 仍然归零 ⇒ 归零的根因就是那条规则本身（而不是 workload、池子或测法）。

### 5.2 边界（4096+100，非 chunk 整数倍）

| | `pgp-pg176m-4196`（per-group） | `pgp-base4g-4196`（标量 8） |
|---|---|---|
| 条目/请求 / `BlockStored:CPU` | 44.6 / **714** | 44.0 / **704** |
| `CPU→GPU` / `hits` | 273.0 MB / **65,536**（= 16×4096） | 273.0 MB / 65,536 |
| replay p50 / fill p50 | **150.5** / 465.8（3.1×） | 155.2 / 466.3（3.0×） |
| 输出 sha256 | ✅ | ✅ |

⇒ 尾部那 100 token（不满 1 个 full chunk）照旧不进 chunk（`storable = 4`），per-group 版的 5 个 chunk 尾巴处理与旧口径**逐项一致**。

---

## 6. 第 4 步：给 A2 的账（用新口径重算 `019` §七 的矩阵）

单位：A2 8 卡（`num_copies = world_size = 8`）⇒ **1 unit 记账 = 131,072 × 8 = 1 MiB**，
**1 unit 物理 = 910,208 × 8 = 7,281,664 B = 6.945 MiB**（§4.4；与 `016` 实测一致）。
每请求条目：**full 组 8 unit/1024 token + 每个 SWA 组（A2 上有 11 个，含 draft）× 1 unit/1024 token**（上游规则下）。

### 6.1 新矩阵（16 并发）

| 上下文 | 口径 | unit/请求 | 记账/请求 | 记账 ×16 | **宿主实占（×6.945）** |
|---:|---|---:|---:|---:|---:|
| **32K** | SWA 全存（现状，`019`） | 384×8 = 3,072 | 3.0 GiB | 48.0 GiB | **333 GiB** |
| | **per-group bpc（本轮）** | 32×8 + 352×1 = **608** | 608 MiB | **9.5 GiB** | **66 GiB** ✅ |
| | SWA 只存窗口（017，**不安全**） | 74×8 = 592 | 9.2 GiB* | — | 64 GiB |
| **128K** | SWA 全存（现状，`019`） | 12,288 | 192.0 GiB | — | **1,333 GiB** ⛔ |
| | **per-group bpc（本轮）** | 128×8 + 1,408×1 = **2,432** | 2,432 MiB | **38.0 GiB** | **264 GiB** ✅ |
| | SWA 只存窗口（017，**不安全**） | 2,128 | 33.2 GiB* | — | 231 GiB |

\* `019` 那两行的"记账"用的是**裁剪后的条目数 × 8 MiB**（旧口径的每条价格），与"只存窗口"的 74/266 条对应；本轮那一行的记账是新口径（每条按 bpc 加权）。

### 6.2 结论

1. **128K × 16 并发**：现状 **1,333 GiB**（A2 宿主余量 442 GiB，**不可能**）⇒ per-group bpc 后 **264 GiB** ⇒ **可以**（占余量 60%），
   而且**是安全口径**（不是"只存窗口"那个对变长前缀会归零的近似）；
2. 比值：**1,333 / 264 = 5.05×**（32K：333 / 66 = 5.05×）——与单卡 tiny 实测的 **4.89×** 同量级，
   差异来自两边**组的构成不同**（tiny 11 个组 = 1 full + 10 SWA；A2 12 个组 = 1 full + 11 SWA ⇒ full 组占比更小、总降幅更大）；
3. `019` 的结论"真正的出路是让 SWA 条目本身变小"【实测·成立】，并且**代价（改动面）比预想小**：只动了 1 个配置字段的解析 + 1 个 manager 子类 + 调度侧 2 处展开（§3）。
4. 【未确认】KV8_swa（`020`）那条线的 `÷1.98` 与本轮的 4.89× **是乘法关系**（页几何 vs 条目粒度），叠加后 128K×16 ≈ **133 GiB**——**没测**，等 `020` 的实测出来再乘。

---

## 7. cannbot 对照（`a2/AGENTS.md` §6 强制）

查的是 **`model/model-infer-kvcache/SKILL.md`**（A3 上 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`，
文件 32,238 B；按 §6 的索引读 `38-52,102-110,142-153,174-199,224-242`，并按"block/slot 映射与 block_size 约束"全文检索）：

| 问题 | cannbot 的回答 | 我们的处置 |
|---|---|---|
| **有没有"不同 group 用不同 block_size"？** | **没有**。§2.1（`SKILL.md:102-110`）把分页注意力写成**单一** `block_size`：物理存储 `[total_num_blocks, block_size, num_kv_heads, head_dim]`、`物理 slot = block_table[..] × block_size + seq_pos % block_size`。§1.1（`:38-52`）只把"混合层"交给 **`attn_type` 分组**（`FullAttention` / `SlidingWindow` / MLA），**不建议也没有 per-group block_size**。 | **不冲突**：我们**没动 GPU 侧的 `block_size`/`block_table`/slot 映射**（仍是 128），动的是**卸载池**里"一个池条目覆盖几个 GPU block"，属于 vLLM `kv_offload` 层，不在这份 skill 的讨论范围内。 |
| **有没有推荐的多级/异构 block 方案？** | 只有一句高阶提示（`SKILL.md:426`）：*"CPU-GPU Offload：HBM 不足时 offload 到 CPU；`torch_npu.empty_with_swapped_memory` + 异步双流；参考 cann-recipes-infer/models/hstu/.../gpu_kv_cache_manager.py"* —— 那是**模型层换分配器**的方案，**没有**关于"外部池条目粒度/记账"的任何内容。 | **没采纳**（不换分配器：A2 侧我们已经用 `aclrtHostRegister` 走通了，见 `logs/014`）。它反而印证了"host 池 + 异步双流"是 CANN 侧的常规形态。 |
| **滑窗的正确性谁负责？** | `SKILL.md:241`：*"长序列 `KV_len > sliding_window` 的正确性必须靠**模型层**保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，**不是 op 层负责**"*。 | 与 `017` §7.1 的观察一致：vllm-ascend 的 DSV4.1 SWA 是"**整序列 KV + 窗口化注意力**"，所以每请求能存出 32 个 SWA chunk——这正是"细粒度却只有 4 条有用"的由来。 |

**⇒ 汇总结论**：cannbot 的 `model-infer-kvcache` **既不支持也不禁止**本改动（它讲模型/算子层的 PA 布局，不讲卸载池的保留策略）。
本任务**没有写算子/kernel、也没做量化数值验证**，所以 `ops/*`、`model-infer-quantization` 那几节不适用；
**采纳情况：不采纳任何写法改动**（本轮不碰模型/算子/布局），只把它的"单一 block_size + attn_type 分组"当作背景确认。

---

## 8. 未确认 / 风险

| 项 | 状态 |
|---|---|
| **A2 8 卡真权重的 4.89×** | 【未确认】本轮是单卡 tiny（TP1、`num_copies=1`）。条目规律与卡数无关（§6 的算式），但 `num_copies=8` 下的物理池/淘汰断崖**没测** |
| `concurrency > 1` | 【未确认】全程 `concurrency=1`（与 013/017 同口径）；并发交错下的 LRU 行为没测 |
| 1.000× 的余量 | 【实测·单点】144 MiB = 1.000× 时四条判据全中（有 10 次淘汰）；`013` 证明拐点很窄 ⇒ **上线仍按 1.2× 配**（176 MiB 臂零淘汰） |
| `prepare_store` 的单位淘汰实现 | 【实测·弱】离线探针覆盖了"刚好满/多一条就淘汰"两条路径；**没有**在 8 卡 + 大池下压测淘汰并发 |
| 数值正确性 | 【实测·弱】只比 `temperature=0` 的 8-token 输出 sha256（与冷算参考一致）；**没做** logprob/长生成对比 |
| 非 1024 的对齐点（如 `prefix-match-unit=32` 下 32/64 的命中） | 【推断】命中长度受 full 组（1024）约束（017 §2.1 推导 + 本轮 `hits` 数字吻合：65,520 = 16×4095、32,752 = 16×2047）；**没有**专门扫非 1024 的命中长度 |
| `×6.945` 在 A2 新口径下是否仍成立 | 【推断·强】它是"Σ page / 记账 per-block 字节"的比值（本机 tiny 与 A2 真机**逐字节相同**），与 `blocks_per_chunk` 无关；**没有**在 A2 上重测宿主实占 |

---

## 9. 复现（全部在 c2；单臂 1.0–1.4 min）

```bash
# 0) 一次性：把补丁 + bench 送到 A3（coscli，不走 ssh 管道）
bash a2/scripts/cos-xfer.sh put <pkg>.tgz share/xfer/021-pgp-pkg.tgz
#   A3 上：coscli cp cos://uploads-new/share/xfer/021-pgp-pkg.tgz ~/tmp/20260922/<任务>/ \
#           && tar xzf ... -C ~/projects/dsv41-upstream-pr/agents/SWA_pergroup/

# 1) 不占卡：unit 池算术 + 谓词 + 展开一致性（~10 s）
bash tools/a3_chip.sh c2 --timeout 300 --name pgp-probe -- \
  python3 /work/agents/SWA_pergroup/scripts/probe_unit_mode.py

# 2) 三批臂（A: 主判据 / B: 变长前缀 / C: 4096+100）
bash tools/a3_chip.sh c2 --timeout 5400 --name pgp-a -- env BATCH=A bash /work/agents/SWA_pergroup/scripts/run_batch.sh
bash tools/a3_chip.sh c2 --timeout 5400 --name pgp-b -- env BATCH=B bash /work/agents/SWA_pergroup/scripts/run_batch.sh
bash tools/a3_chip.sh c2 --timeout 5400 --name pgp-c -- env BATCH=C bash /work/agents/SWA_pergroup/scripts/run_batch.sh

# 3) 观察点
#   out/<tag>.meta.txt        ← [SWA_pergroup] CPU 卸载池 / unit 模式生效 / worker 物理池 / group 表（bpc 列）
#   out/<tag>.kv_events.log   ← BlockStored:CPU / BlockRemoved:CPU
#   out/<tag>.metrics_after.txt ← kv_offload_total_bytes{CPU_to_GPU}、external_prefix_cache_hits/queries
#   out/<tag>.client.json     ← rounds[].ttft.p50_ms 与 *_out_sha256_all
#   python3 scripts/summarize.py --dir out <tag…>
```

**开关**：`BLOCKS_PER_CHUNK`（标量 8 = 旧口径；`{"default":8,"swa":1}` = per-group）；
`SWA_TRIM=off|window`（`off` = 上游规则，**本轮正解**；`window` = 017 的不安全近似，用来复现反例）；
`REPLAY_PROMPT_TOKENS=M`（回放用前 M 个 token 的前缀）。**其余与 017 逐字一致。**

---

## 10. 产物清单

| 文件 | 作用 |
|---|---|
| `a2/logs/021-20260922-swa-pergroup-bpc.md` | 本日志 |
| `a2/agents/SWA_pergroup/patch/sitecustomize.py` | 进程内挂载（meta_path 整文件替换 + 3 个函数钩子 + wo_a dummy 适配） |
| `a2/agents/SWA_pergroup/patch/pgp_scheduler.py` | 要挂的 `offloading/scheduler.py` = **D2 产物副本 + SWA_trim 4 处 + 本轮 per-group bpc / unit 展开**（md5 `15d5548e29af88da71570d5b48abddef`） |
| `a2/agents/SWA_pergroup/patch/pgp_manager.py` | **新**：`PerGroupBPCManager`（unit 池：按 key 分 `bpc_g` 个 unit、按 unit 淘汰） |
| `a2/agents/SWA_pergroup/patch/pgp_hooks.py` | `build_offloading_config`（dict bpc）/ `CPUOffloadingSpec.get_manager` / worker 物理池日志 三个钩子 |
| `a2/agents/SWA_pergroup/refs/*.py` | 镜像内原文副本（md5 与容器逐字节核对过；仅供 diff） |
| `a2/agents/SWA_pergroup/scripts/{run_arm.sh,run_batch.sh,summarize.py,probe_unit_mode.py}` | 单臂 / 三批臂 / 压行 / **不占卡探针** |
| `a2/agents/SWA_pergroup/bench/*.py` | KV 事件探针 + 压测客户端（带 `--replay-prompt-tokens` 与输出 sha256） |
| `a2/logs/raw/021-swa-pergroup/` | **11 条臂的原始产物**（67 个文件）+ 三份 batch 日志 + 探针输出 + `021-summary.md`（压行表） |

---

## 11. 红线遵守

* **只用 c2**（`tools/a3_chip.sh c2` 全程持锁）；**没碰** c0/c1、Phy-ID 8–15、`dsv41-a3`、`mooncake-master`；
* **不手设** `ASCEND_RT_VISIBLE_DEVICES`（由锁脚本注入）；
* **不用 `/tmp`**：本机 `~/tmp/20260922/swa_pergroup/`（`source a2/scripts/tmpdir.sh swa_pergroup`），A3 侧 `~/tmp/20260922/swa_pergroup/`；
* **没写** `upstream-v41/`（只读）；**没改镜像内任何源码**（全部 `PYTHONPATH` + `sitecustomize` 进程内替换，只影响自己起的服务）；**没动** `dsv41-release/`；
* 跨机传文件走 **coscli**（`a2/scripts/cos-xfer.sh`，key `share/xfer/021-*`），ssh 只跑命令（一律 `-o ControlPath=none`）；
* 每臂收尾由 `run_arm.sh` 的 trap 停掉自己起的 `vllm serve`（共享容器本体保留）；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺的格子标 `—`。
