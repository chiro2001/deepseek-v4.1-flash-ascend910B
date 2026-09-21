# DRAM KV 池的"结构性闲置"：**实测只有 ~2×，不是 16×**——但确实能拿回来

> 2026-09-22 02:1x–03:1x，子代理 **P2_poolsizing**（任务书 = [`029`](029-20260922-pool-sizing-waste.md)）。
> 起因：用户问 *"虽然跑通了但是 DRAM 能放的 kvcache 看起来太少了，有没有方案能增大？"*
>
> **一句话结论**：
> 1. **V1 成立**（一个 unit 号确实只对应一个 (group, chunk)，单一名空间）；
>    **V2 不成立**（16 张张量的行数**不必**相同——`cpu_npu.py` 里是同一段循环里的同一个变量）；
>    **V3 不要求"同号存在"**（unit 号只在**组内**当行号用，`CPULoadStoreSpec` 是**按组切片**消费的）。
> 2. ⇒ **原型做出来了**：`a2/agents/P2_poolsizing/`（0 开关、叠加在 `logs/021` 之上）。
> 3. ★ **但 029 的"16×"被实测推翻**：16 张张量**不是互相独立**的
>    （12 张只被 `full` 组用，另 4 张被 **state + 10 个 SWA 组共同引用**），
>    而 SWA 条目的**有效载荷**本来就几乎填满一行 ⇒ **真实可省 ≈ 2×**。
> 4. **单卡 tiny 实测倍率 = 1.96×**（宿主 1000 MiB → 499 MiB，同样的 1152 unit 容量、同样的命中与 sha256）。
> 5. 叠加 `029` 的 **L2（让旋钮诚实）**后，A2 的容量账变成
>    **"32 并发 × 128K" 从"不可能（528 GiB）"变成"可行（269 GiB / 余量 442 GiB）"**（见 §5）。

---

## 1. V1/V2/V3：代码级结论（**不占卡**）

镜像源码（在容器内 `/vllm-workspace/vllm`，本地同版本副本在 `graph_prep/ref/vllm/`，
md5 与 `logs/021` 的 `refs/` 逐字节一致）：

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| **V1** | 一个 unit 号是否只对应一个 (group, chunk)？ | ✅ **成立**（单一名空间） | `vllm/v1/kv_offload/cpu/manager.py:80-94`：`_allocate_blocks()` 只有**一个** `_free_list` + `_num_allocated_blocks`（没有按组切）；`base.py:29-41`：`OffloadKey = block_hash + group_idx` ⇒ 一个 key（= 一个 (group,chunk)）拿一个 `block_id` |
| **V2** | 16 张张量的 `num_blocks` 是否必须相同？ | ❌ **不必**（现状只是"同一个变量"） | `vllm_ascend/.../native/cpu_npu.py:281-313`：`for kv_cache_tensor in kv_caches.tensors:` 里用**同一个** `num_cpu_blocks` 建 `torch.zeros((num_cpu_blocks, cpu_page_size_bytes))`；唯一的形状约束是**行宽** `cpu_tensor.shape[1] == npu_tensor.shape[1] * blocks_per_chunk`（`cpu_npu.py:77`），**与行数无关** |
| **V3** | 拷贝路径是否要求"同一个 slot 号在 16 张张量里都存在"？ | ❌ **不要求** | `cpu_gpu_worker.py:73-120` `compute_sub_block_ptrs()`：行号 = `block_id`（`base_ptr + b * row_stride`）；而 `cpu_npu.py:137-181` 在 `transfer_async()` 里是**逐组**从扁平 spec 里取 `cdiv(group_size, blocks_per_chunk)` 个 id 当该组张量的行号 ⇒ **组与组之间互不影响**，`CPULoadStoreSpec` 的值只要"在该组张量的行数以内"即可 |

**⇒ 立即可得的推论**：每张张量只需要"引用它的那些组真正需要的行数"。
**但**（029 没看到的一点）：**两个组只要引用同一张张量，它们的行区间就必须互不相交**，
否则同一个 `row` 会被两个组互相覆盖 —— 这决定了真实收益上限（§3）。

---

## 2. ★ 实测结构：16 张张量到底是谁的

【实测】tiny（`/work/agents/L1_dummy/models/model-tiny`，与 A2 的结构同构：
Σpage = **910,208 B**、16 张张量的 page 多重集完全一致）——由本补丁的只读探针
在 worker 侧直接打印 `kv_caches.group_data_refs`（臂 `p2-e1-base-144m-8tok`）：

```
tensors=16  pages=[65536, 8192, 128] ×3, [131072, 16384, 256], [131072] ×3, [147712]
group→tensor_idx:
  g0  full      = [0..11]            ← 12 张"平面"张量（4 个 slot × kv/index/scale）
  g1  state     = [12, 13, 14]       ← 不参与卸载（prefix_cacheable=False）
  g2..g11 swa0-9 = [12, 13, 14, 15]  ← ★ 10 个 SWA 组**共用**同样的 4 张张量
group→page(copy):
  g0 = [65536,8192,128]×3 + [131072,16384,256]  (Σ=369,280 B/GPU block)
  g1 = 131072 ×3
  g2..g11 = 131072 ×4                            (Σ=524,288 B/GPU block)
```

【实测】每条目（= 1 个 chunk）的 **unit 消耗**（= 每个 1024-token 对齐窗口）：
`full = 8 unit`，每个 SWA 组 `= 1 unit`（`alignment_chunk_count=8`、`sw_chunks=1` ⇒ 每 8 个
SWA chunk 只留最后 1 个）⇒ **18 unit / 1024 token**（与 `logs/021` 的 4096 token × 16 请求
= **1152 unit** 逐字吻合）。

### 2.1 ★★ 为什么真实收益是 ~2×，而不是 16×

029 的推算假设"16 张张量互不相干，每张只需要自己那组的行"。**实测两份反例**：

1. **4 张 alias 张量被 11 个组共用**（state + 10 个 SWA）⇒ 它们的行区间必须**互不相交**，
   行数 ≈ **Σ 各组的行**，不是 `max`；
2. **SWA 条目的有效载荷 ≈ 一整行**（每条 4 个 ref × 131,072 B = 524,288 B，
   而 4 张张量一行合计 540,928 B）⇒ 那一行本来就写满了，没有"闲置 15/16"。

```
每 1024 token 的宿主字节（单 rank）：
  现状（16 张 × 1152 行）： 18 unit × 910,208 B              = 16,383,744 B
  基线（= 每条目的真实载荷）：8×369,280 + 10×524,288           =  8,197,120 B
  ⇒ 结构性的**真实**浪费 = 1.999×（不是 16×）
  本补丁（按组配额 + 分量行空间，含 tensor[15] 的 16,640 B/行 padding）：
                              8×369,280 + 10×540,928           =  8,363,520 B ⇒ 1.959×
```

### 2.2 ⛔ 对 `029` §2.2 那个"150×"的更正

`029` §2.2 把 `logs/019` 那个「~680 KB 宿主 / 每 cached token（数据本身 4.4 KB）」
分解成 `16 × 6.944 × 1.6 ≈ 150`。**本轮实测只能说清其中的 13.6×**：

```
结构因子：1.96 × 6.944 = 13.6×   ← 两个因子都是【实测】（§2.1 / §4.5）
```

* **`16×` 这个因子不成立**（§2.1：真实结构因子是 1.96×）；
* **`6.944×` 成立**（§4.5 一步钉死）；
* **剩下到 ~150× 的那 ~11× 本轮没有解释** —— `019` 那个"每 cached token"的**分母口径**
  与本轮不同（它摊的是整池宿主 / 某种 token 计数），**要对齐分子分母才能继续拆**。
  ⇒ 标 **【未确认】**，**不要**用 150× 去规划容量（用 §5.2 的实测单位成本）。

## 3. 原型：按组配额 + **分量行空间**（`a2/agents/P2_poolsizing/`）

```
① 调度侧 manager（p2_pool.make_quota_manager）
   * 权重 w_g = 每个对齐窗口该组要占的 unit 数（由 bpc_g、滑窗/裁剪规则算出）；
   * 配额 N_g = N × w_g / Σw（最大余数法，确定性；两侧同算）；
   * unit 号 = (group << 20) | row，row ∈ [base_g, base_g + N_g)；
   * ★ 交给调度/worker 的 spec 里**解码成组内行号**（`_get_load_store_spec`），
     于是 **worker 的拷贝路径一行都不用改**；
   * 配额不足时**只淘汰本组的条目**（`_GroupEvictFilter` 是惰性 `__contains__` 对象，
     直接喂给镜像的 `CachePolicy.evict(n, protected)`，LRU/ARC 都不用复制）。
② worker 侧（p2_hooks）每张张量的行数 = 引用它的组里最大的**行区间上界**；
   并**校验**同一张张量上各组的行区间两两不相交（冲突就直接报错，绝不静默覆盖）。
③ **分量（component）**：同一分量内行号累加、**不同分量之间可以重叠**（它们不引用任何同一张张量）。
   * 自动推导走 `kv_cache_config.kv_cache_tensors[i].shared_by`；★ **实测它在 Ascend 的
     4 张打包张量下会把所有组并成一个分量**（那 4 张张量的 `shared_by` 覆盖了全部层）
     ⇒ 自动模式 = **保守单分量 = 1.29×**（安全，但只拿到 2/3 的收益）；
   * 要拿满 1.96×，必须给 `P2_COMP_JSON`（例如 tiny/A2 的 `[[0],[1,...,11]]`）；
     **worker 会用真实的 `group_data_refs` 校验**它 —— 把共享张量的组拆到不同分量
     就**直接拒绝启动**，绝不静默覆盖；每条臂也会把**真实分量**打印出来供复制。
④ `P2_POOL_PATCH=0`（默认）= 逐字回退 `logs/021` 的行为；`P2_STRUCT_LOG=1` 只打结构日志。
```

| 文件 | 作用 |
|---|---|
| `patch/p2_pool.py` | 配额/行空间/分量算术 + manager 工厂 + `rows_per_tensor()` |
| `patch/p2_hooks.py` | 三个钩子（配置权重+分量 / manager / worker 行数）+ 只读结构日志 + **行区间冲突校验** |
| `patch/sitecustomize.py` | 进程内挂载（**先 exec `logs/021` 的 sitecustomize 再叠本补丁**，不写镜像） |
| `scripts/probe_quota.py` | ★ **不占卡**探针（< 10 s）：配额算术 / 组内行号 / 按组淘汰 / 分量冲突检测（含**反例**断言） |
| `scripts/probe_rss.py` | ★ **不占卡**：判据 D 的机制（`torch.zeros` 是否 lazy、注册的开销） |
| `scripts/run_arm_p2.sh` + `run_batch{2,3,5,6}.sh` | 单臂 / 四条臂 / 第三轮 / 判据 A 六臂 / 判据 D 四臂 |

---

## 4. 单卡 tiny 实测（c2，16 × 4096 token，`blocks_per_chunk={"default":8,"swa":1}`）

### 4.1 ★ 不安全的第一版（`max` 行数）——**留档，不许上线**

【实测】臂 `p2-a2-p2-144m`（`OFFLOAD_BYTES=144 MiB` = 1152 unit）：
worker 物理池 **223,690,752 B（0.208 GiB）= 旧口径的 1/4.69**，
`BlockStored=714`、`CPU→GPU=273 MB`、`hits=65,520`、输出 sha256 与基线相同。

⚠️ 但这一版**把 10 个 SWA 组的行区间重叠**了（每组的行都从 0 开始），
同一张张量的同一行会被多个组写 ⇒ **原理上必然互相覆盖**（本次 `max_tokens=1` 的输出
恰好没露出来，**不足以作为正确性证据**）。⇒ 已改为 §3 的"分量行空间"，并加了冲突断言。

### 4.2 ★ 修复后（分量行空间）的六条臂（全部 rc=0）

原始数据在 [`raw/030-p2-poolsizing/`](raw/030-p2-poolsizing/)（每条臂的 `server.log` /
`client.json` / `metrics_after.txt` / `meta.txt`）。

| 臂 | P2 | knob | 工作集 | unit | **宿主分配（worker 日志）** | alloc / 淘汰 | CPU→GPU | hits | replay p50 | sha256 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `p2-i1-base-144m` | 0 | 144 MiB | 16×4096 | 1152 | **1,048,559,616 B（1000.0 MiB）** | 714 | 273 MB | 65,520 | 48.7 ms | `d23082b3…` ✅ |
| **`p2-j1-p2-144m-comp`** | **1** | 144 MiB | 16×4096 | 1152 | ★ **535,265,280 B（510.5 MiB）= 1.96×** | 714 | 273 MB | 65,520 | 49.8 ms | `d23082b3…` ✅ |
| `p2-n1-base-675m-8k` | 0 | 675 MiB | 16×8192 | 5400 | **4,915,123,200 B（4.578 GiB）** | 1418 | 462 MB | 131,056 | 56.3 ms | `aff7a578…` ✅ |
| **`p2-k1-p2-675m-8k`** | **1** | 675 MiB | 16×8192 | 5400 | ★ **2,509,056,000 B（2.337 GiB）= 1.96×** | 1418 | 462 MB | 131,056 | 59.4 ms | `aff7a578…` ✅ |
| `p2-l1-base-100m` | 0 | 100 MiB | 16×4096 | 800 | 728,166,400 B（0.678 GiB） | **1418 / 928** | **0** | **0** | **465.2 ms** | `d23082b3…` ✅ |
| **`p2-m1-p2-196m`** | **1** | 196 MiB | 16×4096 | 1568 | ★ **728,536,448 B（0.679 GiB）** | 714 | **273 MB** | **65,520** | **47.6 ms** | `d23082b3…` ✅ |

**判据 B（四条判据）**：P2 臂逐项与基线一致 —— `BlockStored:CPU=714` 条、
`CPU→GPU=273 MB`、`hits=65,520/131,328`、**replay 47.6–49.8 ms ≪ fill 472–474 ms（9.5–9.9×）**。
**判据 C（sha256）**：16×4096 臂 = `d23082b36fc1146e2fa3fabc6399dad8acdee4d54d25d5645be42f081d825789`
（与 `logs/017`/`021` 逐字相同）；16×8192 臂 = `aff7a578…`（**P2 与基线逐字相同**）⇒ 补丁不改数值。

### 4.3 ★★ 判据 A（最重要的那条）

⚠️ **先把话说准**：在 `logs/029` 的 **L2 修正后**，`cpu_bytes_to_use` 决定的是 **unit 数**
（与张量行数无关）⇒ "**同一个 knob** 能存更多条目" **不成立**。
真正成立、也是真正有用的表述是：

> **同一份宿主 RAM，P2 能买到的 unit（= 条目）是基线的 1.96×。**

【实测】两条**宿主几乎完全相等**的臂：

| | `p2-l1-base-100m`（基线） | `p2-m1-p2-196m`（P2） |
|---|---:|---:|
| knob | 100 MiB | **196 MiB** |
| unit | 800 | **1568（1.96×）** |
| **宿主分配** | **728,166,400 B** | **728,536,448 B**（+0.05%） |
| 存进池子 | 1418 条、**淘汰 928 条** | **1418 条、淘汰 0** |
| `CPU→GPU` | **0 B** | **272,957,440 B** |
| hits | **0** | **65,520** |
| replay p50 | **465.2 ms（= 冷算）** | **47.6 ms（9.9×）** |

⇒ **同样 0.68 GiB 宿主：基线一条都取不回来，P2 全中。** 这就是用户要的"能放更多 KV"。

### 4.4 ★ 判据 D（宿主实占）与一个**意外发现**

主证据是 worker 的**分配字节**（上表「宿主分配」列，`P2_WORKER_HOST_BYTES`）。
为了确认"分配 = 实占"，用 `scripts/probe_rss.py` 直接量了 RSS（**不占卡**）：

| 量 | 结果 |
|---|---|
| `torch.zeros((1152, page), int8)` × 16 张（分配 0.977 GiB） | **ΔRSS = 1,047,832 KiB（0.999 GiB）** ⇒ **分配即刻常驻**（分配数可以当实占用） |
| 同上 + `fill_(1)` 触碰每一页 | ΔRSS = 1,048,220 KiB（几乎不变 ⇒ 本来就是常驻，不是 lazy） |
| 本补丁的行数形状（分配 0.499 GiB） | ΔRSS = 339,968 KiB（分配器复用导致偏低，**只作旁证**） |
| ★ **第一次 `aclrtHostRegister`**（P1 的 β 后端） | **ΔRSS 再 +589 MiB**；**后续 15 次注册每张都是 0–4 KiB** |

⇒ ★ **新发现（【实测】+【未确认】原因）**：`aclrtHostRegister` 有**每进程一次性的 ~589 MiB**
RSS 开销（不是每张张量）。A2 是 **8 个 worker 进程** ⇒ 记账时**要预留 ~4.7 GiB**；
它是否是 `acl.init()`/`set_device` 引起、能否共用一次，本轮**没有定位**（留作后续）。

---

### 4.5 ★ 主证据：`worker_kv_bytes_per_block` vs `Σ(page)×bpc`（主代理 2026-09-22 指定）

【实测】每条臂的 worker 都会打印这一行（`③ ★ 记账对账`）：

```
worker_kv_bytes_per_block=131072   Σpage=910208   Σ(page)×bpc=910208
aligned_kv_bytes_per_chunk(单位池)=131072   ⇒ 旧口径主机/记账 = 910208/131072 = 6.944×
```

⇒ **一步钉死 `logs/029` 那 6.944× 的来源**：
`worker_kv_bytes_per_block` 是「**一张**张量的 page」（= `total_gpu_kv_bytes // num_blocks`
在 hybrid 几何下塌缩成的那一个数），而池子每行的真实字节是 **`Σ_i page_i × bpc` = 910,208**
（16 张张量之和）⇒ 比值 **6.944×**，与 `logs/016` 从 A2 真机反解的 **6.948** 一致。

**⇒ 这正是 L2 要修的地方**：把 `worker_kv_bytes_per_block` 改成 `Σ_i(page_i)×bpc`
（或让 `num_blocks` 直接用 `cpu_bytes_to_use // Σ(page_i)×bpc`），
旋钮才会等于"宿主字节"，而不是"宿主字节 ÷ 6.944"。

---

## 5. A2 的新容量账（`019` §10 的重算）

### 5.1 三个乘数各自的作用（互相正交）

| 杠杆 | 作用 | 【实测】数值 |
|---|---|---|
| **L1（本补丁）** | **同一份宿主 RAM 装 1.96× 的条目**（按组配额 + 分量行空间） | **1.96×**（§4.2/4.3） |
| **L2（`029` 建议）** | 让旋钮诚实：`worker_kv_bytes_per_block` 用 **Σ over 16**（= 910,208 B）而不是 1 张张量的 page（131,072 B） | 6.944×（§4.5【实测】）⇒ 修好后 **knob 值 = 宿主字节**，不再有隐形放大 |
| L5（`logs/021`，已上线） | per-group bpc + SWA 对齐裁剪 | 5.05×（32K/128K 口径） |

### 5.2 ★ 新的并发 × 上下文边界表（**L1 + L2 + L5**，×8 rank）

【实测】每 unit 的宿主（全 8 rank）：**旧 7,281,664 B → 新 3,711,616 B**
（= `Σpage 910,208 × 8` 与 `535,265,280/1152 × 8`，两条臂逐字吻合）。
unit/请求沿用 `019` §10 的 L5 实测口径（32K=608、128K=2,432）：

| 场景（最坏口径：每个请求前缀互不相同） | unit（×并发） | 旧宿主（×7.28 MB） | **新宿主（×3.71 MB）** | A2（余量 442 GiB） |
|---|---:|---:|---:|---|
| 16 × 32K | 9,728 | 66.0 GiB | **33.6 GiB** | ✅ 宽裕 |
| 16 × 128K | 38,912 | 264.0 GiB | **134.4 GiB** | ✅ 可以（30%） |
| **32 × 128K** | 77,824 | **528.0 GiB** ⛔ | ★ **268.9 GiB** | ✅ **可以（61%）** |
| 64 × 128K | 155,648 | 1,056.0 GiB ⛔ | 537.7 GiB | ⛔ 仍不可能 |

⇒ ★ **L1 把"32 并发 × 128K"从不可能变成可行**（1056 GiB → 269 GiB），
而这正是用户要的那一格。**代价**：`cpu_bytes_to_use` 必须按 **L2 修好后才等于宿主字节**；
在当前（未修）口径下要**再乘 6.944** 才是实际占用。

> **共享前缀场景**（真实 agent 流量：同一 system prompt + 变长历史）**池子只需覆盖唯一前缀**，
> 比上表小一个数量级 —— 这一格 `L3_8card` 正在实测（`019` §10.2）。
> **注意**：L1 是"同样的 RAM 装更多"，**不改变 A2 的 RAM 硬上限**，
> 且 A2 上还要为 §4.4 那 **~589 MiB/进程** 的注册开销留 ~4.7 GiB。

---

## 6. ★ 改动风险面（主代理 2026-09-22 指定要写进结论）

029 原本设想"按组独立名空间"只需改池分配。**实测下来要碰 3 处**，而且每一处都有独立的失败模式：

| # | 改哪里 | 具体改动 | 失败模式 / 风险 | 本轮怎么处理 |
|---|---|---|---|---|
| **R1** | worker 池分配（`native/cpu_npu.py`） | 每张张量行数 = 引用它的组里最大**行区间上界** | 行数给小 ⇒ **DMA 越界**（静默写坏别的分配 / SIGSEGV） | 行数由 `max(组区间上界)` 算出，并在 worker 内**校验行区间互不相交**（冲突直接拒绝启动） |
| **R2** | 调度侧 unit→行的映射（`manager` 的 `_get_load_store_spec`） | unit 号是 `(group,row)`，交给 worker 前**解码成组内行号** | 两侧算出的行空间不一致（权重/分量不同）⇒ **数据错位**（能跑、能命中、**结果错**） | 权重/分量都由**同一个函数**在两侧各自算出（只依赖 `kv_cache_config` + `extra_config`），并在 worker 侧按**真实 `group_data_refs`** 反向校验分量（不细于真值才放行） |
| **R3** | 淘汰语义（`prepare_store` → `CachePolicy.evict`） | **按组淘汰**（惰性 `__contains__` 过滤器） | 若按全局淘汰 ⇒ 把别的组的配额抢走，长期会让某些组饿死（命中率不规则下降） | 已实现按组淘汰；**代价**：某组配额用尽而本组无空闲可淘汰条目时，本轮 store 直接放弃（= 上游 `prepare_store` 返回 `None` 的既有语义），**不会**去借别组 |

**另外两条上不了压力测试就看不出来的风险**：

| # | 风险 | 现状 |
|---|---|---|
| R4 | **配额是静态切分**（`N_g = N × w_g / Σw`）：真实流量的组混合比例若与本配置的 `w_g` 差很远，某些组会先耗尽 ⇒ 命中率下降（但**不会错**，只是少存） | 【未确认】tiny 上 16 请求均衡负载没触发；**A2 上要用真实 trace 压测** |
| R5 | 按组淘汰时 `evict()` 要沿 LRU **跳过别的组**（惰性过滤器，但仍是线性扫描）：池子很大（A2 上 5.9 万 unit）时可能变慢 | 【未确认】tiny 上 replay p50 与基线同档（47.6 vs 48.7 ms）；**A2 上要计时** |

**兼容性 / 回滚**：

```
P2_POOL_PATCH=0（默认）  ⇒ 逐字回退 logs/021 的行为（本轮 I/N/L 三条基线臂就是这么跑的，
                            它们与本补丁无关地证明了基线未变）
P2_STRUCT_LOG=1（只读）  ⇒ 只打"真实分量/记账对账/旧口径宿主字节"，不改任何行为
回滚 = 不设 P2_POOL_PATCH（或把它设 0）；补丁本体在 PYTHONPATH 里，进程结束即消失
```

---

## 7. ★ cannbot 对照（`a2/AGENTS.md` §6 要求）

本任务**不写 AscendC kernel、不做量化数值验证**（改的是**宿主侧池子的容量分配**），
但仍按要求查了 KV cache 那一节：

| 查了哪里 | cannbot 说什么 | 我们怎么用 |
|---|---|---|
| `vendor/cannbot-skills/model/model-infer-kvcache/SKILL.md:38-56`（§1.1 KV 模式选型） | 改造目标是 **Paged（FA + TND/TND_NTD）**；含滑窗的模型要带**滑窗约束**（`attn_type="SlidingWindow"` / `sparse_mode=4`）；MLA 走 TND_NTD + absorb | 我们的 `full` 组 = MLA/long-KV，10 个 SWA 组 = 滑窗 ⇒ **本补丁的裁剪/配额规则与它描述的两类 attention 形态一一对应**（SWA 组每对齐段只留 1 个 chunk，正是"滑窗约束"在卸载层的体现） |
| 同上 `:102-110`（§2.1 物理布局与逻辑映射） | 物理存储 `[total_num_blocks, block_size, num_kv_heads, head_dim]`；**物理 slot = 物理 block ID × block_size + 块内偏移**；`BlockPool` 动态分配 block | ★ **这就是我们的 V1/V3 的"官方版本"**：cannbot 的 `BlockPool` 也是**一个** block-id 名空间、一张张量按 `total_num_blocks` 行分配 ⇒ 镜像里"16 张张量各分 `num_blocks` 行"正是照搬这个布局。**我们没改映射公式**（slot = block_id × …），只改了**每张张量各分多少行**）—— 所以对算子/布局**零影响** |

**没采纳的部分**：cannbot 那份 §1.1 建议"滑窗模型用 `sparse_mode=4` + 另一套 attn_type 接入"，
那是**换 attention 实现**（算子级），与本轮的**容量分配**正交，且会动到模型执行路径 ⇒ 本轮不碰。

---

## 8. 诚实边界（哪些是实测、哪些还没测）

| # | 事项 | 强度 |
|---|---|---|
| 1 | V1/V2/V3 三条代码结论 | 【实测·代码级】（本地副本 md5 与镜像/`021` refs 逐字节一致） |
| 2 | 16 张张量的 group→tensor 映射、每 1024 token 的 unit 构成（8+10=18） | 【实测】（worker 在臂里直接打印 + `BlockStored=714`/1152 unit 交叉核对） |
| 3 | 单卡倍率 **1.96×**（宿主字节，两条不同工作集上都是 1.96×）与四条判据、sha256 | 【实测】（§4.2–4.3） |
| 4 | 判据 A（同宿主：基线 0 命中 / P2 全中） | 【实测】（§4.3） |
| 5 | "分配 = 常驻"（`torch.zeros` 不 lazy）+ 注册的一次性 589 MiB | 【实测】（§4.4，`probe_rss.py`）；**原因【未确认】** |
| 6 | A2 的容量表（§5.2） | 【推断】**按 tiny 实测几何 × A2 的同一份 page 多重集（Σ=910,208 B）外推**；A2 上**还没实测** |
| 7 | A2 的 **13 个组**（多一个 `dspark`）在分量里的归属 | 【未确认】tiny 只有 12 个组；A2 的 `dspark` 引用哪几张张量要在 A2 上打一次结构日志才知道（**本补丁已内置这行日志**，见 §9 用法） |
| 8 | **配额是静态切分**（R4） | 【已知代价】不会错、只会少存；A2 真实 trace 要压测 |
| 9 | R5（大池下按组淘汰的速度） | 【未确认】 |
| 10 | 第一版（`max` 行数）4.69× 的数字 | ⛔ **不可用**（行区间重叠，会静默覆盖）—— 留档只为说明"为什么必须做分量校验" |
| 11 | 用户原问题"DRAM 能放的 kvcache 看起来太少了"的**剩余**部分 | ⚠️ L1 只解决"同样 RAM 装更多"；**A2 的 RAM 是硬上限**。要真正"大很多"仍要 L3（state ring 缩 BF16，`logs/024`）与 KV8 |

---

## 9. 复现

```bash
# 0) 同步（本地 → A3-node1，走 COS）
bash a2/agents/P2_poolsizing/scripts/upload.sh
timeout 90 ssh -o ControlPath=none A3-node1 "bash -s" < a2/agents/P2_poolsizing/scripts/upload.sh fetch

# 1) 不占卡的算术探针（< 10 s；退出码 75 = 没抢到锁）
timeout 240 ssh -o ControlPath=none A3-node1 "cd ~/projects/dsv41-upstream-pr && \
  bash tools/a3_chip.sh c2 --timeout 120 --name p2-probe -- \
  python3 /work/agents/P2_poolsizing/scripts/probe_quota.py"

# 2) 四条臂（E 基线 / F 本补丁 / G 大池 / H 大池基线），~2 min/臂
timeout 900 ssh -o ControlPath=none A3-node1 "cd ~/projects/dsv41-upstream-pr && \
  bash tools/a3_chip.sh c2 --timeout 600 --name p2-batch2 -- \
  bash /work/agents/P2_poolsizing/scripts/run_batch2.sh"

# 3) 不占卡：宿主实占机制探针（判据 D；RSS，不碰 NPU 计算）
timeout 200 ssh -o ControlPath=none A3-node1 "cd ~/projects/dsv41-upstream-pr && \
  bash tools/a3_chip.sh c2 --timeout 300 --name p2-rss -- \
  python3 /work/agents/P2_poolsizing/scripts/probe_rss.py"

# 4) ★ A2 上怎么用（**默认关，开了才走新分配**）
#    a) 先跑一条**只读结构臂**拿到真实分量与旧宿主字节：
#       P2_STRUCT_LOG=1 P2_POOL_PATCH=0 ...   → 看 "[P2_poolsizing] ③ ★ 真实分量" 与
#                                                "③ ★ 记账对账" 两行
#    b) 再用拿到的分量开补丁（分量写错会被 worker 拒绝启动，不会静默出错）：
#       P2_POOL_PATCH=1 P2_COMP_JSON='[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]' \
#       PYTHONPATH=<影子包>:/work/agents/P2_poolsizing/patch \
#       OFFLOAD_SCHED_PATCH=1 OFFLOAD_NPU_WORKER_PATCH=1 NPU_OFFLOAD_HOST_MEM=registered ...
```

**A2 上第一个该看的数**（决定 A2 的分量到底怎么切）：

```bash
grep -a "P2_poolsizing" <server.log> | grep -E "真实分量|记账对账|宿主实占"
```

---

## 10. 交付物

| 位置 | 内容 |
|---|---|
| [`agents/P2_poolsizing/patch/`](../agents/P2_poolsizing/patch/) | `p2_pool.py`（配额/行空间/分量算术 + manager 工厂）、`p2_hooks.py`（三个钩子 + 只读结构日志）、`sitecustomize.py`（叠加 `logs/021` 后挂载） |
| [`agents/P2_poolsizing/scripts/`](../agents/P2_poolsizing/scripts/) | `probe_quota.py`（不占卡探针）、`probe_rss.py`（判据 D 机制）、`run_arm_p2.sh`、`run_batch{2,3,5,6}.sh`、`upload.sh` |
| [`agents/P2_poolsizing/README.md`](../agents/P2_poolsizing/README.md) | 一句话结论 + 怎么用 + 开关表 |
| [`logs/raw/030-p2-poolsizing/`](raw/030-p2-poolsizing/) | 10 条臂的原始 `server.log` / `client.json` / `metrics_after.txt` / `meta.txt` + `probes.log`（两个不占卡探针的完整输出） |
