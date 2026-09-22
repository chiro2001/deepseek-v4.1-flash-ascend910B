# 014 · pinned host 池：**单次 vs 总量**、候选 α/β，以及 A2 该用哪个 API

**日期**：2026-09-22 00:22 – 00:55（A3 本地时钟；五轮电池 + A2 探针自检）　**执行**：子代理 `P1_pinned`（任务书 `/root/p1_pinned`）
**机器**：A3（A3-node1）槽位 **c0 = die3**（避开 c1）　**容器**：`prbench-c0`　**上游**：vLLM `0.27.1` + vllm-ascend `e43cf1e9f`
**标记约定**：【实测】= 本机原始数据（`a2/logs/raw/014-*`，**39 个**文件）；【推断】= 推出来但没直接测；【未确认】= 没跑到。

---

## 0. 一句话结论（★ 第一条推翻了任务的**前提**）

| # | 结论 | 强度 |
|---|---|---|
| **1** | **`logs/009` §3.2 的"pinned 单次分配上限 ∈ (4,8] GiB"不成立。** 干净进程里**单次** `torch.zeros(..., pin_memory=True)` 逐档跑到 **24 GiB 全成功**（2/4/5/6/7/8/10/12/14/16/20/24 GiB）；**同一进程累加** 256 MiB × 512 = **128 GiB 也全成功** | 【实测】 |
| **2** | 把真实臂的形状（**16 块、首块 ≈97%、8 进程并发、每进程 8 / 16 GiB**）在干净容器里复刻：**8/8 进程成功，合计 64 GiB / 128 GiB**。⇒ 也**不是**"形状 / 并发 / 总量" | 【实测】 |
| **3** | 先占 **48 GiB HBM** 再 pin：4/8/12/16 GiB **仍然全部成功** ⇒ 不是显存压力 | 【实测】 |
| **4** | ⇒ 真实八卡臂的 `207001` **不是容量公式问题**；更像"**该进程当时已用掉的驱动侧 host 资源 + 邻居容器**"这类**态**问题。**没有拿到可复现的最小条件** | 【推断】+【未确认】 |
| **5** | **候选 β 的生–死判据通过**：`aclrtHostRegister(ptr, size, flags=0)` 注册的普通 host 内存**能走真实 KV 拷贝路径**，**16 GiB / 24 GiB 单块往返逐字节一致**，带宽 **H2D 58 GB/s / D2H 42.7 GB/s**（与 pinned 同级），且**不经过** `aclrtMallocHost` | 【实测】 |
| **6** | ⇒ **A2 的 260 GB 计划改成：池子用"普通 host 内存 + `aclrtHostRegister(MAPPED)`"，不要用 `aclrtMallocHost` 撑大池**；改动只有 `native/cpu_npu.py` **一处分配**（生成器 + 补丁已备好，§5） | 【实测】+【推断】 |

> **对任务书三个判断的修正**（都基于本轮【实测】）：
> 1. "**与 Engram 无关**" —— ✅ 对（本轮全程 `ENGRAM` 未参与，连 `host_mem_pool` 都没关系）。
> 2. "**触发条件是单次 pinned 分配过大**" —— ❌ **不成立**（单次 24 GiB 都过）。
> 3. "**A2 必须先解决'怎么要到几百 GB 的 DRAM 池'**" —— ✅ 仍然成立，但**卡点不在"单次大小"**，
>    而在"**驱动侧 pinned 路径在真实多 worker 场景下不可靠**"⇒ 用 β 绕开它。

---

## 1. 环境与复现入口

| 项 | 值 |
|---|---|
| 机器 / 槽位 | A3-node1，**c0 = 物理 die 3**（c1 全程未碰）；锁 = `locks/c0.lock`（拿不到 = 退出码 75） |
| 容器 | `prbench-c0`（`/work` = `~/projects/dsv41-upstream-pr`）；`ASCEND_RT_VISIBLE_DEVICES=0` |
| 起跑时宿主 | `MemTotal 2013 GiB / MemAvailable 856–860 GiB`；`host_mem_pool=1`、`host_pin_pre_register=1`、`mem_host_uva=1` |
| 容器 `/dev/shm` | ★ **只有 64 MiB**（Docker 默认）——见 §4.4 的坑 |
| python | `torch 2.10.0+cpu`、`torch_npu`、`acl`（CANN 9.1.0） |
| 探针入口 | `agents/P1_pinned/scripts/`（A3 上 `/work/agents/P1_pinned/scripts/`） |
| 五轮电池的复用命令 | `bash agents/P1_pinned/scripts/run_probe{,2,3,4,5}.sh`（每次只拿 c0 锁、跑完即放；单轮 1–7 min） |
| 原始数据 | `a2/logs/raw/014-*.txt`（**39 个**，A3 原文件在 `agents/P1_pinned/out/` 与 `out2/`） |

**本轮没有**：起服务、加载模型、写 `upstream-v41/`、碰 `dsv4-a3`/`mooncake-master`、碰 Phy-ID 8–15、用 `/tmp`。

---

## 2. 第 1 步【实测】：**是单次上限还是总量上限？**

### 2.1 (a) 单次大分配 —— **没有找到上限**

`pinned_budget_probe.py single <GiB>`，**每档一个干净进程**：

| 单次请求 | 结果 | 原始数据 |
|---:|---|---|
| 2 / 4 / 5 / 6 / 7 / 8 / 10 / 12 GiB | ✅ 全成功（0.9–1.4 s/次） | `014-10-single-fresh.txt` |
| **14 / 16 / 20 / 24 GiB** | ✅ **全成功**（1.5–2.1 s/次，`ptr=0x80000000000`，`is_pinned=True`） | `014-30-single-fresh-big.txt` |
| 单进程**逐档分配→释放** 4→12 GiB | ✅ 全成功（说明释放后再要也 OK） | `014-11-single-ladder.txt` |
| **裸 `aclrtMallocHost`**（ctypes 直连，先 `aclrtSetDevice`）4 GiB ×2 + 8 GiB ×2 | ✅ **24 GiB**（0.06–0.12 s 每次） | `014-37-raw-with-device.txt` |

> ⇒ `logs/009` §3.2 的"上限落在 (4, 8] GiB / worker"**在干净进程里复现不出来**：8 GiB、16 GiB、
> 甚至 **24 GiB 单次**都过。**注意**：第一轮 raw 模式曾报 `rc=107002`，那是探针忘了 `aclrtSetDevice`
> 的假象（见 `014-15-*.txt` vs `014-37-*.txt`），**不是** pinned 容量问题。

### 2.2 (b) 多次小分配累加 —— **128 GiB 也没触顶**

| 形状 | 结果 | 原始数据 |
|---|---|---|
| 256 MiB × 512 = **128 GiB**（同一进程） | ✅ 512 块全成功，总耗时 8.6 s | `014-12-multi-256mib-to128gib.txt` |
| 1 GiB × 128 = **128 GiB**（同一进程） | ✅ 128 块全成功，总耗时 8.6 s | `014-13-multi-1024mib-to128gib.txt` |
| 2 × 4 GiB 大块 + 224 × 256 MiB = **64 GiB** | ✅ 全成功 | `014-14-mixed-2x4gib-then256mib.txt` |

### 2.3 (c) 混测 + **真实形状复刻** + HBM 压力

| 试验 | 结果 | 原始数据 |
|---|---|---|
| 真实形状（16 块，首块 15.50 GiB ≈97%）**单进程** 16 GiB | ✅ | `014-31-uneven-single-16gib.txt` |
| 同形状 **8 进程 × 8 GiB = 64 GiB** | ✅ **8/8**（各 0.5–0.6 s，总 11.9 s） | `014-32-uneven-mp8-8gib.txt` |
| 同形状 **8 进程 × 16 GiB = 128 GiB**（= `d2-dram128-6p` 那条挂掉的臂的分配形态） | ✅ **8/8**（总 13.3 s） | `014-33-uneven-mp8-16gib.txt` |
| 先占 **48 GiB HBM**（剩 13.1 GiB 显存）再 pin 4/8/12/16 GiB | ✅ **全成功** | `014-34-devpressure-48gib.txt` |
| D_off8 早先的旁证 | 单进程 1 GiB 步长 pin 到 128 GiB ✅；8 进程同卡各 16 GiB ✅ | `upstream-v41/logs/51` §4.2 |

### 2.4 判据与它给出的答案

任务书的判据是：**(b) 能累加到远超 (a) 的单次上限 ⇒ 是单次限制 ⇒ 候选 α 可行**。
本轮的实测答案是**更彻底**的一条：

> **在干净进程里，(a) 单次到 24 GiB、(b) 总量到 128 GiB、8 进程并发 128 GiB、HBM 占 48 GiB —— 全都不是限制。**
> ⇒ **α（分片拼池）解决不了真实臂的 `207001`**，因为那个 `207001` 不是这两类上限造成的。
> ⇒ 真实臂的触发条件**仍未复现**（【未确认】），见 §6 诚实清单。

---

## 3. 第 2 步：池子在**哪里**、用**什么 API** 申请的

【实测·代码】（镜像 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3` 内
`/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/`）：

| 文件 | 作用 | 关键行 |
|---|---|---|
| `npu.py` | `NPUOffloadingSpec`（把 vLLM 的 `CPUOffloadingSpec` 换成 Ascend worker） | `create_worker()` 直接 `NPUOffloadingWorker(...)`；`num_chunks = cpu_bytes_to_use // aligned(kv_bytes_per_chunk)` |
| **`cpu_npu.py`** | ★ **CPU 池真正分配的地方** | `NPUOffloadingWorker.__init__`：`cpu_tensor = torch.zeros((num_cpu_blocks, cpu_page_size_bytes), dtype=torch.int8, device="cpu", **pin_memory=pin_memory**)` ——**每个 canonical KV tensor 一次**（DSV4.1 = **16 次**） |
| `offloading_connector.py` | 把 Ascend 的 KV cache 张量规整成 canonical `[num_blocks, page_bytes]` 视图 | `_make_int8_block_view()`（零拷贝 view） |
| （上游）`vllm/v1/kv_offload/cpu/gpu_worker.py` | D2H/H2D 的描述符与搬运 | 走 `torch.ops._C_ascend.swap_blocks_batch`（`csrc/torch_binding.cpp` di 内部 = `aclrtMemcpyAsync`/`aclrtMemcpyBatchAsync`） |

**结论**：

1. 池子**只有一处**分配 —— `cpu_npu.py` 里那句 `torch.zeros(..., pin_memory=True)`；
   `pin_memory=True` ⇒ torch_npu `CachingHostAllocator` ⇒ **`aclrtMallocHostWithCfg`**（就是 `207001` 的出处）。
2. **调度器/manager 侧不分配 host 内存**（只记 block 号），所以改一处就对全局生效。
3. 上游**本来就有一条 mmap 池**（`vllm/v1/kv_offload/cpu/shared_offload_region.py`，`/dev/shm/vllm_offload_*.mmap`
   + `MADV_POPULATE_WRITE`），但 `CPUOffloadingSpec._uses_shared_region()` 限定
   `current_platform.is_cuda_alike()` ⇒ **在 NPU 上被关掉**（`NPUOffloadingSpec` 绕开它直接建 worker）。
   ⇒ 【推断】"把池子换成 mmap"在上游已经是**同构改动**，我们只需在 NPU 侧补一步"**注册**"。
4. ★ 但上游那条 `/dev/shm` 路在容器里**不能直接用**：`/dev/shm` 只有 **64 MiB**，
   写第 65 MiB 会 **SIGBUS**（本轮实测：`014-63-regfile-prefaulted.txt` 直接 `Bus error (core dumped)`）。
   要用文件映射就**必须** `--shm-size`，或**映射磁盘文件/匿名内存**。

---

## 4. 第 3 步【实测】：候选 β 的生死判据 —— 注册内存能不能被 D2H/H2D 用？

### 4.1 API 与 flag（先钉死参数）

| 探针 | 结果 | 原始数据 |
|---|---|---|
| `aclrtHostRegister(ptr, size, flags)` 的 flags 扫描 1 GiB | **只有 `flags=0` 成功**（ret=0，返回 device 指针）；`flags=1/2/4` → **`ret=207000`** | `014-35-hostregister-flags.txt` |
| 注册后 `tensor.is_pinned()` | **变成 `True`**（torch 能看到注册态） | `014-35-*.txt` / `014-40-*.txt` |
| 注册耗时 | **16 GiB → 5.95 s（0.37 s/GiB）**；24 GiB → 8.94 s；Engram 206 GiB → 122 s（0.58 s/GiB，`refs/40`） | `014-41/42-*.txt` |
| 重复注册同一地址段 | **`ret=507899`**（第二次注册失败）⇒ 生产里"一次注册、别重试"即可 | `014-70/71-*.txt` |
| `/dev/shm` 文件映射（**未预热页**） | **`ret=107017`**（未落页的 VMA 注册被拒；与 `refs/40` 的只读 VMA 同款错误） | `014-36-hostregister-devshm-file.txt` |
| `/dev/shm` 文件映射（预热页后） | **SIGBUS**（64 MiB 上限，见 §3.4） | `014-63-regfile-prefaulted.txt` |
| **磁盘文件**映射（`MAP_SHARED` + 预热页）1 / 8 GiB | ✅ **`ret=0`**（0.0 s / 0.2 s）⇒ 预热页就是那个 `107017` 的解 | `014-81-a2probe-copy-on-a3.txt` |

### 4.2 ★ 拷贝判据：**通过**

用 **torch 自己的拷贝路径**（`dev.copy_(host)` / `back.copy_(dev)`，与生产里 torch_npu 的流/事件语义一致），
三种 host 后端逐字节对账（`torch_copy_check.py`）：

| 后端 | 1 GiB ×3 次 | 8 GiB ×2 次 | H2D / D2H 带宽 |
|---|---|---|---|
| `pinned`（现状） | ✅✅✅ | ✅✅ | 56–58 / 42.0–42.7 GB/s |
| `pageable`（普通内存，**不注册**） | ✅✅✅ | ✅✅ | 56.3–58.5 / 42.1–42.7 GB/s |
| **`registered`（mmap/普通内存 + `aclrtHostRegister(MAPPED)`）** | ✅✅（第 3 次是"重复注册"507899，非拷贝问题） | ✅（同上） | **56.5–58.3 / 42.0–42.7 GB/s** |

原始数据：`a2/logs/raw/014-70-torch-copy-1gib.txt`、`014-71-torch-copy-8gib.txt`。

**另有两条独立佐证（更大尺寸 + 裸 acl 路径）**：
* 裸 `acl.rt.memcpy_async` 的 D2H：**16 GiB 与 24 GiB 注册块全部 bit 一致**，`42.0 GB/s`，**真异步**
  （下发 0.2–0.3 ms / 同步 409–613 ms）—— `014-41/42-*.txt`；
* 同一探针对 `pinned` 8 GiB 的 D2H 也是 `41.9 GB/s` bit 一致 —— `014-43-*.txt`。

### 4.3 ⚠️ 一个必须写下来的**反面**结果（以及它的归属）

第 3/4 轮用**裸 `acl.rt.memcpy_async` + 自建 acl 流**做 H2D 时，**三种后端都出现过"H2D 后设备里少了
0.05%–1.1% 的字节（0 值）"**（`014-50/60/61-*.txt`）。
**但同一批内存换 torch 的 `copy_()` 就 100% 一致**（§4.2，1 GiB 与 8 GiB、三种后端、共 12 次）。

⇒ 【实测·归因】**那是我的裸探针的用法问题**（自建 acl 流、没有 torch_npu 的等待语义），
**不是** registered/pageable 内存本身的缺陷；**也不是** pinned 的缺陷（pinned 同样中招）。
⇒ 教训：**验 H2D 必须走生产路径（`swap_blocks_batch` 或 torch `copy_`），不要用裸 `aclrtMemcpyAsync` 下结论。**
（该容器里没有 `swap_blocks_batch` 这个算子 —— 它是这个镜像的 A3 版 csrc；A2 镜像里的算子名要用
`hasattr(torch.ops, "_C_ascend")` + 试调来确认，见 §5.4 的坑。）

### 4.4 候选 β 的结论

| 问题 | 答案 |
|---|---|
| 池子能不能换成 `mmap + aclrtHostRegister`？ | **能**【实测】。普通 `torch.zeros(..., pin_memory=False)` 的内存直接注册就可用，**不需要**先 mmap 文件 |
| 要改多少代码？ | **一处分配**（`cpu_npu.py` 里 `torch.zeros` 那一句）→ helper；补丁生成器 6.5 KB，见 §5.2 |
| 注册内存能不能被 D2H/H2D 用？ | **能**，且 **bit 一致**、带宽与 pinned 同级（H2D 58 / D2H 42.7 GB/s）【实测】 |
| 要不要分片？ | **β 下不需要**（单块 24 GiB 已过；而且池子天然就是"每个 KV tensor 一块"=16 块） |
| 有没有"总量上限"？ | 未测到：注册 24 GiB 单块（`014-42`）；A3 上 Engram 注册过 **206 GiB**（`refs/40`）。**A2 上必须自己量**（§5.3） |

### 4.5 补丁本身的烟测（【实测】PASS）

`patch_smoke_test.sh`（A3，c0 锁内，真实 `prbench-c0` 环境）把补丁版 `cpu_npu.py` 直接 import 进来，
调它的 `_allocate_npu_offload_cpu_tensor(4096, 262144)`（= 1 GiB 池）：

```
[smoke] import OK，helper = True
[smoke] 模块级模式 = registered
[cpu_npu.py:102] [P1_pinned] CPU pool 4096 x 262144 (1.00 GiB): registered dev=0x3ff87e00000 ret=0
[smoke] 池张量 shape=(4096, 262144) is_pinned=True ptr=0xfffed0010000
[smoke] registered 缓冲登记数 = 1
[smoke] ⇒ PASS
```

⇒ 补丁**能在真实镜像里 import、能走 registered 分支、`ret=0`、返回张量 `is_pinned()=True`**。
原始输出：`014-90-patch-smoke-test.txt`。

---

## 5. 第 4 步：**A2 可执行方案**

### 5.1 用哪个 API、池子能到多大

| 项 | 结论 |
|---|---|
| **API** | ✅ `aclrtHostRegister(ptr, size, flags=0)` + **普通 host 内存**（`torch.zeros(..., pin_memory=False)`）；Engram device-index 用的就是这条（`refs/40`：A3 上 206 GiB 注册成功） |
| ❌ 不要用 | `aclrtMallocHost` / `pin_memory=True` 去撑大池 —— 那正是真实八卡臂 `207001` 的出处（原因未复现，但**红线上不要押它**） |
| **池子能到多大** | 由"宿主 RAM"决定，不由驱动池决定：A2 宿主余量 **442 GiB** ⇒ `OFFLOAD_GB=256`（= 32 GB/worker）**在内存账上是安全的**；注册/页固定是**一次性**开销（0.37 s/GiB，8 worker 并行 ⇒ ~12 s/worker）。**A2 必须先按 §5.3 量出实际能注册到多少** |
| **要不要分片** | 【推断】**默认不要**。若 A2 的探测显示"单次注册也有上限"，则**先**把该上限告诉主代理 —— 那时才需要把最大那块（DSV4.1 上占 ~97% 的 full group）再切成多段，改动落在 `SingleDirectionNPUOffloadingHandler` 的指针计算（`compute_sub_block_ptrs` 需要按"段基址+段内偏移"重算），属于**中等改动**，本轮**没有实现** |
| 起服成本 | 注册在 `NPUOffloadingWorker.__init__`（权重加载完、KV cache 建好之后），与现在同一时机，不额外增加一次拷贝 |

### 5.2 改哪些文件（都已在 `a2/agents/P1_pinned/` 备好）

| 文件 | 作用 |
|---|---|
| `scripts/make_cpu_npu_hostmem_patch.py` | 生成器：读镜像内原版 `native/cpu_npu.py` → 输出补丁版（锚点唯一性校验 + `compile()` 自检 + 打印 diff） |
| `patches/offload_dsv41/cpu_npu.py` | 生成好的补丁版（**已对镜像里抽出来的原版 `cpu_npu.py` 复跑过生成器**：328 行 → 388 行，锚点唯一） |
| `scripts/a2_pinned_probe.sh` | ★ A2 一条命令的探测（不占卡、不加载模型，产物落 `~/tmp/<日期>/p1_pinned/`） |
| `scripts/run_probe{,2,3,4,5}.sh` + `in_container{,2,3,4,5}.sh` | A3 上复现本轮五轮电池的入口（拿 c0 锁 → `docker exec prbench-c0`） |
| `scripts/patch_smoke_test.sh` | ★ 补丁烟测：在真实环境里 import 补丁版 `cpu_npu.py` 并注册 1 GiB 池（§4.5） |

**补丁做了什么**（语义上就一件事）：

```python
# 原版（cpu_npu.py，NPUOffloadingWorker.__init__）
cpu_tensor = torch.zeros((num_cpu_blocks, cpu_page_size_bytes),
                         dtype=torch.int8, device="cpu", pin_memory=pin_memory)   # ← aclrtMallocHostWithCfg

# 补丁版
cpu_tensor = _allocate_npu_offload_cpu_tensor(num_cpu_blocks, cpu_page_size_bytes)
```

`_allocate_npu_offload_cpu_tensor()` 按环境变量 `NPU_OFFLOAD_HOST_MEM` 选后端：

| 取值 | 行为 |
|---|---|
| `registered`（**默认**） | `pin_memory=False` 分配 → `aclrtHostRegister(ptr, nbytes, 0)`；成功即用（`is_pinned()` 会变 True） |
| `pageable` | 完全不注册（§4.2 实测往返 bit 一致，带宽同级） |
| `pinned` | 旧行为（回归/对照用） |

注册失败时**自动回落 `pinned`**（= 现状），所以这个改动**不会比现在更差**；注册成功则在日志里打
`[P1_pinned] CPU pool ... registered dev=0x... ret=0`。

**挂载方式**（与 `OFFLOAD_SCHED_PATCH` 同构，新增一个开关）：

```bash
# 1) 生成补丁（在 A3 上先验证过；A2 上也可用同一条命令生成）
python3 agents/P1_pinned/scripts/make_cpu_npu_hostmem_patch.py \
    <镜像内原版 cpu_npu.py> \
    shadow-pkg/patches/files/offload_dsv41/cpu_npu.py

# 2) 起服脚本里加一行（只有 OFFLOAD_NPU_WORKER_PATCH=1 时挂）
[ "$OFFLOAD_NPU_WORKER_PATCH" = 1 ] && MOUNTS+=(-v \
  "$PKG/patches/files/offload_dsv41/cpu_npu.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro")

# 3) 运行时选后端
NPU_OFFLOAD_HOST_MEM=registered    # 或 pageable / pinned
```

> 回滚：`OFFLOAD_NPU_WORKER_PATCH=0`（默认）⇒ 完全回到现在的行为；或 `NPU_OFFLOAD_HOST_MEM=pinned`。

### 5.3 ★ A2 验证命令（**用户粘贴执行**）

**第 0 步（必须先做，不占卡、不加载模型，≈3–5 min）**

```bash
# 把 agents/P1_pinned/scripts/a2_pinned_probe.sh 放到 A2 上（例如 ~/a2_pinned_probe.sh），然后：
#   ① 已经在一个有 CANN + torch_npu 的容器里：
bash ~/a2_pinned_probe.sh
#   ② 或在宿主上、目标容器在跑：
# A2_CONTAINER=<容器名> bash ~/a2_pinned_probe.sh
```

它会（**同一份代码在 A3 上跑过，见 §4**）打印：

| 段 | 内容 | 判读 |
|---|---|---|
| `pinned-single` | 单次 1/4/8/16/32 GiB（`torch.zeros(pin_memory=True)`） | 找出**真实**单次上限 |
| `pinned-multi` | 256 MiB 累加到 128 GiB | 总量上限 |
| `pageable` | 普通内存 D2H/H2D + 逐字节对账 | 不注册能不能用 |
| `register` | 1/8/32/64 GiB 匿名内存注册 + 1 GiB 拷贝对账 | ★ **β 的生死判据** |
| `register-file` | 文件 + `MAP_SHARED` + 注册（Engram 同款；**先预热页**） | 备用形态 |
| `DECISION` | 一行判读 | 直接决定走 β / α |

> ★ **这个脚本本身已在 A3 上跑通**（等于一次"空机自检"，读数见 `014-80/81-a2probe-*-on-a3.txt`）：
> 单次 pinned **1/4/8/16/32 GiB 全过**；累加到 **128 GiB** 过；**注册 1/8/32/64 GiB 全过**
> （64 GiB 用 24.9 s）；磁盘文件映射注册 1/8 GiB 过；拷贝判据
> **H2D 55–58 GB/s / D2H 41 GB/s 往返逐字节一致=True**。
> 脚本里的拷贝对账走的是 **torch `Tensor.copy_`**（生产同款流语义），**不是**裸 `aclrtMemcpyAsync`
> —— 后者会给出假的"H2D 丢字节"（见 §4.3）。

**第 1 步（β，服务级验证）**：按 §5.2 挂上补丁，然后在原臂命令上加两个变量：

```bash
OFFLOAD_NPU_WORKER_PATCH=1 NPU_OFFLOAD_HOST_MEM=registered \
TAG=a2-dram256 ENGRAM=0 OFFLOAD_GB=256 PREFIX_MATCH_UNIT=32 BLOCKS_PER_CHUNK=8 OFFLOAD_SCHED_PATCH=1 \
  bash agents/D_off8/scripts/run_arm_8card.sh      # 口径与 logs/009 的四条臂一致
```

**判据（与 `logs/009` §5 相同的四条 + 两条新增）**：
1. `BlockStored(medium="CPU") > 0`；
2. `kv_offload_total_bytes_total{CPU_to_GPU} > 0`；
3. `external_prefix_cache_hits > 0`（replay 轮应接近 100%）；
4. replay TTFT ≪ fill TTFT；
5. ★ **池子 > 8 GiB/worker 且服务能起来**（= 本轮要解决的那条）；
6. ★ 日志里 8/8 worker 都有 `CPU pool ... registered ... ret=0`（若出现 `falling back to pinned` ⇒ 注册没成，去看第 0 步读数）。

**第 2 步（只在第 0 步显示"注册有上限"时才需要）**：把 `DECISION` 段的原文贴回来，主代理再决定是否上"分片注册"。

### 5.4 A2 上的已知坑（照抄）

1. **`host_mem_pool` 预期 = 0**（A3=1）。`refs/55` §2.1 记载 A2 上 Engram **206 GiB 整表注册 17 min 后
   `ret=207001`**（当时 `MemAvailable` 还有 703 GiB）⇒ **β 在 A2 上也不是白送的**，必须实测（§5.3 第 0 步）。
2. **算子名**：A3 那个镜像里**没有** `torch.ops._C_ascend.swap_blocks_batch`（`prbench-c0` 里实测
   `AttributeError`），它是 `deepseek-v4.1-flash-a3` 镜像/torch_binding 侧才有的名字。
   ⇒ 探针里一律**先 `hasattr` 再试调**，并保留 `acl.rt.memcpy_async` 作为回落（本轮就是这么做的）。
3. **别用 `/dev/shm`**（容器里只有 64 MiB，写超了直接 SIGBUS，`014-63`）。
4. **同一地址段不要重复注册**（`ret=507899`，`014-70/71`）：注册一次、终身使用；重启就随进程释放。
5. 进程退出/`ctrl-c` 时**不需要**显式 `host_unregister`（页随进程释放；Engram 的路径也是这么做的）。

---

## 6. 诚实清单（本轮**没有**解决的）

| # | 项 | 状态 |
|---|---|---|
| 1 | **真实八卡臂 `207001` 的最小复现** | **【未确认】**。单次 24 GiB、总量 128 GiB、8 进程并发 128 GiB、真实形状、HBM 压力 48 GiB —— **全都没复现**。剩下没排除的变量：模型 worker 进程自身的态（权重加载/图编译后驱动侧 host 资源占用）、邻居容器当时的占用、以及"8 个进程**同时** 在**同一时刻**各要 16 GiB"的时序 |
| 2 | `207001` 到底是"池容量"还是"分配粒度/碎片" | **【未确认】**（原 `logs/009` 把它归到"单次大小"，本轮证明该归因不成立） |
| 3 | **A2 的实际上限** | **【未确认】**（A2 本机 ssh 不可达 ⇒ §5.3 第 0 步必须用户粘贴执行） |
| 4 | 分片注册（α 的 β 版） | **【未实现】**：只有单次注册上限被量到才需要，代码改动位置已定位（`compute_sub_block_ptrs` 的段基址化） |
| 5 | 端到端服务级验证（起服 + fill/replay） | **【未做】**：本轮按要求只做到"候选 β 的最小可行形态"（注册 + 真实拷贝路径 + bit 对账），**服务级留 §5.3 第 1 步** |
| 6 | 裸 acl 探针 H2D 丢字节的机理 | **【未确认】**（已用 torch 路径证伪"内存类型"这个解释；但裸流为什么会丢，没有深挖） |
| 7 | 上游 mmap 池（`SharedOffloadRegion`）直接移植 | **【未做】**：它在 `/dev/shm`（容器 64 MiB）⇒ 要用必须改 backing store；本轮选择了更小的改动（就地注册） |

**教训（写给下一个代理）**
1. **"探针全通过、真机必挂"这种悖论，先怀疑结论的归因，不要急着设计补丁。** 本轮一开始照单全收
   "单次上限 (4,8] GiB"，先量了 20 分钟就把它推翻了。
2. **裸 `aclrt*` 流上做 DMA 对账会骗人**：同一批内存，裸流报"丢字节"，torch 路径 100% 一致。
   验 DMA 一律走**生产算子的调用形态**。
3. **walk 别人的探针结果时，注意"探针本身的 bug"**：raw 模式的 `107002` 是漏了 `set_device`；
   `regfile` 的 `107017` 是漏了预热页；`/dev/shm` 的 SIGBUS 是容器默认 64 MiB。

---

## 7. 附：本轮产物

| 路径 | 内容 |
|---|---|
| `a2/logs/014-20260922-pinned-pool-solution.md` | 本文件 |
| `a2/logs/raw/014-*.txt` | **39 个**原始输出（五轮电池 + 环境快照 + A2 探针在 A3 上的自检 2 份 + 补丁烟测 1 份） |
| `a2/agents/P1_pinned/scripts/pinned_budget_probe.py` | (a)/(b)/(c) 单次-总量判定 + 裸 `aclrtMallocHost` |
| `a2/agents/P1_pinned/scripts/pattern_pressure_probe.py` | 真实形状复刻、8 进程并发、HBM 压力、flags 扫描、文件映射注册 |
| `a2/agents/P1_pinned/scripts/hostregister_copy_demo.py` | β 的拷贝判据（三种后端 D2H/H2D + 对账） |
| `a2/agents/P1_pinned/scripts/copy_diag.py` / `h2d_verify.py` / `torch_copy_check.py` | H2D 反面结果的定位与证伪（第 3/4/5 轮） |
| `a2/agents/P1_pinned/scripts/a2_pinned_probe.sh` | ★ **A2 一条命令探测**（自包含，含 DECISION 判读） |
| `a2/agents/P1_pinned/scripts/make_cpu_npu_hostmem_patch.py` | ★ β 补丁生成器（锚点校验 + 语法自检 + diff） |
| `a2/agents/P1_pinned/patches/offload_dsv41/cpu_npu.py` | 生成好的补丁版 `cpu_npu.py` |
| `a2/agents/P1_pinned/scripts/run_probe{,2,3,4,5}.sh` + `in_container{,2,3,4,5}.sh` | A3 五轮电池的复现入口（c0 锁 + `docker exec prbench-c0`） |

**站外写入（A3 侧，全部在 `agents/P1_pinned/` 与 `~/tmp/20260922/p1_pinned/` 之内）**：
`~/projects/dsv41-upstream-pr/agents/P1_pinned/{scripts,patches,out,out2}`、
`~/tmp/20260922/p1_pinned/`（含 `img/cpu_npu.orig.py` 与补丁产物）、`/dev/shm/p1_regfile.bin`（探针临时文件，已随进程结束）。
**没有**改 `shadow-pkg/patches/files/offload_dsv41/`（D2 的 scheduler 补丁原样）、没有改任何别人的目录/容器、
没有碰 `dsv4-a3` / `mooncake-master` / Phy-ID 8–15，没有写 `upstream-v41/`。
