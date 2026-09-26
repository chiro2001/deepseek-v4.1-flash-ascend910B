# 068 · Engram 与卸载池抢 host 资源 —— **源码级定位**

> **任务**：回答"为什么抢、能不能改"，找出**能让两者共存的修改点**。
> **性质**：**纯源码/日志分析，未占卡、未起容器、未跑 vLLM**。
> **读的是哪一份**：容器 `prbench-c0` 内 `/work/shadow-pkg/patches/files/`（md5 见 §1.3）＋
> A3 上失败臂的原始 `serve.log`（路径见 §1.1）。
> ★ 本文所有结论标 **【实测】/【推断】/【未确认】**；错误码/常量/行号都经 `grep` 核对。

---

## 0. 一句话结论

**它们不是"两个资源"，是同一个驱动预算。**
Engram 的表（**183.11 GiB**）和卸载池（**197.20 GiB**）用的是**同一个调用**
`aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED)`。
Engram 先注册（日志 339–340 行），池后注册（985+ 行）⇒ 池只抢到 **74.68 GiB**，
余下 **31 张 / 122.52 GiB 全部 `ret=207001`** ⇒ 池**回落**成 `pin_memory=True`。

★★ **但"起服失败"的直接原因不是池，是 Engram 自己**：
`build_request_ids` 里的 `torch.repeat_interleave(index, counts)` 需要一个
**host 侧小暂存缓冲**（D2H 取标量），而那个缓冲拿不到 ⇒ 抛异常 ⇒ 捕获期整机挂。
⇒ **这条有一个一行修法**（`output_size=`），且**仓里已有验证过的设备侧等价实现**
（`engram_graph.py` 的 `searchsorted`）。见 `agents/Engram_pool_src/CANDIDATES.md`。

---

## 1. 证据来源（先钉住"读的是哪一份"）

### 1.1 失败臂的原始日志

```
臂    : p2-engram1-tierB-dg1-offload      （ENGRAM=1，池 197.20 GiB）
日志  : ~/projects/dsv41-upstream-pr/shadow-pkg/results/
        r8_p2-engram1-tierB-dg1-offload_20260922_174005/serve.log     （2050 行）
```

★ **注意**：`agents/R_8card_int8/logs/p2-engram1-tierB-dg1-offload.serve_a2.log`
只有 177 行，是**启动器**的日志，**不是** serve 本体 —— 别拿它做分析。

### 1.2 对照臂（同机、同日）

| 臂 | 目录 | ENGRAM | 池期望 | 结果 |
|---|---|---|---|---|
| p1a | `r8_p1a-tierB-dg0-offload_20260922_170611` | 0 | 197.20 GiB | **128/128 全绿** |
| p1b | `r8_p1b-tierB-dg1-offload_20260922_172245` | 0 | 197.20 GiB | **128/128 全绿** |
| **p2** | `r8_p2-engram1-tierB-dg1-offload_20260922_174005` | **1** | 197.20 GiB | **97/128 = 74.68 GiB**，31 张失败 |

### 1.3 源码（两条路径的 API 身份）

| 文件 | md5 | 说明 |
|---|---|---|
| `/work/shadow-pkg/patches/files/engram_device_index.py` | `9076716bafd14d8210e674e3489be1ee` | Engram 注册 |
| `/work/shadow-pkg/patches/files/offload_dsv41/cpu_npu.py` | — | 池注册（`0002` 补丁的部署件） |
| `/work/shadow-pkg/patches/files/engram_graph.py` | `adc0bd8683cded42ea4e45cb9a67e654` | ★ 已有设备侧 `searchsorted` 实现 |

---

## 2. Q1 —— 【实测】**是同一个资源**

### 2.1 `207001` 的语义（grep 到定义，不是猜）

```
/usr/local/Ascend/cann-9.1.0/aarch64-linux/include/acl/error_codes/rt_error_codes.h:69
#define  ACL_ERROR_RT_MEMORY_ALLOCATION   207001 // memory allocation error, only used by out of memory
```

⇒ **`207001` 就是 OOM**。

### 2.2 两边调的是同一个函数

**Engram**（`engram_device_index.py`）：

```python
# :239（能力探测，1 页）
dev, ret = acl.rt.host_register(addr, size, ACL_HOST_REGISTER_MAPPED)
# :413（真实路径 _map_and_register）
fd = os.open(self.path, os.O_RDWR)
addr = _libc.mmap(None, maplen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, aligned)
dev, ret = acl.rt.host_register(addr, maplen, ACL_HOST_REGISTER_MAPPED)
```

**卸载池**（`offload_dsv41/cpu_npu.py:96`）：

```python
devptr, ret = acl.rt.host_register(tensor.data_ptr(), nbytes, 0)  # ACL_HOST_REGISTER_MAPPED
```

⇒ **同一个调用、同一个 flag**（`0 == ACL_HOST_REGISTER_MAPPED`）⇒ 同一个驱动预算。

### 2.3 驱动自己的措辞

```
rtsHostRegister execution failed, reason=driver error:out of memory
    [FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:69]
```

### 2.4 ★ "池 / pinned / 注册"三个概念的关系（这是本条最容易搞错的地方）

| 概念 | 表面 | 在这条链里的角色 |
|---|---|---|
| `host_mem_pool` | `/proc/svm/dev0/feature/host_mem_pool` 的**能力位** | **A3 = 1，A2 = 0**（【实测】两机都读过）。它是"有没有专用 host 池"的**特性开关**，不是一个池对象 |
| **pinned** | `aclrtMallocHost` / torch `pin_memory=True` | **独立的 arena**。本臂的证据：池的 31 张注册失败后**回落成 `pin_memory=True` 且未再报错** ⇒ 那 122.52 GiB 的 pinned 分配**成功了** |
| **registered** | `aclrtHostRegister(MAPPED)` | **与 Engram 共用**的那个预算 —— 本节的结论 |

★ 所以：**pinned 与 registered 是两个 arena**（【推断】，依据是"回落未报错"），
而 **registered 在 Engram 与池之间是共享的**（【实测】）。
⇒ 这解释了为什么 A3 早期 8 卡臂在 `aclrtMallocHost` 上撞 207001（`logs/001` §4.2），
而本次是 `rtsHostRegister` 撞 —— **同一个错误码，两个不同的 arena**。

### 2.5 排除项（省得再查一遍）

| 假设 | 判据 | 结论 |
|---|---|---|
| `ulimit -l` 太小 | `ulimit -l = 65536 KiB`（64 MiB），但 p1a/p1b 注进了 **197.20 GiB** | ✗ **排除** |
| 宿主 RAM 不够 | A3 `MemAvailable = 1665 GiB` | ✗ **排除** |
| 注册调用次数上限 | 128 次调用里 97 次成功，失败也在同一批 | ✗ 不像 |

---

## 3. ★★ 顺序：**Engram 先，池后**（更正一个流传的说法）

```
339–340 行   [DEVICE-INDEX] Engram 表已映射为设备可寻址：L1=384006168行, L14=384016682行（每张 256B/行，HBM 占用 0）
985 行       [P1_pinned] CPU pool backend = registered   ← 池从这里开始
1225–1322    ★ 池的 31 次 ret=207001
1536 行      Engram 的 torch.repeat_interleave ⇒ 207001（前向，在 capture 期）
1546+        rtsHostRegister …out of memory（12 处，下游）
```

★ **"池先注册"是错的** —— 池的**第一行**（985）确实晚于 Engram 的**映射行**（339–340）。
1526 那个 Engram 失败是**前向**（capture 期），不是注册。

**Engram 表字节数**【实测，由日志行号算出】：

```
384006168 × 256 B = 98,305,579,008 B
384016682 × 256 B = 98,308,270,592 B
合计              = 196,613,849,600 B = 183.11 GiB
```

### 3.1 上限的边界（不敢说精确值）

```
p1a/p1b：ENGRAM=0 ⇒ 197.20 GiB 注得进去          ⇒ L > 197.20
p2     ：ENGRAM=1 ⇒ 183.11 + 74.68 = 257.79 GiB 后开始失败 ⇒ L ≥ 257.79
        池还想要 122.52 更多（总量 380.27）却没拿到 ⇒ L < 380.27
```

⇒ **257.79 GiB ≤ L < 380.27 GiB**【实测边界】

★★ **两处"不敢定"的地方，必须标清楚**：

1. **Engram 的 183.11 GiB 是 rank0 一份还是 8 rank 各一份？** 【未确认】
   `model.py::_engram_device_setup` 里**只有 print 被 `_bp_rank_zero()` 门控**，
   真实映射**没有 rank 门控**；而日志里只有 2 行 DEVICE-INDEX（都是 TP0）。
   若 R=1 ⇒ L≈258 GiB；若 R=8（1464.6 GiB）⇒ L≈1539 GiB。**两种都与现有观测相容**。
   ⇒ 这是**最值得优先测的一个数**，因为它决定"缩小池子到底有没有用"。
2. **失败是"总量耗尽"还是"大块碎片化"？** 【未确认】
   支持碎片化的证据：**31 个失败全是 ≥2.82 GiB 的大张量**，小张量（1.41/0.18/0.01 GiB）一直成功；
   但 1225 行**之后**仍有 1 张 3.98 GiB、2 张 4.24 GiB **成功** ⇒ 不是单调硬顶，
   存在**8 worker 并发注册的竞争抖动**。

---

## 4. Q2 —— Engram 到底申请了什么（**不是大块 device 内存**）

### 4.1 完整失败栈【实测，日志 1530–1545】

```
engram_graph.py:113   _eager
  → engram_device_index.py:689   device_engram_lookup
  → engram_device_index.py:669   build_request_ids
  → return torch.repeat_interleave(index, counts)
  torch.OutOfMemoryError: LocalScalarDenseNpu.cpp:23
      c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 207001
  [Error]: Failed to apply for memory.
```

### 4.2 机制

`build_request_ids(boundaries, device)` 的**两个操作数都在 device 上**（这是 9/21 那轮
"same device"修复的结果，见函数 docstring），所以**不是**旧的跨设备报错。真正的问题是：

```python
counts = boundaries.diff().to(dev)
index  = torch.arange(len(boundaries) - 1, dtype=torch.int64, device=dev)
return torch.repeat_interleave(index, counts)
```

`repeats` 是**张量**时，ATen 必须先知道**输出长度**（`counts.sum()`）。那是 device 上的标量，
要拿到 host 才能开输出张量 ⇒ **走 `copy_stream` 做一次 D2H** ⇒ 需要一个
**host 侧（pinned）小暂存缓冲** ⇒ 缓冲申请失败（`LocalScalarDenseNpu`）⇒ 207001。

⇒ ★★ **它要的不是"一大块 device 内存"，是一个 host 侧小暂存。修法完全不同。**

### 4.3 表本身怎么注册的

`engram_device_index.py::HostMappedEngramTable._map_and_register`：
`O_RDWR` → `mmap(MAP_SHARED)`（**文件页**，不是匿名）→ `aclrtHostRegister(MAPPED)`。
日志明说 **"HBM 占用 0"** ⇒ 表常驻宿主 DRAM，只被映射成设备可寻址。

★ 顺带一条**已知的坑**（同文件注释）：`MAP_SHARED` + 只读挂载 ⇒ `ret=507899`；
无 device context ⇒ `ret=107002`。**表目录必须 `O_RDWR` 可写**。

---

## 5. Q3 —— 池的注册形态与顺序

| 项 | 值 | 依据 |
|---|---|---|
| 注册粒度 | **逐张量**，**16 张/worker × 8 worker = 128 次调用** | p1a/p1b 各 128 行 `ret=0`；p2 = 97 + 31 = 128 |
| 池总量（本臂几何） | **24.651 GiB/worker**（`P2_WORKER_HOST_BYTES=26469138432`）⇒ 197.20 GiB | 日志 1315–1322 |
| 顺序 | **Engram（339）→ 池（985）** | 行号 |
| 顺序能否改 | 能（池在 `ensure_kv_transfer_initialized`，Engram 在 model `__init__`），**但见下** | — |

★ **"把顺序倒过来"值不值得做**：
只有当上限是**大块碎片化**时才有用（先给池的大块，Engram 的文件页映射本来就不要求连续大块）。
若上限是**总量**，倒序**完全无效**。⇒ 【未确认】，且**成本/收益比差**：它要改模块间时序。

---

## 6. ★★ 对"起服失败"归因的更正（这一条改变了优先级）

**流传的归因**：池注册失败 ⇒ 起服失败。
**实际**：池的 31 次失败**没有**让起服失败 —— 它**回落成了 `pin_memory=True` 并且成功了**
（日志里回落之后一路没有异常，直到 1536 行的 Engram 前向）。

⇒ **起服失败的直接原因是 Engram 的那个 D2H 标量**，不是池。
⇒ ★ **只要修掉 Engram 那一行，这条臂很可能就能起服**，代价是池里 31 张走 pinned
（丢掉那 31 张的 3–4× H2D 提升，是**性能问题，不是可用性问题**）。

这条正是 `CANDIDATES.md` 候选 1/2 的依据。

---

## 7. 与 A2 的关系（红线）

| 项 | A2 | A3 |
|---|---|---|
| `host_mem_pool` | **0** | **1** |
| `host_pin_pre_register` | **0** | **1** |
| 单进程注册实测 | 1/8/32/**64 GiB 全 `ret=0`**（`logs/065` §3c） | 未单测 |
| 8 进程并发注册 | **未测** | 本文件 197.20 GiB（无 Engram）/ 74.68 GiB（有 Engram） |

★ `host_mem_pool` 在两台机器上**相反** ⇒ A3 的结论**不能直接外推"总量上限"**，
只有**机制**（同一个 API、同一个预算）可以外推。

---

## 8. 【未确认】清单（下一位接手先补这些）

1. **Engram 表是 rank0 一份还是 8 份**（§3.1-1）—— 决定"缩池"是否够用；
2. **上限是总量还是碎片**（§3.1-2）—— 决定"倒序"是否值得做；
3. **pinned 与 registered 是否真的两个 arena** —— 本文件只有"回落未报错"这一条间接证据；
4. **`torch.searchsorted` / `repeat_interleave(output_size=)` 在 NPU 上是否都实现** ——
   前者已在 `engram_graph.py` 里被捕获路径用过（强证据），后者**未在本仓验证过**。
