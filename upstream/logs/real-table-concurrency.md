# 40 — ★ 真表 206 GiB 上的注册 / 带宽 / 热行 / 并发扰动（RFC #16375 [47] 补测）

**日期**：2026-09-21 14:21–15:30 CST
**目的**：把 `pr/RFC-16375-CONTRIBUTION.md` §2.1 `Missing from [47]` 里那四个
"unmeasured" 变成有数据 —— 且全部用**真实 206 GiB 表本体**，而不是合成表。
**一句话**：真表测量把两件事同时改了：(a) 合成表上"热行没收益"的结论**在真表上是反的**
（+3.7–4.3×），(b) 真表的**行宽是 256 B，不是 20480 B**，随机 gather 只有 **7.6 GB/s**
（合成表口径的 ~1/12）；另外发现**注册会把整表页弄脏并触发 206 GiB 回写**（内容不变）。

---

## 0. 结论速查

| 项 | 结果 | 标记 |
|---|---|---|
| 单 die 满表注册（as-found） | **206.0 GiB / 122.2 s / 0.580 ms/MiB**，4 分片 **ret=0** | 【实测】 |
| 同上，紧接第二次（热） | 206.0 GiB / **99.9 s** / 0.473 ms/MiB | 【实测】 |
| 上游 #16925 的调用 `V2(MAPPED\|PINNED)` | 206.0 GiB / **83.9 s** / 0.398 ms/MiB，ret=0（+`HostGetDevicePointer`） | 【实测】 |
| **3 die 并发满表注册** | **247.6 / 227.0 / 247.5 s**（第二轮 **240.9 / 241.3 / 238.9 s**），**ret 全 0** | 【实测】 |
| **207001 / 507011** | **一个都没有**（全部 JSON 的 `codes.of_interest` 均为 false） | 【实测】 |
| 真表行宽 | **256 B**（I8，384,016,682 行）＋scale 行 32 B（F32×8） | 【实测】 |
| 单 die 连续读（1 GiB） | **107.0 GB/s**（256 MiB 103.6；16 MiB 64.8） | 【实测】 |
| 单 die 随机行 gather（256 B 行） | **7.5–7.6 GB/s**（16 MiB 臂 10.8） | 【实测】 |
| 部署口径 lookup（weight+scale 两次 gather） | **7.5–7.7 GB/s** | 【实测】 |
| 3 die 并发读 | 连续 **38.5 GB/s/die = 115.6 GB/s 总**；gather 7.56/die = 22.7 总 | 【实测】 |
| **热行（0.1% 行吃 80% 查询）** | **27.9–32.2 GB/s vs 均匀 7.5–10.8** ⇒ **+2.6–4.3×**（3 die 时 +2.8×） | 【实测】 |
| **注册会不会扰动稳态带宽** | 另两 die 各臂比值 **0.992–1.002（≤0.8%）** ⇒ 不扰动 | 【实测】 |
| 注册的副作用 | 文件页被弄脏 → `Dirty` **108–126 GiB** → 整表回写；**内容不变**，mtime 变 | 【实测】/【推断】 |
| 8 rank | **没测到**（只有 3 张空闲 die，3/8 代理） | 【未确认】 |

---

## 1. 机器、内存与安全护栏

### 1.1 开跑前（`agents/T1_realtable/logs/machine_state_before.txt`，14:26:52）

```
               total  used  free shared buff/cache available
Mem:            2013  1442   153    241        670       571     (GiB)
MemAvailable:   598966796 kB  = 571.2 GiB
node free (MB): n0 4320  n1 19605  n2 8518  n3 20058
                n4 61027 n5 42506  n6 369   n7 411
```

三个槽位 c0/c1/c2 = 物理 die **3 / 6 / 7**，`npu-smi` 显示三者只有常驻显存（2891/2893/… MiB）、
**无进程占用**；`dsv41-a3`（die 8–15，`VLLMEngineCor`，32 GB 显存 + 60 GB HBM）全程未被触碰。

### 1.2 护栏执行情况（逐条对照任务书）

1. ✅ 开跑前记录了 `free -g` 与 `numactl -H`（上表）。
2. ✅ `MemAvailable` 由脚本内 2 s 采样线程盯着，floor=100 GiB；**全程最低 561.97 GiB**，
   从未触发（每次运行的 `memory.monitor` 都在 JSON 里）。三个容器**都没有被 stop/rm**，
   没有 `pkill` 任何非自身进程。
3. ✅ 只用 c0/c1/c2；`dsv41-a3` 与其它 ~30 个容器全程只读观察。
4. ✅ 三槽位用**同一个 barrier 目录**逐轮对齐（ready-file + 600 s 超时；某槽位缺席时
   `peers_seen` 如实记录并继续，绝不死等）。
5. ✅ 结束后：三进程退出 → `MemAvailable` 回到 **571.1 GiB**（起点 571.2），
   `Dirty` 回落到基线；所有 `host_unregister` ret=0（见各 JSON 的 `unregister` 数组）。

### 1.3 表与挂载

真实表 4 个分片，208.79 GB ≈ **206.0 GiB**，`st_blocks` 校验**分配率 100.0%**（不是稀疏）：

| 分片 | 大小 | 主张量 | dtype / shape | 行宽 |
|---|---|---|---|---|
| `layers_14_engram_embed.scale.safetensors` | 11.44 GiB | `layers.14.engram.embed.scale` | F32 `[384016682, 8]` | 32 B |
| `layers_14_engram_embed.weight.safetensors` | 91.56 GiB | `layers.14.engram.embed.weight` | I8 `[384016682, 256]` | **256 B** |
| `layers_1_engram_embed.scale.safetensors` | 11.44 GiB | `layers.1.engram.embed.scale` | F32 `[384006168, 8]` | 32 B |
| `layers_1_engram_embed.weight.safetensors` | 91.55 GiB | `layers.1.engram.embed.weight` | I8 `[384006168, 256]` | **256 B** |

行宽是**从 safetensors header 里解析出来的**（脚本 `Shard.parse_header`），不是常量：
合成表用的 20480 B/行**不是**生产形状，真表是 256 B/行、3.84 亿行。

**挂载方式（唯一需要改的公共文件，已备份）**：给 `tools/a3_up.sh` 加了一行
`-v "$TABLE_DIR":/tables:rw`（`TABLE_DIR` 默认 `~/projects/dsv41/models/out/engram-int8`），
三个槽位因此都能看到 `/tables`。**注意这里是 `:rw` 不是 `:ro`**：只读 VMA 会被
`aclrtHostRegister` 拒（**ret=107017** `ACL_ERROR_RT_INVALID_HANDLE`，与我们 `logs/37`
第 8/9 行、`TRACK-D` §111 的既有结论一致），生产路径本身就是 `O_RDWR` + `MAP_SHARED`。
脚本只用 `PROT_READ|PROT_WRITE` **读**，并在 `unmap` 时核对分片 mtime（见 §5）。
原版备份：`agents/T1_realtable/a3_up.sh.bak-20260921`。

---

## 2. 结论 A：真表注册

### 2.1 单 die（阶段 A，c0 = die3，legacy `aclrtHostRegister(MAPPED)`）

| 分片 | 注册长度 | mincore(注册前) | ret | 时间 | ms/MiB |
|---|---:|---:|---:|---:|---:|
| `layers_14_engram_embed.scale` | 11719.3 MiB | 1.00 | 0 | 4.685 s | 0.3998 |
| `layers_14_engram_embed.weight` | 93754.1 MiB | 1.00 | 0 | 51.661 s | 0.5510 |
| `layers_1_engram_embed.scale` | 11718.9 MiB | 1.00 | 0 | **0.065 s** | **0.0055** |
| `layers_1_engram_embed.weight` | 93751.5 MiB | 1.00 | 0 | 59.577 s | 0.6355 |
| **合计** | **206.0 GiB** | | 0×4 | **122.2 s** | **0.5795** |

* 与 `logs/29` 的 **119.4 s** 相差 **+2.3%** —— 真表注册时间**已被独立复现**。
* 第二轮（同进程、同一批文件、页缓存已热）：**99.9 s**（0.4734 ms/MiB）。
* 第三轮换上游 #16925 的调用 `aclrtHostRegisterV2(MAPPED|PINNED)`（flags `0x10000002`）+
  `aclrtHostGetDevicePointer`：**83.9 s**（0.3979 ms/MiB），**ret 全 0**。
  即**上游那条调用在真表 206 GiB 上可用**，而且比 legacy 快 ~15%。

### 2.2 ★ "65× 之谜"的更正：是**页范围**的性质，不是 page cache

`logs/29` 把 `layers_14.scale`（4.7 s）与 `layers_1.scale`（0.065 s）的 65–72× 差
归因为"页是否在 page cache"。本轮把它**量出来并否证了**：

| 证据 | 数字 | 说明 |
|---|---|---|
| 同一轮内两片差 | 4.685 s vs 0.065 s = **72×** | 同尺寸、同 API、同一进程 |
| 两片的 mincore（注册前） | **都是 1.00** | 都**完全**在 page cache 里 |
| **单独**注册 `layers_14.scale` | 4.716 s（0.4025 ms/MiB） | 复现 4.7 s |
| **单独**注册 `layers_1.scale` | 0.066 s（0.0056 ms/MiB） | 复现 0.065 s ⇒ **71×** |
| 真正的冷/热对照（私有 2 GiB 文件，`posix_fallocate` 造、从未触碰） | 冷 66.7 ms → 热 9.3 ms = **7.1×** | cache 效应确有其事 |
| 同上，写满 + fsync + evict 后（真从盘读） | 冷 205.7 ms → 热 9.2 ms = **22.4×** | cache 效应的上限量级 |

⇒ 页缓存状态最多解释 **7–22×**，解释不了 **72×**（两片都 100% 命中）。
**结论**：快/慢是**该文件范围内页的固有属性**（同尺寸独立复现、并发下也不变），
与"是否命中缓存"是两个独立变量。**根因【未确认】**：候选是文件内 THP/hugetext 覆盖不同
（4K 页 vs 2M 页的 per-page 元数据量差 ~512×）或 NUMA 落点不同 —— 本机无法直接验（无驱动源码；
`/proc/PID/numa_maps` 对 file-backed 只给页数不给 THP 状态）。**不要再引用"65× 是缓存冷热"这个说法。**

### 2.3 3 die 并发满表注册（阶段 B，每 die 各注册 206.0 GiB）

| 运行 | c0 (die3) | c1 (die6) | c2 (die7) | ret | 207001/507011 |
|---|---:|---:|---:|---|---|
| 第一轮 `B_*`（注册前先试 evict） | **247.6 s** | **227.0 s** | **247.5 s** | 0/0/0 | 无 |
| 第二轮 `B3_*`（同一 barrier 重跑） | **240.9 s** | **241.3 s** | **238.9 s** | 0/0/0 | 无 |

* 并发下的分片级细节（第二轮）：`l14.weight` **104.5/104.1/103.7 s**、`l1.weight` **121.0/121.8/119.6 s**、
  `l14.scale` **9.4/8.9/9.3 s**、`l1.scale` **0.052/0.063/0.054 s**。
* ⇒ **并发只让"慢"分片慢 ~2.0×**（51.7 → 104 s），**"快"分片几乎零成本**（0.065 → 0.05–0.08 s）。
  这既印证了 §2.2 的 per-range 结论，也说明 3 进程注册同一批文件时**没有互相摧毁性干扰**。
* **物理内存只有一份**：三个进程 mmap 的是同样的 4 个文件（`MAP_SHARED`），
  page cache 全局唯一，所以 3 die 注册 **不是 3 × 206 GiB**；`mincore` 三个进程读数都是 1.00 即证据。
* **ret 码：3 个 die × 4 个分片 × 2 轮 = 24 次注册，全部 ret=0，没有 207001，也没有 507011**
  （#16828 的两个核心判据）。全部 JSON 的 `codes.of_interest` 都是 `{207001: false, 507011: false}`。

---

## 3. 结论 B：真表尺寸下的带宽（注册**保持住**，不进任何 timed loop）

口径：GB/s = 1e9 B/s；gather 的"有用字节" = 行数 × 行宽（256 B）。
每臂 n=7（warmup 2），报**中位数**（best 在日志/JSON 里）。HBM 臂只作锚点。

### 3.1 单 die（阶段 A）

| 臂 | 中位数 GB/s | best GB/s |
|---|---:|---:|
| 连续读 16 MiB | 64.78 | 66.31 |
| 连续读 256 MiB | 103.55 | 104.02 |
| **连续读 1024 MiB** | **107.01** | 107.12 |
| 均匀 gather 16 MiB（65536 行） | 10.80 | 10.91 |
| 均匀 gather 64 MiB | 7.54 | 7.57 |
| **均匀 gather 256 MiB** | **7.55** | 7.56 |
| lookup（weight+scale 两次 gather）16/64/256 MiB | 7.70 / 7.45 / 7.54 | 7.76 / 7.47 / 7.55 |
| HBM 锚点：连续 256 MiB | 523.76 | 525.73 |
| HBM 锚点：gather 16 MiB | 123.20 | 125.11 |

* 连续读 107 GB/s 与 `logs/38` 合成表的 **107 GB/s 完全一致** ⇒ 表从 2 GiB 放大到 206 GiB
  **不改变**连续读上限（带宽是通道属性，不是尺寸属性）。
* 但**随机 gather 塌到 7.6 GB/s**（合成表同口径是 95 GB/s）：差别不在表大小，而在**行宽**——
  256 B 行让每次 gather 只取 6.4 个 cache line、且 92 GiB 上的随机页散列把 TLB/DRAM 行缓冲打散。
  这是"真实 lookup 形状"与"合成 sweep"之间**最重要的一处口径差异**。

### 3.2 3 die 并发（阶段 B，barrier 对齐）

| 臂 | c0 | c1 | c2 | 合计 |
|---|---:|---:|---:|---:|
| 连续读 1024 MiB | 38.56 | 38.51 | 38.50 | **115.6 GB/s** |
| 均匀 gather 256 MiB | 7.56 | 7.58 | 7.55 | 22.7 GB/s |
| lookup weight+scale 64 MiB | 7.45 | 7.04 | 7.08 | 21.6 GB/s |
| HBM 锚点 连续 256 MiB | 514.2 | 515.5 | 501.1 | 1530.8 GB/s |

* **合计 115.6 GB/s ≈ 单 CPU socket 封顶**，与 `logs/38` 在合成表上量到的
  "同一 socket 上并发 ≈115 GB/s（57+57）"**数值重合** ⇒ 该结论**在真表尺寸上成立**。
* per-die 38.5 GB/s 与单 die 107 GB/s 的落差**由拓扑决定**（3 个 die 落到同一 socket），
  不是"表大了就变慢"。注册保持住不动这一点全臂都有 `registration_held: true`。

---

## 4. 结论 C：热行（真表上**有收益**，与合成表相反）⭐

构造：**0.1 % 的行吃 80 % 的查询**（0.1 % × 384,016,682 = **384,017 行 × 256 B ≈ 93.8 MiB**），
其余 20 % 均匀撒在整片 92 GiB 上；**同一份代码、同一份表、同样的字节数**，只有 id 分布不同。

| 臂（同尺寸对比） | 均匀 GB/s | 热行 GB/s | 倍数 |
|---|---:|---:|---:|
| 单 die，16 MiB/iter | 10.80 | **28.54** | 2.64× |
| 单 die，64 MiB/iter | 7.54 | **32.15** | 4.26× |
| 单 die，256 MiB/iter | 7.55 | **27.91** | 3.70× |
| 3 die 并发，256 MiB/iter（每 die） | 7.56 / 7.58 / 7.55 | 21.63 / 21.47 / 21.48 | 2.8× |

* ⇒ **"热行没有收益"是合成表（512 MiB / 2 GiB）太小造成的假象**：小表整体都能被 TLB/LLC 罩住，
  skew 无处发力。**真表 92 GiB 上，0.1 % 的 94 MiB 热点带来 2.6–4.3× 的有效带宽**。
* 这不是"我们实现了 cache"：热点子集仍然住在 host DRAM，只是**访问局部性**（页表/DRAM 行缓冲）
  被恢复了。**没有任何显式 hot-row 缓存被实现或测量**。
* 【推断】生产上若 rank 的热度确实集中在少数行（Engram 的 n-gram 命中通常高度倾斜），
  **显式 hot-row 缓存/预取会有正收益**；但收益上限受 skew 强度控制，**我们没有生产访问 trace**（见 §7）。

---

## 5. 结论 D：注册会不会扰动稳态带宽？

### 5.1 第一轮 `B_*`：**没测到**（如实记录）

registrar（c0）在 b3 阶段先做 `unregister` + **`posix_fadvise(DONTNEED)`**，
而 fadvise 在 91.5 GiB 分片上**花了 306 s 且什么都没回收**（§6.2），
于是它的重注册直到 **go+329 s** 才开始，而两个 reader 按 `--reader-seconds 300` 在
**go+300 s** 就停了 —— **reader 的样本完全没覆盖注册窗口**（`registration_start_epoch` = 1789974471.1
晚于 c1/c2 最后一次迭代 1789973211.0）。⇒ 这一轮**不能**用来回答扰动问题，
它的价值只剩"3 die 并发注册 ret=0"与"fadvise 无效"两条。

### 5.2 第二轮 `B3_*`：**测到了**

registrar 不再做无效 fadvise，reader 改为**等 registrar 的 done-file 才收尾**（上限 600 s），
registrar 延迟 10 s 起跑，于是 reader 有真实的前置基线：

* registrar：`unregister` 4 片约 **9.7 s** → 重注册 **206.0 GiB / 108.9 s**，**ret=[0]**，窗口 **118.6 s**。
  （对比：单 die 热注册 99.9 s ⇒ 有两个 die 在满速读的情况下只慢 **~9%**。）
* reader：c1 **15499** 次、c2 **15766** 次迭代（≈128.5 s），`stopped = peer registration finished`。

**逐臂 before vs during（中位数，GB/s）**：

| 臂 | c1 before → during | c2 before → during | 比值 c1 / c2 |
|---|---|---|---|
| 连续 16 MiB | 56.28 → 56.71 | 56.7 → 56.7 | 0.99 / 1.00 |
| 连续 256 MiB | 92.02 → 91.97 | 92.4 → 92.3 | 1.00 / 1.00 |
| 连续 1024 MiB | 94.74 → 94.64 | 95.1 → 95.0 | 1.00 / 1.00 |
| 均匀 gather 16/64/256 MiB | 7.05→7.05 / 7.29→7.29 / 7.28→7.28 | 7.2→7.2 / 7.5→7.5 / 7.5→7.4 | 1.00 / ~1.00 |
| HBM 连续 256 MiB | 432.81 → 432.16 | 434.1 → 432.5 | 1.00 / 1.00 |
| HBM gather 16 MiB | 68.57 → 69.03 | 70.4 → 70.0 | 0.99 / 1.01 |

**全部 8 个臂 × 2 个 die 的比值落在 0.992–1.002（≤0.8 %）**
⇒ 【实测】**3/8 代理规模下，一个 die 的整表重注册（206 GiB，108.9 s）不会扰动其它 die 的稳态带宽。**
（`before` 窗口 ≈20 s / 2365 次迭代，其中含 registrar 的 9.7 s `unregister` 阶段；
`during` ≈110 s / 13134 次迭代。`per_arm_by_bucket` 与 `buckets_pooled` 都在 JSON 里。）

**口径说明**：这是 **3 die 的 3/8 代理**。8 rank 时多了 5 个 rank 的注册内存/元数据竞争、
以及 socket 上的额外压力，**不能**线性外推（见 §7）。

---

## 6. 两个必须写下来的副作用

### 6.1 注册会把**整张表弄脏**（内容不变）→ 206 GiB 回写 ⚠️

现象（本轮实测）：
* 3 die 并发注册期间 `/proc/meminfo` 的 **`Dirty` 从 0 涨到 108–126 GiB**；
* 真表 4 个分片的 **mtime 恰好在各自被注册的那一刻被更新**
  （15:03:33 / 15:04:48 / 15:05:24 / 15:06:07，与逐分片注册顺序、间隔一一对应）；
* 判定性对照（`agents/T1_realtable/probe_register_dirtying.py`，**私有 1 GiB 文件**，
  每臂独立文件、sha256 前后对比）：

| 臂 | ret | mtime 是否变 | 内容是否变 | 注册后 Dirty |
|---|---|---:|---:|---:|
| 只 `mmap`+`munmap`（负对照） | – | **否** | 一致 | 0 |
| `aclrtHostRegister`（legacy MAPPED） | 0 | **是** | **一致** | 1.0 GB |
| `aclrtHostRegisterV2(MAPPED)` | 0 | **是** | **一致** | 1.0 GB |
| `aclrtHostRegisterV2(MAPPED\|PINNED)`（#16925） | 0 | **是** | **一致** | 1.0 GB |

⇒ 【实测】**内容逐字节不变**（sha256 前后一致），但**页被标脏并触发回写**，mtime 被更新；
【推断】机制是 Linux GUP：驱动为 DMA 钉住**可写**文件映射时会 `set_page_dirty()`，
即使一个字节都没写，内核随后把（未变的）页写回盘。
**只读 VMA 又被 107017 拒绝**（§1.3）⇒ 在这套驱动上没有"既能注册又不弄脏"的选项。
**影响**：一次 bring-up 会在共享机器上产生一次 **206 GiB 级别的写回 I/O**（我们的测量本身
也对这台共享机产生了这个副作用，需要知情）。建议在 RFC 里作为 operational caveat 写出。

### 6.2 `posix_fadvise(DONTNEED)` 在有人映射时**静默无效**（rc=0 也会骗人）

| 场景 | 调用耗时 | rc | mincore 前 → 后 |
|---|---:|---:|---|
| 真表 91.5 GiB 分片，**三个进程仍在映射/注册** | **97.3 s / 208.7 s** | **0** | **1.00 → 1.00**（0 页回收） |
| 私有 512 MiB–2 GiB 文件，无他人映射 | 0.14–3.5 s | 0 | 1.00 → **0.00–0.13** |

⇒ 【实测】`rc=0` **不代表**回收成功；判据只能是 `mincore`。
（我们在 §2.2 的冷/热对照里正是靠 `mincore` 才没写错结论。）

---

## 7. 【实测】/【推断】/【未确认】与"还缺什么"

### 7.1 标记

* 【实测】单 die 真表注册 122.2 s（复现 `logs/29` 的 119.4 s，+2.3%）；
  V2(MAPPED|PINNED) 83.9 s；3 die 并发注册 240.9/241.3/238.9 s（另一轮 227.0/247.5/247.6 s）；
  24 次注册全 ret=0、无 207001/507011；真表行宽 256 B；
  连续读 107 GB/s（1 die）/ 115.6 GB/s（3 die 合计）；均匀 gather 7.55 GB/s；
  热行 27.9–32.2 GB/s；注册期间其它 die 带宽变化 ≤0.8%；
  私有文件上"注册→mtime 变、内容不变、页变脏"；fadvise 在他人映射下 0 页回收。
* 【推断】热行收益的机制是页表/DRAM 局部性（而非显式缓存）；
  注册弄脏文件的机制是 GUP `FOLL_WRITE`；
  115.6 GB/s 合计是**单 socket 封顶**在真表上的重现；
  "快/慢分片"是文件内页属性（THP/hugetext 覆盖或 NUMA 落点）。
* 【未确认】72× 差的确切根因（无驱动源码、无 THP 观测手段）；
  8 rank 的注册时间与是否出现 207001/507011；
  生产的真实访问分布；真表被改 mtime 对上游既有流程是否有影响。

### 7.2 还缺什么（明确写清，别让读者以为 [47] 已经全绿）

1. **8 rank**：本机只有 **3 张空闲 die**（其余被用户自己的 8 卡服务占用），
   本文全部并发数据是 **3/8 代理**。8 rank 的**总注册时间**、rank 间抖动、
   以及 #16828 的 207001/507011 仍**没测到**。
2. **真实模型访问序列**：没有生产 trace。§4 的 skew（0.1 % 行 / 80 % 查询）是**我们构造的模型**，
   不是实测分布；因此"热行缓存值得做"只是**方向性结论**，不是收益预测。
3. **真实 batch shape 下的 lookup 端到端**：本文只测**纯内存臂**（`index_select` 有用字节带宽），
   不含模型侧的 hash/plan、kernel 融合与图捕获；lookup 延迟的那条线仍是 `logs/29`/§2.1 第 2 行的数据。
4. **A2/910B3 对照**：A2 现在不可达，真表测量只在 A3（910C，driver 26.1.1）上做过。
5. 本文的**副作用**（§6.1 回写、§6.2 fadvise）需要在 RFC 里作为 caveat 写出，
   但这属于**产品/运维判断**，需要用户决定口径。

---

## 8. 复现

### 8.1 脚本

| 文件 | 作用 |
|---|---|
| `pr/bench_real_table_regs.py` | 主脚本：`--phase A`（单 die 全流程）/ `--phase B`（3 die barrier 并发 + 扰动）；真表 header 解析、mmap/mincore/fadvise、注册、读臂、MemAvailable 监控、JSON 输出 |
| `agents/T1_realtable/run_stageB.sh` | 阶段 B 第一轮驱动（3 槽位同时起，同一 barrier 目录） |
| `agents/T1_realtable/run_stageB2.sh` | 阶段 B 扰动重跑驱动（reader 等 done-file；registrar 不做无效 fadvise） |
| `agents/T1_realtable/probe_register_dirtying.py` | "注册会不会弄脏文件"的判定性对照（私有文件 + sha256 前后） |
| `agents/T1_realtable/trim_raw.py` | 裁剪规则（见 §8.3） |
| `agents/T1_realtable/analyze_real_table.py` | 把 `logs/raw/40-real-table-*.json` 汇总成日志里引用的表 |

```bash
# 单 die（阶段 A，约 6 min）
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c0 --timeout 2400 --name T1_A_single -- \
  python3 -u /work/bench/bench_real_table_regs.py --phase A --tag c0 \
    --repeat 7 --warmup 2 --sizes-mib 16,256,1024 --gather-mib 16,64,256 \
    --cold-file smallest --evict-control-mib 2048 --extra-api v2_mapped_pinned \
    --out /work/agents/T1_realtable/out/A_single_c0.json

# 3 die（阶段 B 第二轮：并发注册 + 并发读 + 注册扰动，约 13 min）
nohup bash ~/projects/dsv41-upstream-pr/agents/T1_realtable/run_stageB2.sh \
  > ~/projects/dsv41-upstream-pr/agents/T1_realtable/logs/B3_driver.log 2>&1 &

# 副作用对照（私有文件，约 4 min）
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 1200 --name T1_dirty -- \
  python3 -u /work/agents/T1_realtable/probe_register_dirtying.py --size-mib 1024 \
    --out /work/agents/T1_realtable/out/dirtying_probe.json
```

### 8.2 原始数据 + sha256（**原文件留在 A3**，不搬 43 MB 大文件）

目录：A3 `~/projects/dsv41-upstream-pr/agents/T1_realtable/out/`。
本地只有**裁剪版**（`logs/raw/40-real-table-*.json`，最大 75 KB）。

| 原始文件（A3） | 大小 | sha256 | 本地裁剪版 |
|---|---:|---|---|
| `A_single_c0.json` | 74 118 B | `0b6dac5c62ed42af…`（全长见 §8.4） | `logs/raw/40-real-table-A_single_c0.json` |
| `B_c0.json`（阶段 B 第一轮 registrar） | 82 771 B | `796e297f9580fe9c…` | `…-B_c0.json` |
| `B_c1.json`（第一轮 reader，**未覆盖注册窗口**） | 43 366 582 B | `7bc12d1fc948fed1…` | `…-B_c1.json` |
| `B_c2.json`（第一轮 reader，同上） | 43 289 369 B | `772c71ea7e351629…` | `…-B_c2.json` |
| `B3_c0.json`（第二轮 registrar，**扰动结论来源**） | 46 858 B | `bbfa07de4e366caf…` | `…-B3_c0.json` |
| `B3_c1.json`（第二轮 reader） | 18 245 702 B | `67f6be56ac8f316f…` | `…-B3_c1.json` |
| `B3_c2.json`（第二轮 reader） | 18 560 324 B | `d8a3dbfbc267330f…` | `…-B3_c2.json` |
| `only_l14scale.json`（单分片复现 4.7 s） | 21 521 B | `e7ed0814a17cc783…` | `…-only_l14scale.json` |
| `only_l1scale.json`（单分片复现 0.065 s） | 21 430 B | `eae86e3a470d1ba3…` | `…-only_l1scale.json` |
| `dirtying_probe.json`（§6.1 对照） | 7 062 B | `f5fb3ab872d860d6…` | COS key `share/xfer/40-real-table-dirtying-probe` |

日志（A3 `agents/T1_realtable/logs/`）：`A_single_c0.log`、`B_c0/c1/c2.log`、`B3_c0/c1/c2.log`、
`only_shards.log`、`dirtying_probe3.log`、`machine_state_before.txt`、`B_driver.log`、`B3_driver.log`。

### 8.3 裁剪规则（为什么本地那份还是"可追溯"的）

原始 JSON 里 reader 会**逐次迭代**记录（B_c1 有 36 838 条、B3_c1 有 15 499 条），
所以第一轮的 reader 文件各 43 MB。`agents/T1_realtable/trim_raw.py` 的裁剪规则：

* **完整保留**：`registration`（逐分片 ret / ms / mincore / pass 合计 / ret_codes / all_zero）、
  `cold_hot`、`evict_control`、`concurrency_rounds`、`unregister`/`unmap`、`sanity`、`codes`、
  `config`、`environment`、`numa_topology`、`memory_points`、`read_arms`（含每臂逐次迭代，14 臂 × 7 次很小）；
* **替换**：reader 的逐次 dump → (a) 精确计数、(b) **5 s 分箱的中位数序列**、
  (c) 对照 registrar **自身**时刻的 before/during/after 三段统计（`buckets_pooled` +
  `per_arm_by_bucket` + `per_arm_during_over_before`）+ 判定规则字符串；
* **压缩**：`memory.monitor.samples`（每 2 s 一条）→ 每 10 s 最小值序列 + 全局最小值；
* **删除**：`real_table.files[*].tensors`（逐张量 dtype/shape/offset）与 `cold_hot` 之外的重复 meminfo；
* **新增**：`provenance` = 源文件绝对路径（A3）+ 源字节数 + **源文件 sha256** + 裁剪后字节数 + 本规则。

⇒ 每个被引用的数字都能在裁剪版里找到，且能凭 sha256 指回 A3 上的原件。

### 8.4 原始 sha256（完整）

```
0b6dac5c62ed42afa39b22532aedcedec7a7422b6e0d1607c1991ab4c01bdb6c  out/A_single_c0.json
796e297f9580fe9c232e164e2ae34f5d43595d32890f91274465a61e1baa25a9  out/B_c0.json
7bc12d1fc948fed1621056db6d0866cfc29195f3b5ef134bb9cb8795e844bca8  out/B_c1.json
772c71ea7e35162931a2624160676082377a8453d3e5168546400455447b5963  out/B_c2.json
bbfa07de4e366caf02a7f00933fcbe6ac883d1b2baee6d1be3d2836178e09346  out/B3_c0.json
67f6be56ac8f316f588c47e331b9de90afa79e4ffa12cf4a51a0a27acf0f7d98  out/B3_c1.json
d8a3dbfbc267330f61e675695be9852605bd5d71e4ecb9b83f68785a5a97cd36  out/B3_c2.json
e7ed0814a17cc783d7012d66aa7913a270ee3601e4c20dc58740e0d2c3a3dcee  out/only_l14scale.json
eae86e3a470d1ba3b2923f5796274658472196083d454f7a8aaaf7ee17d7da1b  out/only_l1scale.json
f5fb3ab872d860d6e893674e53fac6b52547f2ecfea51b4b02d0f8820b27cbd9  out/dirtying_probe.json
```

### 8.5 裁剪件的搬运（coscli，不走 ssh）

COS key（私有前缀 `share/xfer/`）：

* `share/xfer/40-real-table-trimmed-tar-v2` —— 9 个裁剪 JSON 的 tar.gz（69 108 B，
  sha256 `79f9cca0174c0be4ced6bf05a3e0ef8d3c1c27dd890f2893385afaf2e0814079`）；
* `share/xfer/40-real-table-dirtying-probe` —— §6.1 对照 JSON。

本地落盘后解开即 `logs/raw/40-real-table-*.json`。
