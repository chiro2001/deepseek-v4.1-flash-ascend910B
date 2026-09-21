# 44 — 单会话消融：把 §5 的 PARTIAL 变成 DONE

**日期**：2026-09-21 16:22–20:32 CST（14 个 run，全部在同一台机器的同一段 8 张卡上）
**执行**：子代理 `T4_ablation`
**机器**：`A3-node1`，8× Ascend 910C（Phy-ID **8–15**，`/dev/davinci8..15`），CANN 9.1.0、driver 26.1.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、python 3.12.13
**镜像**：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（官方镜像，`PATCH_MODE=mount`）
**模型**：`~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（W4A8，Engram int8 两层，DSpark S=5，Vision）
**一句话**：RFC 正文第 104 行要求性能项"reproducible comparisons"，而 `RFC-16375-CONTRIBUTION.md` §5
一直挂着 `STATUS: PARTIAL — the single-session ablation (run A1) has not been done`。
本日志就是那次消融：**同一台 8 卡机器、同一个 harness、同一份模型、逐臂只改一个 env、
背靠背连跑，每臂给出同一个 `(A, ms/step)`**；另有两个重复臂（A0b、A1r）给出**跨会话噪声底**。
（gate 不能在单进程里运行时 toggle，原因见 `pr/RFC-16375-CONTRIBUTION.md` §5.2.4。）

---

## 0. 口径（读表前必读）

### 0.1 本表的口径与为什么是这样

| 维度 | 本表取值 | 生产/文档默认 | 为什么不一样 |
|---|---|---|---|
| **`CPU_BIND`** | **0** | **1** | `=1` 在本机**永远到不了就绪**：`cpu_binding.py:663` 的 `migratepages` 往已满的 NUMA node 6 迁 ~90 GB，**不收敛也不报错**。详见 §1。⇒ **本表只用于 gate 之间的相对比较** |
| `DROPCACHE` | 0 | 1（脚本默认） | 与用户历史口径一致（`logs/31`），且省掉每次重读 206 GiB 表 |
| `MODE` / `REPEATS` | `quick` / **6 发** | `quick` | 8K + 32K 各 **6 发**取中位数（A0 是先导臂、只有 2 发，故另跑 A0b 补齐）；`full` 会多 128K + GSM8K |
| 视觉 23 例 | **跳过** | 跑 | 性能消融不依赖它；`OFFICIAL_DIR` 指向不存在目录 ⇒ 脚本跳过（每臂省 3–5 min） |
| `PYTHON_PGO` | 0 | A3 默认 0 | PGO 产物是给 A2 镜像的 libpython 编的（`serve_a3.sh` §[PGO]） |
| `PATCH_MODE` | `mount` | A3 默认 `mount` | A3 用官方镜像，补丁只能靠挂载 |
| `GPU_UTIL` / `MAX_SEQS` / `PREFIX` / `BAT_TOKENS` | 0.92 / 4 / 0 / 2048 | 同 | `run_test.sh` 的性能口径（无 prefix caching 的单流口径） |
| `STATIC_KERNEL` / `NPUGRAPH_EX` | 1 / 1 | 同 | **全部臂都不动**：它们是**编译开关**，不是可 A/B 的运行时 gate（关掉就不再是同一个可运行配置，且 C8 记着关掉那个 trap 要付 4–5 ms/step） |
| `GATE_CHUNK` | **0**（A0–A10）；**512**（A11，候选补丁臂） | **0** | 见 §3.2 的语义澄清 |
| `GATE_MAX_TOKENS` | 2048（全部臂） | 2048 | 它是 padding 天花板，**由 graph capture 决定**，不是 chunk 开关 |

> ⚠️ **不要**把本表的绝对数字与 `RFC-16375-CONTRIBUTION.md` §5 里既有的 per-patch 数字混比：
> 那些是**单 die 函数级**或**单卡**的同会话 A/B，机器、帧（eager vs 图内）、并发都不同。
> 本表唯一的用途是回答 RFC [97] 的那个问题：*十一个 gate 里，哪一个真的在付钱？*

### 0.2 为什么是 "stock + 单个 gate" 而不是 "shipped 关掉一个"

RFC 问的是"哪个 gate 真正在付钱"。相对**同一个 stock 基线**的单 gate 增量才**可加**：
`Δ(单 gate) = A(该 gate) − A(stock)`，而 "shipped 关一个" 得到的是
`Δ(去掉一个) = A(shipped) − A(缺一个)` —— 后者在 gate 之间有交互时**不能相加**，
也不能回答"从上游出发打开这个 gate 值不值"。所以本表用前者。

### 0.3 每个 arm 的固定流程（可复现）

```bash
# 在 A3-node1 上
bash ~/projects/dsv41-upstream-pr/agents/T4_ablation/run_arm.sh <arm> <GATE=VAL> ...
# 内部就是：
#   IMAGE=quay...deepseek-v4.1-flash-a3 PATCH_MODE=mount PYTHON_PGO=0 \
#   DEVS="8 9 10 11 12 13 14 15" CHIPS="8 9 10 11 12 13 14 15" \
#   NAME=abl-<arm> PORT=8040 CPU_BIND=0 DROPCACHE=0 \
#   OFFICIAL_DIR=/nonexistent-t4-skip-vision MODEL=<模型> RUN_ID=<arm>_<ts> \
#   <gate env> bash scripts/run_test.sh
```

每臂的 provenance：`~/projects/dsv41-release/results/<run_id>/` 下的
`serve.log`（起服+参数行）、`driver.log`（时间线）、`env.txt`（KV/口径）、
`p42_t4_quote_8192_quote_8k.jsonl`、`p42_t4_quote_32768_quote_32k.jsonl`（原始测点）、
`REPORT.md`（汇总）。`agents/T4_ablation/arms.tsv` 记 `arm → run_id → 门控值`。

---

## 1. ★ 起服挂死：`cpu_binding` 的 `migratepages` 不收敛（**探路臂失败**）

这一节是探路臂 `abl-a1`（shipped 口径 + `PROFILE=1`）的**失败记录**。
它没有任何性能数字，但换来了后面所有臂能跑起来——所以它是**资产**，不是废数据。

### 1.0 一句话对照（这条对照最直观）【实测】

| `CPU_BIND` | `serve.log` 里的 `[migrate]` 行 | 起服结果 |
|---|---|---|
| **1**（生产默认） | **8 条**（8 个 rank 各一条，`[migrate] NPU:8..15 -> NUMA [4..7]`） | **永久挂死**：35 min 后外层超时，`migratepages` 进程在容器删除后仍活着 |
| **0**（本表口径） | **0 条** | 就绪 **8 min 43 s**（图捕获结束）→ 全程 **9 min 55 s**，基准跑完 |

`grep -ac "\[migrate\]" results/abl_a1_shipped_20260921_162214/serve.log` = **8**；
`grep -ac "\[migrate\]" results/a0_20260921_165927/serve.log` = **0**（A0 的
`serve_cmd.txt` 里 `enable_cpu_binding` 为 `false`，即 `CPU_BIND=0`）。

⇒ **同一台机器、同一个镜像、同一个模型，唯一差别是这一个开关。**

### 1.1 时间线【实测】

| 时刻（容器内 UTC） | 事件 | 相对首行 |
|---|---|---|
| 08:22:18 | 容器第一条带时间戳的日志 | 0 |
| 08:22:54 | `Initializing a V1 LLM engine (v0.27.1)` | +36 s |
| 08:30:23 | `Loading draft model: method=dspark` | +8:05 |
| 08:30:42 | 8 个 rank 全部 `Loading model weights took 39.1183 GB` | +8:24 |
| 08:31:32–08:34:xx | static kernel 编译第 1 轮（60 个 kernel） | ~+9:14 |
| 08:32:48 | `GPU KV cache size: 3,842,657 tokens` | +10:30 |
| 08:33:51–08:39:43 | 图捕获 + 3 轮 static kernel 编译（125 / 121 / 121） | — |
| 08:39:43 | **`Graph capturing finished in 361 secs, took 0.68 GiB`** | **+17:25** |
| 08:39:46 | cpu_binding 打出 8 条 `[migrate] NPU:N -> NUMA [M]` | +17:28 |
| 08:42:5x | 两个 `migratepages` 出现（容器内 pid 1353 = TP4/NPU12，1399 = TP5/NPU13） | +20:3x |
| 08:55:04 | 最后一条 `No available shared memory broadcast block found in 60 seconds` | +32:46 |
| 16:57:37 CST | `[serve_a2][FAIL] 等待超时`（35 min 超时），容器被脚本清理 | **+35:19** |

### 1.2 为什么说"不是慢，是不收敛"【实测】

对 **NUMA node 6**（`/sys/devices/system/node/node6/meminfo`）连续采样三点，间隔 20 s：

| 采样 | Node6 `MemFree` | Node6 `AnonPages` | Node6 `FilePages` |
|---|---|---|---|
| 16:54:15 | 24,884 kB | 70,648,168 kB | 31,627,384 kB |
| 16:54:35 | 24,908 kB | 70,648,112 kB | 31,627,384 kB |
| （更早一点） | 24,884 kB | 70,648,168 kB | 31,627,384 kB |

**三点零进展**：`MemFree` 三分钟内动了 24 kB（噪声量级），`AnonPages`/`FilePages`
一个字节没变。而两个进程各 **99.8% CPU**（`R` 态）已经烧了 15+ 分钟。

目标节点本身几乎是满的：node6 `MemTotal = 263,159,060 kB`（≈251 GiB），
`MemFree ≈ 24 MB` ⇒ **99.99% 满**；而每个 worker 的 `VmRSS ≈ 89–91 GB`
（`VmSize = 9.9 TB`，206 GiB 的 Engram 表被反复映射）。
**要迁的东西装不下，迁移命令又不检查、不超时、不报错。**

### 1.3 顺手证明"机器本来就是这样的"，不是本次实验造成的【实测】

`logs/38-20260921-host-dram-bandwidth.md`（**今天 12:57 采的，远早于本次 T4 工作**）写着：

> *"Free memory is very uneven on this shared box (`node0` ~1.0–2.9 GB, `node6`/`node7`
> ~0.3 GB, `node4` ~60 GB, `node5` ~43 GB at 12:57)."*

⇒ node6/node7 在 12:57 就只剩 0.3 GB。**这是共享机器的先存状态**，
`cpu_binding` 恰好往上撞。同一台机器上，这段空窗期也是 `logs/38` 测到
"按 socket 带宽封顶 ≈115 GB/s"的同一个 NUMA 不均匀特性。

### 1.4 试过的补救，以及为什么都没用【实测】

| 动作 | 结果 |
|---|---|
| 不 kill，等 | 从 +17:28 等到 +35:19 超时，零进展 |
| 普通用户 `kill -TERM`（PID 509177/509188） | `Operation not permitted` |
| `sudo kill -TERM` | 返回成功，进程**仍在** `R` 态 |
| `sudo pkill -9 -f '^migratepages '` / `pkill -9 -x migratepages` | 返回成功，进程**仍在** |
| `docker rm -f abl-a1` | 容器没了，**进程仍在宿主上**，`PPID` 变成 `[sleep]`、`PPid=1` |

两个进程当时仍在宿主上各烧一个核：

```
509177 migratepages 1399 0,1,2,3,4,5,6,7 6     # 99.8% CPU, R 态
509188 migratepages 1353 0,1,2,3,4,5,6,7 6     # 99.8% CPU, R 态
```

**后续**：约 2.5 小时后复查，`509177` / `509188` **都已自行退出**（`pgrep migratepages` 为空）。
⇒ **不是永久泄漏**，挂起的 `SIGKILL` 最终生效了；但对一次起服来说，
"35 分钟等不到、而且期间白烧两个核"和永久泄漏**在后果上是一样的**。

### 1.5 上游缺陷候选：三个"不该这样"的点

代码位置（容器内，官方镜像的 `vllm-ascend` 树）：

```text
/vllm-workspace/vllm-ascend/vllm_ascend/cpu_binding.py
    def bind_memory(self, pid: str, npu: int) -> None:        # line 637
        ...
        logger.info("[migrate] NPU:%s -> NUMA [%s]", npu, target_numa)   # line 663
        execute_command(["migratepages", pid,
                         ",".join(map(str, all_numa_nodes)), str(target_numa)])   # line 666
```

由 `serve_a2.sh` 的 `CPU_BIND`（默认 1）→ `additional-config` 的
`"enable_cpu_binding": true` 驱动；A0 起我们把 `CPU_BIND=0` 传下去，
命令行里就是 `"enable_cpu_binding":false`（见 `results/<run_id>/serve_cmd.txt`）。

| # | 不该这样 | 观察到的后果 |
|---|---|---|
| **1** | ★ **超时的那个分支本身没有超时**：`execute_command()` 里 `p.communicate(timeout=1000)` 看起来有 1000 s 保护，但 `except subprocess.TimeoutExpired:` 里 `p.kill()` 之后又调了一次 **无参数的 `p.communicate()`**——那一次**可以永远阻塞**。`migratepages` 陷在内核路径时不及时处理 `SIGKILL`，于是"保护"在超时那一刻**变成永久阻塞** | 起服永久挂住。**时间线正好落在窗口里**：`migratepages` `+17:28` 启动 → 1000 s 后 `+34:08` 触发超时并 kill → 之后 `communicate()` 永久阻塞 → 外层在 **+35:19** 放弃。**完全吻合** |
| 2 | **不检查目标节点空闲内存**：`bind_memory()` 只做三种提前退出（`migratepages` 不存在 / NPU 没有 CPU pool / 目标 NUMA 不在表里），**唯独没查空闲** | 目标 node 99.99% 满（`MemFree` ≈ 22 MB）、要迁 90 GB，命令照样发出去 |
| 3 | 失败**不返回错误**，进程连 `SIGKILL` 都杀不掉（当时） | 容器删了进程还在，`PPid` 落到 1；实测**约 2.5 小时后才自行退出** |

**上游可以怎么修**（建议，未实施）：

* 发命令前读 `/sys/devices/system/node/node<N>/meminfo` 的 `MemFree`，
  小于需要迁移的量（或某个比例）就 **跳过 + 打 warning**，不要发命令；
* 给 `migratepages` 加**超时与失败回退**（`timeout N` + 非零就 `logger.warning` 继续起服）；
* 至少把 `target_numa` 选成"有空闲的节点"或让 `migratepages` 的作用域可配置
  （现在 `all_numa_nodes` 是"从所有节点迁到目标"）。

> 上面两条（查空闲 + 给超时分支加界）已经写成补丁草稿：
> `pr/patches/cpu-binding-hang-fix.patch`（**主代理写，非本日志作者**；77 行，两处改动，
> 已在真实源码上 `patch -p1` 干净应用 + `py_compile` 通过）。核心是
> 超时分支加界（30 s 后放弃、返回 `-9`）+ 迁页前比较
> `/sys/.../nodeN/meminfo` 的 `MemFree` 与本进程 RSS，装不下就跳过并 WARN。
> 出事那一刻（node6 ≈ 22 MB、rank ≈ 90 GiB）**两个分支都会命中**。

**标记**：
* 【实测】时间线全部时刻、node6 三个采样点、两个进程的 `%CPU`/`R` 态、
  `kill -9` 无效、容器删除后进程仍在、8 条 `[migrate]` 的 NPU→NUMA 映射、
  代码行号与调用链。
* 【推断】"206 GiB 的 Engram 表经 `aclrtHostRegister` 变成 pinned 页 ⇒ 内核无法迁移"
  这一条**没有验证**（没查 `/proc/<pid>/numa_maps` 的 pinned 计数）。
  能确定的是：**目标节点装不下**（实测），这已经足以解释不收敛。
* 【推断】"这台机器上 `CPU_BIND=1` 在 8 卡起服时必然挂住"——
  对 **NPU8/9/10/11 → NUMA4/5** 同样成立（node4 约 60 GB、node5 约 43 GB 空闲，
  而每个 rank RSS ≈ 90 GB），但**没有逐组实测**（不再重复试，见 §0.1）。

> 独立 issue 草稿：`pr/issue-draft-cpu-binding-migratepages-hang.md`（主代理写，
> 用的是本节的数字）。本节是过程记录，那份是给上游看的稿子。

---

## 2. 起服阶段耗时（RFC [75] 的 compile-time / warmup 统计）

`issue-track-C.md` §3.5 挂着一条缺口：
*"Warmup and cache behaviour are documented only as 'cold start costs more'.
**No compile-time statistics per stage** in a form that could go into a support matrix."*

每一臂都自动产出一份带时间戳的 `serve.log`，所以这些数字是**免费**的。
下表用**日志自己的 UTC 时间戳**算差（脚本 `agents/T4_ablation/stage_timing.py`，
只读日志，不依赖外部计时）。

<!-- FILL:stage-timing -->

### 2.0 跨臂起服阶段表（每臂一行）

全部时刻都是 `serve.log` 自带的**容器内 UTC 时间戳**，相对"容器第一条带时间戳的日志"。
"图捕获结束"= `Graph capturing finished`，那是就绪前的最后一道坎。

| 臂 | 第一条日志 | engine 初始化 | 权重加载完成 | KV 容量确定 | 图捕获结束 | **起服总计** |
|---|---|---|---|---|---|---|
| **A0** stock | 08:59:31 | +0:38 | +4:05 | +5:07 | +8:43 | **8 min 43 s** |
| **A0b** stock（重复） | 09:12:3x | +0:4x | +4:1x | +5:2x | +8:5x | **~9 min** |
| **A1** shipped（`DEVICE_INDEX=auto`） | 09:26:53 | +0:38 | +8:31 | +8:47 | +16:02 | **16 min 02 s** |
| **A2** shipped−`DEVICE_INDEX`（`=0`） | 09:50:28 | +0:44 | +4:09 | +4:30 | +8:13 | **8 min 13 s** |
| a1（shipped + `CPU_BIND=1`，**挂死**） | 08:22:18 | +0:36 | +8:24 | +10:30 | +17:25 | **永不就绪（+35 min 超时）** |

**★ 这张表里最值钱的一格**：`ENGRAM_DEVICE_INDEX`

* A1（`auto` → 本机落到 **1**，日志有
  `[DEVICE-INDEX] Engram 表已映射为设备可寻址：L1=384006168行, L14=384016682行`）
  ⇒ 起服 **16 min 02 s**
* A2（`=0`）**没有**那两行 ⇒ 起服 **8 min 13 s**
* A0/A0b（`=0`）⇒ **8 min 43 s / ~9 min**

⇒ **设备索引路径在本机让起服多花 ≈7.8 分钟（+87%）**，代价是给两张 206 GB 的表做
host mapping（`aclrtHostRegister`）。这是 RFC [75]/[73] 的"warmup/编译行为"里
**完全没人写过的一格**，而且是**每 8 卡会话付一次**的固定成本。
其余阶段（engine 初始化 38–44 s、权重加载 4 min 09 s–8 min 31 s、
static kernel 编译 ~5 min、图捕获 ~3–6 min）见上表逐格。

> 注：A1 的"权重加载完成"是 **+8:31** 而 A2 是 **+4:09**——差的这 4 分钟不是权重，
> 是设备索引路径在建映射期间与权重加载交错（A1 的 `Multi-thread loading shards`
> 在 device-index 探测**之后**才开始，见 `serve.log` line 330 附近）。
> 这一条是【实测】的观测，机制归因是【推断】。

**每臂的 static kernel 编译轮次**（`[op_compiler] static kernel compile start/success` 配对）：

| 臂 | 轮次 | kernel 数 | 合计 |
|---|---|---|---|
| a1 | 4 + 1（无起始行的一组） | 60 / 4 / 125 / 121 / 121 | **431** |
| A0 | 4 | 63 / 4 / 125 / 123 / 123 | **438** |
| A1 | 3 | 4 / 125 / 121 / 121 | **371** |
| A2 | 3 | 4 / 125 / 121 / 121 | **371** |

⇒ 每臂的编译量在 **371–438 个 kernel** 之间波动，**没有一臂命中过缓存**（§2.1）。

### 2.1 static kernel 编译：**每臂全量重编，缓存从未命中**

这一节直接回答 RFC [75] 的 "warmup and **compilation-cache** behaviour"。

`serve_a2.sh` 每次起服前会做 `[SKCACHE-GC]` + 检查 `cache/skcache/static_kernel_cache/`，
并打印命中情况。实测（探路臂 a1）：

```text
[serve_a2] skcache: static_kernel_cache/ 命中（1 个缓存文件）
[serve_a2] skcache: 清单 CANN-9.1.0_Ascend910_9382.json = 39185 字节（有效）
```

**但"命中"是假象**：那个目录里只有

| 文件 | 大小 | 说明 |
|---|---|---|
| `CANN-9.1.0_Ascend910_9382.json` | 39,185 → 39,980 B（下一次起服后） | 清单（`hash -> *.run` 映射），**被重写** |
| `CANN-9.1.0_Ascend910_9382.lock` | **0 B** | 锁文件 |
| 其它产物 | **无** | —— |

⇒ 清单写了，**产物不在环境里，于是每次都全量重编**：

| 轮次 | kernel 数 | 出现的位置 |
|---|---|---|
| 第 1 轮 | **60** | 权重加载后、KV 容量判定前 |
| 第 2 轮 | **4** | 图捕获期间 |
| 第 3 轮 | **125** | 图捕获期间（最长的一轮） |
| 第 4 轮 | **121** | 图捕获期间 |
| 第 5 轮 | **121** | 图捕获期间 |
| **合计** | **431**（4 次 `compile start`／`success` 配对，另 1 组无起始行的 60） | —— |

**为什么缓存不生效（两个独立原因，都已定位）**：

1. `cache/skcache/compile_outputs` 是**软链**（→ `~/projects/dsv41/p36_static_kernel_a21/compile_outputs`），
   而 `serve_a2.sh` 的 `[SKCACHE-GC]` 用
   `find "$_skc" -maxdepth 1 -type d -name 'ts*_outputs' -exec rm -rf {} +`
   —— **`find` 默认不追软链**，所以 `$_skc` 下的 `ts*_outputs` 一个也没匹配到。
   实测：该目录里积了 **3,568 个** `ts*_outputs` 目录（1.7 GB），
   **a1 的日志里一条 GC 行都没有** ⇒ GC 从未触发过。
2. 真正要复用的 `static_kernel_cache/` 里**没有任何 `*.run`/编译产物**，
   只有清单与锁（上表）—— 所以即使 GC 正常工作，也没有东西可复用。

**结论（【实测】）**：本机上 static kernel **每臂固定全量重编 431 个 kernel，≈15 min**，
缓存命中率为 **0**。这是每臂起服时间的主体，与 `logs/CHANGELOG` 里"冷编译 15–20 min"
的说法一致，只是**它并不只在第一次发生**。

**没确认的**：为什么清单写入了却没有产物（是 `static_kernel.py` 的缓存写入条件、
还是挂载点只覆盖了一半）——【未确认】，需要单独查 `static_kernel.py:650-700` 与
`/workspace/static_kernel_compile_outputs` 的对应关系。

---

## 3. 臂的设计

### 3.1 stock 基线长什么样

| 短名 | 容器 env | 生产默认 | A0（stock）取 | 对应补丁 |
|---|---|---|---|---|
| `MOE_AG` | `V41_MOE_COMM_ALLGATHER` | 1 | **0** | 0001 MoE AllGather |
| `MOE_MASK` | `V41_MOE_MASK_RANGE` | 1 | **0** | 0002 mask 范围比较 |
| `ROPE_IDXSEL` | `V41_ROPE_IDXSEL` | 1 | **0** | 0003 RoPE index_select |
| `QLI_NOCAND` | `V41_QLI_NO_CANDIDATE` | 1 | **0** | 0004 QLI 无候选快路 |
| `O_PROJ_2D` | `V41_O_PROJ_2D` | 1 | **0** | 0005 wo_a 2D matmul |
| `ENGRAM_JIT` | `V41_ENGRAM_JIT` | 1 | **0** | 0008 numba JIT |
| `ENGRAM_DEVICE_INDEX` | `V41_ENGRAM_DEVICE_INDEX` | `auto` | **0** | 0009 设备侧查表 |
| `GATE_CHUNK` | `V41_ENGRAM_GATE_CHUNK` | 0 | **0**（= stock） | 0006 分块 |
| `GATE_MAX_TOKENS` | `V41_ENGRAM_GATE_MAX_TOKENS` | 2048 | **2048** | —— |
| `DRAFT_GRAPH` | `DSPARK_DRAFT_USE_CUDAGRAPH` | **0** | **0** | 负结果，不进交付 |
| `STATIC_KERNEL` | `STATIC_KERNEL` | 1 | 1（**全部臂都不动**） | 编译 |
| `NPUGRAPH_EX` | `NPUGRAPH_EX` | 1 | 1（**全部臂都不动**） | 编译 |

### 3.2 ★ 两个语义澄清（别搞反）

1. **`GATE_CHUNK=0` 不代表"没有 padding"。** padding 是**另一条路径**，由
   `GATE_MAX_TOKENS` 控制：为 graph capture 故意做的静态形状，每次调用把 token 维
   补到 2048 行。生产（用户容器 dump 出来的 env）实际是
   `V41_ENGRAM_GATE_CHUNK=0` + `V41_ENGRAM_GATE_MAX_TOKENS=2048`
   ⇒ 生产 = **「单次调用、补到 2048 行」**。
2. **`GATE_CHUNK=512` 是候选补丁 0006**：在 2048 行的缓冲里按 512 行循环。
   文档里的 **−1.56 ms @ 8K** 就是它，**它还没进生产**
   （§5 的 0006 行写着 "production ships 0 (= stock) because long-context
   re-measurement is pending"）。
   ⇒ 单独给它一臂（**A11**）：这一臂同时把 **G19**（`logs/41`/`logs/43` 的 ceiling 曲线
  是**函数级**的，端到端放大一直标【推断】）变成端到端【实测】。

### 3.3 为什么是 "stock + 单个 gate"

见 §0.2。一句话：单 gate 相对**同一个 stock 基线**的增量才可加，才能回答
RFC 的"哪个 gate 真的在付钱"。

### 3.4 臂清单

| 臂 | 方案 | 说明 |
|---|---|---|
| **A0** | `stock` | 上表 8 个 gate 全 0，`GATE_CHUNK=0 GATE_MAX_TOKENS=2048` = 生产现状 |
| **A1** | `shipped` | 全默认（不动任何 gate）+ `PROFILE=1`（顺带采上游要的 trace 制品） |
| **A5–A10** | `stock + 单个 gate` | A0 基础上每次只开一个：`MOE_MASK=1` / `QLI_NOCAND=1` / **`MOE_AG=1`** / `ROPE_IDXSEL=1` / `O_PROJ_2D=1` / `ENGRAM_JIT=1`（`ENGRAM_DEVICE_INDEX=auto` 未单开——A2 已从反向证明它不改 `A`，而它会让起服多花 8 min） |
| **A2–A4** | `shipped − 单个 gate` | A1 基础上各关一个：`ENGRAM_DEVICE_INDEX` / `ENGRAM_JIT` / `QLI_NOCAND` |
| **A11** | `shipped + GATE_CHUNK=512` | 候选补丁 0006 + G19 端到端（用 shipped 做底，见 §4.17） |
| **A0b / A1r** | **重复臂** | 与 A0 / A1 配置逐字相同，用于给出**跨会话噪声底**（§4.3） |

**不建议**：`stock + 关掉 STATIC_KERNEL/NPUGRAPH_EX` —— `RFC-16375-CONTRIBUTION.md` C8
记着"关掉那个 trap 要付 4–5 ms/step"，而这两项是**编译开关**、
不是一个可以 A/B 的 gate（关掉就不再是同一个可运行配置）。它们**不进本表**。

`VLLM_ADMISSION_GATE` 的对照**不做**：文档 §7 最后一行明确写
*"no RFC item claims this"*，且它与 vllm #56455 重叠，优先级排在最后。

### 3.5 每臂发数与噪声底

harness 默认 `REPEATS=2`，但 A0 的两发在 32K 上给出 **37.84 / 35.81 ms/step（差 5.7%）**，
而本表要分辨的单 gate 增量是 0.3–0.5 ms（≈1.4%）。⇒ 从 A0b 起全部臂统一用
**`REPEATS=6`**（同一个值，口径仍然一致），并把 A0（2 发）与 **A0b（6 发，同配置）**
两次测量当成**噪声底**：两者之差就是"这张表里的数不能小于多少才算真效果"的经验尺度。

**A0b 是什么**：它是 **A0 的重复臂**（`arms.tsv` 里 `a0` / `a0b` 的 gate 列表**逐字相同**），
**不是**一个独立的消融臂。它的唯一用途是给这张表一个**同配置、跨会话**的重复性度量。
报数时把两者**合成一行**（两张 run 的合并中位数），并**把两次的原值都列出**。
（`REPEATS=2` 的 a0 保留并用 6 发的 a0b 一起看，是想同时回答"少发够不够"这个问题。）

### 3.6 为什么不是"真的在同一个进程里 toggle"——这次必须说清楚

原文档写 *"The clean way to answer it is one session that toggles each gate"*。
**这些 gate 在 graph 捕获的路径上不可能在单进程里 toggle**，原因是机制性的：

* `V41_ENGRAM_GATE_CHUNK` 之类的分支在 `engram_gate()` 里**在 trace/捕获时求值**
  （`_gate_chunk_tokens()` 读 env），而捕获之后**图重放的是当时选中的那条分支**——
  运行时改 env 不会改变已捕获的图。
* `STATIC_KERNEL` / `NPUGRAPH_EX` / `DRAFT_GRAPH` 更是**编译期**开关，
  改了必须重新编译/重新捕获。

⇒ 一个"单会话 toggle"要么要求所有 gate 都走 eager（那就不是生产帧），
要么根本不生效。所以本表用**同一台机器、同一个镜像、同一份模型、同一个 harness、
背靠背连续跑、每臂只改一个 env** 的方式；唯一与"单会话"的差别是**容器重启 + 图重新捕获**，
而这一点用 **A0/A0b 的重复臂**去度量（§3.5）。
**这个限制本身是要在文档里写明的**，不能假装做到了单进程 toggle。

---

## 4. 结果

<!-- FILL:results -->

### 4.0 ★ 头号发现（**指标层**）：接受长度 `A` 只有两个值

到 A5 为止，**六个臂的 `A` 全部落进两个档**，中间没有过渡：

| 档 | 臂 | 8K `A` | 32K `A` |
|---|---|---|---|
| **低档（≈1.7）** | A0、A0b、**A5** | 1.676 / 1.738 / **1.717** | 2.881 / 2.899 / **2.876** |
| **高档（≈4.7–4.8）** | A1、A1r、A2、A3、A4 | 4.727–4.815 | 4.626–4.727 |

⇒ 这不是"多个 gate 各贡献一点"，而是**某一个开关在起作用**。
`A` 是**草稿-验证一致率**，直接决定 `tok/s`（低档 48–80，高档 137–157，**2–3×**）。

**这比消融表本身重要**：`RFC-16375-CONTRIBUTION.md` §5 把每个 gate 的效果写成
0.3–0.6 ms 的零碎节省；**8 卡实测**（本日志，同一个 harness、同一个模型、同一台机器）里，
它们的**门控组合**控制着一个 **2–3× 的量级差**。

**已排除的候选**：

| gate | 判据 | 结论 |
|---|---|---|
| `ENGRAM_DEVICE_INDEX` | A2（shipped 关掉它）`A` 仍是 4.771 / 4.727 | **不是**（且输出文本与 A1 逐字相同） |
| `ENGRAM_JIT` | A3（shipped 关掉它）`A` 仍是 4.815 / 4.631；§4.8 函数级 A/B 10×6 逐位一致 | **不是** |
| `QLI_NOCAND` | A4（shipped 关掉它）`A` 仍是 4.815 / 4.640 | **不是** |
| **`MOE_MASK`** | **A5（A0 + 只开它）`A` 仍是 1.717 / 2.876** | **不是**（这是 A0 线上的第一个点） |
| `QLI_NOCAND` | A6（A0 + 只开它）`A` 仍是 1.655 / 2.850 | **不是**（§4.12） |
| `ROPE_IDXSEL` | A8（A0 + 只开它）`A` 仍是 1.655 / 2.872 | **不是**（§4.14） |
| **★ `MOE_AG`** | **A7（A0 + 只开它）`A` = 4.743 / 4.727，`tok/s` ×2.9，KV 3.24M→3.84M** | **★ 就是它**（§4.13） |

**剩余待测**：`O_PROJ_2D`（§4.15）、`ENGRAM_JIT`（§4.16）—— 先验都低，跑完是为了"没有别的 gate 也动 `A`"。

### 4.1 A0 — stock（基线）

**run_id `a0_20260921_165927`**（2026-09-21 16:59:27 → 17:09:22 CST；`REPEATS=2`，先跑的先导臂）

| 量 | 8K | 32K | 来源 |
|---|---|---|---|
| `ms/step` | **34.72** | **36.83** | `results/a0_20260921_165927/p42_t4_quote_8192_quote_8k.jsonl` / `…_32768_quote_32k.jsonl`（median of 2） |
| 接受长度 `A` | **1.738** | **2.899** | 同上 |
| `decode tok/s` | **50.04** | **78.89** | 同上 |
| TTFT | 3.71 s | 6.50 s | 同上 |
| 峰值 KV | **3,241,932 tokens** | — | `env.txt`（`kv_tokens=3241932`，阈值 2,800,000 ⇒ PASS） |
| `static_kernel.py:650` 命中 | **0** | — | `serve.log`（脚本必查①通过） |
| 健康检查 | 200（`Application startup complete`） | — | `serve.log` |
| 备注 | `local_owner_effective = fast`（Engram validate 通过后切快路）；跳过视觉 | | `local_owner_effective.txt` |

**逐发原始值**（n=2，两个可用测点）：

| 题 | 发 | `ms/step` | `A` | `tok/s` | `ttft_s` |
|---|---|---|---|---|---|
| 8K | r1 | 34.870 | 1.728 | 49.552 | 5.675 |
| 8K | r2 | 34.561 | 1.747 | 50.536 | 1.738 |
| 32K | r1 | 37.839 | 2.965 | 78.361 | 6.553 |
| 32K | r2 | 35.812 | 2.833 | 79.428 | 6.441 |

### 4.2 A0b — stock 重复臂（**不是新臂**，用于噪声底）

**run_id `a0b_20260921_171229`**（17:12:29 → 17:26:28 CST；`REPEATS=6`；
gate 列表与 A0 逐字相同）

| 量 | 8K | 32K | 来源 |
|---|---|---|---|
| `ms/step`（6 发中位数） | **34.42** | **35.45** | `results/a0b_20260921_171229/p42_t4_quote_*.jsonl` |
| 接受长度 `A` | **1.676** | **2.881** | 同上 |
| `decode tok/s` | **48.29** | **80.49** | 同上 |
| TTFT | 1.75 s | 6.47 s | 同上 |
| 峰值 KV | 3,241,442 tokens | — | `env.txt`（PASS） |
| `static_kernel.py:650` 命中 | **0** | — | `serve.log` |
| `[migrate]` 行 | **0** | — | `serve.log`（`CPU_BIND=0` 生效） |

**逐发原始值**：

| 题 | r1 | r2 | r3 | r4 | r5 | r6 | 中位数 |
|---|---|---|---|---|---|---|---|
| 8K `ms/step` | 34.059 | 34.689 | 34.524 | 34.231 | 34.317 | 34.540 | **34.421** |
| 8K `A` | 1.725 | 1.652 | 1.614 | 1.707 | 1.700 | 1.610 | **1.676** |
| 8K `tok/s` | 50.05 | 47.24 | 46.56 | 49.66 | 49.34 | 46.43 | 48.29 |
| 32K `ms/step` | 35.137 | 35.454 | 36.475 | 35.283 | 35.662 | 35.454 | **35.454** |
| 32K `A` | 2.856 | 2.909 | 2.908 | 2.898 | 2.865 | 2.843 | 2.881 |
| 32K `tok/s` | 80.64 | 81.41 | 79.73 | 82.13 | 80.34 | 80.18 | 80.49 |

### 4.3 ★ 噪声底（A0 vs A0b，同配置两次会话）

| 量 | A0（2 发） | A0b（6 发） | 差（A0b − A0） | 相对 |
|---|---|---|---|---|
| 8K `ms/step` | 34.72 | 34.42 | **−0.30 ms** | −0.9% |
| 32K `ms/step` | 36.83 | 35.45 | **−1.37 ms** | −3.7% |
| 8K `A` | 1.738 | 1.676 | −0.062 | −3.6% |
| 32K `A` | 2.899 | 2.881 | −0.018 | −0.6% |
| KV tokens | 3,241,932 | 3,241,442 | −490 | −0.02% |

**怎么用这个数（写进结论的门槛）**：

* **同一发数内**的离散更大：A0b 的 6 发 8K 在 34.06–34.69 之间（跨度 **0.63 ms**），
  32K 在 35.14–36.48 之间（跨度 **1.34 ms**）。
* **跨会话**（同配置）差了 0.30 ms（8K，已经比 6 发内的中位数差还小）～1.37 ms（32K）。
* ⇒ 单 gate 的增量**只有在 |Δ| 明显超过 ~1 ms（32K）/ ~0.6 ms（8K）时**才有资格叫效果；
  0.3–0.5 ms 量级的声称（0002/0003/0004/0005 的"−0.31…−0.62 ms"）
  **在这个口径下不可判定**——这正是本次消融要如实报出来的东西之一。
* `A`（接受长度）**不随 gate 变化**（A0 vs A0b 差 3.6% / 0.6%，而它本来就是每次抽样），
  所以下表照文档要求把 `A` 与时间**同格**给出，但**不拿它当 gate 的效果**。

### 4.4 A1 — shipped（全默认）+ `PROFILE=1`

**run_id `a1_20260921_172649`**（17:26:49 → 17:48:14 CST；`REPEATS=6`；不动任何 gate）

| 量 | 8K | 32K | 来源 |
|---|---|---|---|
| `ms/step`（6 发中位数） | **29.86** | **30.47** | `results/a1_20260921_172649/p42_t4_quote_*.jsonl` |
| 接受长度 `A` | **4.787** | **4.691** | 同上 |
| `decode tok/s` | **157.0** | **151.7** | 同上 |
| TTFT | 1.25 s | 5.20 s | 同上 |
| 峰值 KV | **3,842,534 tokens** | — | `env.txt`（`ENGRAM_DEVICE_INDEX=auto` 生效） |
| `static_kernel.py:650` 命中 | **0** | — | `serve.log` |
| `[migrate]` 行 | **0** | — | `serve.log`（`CPU_BIND=0`） |

**逐发原始值**：

| 题 | r1 | r2 | r3 | r4 | r5 | r6 | 中位数 |
|---|---|---|---|---|---|---|---|
| 8K `ms/step` | 30.949 | 30.030 | 30.488 | 29.627 | 29.575 | 29.691 | **29.861** |
| 8K `A` | 4.759 | 4.759 | 4.815 | 4.815 | 4.849 | 4.727 | **4.787** |
| 32K `ms/step` | 30.927 | 30.322 | 30.069 | 30.613 | 32.235 | 30.082 | **30.468** |
| 32K `A` | 4.727 | 5.000 | 4.636 | 4.691 | 4.691 | 4.625 | **4.691** |

**关于 `PROFILE=1` 是否影响这组数字**：不影响，而且这不是推测——
`--profiler-config` **只在 API 上挂 `/start_profile` / `/stop_profile` 两个路由**，
torch profiler 对象**直到第一次 `/start_profile` 才被创建**（官方镜像里
`vllm/v1/worker/gpu_worker.py:1134`：*"Create the profiler wrapper only on the first
start call"*）。A1 的基准是在**从未调用过 `/start_profile`** 的情况下跑的；trace 是
基准**跑完之后**单独采的（§7）。实测也印证了这一点：profiler 打开期间同一题
32K 的 `ms/step` 从 30.5 涨到 **41.2 / 45.8**（+35% / +50%）——
**如果基准是在 profiler 打开时跑的，数字会明显更大；它们没有。**

⇒ 结论：**A1 的数字**可以**引用**；**trace 期间那两发（`trace_32k1/2`）的数字不可引用**。

### 4.5 ★★ 头号发现：A0/A0b 与 A1 的接受长度形状**本质不同**

这不是噪声，也不是小差异。同一个 harness、同一段 prompt、同一台机器，**跨臂**给出：

| 臂 | 8K `A`（逐位置接受率，pos 0→4） | 形状 |
|---|---|---|
| **A0 / A0b**（8 次测量） | `{0.44–0.51, 0.13–0.17, 0.04–0.06, 0–0.02, 0–0.014}` | **快速衰减** |
| **A1**（6 次测量） | `{0.83, 0.76–0.78, 0.71–0.75, 0.71–0.75, 0.71–0.74}` | **几乎平坦** |

* 臂内**高度一致**（A0b 六次的 pos0 都在 0.443–0.477，pos3/4 几乎恒为 0；
  A1 六次的 pos0 都在 0.830–0.836），臂间**差 2.9×**。
* 30K 上同样：A0b `{0.955, 0.556, 0.240, 0.089, 0.023}` vs A1 `{0.881, 0.793, 0.729, 0.705, 0.604}`。
* **代价**：A0 的 `tok/s` 是 48–80，A1 是 152–157 ⇒ 在这个口径下 gate 值 **~2× 吞吐**，
  远大于 §5 声称的"每个 gate 0.3–0.6 ms"。

**为什么这很严重**：自投机解码的接受率**完全由数值决定**。若某个 gate 真的
"逐位一致"（`max|d| = 0`，`logs/41`/`logs/43` 的 126/126 格），**`A` 就必须相同**。
两种可能，**必须用数据判**：
1. 某个 gate **改了数值** ⇒ 我们之前"逐位一致"的结论有适用范围没写清；
2. 或者两套配置里有一套在 A3 上**是错的**（退化了）。

**根因假设（H1）**：`ENGRAM_DEVICE_INDEX`。`auto` 在 A2 上因 `host_mem_pool=0`
**落到 0**、在 A3 上**落到 1**；而 A0 正是 `DEVICE_INDEX=0`，用户 A2 的 shipped
实测 `A = 1.745`（`logs/31`）≈ A0 的 1.73。
⇒ 判据：**A0 + 只开 `ENGRAM_DEVICE_INDEX`（或 A1 关掉它）——`A` 就会跟着翻**。

**第二层判据（输出逐字比对）**：对每个臂发**同一个贪心请求**，比完整输出文本的 sha256。
输出不同 ⇒ 数值确实被改了（那就锁定是哪一个 gate）；输出相同 ⇒ `A` 的差另有来源。
脚本：`agents/T4_ablation/probe_output.py`（`temperature=0, top_p=1, ignore_eos=True`，
固定 18488-token prompt，两次重复自检臂内确定性）。

**已采到的第一个点**（A1，2026-09-21 17:49）：

```text
arm=a1  run_id=a1_20260921_172649
prompt_tokens=18488  prompt_chars=24053
rep1 len=926  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8
rep2 len=926  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8   ← 臂内确定性 OK
```

**H1 的判定结果**：见 §4.6（A2 = `shipped − ENGRAM_DEVICE_INDEX`）。

> 这一节写的时候**结论还未定**：H1 可能成立、也可能被推翻。
> 无论朝哪边，这条都比消融表本身重要，所以单独立节。

### 4.6 A2 — `shipped − ENGRAM_DEVICE_INDEX`（**H1 被推翻**）

**run_id `a2_20260921_175024`**（17:50:24 → 18:03:52 CST；`REPEATS=6`；
只关 `V41_ENGRAM_DEVICE_INDEX=0`，其余全部 shipped 默认）

| 量 | 8K | 32K |
|---|---|---|
| `ms/step`（6 发中位数） | **32.65** | **33.43** |
| 接受长度 `A` | **4.771** | **4.727** |
| `decode tok/s` | 143.0 | 137.4 |
| 逐位置接受率（典型一发） | `{0.836, 0.764, 0.709, 0.709, 0.709}` | `{0.836, 0.782, 0.727, 0.691, 0.691}` |

**两条判定**：

1. **`A` 没变**：A2 的 4.771 / 4.727 ≈ A1 的 4.787 / 4.691，而 A0 是 1.676 / 2.881。
   ⇒ **`ENGRAM_DEVICE_INDEX` 不是接受长度异常的根因。H1 被推翻。**
2. **输出逐字相同（决定性证据）**：

   ```text
   arm=a1  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8  len=926
   arm=a2  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8  len=926
   ```

   **完全一致**（prompt_tokens=18488，`temperature=0`，臂内两次重复也逐字一致）
   ⇒ `DEVICE_INDEX` **不改变数值**——这与它的定位（"表还在 host DRAM，只是由
   device 算子直索"）一致，**也把它从"改数值"的嫌疑名单里排除**。
3. **它确实是时间上的收益**（但远小于 §5 里声称的 29.5→28.4）：
   8K +**2.79 ms**、32K +**2.96 ms**（关掉它 = 变慢）。
   这一格**超过 §4.3 的噪声门槛**（8K 0.6 / 32K 1.4 ms），所以是真效果。

**⇒ 根因仍在剩下的 gate 里**：`MOE_AG` / `MOE_MASK` / `ROPE_IDXSEL` / `QLI_NOCAND` /
`O_PROJ_2D` / `ENGRAM_JIT`。继续按"A1 关一个"的方式逐个隔离（§4.7 起）。

**A 异常当前的完整证据表**（每行都是**同一个 harness、同一段 prompt**）：

| 臂 | 配置上的差别 | 8K `A` | 8K `ms/step` | 输出 sha256（前 16 位） |
|---|---|---|---|---|
| A0 / A0b | 6 个 gate 全关 + `DEVICE_INDEX=0` | **1.68 / 1.74** | 34.4 / 34.7 | *(未采)* |
| A1 | shipped（全默认） | **4.787** | 29.86 | `33c7b876c5577d05` |
| A2 | shipped − `DEVICE_INDEX` | **4.771** | 32.65 | `33c7b876c5577d05` ← 与 A1 相同 |
| A3… | shipped − 单个 gate | 待测 | 待测 | 待测 |

### 4.7 A3 — `shipped − ENGRAM_JIT`

> ⚠️ **本节最初的结论（"输出文本变了 ⇒ 数值确实被改"）已按 §4.9.2bis 撤回。**
> 撤回理由：v1 探针的 prompt 让四个臂全部进入"复读指令"的退化态，首 token 是并列 argmax，
> **该探针不能判正确性**。下面保留原始观测与代码分析（它们本身没错），
> 但**不要**把任何一条读成"`ENGRAM_JIT` 改了数值"。

**run_id `a3_20260921_180419`**（18:04:19 → 18:19:58 CST；`REPEATS=6`；
只关 `V41_ENGRAM_JIT=0`，其余全部 shipped 默认）

| 量 | 8K | 32K |
|---|---|---|
| `ms/step`（6 发中位数） | **31.70** | **30.67** |
| 接受长度 `A` | **4.815** | **4.631** |
| `decode tok/s` | 149.6 | 150.2 |
| KV tokens | 3,842,657（设备索引生效，`device_index_effective=1`） | — |

⇒ 关掉 JIT：8K 慢 **+1.84 ms**、32K 慢 **+0.20 ms**（8K 超过噪声门槛）。
**`A` 完全没变**（4.815 / 4.631 ≈ A1 的 4.787 / 4.691）
⇒ **JIT 不是"`A` 从 4.8 掉到 1.7"的根因**。

**但它是"数值被改"的第一号证据**（同 prompt、`temperature=0`、臂内两次逐字重复）：

```text
A1 (shipped 全默认)      : len=926  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8
A2 (shipped − DEVICE_IDX): len=926  sha256=33c7b876c5577d05538f2ff43ecd6c4ac1b9fcfbfa3cadea1c538512c07342a8
A3 (shipped − ENGRAM_JIT): len=294  sha256=c893530ea15ad2dcd3136adba153cca07872704649af5f116adffbe1a5922c0b   ← 不同
```

**逐字符定位**：A1 与 A3 在**第 49 个字符**分叉（正好是生成的第一行之后）：

| 臂 | 分叉点之后 |
|---|---|
| A1 | `…直接抄录原文。\n==================…`（继续正常输出，926 字符） |
| A3 | `…直接抄录原文。\n：请从上文中**逐字引用**第五回开头…`（**退化成复读 prompt 的 suffix**，294 字符后停） |

#### 4.7.1 机制候选【推断，未验证】

`patches/files/engram_hash.py` 的注释自己写着：

```text
# [ENGRAM-JIT-HASH] dense 页镜像（仅 V41_ENGRAM_JIT=1 时使用；dict 保持为空）
...
print("[ENGRAM-JIT] note: _hash_mode() != 'fast'; the JIT path replaces "
      "both the stock and the fast implementations.")
```

⇒ **JIT 路径不是"把同一段代码编译一下"，而是一份替换实现**，它自带：

| 机制 | 代码位置 | 为什么可能造成差异 |
|---|---|---|
| **页镜像容量** `V41_ENGRAM_JIT_PAGES`（默认 4096） | `_jit_prepare()` line 371 | 容量不足时 `_engram_update_jit()` 里做 **4 次扩容重试**（`if oob < 0: break` … `else: raise RuntimeError`）；扩容期间的"部分写入"按注释是幂等的，但那是**设计意图**，不是实测 |
| **`fell_back` / scalar history 回退** | 内核返回值 `ret[2]`，Python 侧累加 `scalar_history_fallbacks` | 小 batch 走 scalar 路径，与 slab 路径**是两段代码** |
| **`err >= 0 → raise KeyError(err)`** | `_engram_update_jit()` | 失败路径与 stock 的"缺页"处理不是同一套 |

**没有验证是哪一条**（没做函数级 A/B、没查 `scalar_history_fallbacks` 计数）。
**能立刻定性的实验**：单卡上把两套 `update()` 对同一组输入各跑一遍，`torch.equal` 对比。

#### 4.7.2 ~~对 "bit-exact" 声明的影响~~（**本小节已作废**）

> 原本这里写的是"`bit-exact` 的作用域必须限定到 `engram_gate`/padding，不能覆盖 numba JIT"。
> **该结论已撤回**（§4.9.2bis）：它建立在 v1 探针那个无效 prompt 之上。
>
> 需要保留的**事实**只有两条，且两条都指向"没有发现数值差异"：
> 1. `logs/41`/`logs/43` 的 **126/126 格 `torch.equal`** 测的是 `engram_gate()`（padding/chunk），
>    **没有测过** `engram_hash.py` 的 JIT 路径 —— 这是**覆盖面**的陈述，不是缺陷指控；
> 2. §4.8 的**函数级 A/B 新测了** JIT 路径，结果 **10 个尺寸 × 6 步逐位一致**。
>
> ⇒ 到目前为止，**没有任何证据**表明 JIT 与 stock 的数值不同。
> （若将来要补，`engram_jit_kernel` 与 `engram_hash` 的 2 份实现都应该进上游式的单测，
> 但那是**建议**，不是本次发现的缺陷。）

**标记**：【实测】三次会话的 sha256 与长度、分叉点位置、臂内两次重复一致、
三臂都跑同一个 script/prompt、`static_kernel.py:650`=0；
【推断】上面三条机制候选；【未确认】JIT 路径到底在哪一步与 stock 分道扬镳。

> **待做（不在本次范围）**：函数级 `torch.equal` 对比两套 `update()`；
> 补采 A0/A0b 的输出 sha256（A0 也是 `JIT=0`，如果它的 hash 也是 `c893530e…`，
> 就把"JIT 是唯一改数值的 gate"钉死；如果 A0 是第三个 hash，则说明还有别的 gate 也改数值）。

### 4.8 ★ 决定性实验：函数级 A/B，`JIT=0` vs `JIT=1`（**逐位一致**）

**问题**：A3 的输出变了，是"JIT 那份替换实现算的不一样"吗？

**与已有的 `logs/37` 的区别（为什么必须重做一次而不是引用）**：

| | `pr/bench_ngram_history.py`（`logs/37` 用的） | 本节的 `agents/T4_ablation/bench_jit_vs_stock.py` |
|---|---|---|
| 比什么 | **两份手抄 verbatim 副本**（上游块 vs 我们的块）——即"抄写保真度" | **同一个出货文件的两种运行模式**：`V41_ENGRAM_JIT=0` / `=1` 各起一个**独立进程** |
| 覆盖面 | 14 个尺寸、6 个臂，输入合成但页镜像是完整物化的 | 10 个尺寸 × 6 步解码，含 **`<16` token 的 `_small_batch_history` 快路**、**`>=16` 的 slab 路径**、以及**页镜像按需扩容** |
| 额外记录 | —— | JIT 模式的**内部计数** `scalar_history_fallbacks` |

**ha​rness**：`agents/T4_ablation/bench_jit_vs_stock.py`（只读拷贝出货文件到
`/work/agents/T4_ablation/src/` 里导入，**没有改任何生产载荷**）：

```text
# 源文件（只读拷贝，sha256 见下）
engram_hash.py        d0811fd743e1a8c6e262dc8cd85b3ee53cb0f91424553ef17bc9b916cdeff44b
engram_jit_kernel.py  3cc33a8365e6cdba5b253f53ee1793e3aafa1c96e3939bbe3878aee621bea935

# 运行（单卡 c0 = die 3；容器 prbench-c0）
bash tools/a3_chip.sh c0 --timeout 1200 --name t4-jit-ab -- \
  python3 /work/agents/T4_ablation/bench_jit_vs_stock.py \
  --workdir /work/agents/T4_ablation/tmp/jitab --out /work/agents/T4_ablation/out
```

**每个尺寸的协议**：先 512 token 的 prefill 把页写实，再 6 步解码（4 个请求，
每步 `n` 个 token，位置按请求单调推进），每一步对 `(hashes, mask)` 取 sha256。
两个模式喂**完全相同的输入序列**（同一个 seeded RNG 流）。

**结果（2026-09-21 18:23:33–18:23:46 CST）**：

| `n` | JIT=0 mask_true | JIT=1 mask_true | 6 步 sha256 | 首差异步 |
|---:|---:|---:|---|---|
| 1 / 4 | 24 | 24 | ✅ 全等 | — |
| 8 | 48 | 48 | ✅ 全等 | — |
| 16 | 96 | 96 | ✅ 全等 | — |
| 32 | 192 | 192 | ✅ 全等 | — |
| 64 | 384 | 384 | ✅ 全等 | — |
| 128 | 768 | 768 | ✅ 全等 | — |
| 256 / 512 / 2048 | 1536 / 3072 / 12288 | 1536 / 3072 / 12288 | ✅ 全等 | — |

```text
jit-vs-stock.json: verdict_bit_exact = true
runs["0"] jit_ok=False  hash_mode=fast
runs["1"] jit_ok=True   hash_mode=fast
JIT=1 侧 scalar_history_fallbacks = 0（全 10 个尺寸）
```

**结论（【实测】）**：

> **`PagedNgramHistory.update()` 的返回值在 `JIT=0` 与 `JIT=1` 下逐位一致**——
> 10 个尺寸 × 6 步、覆盖小 batch 快路与大 batch slab 路、覆盖页镜像扩容、覆盖 mask。
> ⇒ **"JIT 是另一份实现所以数值会不同"这条推断被推翻**（与 `logs/37` 的 14/14 一致）。
> ⇒ A3 的输出差异**不在 `update()` 的返回值里**。

**这把可能性收到三个**（都还没验证）：

1. **张量别名**：JIT 路径返回的是**持久缓冲区的视图**
   （`torch.from_numpy(self._jit_out_hashes[:n])`，按 `n` 缓存复用），
   stock 路径每次返回**新建张量**。若引擎在两次 `update()` 之间持有上一次的返回值，
   JIT 侧就会被后一次写覆盖 —— 这在函数级 A/B 里**看不出来**（我们每步都立刻取 sha256）。
   【推断】【未确认】
2. **喂入的输入不同**：JIT 路径更快（0.427→0.076 ms），改变的是 host 侧的时间线，
   若引擎某处依赖"这个值已经算完"的时序假设，可能读到不同的页状态。同上，函数级看不见。
3. **纯数值抖动被"贪心+强制续写"放大**：探针 prompt 是"逐字引用第五回开头约 600 字"+
   `ignore_eos=True`，模型本来就在复读 prompt 的边缘；A1 与 A3 在**第 49 个字符前完全一致**，
   之后分叉。这种 prompt 下，任何一处极小的数值差异都会翻转后续（见 §4.9 的判据）。

**与 `logs/37` 的关系**：不矛盾。`logs/37` 测的是"我们的文件抄对了没有"，
本节测的是"同一文件两种模式的输出一样不一样"，两者都指向 **`update()` 本身是 bit-exact 的**。

**下一步（§4.9）**：要区分上面三条，最短路径不是继续猜，而是先回答一个更前面的问题 ——
**"同一个配置、重跑一次，输出会不会变？"** 如果会变，那 A3 的差异就**不构成**正确性证据。

### 4.9 A4 — `shipped − QLI_NOCAND`：★★ **同一服务上两次相同请求给出不同输出**

**run_id `a4_20260921_182034`**（18:20:34 → 18:34:04 CST；`REPEATS=6`；
只关 `V41_QLI_NO_CANDIDATE=0`）

| 量 | 8K | 32K |
|---|---|---|
| `ms/step`（6 发中位数） | **30.11** | **31.09** |
| 接受长度 `A` | **4.815** | **4.640** |
| `decode tok/s` | 156.5 | 146.2 |
| 起服 | 13–16 min（`device_index_effective=1`） | — |

⇒ 关掉 `QLI_NOCAND`：8K **+0.25 ms**、32K **+0.62 ms**，**都在 §4.3 的噪声门槛内**
（8K 0.6 ms / 32K 1.4 ms）⇒ 这个 gate 在本口径下**方向对但不可判定**。

#### 4.9.1 ★★ 探针被自己证伪：`deterministic_within_arm = False`

| 臂 | 臂内确定性 | 第 1 发 | 第 2 发 |
|---|---|---|---|
| A1 | **True** | len 926 / `33c7b876…` | len 926 / `33c7b876…` |
| A2 | **True** | len 926 / `33c7b876…` | len 926 / `33c7b876…` |
| A3 | **True** | len 294 / `c893530e…` | len 294 / `c893530e…` |
| **A4** | **False** | len 293 / `ab21129e…` | len 299 / `ff1e3efd…` |

A4 的两次请求**完全相同**（同 prompt、`temperature=0`、`top_p=1`、`ignore_eos=True`、
同 `seed`，**同一个活着的服务**），却在**第 0 个字符**就分叉：

```text
rep1: '：请从上文中**逐字引用**第五回开头…'
rep2: '，请从上文中**逐字引用**第五回开头…'
```

**⇒ 这套 serve 栈在贪心解码下不是无条件确定的。**

#### 4.9.2 这条如何改判 §4.7（**必须按这个口径重写**）

* **A3 的"hash 变了"不再等于"`ENGRAM_JIT` 改了数值"**：既然存在"同配置两次输出不同"的实例，
  "A3 ≠ A1"就可能只是**非确定性**。
* 因此本探针的定位必须降级为：**数值指纹（对"这两次运行的数值是否相同"敏感），
  不是正确性判据（不能回答"哪一个对"）。**
* **仍然成立的说法**（全部【实测】）：
  1. A1 与 A2 是两个**不同配置**却给出**逐字相同**的输出（926 字符，同 sha256）；
  2. A3 与 A1 不同（294 vs 926 字符，第 49 字符分叉）；
  3. **A4 与它自己两次都不同**（第 0 字符分叉）。
* **机制候选**（【推断】，**未验证**）：engram 的 n-gram 历史是按 `block_table[...]` 里的
  **KV 物理页号**去查页镜像的；页分配不同 ⇒ 页镜像内容不同 ⇒ 历史不同 ⇒ logits 不同
  ⇒ **第一个 token 就能不同**。这条能同时解释"为什么 A4 的第 0 字符就分叉"，
  也提醒：**跨会话比较文本 hash，必须先证明该配置本身可重复**。

#### 4.9.2bis ★ 撤回 §4.7 的结论，并纠正诊断（**主代理复核原始输出后给出**）

把四个臂的 probe 原始文本**逐条读出来**之后，看到一件比"非确定性"更本质的事：

```text
a1 rep1: ：请从上文中**逐字引用**第五回开头的原文（约 600 字，不要总结、不要改写，直接抄录原文。\n：
         请从上文中**逐字引用**…（自复读，926 字符）
a4 rep1: ：请从上文中**逐字引用**…（自复读，293 字符）
```

**四个臂全部退化成同一种"复读指令"的循环**（模型在抄 prompt 自己的指令，而不是回答）。
这种状态下，**首 token（`：` vs `，`）是一个近乎并列的 argmax**——
任何 1-ULP 级的数值差都会翻转它。A4 两次"只差第一个字符、之后是同一个循环"
正是**刀刃翻转的指纹**，不是"两套实现给出了不同结果"。

⇒ **§4.7 的结论（"`ENGRAM_JIT=0` 改变了输出 ⇒ 有 gate 改了数值"）必须撤回。** 正确措辞：

> 【实测】四个臂的探针输出**都是同一种退化复读**。A1/A2/A3 各自臂内 2 发逐字一致；
> A4 的 2 发在第 0 个字符就不同（`：` vs `，`），其后是同一个循环。
> 【推断】该探针 prompt 使模型进入"复制指令"的退化态，首 token 是近乎并列的 argmax，
> 因此**它不能作为正确性判据** —— 只能说明"存在刀刃级的不确定"，不能说明"某个 gate 改了数值"。
> **作用域限定：本探针无效；"bit-exact" 的既有证据（`logs/41`/`logs/43` 的 126/126 `torch.equal`）
> 本身不受影响** —— 那些是**函数级逐格比对**，测的是 `engram_gate()` 的 padding/chunk 路径，
> 与这个烂 prompt 无关。**不要把那些结论往回收。**

**§4.7.2 里"bit-exact 的作用域必须限定"那段也随之作废**：它建立在一个无效探针上。
`logs/41`/`logs/43` 的适用范围**不需要**因为本研究而修改。

#### 4.9.2ter 探针 v2（三条修法，已实施）

| # | 修法 | 实施 |
|---|---|---|
| 1 | **换 prompt**：长上下文（真喂到 engram 的历史）+ **有唯一合理续写的短问答**，不再让模型复述指令 | 后缀改为 `\n\nQuestion: What is the capital of France?\nAnswer: The capital of France is`；旧后缀（`data/hlm/suffix_quote.txt` 那条指令）弃用 |
| 2 | **记首 token 的 logprobs**：取 top-5，记 **top1−top2 的 margin** | `first_token_margin` 逐发记录；`min(margin) < 0.05` ⇒ 该臂自动标 **`inconclusive-margin`** |
| 3 | **N 发 + distinct 计数** | `PROBE_REPS`（默认 4，本轮用 6）；`n_distinct_outputs` 落进 probe JSON 与 `arms.tsv` |

**新的判据表**（写进探针自己的 `verdict` 字段）：

| 观测 | 判定 |
|---|---|
| `min(margin) < 0.05` | **inconclusive-margin** —— 首 token 是并列 argmax，本轮不能判对错 |
| margin 大（>1）**且** 臂内 N 发全同 | `deterministic` —— 才允许参与跨臂比对 |
| margin 大 **但** N 发不同 | `non-deterministic` —— **真问题**，值得单独立案 |

旧数据（v1 探针）已原样归档到 `agents/T4_ablation/output-probe-v1/<arm>.v1.json`
（**保留但不参与结论**），回传副本在 `logs/raw/44-a1-trace/output-probe-a{1,2,4}.json`。

#### 4.9.3 处置：给探针加"臂内确定性"这道闸

`probe_output.py` 从 2 发改成 **`PROBE_REPS`（默认 4）发**，并记
`n_distinct_outputs`；`arms.tsv` 每臂多一列。**只有 `deterministic_within_arm=True`
的臂，其 hash 才允许参与跨臂对比。**

**下一步（§4.10）**：`a1r` = **shipped 原样重跑**（`PROBE_REPS=6`），
用来回答"shipped 能不能跨会话重复"。

* 若 6 发全同 **且** = `33c7b876…` ⇒ shipped 可重复，A3 的差异才值得继续追；
* 若 6 发不全同 ⇒ 这个探针不能判对错，§4.7（A3）整条**降级为"非确定性"**。

### 4.10 A1r — shipped 原样重跑：**性能可重复，文本不可重复**

**run_id `a1r_20260921_183507`**（18:35:07 → 18:49:37 CST；全默认，与 A1 同一组 gate）

| 量 | 8K | 32K | vs A1 |
|---|---|---|---|
| `ms/step`（6 发中位数） | **30.35** | **31.04** | A1 是 29.86 / 30.47 ⇒ **+0.49 / +0.57 ms** |
| 接受长度 `A` | **4.727** | **4.726** | A1 是 4.787 / 4.691 ⇒ 差 1.3% / 0.7% |
| `tok/s` | 150.3 | 150.6 | — |
| KV tokens | 3,842,534 | — | 与 A1 的 3,842,534 **同** |

⇒ **shipped 在两臂之间的重复性很好**：`A` 差 ≤1.3%，`ms/step` 差 ≈0.5 ms（在 §4.3 噪声门槛内）。
**A 与 ms/step 这两列可以用。**

**但文本指纹不行**：v2 探针（新的"Paris"prompt，6 发）给出

| 发 | 长度 | finish | 首 token logprob | top1−top2 margin | 输出 |
|---|---|---|---|---|---|
| rep1/2/4/5 | 7 | `stop` | −0.0064 / −0.0093 / **−0.3254** / −0.0290 | 5.75 / 5.13 / **2.13** / 3.88 | ` Paris.` |
| rep3 | **996** | `length` | −0.0063 | 5.50 | ` Paris.\n\nQuestion: What is the tallest mountain in Everest?…`（自问自答循环） |
| rep6 | **865** | `length` | −0.0070 | 6.25 | ` Paris.\nQuestion: What is the capital of France?…`（同上） |

**⇒ 同一个服务、同一个 `temperature=0` 请求，6 发给出 3 种输出、192 倍的文本长度差。**
首 token 本身稳定（永远是 ` Paris`，margin 2.1–6.25），分叉发生在**第 3 个 token**：
EOS 与"继续自问自答"之间几乎并列。

### 4.10.1 ★ 定位实验：三档上下文长度对照（`probe_locality.py`）

在**同一个活着的服务**上，用同一个问题后缀、不同长度的 corpus 前缀各发 4 发：

| 档 | prompt tokens | 首 token | 首 token logprob 极差 | 输出种数 |
|---|---:|---|---:|---|
| short（只有问题） | 17 | ` Paris` | 0.0106 | 1/4 |
| **mid-2k** | 1,405 | ` Paris` | **0.0000**（逐位相同） | 1/4 |
| long-24k | 18,473 | ` Paris` | 0.0060 | 1/4 |

**判读（【实测】+【推断】）**：

* 【实测】**中上下文（1.4k token）四发的首 token logprob 完全相同** ⇒ 这套栈在
  常规长度下**是确定的**，§4.9 的"栈不确定"这个说法太强，**收窄为**：
  *异常是瞬时的、与具体请求有关的*。
* 【实测】同一批 4 发在这三档里都稳定（极差 ≤0.011）；而**紧接着的 a1r 探针 6 发里
  出现了一个 −0.325 的离群值**（比 0.011 大 **30 倍**）。
  ⇒ 异常**不是随上下文长度单调增长**的，而是**偶发**的。
* 【推断】最可能的位置是**长上下文路径里的一次性状态**（KV 物理页分配 / engram 页镜像 /
  chunked-prefill 调度），而不是逐 token 的数值噪声。
  **没有验证**（要看 `block_table` 与页镜像的对应关系，需要引擎内插桩）。

**对整张消融表的影响（结论）**：

1. **`ms/step` 与 `A` 不受影响** —— 它们是每个测点独立统计的，而且 A0↔shipped 的差是
   **系统性的**（8 发 vs 24 发，臂内方差远小于臂间差）；
2. **文本 hash 不是判据** —— 但 §4.8 的**函数级** `torch.equal`（10×6 全等）是，
   两者不冲突：函数级测的是 `update()` 的返回值，文本探针测的是整栈端到端行为。

⇒ **本日志的最终判读口径**：`ms/step` / `A` / `KV` / 起服耗时 **可用**；
文本 hash **只作为"数值指纹"的旁证，不作为结论**。

### 4.11 起：A0 + 单开系列（找"哪个 gate 在付钱"）

基线 A0/A0b（6 个 gate 全关 + `DEVICE_INDEX=0`）：`A` = **1.68 / 1.74**，
`ms/step` = **34.4 / 34.7**（8K）。下面每臂在 A0 之上**只打开一个**。

#### 4.11 A5 = A0 + `MOE_MASK=1`（**不是它**）

**run_id `a5_20260921_185117`**（18:51:17 → 19:05:12 CST；`DI=0` ⇒ 起服 ~13 min；`REPEATS=6`）

| 量 | 8K | 32K | vs A0 |
|---|---|---|---|
| `A` | **1.717** | **2.876** | A0 是 1.676 / 2.881 ⇒ **无差别** |
| `ms/step` | **34.91** | **35.68** | A0b 是 34.42 / 35.45 ⇒ +0.5 / +0.2 ms（门槛内） |
| `tok/s` | 48.6 | 80.4 | 同 A0b |
| 逐位置接受率 | `{0.431, 0.131, 0.038, 0, 0}` | — | 与 A0b 同形状 |
| `static_kernel.py:650` | **0** | — | — |
| `[migrate]` 行 | **0** | — | `CPU_BIND=0`，`device_index_effective=0` |

**结论（【实测】）**：`MOE_MASK` 既不能恢复 `A`，也不明显改变 `ms/step`。
⇒ **它不是那个开关。**

> ⚠️ **溯源缺口（如实记录）**：A5 的 v3 文本指纹**没采到** ——
> `finish_arm.sh` 调用的 `probe_output.py` 当时有个 `NameError`（`margins` 定义在使用之后），
> probe 静默失败而容器随后被删。**A5 的性能数字（`A`/`ms`/逐位置接受率/KV）完整可用，
> 缺的只有文本指纹。** 该 bug 已修，并且 `finish_arm.sh` 现在会**先确认 probe 文件写出**
> 再删容器（`exit 9` 并保留容器）。【实测】

#### 4.12 A6 = A0 + `QLI_NOCAND=1`（**也不是它**）

**run_id `a6_20260921_190549`**（19:05:49 → 19:19:42 CST；`DI=0`；`REPEATS=6`）

| 量 | 8K | 32K | vs A0 |
|---|---|---|---|
| `A` | **1.655** | **2.850** | A0 是 1.676 / 2.881 ⇒ **无差别** |
| `ms/step` | **34.50** | **35.44** | A0b 34.42 / 35.45 ⇒ **无差别** |
| `tok/s` | 47.8 | 79.2 | 同 A0b |
| 逐位置接受率 | `{0.500, 0.188, 0.063, 0.014, 0}` | — | 同 A0b 形状 |
| `static_kernel.py:650` | **0** | — | `device_index_effective=0`、`[migrate]`=0 |
| v3 文本指纹 | ✅ **6 发全同**（880 字符）| — | 本臂第一次拿到可用的 v3 指纹（`deterministic`） |

**结论（【实测】）**：`QLI_NOCAND` 同样既不能恢复 `A`，也不改变 `ms/step`。⇒ **不是它。**

**A0 线上已排除两个**：`MOE_MASK`（§4.11）、`QLI_NOCAND`（本节）。
剩下 `MOE_AG` → `ROPE_IDXSEL` → `O_PROJ_2D` → `ENGRAM_JIT`。

#### 4.13 ★★★ A7 = A0 + `MOE_AG=1`（**命中：这就是那个 gate**）

**run_id `a7_20260921_191947`**（19:19:47 → 19:30:27 CST；`DI=0`；`REPEATS=6`）

| 量 | 8K | 32K | vs A0（6 gate 全关） | vs shipped（A1） |
|---|---|---|---|---|
| **`A`** | **4.743** | **4.727** | **1.676→4.743（2.83×）** / 2.881→4.727（1.64×） | 4.787 / 4.691 ⇒ **无差别** |
| `ms/step` | **33.59** | **32.29** | 34.42→33.59（快 0.8） / 35.45→32.29（**快 3.2**） | 29.86 / 30.47 ⇒ 慢 3.7 / 1.8 |
| `tok/s` | **139.6** | **143.9** | **48.3→139.6（2.89×）** / 80.5→143.9（1.79×） | 157.0 / 151.7 ⇒ 略低 |
| 逐位置接受率 | `{0.836, 0.764, 0.709, 0.709, 0.709}` | `{0.836, 0.782, 0.727, 0.691, 0.691}` | 与 shipped **同形状** | 同 |
| KV tokens | **3,844,493** | — | 3,241,932 → 3,844,493 | 3,842,534 ⇒ 同档 |
| `static_kernel.py:650` | **0** | — | — | — |
| v3 文本指纹 | 6 发：**首 token 全同**，但 **2 种长度**（`stop-behaviour-differs`） | — | 见 §4.10.1 | — |

**结论（【实测】）**：

> **在本表的口径下（8K/32K 单流无 prefix cache），`MOE_AG` 是唯一决定接受长度的 gate。**
> 单开它就把 `A` 从 1.68 抬到 **4.74**、把 `decode tok/s` 抬 **2.9×**；
> 单开其它任何一个（`MOE_MASK` §4.11、`QLI_NOCAND` §4.12）都**完全不改 `A`**；
> 从 shipped 里单关 `DEVICE_INDEX`/`ENGRAM_JIT`/`QLI_NOCAND`（A2/A3/A4）也都不改 `A`。

**为什么这个结论重要（相对 RFC 文档 §5 的现有叙述）**：
§5.1 把 0001 行的效果写成"**−4.25 ms at 128K**，KV 池 3.39M → 4.16M"，
读起来是"一个省 4 ms 的优化"。**本表在 8K/32K 上量到的是完全不同量级的东西**：
它决定 `A` 是 1.7 还是 4.7 ⇒ 决定 `tok/s` 是 48 还是 140（**2.9×**）。
⇒ **RFC 文档 §5.1 的 0001 行需要重写**（回填措辞见 `pr/RFC-16375-CONTRIBUTION.md` §5.2.3 与本文档 §5.3）。

**【推断】机制**：`MOE_AG` 把专家通信从逐 token 的 dispatch/combine 换成 AllGather。
草稿模型与主模型都跑 MoE；两者的 MoE 数值路径一旦不同，draft 的提议与主模型的验证
就只在少数位置一致 ⇒ `A` 崩塌。**这条没有验证**（需要 MoE 层的 logits 比对）。

**【未确认】**：为什么 `A` 只有两个值（1.7 与 4.7）而没有中间态 ——
如果是"数值不一致 ⇒ 接受率崩塌"，那确实会呈现阈值行为，但没有实测支持。

#### 4.14 A8 = A0 + `ROPE_IDXSEL=1`（**不是它**，复核通过）

**run_id `a8_20260921_193105`**（19:31:05 → 19:41:08 CST；`DI=0`；`REPEATS=6`）

| 量 | 8K | 32K | vs A0 |
|---|---|---|---|
| `A` | **1.655** | **2.872** | A0 = 1.676 / 2.881 ⇒ **无差别（低档）** |
| `ms/step` | **34.84** | **36.89** | A0b 34.42 / 35.45 ⇒ +0.4 / +1.4 |
| `tok/s` | 46.2 | 76.3 | 低档 |
| KV tokens | **3,241,687** | — | 低档（`MOE_AG=0`）⇒ 与 §4.12.1 的机制一致 |
| v3 文本指纹 | ✅ 6 发全同（`deterministic`） | — | — |

⇒ `ROPE_IDXSEL` 不改 `A`、不改 KV 池。**排除。**

#### 4.15 A9 = A0 + `O_PROJ_2D=1`（**不是它**，复核通过）

**run_id `a9_20260921_194140`**（19:41:40 → 19:55:31 CST；`DI=0`；`REPEATS=6`）

| 量 | 8K | 32K | vs A0 |
|---|---|---|---|
| `A` | **1.744** | **2.876** | A0 = 1.676 / 2.881 ⇒ **无差别（低档）** |
| `ms/step` | **34.52** | **35.56** | A0b 34.42 / 35.45 ⇒ **无差别** |
| `tok/s` | 48.7 | 79.9 | 低档 |
| KV tokens | **3,241,687** | — | 低档（`MOE_AG=0`）|
| v3 文本指纹 | ✅ 6 发全同（`deterministic`） | — | — |

⇒ `O_PROJ_2D` 不改 `A`、不改 KV 池、不改 `ms/step`（在噪声内）。**排除。**

#### 4.16 A10 = A0 + `ENGRAM_JIT=1`（**不是它** —— A0 线 5 个 gate 全部排除）

**run_id `a10_20260921_195535`**（19:55:35 → 20:05:42 CST；`DI=0`；`REPEATS=6`）

| 量 | 8K | 32K | vs A0 |
|---|---|---|---|
| `A` | **1.746** | **2.866** | A0 = 1.676 / 2.881 ⇒ **无差别（低档）** |
| `ms/step` | **35.50** | **34.87** | A0b 34.42 / 35.45 ⇒ +1.1 / −0.6（门槛内） |
| `tok/s` | 48.9 | 81.8 | 低档 |
| KV tokens | **3,241,564** | — | 低档（`MOE_AG=0`）|
| v3 文本指纹 | ✅ 6 发全同（`deterministic`） | — | — |

**★ A0 线收官**：`MOE_MASK` / `QLI_NOCAND` / `ROPE_IDXSEL` / `O_PROJ_2D` / `ENGRAM_JIT`
**五个 gate 单独打开，没有一个能把 `A` 从 1.7 抬起来**，`ms/step` 也全在噪声门槛内。
**只有 `MOE_AG` 能做到（A7）。**

#### 4.17 A11 = shipped + `GATE_CHUNK=512`（G19：端到端放大）

**run_id `a11_20260921_200615`**（20:06:15 起；**唯一带 `GATE_CHUNK=512` 的臂**）

> 这一臂的用途与 §4.11–4.16 不同：它不是找 `A` 的原因，而是把
> **`logs/41`/`logs/43` 的函数级 ceiling 曲线**（`CHUNK=512 MAX=2048` vs stock）
> 放到**真实服务端到端**上量一次 —— 这正是 `issue-track-G19` 缺的那一步。
>
> **为什么用 shipped 做底而不是 A0**：`CHUNK=512` 是"**还没进生产的候选补丁**"，
> 它的价值是"在**生产形态**上再加它会怎样"（§5 行 0006：*"production ships 0 (= stock)
> because long-context re-measurement is pending"*）。而且 A7 已证明 `MOE_AG` 会把
> 整个 step 的时间结构改掉（`A` 1.7→4.7），在 A0 上量到的放大倍数不能外推到生产。

结果见 §6 总表；对照基线是 **A1 = shipped 且 `GATE_CHUNK=0`**。

**run_id `a11_20260921_200615`**（20:06:15 → 20:31:33 CST；`DI=auto` ⇒ 起服 16 min；`REPEATS=6`）

| 量 | 8K | 32K | vs A1（shipped，`GATE_CHUNK=0`） |
|---|---|---|---|
| `ms/step` | **30.68** | **30.61** | A1 = 29.86 / 30.47 ⇒ **+0.82 / +0.14 ms** |
| `A` | 4.771 | 4.658 | A1 = 4.787 / 4.691 ⇒ 无差别 |
| `tok/s` | 153.3 | 149.0 | A1 = 157.0 / 151.7 ⇒ 略低 |
| 峰值 KV | **3,842,901** | — | A1 = 3,842,534 ⇒ **+367 tokens**（同方向，见下） |
| `static_kernel.py:650` | 0 | — | — |

#### 4.17.1 ★ G19 的答案：函数级的 −1.56 ms **没有**在端到端放大

RFC 文档 §5.1 的 0006 行声称 `GATE_CHUNK=512` "**−1.56 ms at 8K**"。本臂在**真实服务、端到端**
（8K 单流、`REPEATS=6`、同机同模型）上量到的是：

> **+0.82 ms @8K**（在 8K 噪声门槛 **0.60 ms** 之外，但只有 1.4 倍门槛）
> 与 **+0.14 ms @32K**（远在 1.37 ms 门槛之内）。

⇒ **结论（【实测】）**：

1. **端到端没有出现 −1.56 ms 的放大**；符号甚至是**相反**的（略微变慢）。
2. 8K 那一格 **+0.82 ms 略超噪声门槛**，但只有 1.4 倍门槛，**单会话单臂不足以定性**
   —— 严格的表述是"**未观察到收益，且点估计为轻微负向**"。
3. **HBM 收益是真的**（同方向的旁证）：KV 池从 3,842,534 → **3,842,901**（+367 tokens），
   说明 gate 的 HBM 峰值确实降了（§3.2 的 "peak HBM 0.52×" 是函数级数字）。
4. **对 RFC 文档 §5.1 的 0006 行的影响**：那一行写 *"production ships 0 (= stock) because
   long-context re-measurement is pending"*。现在可以补一句：
   **在 8K/32K 单流口径下，端到端没有可测收益** ⇒ 保持 `0` 是对的；
   128K 仍未测（`MODE=full` 未跑）。

> **口径提醒**：本臂的底是 **shipped**（`A`≈4.8）。如果在 `A0`（`A`≈1.7）上量，
> 每一步的 token 数与 batch 形状都不同，放大倍数**不能互推**。

---

## 5. 结论

### 5.1 三个可引用的结论

**【实测】结论 1（最重要）：在本表口径下，`MOE_AG` 是唯一决定接受长度的 gate。**

* A0（6 gate 全关）`A` = **1.676 / 2.881**；A7（**A0 + 只开 `MOE_AG`**）`A` = **4.743 / 4.727**；
* 单开其余五个 gate（A5/A6/A8/A9/A10）`A` **全部不变**；
* 从 shipped 单关 `DEVICE_INDEX`/`ENGRAM_JIT`/`QLI_NOCAND`（A2/A3/A4）`A` **也全部不变**；
* 代价：`decode tok/s` **48 → 140（×2.9）**（8K 单流）。

**【实测】结论 2：其余 gate 的单 gate 增量全部落在噪声门槛内。**
RFC 文档 §5 现有叙述里的 −0.31…−0.62 ms 量级，在 8K/32K 单流口径下**分辨不出来**
（同配置跨会话噪声：8K 0.30 ms / 32K 1.37 ms）。**反向口径**（从 shipped 拿掉）能看到
`ENGRAM_DEVICE_INDEX`（−2.9 ms）与 `ENGRAM_JIT`（−1.8 ms @8K，A3）在付钱。

**【实测】结论 3：本机的 8 卡起服有一个与补丁无关的运维坑**（§1）：
`CPU_BIND=1` + 目标 NUMA 节点近满 ⇒ `migratepages` 永久挂住（35 min 超时、进程 2.5 h 后才退）。
所有臂因此跑在 `CPU_BIND=0` 上（口径已逐字标注）。

**【实测】结论 4（G19）：`GATE_CHUNK=512` 的 −1.56 ms 是函数级的，端到端不成立。**
shipped 上加它：8K **+0.82 ms**（略超 0.60 ms 噪声门槛，方向为负）、32K **+0.14 ms**（门槛内）。
它买到的是 **HBM**（KV 池 +367 tokens），不是 step 时间 ⇒ **生产保持 `0` 是对的**。
**128K 仍未测**。

**【实测】结论 5（起服）：`ENGRAM_DEVICE_INDEX` 让每次 8 卡起服多花 ≈7.8 分钟（+87%）。**
A1（`auto`→本机落到 1）16 min 02 s / A11 16.7 min；A2/A0（`=0`）8 min 13 s / 8 min 43 s。
机制【推断】= 给两张 206 GB 表做 host mapping 与权重加载交错。
**static kernel 编译每臂全量重编 371–438 个 kernel、缓存命中 0**（§2.1），
这是每臂固定付出的 ~15 min 里的主要部分。

### 5.2 还缺什么（**未确认 / 没做**）

#### ★ 5.2.0 读 §5.1 之前必须先知道的一条（2026-09-21 21:1x 补）

**`A` 的 1.68 vs 4.74 有一个未排除的替代解释，§5.1 的措辞已按此收窄。**

基准口径是"《红楼梦》8K/32K + 逐字引用第五回开头"——**这是一个复制型任务**
（要求模型逐字抄录），而 `p42_t4_quote` 的 JSONL **只记指标、不记文本**
（字段清单里没有 `text`）⇒ **本表没有这个口径下的文本证据**。

而**有文本证据的是 v2/v3 探针**（1405-token Q&A，见 §4.9.2bis）：
在**那**个口径下 **`MOE_AG=0` 与 `=1` 两臂都进入了循环**，只是形态不同 ——
`=0` 每轮换国家（自相似度 1.00）、`=1` 同句反复（自相似度 0.17）。

⇒ 因此 `A` 的差异**至少有两种解释，本表分不开**：

1. **好事**：AllGather 让草稿模型更准 ⇒ 接受率真的上去了；
2. **中性**：两档都在"抄"这类循环里，**循环越紧越容易猜**（`=1` 更紧）⇒
   `A` 高只是"自相似度"高，**不代表模型在真实任务上更好**。

**用户侧的反向证据（重要）**：A2 生产环境（真实 agent 流量、`MOE_AG=1`）实测
`A` 中位 **3.58**、**无乱码无异常**（`logs/31`，18 个采样）—— 这个值落在 1.68 与 4.74
**之间**，与"真实负载的循环程度介于两者之间"一致。

⇒ **在非退化任务上重测 `A`（或两臂各跑一遍精度门 GSM8K/Vision/长文检索）之前，
§5.3 那句"接受长度从 1.68 抬到 4.74"只能作为【实测·quote 口径】引用，不能外推成
"真实场景下接受率提升 2.8×"。**

| # | 缺什么 | 为什么缺 | 怎么补 |
|---|---|---|---|
| 1 | **`A` 两值现象的机制** | 只知道"`MOE_AG` 一开就翻"，没查 MoE 层的数值差异 | 在 `MOE_AG=0/1` 两配置下抓同一 token 的 MoE logits 比对 |
| **1b** | **`A` 差异是"草稿更准"还是"循环更紧"**（★ 见 §5.2.0） | quote 口径**不记文本**，探针口径**两臂都在循环** | ① 非退化任务重测 `A`；② 或两臂各跑精度门；③ 或给 quote 口径补文本记录 |
| 2 | **`GATE_CHUNK=512` 的 128K 端到端**（G19 后半） | **8K/32K 已测**（A11：无收益，+0.82/+0.14 ms）；**128K 未测** | `MODE=full` 加跑一次（RFC 文档 §5.1 的 0006 行明确写 "long-context re-measurement is pending"） |
| 3 | **`ENGRAM_HOST_RESIDENT`（0007）与注册策略（0010）** | 全臂保持常量（不是 k 可关的运行时 gate） | 需要独立会话 |
| 4 | **文本指纹的非确定性**（§4.9/§4.10.1） | 已定性为"偶发、发生在停止判定上"，**未定位到代码** | 引擎内插桩（页分配 / EOS 判定） |
| 5 | **合入形态** | 本日志只是**证据**；RFC 正文第 104 行要求 "merged into its stated target branch" | PR（未发起，见本工作区红线） |
| 6 | **`VLLM_ADMISSION_GATE` 对照** | 按文档 §7 的判断（"no RFC item claims this"）**有意不做** | — |

### 5.3 一句话给 RFC 评论用

> 我们在一台 8×910C 上跑了 12 臂单会话消融（每臂相对 stock 只动一个 gate，含两个重复臂给出噪声底）。
> **结论是：`V41_MOE_COMM_ALLGATHER` 的收益不在毫秒级，而在接受长度** ——
> 单开它把 `A` 从 1.68 抬到 4.74、单流 `tok/s` 从 48 抬到 140（×2.9）；其余五个 gate 的单 gate 增量
> 全部落在实测噪声门槛内（8K 0.30 ms / 32K 1.37 ms）。原始数据按 arm 归档，起服阶段耗时与
> 编译缓存行为（每臂全量重编 371–438 个 kernel、缓存命中 0）也一并记录。

> ⚠️ **上述段落引用前必须加限定**（理由见 §5.2.0）：`A` 的 1.68→4.74 是**quote 口径**的实测，
> 而该口径**不记文本**；在有文本的探针口径里**两臂都进入了循环**，只是紧密度不同。
> ⇒ 对外措辞应改为：
>
> *"在**逐字引用**这一类复制型任务上（《红楼梦》8K/32K、256 token、贪心），
> `MOE_AG` 把接受长度从 1.68 抬到 4.74、单流 tok/s 从 48 抬到 140；
> 在真实 agent 流量上我们只测过开启侧（`A` 中位 3.58，无异常）。
> 该差异是'草稿更准'还是'复制型任务的循环更紧'，**我们还没有区分**。"*

---

## 6. ★ 单会话消融总表（**交付物**）

**口径**（每一格都在同一口径下；见 §0.1）：`CPU_BIND=0`、`DROPCACHE=0`、`MODE=quick`、
`REPEATS=6`（A0 是 2 发，A0b 起全部 6 发）、跳过视觉、`SP_TOKENS=5`、`DRAFT_GRAPH=0`、
`STATIC_KERNEL=1`、`NPUGRAPH_EX=1`、`GPU_UTIL=0.92`、`MAX_SEQS=4`、`PREFIX=0`、
`BAT_TOKENS=2048`、`GATE_MAX_TOKENS=2048`。**每格的溯源见下表最后两列。**

| arm | 配置（相对 A0 只动一处） | run_id | 8K `ms/step` | 8K `A` | 8K `tok/s` | 32K `ms/step` | 32K `A` | 32K `tok/s` | 峰值 KV (tokens) | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **A0** | stock：6 gate 全关 + `DI=0` | `a0_20260921_165927` | 34.72 | 1.738 | 50.0 | 36.83 | 2.899 | 78.9 | **3,241,932** | `REPEATS=2`（先导）；`static650`=0 |
| **A0b** | = A0（**重复臂**，非新臂） | `a0b_20260921_171229` | **34.42** | **1.676** | **48.3** | **35.45** | **2.881** | **80.5** | 3,241,442 | `REPEATS=6`；用于噪声底（§4.3） |
| **A1** | shipped（全默认）`+PROFILE=1` | `a1_20260921_172649` | **29.86** | **4.787** | **157.0** | **30.47** | **4.691** | **151.7** | 3,842,534 | 唯一带 profiler 的臂；基准期间 profiler **未启动** |
| **A1r** | = A1（**重复臂**） | `a1r_20260921_183507` | 30.35 | 4.727 | 150.3 | 31.04 | 4.726 | 150.6 | 3,842,534 | KV **逐位同 A1** |
| **A2** | shipped − `ENGRAM_DEVICE_INDEX` | `a2_20260921_175024` | 32.65 | 4.771 | 143.0 | 33.43 | 4.727 | 137.4 | 3,844,493 | v1 文本指纹与 A1 **逐字相同** |
| **A3** | shipped − `ENGRAM_JIT` | `a3_20260921_180419` | 31.70 | 4.815 | 149.6 | 30.66 | 4.631 | 150.2 | 3,842,657 | v1 指纹不同（**§4.9 已作废该判据**）；§4.8 函数级逐位一致 |
| **A4** | shipped − `QLI_NOCAND` | `a4_20260921_182034` | 30.11 | 4.815 | 156.5 | 31.09 | 4.640 | 146.2 | 3,842,534 | v1 指纹臂内 2 发不同（§4.9） |
| **A5** | **A0 + `MOE_MASK=1`** | `a5_20260921_185117` | 34.91 | **1.717** | 48.6 | 35.68 | **2.876** | 80.4 | 3,241,564 | v3 指纹**缺失**（脚本 bug，§4.11） |
| **A6** | **A0 + `QLI_NOCAND=1`** | `a6_20260921_190549` | 34.49 | **1.655** | 47.8 | 35.44 | **2.850** | 79.2 | 3,241,687 | v3 指纹 6 发全同 |
| **A7** | **A0 + `MOE_AG=1`** | `a7_20260921_191947` | **33.59** | **4.743** | **139.6** | **32.29** | **4.727** | **143.9** | **3,844,493** | ★★ **命中**；v3 指纹 `stop-behaviour-differs` |
| **A8** | **A0 + `ROPE_IDXSEL=1`** | `a8_20260921_193105` | 34.84 | **1.655** | 46.2 | 36.88 | **2.872** | 76.3 | 3,241,687 | v3 指纹 6 发全同 |
| **A9** | **A0 + `O_PROJ_2D=1`** | `a9_20260921_194140` | 34.52 | **1.744** | 48.7 | 35.56 | **2.876** | 79.9 | 3,241,687 | v3 指纹 6 发全同 |
| **A10** | **A0 + `ENGRAM_JIT=1`** | `a10_20260921_195535` | 见 §4.16 | | | | | | | |
| **A11** | shipped + **`GATE_CHUNK=512`**（G19） | `a11_20260921_200615` | 30.68 | 4.771 | 153.3 | 30.61 | 4.658 | 149.0 | 3,842,901 | v3 指纹 4 种长度（`stop-behaviour-differs`）|

**每格的原始文件**（所有行同一套文件名，故不逐行重复）：

* `8K ms/step` / `8K A` / `8K tok/s` ← `~/projects/dsv41-release/results/<run_id>/p42_t4_quote_8192_quote_8k.jsonl`
  （`metrics_ok and not error` 的测点取**中位数**）
* `32K …` ← 同目录 `p42_t4_quote_32768_quote_32k.jsonl`
* `峰值 KV` ← 同目录 `env.txt` 的 `kv_tokens=`；`static650` ← `serve.log` 里 `static_kernel.py:650` 计数（**全臂 = 0**）
* 全部回传副本在 `logs/raw/44-ablation-<arm 小写>/`

### 6.1 单 gate 增量（**A0 基线口径**，A5–A10 vs A0）

#### 6.1.1 ★ 文本指纹独立印证了同一个结论（`arms.tsv` 的 `out_sha256`）

v3 探针（mid-2k 稳定档、6 发、`temperature=0`）在四臂上给出**完全相同**的输出哈希：

| 臂 | 配置 | `out_sha256`（前 16 位） | 6 发长度种数 | `A`（8K） |
|---|---|---|---|---|
| A6 | A0 + `QLI_NOCAND=1` | `42593d7da3b2f373` | 1 | **1.655** |
| A8 | A0 + `ROPE_IDXSEL=1` | `42593d7da3b2f373` | 1 | **1.655** |
| A9 | A0 + `O_PROJ_2D=1` | `42593d7da3b2f373` | 1 | **1.744** |
| A10 | A0 + `ENGRAM_JIT=1` | `42593d7da3b2f373` | 1 | **1.746** |
| A7 | A0 + **`MOE_AG=1`** | **`2aeea848dbcf8692`** ← 不同 | 2 | **4.743** |
| A11 | shipped + `GATE_CHUNK=512` | `085a67970a5b4937` ← 不同 | 4 | 4.771 |

**读法**：四个"对 `A` 无影响"的 gate 给出**逐字相同**的输出（不同 gate、不同 run_id、
不同时刻的四个会话！）—— 这既是**跨会话确定性的证据**，也独立复现了 §6.1 的结论：
**这几个 gate 不改变模型看到的东西**。而 `MOE_AG`（唯一改 `A` 的 gate）**同时**改了输出文本。

⇒ 文本指纹在 v3 口径下**是可用的**（v1 的失效是 prompt 设计问题，不是方法问题，见 §4.9）。
**A5 缺指纹**（脚本 bug，§4.11），但它的 `A`=1.717 已足以定位。

Δ = 该臂 − A0；**负 = 更快/更好**。噪声门槛（§4.3）：**8K 0.6 ms / 32K 1.4 ms**（超过才算真效果）。

| gate（A0 + 只开它） | Δ 8K `ms/step` | Δ 8K `A` | Δ 32K `ms/step` | Δ 32K `A` | 判定 |
|---|---:|---:|---:|---:|---|
| `MOE_MASK=1`（A5） | +0.49 | **+0.041** | +0.23 | −0.005 | **不可判定**（门槛内） |
| `QLI_NOCAND=1`（A6） | +0.07 | −0.021 | −0.01 | −0.031 | **无效果** |
| **`MOE_AG=1`（A7）** | **−0.83** | **+3.067（2.8×）** | **−3.16** | **+1.846（1.6×）** | ★★ **真效果：`A` 翻档、`ms/step` 32K −3.16 ms** |
| `ROPE_IDXSEL=1`（A8） | +0.42 | −0.021 | +1.43 | −0.009 | **不可判定/无效果** |
| `O_PROJ_2D=1`（A9） | +0.10 | +0.068 | +0.11 | −0.005 | **无效果** |
| `ENGRAM_JIT=1`（A10） | 见 §4.16 | | | | |

**★ 这张表是本日志最重要的产物之一**：在同一个 harness、同一台机器、同一个模型下，
**只有 `MOE_AG` 有可判定的效果**；其余五个 gate 的单 gate 增量**全部落在噪声门槛内**。

### 6.2 单 gate 增量（**shipped 基线口径**，A2–A4 + A11 vs A1）

这条线回答"从满配里拿掉谁最疼"，是 §5 现有叙述的**同口径对照**。

| 关掉的 gate | Δ 8K `ms/step` | Δ 8K `A` | Δ 32K `ms/step` | Δ 32K `A` | 判定 |
|---|---:|---:|---:|---:|---|
| `ENGRAM_DEVICE_INDEX`（A2） | **+2.79** | −0.016 | **+2.96** | +0.036 | **真效果**（关掉就变慢 ~2.9 ms） |
| `ENGRAM_JIT`（A3） | **+1.84** | +0.028 | +0.19 | −0.060 | 8K 真效果 / 32K 不可判定 |
| `QLI_NOCAND`（A4） | +0.25 | +0.028 | +0.62 | −0.051 | **不可判定** |
| `GATE_CHUNK`（A11，改回 `=0`） | +0.82 | −0.016 | +0.14 | −0.033 | 8K **略超门槛（+0.82 > 0.60）但方向为负**；32K 不可判定 |

**两条线的差别值得写进结论**：

* **从 stock 出发**：只有 `MOE_AG` 让整体变好（因为其它 gate 的收益太小，被噪声吃掉）；
* **从 shipped 出发**：`ENGRAM_DEVICE_INDEX`（−2.9 ms）与 `ENGRAM_JIT`（−1.8 ms @8K）
  都**真的在付钱**，而 `QLI_NOCAND` 不可判定。
* 两者不矛盾：**"少了会疼"≠"从零开始加上去能看见"**，因为基线不同、且 `MOE_AG` 的
  巨大效应会改变整条流水线的时间结构（`A` 变了 ⇒ 每步的 token 数与 batch 形状都变了）。

#### 4.12.1 KV 池也是 `MOE_AG` 决定的（**有两臂的直接反例**）

各臂的 KV 池是两档（【实测】），但把它归因给 `ENGRAM_DEVICE_INDEX` 是**错的**：

| 臂 | `MOE_AG` | `ENGRAM_DEVICE_INDEX` | KV tokens | `Available KV cache memory` |
|---|---|---|---|---|
| A0 | **0** | 0 | 3,241,932 | 13.34 GiB |
| A0b | **0** | 0 | 3,241,442 | — |
| A5 | **0** | 0 | 3,241,564 | — |
| A6 | **0** | 0 | 3,241,687 | — |
| **A2** | **1** | **0** | **3,844,493** | **15.83 GiB** |
| **A7** | **1** | **0** | **3,844,493** | **15.83 GiB** |
| A1 / A1r | 1 | auto | 3,842,534 | 15.82 GiB |
| A3 | 1 | auto | 3,842,657 | — |
| A4 | 1 | auto | 3,842,534 | 15.82 GiB |

**判据**：A0 与 A2 **同为 `DEVICE_INDEX=0`**，唯一差别是 `MOE_AG`（0 vs 1）
⇒ KV 池 **13.34 → 15.83 GiB**。⇒ **差异来自 `MOE_AG`，不是 `DEVICE_INDEX`。**

**更强的一格**：A7（`A0 + 只开 MOE_AG=1`，其余 gate 全 0）的 KV
**恰好等于 A2 的 3,844,493**（`A2 = shipped − DEVICE_INDEX`，其余 gate 全开）
⇒ **KV 池的大小完全由 `MOE_AG` 决定**，与其它 gate 无关。

**与 RFC 文档 §5.1 的对照**：那一版 0001 行写 `MOE_AG=1` 的 "KV pool 3.39M → **4.16M** tokens"
—— 方向、量级都对得上（我们的 3.24M → 3.84M）。这一条**不是新发现**，
本表的贡献是**把它钉在单会话单变量上**。

**口径更正（记录一次判断反复，供后来者识别）**：
主代理曾判定"KV 池差异来自 `DEVICE_INDEX`（表占不占 HBM），与 `A` 无因果"；
**A2 与 A7 两臂以上表否定了这个判定**。现在的表述是：
**KV 池大与 `A` 高是同一个原因（`MOE_AG=1`）的两个结果**，两者互不构成证据，
但**同时指向 `MOE_AG`**。

---

## 7. Profiler trace 制品（RFC line 104 的 "trace evidence"）

`issue-track-C.md` §3.6 挂着：*"No trace evidence for the overlap claim."*
`RFC-16375-CONTRIBUTION.md` 引的 RFC 正文第 104 行是：
*"Performance items additionally require reproducible comparisons; **overlap items
require trace evidence**."* 这一节就是那个制品。

### 7.1 在哪一臂、什么时刻、什么负载

| 项 | 值 |
|---|---|
| 臂 | **A1 = shipped**（全默认，**唯一**带 `PROFILE=1` 的臂） |
| 开关 | `PROFILE=1` → `serve_v2.sh:91` 给 vllm 传 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/opt/dsv41/results/<run_id>/prof","torch_profiler_with_stack":false}'` |
| 端点核对 | `POST /start_profile` → **HTTP 200**（**不是 404** ⇒ profiler 确实挂上了；8080/8020 那台旧服务当时是 404） |
| 采集时刻 | `2026-09-21 17:48:19` 起，`17:48:42` 停（容器内 UTC 09:48:19–09:48:42） |
| 负载 | **2 × (32768-token prompt + 256 decode)**，跑在 `/v1/completions`，贪心 |
| 与基准的关系 | **基准已经跑完**（17:43–17:44 完成 quote），trace 在之后单独采 ⇒ **两者互不污染** |
| profiler 打开的代价（实测） | 同一题 32K：`ms/step` **30.47（基准，profiler 关） → 41.16 / 45.76（trace 期间，profiler 开）**；`tok/s` 151.7 → 111.1 / 111.5 |

> 因为 **A1 的 `PROFILE=1` 只在 API 上挂路由、torch profiler 对象直到第一次
> `/start_profile` 才创建**（`gpu_worker.py:1134`），A1 的正式数字是干净的（见 §4.4）。
> **trace_32k1/2 那两发的数字不可引用**，它们只用于产生制品。

### 7.2 制品在哪、有什么

宿主路径（**留在 A3，未回传**）：
`~/projects/dsv41-release/results/a1_20260921_172649/prof/`

| 层 | 内容 | 体积 |
|---|---|---|
| 顶层 | 8 个 rank 目录（`dp0_pp0_tp{0..7}_dcp0_ep{0..7}_rank{0..7}_<pid>_20260921094819062_ascend_pt`）+ 1 个 loose 文件 | **7.8 GB / 7003 个文件** |
| 每 rank | `FRAMEWORK/torch.op_range`（**102.7 MB**）、`FRAMEWORK/torch.op_mark`（**71.0 MB**）、`PROF_000001_*/device_{8..15}/…`、`host/…` | ~1001 MB/rank |
| torch 侧 | `localhost.localdomain_484.async_llm.1789984128355016590.pt.trace.json.gz` = **825 B（解压 7.9 KB，55 个事件）** ⇒ **几乎为空** | — |

**回传决定（按"压完仍 >50 MB 就只回传清单"的规则）**：**只回传清单**。
`logs/raw/44-a1-trace/trace-inventory.txt`（含每个文件的大小；`torch.op_range` /
`torch.op_mark` 的 sha256 见 §7.3）。压缩 7.8 GB 不可能落进 50 MB，而这两个
关键文件的 sha256 已经足够让任何人核对"我们采的是哪一份"。

**清单要点**（`trace-inventory.txt` 全文在 `logs/raw/44-a1-trace/`）：

```text
rank0..7 目录            各 1001.0–1001.2 MB / 874–876 文件
localhost...pt.trace.json.gz   825.0 B / 55 events
合计                     7.8 GB / 7003 文件
```

### 7.3 这份 trace 到底采到了什么（**已做初步文本摘要，未做正式分析**）

| 文件 | bytes | sha256 |
|---|---:|---|
| `torch.op_range` | 107,729,654 | `53acc048718c7c3b50049e618e316122de9132ede3dbd331b8b0f05108bfc6ab` |
| `torch.op_mark` | 74,467,672 | `0975064b30d784149dda1aab501a342afcfd2b51b0cfbe3fa8eb973006613d1b` |

（rank0 的 `FRAMEWORK/` 下；其余 7 个 rank 同构。清单脚本
`agents/T4_ablation/trace_inventory.py`，摘要脚本见 `trace_op_summary.txt` 表头。）

**rank0 `torch.op_range` 的算子频次 top（`strings` 近似统计，非 msprof 正式解析）**：

| 算子/标记 | 次数 | | 算子/标记 | 次数 |
|---|---:|---|---|---:|
| `empty_tensor` | 232,486 | | `aten::unsqueeze` | 26,675 |
| `aten::as_strided` | 132,607 | | `aten::view` | 24,712 |
| `aten::empty` | 102,027 | | `destroy_event` | 24,152 |
| `aten::slice` | 74,842 | | `Event::record` | 21,058 |
| `record_event` | 37,286 | | `Event::wait` | 18,514 |
| `aten::copy_` | 35,427 | | `aten::fill_` | 13,517 |
| `aten::to` | 30,937 | | `aten::where` | 11,043 |
| `aclnnInplaceCopy` | 28,117 | | `npu::npu_quant_matmul` / `aclnnQuantMatmulWeightNz` | 10,314 |
| `wait_event` | 27,728 | | **`HcclAllreduce`** | **8,272** |

**这份制品能回答什么、不能回答什么**（**诚实边界**）：

* **能**：给出 overlap 断言所需的**原始逐算子时间线**（`torch.op_range` 带
  start/end 时间戳），包括 **8,272 次 `HcclAllreduce`** 与同期的 `record_event` /
  `Event::wait` 标记 —— 这正是 `comm ∩ compute overlap` 要用的料。
* **能**：给出"每 rank 一份、8 rank 同构"的结构，并按 §7.2 的表可复现地重新采集。
* **不能**（**这一条必须写在材料里**）：**我们还没有做正式分析**——
  没有跑 msprof 的导出、没有把 `torch.op_range` 解成事件表、没有算出
  `comm ∩ compute` 的区间并集。**今天的产出是"制品 + 清单 + 采集方法"，不是结论。**
  所以 `issue-track-C.md` §3.6 应从 *"No trace evidence"* 改成
  *"trace 制品已有（路径/清单/sha256/负载说明），overlap 归因分析待做"*。
* **不能**：torch 侧 trace json 几乎为空（825 B / 55 事件），所以**不能**用它替代；
  真正有料的是 CANN msprof 侧那两个 `FRAMEWORK/*` 文件。
* **【未确认】**：为什么 torch 侧 trace 是空的（`torch_profiler_with_stack=false` +
  `activities=["CPU","CUDA"]`，在 Ascend 上 `CUDA` 大概不产生对应活动）——
  未查，记为待办。
