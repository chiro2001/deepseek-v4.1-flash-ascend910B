# Issue 草稿 —— `enable_cpu_binding` 在目标 NUMA 节点已满时**永久挂死**（草稿，未提交）

- **目标仓**：`vllm-project/vllm-ascend`
- **建议标题**：`[Bug] enable_cpu_binding hangs engine startup forever when the target NUMA node is full (migratepages never converges)`
- **建议 label**：`bug`（若可加：`CPU affinity` / `startup`）
- **严重性**：**阻塞级** —— 服务永远不到就绪，并留下**内核态空转、`kill -9` 无效**的进程
- **发现于**：A3（Atlas A3，8×910C 类），2026-09-21
- **状态**：**草稿，未提交**（用户未授权发 issue）

---

## 1. 一句话

`additional-config: {"enable_cpu_binding": true}` 在起服时对每个 rank 调
`migratepages <pid> 0,1,...,7 <target_node>` **同步等待且无超时**；当**目标节点的空闲内存
远小于要迁移的页量**时，`migratepages` 不返回错误、而是**在内核里无限空转（100% CPU）**，
于是 vLLM 永远停在"图捕获已完成、等待就绪"的状态。

---

## 2. 现象（可复现的时间线）

8 张 die、TP8/EP8、模型带 **206 GiB 的 Engram 表**（host-mapped，`aclrtHostRegister` 路径）：

| 相对时间 | 事件 |
|---:|---|
| 0 | 容器启动 |
| +36 s | `Initializing a V1 LLM engine` |
| +8:24 | 权重加载完成（8 × 39.118 GB） |
| +10:30 | KV cache 容量确定（3,842,657 tokens） |
| **+17:25** | **`Graph capturing finished in 361 secs`** ← 一切都正常 |
| **+17:28** | 8 条 `[migrate] NPU:8..15 -> NUMA [...]` |
| +17:28 起 | **永不就绪**；35 分钟后等待超时 |

卡住的是 `NPU12` 与 `NPU13` **两个** rank（两者都映射到 **NUMA node 6**）。

```
$ ps -eo pid,etime,pcpu,args | grep migratepages      # 宿主视角，已持续 20+ 分钟
509177  20:10  99.8  migratepages 1399 0,1,2,3,4,5,6,7 6
509188  20:10  99.8  migratepages 1353 0,1,2,3,4,5,6,7 6
```

---

## 3. ★ 根因：目标节点是满的（而代码不检查）

```
node 6 size: 256991 MB
node 6 free:     22 MB          ← 99.99% 满
Node 6 AnonPages: 70,721,640 kB   (~67 GiB)
Node 6 FilePages: 31,627,440 kB   (~30 GiB)
```

每个 worker 的 `VmRSS ≈ 89–91 GB`。**要往一个只剩 22 MB 的节点迁 90 GB ⇒ 永远不可能成功。**

### 零进展的直接证据（3 分钟采样）

| 量 | 3 分钟前 | 3 分钟后 | Δ |
|---|---:|---:|---:|
| `Node 6 MemFree` | 24,884 kB | 24,908 kB | **+24 kB** |
| `Node 6 AnonPages` | 70,648,168 kB | 70,648,112 kB | **−56 kB** |
| `Node 6 FilePages` | 31,627,384 kB | 31,627,384 kB | **0** |

⇒ 不是"慢"，是**不收敛**。

### 这不是本次实验造成的

同一台机器在**更早的一次测量**里就已经记录到这个不均衡
（`logs/38-20260921-host-dram-bandwidth.md` §1.6，12:57 采，远早于本次起服）：

> *"Free memory is very uneven on this shared box (`node0` ~1.0–2.9 GB, `node6`/`node7`
> ~0.3 GB, `node4` ~60 GB, `node5` ~43 GB at 12:57)."*

---

## 4. 代码位置

根因是**两层**的，第二层才是真正让它"永久"挂住的原因。

### 4.1 触发：`bind_memory()` 不检查目标节点能不能装下

`vllm_ascend/cpu_binding.py`，`CpuAlloc.bind_memory()`（我们这份是第 639–671 行）：
由 `[migrate] NPU:<n> -> NUMA [<node>]` 触发的
`migratepages <pid> 0,1,...,7 <node>`，经 `execute_command()` **同步等待**。
调用链：`run_all() → bind_threads() → bind_memory()`（第 689 / 694 行）。
由 `additional-config` 的 `enable_cpu_binding: true` 驱动
——它在 `ascend_config.py` 里**默认就是 `True`**（我们这份第 387 行），
我们起服脚本里对应 `CPU_BIND=1`。

它只做了三种"提前退出"：`migratepages` 不存在、NPU 没有 CPU pool、目标 NUMA 不在表里。
**唯独没有检查目标节点还剩多少内存**。

### 4.2 ★ 真正致命的：`execute_command()` 的 timeout 分支**本身没有 timeout**

```python
def execute_command(cmd: list[str]) -> tuple[str, int]:
    ...
    with subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env) as p:
        try:
            out, _ = p.communicate(timeout=1000)          # ← 看起来有 1000 s 保护
        except subprocess.TimeoutExpired:
            p.kill()                                      # ← SIGKILL
            out, _ = p.communicate()                      # ← 没有 timeout，可以永远阻塞
    return out.decode(), p.returncode
```

`p.kill()` 之后那次 `p.communicate()` **没有超时**。而 `migratepages` 陷在内核路径里时
**不会及时处理 SIGKILL**（信号只在检查点生效），于是：

> **1000 秒的"保护"在超时那一刻变成了永久阻塞** ——
> engine 卡在 `bind_memory()` 里，永远到不了 READY。

我们的时间线正好落在这个窗口上：`migratepages` 于 `+17:28` 启动，
1000 s 后（`+34:08`）超时并调用 `p.kill()`，随后 `p.communicate()` 永久阻塞；
外层等待在 **35 分钟**处放弃 —— 与观测完全吻合。

**这也解释了"杀不掉"**：SIGKILL 确实发了，但进程要到能处理信号时才会退出。
我们那两个进程**约 2.5 小时后自己消失了**（内核路径最终返回、挂起的 SIGKILL 生效）
——所以它不是永久泄漏，但对一次起服来说已经足够致命。

### 三个"不该这样"的点

1. **不检查目标节点的空闲内存** —— 一次 `MemFree` 读取就够；
2. **超时分支没有超时** —— 见 §4.2，这把"有界等待"变成"无界阻塞"；
3. **失败被静默吞掉** —— `bind_memory()` 把 `execute_command()` 的返回值**直接丢弃**，
   连 `returncode` 都不看；即使 `migratepages` 快速失败，日志里也不会有任何痕迹；
4. **没有降级路径** —— 迁页失败时不能退回"只绑 CPU、不迁页"，而是拿整个起服陪葬；
5. **杀不掉的那段时间** —— `R` 态持续 100% CPU，容器 `docker rm -f` 之后仍在宿主上活着
   （父进程变成 `[sleep]`、PPid=1），约 2.5 小时后才自行退出。一次配置问题因此
   变成"宿主上两个核被占几小时"。

---

## 5. 影响面

* **共享集群 / 多租户机器上风险最高** —— NUMA 分布不均是常态，而排序靠后的节点
  （本例 node6/node7）往往就是被邻居占满的那些；
* 一旦命中，**服务无法通过重试解决**（每次重起都会重新挂住），
  且**每挂一次留一对杀不掉的进程**；
* 由于 `enable_cpu_binding` 是绑核优化的一部分（我们这边的 RFC [75]/[77] 语境），
  用户很难想到"起不来"是因为 NUMA 空闲量。

---

## 6. 修法（附一个已验证可用的补丁）

我们写了一个 77 行的补丁，**改两处**，已在真实源码上 `patch -p1` 干净应用 + 语法检查通过：

`pr/patches/cpu-binding-hang-fix.patch`

### 6.1 `execute_command()`：把超时分支变成有界的（**最重要**）

`p.kill()` 之后用**带 timeout 的** `communicate(timeout=30)`；如果还是不出来，
打一条 WARN **直接返回 `("", -9)`**，让调用方继续往下走，而不是永远等：

```python
            p.kill()
            try:
                out, _ = p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("[cpu_binding] '%s' survived SIGKILL for 30 s; abandoning it. "
                               "It may stay on the host until the kernel reclaims it.", cmd[0])
                return "", -9
```

### 6.2 `bind_memory()`：迁页前先看目标节点装不装得下

加了一个 `_node_mem_free_bytes(numa_node)`（读
`/sys/devices/system/node/node<N>/meminfo` 的 `MemFree` —— 我们实测**容器内可读**），
并与本进程 RSS 比较；装不下就**跳过迁页 + WARN**，只保留绑 CPU：

```python
        target_free = _node_mem_free_bytes(target_numa)
        wanted = psutil.Process(int(pid)).memory_info().rss
        if target_free is not None and wanted and target_free < wanted:
            logger.warning("[migrate] NPU:%s -> NUMA [%s] skipped: the node has %.1f GiB free "
                           "but this rank holds %.1f GiB, so migratepages would not converge. "
                           "...", npu, target_numa, target_free / 1024**3, wanted / 1024**3)
            return
```

**验证**：把补丁后的 `_node_mem_free_bytes()` 放进出问题的那个容器里跑（同一台机器、
同样的 `/sys` 挂载），读数是 `node0 37.2 / node4 0.3 / node5 13.7 / node6 0.9 / node7 0.3 GiB`、
不存在的 node 返回 `None`。
⇒ **在出问题的那一刻（node6 ≈ 22 MB、rank ≈ 90 GiB）这两个分支都会命中**：
迁页被跳过并打 WARN，engine 继续起服。

### 6.3 建议同时做的（我们没写进补丁，属于设计层面）

* **把"绑 CPU"和"迁内存"拆成两个开关** —— 现在一个 `enable_cpu_binding` 同时控制两件
  风险完全不同的事：绑 CPU 无害、迁页会挂死；
* **别吞返回值**：`bind_memory()` 应该看 `execute_command()` 的 `returncode` 并记录；
* 或者干脆**把 `migratepages` 换成 `set_mempolicy(MPOL_BIND)`**：
  它是**设置策略**、立即返回、由内核在首次触碰时落地，**不存在"迁移一个满节点"这种
  无界操作**（我们在这台机器上测过 `numactl --membind` 被容器 seccomp 拒，
  但 vllm 进程本身有权限的话这是更稳的语义）。

> 补丁是**给讨论用的最小修改**，不是要求你们照抄 —— 6.1 那一处我们认为是必须的
> （它把一个"神秘卡死"变成"一条可读的报错 + 继续起服"）。

---

## 7. 复现与证据（self-contained）

```bash
# 0) 前提：目标 NUMA 节点已满（MemFree 远小于每 rank 的 RSS）
npu-smi info | sed -n '/Process id/,$p'          # 确认 8 个 VLLMWorker_TP 在跑
grep -E "MemFree" /sys/devices/system/node/node6/meminfo
ps -eo pid,etime,pcpu,args | grep [m]igratepages  # 期望：两个 99% CPU 的 migratepages

# 1) 起服（关键开关）
CPU_BIND=1 ... bash scripts/run_test.sh           # 这一步会挂；日志里能数出 8 条 [migrate]

# 2) 观测"零进展"
for i in 1 2 3; do sleep 60; grep -E "MemFree|AnonPages" /sys/devices/system/node/node6/meminfo; done
```

**边界（我们没测到的）**：我们没有去二分"目标节点要留多少空闲才会成功"
——这台机器上 node6/node7 长期是满的，找不到一个能成功的中间点。
也没有验证内核侧**为什么** `SIGKILL` 要等约 2.5 小时才生效
（没有内核调试手段；`/proc/<pid>/stack` 在 R 态下也没给我们可读的调用栈）。
我们把 `CPU_BIND=0` 当作绕过办法（代价是失去绑核优化，
而本机的绑定诊断本身是 RFC [75] 语境下的另一项工作）。

**两个进程最终自己消失了**（约 2.5 小时后），所以它不是永久泄漏 ——
但对"起服"这个场景来说，35 分钟等不到就等同于失败。

**复现的最小判据**（不需要 8 卡）：只要目标 NUMA 节点为空闲 < rank RSS，
`migratepages <pid> <all-nodes> <full-node>` 就会进入同样的不收敛状态；
用 `free -g` + `numactl -H` 挑一个满节点即可。

---

## 8. 附：这条与我们的其它材料的关系

* 它**不是** RFC 条目的"完成证据"，而是一条**独立的 bug 报告**；
* 与 RFC [75]（`npugraph_ex` + static-kernel 的 warmup/编译缓存行为）相邻：
  同一批测量里我们还记录到 static kernel **每臂全量重编 431 个**、
  `static_kernel_cache/` 里**没有任何可复用产物**、`compile_outputs/` 里 3568 个
  `ts*_outputs` 从未被复用、`[SKCACHE-GC]` 因**软链 + `find` 不追软链**从未触发
  —— 详情见 `logs/44-20260921-single-session-ablation.md`；
* 提交前需要补的：把上面的数字换成**你们自己机器**的可复现值，
  并把标题里的 `migratepages` 换成你们代码里实际调用的命令名（我们的版本是 CANN 的 `migratepages`）。
