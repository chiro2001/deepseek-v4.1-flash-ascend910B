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

* `vllm_ascend/cpu_binding.py:663` 起的 `bind_memory()`：
  由 `[migrate] NPU:<n> -> NUMA [<node>]` 触发的 `migratepages <pid> 0,1,...,7 <node>`；
  通过 `execute_command()` **同步等待**。
* 由 `additional-config` 的 `enable_cpu_binding: true` 驱动
  （我们这边对应起服脚本的 `CPU_BIND=1`，是**默认值**）。

### 三个"不该这样"的点

1. **不检查目标节点的空闲内存** —— 一个 `MemFree` 读一下就够；
2. **没有超时、没有回退** —— 失败时不报错、不降级为"不绑"或"绑 CPU 不迁页"，而是无限等；
3. **进程杀不掉** —— `kill -TERM` / `kill -9` 都无效，`R` 态持续 100% CPU，
   **容器 `docker rm -f` 之后仍在宿主上活着**（父进程变成 `[sleep]`、PPid=1），
   只能等机器重启（或内核最终回收）。这会把"一次配置错误"升级为"宿主上永久漏两个核"。

---

## 5. 影响面

* **共享集群 / 多租户机器上风险最高** —— NUMA 分布不均是常态，而排序靠后的节点
  （本例 node6/node7）往往就是被邻居占满的那些；
* 一旦命中，**服务无法通过重试解决**（每次重起都会重新挂住），
  且**每挂一次留一对杀不掉的进程**；
* 由于 `enable_cpu_binding` 是绑核优化的一部分（我们这边的 RFC [75]/[77] 语境），
  用户很难想到"起不来"是因为 NUMA 空闲量。

---

## 6. 建议的修法（供参考，不是要求）

任选其一即可止血：

1. **迁页前读 `MemFree`**：如果目标节点空闲 < 该进程可迁移页量（或低于某个阈值），
   跳过迁页并打一条 WARN，只保留**绑 CPU**（那部分与内存无关）；
2. **给 `execute_command` 加超时 + 失败回退**：超时后 kill 子进程并按"未绑定"继续启动；
3. **把失败暴露成错误**：宁可起服失败并打印"node N 满、无法迁页"，
   也不要静默挂死（**这一条最重要**，它把问题从"神秘卡死"变成"一条可读的报错"）；
4. 顺手：`migratepages` 失败时不要让它留在宿主上空转 —— 也就是 2 的 kill 分支。

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
也没有验证内核侧为什么 `SIGKILL` 拿不掉它（没有内核调试手段）。
我们把 `CPU_BIND=0` 当作绕过办法（代价是失去绑核优化，
而本机的绑定诊断本身是 RFC [75] 语境下的另一项工作）。

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
