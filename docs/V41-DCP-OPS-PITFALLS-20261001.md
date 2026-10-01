
---

## 坑 4：`ENGRAM=1` + `CPU_BIND=1` ⇒ `migratepages` 卡死在 D 状态（2026-10-01 本轮踩到）

### 症状

图捕获 100% 完成后**再无任何日志推进**，EngineCore 每 60 s 刷：

```
INFO [shm_broadcast.py:802] No available shared memory broadcast block found in 60 seconds.
     This typically happens when some processes are hanging or doing some time-consuming work
```

`/metrics` 不存在、health 000，但容器不退出、8 个 worker 各烧 ~28% CPU（看起来像在干活）。

### 误判路径（值得记，避免下次又走过去）

1. 先看日志尾部有 `AOE_RTKB` 线程 ×16 ⇒ 以为在**编译 static kernel**。
   实际那是 **AOE 的常驻线程池**，编译结束后线程仍在，**不能**作为"在编译"的证据。
2. 查 `static_kernel_compile_outputs/` 最新 `ts*` 目录：**mtime 7 分钟没动、文件数没涨** ⇒ 排除编译。
3. `ps -eo pid,pcpu,etime,stat,comm` 才看到真凶：

```
1178590 1.8  05:18 D  migratepages 2492 0,1,2,3,4,5,6,7 4
1178623 4.4  05:18 D  migratepages 2628 0,1,2,3,4,5,6,7 6
1178648 4.4  05:18 D  migratepages 2674 0,1,2,3,4,5,6,7 6
1178672 1.6  05:18 D  migratepages 2473 0,1,2,3,4,5,6,7 4
```

**`D` = uninterruptible sleep**，`etime` 已 5 分 18 秒。这正是 `cpu_binding.py:663` 那行
`[migrate] NPU:x -> NUMA [y]` 派生的动作。

### 为什么这次特别慢

| 变量 | 基线 `dcpcap_1001_102816` | 本次 `dcpcap_1001_1135_spec1` |
|---|---|---|
| `ENGRAM` | **0** | **1** |
| 容器 host 内存 | 小 | **242.8 GiB** |
| `CPU_BIND` | 1 | 1 |

`ENGRAM=1` 且宿主常驻（`V41_ENGRAM_HOST_RESIDENT`）时进程的 host 驻留内存极大，
`migratepages` 要把这些页跨 NUMA 节点搬 ⇒ 在**宿主机 swap 已满（3/3 GiB）**的内存压力下
几乎不动。

对照：`dcpcap_1001_102816`（ENGRAM=0）同样有 `[migrate]` 行，但**没有**卡住。

### 处置

任选其一（按推荐度）：

1. **`ENGRAM=0`** —— 推荐。除了绕开这个坑，它还让 DSpark 的 A/B **单变量**
   （line A 的 32.58 基线就是 SPEC=0/ENGRAM=0）。
2. `CPU_BIND=0` —— 关掉 CPU 绑定，不派生 `migratepages`。代价是失去绑核。
3. 等 —— 不推荐，宿主机 swap 满时可以是数十分钟量级。

### 判据（如何一眼分辨"编译中" vs "卡在 migratepages"）

```bash
# 编译中：最新 ts* 目录的 mtime 与文件数在涨
C=~/cedpd-repo/cache/skcache/compile_outputs
ls -td $C/ts*/ | head -1 | xargs stat -c %y
# 卡住：migratepages 在 D 状态且 etime 很大
ps -eo pid,etime,stat,args | grep -E "migratepages" | grep -v grep
```

⚠️ **不要**用 `AOE_RTKB` 线程数判断是否在编译 —— 它是常驻线程池。
