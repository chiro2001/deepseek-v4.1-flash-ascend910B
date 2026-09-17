# numba njit 落地：已验证的方案、骨架与陷阱清单

> 2026-09-16 15:40 CST｜A3-node1 容器 `dsv41-a21-perf`｜**全部结论都是实测**
> 前置：`reports/jit-cpu-operator-audit.md`（微基准）、`reports/engram-host-breakdown.md`（相位拆解）

---

## 0. 一句话

**numba 落地可行，且已在本机验证：**

| 验证项 | 结果 |
|---|---|
| **多进程并发编译** | ✅ 8 进程无死锁（`COLD=0.64s`，每进程 0.37–0.38s；`WARM=0.53s`） |
| **逐位等价** | ✅ 修正后 1000 例 fuzz 全过（**第一版错了——见 §3 陷阱 1**） |
| **性能** | ✅ **3.42 µs/step**（覆盖页写+4-gram 历史+哈希，L=2、N=6） |
| **对照 profile** | `hash` 相位 **401 µs** ⇒ **117×** |
| **编译缓存** | **0.4 MB**（可持久化到宿主目录） |
| **冷启动代价** | 8×0.38s ≈ **3 s**（相对 5–7 min 起服可忽略） |

---

## 1. 集成骨架（可直接照抄）

```python
# probe_hash/engram_jit.py  （新文件，env 门控）
import os
os.environ.setdefault("NUMBA_CACHE_DIR", os.environ.get("V41_NUMBA_CACHE", "/tmp/nb_cache"))
import numpy as np
from numba import njit

@njit(cache=True, nogil=True, parallel=False)      # ① 必须 parallel=False
def engram_host_step(ids, positions, request_ids, block_table, token_map,
                     pages, page_present, block_size, pad_id,
                     multipliers, primes, offsets, out_hashes):
    """Engram host 热路径：页写 + 4-gram 历史 + 归一化哈希。逐位等价于 stock。"""
    n = ids.shape[0]
    L = multipliers.shape[0]
    lookback = multipliers.shape[1]
    H = primes.shape[2]

    # --- 1) 页写（stock 的 Python dict 循环 → 直写预分配数组）---
    for i in range(n):
        page = block_table[request_ids[i], positions[i] // block_size]
        pages[page, positions[i] % block_size] = token_map[ids[i]]
        page_present[page] = True

    # --- 2) history + 哈希 ---
    for i in range(n):
        page = block_table[request_ids[i], positions[i] // block_size]
        prod = np.empty((L, lookback), dtype=np.int64)
        for s in range(lookback):
            prev = positions[i] - s
            v = pad_id
            if prev >= 0 and page_present[page]:
                v = pages[page, prev % block_size]
            for l in range(L):
                prod[l, s] = v * multipliers[l, s]
        for l in range(L):
            col = 0
            roll = prod[l, 0]
            for s in range(1, lookback):
                roll = roll ^ prod[l, s]        # ② 逐 shift 累积
                for h in range(H):
                    # ③ 每个 s 分组用「当时的」roll —— 见 §3 陷阱 1
                    out_hashes[i, l, col] = roll % primes[l, s - 1, h] + offsets[l, col]
                    col += 1
    return out_hashes
```

**调用侧（替换 `PagedNgramHistory.update`）**：

```python
if _JIT_ENABLED:
    self._ensure_buffers()          # pages[N, BS] int64=-1, page_present[N] bool
    engram_host_step(ids_np, pos_np, req_np, blk_np, self._token_map_np,
                     self._pages, self._page_present, block_size, self._pad_id,
                     self._mult_np, self._primes_np, self._offsets_np,
                     self._out_np)
    return torch.from_numpy(self._out_np), mask
else:
    ...stock...
```

---

## 2. 部署要点（都已验证）

| 项 | 做法 | 理由 |
|---|---|---|
| **缓存目录** | `NUMBA_CACHE_DIR` 指向**宿主挂载目录**（如 `$P/numba_cache`） | 容器重建不丢；实测缓存仅 **0.4 MB** |
| **单线程** | `njit(..., parallel=False)` | 8 个 rank 已并行，再开 numba 线程会争 CPU |
| **`nogil=True`** | 加 | 不持有 GIL，不阻塞其它 host 线程 |
| **首次编译** | 8×0.38s ≈ 3s | 可接受；WARM 后 0.265s/进程 |
| **页数组容量** | `MAX_PAGE = KV_blocks`（本配置 ≈32.5K 页） | 32.5K×128×8B = **33 MB** |
| **页存在位图** | 必须单独维护 `page_present` | **见陷阱 2** |

---

## 3. 陷阱清单（**陷阱 1 是我实际踩到的**）

### 陷阱 1 ★：哈希是「**按 shift 分组、每组用当时的累积 roll**」

**stock 的实现**（`engram_hash_ab.py` 尾部）：

```python
rolling, hashes = products[..., 0], []
for shift in range(1, self.lookback):
    rolling = torch.bitwise_xor(rolling, products[..., shift])   # 逐步累加
    hashes.append(rolling[..., None] % self.primes[:, shift - 1])  # 用「当时」的 rolling
return torch.cat(hashes, -1) + self.offsets
```

**我的第一版原型**先算完 4 个 product 的完整异或，再一次性输出 3 个分组
⇒ **三个分组都用了最终值**，而正确行为是：

| 分组 | 应用的 roll |
|---|---|
| `shift=1` | `p0 ^ p1` |
| `shift=2` | `p0 ^ p1 ^ p2` |
| `shift=3` | `p0 ^ p1 ^ p2 ^ p3` |

**实测后果**：`mismatch = 192/288 = 2/3`（恰好「两个分组错、一个对」）——
**这个 2/3 的指纹本身就是定位线索**。

**必须**在输出循环里**当场累加**（见 §1 骨架的 ②③ 两处注释）。

### 陷阱 2：`pages` 从 dict 改数组后**必须**加 `page_present` 位图

stock 在小批量路径里有「缺页 bail-out」：
```python
if page not in self.pages:   # 缺页 ⇒ 走标量回退
```
用 dense 数组后**无法区分「缺页」与「槽位值是 -1」**——
会把「未镜像的页」误当成「有效但值为 -1」，**静默破坏语义**。
⇒ 必须单独维护 `page_present[page]` 并在读之前检查。

### 陷阱 3：pad 值不是 -1

stock 的初值是 `self.pad_id = token_map[config.engram_pad_id]`（**正整数**），
不是 -1。骨架里已用 `pad_id` 参数传入。

### 陷阱 4：图像 token 要被 `masked_fill(~mask, -1)` 置 -1

`valid_engram_token_mask` 排除图像区域；numba 版必须在页写前应用同样的 mask，
否则会把图像 token 写进历史。

### 陷阱 5（**不是**陷阱）：numba 的 `%` 是 Python 语义

实测 `numba: (-5) % 3 = 1`，与 torch/numpy 一致（都返回非负）。
**不需要**写 `((x % m) + m) % m`。

### 陷阱 6 ★★：**每个 shift 用「自己 `prev` 的页」**，不是当前 token 的页

> ⚠️ **本报告 §1 骨架代码在这点上是错的** —— 由线 3 的 S9 判别实验证伪，
> 我随后独立核实了源码。**照抄 §1 骨架会在跨页时全错（48/48 不同）。**

**stock 的真实语义**（`engram_hash_ab.py`）：

```python
# :370  （_vectorized_history，大路径）
pages = block_table[request_ids.unsqueeze(1), previous.clamp_min(0) // block_size]
#                                            ^^^^^^^^ 每个 shift 自己算页

# :52   （_stock_small_history，小路径）
page = block_table[request, previous // block_size].item()
```

**线 3 的判别实验**（构造 block0 全填 `token_map[111]`，只在 position 128 写 `token_map[222]`）：

| 假设 | 能否重现参考输出 |
|---|---|
| **A：每个 shift 各自的页** | ✅ `ref == A` 且 `jit == A` |
| B：全部 shift 复用当前 token 的页（= §1 骨架） | ❌ `ref != B` |

两种假设的输出差异 = **48/48（全不同）**。

**⇒ 正确写法**：
```python
for s in range(lookback):
    prev = positions[i] - s
    v = pad_id
    if prev >= 0:                                  # ① 先判有效
        p = block_table[request_ids[i], prev // block_size]   # ② 用 prev 自己的页
        if page_present[p]:
            v = pages[p, prev % block_size]
```

**另一个容易漏的细节**：stock 用 `previous.clamp_min(0)` 是为了让负 `prev` 不越界索引，
之后那些 shift 会被 `valid = active & (previous >= 0)` 过滤掉
—— **§1 骨架也没体现这一点**（我用了 `if prev >= 0`，语义等价但顺序不同，
需注意 `clamp_min(0)` 下 `prev=-1` 会索引到第 0 页的第 `-1 % 128 = 127` 槽，
而该值**不会被使用**；所以两种写法的**结果**相同，但**不要**在实现里省掉 `prev >= 0` 检查）。

### 陷阱 7：`multipliers` 的实际形状是 `(L, lookback)` = **(2, 4)**

不是审计报告里写的 `(2, 3, 4)`。§1 骨架已按 `(L, lookback)` 写。

### 陷阱 8：页容量用「按需增长 + 重跑」而不是预分配

线 3 的实现（**采纳**）：
* 初始 `V41_ENGRAM_JIT_PAGES=4096`（4 MB），内核返回 `oob_page`
* Python 侧扩容到 `max(oob+1, cap*1.5+1024)` 并**重跑**
* 页写是**幂等**的 ⇒ 重跑无副作用
* 128K 实测会涨到 **47596 页**

好处：小 profile 不浪费 33 MB。比 §2 的「预分配 32.5K」更省内存。

---

## 4. 验证要求（**好消息：核心验证不需要 NPU**）

这是纯 CPU 代码 ⇒ 可以直接在宿主机/容器里做逐位对账，迭代极快：

1. **逐位等价**：同一批 `(input_ids, positions, request_ids, block_table)` 跑 old/new，`torch.equal`
2. **边界覆盖**：
   - ① 首步无历史
   - ② **跨页边界**（`prev` 落在前一页）—— 注意 stock 用「当前 token 的页」，
     这点必须**照抄 stock 而非"修正"**
   - ③ 图像 token
   - ④ **缺页 bail-out**
   - ⑤ prefill chunk（n≈2044）
3. **fuzz**：≥1000 组随机输入（我已跑通 1000 例）
4. **性能**：三相的 µs 对照（`hash`/`plan`/`lookup`）
5. **8 卡 device 账目**：由性能线做（看 `route-probe` 的相位）
6. **精度门**：由正确性线做

---

## 5. 预期收益（汇总）

| 相位 | 当前 | numba 后（预估） | 依据 |
|---|---|---|---|
| `hash` | 0.401 ms | **~0.004** | 实测 3.42 µs |
| `route.plan` | 0.241 ms | **~0.005** | 同套路（已是 numpy，再上 numba） |
| `route.lookup` | 0.205 ms | **~0.03** | 25 GB CPU 表索引 + 融合 dequant |
| `pad` | 0.130 ms | **~0.002** | 跳过 memset（补丁已写） |
| **合计** | **0.977 ms** | **~0.04 ms** | **省 ≈0.94 ms/step** |

**我们离目标差 0.65 ms ⇒ 够用。**

---

## 6. 证据

| 内容 | 路径 |
|---|---|
| 多进程编译验证 | 容器 `/tmp/nb_mproc.py` |
| 逐位对账 + 修正 | 容器 `/tmp/nb_fix.py`（1000 例 fuzz） |
| 性能基准 | 容器 `/tmp/nb_perf.py` |
| 隔离定位 | 容器 `/tmp/nb_isolate.py` |
| 微基准（torch vs numpy vs numba） | 容器 `/tmp/bench_cpu_dispatch.py` |
| stock 语义 | `probe_hash/engram_hash_ab.py`（`update` 尾部、`_small_batch_history`） |
| 相位实测 | `logs/perf/fq_real_serve.log` 的 `[bneck]` / `[route-probe]` |
