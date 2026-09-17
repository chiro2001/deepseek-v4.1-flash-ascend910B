# Engram host 侧：JIT/numpy 重写的量化审计（面向 A2 弱 CPU）

> 2026-09-16 15:25 CST｜实测环境：A3-node1 容器 `dsv41-a21-perf`
> 前置：`reports/engram-host-breakdown.md`（相位拆解）、`reports/cpu-contention-probe`（竞争实验）

---

## 1. 环境：JIT 工具链**全部现成**

| 工具 | 版本 | 备注 |
|---|---|---|
| **numba** | **0.67.0** | `njit(cache=True)` 可用，编译缓存已验证落盘 |
| llvmlite | 0.49.0 | numba 的 LLVM 后端 |
| **numpy** | **1.26.4** | |
| Cython | 3.2.9 | 备选 |
| pybind11 | 3.1.0 | 备选（写 C++ 扩展） |
| gcc / g++ | 有 | `/usr/bin/gcc` |

⇒ **不需要装任何东西，不需要联网。**

---

## 2. 微基准：torch CPU vs numpy vs numba（N=6 小数组）

| 操作 | torch CPU | numpy | **numba `njit`** | torch→numba |
|---|---|---|---|---|
| 单算子 `a*2` | 3.93 µs | 1.28 µs | — | — |
| `a[0].item()` / `int(a[0])` | 2.55 µs | **0.23 µs** | — | 11× |
| `.tolist()` | 0.47 µs | 0.18 µs | — | 2.6× |
| **索引查表** `big[idx]`（6 元素） | **5.03 µs** | **0.28 µs** | — | **18×** |
| **15 算子链** | **61.02 µs** | **16.68 µs** | **0.79 µs** | **77×** |

**结论**：
* 光换 **numpy** 就 3.1×（单算子）到 18×（查表）；
* **numba** 再把 numpy 提升 21×（15 算子链：16.68 → 0.79 µs）；
* 根因：torch CPU 的 dispatcher 对小张量太重（每算子 ~4 µs 固定开销），
  而我们的工作负载是「**几万个固定开销 ≫ 实际计算**」。

### 2.1 numba 原型（真实的 Engram 形状）

一个 `@njit(cache=True)` 函数完成「页写 + 4-gram 历史读 + 归一化哈希」：

```
first call (incl. JIT compile) = 540.3 ms     <- 一次性
steady-state                   = 1.713 us/call
cache dir                      = /tmp/nb_cache, files=1   <- 可持久化
```

对照当前 `hash` 相位 **401 µs** ⇒ **该相位可降两个数量级**。

**编译成本可控**：8 个 worker × 540 ms = 4.3 s（仅首次）；`cache=True` 后重启近乎免费
（需把 `NUMBA_CACHE_DIR` 指到**宿主机挂载的目录**，否则容器重建即丢）。

---

## 3. 逐相位的可回收量

| 相位 | 当前 ms | 现状实现 | 目标实现 | 预估 | 依据 |
|---|---|---|---|---|---|
| **`hash`** | **0.401** | ~20 个 torch CPU 算子 + Python 页写循环 + 3× `.tolist()` | **numba**（页写+历史+哈希一体） | **→ ~0.01** | 原型 1.71 µs |
| **`route.plan`** | **0.241** | 已是 numpy（`argsort`/`bincount`/`cumsum`） | **numba**（同上套路） | **→ ~0.005** | numpy 6 算子 → numba |
| **`route.lookup`** | **0.205** | `torch.index_select`（25.75 GB CPU 表）+ dequantize | **numpy/numba** 索引 + 融合 dequant | **→ ~0.03** | 查表 18× + 去掉 dequant dispatch |
| `route.evt` | 0.105 | `torch.npu.Event()` record | ❌ **不可 JIT**（NPU API） | 0（另想办法） | — |
| ~~`pad`~~ | ~~0.130~~ | ~~`padded.zero_()` 50 MB~~ | **❌ 已撤回（见 §7.5）** | **0** | 实测 memset 只要 17 µs |
| `route.h2d/a2a/scatter/bcast` | 0.40 | DMA / HCCL / 设备 index_copy | ❌ 不是 CPU 工作 | 0 | — |
| **合计（CPU 敏感部分）** | **0.95**（剔除 pad） | | | **~0.04** | **省 ~0.91 ms/step** |

**⇒ 这套改造能回收 ~0.9 ms/step，超过我们差的 0.65 ms。**

---

## 7.5 ⚠️ 撤回：`pad` 那 0.13 ms 是**假的**

我原先假设 `pad` 相位的 0.13 ms 是「50 MB memset」，并据此写了 `patch_pad_skip.py`。

**实测推翻**（`/tmp/pad_bench.py`，NPU 上直接测）：

| 操作 | 实测 |
|---|---|
| full `zero_()` **25.2 MB** | **8.48 µs** |
| slice `[:6].zero_()` **74 KB** | **16.72 µs**（**更慢**！） |

⇒ 50 MB memset ≈ 17 µs，**不是 0.13 ms**。`pad` 的真实构成是 **6 次小设备 kernel 提交**（各 ~16–25 µs）。

⇒ **`pad-skip` 撤回**（A/B 实测：`pad` 0.153 → 0.175–0.193，无改善）。
⇒ 本表合计从 1.08 改为 **0.95 ms**。

**教训**：我当初用「50 MB ÷ 0.13 ms = 387 GB/s，看起来合理」来"验证"自己的假设——**这是循环论证**。
正确做法是先单独测 `zero_()` 本体。参见 `engram-host-breakdown.md` §3.5。

---

## 4. 为什么这对 **A2** 尤其重要

* 这些是**纯 CPU dispatch 开销** ⇒ 弱 CPU 上绝对值更大（竞争实验：CPU 100% 竞争时
  `hash` 0.40 → 2.59 ms，6.5×）；
* A2 上若这 1.08 ms 变成 1.5–2 ms，**重写后同样的绝对值也省下来**；
* 反过来说：**不重写的话，A2 会比本机多花 1–2 ms，直接吃掉性能预算**。

---

## 5. 实现方案（给线 3）

### 5.1 核心：把 host pipeline 收敛成 2–3 个 `@njit` 函数

```
host_engram_step(
    input_ids[n], positions[n], request_ids[n], block_table[n, nb],
    token_map[vocab], pages[MAX_PAGE, BS], page_present[MAX_PAGE],
    multipliers, primes, offsets, image_token_id, image_pad_token_id,
    out_hashes[n, L, 3, 24], out_mask[n]
)
```

**关键改造**：`self.pages` 从 **Python dict** 变成 **预分配 numpy 数组**
`pages[MAX_PAGE, 128]`（int64，初值 -1）+ `page_present[MAX_PAGE]`（bool）。
* 容量：KV 4.16M token / 128 = 32.5K 页 × 128 × 8 B = **33 MB**（可接受）
* 收益：页写从「Python 循环 + dict 查 + torch 标量 setitem × 6」变成 **一次向量化写**

### 5.2 必须保留的语义（逐位对账点）

1. **`_small_batch_history` 的 bail-out**：当前在「镜像里缺页」时回退到标量走法。
   用 dense 数组后无法区分"缺页"与"槽位是 -1" ⇒ **必须用 `page_present` 复刻该判据**，
   否则会把「未镜像的页」错误地当成「有效但值为 -1」，破坏语义。
2. **`masked_fill(~mask, -1)`**：图像区域的 token 必须置 -1（`valid_engram_token_mask`）。
3. **`primes` / `multipliers` / `offsets` 的形状与 dtype**：
   `primes` 实际形状是 `(L=2, lookback-1=3, n_heads=8)`，
   `products = history[:,None] * multipliers` ⇒ `multipliers` 形状 `(2, 3, 4)`，
   `offsets` 形状 `(2, 24)`。**必须逐一对账张量形状**。
4. **整数语义**：全是 int64 的乘/异或/取模 ⇒ 天然逐位可复现；
   但注意 `torch` 的 `%` 与 C 的 `%` 对负数行为不同（本流程里被 `>=0` 过滤过）。

### 5.3 部署

* `NUMBA_CACHE_DIR` 指向**宿主机挂载目录**（否则容器重建丢失缓存）
* `njit(cache=True, parallel=False)` —— **必须单线程**：8 个 rank 已在并行，
  再开 numba 线程会争 CPU
* 首次编译 8×540 ms ≈ 4.3 s，加在起服里（可接受；缓存后更快）

---

## 6. 验证要求

| 项 | 方法 |
|---|---|
| **数值逐位等价** | 同一批 (input_ids, positions, request_ids, block_table) 跑 old/new，`torch.equal` |
| 覆盖边界 | ① 首步（无历史）② 跨页边界 ③ 图像 token ④ 缺页 bail-out ⑤ prefill chunk |
| 性能 | 单卡微基准 + 8 卡 device 账目（`route-probe` 的 `hash`/`plan`/`lookup` 三相） |
| 缓存 | 重启后确认 `NUMBA_CACHE_DIR` 命中（首调 < 50 ms） |
| **精度门** | GSM8K-100 + Vision（由正确性线做） |

---

## 7. 证据

| 内容 | 路径 |
|---|---|
| 微基准脚本 | 容器 `/tmp/bench_cpu_dispatch.py` |
| numba 原型 | 容器 `/tmp/nb_probe.py` |
| 相位拆解 | `reports/engram-host-breakdown.md` |
| CPU 竞争实验 | `exp_tools/cpu_contention_probe.sh`、`/tmp/cpu_contention.log` |
| 被重写的代码 | `probe_hash/engram_hash_ab.py`（`PagedNgramHistory.update`）、`probe_bneck/engram_host_ws_opt.localowner_v2.py`（`_local_owner_plan_numpy` / `lookup_local`） |
