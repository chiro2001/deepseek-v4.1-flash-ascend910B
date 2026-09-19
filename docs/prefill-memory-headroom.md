# Prefill 速度的真正命门：显存余量（`GPU_UTIL`）

> 2026-09-19 实测（A3，8×910C，TP8+EP8，Engram 开，真实权重，
> `BAT_TOKENS=8192`，关前缀缓存保证每个 prompt 都真跑 prefill）

---

## 0. 一句话

**`--gpu-memory-utilization` 不是"越大越好"。0.94 会让真实 prefill 的
activation 峰值贴住显存上限，模型 `forward` 慢 2.1×、每条新请求还要多等 ~6 秒。
0.92 是实测拐点：prefill 快 6~7×，代价是 KV 池少 8.6%。**

| `GPU_UTIL` | 设备余量 | 8K prefill TTFT | 判定 |
|---:|---:|---:|---|
| 0.94（旧默认） | 6.11 GiB | **8.0 – 8.6 s** | ❌ 断崖 |
| **0.92（现默认）** | **7.36 GiB** | **1.14 s** | ✅ 推荐 |
| 0.88 | 9.80 GiB | 1.28 s | ✅ 等效，但 KV 更少 |

---

## 0.1 发布默认配置下的实测（权威口径）

上面那组数字来自探索阶段的配置（为做单变量对照，关掉了静态 kernel 与前缀缓存）。
**下面是发布默认配置**（`STATIC_KERNEL=1` + `PREFIX=1` + `GPU_UTIL=0.92` + `MOE_AG=1`）
的真实服务实测，两者差异 **< 1%**：

| prompt tokens | chunk 数 | **首 token（发布默认）** | 探索配置 | 差异 |
|---:|---:|---:|---:|---:|
| 8 192 | 1 | **1.136 / 1.161 / 1.173 s** | 1.138 s | +0.2% |
| 32 768 | 4 | **4.218 / 4.221 s** | 4.191 s | +0.7% |
| 131 072 | 16 | **18.165 s** | 18.068 s | +0.5% |

> 这同时说明 **`STATIC_KERNEL` 与前缀缓存对 prefill 没有影响** ——
> 静态 kernel 只服务 aclgraph（decode），前缀缓存按内容命中（本组测试用互不重叠的
> token 区间、服务端命中率 0.0%）。
>
> 该服务器的 KV 实测为 **`GPU KV cache size: 2,823,080 tokens`**（13.06 GiB），
> 起服含静态内核编译共 **695 s**。

---

## 1. 决定性证据：一次真实 prefill 的显存轨迹

为什么 0.94 会出问题？因为 **vLLM 切分显存时依据的 activation 峰值是偏低的**：
它在 startup profiling 阶段量峰值，而**那时 KV cache 还没分配** ——
量的是一个没有 KV cache 竞争、也更轻量的场景。

在引擎里加一个探针（在每次 `num_tokens > 1000` 的 forward 前后各打一行，
并把峰值统计归零），同一次起服的四个时刻：

| 阶段 | alloc | reserved | **设备剩余** |
|---|---:|---:|---:|
| profiling dummy（n=8192，**KV cache 尚未分配**） | 41 072 MiB | 41 406 MiB | 21 090 MiB |
| ── **KV cache 在此分配（+13 GiB）** ── | | | |
| warmup#1（n=4096）forward 前 | 54 375 MiB | 55 228 MiB | 6 029 MiB |
| warmup#1（n=4096）forward 后 | 54 535 MiB | 57 872 MiB | 3 389 MiB |
| 被测请求（n=8064）forward 前 | 54 404 MiB | 57 872 MiB | 3 384 MiB |
| **被测请求（n=8064）forward 后** | 54 719 MiB | **60 774 MiB** | **476 MiB** |

设备总量 `62 748 MiB (61.27 GiB)`。

**读法**：

1. profiling 报的 `peak activation = 3.21 GiB`，是在**没有 KV cache** 时量的。
2. 真实 prefill 一次就把 `reserved` 从 57 872 推到 **60 774 MiB（+2.9 GiB）**，
   只剩 **476 MiB** —— 已经贴到 `gpu_memory_utilization` 配额（61 284 MiB）。
3. 从 warmup 到被测请求，`reserved` 累计增长 **5.5 GiB**（55 228 → 60 774）。

⇒ **真实 prefill 需要 activation/workspace ≈ 5.5–6.0 GiB，而 vLLM 只预留了 3.21 GiB。**
差额全靠"余量"兜；余量不够，分配器必须反复向驱动申请/归还
（`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 下这一步是同步的），
于是 `forward` 变慢、新请求还要等池子长大。

---

## 2. 单变量对照（同机、同模型、同请求形状）

每一步只改一个变量，各自独立起服：

| # | `PREFIX` | `STATIC_KERNEL` | **`GPU_UTIL`** | 8K prefill TTFT |
|---|---:|---:|---:|---:|
| A | 1 | 1 | **0.94** | 8.0 – 8.6 s |
| B | 0 | 0 | **0.88** | 1.34 s |
| C | **1** | 0 | 0.88 | 1.34 s |
| D | 0 | **1** | 0.88 | 1.35 s |
| **E** | 0 | 0 | **0.94** | **3.67 s** |

**B ↔ E 是唯一变量对照（只差 `GPU_UTIL`）：1.34 s vs 3.67 s。**
C 与 D 分别排除了 `PREFIX` 与 `STATIC_KERNEL` 的影响。

另外用**已发布默认配置**（`PREFIX=1` `STATIC_KERNEL=1` `GPU_UTIL=0.94`）
跑同一请求，得到 6.7 s，与 A 一致 —— **发布配置确实落在这个断崖上**。

### 2.1 三档完整曲线（`PREFIX=0` `SK=0`，其余同）

| `GPU_UTIL` | KV cache | 设备余量 | 8K prefill |
|---:|---:|---:|---:|
| 0.88 | 8.12 GiB | 9.80 GiB | 1.277 s |
| **0.92** | **10.57 GiB** | **7.36 GiB** | **1.265 s** |
| 0.94 | 11.79 GiB | 6.11 GiB | 2.652 s |

**0.92 与 0.88 在噪声内等效，0.93 未测但夹在中间，0.94 是断崖。**

---

## 3. 扩展验证：32K / 128K

同一台服务器、同一配置，输入变长（chunked prefill，每 chunk 8192 token）：

| 输入 | chunk 数 | `GPU_UTIL=0.94` | **`GPU_UTIL=0.92`** | 改善 |
|---:|---:|---:|---:|---:|
| 8 192 | 1 | 7.99 / 8.58 s | **1.138 s** | 7.0× |
| 32 768 | 4 | 29.12 s | **4.191 s** | 6.9× |
| 131 072 | 16 | 102.34 s | **18.068 s** | 5.7× |

**per-chunk 成本在 0.92 下恒定（1.03–1.18 s）⇒ prefill 吞吐 ≈ 7.2–7.8 K token/s，
与序列长度无关。**

---

## 4. 顺带排除的其他猜想

同一个"8K prefill 要 8 秒"的现象，此前有过几种候选解释，都已用单变量对照排除：

| 猜想 | 结论 | 证据 |
|---|---|---|
| Engram host 路径太慢 | ❌ 全部只值 **156 ms（13.6%）** | 用运行时开关逐段关闭（见 §4.1） |
| `.cpu()` 同步（`d2h`） | ❌ 只值 **7 ms（0.6%）** | 同上 |
| 前缀缓存 | ❌ 无关 | 只改 `PREFIX`：1.34 vs 1.34 s |
| 静态 kernel | ❌ 无关 | 只改 `STATIC_KERNEL`：1.34 vs 1.35 s |
| CPU 绑核模式 | ❌ 无关 | 所有运行均为 `global_slice` |

### 4.1 Engram 各段的边际代价（同服务器、交替测量）

| 模式 | 关掉了什么 | TTFT | 相对基线 |
|---|---|---:|---:|
| `nohost` | 全部 host 路径 | 1.012 s | −156 ms |
| `nocomm` | route（集合通信 + CPU 查表） | 1.113 s | −55 ms |
| `mirror` | `d2h` 同步 | 1.161 s | −7 ms |
| `stock` | 无（基线） | 1.168 s | — |

**结论**：Engram 的 host 路径整体只占 13.6%，把它优化到零也换不来数量级提升。
真正的杠杆是 `GPU_UTIL`。

---

## 5. 怎么选 `GPU_UTIL`

需要的设备余量 = **真实 prefill activation 峰值（≈6 GiB）+ 碎片缓冲（≥1 GiB）**。

| `GPU_UTIL` | 余量 | 适用 |
|---:|---:|---|
| 0.88 | 9.80 GiB | 想要最大安全边际（代价：KV 最少） |
| **0.92** | **7.36 GiB** | **默认**，实测拐点 |
| 0.93 | ~6.7 GiB | 未测，风险未知 |
| 0.94 | 6.11 GiB | ❌ 断崖，prefill 慢 2.1× |
| ≥0.95 | — | ❌ 已实测 OOM |

### 5.1 KV 容量的取舍

`GPU_UTIL` 从 0.94 降到 0.92，KV 池**约少 8.6%**（3,088,412 → **2,823,080** tokens，前者实测、
后者为发布默认配置下的实测值，见 §0.1）。
换来的是 prefill 快 6~7×。

> 若你的场景**极度**依赖 KV 容量、且几乎不跑长 prompt（上下文主要在 20K 以内），
> 可以显式 `GPU_UTIL=0.94` —— 但要知道代价是首 token 延迟从 1.1 s 涨到 8 s。
> **不要**通过调大 `GPU_UTIL` 补 KV：0.95 已实测 OOM。

### 5.2 与 `MAX_SEQS` 的关系

`BAT_TOKENS=8192` + `MAX_SEQS=64` 的组合在 `GPU_UTIL=0.94` 下实测 OOM
（ACL graph 重放失败）；`MAX_SEQS=32`（默认）已验证稳定。
`MAX_SEQS=64` + `GPU_UTIL=0.92` 尚未复测，脚本会给出提示。

---

## 6. 复现

```bash
# 单变量对照：只改 GPU_UTIL，其余完全相同
GPU_UTIL=0.94 bash scripts/serve_a3.sh     # 8K prefill ≈ 8 s
GPU_UTIL=0.92 bash scripts/serve_a3.sh     # 8K prefill ≈ 1.14 s

# 用本包自带的测试脚本量 prefill（它会精确切到目标 token 数）
GPU_UTIL=0.92 MODEL=/path/to/model bash scripts/run_test.sh
```

**判据**：起服日志里的 `Available KV cache memory` 应约 13 GiB（`SK=1`、`util=0.92`），
且第一个请求之后 TTFT 应稳定在 **~1.1 s / 8K token**。
如果你的第一个请求要 7 s 而后续 1.2 s，说明池子在首次请求时被撑大 ——
那是同一机制的另一面，不是故障。

---

## 7. 附：为什么"第一个请求特别慢"是同一件事

六轮独立起服里，**第一条请求（无论 4096 还是 8192 token）恒定要 7.2–7.5 s**，
第二、三条只要 1.1–1.4 s，且与 `PREFIX` / `STATIC_KERNEL` / `moe_comm` 都无关。

机制：第一条请求要把 `reserved` 从 55.2 GiB **撑到 60.8 GiB（+5.5 GiB）**，
在 `expandable_segments` 下这一步是逐步同步完成的；撑满后后续请求复用同一片池子。

**这不是故障，是预热。** 若要生产环境避免它，用一次足够大的预热请求即可。
