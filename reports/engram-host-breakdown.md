# Engram host 侧开销的完整拆解（面向 A2 弱 CPU）

> 2026-09-16 15:10 CST｜数据源：真权重 `fq_real` 会话（8 rank，decode-only 均值）
> 相关：`engram-final-quantification.md`（旧量化）、`target-requires-A3493.md`（目标算术）

---

## 1. `[bneck]` 的 7 个字段里，只有 3 个是真实 host 工作

真权重实测（`fq_real_serve.log`，8 rank 一致）：

```
[bneck] steps=2240 dec=20
  d2h=10.55  hash=0.41  hp=41.17  meta=0.002  pad=0.13  route=1.02  total=12.81
```

| 字段 | ms | 是不是 host 开销 | 判定依据（代码） |
|---|---|---|---|
| **`d2h`** | **10.55** | 否 —— **是"等设备"** | `input_ids[:n].cpu()` 阻塞等 NPU 流水线排空；同期 host 空闲 |
| **`hp`** | **41.17** | 否 —— **是间隔参考值** | `mark_step()` 的 `now - self.last_prep` = 距上次 engram 调用的间隔 |
| `hash` | **0.41** | **是（纯 CPU）** | `PagedNgramHistory.update()`：token_map 索引 + page 写 + 4-gram 历史 + 哈希 |
| `meta` | 0.002 | 是（可忽略） | metadata 构造 |
| **`pad`** | **0.13** | **是（CPU/内存带宽）** | 见 §3 —— 每步 50 MB memset |
| **`route`** | **1.02** | **是（CPU + 集合通信）** | 7 个相位，见 §2 |
| **`total`** | 12.81 | = d2h + hash + route + pad + meta | |

注意 `hp=41.17` 比客户端实测 `ms/step=32.4` 还大 ⇒ 它不能当步周期用（含打印开销与口径差）。本文只用 `hash/route/pad`。

**⇒ 真实 host CPU 工作 = hash + route + pad = 0.41 + 1.02 + 0.13 = 1.56 ms/step**

---

## 2. `route` 的 7 个相位（`[route-probe]` 实测）

代码顺序（`engram_host_ws_opt.localowner_v2.py:811-844`，每个 engram 层一次，共 2 层）：
`plan` → `lookup` → `h2d` → `evt` → `a2a` → `scatter` → `bcast`

| 相位 | ms/step | 做什么 | **随 CPU 变慢而放大？** |
|---|---|---|---|
| `plan` | **0.254** | numpy 规划 + `[row.tolist() for row in gathered]` 等列表转换 | **是**（纯 Python/numpy） |
| `lookup` | **0.204** | `index_select` CPU 常驻的 25.75 GB/rank int8 表 + dequantize | **是**（dispatch + 内存延迟） |
| `h2d` | 0.067 | `values.to(device)`（6×256 bf16 ≈ 3 KB） | 否（DMA） |
| `evt` | **0.105** | `_record_offload_use(...)` 记账 / Event | **是**（Python + Event 开销） |
| `a2a` | 0.199 | `dist.all_to_all_single` 值回传 | 部分（下发 CPU，等待设备） |
| `scatter` | 0.032 | `result[order.to(device)] = returned` | 否（设备侧 index_copy） |
| `bcast` | 0.102 | `dist.broadcast` TP 组内同步 | 部分 |
| **合计** | **0.963** | | |

**⇒ route 里纯 CPU 部分 = plan + lookup + evt = 0.563 ms；通信/DMA = 0.40 ms**

（注：某些 rank 的 `scatter` 达 0.46–1.00 ms，是设备侧异常项，与 CPU 无关。）

---

## 3. ⚠️ `pad` 相位：**原始归因已撤回**（实测推翻）

### 3.1 我原来的说法（**错的**）

我原先写：「每步 50.3 MB memset，实测 387 GB/s，是纯浪费，去掉可省 0.13 ms」。

### 3.2 实测推翻了它

在 NPU 上直接测（`/tmp/pad_bench.py`）：

| 操作 | 实测 |
|---|---|
| full `zero_()` **25.2 MB**（2048×6144 bf16） | **8.48 µs** |
| slice `[:6].zero_()` **74 KB** | **16.72 µs**（**比 full 更慢！**） |
| slice `[:2048].zero_()` | 16.46 µs |

**⇒ 50 MB 的 memset 只要 ~17 µs，不是 0.153 ms。**

（注：循环内只 `synchronize()` 一次，所以测的是**提交速率**；但这正是流水线里的真实成本——
而且 25 MB 的提交比 74 KB **更快**，说明小 slice 走了更贵的路径。）

### 3.3 `pad=0.153 ms` 的真实构成

`pad` 窗口内共有 **6 次小的设备 kernel 提交**：
1. `padded_mask.zero_()`
2. `padded_mask[:6].copy_(mask)`
3. 每层 ×2：`padded[:6].zero_()` + `padded[:6].copy_(values)`（2 层 = 4 次）

每次 ~16–25 µs ⇒ 约 **0.10–0.15 ms** ⇒ **与实测吻合**。

**⇒ 瓶颈是「小 kernel 的提交开销」，不是带宽。**

### 3.4 结论：**`pad-skip` 补丁撤回**

* 去掉 `padded.zero_()` 的收益 ≈ 8 µs（一次 full-zero 被 slice-zero 替代后反而可能 **更慢**）
* 实测 A/B（tvA vs tvB）证实：`pad` 从 0.153 → 0.175–0.193，**没有改善**
* **不要把它写进任何交付配置**

**真正能省的是减少那 6 次提交**（例如把 mask 与 lookups 的零化合并、
或用一次 `torch.zeros` 分配替代 zero_ + copy）——但收益上限也只有 ~0.1 ms，**优先级低**。

### 3.5 这条错误的教训（值得记录）

我当初的推理链是「**50 MB ÷ 0.13 ms = 387 GB/s，看起来合理**」——**这是循环论证**：
我先假设 pad 就是 memset，再用它反推出带宽来"验证"自己。
**正确的做法是先单独测 zero_() 本体**（30 秒的实验），再谈归因。
这与本项目早先 `profiler-overhead-analysis.md` 里记录的「用外部假设反推设备步数」是同一类错误。

---

## 4. A2 弱 CPU 的定量投影

| 类别 | ms/step | 说明 |
|---|---|---|
| **CPU 敏感** | **1.10** | hash 0.41 + plan 0.25 + lookup 0.20 + evt 0.11 + pad 0.13 |
| **CPU 不敏感** | **0.46** | h2d 0.07 + a2a 0.20 + scatter 0.03 + bcast 0.10 + meta 0.002 |
| `d2h`（等设备，不是开销） | 10.55 | 与 CPU 无关 |
| 合计 `total` | 12.81 | |

### 4.1 投影（假设暴露率不变，历史实测约 64%）

| A2 有效 CPU 相对本机 | CPU 敏感部分 | host 合计 | **暴露到 ms/step 的增量** |
|---|---|---|---|
| 1×（本机） | 1.10 | 1.56 | — |
| **2× 慢** | 2.20 | 2.66 | **+0.70 ms/step** |
| **3× 慢** | 3.30 | 3.76 | **+1.41 ms/step** |
| 4× 慢 | 4.40 | 4.86 | +2.11 ms/step |

### 4.2 对 110 tok/s 目标的影响

目标要求 A=3.493 时 **ms ≤ 31.75**。本机当前 32.4（7 补丁）⇒ 差 0.65 ms。

**A2 上若 +1.0 ms ⇒ 差 1.65 ms** ⇒ A2 的弱 CPU 会直接吃掉我们压出来的性能预算。

---

## 5. 可行动项（按收益/风险排序）

| # | 动作 | 预期收益 | 风险 | 归属 |
|---|---|---|---|---|
| **1** | **去掉 `padded.zero_()`** | 本机 0.13 ms；A2 上按带宽比放大 | 低 | 我（性能线） |
| **2** | `plan` 的 numpy/列表转换优化 | 0.25 ms（A2 上更大） | 中 | 线 3 |
| **3** | `lookup` 的 dequantize 移到设备 / 换实现 | 0.20 ms | 中（数值） | 线 3 |
| **4** | `evt` 的 Event 复用/去记账 | 0.11 ms | 低 | 线 3 |
| 5 | Engram `wkv` 量化 int8 | 0.27 ms（device 侧，与 CPU 无关） | 中（重导 ckpt + 精度门） | 量化团队 |

**⇒ 前 4 项合计约 0.69 ms（本机）/ 更多（A2）—— 这正是弱 CPU 上该重点做的。**

---

## 6. CPU 竞争实验（已完成，A-B-A 结构可复现）

在容器 cpuset（320-639）上加 0 / 64 / 160 / 320 个 `nice 19` 忙循环：

| 臂 | 忙循环 | **ms/step** | hash | plan | lookup | evt | pad | **CPU 敏感合计** | route.scatter |
|---|---|---|---|---|---|---|---|---|---|
| **base** | 0 | **33.27** | 0.401 | 0.241 | 0.205 | 0.104 | 0.116 | **1.067** | 0.776 |
| h64 | 64（20%） | **40.05** | 0.430 | 0.283 | 0.219 | 0.117 | 0.133 | **1.182**（+10.8%） | 0.841 |
| h160 | 160（50%） | **47.57** | 0.562 | 0.356 | 0.270 | 0.146 | 0.172 | **1.506**（+41.1%） | 1.053 |
| h320 | 320（100%） | **81.37** | **2.594** | 0.374 | 0.282 | 0.150 | 0.191 | **3.591**（+237%） | **12.910** |
| **restore** | 0 | **~31.5** | 0.406 | 0.246 | 0.203 | 0.108 | 0.120 | **1.083**（+1.5% vs base ✓） | 0.674 |

### 6.1 两个读数

**① 分类正确，A2 投影可信**：
CPU 敏感相位随竞争单调上涨（1.067 → 1.182 → 1.506 → 3.591），
且 `restore` 臂回到 base（+1.5%）⇒ **实验可复现，§4 的分类与投影成立**。

**② ⚠️ 但 Engram 只占 CPU 损失的一小部分**：
h320 臂里 Engram 相位只涨了 **+2.52 ms**，而 **ms/step 涨了 +48.1 ms**。
⇒ 大部分损失来自 **vLLM/CANN 的其它 host 工作**（不在我们的 Engram 路径里）。

**⇒ 诚实结论**：numba 重写 Engram 能回收 ~1.0 ms（本机）/ 更多（A2），
但**不能完全解决 A2 的弱 CPU 问题** —— 那需要 vLLM 运行时层面的改动（超出当前范围）。

**③ `route.scatter` 在 100% 竞争下从 0.776 暴涨到 12.91 ms（17×）**，
而它是**设备侧**的 `result[order.to(device)] = returned`。
异常幅度远超纯 CPU 解释 ⇒ **需要单独追**（可能是 HCCL 线程或驱动线程被饿死导致设备侧命令延迟）。

---

## 7. 证据

| 内容 | 路径 |
|---|---|
| `[bneck]` / `[route-probe]`（真权重） | `logs/perf/fq_real_serve.log` |
| pad 代码 | `probe_bneck/model.py.probe:845-858` |
| route 代码 | `probe_bneck/engram_host_ws_opt.localowner_v2.py:811-844` |
| hash 代码 | `probe_hash/engram_hash_ab.py:236-300` |
| 旧量化 | `reports/engram-final-quantification.md` |
