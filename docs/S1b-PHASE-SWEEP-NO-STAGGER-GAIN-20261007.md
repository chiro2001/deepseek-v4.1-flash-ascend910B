# Shunt 实现细节 vs "错开/加锁"设想：相位扫描的判定（2026-10-07）

> 触发：讨论分流实现细节时提出的两个问题——
> ① 现在两条流是"直接启动让它们自己走"，还是由锁/barrier 调度？
> ② 如果让两条流**错开相位**（A 流的 cube 相位对上 B 流的 vector 相位），
>    能不能绕开"执行单元数量"的限制？
> 环境：a3-21 chip 5（`dsv41-op-hcfuse`）。全部【实测】。

---

## 0. 先回答：现在的实现是"分段 fork/join"，相位**不受控**

```python
for s in range(nseg):                      # 每 seg 个单元一个同步点
    ef[s].record(root)                     # fork：event 必须 record 在【捕获根流】上
    with torch.npu.stream(s2):
        s2.wait_event(ef[s])               # 侧流等 fork
        for i in range(lo, hi): aiv(i)     # 侧流跑 AIV 段
        ej[s].record(s2)
    for i in range(lo, hi): aic(i)         # 根流跑 AIC 段
    root.wait_event(ej[s])                 # 根流等 join
```

| 问题 | 答案 |
|---|---|
| 是"两条流自由跑"吗？ | **不是**。每 seg 个单元有一次 **fork/join barrier**（实测 seg = 2/4/8/16/32 都试过） |
| 是"由锁控制错开"吗？ | **不是**。event 只保证**依赖顺序**，**不控制相位**。段内两条流的相位由各自算子的**时长**自然形成 |
| 相位是"设计"出来的吗？ | **不是**。从未显式控制过 —— 这正是本轮要测的 |

---

## 1. 直接测"错开"：相位扫描 → **平的**

**做法**：两条流各跑 K=64 个 MIX 算子（`moe_init_routing`，MIX_AIC，48 blocks）。
在 B 链**头部插入 n 个"移相算子"**（`dynamic_quant`，≈7 µs/个），
n = 0~6 覆盖 **约一个 MIX 算子周期**（≈39 µs）。若相位重要，makespan 应出现明显谷底/峰值。

| 移相算子数 | makespan | 相对 n=0 |
|---:|---:|---:|
| **0（对齐）** | 2.498 ms | 1.000× |
| 1 | 2.525 ms | 0.989× |
| 2 | 2.512 ms | 0.994× |
| 3 | 2.505 ms | 0.997× |
| 4 | 2.467 ms | 1.013× |
| 5 | 2.496 ms | 1.001× |
| 6 | 2.494 ms | 1.001× |

**⇒ 完全平坦（±1.3%）。相位错开对 MIX 算子毫无影响。**

**参照**：同样的 128 个 MIX 算子放在**单流串行**上只要 **1.724 ms**
⇒ **两条流并发反而慢 45%（0.69×）**。不是"没赚到"，是"亏了"。

---

## 2. 为什么错开没用：block 分配是"整核、整段"持有的

逐算子 profiler 证据（同一个 `moe_init_routing`，128 个都算上）：

| 臂 | kernel 数 | 中位时长 | 设备时长合计 |
|---|---:|---:|---:|
| **单流** | 128 | **10.38 µs** | **1.328 ms** |
| **两流并发** | 128 | **35.85 µs** | **4.404 ms** |

**同一个 kernel：10.4 µs → 35.9 µs，慢 3.45×。**

⇒ 机制【推断】：**一个 48-block 的 kernel 把 24 个核全部占住，并且是"整个 kernel 生命周期"持有**，
　 不只在它用到 cube（或 vector）的那一段持有。
　 ⇒ 另一个 kernel **没法**趁它"vector 相位时用它的 cube"——因为核根本没被释放。
　 ⇒ 这解释了为什么"错开"无效：**没有可利用的空窗，无论相位如何。**

（对照：`MIX_AIC ∩ MIX_AIV = 0.000` —— 连同一个 MIX 核**内部**的 cube/vector 也是先后执行，
　 说明这个"相位"本来就是顺序的，不构成可填充的空窗。）

---

## 3. 加锁（更细的分段）反而更慢

同一组实验，把段长从 32 缩到 8（同步更频繁）：

| 结构 | makespan | vs 串行 |
|---|---:|---:|
| 无同步（一次 fork 到底） | 1.310 ms | 0.985× |
| **seg=32** | 1.299 ms | 0.994× |
| **seg=16** | 1.416 ms | **0.912×** |
| **seg=8** | 1.471 ms | **0.878×** |

⇒ **barrier 越密越差**。你设想中的"由锁来调度"在本平台上是**负收益**：
　 锁只能**限制**并发，不能**创造**并行。

---

## 4. 完整的重叠效率表（决定"什么值得分流"）

| 组合 | vs 串行 | 结论 |
|---|---:|---|
| pure cube(23blk) ∥ AIV **窄**(4blk) | **1.44×** | ✅ 有效 |
| **AIV(48blk) ∥ AIV(48blk)** | **1.61×** | ✅ 有效 |
| pure cube(23blk) ∥ AIV(48blk) | 1.31× | ⚠️ 部分 |
| pure cube(23blk) ∥ cube(23blk) | 1.18× | ❌ 差 |
| **MIX(48blk) ∥ AIV 窄** | **1.11×** | ⚠️ 微弱 |
| **MIX(48blk) ∥ AIV(48blk)** | **0.985×** | ❌ **无** |
| **MIX(48blk) ∥ MIX(48blk)** | **0.69×** | ❌ **负** |

**规律**：重叠收益由**两个 kernel 的 block 数（占核比例）**决定，**与相位无关**。
两个都"宽"（≈占满 24 核）⇒ 必然互相压缩。

---

## 5. 对 Shunt 的影响（重要修正）

生产主流的构成（`DECODE-AIC-AIV-PIPELINE` / `shunt_probe2`）：

| 算子 | 类型 | **Block Num** |
|---|---|---:|
| `HcPre` | MIX_AIC | **24** |
| `SparseFlashMla` | MIX_AIC | **24** |
| `GroupedMatmulSwigluQuant` / `GroupedMatmul` | MIX_AIC | **24** |
| `QuantBatchMatmulV3` | MIX_AIC | **16~20** |
| `RmsNorm` / `HcPost` | AI_VECTOR | **48** |
| `DynamicQuantV2` | AI_VECTOR | **4~16** |

⇒ **主流上的大项（贡献 AIC 17.2 ms 与 AIV 大部分时长的那些）几乎都是"宽"算子，
　 恰恰落在"重叠无效或为负"的区**。

**修正后的可重叠空间**（用 AIV 的 block 分布算）：

| AIV block 数 | 算子占比 | 时长 | 可重叠性 |
|---|---:|---:|---|
| 1 block | 40.2% | 1.3 ms | ✅ 好（但总量小） |
| 2/4/6 | 5.4% | 0.5 ms | ✅ 好 |
| 8/12/16/24 | 8.6% | 1.1 ms | ⚠️ 中 |
| **48** | **30.0%** | **7.4 ms** | ❌ **基本不可重叠** |

⇒ **可重叠的 AIV 总量 ≈ 2.9 ms**（不是之前说的 8.23 ms），
　 其中真正能兑现的按 1.1~1.4× 折算 ⇒ **约 0.3 ~ 1.0 ms/步**。

---

## 6. 三个结论（对应你的三句话）

| 你的设想 | 实测判定 |
|---|---|
| "两个流是错开的" | ❌ **错开无效**：相位扫描平坦（±1.3%），覆盖整一个算子周期 |
| "由某些锁来调度" | ❌ **反向**：barrier 越密越差（seg=8 时 0.878× vs 无同步 0.985×） |
| "错开就能绕开执行单元数量限制" | ❌ **绕不开**：48-block kernel **整段持有全部 24 核**，没有"相位空窗"可以填 |

**根因一句话**：**限制不是"cube 和 vector 谁在忙"，而是"核被谁整段占着"。**
核一旦被某个 kernel 按 block 占住，它的 cube 和 vector 就都归那个 kernel 调度，
外面的流插不进去 —— 这与相位无关。

---

## 7. 复现

```bash
ssh a3-21 'docker cp ~/tmp/shunt_phase_sweep.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && K=64 python3 shunt_phase_sweep.py"'
ssh a3-21 'docker cp ~/tmp/shunt_mix_verify.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && K=64 python3 shunt_mix_verify.py"'
ssh a3-21 'docker cp ~/tmp/shunt_mix_vs_aiv.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && K=64 python3 shunt_mix_vs_aiv.py"'
```

工具：`tools/shunt_phase_sweep.py`、`tools/shunt_mix_verify.py`、`tools/shunt_mix_vs_aiv.py`、
`tools/shunt_block_sweep.py`（block 数扫描，注意：rms 的 tiling 恒取 48 blocks，宽度改不动 block 数）。

> ⚠️ **代理算子的局限**：本轮 MIX 代理是 `moe_init_routing`（**48 blocks**），
> 而生产 `HcPre`/`SparseFlashMla`/`GroupedMatmul` 是 **24 blocks**、`QuantMatmul` 是 **16~20 blocks**。
> 24 核已被占满（= 48 vector 单元 / 24 cube 单元），所以"宽算子不可重叠"的结论方向应成立，
> **但 24-block 与 16-block 的具体重叠系数需要复测**（待办 S1b）。
