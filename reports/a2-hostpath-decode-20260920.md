# A2（910B3）Engram **host 路径** 单流 decode 实测（2026-09-20）

> 口径：`ENGRAM_DEVICE_INDEX=0`（关闭 Engram 算子入图，走 host 路径）。
> 数据源：用户提供的 A2 真机 `[bneck]` / `[route-probe]` 日志（`steps=3320…3520`，
> 每 20 步一次滚动窗口，`n=6 padded=6 max=2048 lookup=(2048, 6144)`）。

---

## 0. 先说 `[bneck]` 字段口径（源码级，避免误读）

`patches/files/model.py`：

```
mark_step()   在 prepare_engram_inputs() 开头调用，累计 (now - 上一次 mark_step)
              ⇒ acc["hp"] = **相邻两次 engram prep 起点的间隔** = 完整 decode step 时间
stat("total") 包住 prepare_engram_inputs 本体 ⇒ **engram host 路径耗时**
```

⇒ 两个关键量：

| 字段 | 含义 |
|---|---|
| **`hp`** | **decode step 时间（ms/step）** —— 用它算 tok/s，不要拿 `total` |
| `total` | 该 rank 的 engram host 路径耗时（`d2h + hash + route + pad + meta` 量级） |

**自洽校验**：`hp ≈ 64.8 ms/step` 与 APIServer 的 `A=3.44 / 54.7 tok/s`
⇒ `1000 × 3.44 / 54.7 = 62.9 ms/step` —— 与 `hp` 一致（±3%）。✅

---

## 1. 结论：A2 host 路径单流 decode ≈ **58–65 ms/step（≈55 tok/s）**

稳态（`steps=3460…3520`，四个窗口）8 rank 的 `hp`：

| rank | 3460 | 3480 | 3500 | 3520 |
|---|---:|---:|---:|---:|
| TP0 | 64.92 | 64.61 | 65.28 | 64.80 |
| TP1 | 64.82 | 64.78 | 65.07 | 64.79 |
| TP2 | 64.90 | 64.62 | 65.24 | 64.89 |
| TP3 | 64.96 | 64.61 | 65.27 | 64.78 |
| TP4 | 64.95 | 64.65 | 65.19 | 64.77 |
| TP5 | 64.98 | 64.64 | 65.25 | 64.77 |
| TP6 | 64.94 | 64.59 | 65.32 | 64.76 |
| TP7 | 64.91 | 64.70 | 65.20 | 64.78 |

⇒ **8 rank 间离散 < 0.8 ms（1.2%）**，非常整齐。
⇒ 单流吞吐 `A=3.44 / 64.8 ms` ≈ **53 tok/s**（与 APIServer 的 54.7 tok/s 一致）。

**⚠️ 起服后有时间漂移**（同一进程内）：

```
steps=3320  hp ≈ 58.0–58.8       ← 早期
steps=3340  hp ≈ 62.4
steps=3360  hp ≈ 63.9
steps=3380–3440  hp ≈ 61.7–62.9
steps=3460–3520  hp ≈ 64.6–65.3  ← 稳态，比早期高 ~11%
```

这条**必须记**：拿早期窗口比后期窗口会得出错误的"优化收益"。（成因未查：可能是
热、也可能与 KV/页增长有关；`GPU KV cache usage` 只有 0.5%，不像 KV 压力。）

---

## 2. Engram host 路径占多少：**中位 6.0 ms/step（9.3%），最慢 rank 8.9 ms（13.7%）**

稳态四窗口的 `total`（ms/step，中位）：

| rank | total 中位 | 占 `hp` | 主要构成 |
|---|---:|---:|---|
| **TP7** | **8.9** | **13.7%** | `d2h 5.4`（等设备排空） |
| **TP6** | **8.5** | **13.1%** | `d2h 4.8` |
| TP2 | 6.8 | 10.5% | `d2h 2.1–3.4` |
| TP3 | 6.0 | 9.3% | `d2h 2.2` |
| TP4 | 5.9 | 9.1% | `d2h 1.9–2.5` |
| **TP0** | **6.0** | **9.3%** | **`route 2.9`**（全 rank 最高） |
| TP5 | 5.0 | 7.7% | `d2h 1.2–2.4` |
| **TP1** | **3.5** | **5.4%** | `d2h 0.2–0.5`（最轻） |

⇒ **8 rank 的 `total` 极不均衡：3.5 → 8.9 ms（2.5×）**，主要是 `d2h` 的 rank 间差异。

### 2.1 `d2h` 不是工作量，是"等设备排空"

`d2h` 在 8 个 rank 上从 **0.21** 到 **5.42 ms** 都有，而 `hash`/`route`/`pad` 都很稳。
按项目既有结论（`reports/engram-final-quantification.md`）：`d2h` 是**阻塞等 NPU 流水线
排空**，不是真实 CPU 工作 ⇒ 它是**rank 间 skew 的显影**，不是可优化的"计算"。

### 2.2 `route` 才是真实工作量，且 **TP0 是异常点**

`route`（CPU 查表 + staging + 集合通信下发）：

| rank | route（稳态） |
|---|---:|
| TP1–TP7 | **1.52 – 1.72**（很整齐） |
| **TP0** | **2.82 – 2.96**（比其它 rank 高 **~75%**） |

`route-probe` 的内部相位能解释一半：

```
TP1–TP7: a2a≈0.33  bcast≈0.19  evt≈0.16  h2d≈0.11  lookup≈0.47  plan≈0.12  scatter≈0.06  ⇒ 合计 ≈1.44
TP0    : a2a≈0.37  bcast≈0.24  evt≈0.16  h2d≈0.11  lookup≈0.46  plan≈0.13  scatter≈0.8–1.4  ⇒ 合计 ≈2.2–2.8
```

⇒ **TP0 的 `scatter` 是 0.8–1.4 ms，其它 rank 只有 0.055–0.09 ms（10–20×）**。
这是 **rank0 特有的额外工作**（metadata gather / owner 路由的 rank0 侧代价）。

> 注：**同样的 TP0 `scatter` 异常在 A3 日志里也存在**（早先 A3 数据里 TP0 `scatter=0.892/0.995`，
> 其它 rank 0.038–0.050）⇒ **不是 A2 特有**，是 rank0 的角色代价。见
> `results/e2e_fix_H_final/serve.log` 的 `[route-probe]` 行。

### 2.3 `hash` 在 A2 上很便宜

`hash ≈ 0.12–0.15 ms/step`，**8 rank 完全一致**（`LOOKUP` 侧已向量化 + numba JIT）。
⇒ 用户此前担心"A2 CPU 性能低"，**在这条路径上不成立**（hash 不是瓶颈）。

---

## 3. ★ 最重要的观察：**A2 的慢不是 Engram 造成的**

把 A2 与 A3 的 host 路径放在一起（两机都走 host 路径）：

| 量 | A3（910C，2.0 TiB） | **A2（910B3，754 GiB）** | 比 |
|---|---:|---:|---:|
| **`hp`（ms/step）** | **34.61** | **64.8** | **A2 慢 1.87×** |
| engram `total` | 8.22 | **6.0（中位）** | A2 **反而小 27%** |
| `route` | 1.00 | 1.6 | A2 大 60% |
| `hash` | 0.42 | 0.135 | A2 **小 3×** |
| `pad` | 0.128 | 0.215 | A2 大 68% |
| `d2h` | 5.92 | 0.2–5.4（rank 差异极大） | — |

> A3 的基线出自 `reports/engram-final-quantification.md`
> （`[bneck] mode=stock steps=340 dec=20 d2h=5.92 hash=0.42 hp=34.61 meta=0.002 pad=0.128 route=1.00 total=8.22`）。
> ⚠️ **两次运行的 `n` 未记录、配置未必完全一致**，所以这是**量级对比**，不是严格 A/B。

**推论**（标【推断】）：即便把 Engram host 路径**完全消掉**，
A2 也只能从 64.8 → **~58.8 ms/step**（−9.3%）。
要接近 A3 的 34.6 ms/step，缺口在**engram 之外**（target forward + draft + 其它 host 工作）。

⇒ **对 A2 而言，`ENGRAM_DEVICE_INDEX=0 → 开启` 的天花板是 ~9–14%，不是 2×。**
这条直接决定"A2 上是否值得改分片注册架构"的性价比判断。

---

## 4. 与 A2 device-index 失败的对照（为什么这条数据重要）

| A2 配置 | decode 单流 | 状态 |
|---|---:|---|
| `ENGRAM_DEVICE_INDEX=0`（本报告） | **~65 ms/step（53–55 tok/s）** | 可用 |
| `ENGRAM_DEVICE_INDEX=auto`（整表注册） | — | ❌ **不可用**（`ret=207001`，见 `CHANGELOG.md` v8 §3） |
| 分片注册（**未实现，评估中**） | 预估 **+1.5–2.3 ms** 的跨 rank 通信税 | 上限收益 = 6.0 − 2.0 ≈ **4 ms/step（6%）** |

⇒ **分片注册在 A2 上的净收益上限只有 ~6%，而不是 6 ms 的"全额回收"** ——
因为分片方案必须把 `route` 那条集合通信（a2a+bcast ≈ 0.5 ms）**加回来**，
只能省掉 `d2h`（等待，不是工作）与部分 `lookup`/`scatter`。

**建议**：把 A2 的分片注册从"值得做"降级为"**收益 ≤6%、需先做小规模验证**"。

---

## 5. 复现命令

```bash
# A2，host 路径（本次口径）
MODEL=/home/user/models/out/v41-w4a8-flat IMAGE=dsv41-a2:v8 \
  ENGRAM_DEVICE_INDEX=0 GPU_UTIL=0.91 PROFILE=1 MODE=full \
  bash scripts/run_test.sh

# 看探针输出（每 20 步一行，hp 就是 ms/step）
docker logs -f dsv41-a2 2>&1 | grep -E "\[bneck\]|\[route-probe\]"
```

**读法**：
* `hp=` → **ms/step**（除以 A 再 ×1000 得 tok/s 的单流换算）
* `total=` → 该 rank 的 engram host 路径耗时
* `route-probe` 的 `scatter=` → **TP0 异常大是预期的**（rank0 角色代价，A3 同样）

---

## 6. 未验证 / 待办

1. **`hp` 的时间漂移（58→65 ms）成因未查** —— 建议下次起服后连续采 10 分钟，
   确认是单调爬升还是稳定在 65。
2. **A2/A3 的 `n` 与配置未对齐** ⇒ 第 3 节的量级对比**不能**当严格 A/B；
   要做严格对比需在 A2 上用 A3 同一配置重跑一次。
3. **分片注册的净收益上限（~6%）是推算**，未实测；建议先在单卡上量
   `route` 拆分后的真实代价，再决定是否改架构。
