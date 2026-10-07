# 真实算子 + 真实 shape 的分流实验：怎么做、做了什么、结果如何（2026-10-07）

> 触发：用户指出「之前的数据都不是真实的算子和真实的 shape」。
> 本文给出**正确做法**，并完成第一版执行。环境 a3-21 chip 5。全部【实测】。

---

## 0. 用户的判断是对的：之前全是代理

| 轮次 | 算子 | shape | block |
|---|---|---|---|
| S2/S3 | **合成**（`matmul`+`rms_norm` 交替） | 合成 | 未核对 |
| S4 | 真实**块结构**（AIC 1 算子 / AIV 2 算子） | 近似 | **核对过**（24blk） |
| S5 | 真实**时长比例** | 近似 | 部分 |
| S6/S7 | trace 的**真实统计**（引擎级计时） | — | 真实 |
| **S10（本文）** | **真实算子** | **真实 shape** | **逐个核对** |

---

## 1. 正确做法（四步）

### 第 1 步：从 trace 提取一层的真实规格

以 `HcPre` 为锚（**每层 2 个 HcPre**），取 `anch[430] ~ anch[432]` ⇒ **一层 = 32 个算子 / 690.9 µs**。

产出（`tools/shunt_real_layer_spec.py`）每个算子的：
`core / block / duration / 输入 shape / 输出 shape / dtype`
⇒ `/tmp/layer_spec.json`，可直接用于实例化。

### 第 2 步：用真实算子 + 真实 shape 实例化，并**逐个核对**

核对两个量：**`Block Num` 与 `Duration(us)`**（两者都对上才算复刻成功）。

| 算子 | 目标 blk | 实测 blk | 目标 µs | 实测 µs | 判定 |
|---|---:|---:|---:|---:|---|
| `RmsNorm`(5120) | 48 | **48** | 18.7 | 7.42 | ⚠️ 时长差 2.5× |
| `RmsNorm`(1280) | 48 | **48** | 12.4 | 7.66 | ⚠️ 时长差 1.6× |
| `DynamicQuant`(5120) | 16 | **16** | 5.6 | 5.22 | ✅ |
| `DynamicQuant`(1280) | 4 | **4** | 3.7 | 4.04 | ✅ |
| `MatMulV2`(4096×1024) | 22 | **22** | 28.0 | 22.36 | ✅ |
| `MoeGatingTopK` | 48 | **48** | 15.5 | 10.80 | ✅ |
| `MoeInitRouting` | 48 | **48** | 16.4 | 6.70 | ⚠️ 时长差 2.4× |
| `TensorMove` | 24 | 48 | 5.1 | 6.42 | ⚠️ blk 不符 |
| `Cast` | 1 | 48 | 1.2 | 4.14 | ⚠️ blk 不符 |
| `RoPE` | 48 | — | 10.9 | — | ❌ 实例化失败 |
| `HcPre`(`npu_mhc_pre`) | 24 | — | 38.2 | — | ❌ A3 不支持（561103） |
| `MatMulV3`(fp32) | 24 | — | 19.3 | — | ❌ 实例化失败 |

**⇒ 14 个候选里 5 个精确匹配、4 个 blk 匹配但时长差、2 个 blk 不符、4 个无法实例化。**

### 第 3 步：**隔离时长与 trace 时长的系统性偏差**（重要发现）

| 算子 | 隔离跑 | trace | 比值 |
|---|---:|---:|---:|
| `RmsNorm`(5120) | 7.98 | 18.7 | **0.43×** |
| `MoeInitRouting` | 9.34 | 16.4 | **0.57×** |
| `RmsNorm`(1280) | 8.05 | 12.4 | 0.65× |
| `MoeGating` | 10.76 | 15.5 | 0.69× |
| `MatMulV2` | 22.48 | 28.0 | 0.80× |
| `DynamicQuant`(5120) | 5.26 | 5.6 | 0.94× |
| `DynamicQuant`(1280) | 4.16 | 3.7 | 1.12× |

**⇒ 大多数算子在隔离环境里比 trace 里快 1.2~2.3 倍。**
原因【推断】：真实模型里 1624 个算子排队、指令缓存压力、跨核同步、以及真实的内存布局。

**⇒ 这条本身就说明：任何"隔离测单算子 → 外推全模型"的做法都会系统性偏差。**

### 第 4 步：按真实顺序串成链，测不同分流设计

---

## 2. 结果：真实算子 + 真实 shape 的分流对照

**能实例化的 7 个真实算子**（真实 shape，块数已核对）：
`RmsNorm5120`(V,18.7) `DynQuant5120`(V,5.6) `RmsNorm1280`(V,12.4) `DynQuant1280`(V,3.7)
`MatMulV2`(C,28.0) `MoeGating`(V,15.5) `MoeInitRouting`(M,16.4)

| 臂 | makespan | **vs 单流** |
|---|---:|---:|
| **A 单流基线** | 2.020 ms | 1.000× |
| **B 均匀分半**（奇偶交替到两条流） | 1.563 ms | **1.292×** |
| **C 按引擎分**（AIC+MIX 一流，AIV 另一流） | **1.404 ms** | **1.439×** |
| D 按引擎分 + 逐层屏障 | 1.647 ms | 1.226× |

**三条结论**：

1. **按引擎分最好：1.439×** —— 与用户的直觉一致（把 cube 型和 vector 型分开）；
2. **均匀分半也有 1.292×** —— 即使两条流内容相似，仍有重叠收益；
3. **屏障反而最差（1.226×）** —— 与 S4 的结论相反，
   因为这里的"层"已经足够大（100 µs），**逐层屏障把并行的窗口切碎了**。

---

## 3. 三轮对照（同一问题，三种保真度）

| 保真度 | 结果 | 说明 |
|---|---:|---|
| **7 个算子 / 1 层**（工作量 100 µs） | **0.999×** | ❌ 被图重放的 **~190 µs 固定开销**淹没 |
| **7 个算子 / 20 层**（工作量 2 ms） | **1.439×** | ✅ 本文结果 |
| S4：真实块结构 / 190 块 | 1.410× | 结构对但算子代理 |

**⇒ 方法论教训：微基准的工作量必须显著大于图重放的固定开销（~190 µs），否则测的是开销不是收益。**

---

## 4. 这次实验的局限（必须说明）

| # | 局限 | 影响 |
|---:|---|---|
| 1 | **只用了 7 / 32 个算子** | 缺失的包括**最大的三个 MIX 算子**：`GroupedMatmul` 对（294.6 µs）+ `SparseFlashMla`（55.3）+ `HcPre`×2（77）⇒ **占该层的 62%** |
| 2 | `HcPre` / `MatMulV3` / `RoPE` 无法实例化 | A3 不支持 / 签名不符 |
| 3 | **时长偏差未校正**（隔离 0.43~1.12× trace） | 绝对数值不可外推 |
| 4 | 算子间**无真实数据依赖**（各用独立张量） | 真实层里相邻算子多是依赖链 |
| 5 | MIX 算子只用 `MoeInitRouting`（48blk）代理 | 真实 `HcPre`/`SparseFlashMla` 是 24blk |

**⇒ 1.439× 是"在真实算子、真实 shape 下、对可实例化子集、无依赖约束"的结果，
不是真实模型的预测值。**

---

## 5. 要拿到真实答案，只有两条路

| 路径 | 做法 | 保真度 | 成本 |
|---|---|---|---|
| **A. 继续补全 chip5 复刻** | 解决 HcPre（找替代或自建）、`SparseFlashMla`、`GroupedMatmul` 的实例化；引入真实依赖 | 中 | 2~3 天 |
| **B. 直接在 tiny 真模型上做** | 用真实模型 + 真实算子 + 真实依赖 + 真实 shape，只改 stream 归属 | **最高** | 3~5 天 |

**建议：先做 B 的"最小版本"** —— 在 tiny 上只把**一对**算子（如 `MatMulV2` 与 `RmsNorm`）分到两条流，
用 profiler 看真并发与步长。**这一对的收益符号就能决定整条线要不要继续。**

---

## 6. 复现

```bash
# ① 提取真实规格
ssh a3-21 'python3 ~/tmp/shunt_real_layer_spec.py <PROF_DIR>'          # -> /tmp/layer_spec.json
# ② 逐个核对 block/duration
ssh a3-21 'docker cp ~/tmp/shunt_real_ops.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 shunt_real_ops.py"'
# ③ 真实顺序链 + 隔离对比
ssh a3-21 'docker cp ~/tmp/shunt_real_chain.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 shunt_real_chain.py"'
# ④ 分流对照（务必 LAYERS=20 以上）
ssh a3-21 'docker cp ~/tmp/shunt_real_split.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && LAYERS=20 python3 shunt_real_split.py"'
```

工具：`tools/shunt_real_layer_spec.py`、`tools/shunt_real_ops.py`、
`tools/shunt_real_chain.py`、`tools/shunt_real_split.py`。
