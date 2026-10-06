# 融合优化执行状态：使能完成、F3 重估、目标可行性（2026-10-07）

> 承接 `FUSION-CANDIDATES-20261007.md` 与 `F3-ROOTCAUSE-AND-CODE-FIDELITY-20261007.md`。
> 本文记录**执行第一天的实际进展**与两处必须更正的估算。全部为【实测】/【推断】。

---

## 0. 本轮做完的事

| # | 事 | 结果 |
|---:|---|---|
| 1 | **使能主层代码迭代**（F3/F7 的前置） | ✅ 已入库 + 已同步到 a3-21（commit `e31e6e1`） |
| 2 | **采到纯 conc=1 的新 profile** | ⏳ raw 6.2 GB 已落盘；导出进行中（`operator_details.csv` 188 MB 已出，`kernel_details.csv` 待出） |
| 3 | **更正 F3 的可回收量估算** | ⚠️ 从 1.127 ms 下修到 **0.3~0.5 ms**（见 §2） |
| 4 | **更正常用 profile 的口径** | ⚠️ `armF_r6_base` 是 **conc=1/4/8 混合**窗口（见 `F3-ROOTCAUSE` §5） |
| 5 | 定位 MoE 路由 cast 的位置 | ✅ 在 gate MatMul 与 `MoeGatingTopKHash` 之间，其 `[48]` 输出**直接喂给 `MoeGatingTopKHash` 的第 2 个输入**；但**具体代码行仍未定位**（见 §3） |

---

## 1. 使能：`V41_MAINPY_MOUNT=1`（关键）

### 1.1 为什么必须先做这件事

主层（40 层）的 attention prolog **不在 `dsa_v1.py` 里**，而在**镜像自带**的两个文件：

| 文件 | 行数 | 是否被 tp8k5 挂载 |
|---|---:|---|
| `attention/dsa_v1.py` | 2452 | ✅ 来自 `patches/files/draft/dsa_v1.py` |
| **`attention/dsa_v41.py`** | **1092** | ❌ **镜像自带** |
| **`models/deepseek_v41/indexer.py`** | **244** | ❌ **镜像自带** |
| `ops/fused_moe/router/fused_topk_router.py` | 224 | ❌ 镜像自带 |

⇒ **不改挂载，F3/F7 根本无法迭代**（只能重建镜像）。

### 1.2 做法

* 从 tp8k5 容器**逐字节抽出**两份基线并入库
  （`patches/files/mainpy/dsa_v41.py` md5 `6e60fc4d…`、`indexer.py` md5 `eabbce97…`）；
* `serve_a2.sh` 新增 **`V41_MAINPY_MOUNT=1`**（默认 `0`，**不改变交付行为**）：
  开启后把这两份挂到真实路径；**只开开关不改文件 ⇒ 行为与镜像逐字节一致**，可作 A/B 的 A 臂；
* 与 CED decode 路径互斥（两者都挂 `dsa_v41.py`，会 duplicate mount）——已在脚本里 `die` 拦住。

---

## 2. ★ 更正：F3 的可回收量被高估了

### 2.1 错在哪

`FUSION-CANDIDATES` 的"4:1 融合可省"用的是：

```
省 = (n − n/4) × 该族中位单算子时长
```

对 F3 代入 n=370、中位 10.52 µs ⇒ 2.92 ms(profile) = **1.794 ms(真实)**，
与"F3 真实暴露 1.127 ms"量级一致，于是取 ~1.1 ms。

### 2.2 但 F3 的 370 个算子**不是同一条链上的等长算子**

按形状拆开（`q_lora_rank=1280`、`head_dim=512`、`hidden=5120`）：

| 算子 | 形状 | 次/步 | 中位 |
|---|---|---:|---:|
| `RmsNorm` | `[48,5120]` | 32.1 | 18.66 µs |
| `RmsNorm` | `[48,512]`（kv_norm） | 35.9 | 13.43 |
| `RmsNorm` | `[48,1280]`（**q_norm**） | 30.6 | 13.57 |
| `DynamicQuantV2` | `[48,5120]` | 63.4 | 6.21 |
| `DynamicQuantV2` | `[48,1280]` | 36.7 | 3.81 |
| `RmsNormCast` | `[48,5120]` | 30.6 | 10.48 |
| `DequantSwigluQuant` | — | 43.0 | 11.98 |

**只有 `q_norm [48,1280]` + `DynamicQuant [48,1280]` 是真正成对的**（≈30 对/步）。
`kv_norm [48,512]` **没有量化伙伴**（后面直接进 RoPE/cache），
`RmsNormCast`/`DequantSwigluQuant` **本身就是官方融合算子**，不能再合。

### 2.3 修正后的账

| 项 | 值 |
|---|---:|
| 真正可融合的对 | **≈30 对/步**（q_norm + DynamicQuant） |
| 每对省 | 约 **1 次 launch**（保守 4~8 µs，因为数据流仍要跑一遍） |
| 合计 | **0.12~0.24 ms(profile) = 0.07~0.15 ms(真实)** |

⇒ 即使全做，**F3 只值 0.07~0.15 ms（0.3~0.6%）**，不是 1.1 ms。

> **方法论教训**：族级"4:1 融合"估算**必须逐形状核实"谁和谁真能合"**。
> 把三个不同形状、不同伙伴关系的算子放进一个族，会**系统性高估**。

---

## 3. MoE 路由的 `[48] INT32→INT64`：位置已知、代码行未知

### 3.1 profile 证据（邻居法）

```
−0.028  RmsNormCast  "48,5120;5120"
−0.018  MatMulV3     "48,5120;384,5120"      ← gate 权重投影（384 专家）
−0.017  DynamicQuant "48,5120"  (s106)
±0.000  Cast "48" INT32→INT64   (s109)      ← ★ 目标，46.6 次/步
+0.001  QuantMatmul "48,5120;18,320,16,32;576" (s106)
+0.001  MoeGatingTopKHash "48,384;384;48;;384"   ← 第 2 个输入就是那个 [48]
```

⇒ 该 cast 的产物**直接作为 `MoeGatingTopKHash` 的 ids 输入**，频率 ≈ 每层一次。

### 3.2 但**不是** `fused_topk_router.py:164`

`ARMH-MECHANISM-AUDIT-20261005` 已经查明：那一行被
`if self.tid2eid is not None or self.bias_vl is not None:`（视觉/hash 路由专用）包住，
**纯文本配置下不执行** ⇒ `IDS64_HOIST` 无效是预期行为。

该审计当时就记下了"**我们观测到的 36.5 个/步的 INT32→INT64 出自另一处（未定位）**"。
**本轮仍未定位到具体代码行**（`operator_details.csv` 的 `Call Stack` 列为空，无法用调用栈追）。

**价值**：46.6 × 1.28 µs = 0.059 ms(profile) = **0.036 ms(真实) = 0.15%**。小，但是零风险（若找到）。

---

## 4. ★ 目标可行性：融合program 的现实产出可能**够不到 12%**

### 4.1 逐族重估（用"真正可合的对"而不是族总数）

| 族 | 族暴露(真实) | **现实可回收** | 依据 |
|---|---:|---:|---|
| F3 norm+quant | 1.127 | **0.07~0.15** | §2：只有 q_norm 成对（30 对），且只省一次 launch |
| F2 dtype/搬运 | 0.888 | 0.3~0.6 | 项多但每个 1.4~9 µs；最大单 0.106（`_forward_o_proj`） |
| F7 RoPE/取表 | 0.877 | 0.2~0.5 | 取表链**已做过**（6→2）；剩 RoPE 需与 norm/quant 合 |
| F4 MoE 路由 | 0.384 | 0.15~0.3 | gating+routing 可合（unpermute 不能） |
| F1 位置/槽位 | 0.356 | 0.15~0.3 | 可下沉成 1 kernel，有先例 |
| F5/F6/F8 | 1.076 | 0.2~0.5 | 分散 |
| **合计** | **4.708** | **≈1.1~2.4** | |

⇒ 对应步长 **24.59 → 22.2~23.5 ms**，即 **+4.6% ~ +9.7%**。
**目标 ≤22.0 ms（+12%）很可能够不到。**

### 4.2 缺的那一截在哪：**`HcPre`（单项最大）**

| 项 | 值 |
|---|---:|
| 次数/步 | **86~101** |
| 单次 | **~38~40 µs**（实测 38.13 µs @ `[48,4,5120;…]`） |
| 每步合计 | **≈3.28 ms(profile) = 2.02 ms(真实)** |
| 真实暴露 | **1.987 ms = 8.1% 步长**（**单项最大**） |
| 数据量 | 48×4×5120×2 B ≈ **1.9 MB ⇒ 按带宽只需 ~1.5 µs** |

⇒ **单次 38 µs 里有 ~36 µs 是纯开销**（比带宽所需高 **26×**）。
官方文档已定性：**AIV 同步占 83%**，tiling 把 K 切成 **20 块 × 24 核**。

**这一项就是"融合 program 够不到 12%"的补足来源**（潜在 0.5~1.5 ms）。
但它**不是 Python 层能解决的**——需要改 kernel tiling / 跨核同步，
正是 `kernels/gmm1_armF/` 那条路（当年靠 `SyncAll 4→2` 拿到 ≈3%）。

### 4.3 已排除的 HcPre 替代路线（别重走）

| 尝试 | 结论 |
|---|---|
| 原生 PyTorch 实现 | **慢 35×** |
| 减 Sinkhorn 迭代（20→12） | **不可减**（eps 下限，12 次仍差 8e-2） |
| A1（自适应 K_L0） | 服务内**从未执行**（静态内核缓存未失效）；澄清后效应也低于噪声底 |
| HcPre+RMSNorm 融合 | M=6 实测**净亏 672 µs/步** |

---

## 5. 另一处必须记录的坑：`operator_details.csv` 的计数**不可用于算子清单**

导出中间产物 `operator_details.csv`（188 MB、497.9 万行）的计数**自相矛盾**：

| 算子 | 行数 | 若按"每步 40~86 次"推 |
|---|---:|---|
| `aclnnSparseFlashMla` | **80** | ≈ 2 步 |
| `aclnnHcPre` | 172 | ≈ 2 步 |
| `aclnnInplaceCopy` | **147,407** | ≈ 1,866 步 |

**原因（推断）**：ACL graph 模式下，`aclnn*` 的**host 调用**只在**捕获时**发生一次，
而设备 kernel 在每次 replay 都跑 ⇒ 两者混在同一张表里，比值毫无意义。

⇒ **做算子清单/暴露度分析，只能用 `kernel_details.csv`**（纯设备侧、每次 replay 都记）。
`Call Stack` 列在本导出里**全为空** ⇒ 无法用调用栈定位代码行。

---

## 6. 下一步（按可行性重排）

| 序 | 动作 | 预期 | 风险 | 依赖 |
|---:|---|---:|---|---|
| 1 | **等 `kernel_details.csv` 导出完**，用纯 conc=1 数据重算八族 | — | — | 进行中 |
| 2 | **F2**：`_forward_o_proj` 的 `output[...] = self.wo_b(...)`（消 TensorMove） | 0.106 ms | 低 | `dsa_v1.py`（已挂载） |
| 3 | **F1**：位置/槽位链下沉（有 `_compute_slot_mapping_kernel` 先例） | 0.15~0.3 ms | 低 | `model.py`/`dsa_v1.py` |
| 4 | **F3**：仅 `q_norm[48,1280]`+`DynamicQuant`（需 `dsa_v41.py`+`indexer.py`） | 0.07~0.15 ms | 中 | ✅ 使能已完成 |
| 5 | **HcPre tiling/同步**（补足 12% 的关键） | 0.5~1.5 ms | **高**（kernel 级） | OPP 覆盖机制 + 重编 |

---

## 7. 复现

```bash
# 使能开关（默认关；开启后主层文件由 patches/files/mainpy/ 提供）
grep -n 'V41_MAINPY_MOUNT' scripts/serve_a2.sh
md5sum patches/files/mainpy/dsa_v41.py patches/files/mainpy/indexer.py
# 期望：6e60fc4d40e486e75a3bac5588c3b77f / eabbce97e9c5816e478988a5bbcca68c

# 主层 prolog（未融合）—— 这就是 q_norm + DynamicQuant 的来源
ssh a3-21 'docker exec dsv41-tp8k5 sed -n "345,375p" \
  /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py'

# 形状归属
ssh a3-21 'python3 ~/tmp/shapeagg.py <PROF>/ASCEND_PROFILER_OUTPUT "RmsNorm|DynamicQuant"'
# 邻居法定位 cast
ssh a3-21 'python3 ~/tmp/neigh.py <PROF>/ASCEND_PROFILER_OUTPUT "CastAiCore_Cast" "\"48\""'
```
