# dummy 性能线协议 v2（基于实测校准）

> 2026-09-16 12:10 CST｜起因：用已知答案（F3）校准 dummy 墙钟测量，发现**墙钟不可用**

---

## 0. 校准结果：dummy 墙钟分不出 0.3–0.8 ms

用 **F3（`wo_a` 退化 batch matmul → 2D）** 做已知答案校准。
真权重下 F3 的收益是 **−0.3 ~ −0.8 ms/step**（A/B/A2 三轮方向一致，`reports/f3-wo-a-2d.md`）。

**dummy 下的 A/B/A2（`V41_DUMMY_WO_A_FIX=1` 让两臂都能起服）：**

| 臂 | 会话 | 32K ms/step（逐发） | 32K **中位** | p50 中位 |
|---|---|---|---|---|
| **A（F3=0）** | ① | 33.139 / 31.141 / 31.047 / 31.808 | 31.474 | **30.276** |
| **B（F3=1）** | ① | 35.603 / 33.152 / 33.625 / 32.943 | 32.241 | 30.944 |
| **B（F3=1）** | ② | 31.373 / 31.539 / 31.475 / 31.237 | — | **30.3** |

**读数**：
1. **同一个配置（B）在两个会话间差了 ~2 ms**（session① 中位 32.24 / p50 30.94；session② p50 30.3）
   ⇒ **会话间方差 ≈1.5–2 ms**，**远大于 F3 的 0.3–0.8 ms**。
2. **预热后的 A(p50 30.28) 与 B(p50 30.3) 完全一致** ⇒ dummy 墙钟**分不出 F3**。
3. 会话①的 B 偏高是因为**前几发落在冷启动尾巴上**（第一发 35.6）。

### 0.1 另一个坑：dummy + F3=0 无法起服

`--load-format dummy` **不调 `weight_loader`** ⇒ `wo_a.weight` 停在声明的 2D 形状
⇒ stock 的 `npu_transpose_batchmatmul` 崩：

```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

⇒ 必须加 `V41_DUMMY_WO_A_FIX=1`（把 2D 权重按 loader 的同一变换重建成 3D）才能跑 A 臂。

---

## 1. 协议 v2：**用 device 侧账目，不用墙钟**

既然用户的目标就是"**只关注 device 侧性能**"，而墙钟混入了 host 抖动与会话漂移，
那就直接测 device：

### 1.1 步骤（每臂 ~7 min，其中起服 ~5 min）

```bash
# 1) 起服（dummy，带 profiler 端点）
RUN_ID=xx MAX_SEQS=1 LOAD_FORMAT=dummy SP_TOKENS=5 O_PROJ_2D=1 \
  MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 PROFILE=1 PROFILE_DIR=$P/logs/prof_dummy \
  EXTRA_ENV="V41_ENGRAM_WITH_DUMMY=1 V41_DUMMY_WO_A_FIX=1" \
  bash scripts/serve_a21.sh

# 2) 等 READY + 预热（关键：必须跑到会话稳定，见 §0 读数 3）
#    预热 = 先跑 2 发 32K（丢掉），再正式采集

# 3) start_profile → 跑 1 发 32K decode → stop_profile

# 4) 从 op_summary 抽 device 账目（**不看墙钟**）：
#    - 每个算子的「次数/步 × 平均 µs」→ 每步 ms
#    - device busy 并集、comm/compute 占比
#    用 scripts/dev_account.py（锚算子自动选），步数用同会话客户端实测
```

### 1.2 为什么这能行

| 指标 | 噪声 | 能否分辨 F3 |
|---|---|---|
| 墙钟 ms/step | **±1.5–2 ms（会话间）** | ❌ |
| **device 算子时长求和** | profiler 开销是**常数**（同会话同配置） | ✅ 应能（F3 直接把 TransposeBatchMatMul 47.26 µs×40 换成更小的 matmul） |

`dev_account.py` 的 `--op auto` 已支持在 `DispatchFFNCombineW4A8` / `GroupedMatmulSwigluQuantV2` 之间自动选锚算子。

### 1.3 判据（dummy 线的"通过"标准）

对每个待验改动：
1. **device busy 并集**下降 ≥0.3 ms/step，**且**
2. 目标算子的每步耗时下降与离线预测一致（量级匹配），**且**
3. 总算子数下降（若改动是"合并/删除算子"类）

→ 三条都满足才转给**线 1（正确性线）**做精度门；否则打回线 3。

---

## 2. 协议的适用范围（写清楚，别误用）

| dummy 线能做 | dummy 线不能做 |
|---|---|
| device 侧整步账目（算子时长/次数/占比） | **墙钟 ms/step 的精细 A/B**（会话漂移 1.5–2 ms） |
| **>1.5 ms** 量级的改动（墙钟可分辨） | **<1 ms** 量级的改动（必须走 device 账目） |
| 图/padding 类问题（capture size） | 接受率 A（dummy 恒为 1.0） |
| Engram host 路径时延 | 精度/正确性 |
| 通信/重叠结构（`hcom_*` 的每步账目） | 端到端 tok/s 的绝对判定 |

---

## 3. 起服成本（已实测，仍有效）

| 配置 | TIME-TO-READY |
|---|---|
| 真权重 + `MAX_SEQS=4`（旧默认） | 509 s |
| 真权重 + `MAX_SEQS=1` | 417 s |
| **dummy + `MAX_SEQS=1`** | **296–306 s** |

---

## 4. 本次校准的附带产出

1. **`V41_DUMMY_WO_A_FIX`**：让 dummy 下 F3=0 臂也能起服（`exp_tools/patch_wo_a_dummy_stock.py`）。
2. **dummy 的时延可信度**：128K dummy 31.749 vs 真权重 32.894 ⇒ **差 3.5%**（结构一致）。
3. **Engram 时延代价 2.105 ms/step**（dummy+Engram 33.854 vs dummy 31.749）。
4. **会话预热必须做**：冷启动尾巴会把第一发抬高 2–4 ms。

---

## 5. 证据

| 内容 | 路径 |
|---|---|
| A/B/A2 原始 jsonl | `logs/perf/a21/p42_t4_quote_{32768,131072}_df3{a,b,b2}_*.jsonl` |
| 运行日志 | `/tmp/dummy_f3_ab.log` |
| 起服日志 | `logs/perf/df3{a,b,b2}_*_serve.log` |
| 补丁 | `exp_tools/patch_wo_a_dummy_stock.py`、`patch_wo_a_dummy_shape.py`、`patch_engram_dummy.py`、`patch_load_format.py`、`patch_loader_mt_guard.py` |
