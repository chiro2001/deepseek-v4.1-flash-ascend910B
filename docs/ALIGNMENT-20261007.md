# 行动前对齐（2026-10-07）：指标口径 / 24.6 ms 的分解 / AIC∥AIV 尝试清单

> 用户要求先对齐再行动。本文回答三件事，全部为【实测】。

---

## 1. 指标口径（确认）

| 项 | 定义 |
|---|---|
| **A** | **接受长度**（每步被接受的 draft token 数） |
| **A_out** | `1 + A` = 每路每步产出 token |
| **主指标** | **单流 ms/step**（引擎侧步周期） |
| **tok/s** | 只用**地火口径**报；否则一律用 ms/step |

**用户判断成立**：`A` 随内容与并发变化 ⇒ **tok/s 不可作为跨档比较的主指标**。
唯一能跨档比较的纯结构量是 **ms/step**（以及由它派生的**步长膨胀**）。

---

## 2. ★ 24.6 ms 到底含不含推测解码 —— **含**

### 2.1 决定性证据：同配置 SPEC=0 vs SPEC=1（两次 run，只差 SPEC）

| run | SPEC | DRAFT_GRAPH | `[bneck] hp` | 佐证 |
|---|---|---:|---:|---|
| `armNOSPEC2_1006_231507` | **0** | 0 | **17.729 ms** | `speculative_config=None`、日志中 dspark 出现 **0** 次 |
| `armFINAL_1006_235454` | **1** | 1 | **24.614 / 24.694 ms** | 日志中 dspark 出现 **317** 次 |
| **Δ** | | | **+6.89 ms** | |

两侧**逐项同配置**（同镜像 `local/dsv41-a3-tp8:20261005-1001`、同模型、`MAX_SEQS=32`、
`BAT=8192`、`sptok=5`、`capture_sizes` 相同、`ENGRAM_DEVICE_INDEX=1`）。

### 2.2 机制（读源码）

```python
# patches/files/model.py:1377
def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
    _bp = _BP_STATE.refresh()
    _bp.mark_step()          # ← hp 在这里打点

# patches/files/model_runner_v1.patched.py:3020
prepare_engram = getattr(self.model, "prepare_engram_inputs", None)
if prepare_engram is not None:
    model_inputs.update(prepare_engram(input_ids, positions, num_tokens_padded))
run_model = partial(self.model, **model_inputs)     # ← 然后才 run_model()
```

`prepare_engram_inputs()` 由 runner 在 **`run_model()` 之前**调用，每步一次
（源码注释自证：*"`prepare_engram_inputs()` runs **before** `run_model()`, i.e. outside the model's capture"*）。

⇒ `hp` = **相邻两次"主模型 forward 之前"打点之间的墙钟间隔** = **覆盖整步**：
包含 draft、verify、采样、metadata 提交。

**交叉验证**：`/metrics` 差分（`Δdrafts/并发`，按定义就是完整引擎步周期）在 conc=1 得 **24.793 ms**，
与 `hp` 的 **24.590 ms** 差 **0.8%**（`FOUR-GATES-CLOSURE` §3）。

### 2.3 正确的分解（与用户记忆的对照）

| | SPEC=0（= 主干整步） | + 推测解码 | = 总计 |
|---|---:|---:|---:|
| **用户记忆中的旧数**（`CED-PD-DYNAMIC-SPEC-20260926`） | **24.35** | **+8.33** | 32.68 |
| **当前 TP8 交付（实测）** | **17.73** | **+6.89** | **24.61** |

⇒ **结论三句**：

1. **24.6 ms 已经包含推测解码**，不存在"额外 6 ms"没算进去的问题；
2. 用户记忆里的那个 **"24 ms"其实是旧配置的 SPEC=0 基线**（24.35），不是"主干"；
3. **主干整步从 24.35 → 17.73（−27%）是真实进步**，推测解码的净增量也从 +8.33 → +6.89。

---

## 3. 多流损失（按用户要求，只用 ms/step）

同语料受控实测（地火 × 统一 `continue` 任务，nonce 破缓存，tp8k5）：

| conc | ms/step | A | A_out | **每 token 延迟** | 聚合 |
|---:|---:|---:|---:|---:|---:|
| 1 | **24.860** | 1.66 | 2.66 | **9.35 ms** | 106.9 tok/s |
| 8 | **41.890** | 2.16 | 3.16 | **13.26 ms** | 603.4 tok/s |

| 口径 | 值 |
|---|---:|
| **步长膨胀** | **×1.685** |
| **每 token 延迟（实测，A 变化）** | **+41.8%** |
| **每 token 延迟（A 归一化，纯结构代价）** | **+68.5%** |

⇒ **用户判断成立：多流的性能损失仍然很大。**
（此前报的"损失 13%~20%"是被 `A` 的变化掩盖的结果。）

---

## 4. ★ AIC ∥ AIV：我们做过哪些**专门**尝试

### 4.1 已做（7 项）

| # | 尝试 | 规模 | 结果 | 产物 |
|---:|---|---|---|---|
| **1** | **`limit_core_num` 层内控核（A1）** | tiny 真实权重 | **+0.8%**（最优 kv=8 核，省 0.191 ms/步） | `tools/a1_ab2.py`、`a1_budget.py`；文档 `LIMIT-CORE-TINY-RESULT` |
| **1a** | ├ 官方用法调研（源码级） | `cann-recipes-infer` | 官方仅 **5 处 / 2 组**支路：MLA 的 KV‖Q（12+8 cube）、Compressor‖Indexer Q（16+8） | `OFFICIAL-LIMIT-CORE-WHAT-IS-PARALLEL` |
| **1b** | └ 核预算→时间曲线 + 关键发现 | tiny 微基准 | **限 AIC 代价超线性**（24→12 慢 1.92×）；**cube 密集算子对 AIV 预算完全不敏感（48→2 全一样）**；AIV 密集算子 48→32 无代价 ⇒ **约 16 个 AIV 核是多余的** | `tools/tiny_limit_{verify,key,sweep}.py` |
| **2** | **多流 pingpong 微基准**（同工作量切 k 条独立流，图内） | tiny | **6 流 1.37×**（上界 1.46×，达 **94%**）⇒ **设备有能力重叠 AIC 与 AIV** | `tools/tiny_pingpong_{v3,v4}.py`、`PINGPONG-AIC-AIV-VERDICT` |
| **3** | **两条大流并发（更接近真实）** | tiny | AIC‖AIC：**−20% ~ +8%**（HBM 争抢，并发需求 1308 GB/s > 可达 1182）；**AIC‖AIV：+6.9%**（唯一正组合） | `tools/tiny_limit_concur.py` |
| **4** | **图内多流捕获的可行性** | tiny | 失败→成功：**fork event 必须在"捕获根流"上 record**；图重放 0.989 vs 串行 1.293 ⇒ **1.31×** | `tools/tiny_graph_ms.py` |
| **5** | **MIX 核内部相位**（profile） | 交付 profile | **`MIX_AIC ∩ MIX_AIV = 0.000`** ⇒ 即使同一个混合核，cube 与 vector 相位也是**先后**执行 | `DECODE-AIC-AIV-PIPELINE` §2 |
| **6** | **现有侧流（MULTISTREAM/DSA_OVERLAP）的覆盖面** | 交付 profile | 主流 busy 仅 **51%**，空闲窗口被侧流填掉 **7.89 ms/步（20%）**；其中 **engram `AivKernel` 4.11 ms 已 100% 与 HCCL 并行**（唯一真正的 AIV‖通信先例） | `DECODE-PARALLELISM-WHAT-IS-HIDDEN` |
| **7** | **资源账与上界** | 交付 profile | AIC 忙 **22.05**、AIV 忙 **14.75**、重叠仅 **3.50 ms** ⇒ AIV 全藏进 AIC = **1.39×**；连通信也藏 = **1.63×**；理论上界 **1.82×** | `DECODE-AIC-AIV-PIPELINE` |

### 4.2 ⚠️ 两个必须说清的事实

1. **A1 从未落到代码里**：`patches/files/dsa_v1.py:1891` 至今仍是
   `main_stream.wait_event(e_kv_matmul_done)`（**强制串行**），
   而 `limit_core_num` 在我们 `patches/`、`scripts/`、`experimental/` 里**零使用**
   （只出现在 `tools/` 的微基准中）。⇒ **A1 是一个"测过但没做"的候选。**
2. **A3 上官方反而更保守**：源码注释
   `# ensure wkv matmul does not overlap with wq_a or wq_b` ⇒
   **A3 上 `wkv` 必须等 `wq_a` 完成**（950 上不必）。这是个 A3 专属的额外串行约束。

### 4.3 未做（3 项，按预期收益排序）

| # | 未做项 | 为什么它才是"真正的 AIC‖AIV" | 上界 | 难度 |
|---:|---|---|---:|---|
| **B1** | **跨层软件流水**（层 L 的 AIV ‖ 层 L−1 的 AIC） | 同层内 `RmsNorm→Quant→Matmul` 是真依赖，**只有跨层才有空间**；这是唯一能触碰 11.25 ms 的那个 | **≤11.3 ms（1.39×）** | **高**（要重构 40 层循环） |
| **B2** | **给每个 ubatch 独立 `compute_stream`** | 现有 DBO 骨架只有一条 `compute_stream`（所有 ubatch 共用）⇒ 计算之间物理上不可能并行 | 同 B1 | 高 |
| **B3** | **把更多 AIV 算子挂到已有侧流**（路径 A 扩展） | 只做了 engram 一处；`AivKernel` 已证明机制可行 | ≤4.2 ms（1.10×） | **低** |

### 4.4 一个结构性事实（决定了上面的排序）

主流上 **272 对 AIC↔AIV 是直接依赖交替**，且 **AIC 连续块中位只有 1 个算子**
⇒ **层内可压的空间几乎为零**（这是 A1 只有 +0.8% 的原因）。
**要拿那 11.25 ms，只能在"层与层之间"或"请求与请求之间"找独立工作。**

---

## 5. 待确认的对齐点

| # | 待确认 | 我的建议 |
|---:|---|---|
| 1 | 24.6 ms 是否按"含推测解码的完整步"记账 | **是**（§2 已证）；口径改为 **17.73（无 spec 整步）+ 6.89（spec 增量）= 24.61** |
| 2 | 多流损失的口径 | **用步长膨胀 ×1.685**（跨语料可比）；每 token 延迟 +68.5%（A 归一化） |
| 3 | AIC‖AIV 的下一步 | 先做 **B3（低风险，≤4.2 ms）**；B1/B2 需重构，建议先做归因实验 |

---

## 6. 复现

```bash
# §2 SPEC=0 vs SPEC=1（历史 run，只差 SPEC）
ssh a3-21 'grep -oE "\[bneck\][^\n]*hp=[0-9.]+" \
  ~/cedpd-repo/results/armNOSPEC2_1006_231507/serve.log | tail -2'   # → hp=17.729
ssh a3-21 'grep -oE "\[bneck\][^\n]*hp=[0-9.]+" \
  ~/cedpd-repo/results/armFINAL_1006_235454/serve.log | tail -2'     # → hp=24.614/24.694
ssh a3-21 'grep -c dspark ~/cedpd-repo/results/armNOSPEC2_1006_231507/serve.log'  # → 0

# §3 多流（同语料受控）
ssh a3-21 'python3 ~/tmp/dihuo_homo_ab.py http://127.0.0.1:19210 1,8 25'
```
