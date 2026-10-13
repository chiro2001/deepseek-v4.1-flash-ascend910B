# armH 机制审计：`PAD_SKIP` 生效、`IDS64_HOIST` **在本配置下无可优化对象**（2026-10-05）

> 背景：armH = 交付口径 + `IDS64_HOIST=1` + `ENGRAM_PAD_SKIP=1`。
> 端到端只测到 **−0.165 ms/步**（`hp` 配对中位），落在单跑噪声底（±0.16 ms）内，
> 当时判为"弱正、不采纳"。本文用**逐算子族计数**（零噪声判据）审计两者到底有没有生效。
> 数据：两臂都导出了 rank0 profile（每臂 3 个捕获，覆盖 N=1/N=4/N=8）。
> 工具：`tools/compare_op_counts.py`（新增）。标注：【实测】。

## 1. 逐算子族计数（按"每步"归一；步数 = HcPre 数 ÷ 80）

| 算子族 | armF（两开关都 0） | **armH**（都 1） | 判定 |
|---|---:|---:|---|
| **`ZerosLike` 形状 `8192,6144`** | **1.86 / 步** | **0.00 / 步** | ✅ **PAD_SKIP 生效**（整类消失） |
| `ZerosLike` 全部 | 11.2 | 11.2 | 其余 ZerosLike 不受影响（预期） |
| `Fill` | 55.8 | 55.8 | 不变 |
| `Cast` 全部 | 253.1 | 252.4 | 不变 |
| `ViewCopy` | 28.8 | 28.8 | 不变 |
| `Index` / `IndexCheck` | 19.4 / 19.4 | 19.4 / 19.4 | 不变 |

### 1.1 进一步按 (形状, dtype) 拆 `aclnnInplaceCopy_CastAiCore_Cast`

| 形状 | dtype | armF /步 | armH /步 |
|---|---|---:|---:|
| `6,6` | FLOAT→BF16 | 36.58 | 36.54 |
| **`6`** | **INT32→INT64** | **36.58** | **36.54** |
| `6,32` | FLOAT→FP16 / BF16→FLOAT | 7.32 / 7.32 | 7.31 / 7.31 |
| `6,4,5120` | BF16→FLOAT | 6.40 | 6.40 |

⇒ **两臂逐项相同**，`IDS64_HOIST` 没有减少任何 Cast。

## 2. 为什么 `IDS64_HOIST` 无效（代码级原因）

`IDS64_HOIST` 的目标是 `fused_topk_router.py:164` 的
`input_ids = input_ids.to(torch.int64)`（每层一次）。但该行**被一个条件包住**：

```python
if self.scoring_func == "sqrtsoftplus":
    if self.tid2eid is not None or self.bias_vl is not None:   # ← 视觉 / hash 路由专用
        input_ids = input_ids.to(torch.int64)
```

交付实例是**纯文本模型**（`v41-flat-verify3`，`limit-mm-per-prompt` 只给 4 张图但路由不用
`tid2eid`）⇒ `tid2eid` 与 `bias_vl` 都为 `None` ⇒ **这段代码根本不执行** ⇒
hoist 想优化的那个 cast 在本配置下**不存在**。

**我们观测到的 36.5 个/步的 `INT32→INT64`（形状 `6`）出自另一处**（未定位；特征是每层一个、
在 MoE 路由附近）。它与 `IDS64_HOIST` 无关，因此开关无效是**预期行为**，不是 bug。

> 教训（写给后续"接线开关"类工作）：**要把开关和它要消除的算子对上账**。
> 本次如果只测端到端，会得到"−0.165 ms，但可能只是噪声"这种无法判决的结论；
> 而"逐算子族计数"是**零噪声**的，一次就能判定机制是否生效。

## 3. `PAD_SKIP` 的量化收益与安全性

| 项 | 值 |
|---|---|
| 消除的算子 | `ZerosLike "8192,6144"`，**1.86→0 /步**（即 2 次/步，两次捕获的平均） |
| 单次时长（交付口径 profile） | 29–33 µs |
| **折算收益** | **≈ 58 µs/步（≈0.22%）** |
| 安全性 | 只改"清零哪些行"：模型只读 `lookups[layer][:n]`，而 `_n_read = max(values.shape[0], padded_tokens) ≥ n` ⇒ **可读区仍被清零** |
| 正确性 | armH 的 144K 验收 **11/11 通过** |

## 4. 处置

1. **`ENGRAM_PAD_SKIP` 转为交付默认**（`serve_a2.sh` 里 `PAD_SKIP=${PAD_SKIP:-1}`）。
   理由：机制已用零噪声判据证实、收益 0.22%、零精度风险、验收通过。
2. **`IDS64_HOIST` 保持默认 0**，并在启动器注释里写明"**本配置下无对象**"，
   避免后续有人再花一轮去验证它。
3. armH 的"−0.165 ms"解释为：**PAD_SKIP 的 ~0.06 ms + 运行间漂移**，不是两项相加。

## 5. 复现

```bash
# 逐算子族计数（零噪声判据）
python3 tools/compare_op_counts.py armF_r6_base armH_r6_flags --ops ZerosLike,Fill,Cast,ViewCopy
# 更细的 (形状, dtype) 拆分见本文 §1.1 的脚本（cast_shape2.py，同目录）
```
