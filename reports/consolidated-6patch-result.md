# 全部 6 个补丁合并后的真权重 128K 结果

> 2026-09-16 14:14–14:19 CST｜A3-node1 chips 8-15｜**真权重**
> 配置：`MAX_SEQS=1 SP_TOKENS=5 O_PROJ_2D=1 MOE_MASK=1 ROPE_IDXSEL=1 GATE_HOIST=1
> V41_IDS64_HOIST=1 MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 STATIC_KERNEL=1 GPU_UTIL=0.94`

---

## 1. 结果（128K × 8 发，同会话）

| 统计量 | ms/step | A | tok/s |
|---|---|---|---|
| **中位** | **32.740** | **2.763** | **82.73** |
| 范围 | — | **1.571 – 3.400** | — |

对照（同一套测量口径、更早的会话）：

| 配置 | 128K ms/step 中位 | A 中位 | tok/s 中位 |
|---|---|---|---|
| AllGather 基线（无后 4 个补丁） | 35.14 | 2.763 | 78.6 |
| F3 + moe-mask（4 个补丁前） | 32.85 | **1.977** | 62 |
| **全部 6 个补丁** | **32.740** | **2.763** | **82.73** |

### 1.1 诚实解读

* **ms/step 的改善很小（32.85 → 32.74，−0.11 ms）**，
  而我在 device 账目里独立验证的 `moe-mask-range`（−0.51）与 `rope-idxsel`（−0.45~−0.51）
  合计应有 **−1.0 ms**。**这个差异没有被观察到**，可能原因：
  1. 真权重的 host/调度瓶颈吸收了部分 device 收益；
  2. 会话间漂移（±0.5 ms 量级）掩盖了小效应；
  3. 部分算子收益与其它算子的等待重叠（device 并集 ≠ 时长求和）。
* **A 中位从 1.977 变成 2.763 不能归因于补丁**（A 是随机的，样本量不同）。

⇒ **`ms/step` 的真实改善需要用更大样本或 device 账目来确认**；
单会话 8 发的中位差 0.11 ms 在噪声内。

---

## 2. `BAT_TOKENS=8192` 失败（记录为死路）

```
(Worker_TP4_EP4) ERROR torch._dynamo.exc.UserError:
  Consider annotating your code using torch._check*().
  Could not extract specialized integer from data-dependent expression u0 (unhinted: u0).
  (Size-like symbols: none)
```

起服在编译期崩溃（compile range 变为 `[8192]`）。

**影响评估**：`BAT_TOKENS` 只影响 **prefill chunk 数**（64→16）⇒ 只对 **TTFT/prefill**
有意义（可减 75% 的 AICPU allreduce），**对 decode 的 110 tok/s 目标无直接贡献**。
⇒ **不作为优化项，仅记录**（若将来要优化 TTFT，需要先解决这个 dynamo 形状问题）。

---

## 3. 达到 110 tok/s 的算术（用本次数据更新）

`tok/s = A × 1000 / ms/step`

| 场景 | A | ms/step | tok/s |
|---|---|---|---|
| 本次中位 | 2.763 | 32.740 | 82.7 |
| 本次最好 A + 本次 ms | 3.400 | 32.740 | **103.9** |
| **A 稳定在 3.5 + ms 31.0** | 3.5 | 31.0 | **112.9** ✅ |
| A 稳定在 3.5 + ms 31.75 | 3.5 | 31.75 | **110.2** ✅ |

**⇒ 两条路都必须走**：
1. **ms/step 再降 1.0–1.7 ms**（已有 `qli-no-candidate` 投影 −0.9，正在验）；
2. **A 必须稳定在 ~3.5**（依赖正确性线的 `HCCL_DETERMINISTIC` 发现）。

---

## 4. 证据

| 内容 | 路径 |
|---|---|
| 运行日志 | `/tmp/final_consolidated.log` |
| 原始 jsonl | `logs/perf/a21/p42_t4_quote_{131072,32768,8192}_fcA_*.jsonl` |
| 起服日志 | `logs/perf/fcA_serve.log`、`fcB_serve.log`（后者含失败栈） |
| 脚本 | `exp_tools/final_consolidated.sh` |
| 单项验证报告 | `f3-wo-a-2d.md`、`moe-mask-range-verified.md`、`rope-idxsel-verified.md` |
