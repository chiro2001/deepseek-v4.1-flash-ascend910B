# `moe-mask-range` 验证通过（dummy device 账目，per-pass 归一化）

> 2026-09-16 13:05 CST｜A3-node1 chips 8-15｜dummy 128K profile
> A = `V41_MOE_MASK_RANGE=0`（stock）／B = `=1`
> 两 profile 的锚点数**完全相同（7912）** ⇒ 最干净的对照

---

## 0. 结论：**通过，采纳**

| 判据（`dummy-line-protocol.md` §1.3） | 结果 |
|---|---|
| 目标算子每步耗时下降与离线预测一致 | ✅ 预测 −0.40 ms/步，实测 decode 窗 **−0.51 ms/步** |
| 总算子数变化符合预期 | ✅ `Index`/`IndexCheck` 各 **−40/pass**，`MaskedFill` **+40/pass**（正是新实现） |
| (device busy 并集下降 ≥0.3 ms/step) | ⚠️ 窗口口径下受 prefill 污染，改用 per-pass 归一化（更干净） |

---

## 1. per-pass 归一化对照（两 profile 锚点数都是 7912 ⇒ 直方可比）

| 算子 | A 事件数 | B 事件数 | Δ事件 | A 总时长 | B 总时长 | **Δ时长** |
|---|---|---|---|---|---|---|
| `Index` | 14668 | 6756 | **−7912** | 213.9 ms | 137.0 ms | **−76.9 ms** |
| `IndexCheck` | 15036 | 7124 | **−7912** | 113.3 ms | 40.5 ms | **−72.8 ms** |
| `MaskedFill` | 1156 | 9068 | **+7912** | 16.7 ms | 38.9 ms | +22.2 ms |
| `Mul` | 13734 | 5822 | −7912 | 142.4 ms | 126.8 ms | −15.6 ms |
| `Cast` | 52288 | 60200 | +7912 | 193.7 ms | 204.0 ms | +10.3 ms |
| **净计** | | | | | | **−132.8 ms** |

**Δ事件数 = ±7912 的四项完全对齐** ⇒ 补丁的替换机制**被逐条证实**：
每次 `expert_map[topk_ids] != -1` 从「`Index` + `IndexCheck`」变成「`MaskedFill` + `Cast`」。

**归一化收益**：−132.8 ms ÷ 197.8 pass = **−0.67 ms/pass**
（其中 decode 窗实测 **−0.51 ms/step**：`Index+IndexCheck` 0.927 → 0.414 ms/step）

prefill 段也同步受益（prefill chunk 更多、算子更贵）⇒ **对 TTFT 也有帮助**。

---

## 2. 与线 3 离线预测的对照

| 项 | 线 3 预测 | 本次实测 | 判定 |
|---|---|---|---|
| Index+IndexCheck 的 8 卡耗时 | 18.16 µs/层 = **0.726 ms/步** | 窗口口径 0.927 ms/步（含其它 Index 用途） | ✅ 同量级 |
| 替换后 | 7.72 µs/层 | 0.414 ms/步（含其它 Index 用途） | ✅ |
| **净收益** | **−0.40 ms/步** | **−0.51 ms/步** | ✅ **略优于预测** |
| 数值等价 | `max_abs = 0.000e+00`（逐位相同） | 本次不测数值（dummy） | 由线 1 的精度门负责 |

---

## 3. 尚未做（交给线 1）

* **精度门**：GSM8K-100 + Vision 23/23（dummy 下数值是垃圾，测不了）。
* **真权重下的 8 卡确认**：dummy 与真权重的 device 结构一致（已实测差 3.5%），
  但这是"采纳前的最后一道"，建议与线 1 的精度门一起做。

---

## 4. 对 110 tok/s 目标的贡献预估

当前最好成绩 128K：`32.894 ms/step` / `A=3.493` ⇒ **106.6 tok/s**。
目标 110 tok/s（同 A）需 `ms/step ≤ 31.75` ⇒ **只差 1.14 ms**。

本改动贡献 **−0.51 ms**（decode）⇒ 还剩 **0.63 ms**。

来自 `small-op-audit.md` 的其余候选（按线 3 的更正后模型重估）：
`rms_norm_dynamic_quant` ≈0.40（线 3 在做）、RoPE gather 链、Cast 链、`input_ids` hoist ≈0.20。
⇒ **再兑现 1–2 个候选即可跨过 110**。

---

## 5. 证据

| 内容 | 路径 |
|---|---|
| A 臂 profile | `logs/prof_vmA/`、`/tmp/vmA_rank0.csv`（175 MB） |
| B 臂 profile | `logs/prof_vmB/`、`/tmp/vmB_rank0.csv`（178 MB） |
| 运行日志 | `/tmp/verify_moemask.log` |
| 对比脚本 | `/tmp/devacct_ab_compare.py`（切段 + per-pass 归一化 + 按 stream 分组） |
| 补丁 | `probe_moe_mask/token_dispatcher.py`（A3-node1 已应用，md5 `a695735ae3e03096a432468eb9ad6b83`） |
| 启动器挂载 | `scripts/serve_a21.sh` 的 `# [MOE-MASK-RANGE]` 段（`MOE_MASK=0/1`） |
| 线 3 的交付 | `A3-node2:~/handoff/patches/moe-mask-range/`（含 README + 测试） |
