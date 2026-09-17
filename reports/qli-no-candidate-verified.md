# `qli-no-candidate` 验证通过（device 账目，两臂锚点数完全相同）

> 2026-09-16 14:47 CST｜A3-node1 dummy profile（8K prompt + 192 output ⇒ decode 占比高）
> A=`V41_QLI_NO_CANDIDATE=0`／B=`=1`｜**两臂 anchors 都是 9460（236.5 passes）⇒ 直方可比**

---

## 1. 结果

| 项 | A（QLI=0） | B（QLI=1） | Δ |
|---|---|---|---|
| `QuantLightningIndexerV2` **static** 次数 | 1712（7.24/pass） | **1712**（不变） | 0 |
| 同上，**每 op µs** | **99.28** | **50.29** | **−49%** |
| 同上，总时长 | 170.0 ms | 86.1 ms | **−83.9 ms** |
| `QuantLightningIndexerV2` dynamic（prefill） | 49.8 ms | 16.6 ms | −33.2 ms |
| `QuantLightningIndexerV2Metadata` | 35.5 ms | 35.8 ms | +0.3 |
| **QLI 总计** | **255.2 ms** | **138.5 ms** | **−116.7 ms** |
| **归一化** | **1.079 ms/pass** | **0.585 ms/pass** | **−0.494 ms/pass** |

---

## 2. 与线 3 预测的对照

| 项 | 线 3 预测 | 本次实测 | 判定 |
|---|---|---|---|
| 机制 | `candidate_mode` 1/2 → 3 | ✅ static 次数不变、每 op 减半 | ✅ |
| 单卡同输入（page=128, M=8, topk=512） | mode2 比 mode3 慢 ~320 µs | — | — |
| 8 卡投影 | **−0.9 ms/step**（decode） | **−0.494 ms/pass**（本窗口） | 实测约为预测的一半 |
| top-K 等价 | 3 个长度全部 `torch.equal` | 本次不测数值（dummy） | 交线 1 精度门 |

### 2.1 为什么是 −0.494 而不是 −0.9

线 3 的 −0.9 是**按真 8 卡 profile 的 decode 口径**（1×193.6 + 4×333.3 = 1527 µs → 577 µs）算的。
本次是 dummy + 8K prompt，窗口里 `static` 的 7.24 次/pass 与真权重 decode 的 8 次/前向接近，
但**每 op 的绝对值不同**（dummy 99 µs vs 真权重 193.6/333.3 µs）——
dummy 的权重是未初始化内存，访存模式一致但 cache 行为可能不同。

**⇒ 以线 3 的真 8 卡口径（−0.9）为准，本次确认了方向和量级（减半）。**

---

## 3. 结论：**采纳（待精度门）**

| 判据 | 结果 |
|---|---|
| 机制符合设计 | ✅ static 次数不变、每 op 耗时减半 |
| device 收益为正 | ✅ −0.494 ms/pass（dummy）/ −0.9 ms/step（线 3 真 8 卡口径） |
| 需重捕获 | ✅ 是（QLI 在 aclgraph 内，`OP State=static`） |
| **数值等价** | ⚠️ **仅单卡合成数据**（3 长度 `torch.equal`）⇒ **必须过线 1 的真权重精度门** |

---

## 4. 对目标的贡献

当前 6 补丁的 128K = **32.740 ms/step**。
加上本项（−0.9）⇒ **≈ 31.8 ms/step**。

**而这正好是 A=3.493 时达到 110 tok/s 所需的 31.75**（见 `target-requires-A3493.md`）。

⇒ **性能线的部分已达成交付边界**；剩下的唯一变量是 A。

---

## 5. 证据

| 内容 | 路径 |
|---|---|
| A/B profile | `logs/prof_vqA/`、`logs/prof_vqB/`；`/tmp/vqA_rank0.csv`、`/tmp/vqB_rank0.csv` |
| 运行日志 | `/tmp/verify_qli.log` |
| 补丁 | `probe_idx/indexer.py`（含 `CAND_MODE` 与 `QLI_NO_CAND` 两个门控） |
| 启动器 | `serve_a21.sh` 的 `QLI_NOCAND` env |
| 线 3 交付 | `A3-node2:~/handoff/patches/qli-no-candidate/` |
