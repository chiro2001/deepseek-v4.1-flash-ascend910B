#!/usr/bin/env python3
"""把「§7 执行状态」追加到 OPTIMIZATION-ROADMAP-CORRECTED-20261006.md 末尾。

注：该文件每行带一个前导 `+`（历史编辑遗留），追加内容保持一致风格。
"""
from pathlib import Path

P = Path("/home/chiro/projects/dsv41/main-merge/docs/OPTIMIZATION-ROADMAP-CORRECTED-20261006.md")
txt = P.read_text()
if "## 7. ★ 执行状态" in txt:
    print("已存在，跳过")
    raise SystemExit(0)

body = """
---

## 7. ★ 执行状态（2026-10-06 深夜）：**全部关闭**

按本文顺序逐项执行完毕，每项都有**实测**判据。结论：**线 A 与线 B 的可做空间已探明结束**，
剩下的大项（MoE 带宽、HcPre 固定开销）都需要 **kernel 级**工作。

| 项 | 预期 | 实测结论 | 证据文档 |
|---|---:|---|---|
| A1 层内控核 | +4~9% | ✅ 做过，**+0.8%**（kv=8 核最优） | 本文 §2 |
| A1' Scatter 换算子 | +2.9% | ❌ **三条路径全否**：换 `_asc` 无效（同为 aclnn 启动路径）；换 PA 写算子形状不兼容；**合并调用被「同层先写后读」挡住** | `LINE-A-SCATTER-MICROBENCH-20261006.md` |
| A2 通信与计算重叠 | +5~12% | ❌ **单批内不可做**：通信窗口内 AIC 忙 **0.000 ms**，但紧跟其后的消费者立即依赖 allreduce 结果 | `COMM-STRUCTURE-AND-DBO-REGIME-20261006.md` |
| A3 metadata 提前 | +2~3% | ❌ 暴露仅 **0.032 ms** | 本文 §2 |
| A4 SwiGLU 融合 | +0.5~1% | ❌ 暴露仅 **0.025 ms** | 本文 §2 |
| A4 RmsNorm+Quant | +1~2% | ❌ 暴露 ≤0.200 ms | 本文 §3.1 |
| **HcPre（新发现）** | — | ❌ **固定开销主导**：101 次/步 × ~40 µs（Sinkhorn 只占 13%）；迭代数**不可减**（eps 使 12 次仍差 8e-2）；原生实现**慢 35×**；容器内只有 `npu_hc_pre_v2` | `HCPRE-AND-SCATTER-DEEP-DIVE-20261006.md` |
| B pingpong / DBO | **+25~35%** | ❌ **图模式净亏 42%**（conc4: 98.9→59.3；conc8: 130.6→75.7），且**图内并发=串行**（重叠收益 0） | `DBO-GRAPH-VERDICT-NEGATIVE-20261006.md` |

### 7.1 线 B 为什么与预测相反（方法论）

微基准那 1.37× 是「**固定总工作量**切多流」——每个 kernel 的效率不随 shape 变化；
真实推理里「**拆 batch**」会改变 kernel 自身的效率：

* 每半批 token 数减半 ⇒ MoE/GEMM 算术强度与专家利用率下降；
* 每半批各自 allreduce ⇒ 通信次数翻倍且每次更小；
* 而收益上限只有 **~10%**（通信暴露 2.5 ms / 24.6 ms）。

⇒ **不能用合成微基准外推 ubatching 在真实模型上的收益。**

### 7.2 剩余空间（都需要 kernel 级工作）

| 项 | 真实暴露 | 性质 |
|---|---:|---|
| MoE grouped GEMM（w1/w3 + w2） | 4.44 ms | 实测 683~702 GB/s ≈ 58% 可达带宽 ⇒ 理论上限约 1.7× |
| HcPre | 1.99 ms | 101 次/步 × ~40 µs，**43 µs 固定开销**（有效 ~40 GB/s） |
| 通信 | 2.50 ms | 需"另一个 micro-batch"才能填，而拆 batch 已证净亏 |
"""

P.write_text(txt.rstrip("\n") + "\n" + body)
print("appended")
