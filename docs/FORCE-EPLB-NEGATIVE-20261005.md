# `enable_force_eplb`（强制专家打散）：**负结果**（2026-10-05）

> 动机：上一轮把 allreduce 的 2.3× 不对称拆到"**到达偏差**"（HCCL 记 100% Idle；
> rank0 中位 63.7 µs vs rank5 47.6 µs），于是"让 MoE 完成时刻更齐"成为唯一方向。
> 本栈现成的相关旋钮只有两个：`enable_force_eplb`（本文件）与
> `eplb_config.dynamic_eplb`（见 `HCCL-ALLREDUCE-NEGATIVE` §3.6，尚未验证）。
> 标注：【实测】。

## 1. 这个开关做什么（读代码）

`vllm_ascend/ops/fused_moe/routed_experts.py:597`：

```python
if get_ascend_config().enable_force_eplb:
    topk_ids = get_force_eplb_topk(topk_ids, num_logical_experts)   # 人为打散
elif enable_force_load_balance:
    topk_ids = torch.argsort(torch.rand(...))[:topk_ids.size(1)]     # 随机
```

⇒ 它**不是**"负载均衡"（不搬权重、不测热度），而是**在路由处人为打散 topk**。
`ascend_config.py:480-494` 明确它与 `dynamic_eplb` **互斥**（同时开会在启动时 ValueError）。

## 2. 实验与结果

臂：`armFE_forceeplb` = 发布镜像 + `FORCE_EPLB=1`（其余与基线逐项相同）。

| 判据 | 结果 |
|---|---|
| 能否起服 | ✅ 正常（冷编译，与基线同） |
| **KV 容量** | **2,987,509**（与基线**逐字相同** ⇒ 没引入额外权重/缓冲） |
| 单流 tok/s | 113.8（基线 116.4） |
| N=8 聚合 | 335.4（基线 362.7） |
| `ab_gate`（全桶中位） | **+0.166 ms**（vs `armF_r7_padskip`）/ **+0.354 ms**（vs `armIMG_v3_final`） |
| `ab_gate` 轮次 | 0/1 为负 |

### 2.1 按稳定桶复核（与前一轮同样的口径）

| 真实并发档 | 样本（A/B） | 基线 A | force-EPLB B | Δ |
|---:|---|---:|---:|---:|
| n=6（conc=1） | 3168 / 1056 | 24.60 | 24.46 | **−0.14 ms（−0.6%）** |
| n=12（conc=2） | 1248 / 344 | 27.28 | 27.34 | **+0.06 ms（+0.2%）** |
| n=24（conc=4） | 400 / 128 | 32.02 | 31.85 | **−0.17 ms（−0.5%）** |
| n=48（conc=8） | 200 / 72 | 40.63 | 41.04 | **+0.41 ms（+1.0%）** |
| n=18（过渡桶） | **192 / 40** | 31.54 | 31.89 | +0.35（**p90 = 1738 ms**，样本太少，不作判据） |

⇒ **四个稳定桶全部落在 ±0.6% 内、方向不一致** ⇒ **无收益**。

## 3. 判决与推论

**不采纳。** 并且这个负结果**加强了 §3.5 的结论**：

* 若 allreduce 的等待真的来自"专家负载不均导致各 rank 完成时刻不齐"，
  那么**人为打散 topk**（本实验）本应部分缓解它 —— 实测**没有**。
* ⇒ 那条 2.3× 不对称更可能来自**别的机制**：比如 MoE 侧那次 allreduce
  与**共享专家**的输出合并（`fused_moe.py:147` 的 `_reduce_shared_output_if_needed`）
  导致它必须等两条分支都完成，而不是等其它 rank。
  【推断】若成立，可动的量在"**共享专家与 MoE 的调度**"而不是通信参数。

## 4. 复盘：`ab_gate` 的中位数会被过渡桶带偏（第三次踩到）

这是**同一类问题第三次出现**（前两次：`armH`、`ENGRAM_WKV_TP`）。规律：

* `ab_gate.py` 取的是**全部 batch 桶的中位**，而 **n=18 桶永远是过渡桶**
  （采样少、且 p90 出现 1.7×10³ ms 级的 ramp 值）；
* ⇒ **只要有一个真实档变快、而 n=18 变慢，中位数就可能为正**，把真收益吞掉；
* **正确做法**：先看**每个桶的样本数与 p10/p90**，只在"样本 ≥64 且 p90 与 p10 同量级"的
  桶上做判决（本轮 4 个稳定桶全部合格）。已把这条写进 `tools/ab_gate.py` 的建议里。

## 5. 复现

```bash
sed -e 's|^export IMAGE=.*|export IMAGE=local/dsv41-a3-tp8:20261005-1001|' \
    -e '/V41_HC_OPP_PKG/d' -e '/^export ENGRAM_DEVICE_INDEX/a export FORCE_EPLB=1' \
    ~/tmp/launch_armF.sh > ~/tmp/launch_armFE.sh
BENCH_CONC=1,2,4,8 BENCH_REPS=3 BENCH_OUT=192 \
  bash tools/run_arm_suite.sh ~/tmp/launch_armFE.sh armFE_forceeplb 1 0 0
# 判据：先看分桶样本数/p90，再看稳定桶的配对差
python3 ~/tmp/hp_buckets.py armIMG_v3_final armFE_forceeplb
```
