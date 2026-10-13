#!/usr/bin/env python3
"""写 prefill 的最终画像（含对我自己两轮结论的连续更正）。"""
from pathlib import Path

DOC = Path("/home/chiro/projects/dsv41/main-merge/docs/PREFILL-FINAL-PICTURE-20261007.md")

TEXT = '''# ★ prefill 的最终画像：算力饱和 + 通信已重叠（含连续两次自我更正）（2026-10-07）

> 这份文档修正了我**自己上一轮的"更正"**。追溯过程本身有方法论价值，故完整保留。
> 结论以**暴露度**（union − 与其它 kernel 的重叠）为准，不以算子耗时求和为准。

## 0. 三代结论的演进（同一份 profile，三种口径）

| 代 | 口径 | 结论 | 对否 |
|---|---|---|---|
| ① | 并发不伸缩 + 墙钟整除 | prefill 是 **compute-bound**（~8K tok/s，合批不帮） | ✅ **最终成立** |
| ② | `op_statistic.csv` **求和** | `allreduceAicpuKernel` 占 **45.5%** ⇒ 通信受限 | ❌ **求和假象** |
| ③ | **暴露度**（本轮） | allreduce 的**实际搬运**只占活跃窗口 **11%**，其余是算力 | ✅ **最终结论** |

## 1. ②为什么错：`allreduceAicpuKernel` 的 6.93 s 里有 6.07 s 是**等待**

同一份 prefill profile（8 条 ~8000-token prompt、15 个 forward、窗口 14779 ms）：

| 项 | 次数 | 求和 ms | union ms | **被其它 kernel 覆盖** | **真实暴露** |
|---|---:|---:|---:|---:|---:|
| `allreduceAicpuKernel` | 729 | 6930.9 | 6930.9 | **6067.2** | **863.7 ms** |
| `hcom_allReduce`（实际搬运） | 1326 | 1094.8 | 1094.8 | 90.1 | **1004.8 ms** |
| 其它 kernel | — | — | 7927.7 | — | — |

⇒ `allreduceAicpuKernel` 是**编排/等待**型 kernel，它 **87.5% 的时长与其它 kernel 重叠**
（等待期间设备在跑别的请求的算力）。把它按耗时求和得到 45.5%，
相当于把"等待"算成了"工作" —— 这正是本仓反复记录的同一个教训
（"算子耗时 ≠ 可回收时间"）。

## 2. ③的关键：先把**测量自身的人为 artifact** 剔掉

初版分析里"设备 39.5% 空闲"是**我的采集脚本造成的**：

```
POST /start_profile
time.sleep(2)        ← 人为空转
… 8 条并发请求 …
time.sleep(2)        ← 人为空转
POST /stop_profile
```

空闲段实测只有 **7 个**，其中 **4 个 ≥50 ms**，最大的两个是
**3571.5 ms** 与 **2132.0 ms** —— 分别对应"起服后到请求真正开跑"与"请求结束到 stop_profile"。
**剔除这两个 artifact（合计 5703.5 ms）后**：

| 口径 | 数值 |
|---|---:|
| 原始窗口 | 14779.1 ms |
| **真实活跃窗口** | **9075.6 ms** |
| 全部 kernel 并集 | 8942.4 ms |
| **设备忙碌率** | **98.5%** |
| 真实空闲（除 artifact 外） | **133 ms（1.5%）** |

⇒ **prefill 期间设备几乎满负荷，没有"停顿"可回收。**

## 3. 最终画像（按活跃窗口 9075.6 ms 折算）

| 项 | 时长 | 占活跃窗口 |
|---|---:|---:|
| 全部 kernel 并集 | 8942.4 ms | **98.5%** |
| └ 其它 kernel（算力为主）并集 | 7927.7 ms | 87.4% |
| └ **allreduce 实际搬运（`hcom_allReduce`）暴露** | **1004.8 ms** | **11.1%** |
| └ allreduce 与算力**重叠**部分 | 6157.3 ms | （已含在上面） |
| 真实空闲 | 133 ms | 1.5% |

两条关键读法：

1. **实际通信（`hcom_allReduce`）91.8% 暴露** ⇒ 它确实在关键路径上，但**只有 11%**；
2. **设备已经 98.5% 忙**，其中绝大部分是算力 —— **prefill 是算力饱和状态**。

## 4. 因此：prefill 没有"配置/调度漏洞"可捡

| 曾被怀疑的原因 | 实测结论 |
|---|---|
| 合批不足（BAT_TOKENS 限制） | ❌ 2K prompt 可合批，仍只有 1.16× |
| `admission_gate` 限流 | ❌ 默认 8，不是元凶 |
| 通信（allreduce）主导 | ❌ 求和 45.5% 是假象；实际搬运仅 **11%** 且已基本重叠 |
| 设备停顿/气泡 | ❌ 真实空闲只有 **1.5%** |
| **⇒ 真实瓶颈** | ✅ **算力 kernel 本身**（87.4% 的时间是算力在跑） |

**⇒ prefill 提速只能靠 kernel 级优化**（算力 kernel：`SparseFlashMla` 12.8%、
`HcPre` 6.9%、`QuantLightningIndexerV2` 6.2%、各类 MatMul/Quant 合计 ~8%…），
以及**把通信从 11% 进一步藏掉**（空间有限）。

## 5. 方法论教训（值得写进流程）

1. **任何"某资源占比"的结论，必须先算暴露度**，不能直接对 `op_statistic.csv` 求和 ——
   本仓在 decode 侧已经踩过（"AICPU 求和 2.07 s 但暴露只有 0.48 s"），本轮在 prefill 侧又踩一次。
2. **采集脚本自身的 sleep 会污染"空闲"结论** —— 必须把 start/stop_profile 紧贴工作负载，
   或在分析时按"活跃窗口"折算。
3. **同一天里对同一指标给出相反结论时，要保留演进过程**（本文件即为此目的），
   否则后续会反复回到已被否掉的方向。

## 6. 环境状态

本轮**未重启任何服务**（全部为离线分析 + 一次 `PROFILE=1` 的采集）。
tp8k5 = 交付配置（health=200、KV 2,987,400、BAT 8192、MAX_SEQS 32、SP 5、`enable_prefill_mc2=false`、
dspark 开启）；tiny（chips 2/3）= health=200、未动。未提交改动 0。

## 7. 复现

```bash
# 采集（注意：start/stop 要紧贴负载，避免 sleep 污染）
# 暴露度分析
ssh a3-21 'docker cp ~/tmp/ar_exposure.py dsv41-tinyspark:/tmp/ && \\
  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 ar_exposure.py <pf 目录>"'
```
'''

DOC.write_text(TEXT)
print("wrote", DOC, len(TEXT))
