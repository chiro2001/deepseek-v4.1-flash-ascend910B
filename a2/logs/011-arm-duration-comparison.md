# 两条线的迭代成本对比：单卡 tiny+dummy vs 8 卡真权重

> 2026-09-22 00:2x。【实测】，来自 L1_dummy 的 `arm-d4k` 与 D_off8/D2 的 8 卡臂。

---

## 一、结论：单卡方案把每臂从 **25 分钟** 压到 **70 秒**

| | 8 卡真权重（`001`/D2 的臂） | **单卡 tiny + dummy**（`arm-d4k`） |
|---|---|---|
| 模型 | `v41-w4a8-engram-dr-vision-qrot-mtpq` | **`model-tiny`（6.3 MB）** |
| 权重加载 | 真权重 273 GB，加载 **471 s** | **`--load-format dummy`，秒级** |
| **起服到就绪** | **~1320 s（22 min）** | **45 s** ← ★ |
| 压测 | 16×32768 × 2 轮 | 16×4096 × 2 轮 |
| **整臂（起服→压测→收尾）** | **~25 min** | **70 s**（16:00:15 → 16:01:25） |
| 占用 | Phy-ID 8–15（8 张） | **c0 一张** |
| **加速比** | 1× | **≈19×** |

---

## 二、它**确实**跑起来了（关键判据）

```
model=/work/agents/L1_dummy/models/model-tiny model_bytes=6376695
prefix_match_unit=32 l1_patch=1 wo_a_fix=1  engram=0  load_format=dummy
GPU KV cache size: 22,719 tokens                       ← KV 池分配成功
[kv_events] BlockStored:GPU=11286  BlockStored:CPU=704  BlockRemoved:GPU=7322
[fill]    ttft p50 = 461.4 ms
[replay1] ttft p50 = 453.1 ms                          ← 目前只快 1.8%（取回还没修好）
```

⇒ **DSV4.1 的 KV group 结构（`full` + `state`(block 32) + 多个 SWA）在截断 config 下保留了**，
服务能起、KV 能分配、**CPU 侧确实存了 704 个块**。**"能不能单卡测 offload" 的答案是：能。**

> ⚠️ 一个待解：`kv_events` 里 `bytes={'GPU': 1444608, 'CPU': 0}` —— CPU 侧记了 704 个块却报 0 字节，
> 且 replay ≈ fill（取回未生效）。**L1 正在迭代**。

---

## 三、代价（要如实记）

| 项 | 说明 |
|---|---|
| **需要一个 `wo_a` dummy 适配** | `arm-d4k` 的日志里有 `[L1_dummy] wo_a dummy 适配：(4096, 4096) -> (8, 4096, 512)` —— dummy 权重下 `wo_a` 的形状与 V4.1 期望不符，L1 打了适配 |
| **结构不等价** | 截断层数（40 → 4/8）后 group 的**数量**变了，但**类型不变**。若 bug 依赖"13 组"这个数量，可能不复现 |
| **TP1 丢掉了跨 rank 行为** | 候选根因 ①（`group_idx` 错位）是跨 rank 问题，D2 已排除；但其它跨 rank 效应仍可能被掩盖 |
| **丢掉了 Engram** | `engram=0`；真机上 Engram 打开会撞 `207001` |
| 前 6 个臂都失败 | `nopmu` / `pmu32` / `nopatch` / `nopatch2` / `woa` / `woa2` 全是 `ERR99999 UNKNOWN application exception`，只有 `d4k` 通过 ⇒ **踩了 6 次坑才通，这些坑本身有价值** |

---

## 四、对决策的影响

**⇒ 迭代策略改成"单卡快跑、8 卡终验"：**

1. **根因定位 / 修复迭代** —— 全部在单卡 tiny 上做（70 s/臂）；
2. **8 卡只用于**：① 复现"单卡不复现"的问题（如跨 rank）；② 生产配置的最终确认。

**⇒ 对 A2 的直接影响**：以后 A2 上的调参（`cpu_bytes_to_use`、`blocks_per_chunk`、`prefix-match-unit`）
也可以先在 A3 单卡上扫一遍，再拿到 A2 上验一次，**不必每次都在 A2 上从零试**。

