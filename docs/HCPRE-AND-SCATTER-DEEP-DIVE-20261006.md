# HcPre / Scatter 深挖：三条被证伪的优化路径 + HcPre 的真实结构（2026-10-06）

> 目的：线 A 里"真实暴露"最大的三项是 MoE(4.44ms)、**HcPre(1.99ms)**、通信(2.50ms)。
> 通信已证只能靠另一个 micro-batch（线 B，已证伪）；本轮把 HcPre 与 Scatter 挖到底。
> 全部为【实测】。

## 0. 一页纸

| 路径 | 结论 |
|---|---|
| HcPre 减少 Sinkhorn 迭代（20 → 8） | ❌ **不能**：eps 下限使收敛极慢，12 次仍差 8e-2 |
| HcPre 换原生 PyTorch 实现 | ❌ **慢 35×**（1897 µs vs 55 µs/call） |
| HcPre 换其它已注册算子 | ❌ 容器内只有 `_C_ascend.npu_hc_pre_v2` 一个 |
| Scatter 换 `npu_scatter_pa_kv_cache_functional` | ❌ 该算子拒绝我们的形状（head=1） |
| **HcPre 的真实结构** | ✅ 101 次/步 × ~40 µs，**固定开销主导**（迭代只占 ~1/3） |

## 1. HcPre 的真实结构（profile 权威数据）

```
op_statistic.csv:  HcPre  MIX_AIC  Count=7310  Total=294538 us
                   Min=26.86  Avg=40.29  Max=289.53   (72 步窗口)
⇒ 101.5 次/步，均 40.3 µs，合计 4.09 ms/步（union 3.279，暴露 1.987）
```

调用点（`models/deepseek_v41/model.py:999 / :1033`）：**每层两次**
（attn 用 `hc_attn_*`、ffn 用 `hc_ffn_*`），输入不同（ffn 吃 attention 之后的 hidden_states）
⇒ **无法合并**。40 层 × 2 = 80 次/步，其余来自 draft/预热步。

### 1.1 微基准：单次成本 vs T 与迭代数（K=40 批量，去掉同步开销）

| T | iters=1 | iters=20 | 数据量 |
|---:|---:|---:|---:|
| 48（decode 档） | **48.4 µs** | 54.9 µs | 2.05 MB |
| 96 | 52.6 µs | 62.6 µs | 4.01 MB |
| 512 | 99.1 µs | 143.7 µs | 21.05 MB |

三条读法：

1. **固定开销约 43 µs**：T 从 48 → 512（10.7×），耗时只从 48 → 99 µs（2.06×）；
2. **Sinkhorn 迭代只占 ~13%**（48.4 → 54.9 µs）——文档里"受 20 次 Sinkhorn 串行限制"
   的说法**不成立**；
3. 2 MB 数据若按 1182 GB/s 只需 1.7 µs ⇒ 当前**有效带宽约 40 GB/s（3.5%）**，
   说明瓶颈是**核内固定开销/并行度**，不是数据量。

## 2. 三条被证伪的路径

### ❌ 2.1 减少 Sinkhorn 迭代

`hc_split_sinkhorn_torch` 的循环里每步都 `/(sum + eps)`，**eps 下限让不动点偏移**，
收敛极慢（fp64 下同样慢，与精度无关）：

| iters | max\|X_iters − X_20\|（fp32，300 组随机输入） |
|---:|---:|
| 8 | 3.43e-01 |
| 12 | 8.02e-02 |
| 16 | 2.28e-02 |
| 20 | 0 |

⇒ 迭代次数**是模型语义的一部分**（20 次本身也未收敛），**不能改**。

### ❌ 2.2 换原生 PyTorch 实现

`hc_pre_native`（flatten→rsqrt→linear→sinkhorn→collapse）在 T=48 实测
**1897 µs/call**，比融合算子（55 µs）**慢 35×**（每个 torch 算子一次 kernel 发射）。
⇒ 融合算子已经是正解。

### ❌ 2.3 换 Scatter 专用算子

`npu::npu_scatter_pa_kv_cache_functional` 的签名是
`(key, value, key_cache, value_cache, slot_mapping, *, cache_mode)`，
四种 cache_mode（PA_NZ / PA / PA_BNSD / PA_BLK_BNSD）**全部** `aclnnScatterPaKvCache failed`
——我们的 `num_kv_heads=1` 形状不被接受。`scatter_update` 亦失败。

（Scatter 本身的账见 `docs/LINE-A-SCATTER-MICROBENCH-20261006.md`：
固定 43 µs/次，58 次/步 = 1.01 ms，**唯一可行方向是合并调用**。）

## 3. 顺带确认的两件事

1. **tp8k5 就是权威基线**：`results/armIMG_restore10_*/inner.sh` 显示
   TP=8 / MAX_LEN=1M / MAX_SEQS=32 / BAT=8192 / KV=bf16 / PREFIX=1 / SPEC=1 / SP=5，
   且 `[bneck] hp = 24.590 ms`（6 并发）——**与目标里"真实步长 24.59 ms"完全一致**。
2. **tiny 上拿不到 `[bneck] hp`**：bneck 探针（`model.py` 的 `_BneckState.tick()`）
   只在 engram 路径被调用，而 tiny 跑 `ENGRAM=0` ⇒ 探针日志 0 行。
   要在 tiny 用门 #3 口径，需要开 ENGRAM 或给非 engram 路径补一次 `tick()`。

## 4. 结论：线 A 的"真实暴露"项已全部探明

| 暴露项 | 真实暴露 | 结论 |
|---|---:|---|
| MoE w1/w3 + w2 | 4.44 ms | 带宽受限（683/702 GB/s ≈ 58% 可达），需 kernel 级优化 |
| **HcPre** | **1.99 ms** | **固定开销主导（~43 µs/次 × 101 次）**，官方融合算子已是最优实现 |
| **通信** | **2.50 ms** | 单批内无独立工作可填（已证），只能靠 micro-batch（已证净亏） |
| SparseFlashMla | 1.38 ms | 依赖链上的计算 |
| RmsNorm / HcPost / RoPE | 2.37 ms | 依赖链上的计算 |
| Scatter | 0.72 ms | 固定开销，**合并调用**是唯一可行方向（+3.3% 上界） |

⇒ 除 **Scatter 合并（+3.3%）** 外，线 A 剩余项都需要**写/改 AscendC kernel**
（HcPre 固定开销、MoE 带宽利用率），不属于"低风险小改动"。

## 5. 复现

```bash
ssh a3-21 'python3 ~/tmp/sinkhorn_conv.py'                    # 迭代收敛性
ssh a3-21 'docker cp ~/tmp/hcpre_bench2.py dsv41-tinyspark:/tmp/ && \
  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 hcpre_bench2.py"'   # T×iters
ssh a3-21 'docker cp ~/tmp/hcpre_native.py dsv41-tinyspark:/tmp/ && \
  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 hcpre_native.py"'   # 原生对照
ssh a3-21 'docker cp ~/tmp/scatter_bench3.py dsv41-tinyspark:/tmp/ && \
  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 scatter_bench3.py"' # PA 算子
```
