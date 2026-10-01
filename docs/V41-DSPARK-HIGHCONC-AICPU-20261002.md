# 高并发 DSpark decode：allreduce 记账纠正 + 分块优化证伪（2026-10-02 实测，TP8+DCP8）

> 任务：拆解并优化 DSpark 入图后的 decode `ms/step` 开销（用户指定「采 1/4/8 并发的 profiler，
> 看高并发下 DSpark 的优化怎么进一步突破」）。本文记录高并发档的发现、一个统计口径错误的纠正，
> 以及一条被实测证伪的优化路径。

## 0. 结论摘要

1. **【实测】`allreduceAicpuKernel` 与 `hcom_allReduce` 是同一份工作的两种记账，不能相加。**
   基线 N=8 里两者分别是 33 934 / 34 600 µs/步，数值几乎相等 ⇒ 前者是后者的设备侧执行体。
   此前把它记成"额外 32.9% 开销"是**重复计数**。
2. **【实测】分块 allreduce（绕开 8 MiB 阈值）是负优化**：`hcom_allReduce` 34.6 → 77.6 ms/步，
   设备总计 145 → 156 ms/步。已回退。
3. **【实测】8 MiB 是 HCCL 在图捕获路径下的真实拐点**（微基准 + HCCL 源码双向印证）。

## 1. 五档性能拆解（run `dcpcap_1001_3200_mk0ref`，rank0 op_summary）

配置：`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8 MAX_SEQS=16 BAT_TOKENS=2048 ENGRAM=0`。

| 并发 | steps | 设备总计 µs/步 | hcom_allReduce | hcom_allGather | AICPU_ar | AICPU_ag |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.0 | 43 838 | 13 108 | 2 239 | 439 | 273 |
| 4 | 86.5 | 72 165 | 22 264 | 3 868 | 1 662 | 1 081 |
| 6 | 199.0 | 71 405 | 23 032 | 3 921 | 1 194 | 1 088 |
| 7 | 199.0 | 135 001 | 24 086 | 9 485 | 44 059 | 1 658 |
| 8 | 108.5 | 145 126 | 34 600 | 7 293 | 34 998 | 1 064 |

* `hcom_allReduce` 按 gid 拆（N=8）：`gid=097`（**DCP merge**）13 016 µs/步、`gid=503`（TP/EP）9 486、
  `AivKernel`（其它）12 099。
* 每步 38 次（= 层数）的形态在图内可见：`RunAicpuRpcSrvLaunchV2_allreduce`（`Task Type=AI_CPU`，stream 169）。

## 2. 拐点：8 MiB（微基准 + 源码）

8 卡图捕获微基准（`~/tmp/graph_sweep.py`，与生产同容器/同 HCCL env，38 次捕获进一张图取中位）：

| 包大小 | 单次 | |
|---:|---:|---|
| 3.75 MiB | 77.6 µs | |
| 7.50 MiB | 143.1 µs | ← N=6 |
| **8.75 MiB** | **251.0 µs** | ← 拐点 |
| 15.00 MiB | 254.6 µs | ← N=7/8 |
| 20.00 MiB | 306.9 µs | |

HCCL 源码（`all_reduce_operator.cc: SelectAlgfor91093`）与常量
（`pub_inc/hccl_aiv.h`：`AIV_ALL_REDUCE_A3_ENTRY_SIZE=1 MiB` 单算子、
`AIV_ALL_REDUCE_A3_GRAPH_ENTRY_SIZE=4 MiB` 图模式）给出了门限机制；
`isOnlyAiv`（`CommConfig` 值 4 = `COMM_CONFIG_OPEXPANSION_ONLY_AIV`）可绕过门限，
但 **torch_npu 当前未通过环境变量暴露它**（`HCCL_OP_EXPANSION_MODE` 只认 AI_CPU/AIV/HOST/HOST_TS）。

## 3. ★ 分块优化：设计与实测（**证伪**）

**假设**：把 `[T,64,640]` fp32 沿 dim 0 切成 ≤7 MiB 的块，避开 AICPU-RPC 回落。

**实现**：`_v41_chunked_allreduce()`（`dsa_v41.py`），env `V41_DCP_AR_CHUNK_MB`（默认 0 = 原行为）。
行切片对连续张量零拷贝，数学上与单次 `all_reduce` 等价。

**正确性**：8 卡容器内单测 `~/tmp/archunk_test.py`，`[96,64,640]` 单次 vs 3 分块：
`torch.equal=False`、`max|d|=2.86e-06`（fp32 求和顺序差异，量级正常）。

**性能实测**（`V41_DCP_AR_CHUNK_MB=7`，其余配置逐项对齐，KV 容量同为 7 437 469 tokens）：

| 指标 | 基线 | 分块 |
|---|---:|---:|
| `allreduceAicpuKernel` | 31.2 次/步、33 934 µs/步 | **0（消失）** |
| `hcom_allReduce` | 431 次/步、34 600 µs/步 | **660 次/步、77 592 µs/步** |
| 设备总计 | 145 126 µs/步 | 156 489 µs/步 |
| ms/step（N=8 实测，同脚本同配置） | **136.40** | **143.09** |

* **同脚本同配置的干净 A/B**：基线 **136.40 ms/step** vs 分块 **143.09 ms/step**（**+4.9%**，负优化）。
* 分块**确实消除了 AICPU-RPC 内核**，但 `hcom_allReduce` 反而翻倍
  ⇒ 通信次数从 431 → 660 次/步，每次的固定开销叠加，得不偿失。
* 微基准里分块同样不占优（283–343 µs vs 单次 254 µs）。
* **处置：已回退**（`~/dcpw/.../dsa_v41.py` 恢复 `c21d4afa`，备份 `*.bak_archunk_failed`）。

## 4. 复现

```bash
# 采集（1/4/8 并发，带 warmup）
python3 ~/tmp/n8_test.py 19210 8 200
# 分档账本
python3 ~/tmp/conc5.py /opt/dsv41/results/<run>/prof     # 容器内
python3 ~/tmp/aicpu_only.py <op_summary.csv> <tag> <steps>
# 拐点微基准（8 卡图捕获）
docker exec <ctr> bash -lc "cd /tmp && HCCL_NPU_SOCKET_PORT_RANGE=61000-62000 \
  torchrun --nproc_per_node=8 --master_port=29780 /tmp/graph_sweep.py"
```

## 5. 下一步候选（按证据强度排序）

| # | 方向 | 依据 | 风险 |
|---|---|---|---|
| 1 | merge 包 **宽度 640 → 528**（对齐 128 → 16） | 可省 17.6% 通信量；纯 padding，无精度风险 | 需确认 16 B 对齐满足 |
| 2 | merge 包 **fp32 → bf16** | 通信量减半，且能跨过 8 MiB 门限 | 归约精度（bf16 累加） |
| 3 | `RS_MERGE`（只收 1/8 head） | 理论省 7/8 接收量；已实现但有深层 bug（A→1.00） | 未解决 |
| 4 | 减少 `gid=503`（TP/EP）allreduce 次数 | 9.5 ms/步，与 merge 同域排队 | 需改上游 |


## 6. ★ bf16 归约：设计、实测（**同样证伪**）

**假设**：把归约包从 fp32 降到 bf16，通信量减半并跨过 8 MiB 门限。
**实现**：`_v41_bf16_allreduce()`（`dsa_v41.py`），env `V41_DCP_AR_BF16`（默认 0）；
只在 `t.numel()*element_size() <= 32 MiB`（decode 档）时生效，避开 prefill 的 GB 级分配。

**微基准（8 卡图捕获，`~/tmp/bf16_test.py`）**

| 形状 | dtype | 大小 | 单次 | 38 层/步 |
|---|---|---:|---:|---:|
| [96,64,640] | fp32 | 15.00 MiB | 260.8 µs | 9.91 ms |
| [96,64,640] | bf16 | 7.50 MiB | 143.3 µs | 5.45 ms |
| [96,64,528] | bf16 | 6.19 MiB | 119.8 µs | 4.55 ms |

微基准里 bf16 确实快 **45%**；精度 `max|d|=0.128`、相对 **0.84%**。

**端到端实测（N=8，同脚本同配置）**

| 指标 | 基线 | bf16 |
|---|---:|---:|
| ms/step | **136.40** | **153.43** |
| A（接受长度） | 1.90 | **1.66** |
| 聚合 tok/s | 111.8 | 87.2 |

⇒ 端到端 **−12.5%（更慢）**，且 **A 掉 12.6%**（精度受损）。
原因：38 层各多两次 dtype 转换 + 一次 `copy_`，抵消并超过了通信上的收益；
bf16 累加使 merge 结果偏离，直接反映为草稿接受率下降。
**处置：已回退**（`*.bak_arbf16_failed`）。

## 7. 至此证伪的三条路径

| # | 路径 | 结果 |
|---|---|---|
| 1 | 分块 allreduce（≤7 MiB） | ms/step +4.9%，hcom_allReduce 翻倍 |
| 2 | bf16 归约 | ms/step +12.5%，A −12.6% |
| 3 | `RS_MERGE`（reduce_scatter，只收 1/8） | A → 1.00（早前轮次，深层 bug 未解） |

## 8. 下一步：padding 浪费（新发现）

`CAPTURE_SIZES = 1,2,3,4,8,12,16,20,24,32,40,48,96,128` **没有 56/64 档** ⇒
N=7（T=56）与 N=8（T=64）都被 padding 到 **T=96**，多算 43–71% 的行，
且包被推过 8 MiB 门限（15.75 MiB vs 10.5 MiB）。

候选：把 56/64 加进 `CAPTURE_SIZES`，用实测确认 N=7/N=8 的 gain。


## 9. ★ 有效的优化：bucket 稠密化（padding 消除）

**根因**：`CAPTURE_SIZES` 的自动推导用几何级数（…32,40,48,**96**,128），
而每步 token 数 = 并发 × 8 ⇒ 并发 7–11 都落在 48→96 的空档里，被 padding 到 **96**：

* profiler 实测 `HcPre` 的 Input Shapes 就是 `96,4,5120`（真 batch 只有 64 行）；
* 白算 50% 的行；
* merge allreduce 包 10.5 MiB → **15.75 MiB**，跨过 HCCL 8 MiB 门限。

**改动**（`scripts/serve_a2.sh`，`[CAPTURE-DENSE]`）：在 48→96 这个最大缺口里按
「每步 token 数」补桶（最多 5 个），`MAX_SEQS=1/4` 的桶列**保持不变**：

| MAX_SEQS / SP_TOKENS | 桶列 |
|---|---|
| 1 / 7 | `1,2,3,4,8,12,16,20,24,32`（**不变**） |
| 4 / 7 | `1,2,3,4,8,12,16,20,24,32`（**不变**） |
| 16 / 7 | `…,48,`**`56,64,72,80,88`**`,96,128` |
| 32 / 7 | `…,48,`**`56,64,72,80,88`**`,96,192,256` |

**实测（同脚本同配置 A/B）**

| 口径 | 并发 | 基线 | 稠密桶 |
|---|---:|---:|---:|
| profiler（3 次中位） | 8 | 136.40 | **127.60** |
| 非 profiler | 8 | 110.09 | 待测 |

## 10. ★ 新发现的独立缺陷：并发 ≥10 接受长度塌成 1.00

非 profiler 基线曲线（run `dcpcap_1002_024731_base_np2`）：

| 并发 | 真 batch | 桶 | ms/step | A | 聚合 tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 43.63 | 2.88 | 62.2 |
| 4 | 32 | 32 | 71.44 | 2.28 | 112.0 |
| 8 | 64 | **96（padding）** | 110.09 | 1.85 | 118.5 |
| **10** | **80** | **96（padding）** | 99.14 | **1.00** | 91.2 |

`A=1.00` ⇒ **所有草稿全部被拒**（`accepted_tokens` 零增长），且 `steps` 恰好钉在 199
（= 池化的上限），说明解码退化成"每步只出 1 个 token"。

**注意**：这不是本次 padding 改动的产物，基线就有；`A=1.00` 在高并发下会**污染整个引擎**
（实测一次之后，后续 N=1/N=4 也变成 A=1.00，直到重启服务）。
⇒ 与早前记录的「并发 16 是独立问题」是同一族。**这是下一个要修的目标。**

## 11. 口径提醒（避免误引用）

* 本文所有 ms/step 若标注 "profiler" 都在 `start_profile/stop_profile` 区间内测的，
  设备侧 profiling 会抬高绝对值；**只有同一口径内的 A/B 差值可比**。
* `ms/step` 用 `draft_tokens_增量 / (K×并发)` 算步数，与 `HcPre` 次数交叉校验过
  （N=8：HcPre 13072 / 38 ≈ 344 步… 见各 run 的 op_summary）。
* **A 随版本/形状漂移**（同配置不同 run：1.85 / 1.90 / 2.24）——HCCL allreduce 在该形状上
  求和顺序可变（本文件早前已记录），所以**跨 run 比 A 不可靠**，要按同一次生成内的
  `accepted/drafted` 比。
