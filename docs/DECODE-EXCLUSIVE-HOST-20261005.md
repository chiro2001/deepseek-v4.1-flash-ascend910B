# decode 步的「独占贡献」分解，与 host 侧下发账（2026-10-05）

> 目的：回答"下一个该动谁"。前几轮用的是"busy 之和 / 并集 / 真空闲"，
> 这些口径能把一个 stream 的工作量说清楚，但**说不清它到底占不占墙钟**。
> 本轮换一个判据：**独占贡献 = union(全部任务) − union(去掉该 stream 的任务)**。
> 它直接给上界——"把这个 stream 的活儿全删掉，墙钟最多能缩多少"。
>
> 数据：`k6full_1004_100156` PROF_000003（rank0，626 步 / 17.69 s，TP8 + SP5 + DRAFT_GRAPH）。
> 分析窗口 3728.88–3754.47 ms（25.59 ms，2549 个任务）。纯离线，未占卡、未起停服务。

## 0. 一句话结论

1. **主模型图（stream 158）独占 13.27 ms（51.9%）**——大头仍然在这里，但已被反复攻过。
2. 三个非主流加起来 **4.70 ms（18.4%）**：**draft 1.87** > **采样/输入准备 1.64** > **metadata 1.19**。
3. metadata 的 7 次/步 **不是"算得慢"，是"被争用放大"**：隔离基准里 device 排队吞吐只有
   **1.4 µs/次**，服务内实测 165–232 µs/次。
4. **host 侧下发账（新证据）**：本 profile 的 acl host 时间合计 **33.9 ms/步**，
   而步周期 ~28 ms ⇒ 【推断】decode 墙钟**至少部分是 host-launch-bound**，
   "并集里的 2.88 ms 空隙"很可能不是设备的锅。

## 1. 各 stream 的独占贡献（decode 步，25.59 ms）

| stream | 身份 | 任务数 | busy 之和 | **独占贡献** | 占比 |
|---:|---|---:|---:|---:|---:|
| **158** | 主模型 40 层（图） | 1348 | 14.863 ms | **13.271 ms** | **51.9%** |
| **154** | DSpark draft（3 层 + LM head） | 250 | 2.034 ms | **1.867 ms** | **7.3%** |
| **47** | 采样 / 输入准备 / engram gather | 424 | 1.700 ms | **1.638 ms** | **6.4%** |
| **35** | AICPU metadata ×7 | 37 | 1.270 ms | **1.190 ms** | **4.7%** |
| 155 / 156 | KV / 共享专家副流 | 146 / 144 | 1.183 / 0.838 | 0.148 / 0.307 | 0.6% / 1.2% |
| 151/152/157/38/41 | 小副流 + 通信 | — | — | 0.035 / 0.045 / 0 / 0 / 0 | ≤0.2% |

* 全部任务并集 22.708 ms（88.7%），**空隙 2.882 ms（11.3%）**；
  但其中 ≥20 µs 的"真空闲"只有 0.147 ms（见 `STEP-FULLY-PACKED`）⇒ 空隙是碎的。
* stream 157（1.97 ms busy）与 153（0.37 ms）**独占为 0** ⇒ 这两条纯通信/伴随流
  完全被别的流盖住，动它们不影响墙钟。

### 1.1 对上一轮口径的修正

`LEVERS-R5` 里把 stream 47 的小算子海估成「~0.5 ms/步」，那是按"每个小算子 5 µs 下发空隙"
累加出来的**下界**。用独占口径重测，它的上界是 **1.64 ms/步**，
而且其中 21/22 个算子族的独占 ≈ 自己的 busy（即**几乎没有被重叠掉**）。
⇒ 这条线的天花板比原来估的高 3 倍，值得按 `LEVERS-R5 §D` 的方向做融合。

## 2. 算子级独占贡献（谁真的占墙钟）

### 2.1 stream 47（1.638 ms）

| 算子 | 次数 | busy | **独占** |
|---|---:|---:|---:|
| ViewCopy | 24 | 0.191 ms | **0.188 ms** |
| MatMulV2（LM head 为主） | 2 | 0.177 | **0.177** |
| Index | 17 | 0.167 | **0.166** |
| SparseAttnSharedkvMetadata | 1 | 0.165 | **0.165** |
| Cast | 84 | 0.114 | 0.098 |
| IndexCheck | 19 | 0.070 | 0.069 |
| ZerosLike | 3 | 0.067 | 0.067 |
| SelectV2 / Fill / GatherV3 | 33 / 48 / 18 | 0.188 | 0.180 |
| FloorMod / FloorDiv / ClipByValueV2 / GreaterEqual | 18/17/18/17 | 0.169 | 0.151 |
| 其余（Copy/Transpose/ArgMax/IndexFill/IndexPut/Range…） | ~60 | 0.24 | ~0.22 |

> 特征：**单算子 1–10 µs 的 device 时间，却基本不被重叠**。
> 与 host 侧账（§3）结合看，这批 op 的真实身份是"**host 一条条下发、设备来不及盖住**"。

### 2.2 stream 35（1.190 ms）——全部是 7 次 AICPU metadata

| 算子 | 次数 | busy | **独占** |
|---|---:|---:|---:|
| SparseFlashMlaMetadata | 3 | 0.603 ms | **0.581 ms** |
| QuantLightningIndexerV2Metadata | 2 | 0.389 | **0.389** |
| SparseAttnSharedkvMetadata | 1 | 0.190 | **0.190** |

**7 次/步的构成**（全 profile 计数反推，626 步）：
`SparseFlashMlaMetadata 3`（按 `smla:c{ratio}` 去重后仍有 3 个 ratio）
`＋ QuantLightningIndexerV2Metadata 2（qli:c{ratio}）`
`＋ SparseAttnSharedkvMetadata 2（一条在 s47、一条在 s35）`。
**注意：这已经是"按 (ratio, stage) 去重之后"的调用次数**——
`dsa_v41.py` 里 `_publish_task(batch_shared, f"smla:c{ratio}", …)` 已经在做每步缓存，
想再降只能**跨 ratio 合并**或**换执行位置**。

## 3. metadata 算子的隔离基准（新实测）

在 tiny 容器（单卡、`dsv41-tinyspark`，B=1/8/16/32，`npu_sparse_flash_mla_metadata`）：

| 测法 | 结果 |
|---|---|
| host 下发（不含 device 等待） | **55–57 µs/次，且与 batch(1→32) 无关** |
| 连续入队 50 次后的 device 吞吐 | **1.3–1.4 µs/次** |
| 一次一同步（含往返延迟） | 129–219 µs/次 |
| 服务内实测 device duration | **165–232 µs/次**（＝隔离值的 100 倍量级） |

**读法**：

* 这个算子的**本体确实很便宜**（排队执行 1.4 µs），早前"内核本体仅 9.5 µs"的结论方向正确；
* **host 侧 56 µs/次是硬的**，7 次/步 ≈ **0.39 ms/步**的 host CPU，且与 batch 无关；
* 服务内 165–232 µs 的 device duration **是争用/调度膨胀**（同一时刻 AI core / HCCW 都在跑），
  不是算法工作量 ⇒ **想省它只能"少发几次"或"挪到不争用的地方"，优化 kernel 本体没有意义**。

## 4. host 侧下发账：33.9 ms/步 vs 步周期 ~28 ms【推断：host-bound】

同一 profile 的 `api_statistic`（Level=acl，host 侧）：

| 项 | 全 profile | 每步 |
|---|---:|---:|
| acl host 合计 | 21 196.8 ms | **33.86 ms/步** |
| 其中 `*GetWorkspaceSize` | 1 286.9 ms | **2.06 ms/步** |
| 其中名字含 `Metadata` 的 | 1 013.3 ms | **1.62 ms/步** |

（步数 = 主模型 gmm1 锚点数 25040 ÷ 40 = **626 步**；profile 跨度 17.69 s。）

**为什么值得记一笔**：每步的 acl host 时间（33.9 ms）**大于**步周期（~28 ms）。
单进程不可能真的忙 120%，所以要么（a）统计含嵌套/多线程重复计入，
要么（b）host 确实在多个线程上并发下发、但**某些线程排队**。
无论哪种，它和 §1 的现象一致：**设备侧 2.88 ms 的空隙更像"等 host"**，
而不是"设备没活干"。

> 待补的直接证据（下一轮做）：把 host 侧 trace（`FRAMEWORK/`）按线程拆开，
> 看 E2E 每步里 host 的 gantt。本轮只有 api_statistic 聚合值，故标【推断】。

## 5. 下一步排序（含可执行实验）

| # | 靶点 | 上界 | 成本 | 备注 |
|---|---|---:|---|---|
| **1** | **metadata 7 次/步 → 更少**（跨 ratio 合并成一个 kernel，或改为 host 计算 + H2D） | **0.6–1.2 ms/步** | 高（要动 vendor 算子 `csrc/attention/sparse_flash_mla_metadata/` 或写 host 参考实现） | 源码在仓库里，可改可重编；tiny 上可先做"逐位一致"的 host 参考实现 |
| **2** | **stream 47 融合**（ViewCopy/Index/Cast/Floor*/SelectV2/Fill 一族，420 个算子） | **0.5–1.6 ms/步** | 中高（有 `_compute_slot_mapping_kernel` 先例） | 与 1 同源：都是"少发几次" |
| **3** | 减少 eager 算子总数（host 调用次数） | 最大，但最难 | 高 | 需要 §4 的 host trace 才能定位"谁在排队" |
| **4** | HcPre A1 重测（必须先清 static kernel 缓存） | 0.12–0.14 ms | 低 | 见 `KERNEL-CACHE-STALE`；重启实例即可做 |
| 5 | 主模型图内（HcPre 3.42 / gmm1 2.66 / MatMulV2 2.34 ms） | 已多次攻过 | 高 | 参考 `LEVERS-R5` |

### 5.1 建议的第一枪（不需要新卡）

**在 tiny 上做 metadata 的 host 参考实现**：

1. 用 §3 的夹具对 `npu_sparse_flash_mla_metadata` 生成 `(输入, 输出)` 对（batch 1–32、各 ratio/模式）；
2. 按 `csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/*.cpp` 的切分逻辑写 Python 实现；
3. **逐位比对**，全通过再谈接到服务里（写进预分配 buffer + H2D，一次 ~10–20 µs）。

若第 3 步成立，7 次 AICPU（0.39 ms host + 1.19 ms 争用膨胀）可以被
**1 次 H2D + 纯 host 计算**替掉，且顺带解掉"metadata 在 AICPU 上崩"的那类问题。

## 6. 复现命令

```bash
# 独占贡献（全部 stream）
python3 ~/tmp/excl.py <mindstudio_profiler_output> 3728.88 3754.47
# 单 stream 的算子级独占
python3 ~/tmp/excl_op.py <mindstudio_profiler_output> 3728.88 3754.47 47
# metadata 隔离基准（tiny 容器，勿在 tp8k5 上跑）
docker cp ~/tmp/meta_bench2.py dsv41-tinyspark:/tmp/ && \
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 dsv41-tinyspark bash -lc "cd /workspace && python /tmp/meta_bench2.py"
```
