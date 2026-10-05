# decode 步的三段式分解：计算 / allreduce / 尾部（2026-10-05）

> 数据：`results/armF_r6_base/prof`（交付口径 devidx=1，**纯 decode**：该 profile 里
> stream 146 的 gmm1 **全部 M=36**，58 个捕获步，无 prefill 混入）。
> 工具：`tools/stream_gap_attrib.py`（新增）、`tools/step_gap_detail.py`。
> 标注：【实测】/【推断】。

## 0. 一句话

一个 25.96 ms 的 decode 步可以拆成三块：

| 段 | 时长 | 内容 |
|---|---:|---|
| **主图计算** | ~16.4 ms | 40 层的前向（其中 80 次 allreduce **完全暴露**） |
| **allreduce 等待** | **2.22 ms** | 主图流里 80 个空隙 = `hcom_allReduce_` 自身时长（事件开销仅 ~1 µs） |
| **尾部** | **~5.5 ms** | 采样 → draft → metadata（主图流此时**完全空闲**） |
| 其它（重叠覆盖） | ~1.8 ms | |

⇒ 两个明确、可量化的靶点：**allreduce 2.22 ms（8.6%）** 与 **尾部 5.5 ms（21%）**。

## 1. 方法：看"**这条流自己被卡住**"而不是"全系统真空闲"

`idle_who.py` 那套口径看的是"全系统无任务"（只有 0.15 ms/步），会**漏掉**"主图在等 allreduce
而别的流在跑"这一类。本轮的判据换成**单流内部空隙**：

```
该流内空隙（≥5 µs）合计 = 8.429 ms/步（占 25.96 ms 的 32.5%）
```

按 (前驱 → 后继) 聚合，前三名是：

| 空隙 | 次数/步 | ms/步 | 真身 |
|---|---:|---:|---|
| **`Add` → `HcPost`** | 40 | **1.524** | **`hcom_allReduce_`**（MoE 后的 row-parallel 归约） |
| **`MatMulV2` → `HcPost`** | 40 | **0.779** | **`hcom_allReduce_`**（attention o_proj 后的归约） |
| `QuantBatchMatmulV3` → `InplacePartialRotaryMul` | 40 | 0.308 | KV 路径的等待 |
| `DynamicQuant` → `QuantBatchMatmulV3` | 34 | 0.207 | 量化→GEMM 的等待 |
| **`RmsNorm` → `Cast`** | 1 | **5.461** | **尾部**（见 §3） |

**逐个空隙现场核对**（`Add → HcPost`）：空隙 93.4 µs，其中 `hcom_allReduce_` 占 92.46 µs，
`HcPost` 在 allreduce 结束的下一个时刻开始 ⇒ **事件开销只有 ~1 µs，allreduce 完全暴露**。

## 2. allreduce：2.22 ms/步，80 次/步，全在关键路径上

| 量 | 值 |
|---|---|
| 次数 | **81/步**（层内 2 × 40 + 1） |
| 设备时长合计 | **2.22 ms/步**（128.65 ms / 58 步） |
| 单次分布 | p50 **16.8 µs**；9.9 次/步 <10 µs、39.8 次/步 10–20 µs、**12 次/步 >50 µs** |
| 关键路径暴露 | **≈ 全部**（空隙时长 = allreduce 时长 + ~1 µs） |

### 2.1 为什么它跑不掉（已排除的路径）

| 路径 | 判定 | 依据 |
|---|---|---|
| `FUSED_MC2=1` / `MC2=1`（把通信融进 GEMM） | **实测负结果** | R5：N=1 −3.8%、N=4 −2.9%、N=16 −2.7% |
| `fuse_gemm_comms` pass | 同上 | 就是 MC2 |
| `fuse_allreduce_rms` pass | **本栈没实现** | `vllm_ascend/` 里只有 `fuse_norm_quant` |
| `enable_sp`（序列并行，把 allreduce 换成 reduce-scatter） | **不适用** | 消息只有 60 KB，是延迟受限；换成两个集合操作只会更慢 |
| custom allreduce（小消息专用实现） | **Ascend 上不可用** | vLLM 的 custom_ar 需要 `ops.meta_size()`（CUDA 专有） |
| 减少 allreduce 次数 | 不可行 | 每层 2 次由张量并行语义决定 |

### 2.2 一个方法论副产品：隔离支架测不到设备延迟

用 `torchrun` 直接跑 8 卡 allreduce（`tools/hccl_allreduce_latency.py`）得到
**112 µs/次（60 KB，流水化）**，远高于服务里的 16.8 µs —— 因为那测的是**主机下发**
（Python + torch 派发 ≈ 100 µs/次），而服务里的 allreduce **在图内**由设备运行时发射。
⇒ **要测设备侧集合通信延迟，必须放进 ACLGraph 里**；直接用 Python 循环测会得到 6× 偏高的数。

## 3. 尾部 5.5 ms：主图完全空闲，串行跑「采样 → draft → metadata」

按 0.5 ms 分桶看单步内各流的忙碌轮廓（`step_profile2.py`）：

```
t 0.0 – 19.5 ms : stream 146（主图）占空 ~80%，allreduce/共享专家/KV 交替填充
t 19.5 – 21.0   : s146 = 0；s47（采样链）在跑
t 21.0 – 23.5   : s146 = 0；s142（draft 3 层）在跑
t 23.5 – 25.9   : s146 = 0；s35（metadata 7 次）在跑
t 25.0 起        : s146 重新开始（下一层的 HcPre）
```

尾部里还有两段**全系统真空闲**：`20.87→21.48`（0.6 ms）与 `23.22→24.38`（**1.16 ms**）。
后者正是之前定位的 **24 组 `Fill+ViewCopy`**（0.19 ms 设备时间）所在位置 ⇒ 
【推断】**这 1 ms 是主机侧串行下发/等待**。

**注意**：PGO（换 libpython）实测**中性偏负**、metadata 主机注入 1 ms 也**零影响**
⇒ 这段空闲**不是 Python 字节码开销**，更像**一次 D2H 同步的往返延迟**
（等待设备把"本轮接受几个 token"回传后才能组下一批）。
要消除它需要 vLLM 层的异步调度，属大工程。

## 4. ★ 本轮找到并已实现的一个新杠杆：engram gate 的 `wkv` 是**复制的**

在"最大的单个算子"清单里发现：

| 形状 | 次数/步 | p50 | 合计 |
|---|---:|---:|---:|
| **`MatMulV2 "6,6144;25600,6144"`** | **2** | **355 µs**（max 447） | **0.709 ms/步（2.7%）** |

它是 `layer.engram.wkv`：`nn.Linear(6144, 25600)`，权重 **315 MB**，而 M 只有 6
⇒ 纯**权重载入**受限（隔离实测 254 µs @1237 GB/s；服务内 355–447 µs 是争用放大）。

**关键事实**：它是**普通 `nn.Linear`（每卡复制全量）**，而 TP=8 的输入在 8 卡上完全相同
⇒ 8 张卡各自读了同一份 315 MB，**纯浪费**。

| 隔离实测（tiny，单卡） | 权重 | 时长 | 有效带宽 |
|---|---:|---:|---:|
| 全量（现状） | 314.6 MB | **254.2 µs** | 1237 GB/s |
| **1/8 输出分片** | 39.3 MB | **24.4 µs** | 1609 GB/s |
| 1/8 输入分片 | 39.3 MB | 19.8 µs | 1988 GB/s |

### 4.1 改法（已实现，逐位不变）

按**输出维**分片 + `all_gather` 拼回：

* 每卡只算 `[6, 6144] × [6144, 3200]` ⇒ 权重 39 MB、~24 µs；
* 再把 `[6, 3200]` 沿最后一维 all_gather 成 `[6, 25600]`（每卡 38 KB）。

**数值逐位不变的理由**：每个输出元素仍由**一次 matmul** 算出，gather 只搬字节 ——
与"输入分片 + all_reduce"（会改累加顺序）不同，这条路径**零数值风险**。

| 预估 | 现在 | 改后 |
|---|---:|---:|
| 每层 wkv（服务内） | 355–447 µs | ~25 µs + all_gather（~40–60 µs，按本文 §2 的集合通信量级估） |
| 2 层/步 | 0.71–0.89 ms | ~0.13–0.17 ms |
| **净省** | | **≈0.55–0.7 ms/步（2.1–2.7%）** |

开关：`V41_ENGRAM_WKV_TP=1`（默认 0）。

## 5. 复现

```bash
# ① 单流内部空隙归因（本轮的核心判据）
python3 tools/stream_gap_attrib.py <profdir> 146 16 0.005 146
# ② 在最小的窗口里看"空隙被谁占着"
python3 tools/step_gap_detail.py <profdir> <n_step> 0.2
# ③ 找最大单算子
python3 tools/op_profile_stats.py <profdir> 58 146
# ④ 8 卡 allreduce 延迟（注意：这测的是**主机下发**，不是设备侧）
torchrun --nproc_per_node=8 tools/hccl_allreduce_latency.py 200
```
