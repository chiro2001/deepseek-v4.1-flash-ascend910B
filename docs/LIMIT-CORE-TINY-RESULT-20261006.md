# tiny 上实测 `limit_core_num`：它能做什么、不能做什么（2026-10-06）

> 起因：核对 CANN recipes 后发现官方用 `limit_core_num` 做核预算分区（且注明只支持 A3），
> 而我们在 vllm-ascend 里零使用。本文件是在 **tiny 夹具（a3-21 chips 2/3）** 上的实测结论。
> 全部为【实测】。工具：`tools/tiny_limit_*.py`。**未碰交付实例**（tp8k5 全程在跑）。

## 0. 一句话

**API 可用、量化了核预算→时间的曲线，也找到了一个真实可用的头寸**
（**cube 密集算子对 vector 核预算完全不敏感 ⇒ 可以把它那 48 个 vector 核让出来**）；
**但"两条大计算流并发"本身收益很小甚至为负**（互相争抢），
官方那种"主 16 / 侧 8"的分法只在**两分支在各自预算下耗时相当**的窄区间才赢。

---

## 1. 环境与 API 可用性

```python
torch_npu 2.10.0.post4
torch.npu.npugraph_ex.scope.limit_core_num(op_aicore_num, op_vectorcore_num, stream=None)
```

实测（tiny 容器内）：

```
has npugraph_ex: True / scope: True / limit_core_num: True
签名: (op_aicore_num: int, op_vectorcore_num: int, stream=None)   ← 比官方多一个 stream 参数
get_stream_limit(stream) -> {'cube_core_num': 24, 'vector_core_num': 48}   ← 默认就是满核
```

底层走 `torch_npu._C._npu_set_stream_res_limit(stream_id, device_index, device_type, type, value)`，
`type=0` 是 cube、`type=1` 是 vector。

**两个坑（都踩过）**：

* `limit_core_num(None, None, s)` **不会**变成"不限制"，而是把 `None` 传给
  `set_stream_limit` ⇒ `TypeError: an integer is required`。要"不限制"必须**不进**这个上下文。
* `op_aicore_num=0` 在本栈会触发 `aclnnAdd` 的 inner error。最小要 ≥1。

---

## 2. 核预算 → 时间的定量曲线【实测】

单个 `2048×4096×4096` bf16 GEMM（cube 密集）：

| AIC 预算 | 耗时 | 相对满核 |
|---:|---:|---:|
| 24（满） | 0.220 ms | 1.00× |
| 20 | 0.259 | 1.16× |
| 16 | 0.313 | **1.42×** |
| 12 | 0.424 | 1.92× |
| 8 | 0.620 | 2.81× |
| 4 | 1.235 | 5.59× |

⇒ **限核的代价超线性**（12→8 掉 1/3 的核，时间涨 46%）。这条曲线是判断"值不值得分区"的基础。

---

## 3. ★ 最有价值的发现：cube 密集算子对 **vector 核预算完全不敏感**

同一个 GEMM，在 AIC=24 固定、只改 AIV 预算：

| AIV 预算 | 48 | 32 | 16 | 8 | 4 | 2 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM 耗时 | 0.222 | 0.221 | 0.221 | 0.221 | 0.221 | **0.221** |

**六档完全一样**（差异在噪声内）。

⇒ **cube 密集算子根本不用 vector 核** ⇒ 可以放心把主流的 AIV 预算压到很小，
**把 40+ 个 vector 核"让"给一条纯向量流**，而**不会拖慢主流**。

这条是"零风险的头寸"：与"限 AIC"（1.42× 代价）完全不同性质。

> 反向也测了：**AIV 密集型**（fp32 elementwise，256 MB 足迹）对 AIV 预算敏感但**有饱和点**：
> AIV 48→32 无代价（0.352→0.353），24 → 1.10×，16 → 1.24×。
> ⇒ 48 个 vector 核里约有 **16 个是"多余"的**（至少对这种 elementwise）。

---

## 4. 但"两条大流并发"本身收益很小【实测·负结果】

### 4.1 两个都是 AIC 密集（GEMM ∥ GEMM）

| 侧/主规模比 | 串行 | 并发(不限) | 并发(主16,侧8) |
|---:|---:|---:|---:|
| 0.25 | 0.287 | — | 0.316（**−9%**） |
| 0.50 | 0.347 | — | 0.320（**+8.3%**） |
| 0.75 | 0.386 | — | 0.429（**−10%**） |
| 1.00 | 0.445 | 0.442（−0.6%） | 0.433（**+2.9%**） |

**只有"两分支在各自预算下耗时相当"时才小赢**，其余区间都是负收益。
这与官方文档的警告一致（"核资源若不能形成有效重叠，时间线仍可能串行或产生拖尾"）。

### 4.2 一个 AIC 密集 ∥ 一个 AIV 密集（更接近我们的情形）

（主 = GEMM，侧 = fp32 elementwise 256 MB 足迹）

| 分配（主AIC,主AIV）/（侧AIC,侧AIV） | 总耗时 | vs 串行 |
|---|---:|---:|
| 不限 / 不限 | 0.566 | +1.5% |
| **主满 / 侧 AIV24** | **0.538** | **+6.9%** |
| 主满 / 侧 AIV16 | 0.579 | −0.8% |
| 主AIC16 / 侧AIV16 | 0.665 | −13.4% |
| 主AIC20 / 侧AIV16 | 0.607 | −5.7% |
| 主AIC12 / 侧AIV24 | 0.724 | −20.6% |

### 4.3 为什么并发收益这么小：**HBM 争抢**【推断，有数字支撑】

| 分支 | 字节 | 单独耗时 | 隐含带宽 |
|---|---:|---:|---:|
| GEMM 2048×4096×4096 | 48 MB | 0.221 ms | 217 GB/s |
| elementwise 32M fp32（读2写1） | 384 MB | 0.352 ms | **1,091 GB/s** |
| **并发时合计需求** | | | **1,308 GB/s > 可达 1,182 GB/s** |

⇒ **两个分支单独都不慢，但并发时把 HBM 打爆了** ⇒ 争抢把收益吃光。
（另加空 fork/join 开销 0.022 ms/对。）

> ⚠️ **这一条对实验外推有限制**：我们的真实 decode **HBM 只有 11% 利用率**，
> 所以**真实场景不会像这个微基准那样被带宽卡住**。微基准用的是"大张量"，
> 而真实 decode 是"1969 个小 AIV 算子、总带宽需求很低"。
> ⇒ **微基准的负结果不能直接推翻真实场景，但也不能证明真实场景会赢。**

---

## 5. 结论与下一步

### 5.1 可以确定的

| # | 结论 | 证据 |
|---|---|---|
| 1 | `limit_core_num` 在本栈可用、可量化 | §1、§2 |
| 2 | **限 AIC 代价超线性**（16→1.42×、8→2.81×） | §2 |
| 3 | **cube 密集算子对 AIV 预算完全不敏感**（48→2 都一样） | §3 |
| 4 | **AIV 约有 16 个核是"多余"的**（elementwise 48→32 无代价） | §3 |
| 5 | 两条大计算流并发**收益小且脆弱**（−20% ~ +8%），主因是 HBM 争抢 | §4 |

### 5.2 建议：把它当"零风险试探"，而不是当"确定收益"

**可做**：给现有的 `aux_stream`（`dsv4_dsa_overlap_stream`）加一个 **AIV 预算**，
把主流的 **AIV 压小**（GEMM 类不吃 vector 核，§3 证明无代价），
把让出来的 vector 核配给侧流。**这一步的下行风险几乎为零**（不动 AIC 预算）。

**不该做**：不要照搬官方的"主 16 / 侧 8"—— 那会限 AIC，
在 cube 密集的步骤上直接付 1.42× 代价（§2），而我们的步恰好是 cube 密集的
（AIC 22.05 ms > AIV 14.75 ms）。

**验证方式**：在 tiny 上给 `dsa_v1.py` 的 `aux_stream` 加预算后，
跑 `walk_blocks` 与 `bench_concurrency` A/B，判据是 **`[bneck] hp`（ms/step）** 与带宽采样。
真实 decode 的 HBM 只有 11%，所以微基准的带宽争抢在那里**不该出现**——
如果 tiny 上仍然没有收益，就说明瓶颈不在流编排，而在别处（回到算子数/依赖）。

---

## 6. 复现

```bash
ssh a3-21 'docker cp ~/tmp/limit_verify.py dsv41-tinyspark:/tmp/ && \
  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 limit_verify.py"'   # 核预算曲线
ssh a3-21 'docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 limit_key.py"'    # AIV 不敏感
ssh a3-21 'docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 limit_concur.py"' # 并发收益
```

工具已入仓：`tools/tiny_limit_{verify,key,concur,aic_aiv,sweep}.py`。
