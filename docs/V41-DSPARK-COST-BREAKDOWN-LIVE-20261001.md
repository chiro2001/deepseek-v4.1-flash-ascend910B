# DSpark 入图开销拆解（2026-10-01 实测，TP8+DCP8）

> 目标：找出 DSpark 入图后 decode `ms/step` 增量落在哪，找可优化项。
> 方法：同配置 SPEC=0 / SPEC=1 各采一次 profile，归一化到 per-step 差分。

## 0. 两个口径（必须分清）

| 数据源 | 覆盖范围 | SPEC=0 | SPEC=1 | Δ |
|---|---|---:|---:|---:|
| `op_statistic`（**纯计算**，不含通信） | 69→82 种算子 | 18778.5 | 27655.2 | **+8876.7** |
| `op_summary` 里的 `hcom_*`（**通信**） | allReduce+allGather | 9612.4 | 11601.9 | **+1989.5** |
| **设备合计** | | **28390.9** | **39257.1** | **+10866.2** |
| **端到端墙钟** | | **29.35 ms** | **41.4 ms** | **+12.05 ms** |

步数：`HcPre` 次数 ÷ 每步次数（SPEC=0 为 80、SPEC=1 为 86）。
墙钟 SPEC=1 = `ms/token 13.15 × A 3.15 = 41.4 ms`。

**设备时间解释了墙钟的 95%**（28391/29350、39257/41400）⇒ 没有大的隐藏 host 开销。

## 1. 参照基线：A2 生产的历史数字

| 项 | 值 | 出处 |
|---|---|---|
| `SPEC=0` | 24.35 ms/step（A=1，41.06 tok/s） | `docs/CED-PD-DYNAMIC-SPEC-20260926.md` §11.4 |
| `SPEC=1 DRAFT_GRAPH=1` | 32.68 ms/step（A=3.10，**94.79 tok/s**） | 同上 |
| ⇒ DSpark 代价 | **ms/step +34.2%（+8.33）**，换来 ms/token ÷2.31 | |

⚠️ 那是 **CED-PD 拓扑（无 DCP）**。本轮在 **DCP8** 上重测，得到 **+12.05 ms/step**。

## 2. ★★ 反直觉结论 1：target M=1→8 几乎不花钱（证伪既有假设）

`HcPre` 的 Input Shapes（决定性证据）：

| | SPEC=0 | SPEC=1 |
|---|---|---|
| decode 行数 | **1** | **8**（= 1+K = 1+7） |
| 单次耗时 | **28.14 µs** | **32.26 µs** |
| 涨幅 | | **仅 +14.6%** |

同理 `SparseFlashMla`：次数 78→78，耗时 1723.8→1855.5（**+7.6%**）。

⇒ **M=8 是"权重带宽 bound"，多 7 行几乎免费。** "target M=1→8 是主要开销"不成立。

## 3. 纯计算增量明细（`op_statistic`，>100 µs/step）

| OP | S0 次/步 | S1 次/步 | S0 µs/步 | S1 µs/步 | **Δ** |
|---|---:|---:|---:|---:|---:|
| GroupedMatmulSwigluQuantV2（MoE gmm1） | 40.00 | 43.00 | 2100.0 | 3336.9 | **+1236.9** |
| MatMulV2 | 94.23 | 109.53 | 1681.9 | 2302.0 | +620.1 |
| HcPre | 80.00 | 86.00 | 2289.2 | 2889.5 | +600.3 |
| **Transpose** | 0.38 | **78.99** | 11.8 | 462.7 | **+450.9** |
| GroupedMatmul（MoE gmm2） | 40.00 | 43.00 | 1330.9 | 1718.1 | +387.2 |
| QuantBatchMatmulV3 | 208.00 | 226.00 | 2022.7 | 2350.8 | +328.1 |
| Cast | 337.99 | 414.94 | 425.1 | 722.9 | +297.8 |
| DynamicQuant | 132.00 | 141.03 | 295.6 | 571.6 | +276.0 |
| Add | 71.00 | 85.05 | 116.1 | 382.1 | +266.0 |
| ViewCopy | 75.75 | 106.11 | 344.3 | 604.5 | +260.2 |
| **SparseAttnSharedkv**（draft attention） | **0.00** | **3.00** | 0.0 | 255.4 | **+255.4** |
| **SparseAttnSharedkvMetadata** | **0.00** | **2.00** | 0.0 | 231.1 | **+231.1** |
| Mul | 157.99 | 162.99 | 334.6 | 564.0 | +229.4 |
| InplacePartialRotaryMul | 136.00 | 145.00 | 365.4 | 592.7 | +227.3 |
| RmsNorm | 129.00 | 140.03 | 633.3 | 854.0 | +220.7 |
| DequantSwigluQuant | 40.00 | 43.00 | 243.0 | 461.9 | +218.9 |
| HcPost | 80.00 | 86.00 | 522.7 | 707.1 | +184.4 |
| ScatterNdUpdateSk | 52.00 | 58.00 | 211.2 | 392.4 | +181.3 |
| **Index** | 3.00 | **19.87** | 27.5 | 202.3 | **+174.8** |
| **Slice** | 38.38 | **124.99** | 63.6 | 225.0 | **+161.4** |
| Sub | 162.00 | 172.06 | 218.1 | 375.8 | +157.6 |
| MoeInitRoutingV3 | 40.00 | 43.00 | 385.3 | 520.6 | +135.3 |
| **ArgMaxV2** | 1.00 | **8.99** | 16.4 | 149.5 | **+133.1** |
| SparseFlashMla | 78.00 | 78.00 | 1723.8 | 1855.5 | +131.7 |
| allreduceAicpuKernel | 0.19 | 0.44 | 97.1 | 216.5 | +119.4 |
| GatherV3 | 6.00 | 36.00 | 16.2 | 126.2 | +110.0 |
| …（其余 <100） | | | | | |
| **>100 µs/step 合计** | | | | | **+8559.6** |
| **设备计算总增量** | | | | | **+8876.7** |

### 3.1 ★ `Transpose` 的来源（+450.9 µs/step）

`_v41_dcp_gather_heads`（`dsa_v41.py:1913/1922`）里的**两次转置**：

```python
q_t = q.transpose(0, 1).contiguous()       # [T,8,512] -> [8,T,512]
...
return gathered.transpose(0, 1).contiguous()  # [64,T,512] -> [T,64,512]
```

实测 shape 完全对上：`8,8,512`（3268 次）与 `64,8,512`（3268 次），
**3268/87 ≈ 37.6 次/步 ≈ 层数** ⇒ 每层 2 次转置。

**优化机会**：纯拷贝、无计算 —— 可考虑改用 `all_gather_into_tensor` 直接产出
`[T,8,8,512]` 布局后一次 reshape，把 2 次转置降到 1 次（省 ~225 µs/step）。

### 3.2 ★ 新增的 draft 专用算子（合计约 +660 µs/step）

| OP | 次数/步 | µs/步 |
|---|---:|---:|
| `SparseAttnSharedkv` | 3.00（= 3 draft 层） | 255.4 |
| `SparseAttnSharedkvMetadata` | 2.00 | 231.1 |
| `Index` | 3.00 → 19.87 | +174.8 |
| `ArgMaxV2` | 1.00 → 8.99 | +133.1 |
| `copy_and_expand_dflash_and_dsp` | 0 → 1 | +39.4 |

`Index`/`Slice`/`GatherV3`/`IndexCheck` 的暴增来自 **draft 的 SWA 索引构建**
（`build_dspark_swa_indices`），是大量小算子。

## 4. ★★★★ 通信增量明细（+1989.5 µs/step）

| 项 | SPEC=0 | SPEC=1 | Δ |
|---|---:|---:|---:|
| `hcom_allReduce` 总 | 8106.1 | 9873.0 | **+1766.9** |
| `hcom_allGather` 总 | 1506.3 | 1728.9 | +222.6 |

### 4.1 按 group 拆（最关键的发现）

| group | 含义 | S0 µs/步 | S1 µs/步 | Δ |
|---|---|---:|---:|---:|
| `gid=503` | MoE/TP | 3060 | 3172 | +112 |
| `gid=097` | **DCP merge** | 1117 | 1982 | **+865** |

### 4.2 `gid=097` 的 buffer 级拆解

| 项目 | SPEC=0 | SPEC=1 |
|---|---:|---:|
| distinct k（buffer 数） | **199** | **86**（−57%） |
| 每步调用总次数 | 74.1 | 74.3（**几乎不变**） |
| k=1 大 buffer 次数/步 | 0.76 | **1.75** |
| **其它 k 单次耗时** | **13.0 µs** | **21.7 µs（+67%）** |

⇒ **调用次数没变，是每个 buffer 的数据量变大了。**

## 5. ★★★ 通信变慢的根因（已定位到代码）

`dsa_v41.py:805-817` 的注释自己写着：

> 「瓶颈是**集合通信的次数（延迟）**，不是带宽（**T=1 时单次才 ~128 KB**）」

而 DSpark 把 target 的 T 从 **1** 提到 **8**：

| | T | 打包张量 `[T,H,W=640]` fp32 | 字节 |
|---|---:|---|---:|
| SPEC=0 | 1 | `1×64×640` | **160 KB** |
| SPEC=1 | 8 | `8×64×640` | **1280 KB** |

⇒ **数据量 ×8**，从"延迟域"进入"带宽域"。
实测有效带宽 ≈ **98 GB/s**（2.24 MB 搬运 ÷ 21.7 µs）。

**而每个 rank 归约后只需要自己的 8 个 head（`o_proj` 是 TP 切的）
⇒ all_reduce 让每 rank 收到 64 head 的和，87.5% 是白拿的。**

## 6. 优化靶点清单

| # | 靶点 | 预期收益 | 状态 |
|---|---|---:|---|
| **1** | **DCP merge 的 reduce_scatter**（只收 1/8 head） | ~600 µs/step | 实现已有，**首次实测 A 掉到 1.00**（见 §7） |
| 2 | `_v41_dcp_gather_heads` 的 2 次 Transpose → 1 次 | ~225 µs/step | 未做 |
| 3 | draft SWA 索引构建的小算子（Index/Slice/GatherV3/ArgMaxV2） | ~500 µs/step | 未做 |
| 4 | AscendC 融合 merge kernel（`V41_DCP_MERGE_KERNEL=1`） | tiny 上 +0.1%（无收益） | DCP=2 测不出，需 DCP=8 |

## 7. `V41_DCP_RS_MERGE` 的实测结论

| 环境 | 结果 |
|---|---|
| tiny（DCP=2） | 36.49 vs 基线 36.24 → **+0.7%，无收益** |
| **TP8（DCP=8）** | **A 从 2.5–3.0 掉到 1.00（所有草稿被拒）⇒ 正确性回归** |

### 7.1 根因（已定位）

RS 分支返回的 `_pack` 是**非连续视图**：

```python
_pack = _rs_out.view(_rows, T, _W).permute(1, 0, 2)   # [T,8,640] 非连续
head_slice = None
```

而下游立刻做逐元素减法/除法：
```python
scaled = _pack[..., :_out_dim] - _onum
wsum   = _pack[..., _out_dim:_out_dim+1] - dcp*_keep
```

**同一文件的 `V41-DENFIX` 注释精确记录过这个失败模式**：
> 设备侧那次「`[T,H,1]` 视图 + 标量减法」**没有读到真实数据**（读到的等价于 padding 的 0）

代码库另有三处同类记录（`PACKDIRECT-ABORT` / `SUBALPHA-ABORT` / `contigw`）
⇒ **「非连续视图 + 逐元素算子」在本平台一律不可信。**

### 7.2 修法

permute 之后加 `.contiguous()`（`[V41-RSCONTIG]`）。代价一次 164 KB 拷贝，
远小于 reduce_scatter 省下的搬运（2.24 MB → 1.15 MB）。

**验证结果见下一节。**

## 8. 复现

```bash
# 两臂起服
env SPEC=0 SP_TOKENS=5 DRAFT_GRAPH=0 DCP=8 ... STAMP=<t0> bash ~/dcp_stage_capacity.sh
env SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8 ... STAMP=<t1> bash ~/dcp_stage_capacity.sh

# 采集 + 导出 + 解析（大文件不出容器）
bash ~/tmp/prof_cycle.sh 19210 <run> 200
bash ~/tmp/export_and_parse.sh dsv41-dspark8 <prof_root> <ts> <tag>

# 严格对比（op_statistic 是小文件）
python3 ~/tmp/cmp_stat.py <s0.csv> <s1.csv> <s0_steps> <s1_steps>
python3 ~/tmp/shape_probe.py <op_summary.csv> <OP> --field in
python3 ~/tmp/by_stream.py <op_summary.csv> --op hcom_
python3 ~/tmp/gid_timeline.py <op_summary.csv> <gid>
```
