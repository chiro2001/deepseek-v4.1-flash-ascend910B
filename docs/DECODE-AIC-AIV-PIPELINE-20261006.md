# ★ decode 串行链的算子/资源分解，以及 AIC-AIV 流水的可行性与上界

> 起因：用户追问 ——「串行链上具体是哪些算子、各占什么资源？能否用 910 的 AIC/AIV 分离特性做流水？」
> 数据：`results/armF_r6_base/prof/...rank0.../kernel_details.csv`（交付口径、纯 decode，
> 72 步稳态，profile 步长 **40.02 ms**，服务内实测 **24.59 ms**，msprof 膨胀 1.63×）。
> 全部为【实测】；工具 `tools/prof_chain_core.py`、`tools/prof_chain_blocks.py`。

## 0. 一句话

**AIC 忙 22.05 ms/步、AIV 忙 14.75 ms/步，而两者的重叠只有 3.50 ms** ——
即 **AIC 有 45% 的时间闲着、AIV 有 63% 闲着，却仍在同一条流上排队**。
⇒ 完美流水的理论上界是 **max(22.05, 14.75) = 22.05 ms，即 1.82×**；
把通信也藏进去同样是 22.05（通信 4.21 ms 同样零重叠）。

---

## 1. 资源总账（每步 = 40.02 ms profile）

| 类别（`Accelerator Core`） | 算子/步 | 时长合计 | **并集 busy** | 中位 | 占步长 |
|---|---:|---:|---:|---:|---:|
| `MIX_AIC`（混合核的 cube 相位） | 409.1 | 16.981 | **16.981** | 35.9 µs | 42.4% |
| **`AI_VECTOR_CORE`（纯 AIV）** | **1969.5** | 12.927 | **12.805** | **4.7 µs** | 32.0% |
| `COMMUNICATION`（HCCL + AivKernel） | 183.0 | 8.329 | 4.215 | 29.1 µs | 10.5% |
| `AI_CORE`（纯 AIC） | 200.0 | 5.068 | 5.066 | 18.4 µs | 12.7% |
| `AI_CPU` | 14.0 | 2.507 | 2.066 | 194.1 µs | 5.2% |
| `MIX_AIV`（混合核的 vector 相位） | 112.0 | 1.945 | 1.945 | 17.3 µs | 4.9% |

**资源并集**：

| 资源 | busy/步 | 占步长 | 按块数的核利用率 |
|---|---:|---:|---:|
| **AIC**（`AI_CORE` + `MIX_AIC`） | **22.05 ms** | **55%** | **50.8%**（488.3 / 960 核·ms） |
| **AIV**（`AI_VECTOR_CORE` + `MIX_AIV`） | **14.75 ms** | **37%** | **28.2%**（541.7 / 1921 核·ms） |
| 通信 | 4.21 ms | 11% | — |
| AICPU | 2.07 ms | 5% | — |

> 核利用率的算法：Σ(算子时长 × Block Num) ÷ (核数 × 步长)。A3 每 die **24 cube / 48 vector**。

## 2. 重叠矩阵：几乎全是串行【实测】

| ∩ (ms/步) | AIC纯 | AIC混合 | AIV纯 | AIV混合 | 通信 | AICPU |
|---|---:|---:|---:|---:|---:|---:|
| **AIC纯** | 5.070 | 0.000 | 1.046 | 0.000 | **0.000** | 0.063 |
| **AIC混合** | 0.000 | 16.981 | 2.449 | 0.000 | **0.000** | 0.273 |
| AIV纯 | 1.046 | 2.449 | 13.009 | 0.000 | 0.028 | 0.866 |
| AIV混合 | 0.000 | 0.000 | 0.000 | 1.945 | 0.000 | 0.019 |
| 通信 | 0.000 | 0.000 | 0.028 | 0.000 | 8.329 | 0.188 |

三条读法：

1. **AIC ∩ AIV 只有 3.50 ms**（1.046 + 2.449）⇒ 两大资源 **76% 的时间互不相干地排队**；
2. **通信 ∩ AIC = 0.000、通信 ∩ AIV纯 = 0.028** ⇒ **集合通信完全没有和计算重叠**；
3. **`MIX_AIC` ∩ `MIX_AIV` = 0.000** ⇒ 连**同一个混合核内部**，cube 相位与 vector 相位
   也是**先后执行**，不是并行。

> 唯一的例外：`COMMUNICATION` 类内部 sum 8.329 / union 4.215，**差值 4.114 恰好 = `AivKernel` 的 4.114**
> ⇒ **AivKernel（engram wkv 的 vector 核）确实与 HCCL 集合通信并行**。
> 这条很重要：**"把 vector 工作藏进通信/计算"的机制在本栈已经跑通，只是覆盖面很小。**

## 3. 串行链的具体算子（每层 × 40 层 + engram + 通信）

从主计算流（stream 109，1167 算子/步）的相邻算子对 + 资源分类重建：

```
 ① HcPre                       AIC  24块  37.8 µs   ← hyper-connection 前投影
 ② RmsNorm                     AIV  48块  13.4 µs
 ③ DynamicQuant                AIV  14块   5.5 µs
 ④ QuantBatchMatmulV3          AIC  16块  15.3 µs   ← attention qkv 投影
 ⑤ InplacePartialRotaryMul     AIV  48块   8.9 µs   ← RoPE
 ⑥ SparseFlashMla              AIC  24块  56.4 µs   ← 稀疏注意力
      └ SparseFlashMlaMetadata AICPU 48块 258.0 µs（3 次/步，独立 eager 流）
 ⑦ MatMulV2                    AIC  23块  19.1 µs   ← o_proj
 ⑧ HcPost                      AIV  48块  16.7 µs
 ⑨ MoeGatingTopKHash           AIV  48块  13.7 µs
 ⑩ MoeInitRoutingV3            AIV  48块  14.5 µs
 ⑪ GroupedMatmulSwigluQuant    AIC  24块 119.3 µs   ← MoE w1/w3（**单步最贵算子**）
 ⑫ GroupedMatmul               AIC  24块  62.2 µs   ← MoE w2
 ⑬ DequantSwigluQuant          AIV   3块  12.0 µs
 ⑭ MoeTokenUnpermute           AIV  48块  10.8 µs
 ⑮ allreduce                   COMM          ← 40 层 × 2 次 = 80 次/步
```

步级汇总（每步）：

| 算子 | 资源 | 个数/步 | ms/步 | 块数 |
|---|---|---:|---:|---:|
| `GroupedMatmulSwigluQuant`（w1/w3） | AIC | 43.0 | 5.185 | 24 |
| `AivKernel`（engram wkv all_gather） | COMM/AIV | 91.0 | 4.114 | 8 |
| `QuantBatchMatmulV3` | AIC | 226.0 | 3.578 | 16 |
| `MatMulV2` | AIC | 111.0 | 3.425 | 23 |
| `HcPre` | **AIC** | 86.0 | 3.279 | 24 |
| `GroupedMatmul`（w2） | AIC | 43.0 | 2.740 | 24 |
| `SparseFlashMla` | AIC | 40.0 | 2.200 | 24 |
| `RmsNorm` | AIV | 140.0 | 2.012 | 48 |
| `HcPost` | **AIV** | 86.0 | 1.401 | 48 |
| `InplacePartialRotaryMul` | AIV | 145.0 | 1.285 | 48 |
| `ScatterNdUpdateSk` | AIV | 58.0 | 1.180 | 48 |
| `DynamicQuantV2` | AIV | 141.0 | 0.747 | 14 |

> **`HcPre`(AIC) 与 `HcPost`(AIV) 是一对**：相邻出现 4070 次/窗口 ⇒ 天然的流水候选对。

## 4. AIV 为什么有 1969 个算子：40% 只用 1 个块

| AIV 块数 | 算子个数 | 占比 | 时长合计 |
|---:|---:|---:|---:|
| **1 块** | 56969 | **40.2%** | 1.3 ms |
| 2/4/6 块 | 7743 | 5.4% | 0.5 ms |
| 8/12/16/24 块 | 12169 | 8.6% | 1.1 ms |
| **48 块** | 42510 | **30.0%** | **7.4 ms** |

只用 1 块的那 791 个/步（56969/72）几乎全是 **1.3 µs 的标量/掩码/类型转换**：

| 算子 | 个数/步 | 中位 | 块数 |
|---|---:|---:|---:|
| `aclnnInplaceCopy_CastAiCore_Cast` | 172.5 | 1.3 µs | 1 |
| `aclnnInplaceFillScalar_Fill` | 60.0 | 1.3 µs | 1 |
| `aclnnInplaceMaskedFillScalar` | 50.0 | 1.4 µs | 1 |
| `aclnnSWhere_SelectV2` | 48.0 | 1.9 µs | 1 |
| `aclnnLtTensor_Less` | 46.0 | 1.4 µs | 1 |
| `aclnnNeg_NegAiCore` | 45.0 | 1.5 µs | 3 |
| `aclnnAbs_AbsAiCore` | 45.0 | 1.3 µs | 1 |
| `aclnnBitwiseOrTensor_LogicalOr` | 44.0 | 1.3 µs | 1 |
| `aclnnGeTensor_GreaterEqual` | 43.0 | 1.3 µs | 1 |
| `aclnnDivMods_Cast` | 36.0 | 1.3 µs | 1 |

⇒ 这些**单个只值 1.3 µs**（时长合计仅 1.3 ms/步），但**数量占 AIV 的 40%**。
它们的真实代价是**每个都要占一次算子调度槽位**（2888 个/步里它们占 ~791）。

## 5. ★ 流水可行性与上界【实测 + 推断】

### 5.1 上界

| 方案 | 步长 | 加速 |
|---|---:|---:|
| 现状 | 40.02 ms | 1.00× |
| **AIV 全藏进 AIC**（AIV 14.75 全部重叠） | 40.02 − 11.25 = **28.8 ms** | **1.39×** |
| 再把通信（4.21）也藏掉 | **24.6 ms** | **1.63×** |
| **理论上界**（所有资源完美重叠）= max(AIC, AIV) | **22.05 ms** | **1.82×** |

（AIV 已重叠 3.50，所以"可新增的隐藏"是 14.75 − 3.50 = 11.25 ms。）

### 5.2 三条可行路径

| # | 路径 | 可回收 | 难度 | 依据 |
|---|---|---:|---|---|
| **A** | **把 AIV 工作藏进通信/相邻 AIC** | ≤4.2 ms | **低** | **已有成功先例**：`AivKernel` 4.11 ms 已完全与 HCCL 并行（§2 注）；`MULTISTREAM=1` / `DSA_OVERLAP=1` 机制已存在，只是覆盖面小 |
| **B** | **跨层软件流水**（层 L 的 AIV 与层 L−1 的 AIC 重叠） | ≤11.3 ms | **高** | 需要把 40 层循环重构成流水；同层内是真依赖（RmsNorm→Quant→Matmul），只有跨层才有空间 |
| **C** | **消掉 1 块 AIV 小算子**（Cast/Fill/MaskedFill/Select…） | ~1–3 ms | 中 | 791 个/步 × 1.3 µs；靠融合——但它们分散在 MoE 路由/掩码逻辑里 |

### 5.3 为什么现在做不到：两个硬约束

1. **同层内是真实数据依赖**：`RmsNorm(AIV) → DynamicQuant(AIV) → QuantMatmul(AIC)` 必须串行；
   `HcPost(AIV) → HcPre(AIC)` 同理（4070 次/窗口的相邻对）。
   ⇒ 层内可压的只有"不相邻的 AIV 对 AIC"（例如 `HcPost(L)` 与 `GroupedMatmul(L)` 无依赖？需逐层核实）。
2. **整个前向在一张 NPUGraph 里、单流下发** ⇒ 编译器不做跨算子流分离，就一定是串行。
   `MIX` 核本可让 cube/vector 并行，但实测 **`MIX_AIC` ∩ `MIX_AIV` = 0**
   ⇒ 当前 MIX 核内部也是相位串行的。

### 5.4 建议的下一步（按性价比）

1. **先做 A（低风险、有先例）**：核对 `MULTISTREAM` / `DSA_OVERLAP` 的现有覆盖点，
   找出**还有哪些 AIV 算子与其前驱 AIC 无数据依赖**，把它们挂到已有的侧流上。
   判据：`通信 ∩ AIC = 0` 说明侧流机制**没有**用在集合通信上 ⇒ 这里是空白。
2. **再评估 B**：B 的收益最大（11.3 ms），但需要跨层重构；
   可以先用**并发 8（M=48）**验证"token 维切分后的跨层流水"是否可行（M=6 切不动）。
3. **C 作为顺带**：那 791 个 1.3 µs 算子如果能并入邻居，除了省 ~1 ms，还能减少 2888 里的 27%。

## 6. 复现

```bash
PROF=~/cedpd-repo/results/armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1434_20261004195431495_ascend_pt
python3 tools/prof_chain_core.py   $PROF/ASCEND_PROFILER_OUTPUT   # 资源并集 + 重叠矩阵
python3 tools/prof_chain_blocks.py $PROF/ASCEND_PROFILER_OUTPUT   # 块数 → 核利用率
python3 tools/prof_aiv_tail.py     $PROF/ASCEND_PROFILER_OUTPUT   # AIV 构成与块数分布
```

> 需 pandas/numpy ⇒ 用交付镜像跑（宿主 python3 没有 pandas）：
> `docker run --rm --entrypoint python3 -v $PROF:/p:ro -v ~/tmp:/t:ro <image> /t/<script>.py /p/ASCEND_PROFILER_OUTPUT`
