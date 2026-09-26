# CED-PD 的 D 侧启用 static kernel：单变量实验（2026-09-26）

> 起因：溯源历史「≈24ms/step」时发现那一发用的是 `STATIC_KERNEL=1`，
> 而 CED 从第一版起一直是 `=0`（见
> [`CED-PD-HISTORICAL-24MS-PARAMS-20260926.md`](CED-PD-HISTORICAL-24MS-PARAMS-20260926.md)）。
> 本文是把它拿回来的单变量验证。

## 0. 结论

| 项 | 结果 |
|---|---|
| **ms/step（并发4，同为4路）** | **45.42 → 41.07 ms（−4.35，−9.6%）**，回切复现 |
| 单路（batch=1） | 33.57 → **30.05**（−10.5%） |
| 接受长度 A | **不变**（SK=0 中位 2.41 vs SK=1 中位 2.39，5+3 个样本） |
| 正确性 | 144K 四针 **4/4**、1M 四针 **4/4** |
| 端到端（并发4 总吞吐） | 中位 135.0 → **152.6 tok/s**（噪声大，见 §4） |

**判定：采纳方向成立**，但完整 21 项验收矩阵要重跑一遍才算交付（本轮只覆盖四针）。

## 1. 实验设计

口径与既有的 B 臂 profiler **逐项相同**，只改 `STATIC_KERNEL` 一个变量：

```
SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1   V41_CED_ALLOW_DSPARK=1
MAX_SEQS=4  MULTISTREAM=0  DSA_OVERLAP=0  PREFIX=0
2048 prompt / 256 输出 / 并发 4 / ignore_eos / 2 轮热身（不计入）
```

| 臂 | `STATIC_KERNEL` | 会话 |
|---|---|---|
| SK=0 会话1 | 0 | `ced_d4b_draft_graph_0926_123034` |
| SK=1 会话1 | 1 | `ced_d4b_draft_graph_0926_134021` |
| **SK=0 回切** | 0 | `ced_d4b_draft_graph_0926_141209` |
| SK=1 会话2 | 1 | `ced_d4b_draft_graph_0926_143141` |

## 2. ★ 先修判据：`compile start` 是错的判据

第一版检查脚本用「`static kernel compile start` > 0」判断静态核是否生效，
结果在 SK=1 第二次起服时给出**假阴性**（`compile start = 0`），差点误判成静默失效。

真实原因是**编译缓存命中**：`cache/skcache/compile_outputs/static_kernel_cache`
（首次 SK=1 起服时 13:49 建立），第二次不需要重新编译。

四个会话的正确分类（判据 = 是否打印
`enable_static_kernel is enabled, static shape kernel will be used`）：

| 会话 | `static shape kernel will be used` | `compile start` | `torch.compile took` |
|---|---:|---:|---:|
| SK=0 会话1 | **0** | 0 | 0.20 s |
| SK=0 回切 | **0** | 0 | 0.20 s |
| SK=1 会话1 | **1** | 4 | 2.80 s |
| SK=1 会话2 | **1** | **0（缓存命中）** | 1.52 s |

⇒ 判据已改为 ① `static_kernel.py:650` warning == 0、②
`static shape kernel will be used` > 0，并把这个坑写进
`experiments/dspark/check_static_kernel.sh`。

## 3. ms/step：可复现的 −4.4 ms

用 `HcPre` 次数定步（每步 86 次），按**步索引**窗口对齐（避免时间窗偏差）：

| 会话 | 步 28–56 | 步 56–84 | 步 84+ |
|---|---:|---:|---:|
| SK=0 会话1 | 45.42 | 44.86 | 33.57 |
| **SK=0 回切** | **45.50** | **45.27** | — |
| **SK=1** | **41.07** | **40.50** | 30.05 |

**两个互相独立的 SK=0 会话差 < 0.5 ms** ⇒ 排除 session 漂移（历史文档记录过
同配置跨会话差 12% 的先例，所以这一步是必须的）。SK=1 稳定低 **4.4 ms**。

### 3.1 省在哪：**每个 kernel 都便宜一点**

逐算子 A/B（`tools/ced_prof_ab.py`，同为 4 路窗口）：

| 算子 | Δ ms/step |
|---|---:|
| `aclnnQuantMatmulWeightNz` | −0.664 |
| `HcPre` | −0.660 |
| `SparseFlashMla` | −0.492 |
| `aclnnGroupedMatmulWeightNz`（MoE gmm2） | −0.434 |
| `aclnnMatmul` | −0.427 |
| `hcom` / `AivKernel`（通信） | −0.226 / −0.224 |
| `MoeGatingTopKHash` | −0.186 |
| `aclnnScatterNdUpdateSk` | −0.184 |
| `InplacePartialRotaryMul` | −0.154 |
| `HcPost` | −0.129 |
| `QuantLightningIndexerV2` | −0.120 |
| 其余 10 项合计 | ≈−0.6 |
| `RmsNorm` | **+0.106**（唯一变差） |
| **合计** | **−4.43** |

**没有单项大赢，而是普遍性地每项便宜几个百分点** —— 这正是形状特化内核对
每次调用开销的典型作用。

`core/step`（累加 core 时 ÷ step 周期）：两臂分别 **102.2% / 102.8%**
（>100% 说明多流之间有重叠）⇒ **省下来的全是真实设备工作量，仍然没有气泡可挖**。

## 4. A 没有被影响（先否证再确认）

第一轮 SK=1 的客户端基准报出 A@4 = 2.15，而 SK=0 是 2.41/2.45，
看起来像"静态核改变了数值 ⇒ 接受率下降 11%"。这如果成立，
就会正好抵消掉 ms/step 的收益（`ms/token = ms/step ÷ A`）。

补测 4 次重复后否证：

| 臂 | A@conc4 各样本 | 中位 |
|---|---|---:|
| SK=0 | 2.41 / 2.45 / 2.41 | **2.41** |
| SK=1 | 2.43 / 2.36 / 2.20 / 2.42 / 2.15 | **2.39** |

⇒ **A 在噪声内不变**，2.15 是离群样本。这不意外：`A` 的 run 间抖动是
已知现象（`reports/a-basin-and-acceptance-shape.md`），单样本不足以定性。

⇒ 由于 A 不变而 ms/step 降 9.6%，**每 token 时延也降 9.6%**。

## 5. 端到端（并发 4，总吞吐）

| 臂 | 总吞吐各样本 | 中位 |
|---|---|---:|
| SK=0 | 144.0 / 126.6 / 135.0 | **135.0** |
| SK=1 | 83.6 / 155.3 / 150.0 / 158.9 / 141.5 / 155.0 | **152.6** |

方向一致（+13%），但**单次方差很大**（SK=1 有一次 83.6 的离群）。
以 §3 的 ms/step 为准，客户端数字只作旁证。

## 6. 建议

1. **采纳 `STATIC_KERNEL=1`** 作为 CED D 侧的新基线 —— 但先跑完整验收矩阵
   （本轮只验了 144K/1M 四针）。
2. 起服后**必须**用 `check_static_kernel.sh` 核对，判据是
   `static shape kernel will be used`（不是 `compile start`）。
3. 顺手确认 P 侧（`experiments` 之外的 `serve_a3_ced_pd.sh` 默认
   `STATIC_KERNEL=${STATIC_KERNEL:-0}`）是否也该改 —— 本轮只测了 D。
4. 下一项待验：`SP_TOKENS=5`（历史那一发用的值，见溯源文档 §5）。

## 7. 复现命令

```bash
# 起服（单变量）
ARM=draft_graph STATIC_KERNEL_ARM=1 V41_SLOT_MAP_FUSED=on bash launch_d_dspark.sh
bash check_static_kernel.sh <run_dir>          # 必须 ✓

# 正确性
python3 tools/ced_pd_acceptance.py --base-url http://127.0.0.1:18992 \
  --tokenize-url http://127.0.0.1:18990 --model deepseek-v41-ced-pd \
  --corpus data/hongloumeng.txt --mode needle --context-tokens 144000 \
  --out results/sk1_needle144k.json

# ms/step（HcPre 定步：SPEC=7 → 86/步）
python3 step_period.py <...>/kernel_details.csv 86 "label"
# 两臂对齐（按步索引窗口）
python3 cmp_steps.py a.csv 86 "SK=0" b.csv 86 "SK=1"
```
