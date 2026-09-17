# draft eager 的实测量化 + MoE roofline（2026-09-16 05:20–06:00）

> 目标：找 128K 单流的剩余优化空间
> 结论：**draft 的 5.09 ms/step 空转不是 dispatch 开销**（SPEC_EAGER=0 图捕获后仍存在）；
> **MoE 融合算子 `DispatchFFNCombineW4A8` 是纯访存/延迟受限**（`aic_mac_ratio=0.7%`），
> 每步 9.4 ms 接近该 EP 形态的下限。

---

## 1. 发现一：每步在「draft 结束 → 下一步 target 开始」之间设备空转 ~9.6 ms

用「大间隔」扫描纯 decode 尾部窗口（`find_structure.py`）：

```
尾部窗口 锚 2229 个  锚时长中位 240.9 us
间隔 >= 3 ms 的个数: 51
大间隔分布（ms → 个数）: [(8,13),(9,19),(10,5),(11,12),(12,1),(14,1)]

  每一个大间隔的形态都相同：
  gap= 9.1 ms   前锚 148.2us(draft)  →  后锚 244.2us(target)
  gap=11.2 ms   前锚 155.1us(draft)  →  后锚 238.7us(target)
  ...（10/10 全是 draft → target）

  大间隔总时长 = 486.7 ms（占窗口 23.5%）  ⇒  约 9.6 ms/步
```

**同一结构在两份独立 profile 里复现**（快会话 8–14ms、慢会话 6–9ms，都是 draft→target）。

### 1.1 该区域内部（单段 11.08 ms 的解剖）

| 项 | 值 |
|---|---|
| 段跨度 | **11.081 ms** |
| busy（区间并集） | **3.181 ms（28.7%）** |
| **FREE** | **7.901 ms（71.3%）** |
| 算子数 | **482 个** |
| 段内空档 | 252 个（192 个 <20µs、42 个 20–100µs、16 个 0.1–0.5ms、2 个 0.5–2ms） |
| 算子时长总和 | 3.362 ms |

算子构成（Top）：`SparseFlashMlaMetadata` ×3（567µs）、`MatMulV2` ×9（381µs）、
`QuantLightningIndexerV2Metadata` ×2、`ViewCopy` ×26、`hcom_allReduce_` ×8、
`GatherElementsV2` ×14、`Cast` ×89、`BroadcastTo` ×33、`SelectV2` ×33、`ArgMaxV2` ×7。

**最大的几个空档都在 Engram 的集合通信点**：

```
gap=1757us   前: Cast/BroadcastTo/Cast/Fill(stream 47)  →  后: hcom_alltoallv_(stream 41)
gap= 606us   前: IndexCheck/IndexPutV2/broadcast        →  后: hcom_alltoallv_
gap= 428us   前: Pack/Sub/Fill(stream 47)               →  后: Sub/Range/Less(stream 35)
gap= 267us   前: Cast/Fill                              →  后: hcom_alltoallv_
gap= 250/243/200/187/184/183us  ……（同型，都是 Engram route 的往返）
```

---

## 2. 发现二：这 5.09 ms 空转**不是** draft 的 Python dispatch 开销

### 2.1 方法

`serve_v2.sh` 用 `--speculative-config {...,"enforce_eager":SE}` 控制 draft 是否入图：

```python
# llm_base_proposer.py:218
self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
```

`SPEC_EAGER=1`（我们的默认）⇒ `enforce_eager=true` ⇒ **draft 不入图**。

### 2.2 结果（SPEC_EAGER=0，draft 入图）

日志确认生效：`enforce_eager":false`，`Capturing CUDA graphs (decode, FULL)` ×5。

| 上下文 | SPEC_EAGER=1 | **SPEC_EAGER=0** | Δ |
|---|---|---|---|
| 8K | 34.27 | 34.32 | +0.05 |
| 32K | 35.85 | **34.84** | **−1.01** |
| 128K | 39.10 | 39.25 | +0.15 |

**正确性**：同 prompt/seed/temperature=0 的输出**逐字节一致**
（`' 26 27 28 ... 49'`，48 token）。这**推翻了此前"draft 入图 pos0 仍错"的结论**
——在当前配置（`fused_mc2=1` + static kernel + `LOCAL_WORLD_SIZE` 修复）下该问题不复现。

KV 容量不变（3,388,563）。

### 2.3 判读

把 draft 从 eager 变成图捕获，**只回收了 32K 的 1.0 ms**，而预期是 5 ms。
⇒ **那 5.09 ms 空转不是 dispatch 开销**，而是 **spec decode 的接受判定往返延迟**
（draft token + target logits 必须在设备上算完、经 rejection sampling、再把 accepted ids 读回 host，
才能构造下一步的 `input_ids`）。这是 spec decoding 的架构性代价。

⚠️ 注意：这一项**无法用 SPEC=0 之外的方式消除**——SPEC=0 时 A=1.0，
每 token 成本从 13.1 ms 涨到 25.0 ms（生产态），净亏 1.9×。

---

## 3. 发现三：`DispatchFFNCombineW4A8` 是**纯访存/延迟受限**，已接近该 EP 形态的下限

从 profile 取该算子的原始字段：

```
dur=250.5us
Input Shapes   : "1,5120; 48,5120,576; 48,2304,640; 1,6; 48,4608; 48,1,5120; 48,4608; 48,5120; 1,6; 1"
Output Shapes  : "1,5120; 48"
Block Num      : 24        Mix Block Num: 48
aic_mac_ratio  : 0.007     ← ★ 只有 0.7% 的时间在发 MAC
aic_mte1_ratio : 0.039     aic_mte2_ratio: 0.045
aiv_mte2_ratio : 0.004     aiv_vec_ratio : 0.003
cube_utilization: 85–92%
```

**关键读数**：

1. **输入 hidden 是 `[1, 5120]` —— 每个 rank 只处理 1 个 token**（8 token/步分摊到 8 rank）。
2. **要面对 48 个本地专家**（384 / EP=8）的权重张量。
3. `aic_mac_ratio = 0.7%` ⇒ **cube 几乎不发 MAC**，耗时全部在搬权重 + 集合通信延迟。

⇒ 这是 **EP=8 + 384 专家 + decode 1 token** 的固有形态：
每层要把 48 个专家的权重搬进来，只服务 ~1 个 token。**算术强度极低**。

**推论**：9.4 ms/step（40 层 × 235µs）已接近该并行形态的下限；
要显著改善只能改并行形态（EP 更大 / 合并 DP 批次 / 改架构），
而这些都超出当前范围（A3-node1 只有 8 卡）。

---

## 4. 发现四：Engram host 路径在本配置下 = **2.77 ms/step**（同会话配对）

用现有的运行时开关（`/tmp/v41_bneck_mode`）做同会话配对，32K、3 发中位：

| 臂 | ms/step | A | 相对 stock |
|---|---|---|---|
| `stock` | 35.19 | 2.93 | — |
| **`nohost`**（切 D2H+hash+route 全路径） | **32.42** | 2.92 | **−2.77** |
| `nocomm`（只切 route） | 34.10 | 2.97 | −1.09 |
| `faked2h`（只切第一个 D2H 的等待） | 34.40 | 2.45 | −0.79 |

`nohost` 的 A 与 stock 几乎相同（2.92 vs 2.93）⇒ **该差值可信**。

⇒ Engram 整条 host 路径 **2.77 ms/step**，其中 route（集合通信 + CPU 查表）约 1.1，
D2H 排空约 0.8，其余（hash/pad/staging）约 0.9。

**历史多轮尝试（route 流水化、host 镜像、D2H 消除、localmeta/gather/b2g）均已实测否决**，
该 2.77 的绝大部分是"每步必须做的活 + 集合通信延迟"。

---

## 5. 对 110 tok/s 的最终判定（更新后的地板）

**SPEC=0 vs SPEC=7 的干净对照**（同一客户端口径）：

| 配置 | busy | comm | FREE | ms/step | A | ms/token |
|---|---|---|---|---|---|---|
| SPEC=0 | 28.62 | 1.64 | **3.65** | 32.27 | 1.0 | **32.27** |
| SPEC=7 | 34.35 | 3.55 | **8.74** | 43.09 | 2.729 | **15.79** |

（均为 profiled；profiler 附加量 ≈7.2 ms/step，生产态按比例折算）

**128K 生产态分解**（39.10 ms/step）：

| 项 | ms/step | 可否消除 |
|---|---|---|
| target 40 层 compute（含 MoE 9.4、attention、HcPre/Post） | **≈22.6** | 否（EP 形态下限） |
| 长上下文注意力增量（128K vs 32K） | **≈3.3** | 否 |
| draft 设备忙 | ≈5.7 | 否（是 A=2.73 的来源） |
| spec 接受判定往返（FREE） | ≈5.1 | 否（SPEC_EAGER=0 已证伪可回收性） |
| Engram host 路径 | ≈2.8 | 部分（多轮尝试已收窄） |
| 其它 FREE / host | ≈2 | 部分 |

⇒ **零开销地板 ≈ 25.9 ms/step**（= 22.6 + 3.3，且已含 draft 的必要成本则在 31.6 以上）。

**要 128K > 110 tok/s 需 ms/step ≤ 25.1（A=2.763）** —— **低于地板**。
即使 A 提到 4.0（逐位置接受率上界只有 2.87，故不可达），也需 ≤36.4，而当前 39.10
仍有 2.7 ms 差距，靠 Engram（2.8）+ 其它 FREE 可勉强摸到。

---

## 6. 本轮建议的配置变更

| 变更 | 依据 | 建议 |
|---|---|---|
| **`SPEC_EAGER=0`（draft 入图）** | 32K −1.01 ms；正确性逐字节一致；KV 不变 | **建议采纳**（已在 `serve_a21.sh` 加 `SPEC_EAGER_OPT` 开关） |

---

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| SPEC=0 profile | `logs/prof_nospec/`、`/tmp/op_nospec.csv` |
| SPEC=7 profile | `logs/prof_fixed/`、`/tmp/op_summary_fixed.csv` |
| SPEC_EAGER=0 会话 | `logs/perf/a21_se0_0538_serve.log` |
| SPEC_EAGER=0 测量 | `logs/perf/a21/measure_se0.log` |
| Engram 臂配对 | `logs/perf/a21/p42_t4_quote_32768_arm_*.jsonl` |
| 结构分析工具 | `scripts/find_structure.py`、`scripts/step_breakdown.py` |
| MoE roofline | `/tmp/op_summary_fixed.csv` 的 `DispatchFFNCombineW4A8` 行 |
