# CED-PD 上启用 DSpark：验收结果与可复现启动方式（2026-09-26）

> **结论**：在 CED-PD（P 只跑前 20 层 + 层 20 全局源；D 做 128-token 有界重放 +
> 全 40 层，BF16 KV）的基础上，**D 侧 DSpark 已打通并通过完整验收矩阵 21/21**，
> 与未开 DSpark 的交付口径**结果逐项一致**。P 侧保持 `SPEC=0`（架构性要求）。

## 1. 验收矩阵（21/21 通过）

口径：`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`、`V41_SLOT_MAP_FUSED=on`、
其余与 CED 交付口径逐项相同（`MULTISTREAM=0 DSA_OVERLAP=0 PREFIX=0`
`KV_DTYPE=bfloat16` `MAX_LEN=1048576`）。

| 项 | 144K | 1M |
|---|---|---|
| 22-token 短针 | ✅ | — |
| 四针 A/B/C/D | **4/4** | **4/4** |
| 流式（TTFT） | ✅ 10.57 s | ✅ 100.04 s |
| 多轮（三轮） | **3/3** | **3/3** |
| 缓存命中（`PREFIX=0` 口径，`cached_tokens=0` 属预期） | ✅ a1/a2 | ✅ a1/a2 |

**总计 21 条，通过 21，失败 0。**
四针答案与 `SPEC=0` 交付口径**逐字节相同**（`ZQ7K-3341` / `VX2M-8890` /
`HT4P-5527` / `RB9N-6014`）。

原始证据：`results/slotfused_accept_all.json` +
`results/slotfused_accept_all_evidence/`（a3-21，run `ced_d4b_draft_graph_0926_051357`）。

## 2. 性能（step 口径）

| ctx | ms/step | A | ms/token | 相对 `SPEC=0`（26.8 / 27.4 ms/token） |
|---|---:|---:|---:|---|
| 32K | 34.22 / 34.22 | 2.59 / 3.61 | 13.20 / 9.70 | **≈2.4×** |
| 144K | 36.43 / 36.56 | 3.17 / 3.07 | 11.47 / 12.09 | **≈2.3×** |

* **A（平均接受长度）3.1–3.4**，高于 A2 单实例的 2.7–3.0。
* step 墙钟 34–37 ms，与 A2 单实例的 38.5 ms 同量级。
* 详细剖析（设备利用率 97.5%、算子构成、已排除的优化）见
  [`CED-PD-DSPARK-PROFILING-20260926.md`](CED-PD-DSPARK-PROFILING-20260926.md)。

## 3. 可复现的启动方式

### 3.1 前置：P 侧（不变）

P 与 DSpark **架构性无关**，保持 CED 交付口径即可（`SPEC=0`）。
launch 脚本会**硬拒绝** P 开 SPEC，理由是 DSpark 的 aux hidden state 取自
目标层 37/38/39，而 P 在第 20 层 break。

### 3.2 D 侧：开 DSpark

```bash
ARM=draft_graph V41_SLOT_MAP_FUSED=on bash launch_d_dspark.sh
```

（脚本在 `experiments/dspark/launch_d_dspark.sh`，逐项对齐交付口径的 D，
唯一变量是 `SPEC` / `DRAFT_GRAPH` / `V41_SLOT_MAP_FUSED`。）

关键 env：

| env | 值 | 作用 |
|---|---|---|
| `SPEC` | 1 | 开推测解码（P 侧必须为 0） |
| `SP_TOKENS` | 7 | 已扫过 5/7/9，7 最优 |
| `DRAFT_GRAPH` | 1 | 草稿入图（必需 `DSPARK_GRAPH_CAPTURE_METADATA=1` 配套，脚本自动设） |
| `V41_CED_ALLOW_DSPARK` | 1 | **2026-09-27 起 D 侧 DSpark 是交付默认**；显式 `=0` 表示不要 DSpark（退回 `SPEC=0`），此时引擎侧的门仍会拒绝矛盾配置 |
| `V41_SLOT_MAP_FUSED` | on | slot-mapping 融合；**注意这是 host-only 优化，实测对 step 时延无影响**，只修了原来的死开关 |

### 3.3 起服后的硬门（必须逐条确认）

```bash
# 1) 组拓扑：P 12 组、D 13 组，上半层 SWA 必须仍落在 7..11
grep -a "CED decode: upper SWA groups" d/serve.log
#    期望：upper SWA groups=(7, 8, 9, 10, 11) draft(g12) groups=(12,) total_groups=13

# 2) 4 GiB 寻址上界（G12 与 target SWA 共享槽位，上界不变）
grep -a "CED-32BIT-GUARD" d/serve.log | head -1
#    期望：num_blocks=29076 max_page_stride=147712

# 3) 草稿图确实带 metadata 捕获（不是 LEGACY 回退分支）
grep -a "dspark-graph-capture" d/serve.log | head -1
#    期望：... built draft attention metadata (groups=1 layers=3 ...)；出现 "LEGACY" 就是坏的

# 4) 接受长度（唯一能证明草稿在干活的判据）
grep -a "SpecDecoding metrics" d/serve.log | tail -1
#    期望：Mean acceptance length ~3；**A≈1.0 就是草稿没产出**，此时 ms/token 反而更好看
```

### 3.4 常见故障与判据

| 现象 | 原因 | 位置 |
|---|---|---|
| 起服正常、**第一条请求**时引擎死，`'MooncakeConnectorWorker' object has no attribute 'ced_draft_swa_groups'` | 组探测只写在调度侧、执行侧没算 | 已修：抽出 `_ced_detect_swa_groups()` 两边共用 |
| 启动期 `CED decoder expected the upper SWA groups ... got [...]` | DSpark 把某个组插到了索引 12 之前 | `[CED-DSPARK-GUARD]`，重启期硬错误（好过静默清错页） |
| `CED decoder expected 12 remote ... N local` | P/D 组数契约不匹配 | 远端恒为 12（P 无 DSpark），本地 = 12 + 草稿组数 |
| A ≈ 1.0 | 草稿窗口没被填（重放没走 `build_model_inputs_first_pass`）或草稿图里没有 attention | 见 §3.3 第 3/4 条 |

## 4. 未做/已知限制

* **草稿 SWA 路径没有独立的越界读修复**：`[CED-SWA-CLIP]` 修的是 target 的
  `dsa_v41.py`，而草稿走 `dsa_v1.py` + `AscendDSparkProposer`。
  1M 四针与多轮全部通过，说明**在本轮口径下没有复现该越界**，
  但这是"没复现"而不是"已证明不存在"。
* **draft 入图没有兑现性能收益**：144K 上图臂 11.2 ms/token vs eager 臂 11.1
  （噪声内），32K 上反而更差。draft 只有 3 层，图 replay 的固定开销盖过了
  省下的 launch。保留它是为了与 A2 口径一致。
* **`PREFIX=1` 的缓存命中未在 DSpark 下验证**：本轮 `PREFIX=0`，
  验收里的 `cached_tokens=0` 是预期行为。缓存命中是 CED 的独立实验臂
  （见 [`CED-PD-CACHE-HIT-PLAN-20260925.md`](CED-PD-CACHE-HIT-PLAN-20260925.md)），
  与 DSpark 组合需要单独一轮。

* **DSpark 的收益只在低并发成立**。2K prompt + 128 token 输出、`MAX_SEQS=4`：

  | 并发 | DSpark 总吞吐 | SPEC=0 交付口径总吞吐 | 比 |
  |---:|---:|---:|---:|
  | 1 | **73.1 tok/s** | 41.4 | **1.77×** |
  | 2 | 57.2 | 70.6 | **0.81×** |
  | 4 | 76.7 | 114.5 | **0.67×** |

  原因是推测解码把每步的行数从 `batch × 1` 抬到 `batch × (1 + SP_TOKENS)`：
  并发 4 时 M = 32（vs SPEC=0 的 M = 4），而产出只多 A≈3 倍
  ⇒ 每步的算子时间增长快过 token 产出。

  **⇒ 部署建议**：DSpark 适合**低并发、交互式**场景（单流时延/吞吐最优）；
  高并发吞吐场景应保持 `SPEC=0`，或按并发自适应开关。
  这组数字是短 decode 窗口（128 token）测的，绝对值有噪声，但**趋势是明确的**。
