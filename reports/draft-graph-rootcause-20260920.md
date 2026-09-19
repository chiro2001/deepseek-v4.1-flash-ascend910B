# DSpark draft 入图 —— 根因收窄记录（2026-09-20）

## 0. 一句话

**draft 那三个文件是好的；坏的只有"把 draft 前向放进 ACLGraph"这一件事。**

| 臂 | draft 前向 | A（conc=1，8 条中位） | 单流 tok/s | ms/step |
|---|---|---:|---:|---:|
| **E**：draft 文件 + draft **eager** | eager | **2.648** | **92.0** | 29.5 |
| **C**：draft 文件 + draft **入图** | ACLGraph | **1.07** | **42.7** | — |

⇒ 同一份 draft 文件，只切换"入不入图"，A 从 **2.65 掉到 1.07**（acceptance rate 31% → 1.5%）。
**这是可复现的确定性差异**（8 条请求，两条臂各一次；不是单发抽签）。

## 1. ★ 先修正一个方法论错误（我们为此误判了一整轮）

`tools/bench_concurrency.py` 的 **prompt 条数 = `--concurrency` 列表里的最大值**。
所以 `--concurrency 1` **只发 1 条请求**，而 README §3.2 的基线是 **64 条的中位数**。

* A 是**发放级的抽签**（历史 163 发全量：steep 15% / flat 7% / shallow 77%），
  拿"单发"去比"64 发中位"会得出**方向都错**的结论；
* 本轮实测：同一条臂，单发 A=2.04 / 67 tok/s，**8 发中位 A=2.65 / 92.0 tok/s** —— 判据直接从 FAIL 变 PASS。

**已修**：`tools/draft_graph_guard.sh` 与 `tools/draft_arm_probe.sh` 现在都用
`--concurrency 1,2,4,8`（⇒ 8 条 prompt），并显式取 `conc=1` 那一行。脚本里写了这条坑。

## 2. 上游现状（说明这条路没人走过）

`vllm-project/vllm-ascend` **main（2026-09-17 fetch 到 b255ab5）**：

```python
# vllm_ascend/spec_decode/dspark_proposer.py:73
# DSpark runs eager only (Ascend cudagraph unsupported on this path).
self.use_cuda_graph = False
```

而且 DSv4.1 支持（#16544 / `200309d`）**合入后又被回退**（#16905 / `9dc6704`）；
被回退的那份里 dspark 同样是硬编码 eager。⇒ **上游没有可抄的实现，我们是第一个试的。**

## 3. 已经排除的原因（本轮实测）

1. **draft 文件本身**（`dspark_proposer.py` / `llm_base_proposer.py` / `dsa_v1.py` 的 294 行）：
   在 draft eager 下 A=2.648 / 92.0 tok/s，**与 stock 基线同量级** ⇒ 不是它们。
2. **`start_pos` 的常驻化**（`start_pos_draft`，[STARTPOS-DRAFT-FIX]）：回退成 stock 表达式后无变化。
3. **`target_positions` 的常驻替换**（[TARGETPOS-FIX]）：eager 路径回退成真实值后无变化。
   （但这条改动**保留了**：图路径必须用捕获时绑定的常驻地址，eager 路径用真实值 —— 两者都对。）
4. **主模型图 vs eager**：不是变量。两条臂的日志都显示
   `Capturing CUDA graphs (decode, FULL): 9/9 ... finished in ~340 secs` ——
   `speculative-config.enforce_eager` 只门控 **draft**，主模型在两条臂里都走 FULL 图。
5. `DSPARK_GRAPH_CAPTURE_METADATA=1`（服务端已断言容器内确实为 1）、
   `DSPARK_GRAPH_DEVICE_METADATA=1`、capture bucket 齐全、replay 重建 metadata —— 前一轮已逐一排除。

## 4. 剩下的方向（下一步该做什么）

症状是 **pos0 ≈ 0.2**（第一个 draft token 就错）⇒ 图里 draft attention 要么没算，
要么读到的 metadata 是捕获时的陈旧值。

**建议的下一步（成本 ~1 次起服 + 1 次测量）**：用现成的影子探针定界
`DSPARK_GRAPH_SHADOW_EAGER=1`（`llm_base_proposer.py` 里已有实现）——
它在一次 replay 之后用**同一批 buffer** 再跑一遍 eager draft 前向，并 diff 两边产出的
draft token ids：

| 影子对比结果 | 结论 | 下一步 |
|---|---|---|
| **一致** ⇒ 图算得没错，是**喂给图的输入**在 replay 时没刷新 | 去审"replay 时哪些张量不是常驻 buffer 的切片"（`sin/cos`、`sas_metadata`、`dspark_swa_indices` 已经确认是常驻的，剩下的是 per-group 的 block_table / query_slot_mapping） |
| **不一致** ⇒ 图本身算错 | 逐算子比 capture vs replay 的输入，重点查 DSA attention 的 metadata 选择分支 |

**当前交付口径不变**：`DRAFT_GRAPH=0`（draft 永远 eager，A≈2.65–2.85、92–111 tok/s），
`serve_a2.sh` 里的 DRAFT-GUARD 会拒绝"stock 文件 + DRAFT_GRAPH=1"这种静默失效组合。

## 5. 本轮用到的臂与命令

```bash
# 起服（都在 chip 8-15；CPU_BIND=0 绕开 migratepages 卡死）
DEVS="8 9 10 11 12 13 14 15" CPU_BIND=0 \
  MODEL=/home/user/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  DRAFT_GRAPH=1 DSPARK_DRAFT_USE_CUDAGRAPH=0 bash scripts/serve_a3.sh   # 臂 E（draft eager）
DEVS="..." CPU_BIND=0 MODEL=... DRAFT_GRAPH=1 bash scripts/serve_a3.sh  # 臂 C（draft 入图）

# 测量（8 条中位 + 逐位置接受率 + [bneck] hp）
bash tools/draft_arm_probe.sh http://127.0.0.1:8020 results/<run_id> <标签>
```

原始产物：`results/a2_20260920_063841/arm_armE8/`（臂 E）、
`results/a2_20260920_064902/arm_armC8/`（臂 C，guard.log 里有逐档数据）。
