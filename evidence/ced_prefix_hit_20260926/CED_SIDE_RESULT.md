# CED 臂开 `PREFIX=1` 的**实测**结果（2026-09-26 00:49–01:20）

> 结论先行：**144K 串行可用且正确；1M 会让 D 引擎崩溃**。
> 崩溃点精确到 `experimental/ced/mooncake_hybrid_connector.py::start_load_kv`
> 的 `raise RuntimeError("CED decoder expected 12 KV cache groups without DSpark")`。

## 1. 怎么跑到这一步的

`scripts/serve_a3_ced_pd.sh` 原先对 `PREFIX=1` 是**硬门**（直接 exit 2）。
2026-09-26 改成"默认仍然拒绝，但允许显式放行"：

```bash
PREFIX=1 V41_CED_ALLOW_PREFIX=1 ...   # 实验臂；会打 WARN，明确结果不可当交付证据
PREFIX=1                              # 不带开关 → 仍然 exit 2
```

目的是**先实测会坏在哪**，而不是按推断改三处代码。

## 2. 144K：可用且正确（12 次请求全对）

探针 `pfx_probe2.py`（针插 80% 深度、`temperature=0`、判据用服务端 metrics）：

| prompt | 步骤 | wall | D hits 增量 | 答案 | 与冷是否一致 |
|---|---|---:|---:|---|---|
| P1 | cold | 31.50 s | 0 | ✅ `RB9N-6014` | — |
| P1 | hit1 | 3.08 s | 144,000 | ✅ | 相同 |
| P1 | hit2 | 1.19 s | 144,000 | ✅ | 相同 |
| P2 | cold | 11.24 s | 0 | ✅ | — |
| P2 | hit1 | 1.26 s | 144,000 | ✅ | 相同 |
| P2 | hit2 | 1.24 s | 144,000 | ✅ | 相同 |

**交错命中**探针 `pfx_interleave.py`（P1冷→P2冷→P1热→P2热→P1热→P2热，6 次）：

```
  1 P1(冷)  wall= 1.25s  D_hits增量=144000 正确=True 与首次相同
  2 P2(冷)  ...
  3 P1(热)  wall= 1.25s  D_hits增量=144000 正确=True 与首次相同
  4 P2(热)  wall= 1.29s  D_hits增量=144000 正确=True 与首次相同
  5 P1(热)  wall= 1.21s  D_hits增量=144000 正确=True 与首次相同
  6 P2(热)  wall= 1.24s  D_hits增量=144000 正确=True 与首次相同
```

这条专门验"CED 的 D 侧预清零会不会把**别的请求**的 hashed 块清零"——
**6/6 全对且各自与冷值逐字节相同**，在这个配置下没有出现污染。

## 3. 1M：命中有，但（a）没有加速（b）随后 D 崩了

同一探针、目标 1M：

| # | 请求 | wall | D hits 增量 | 结果 |
|---|---|---:|---:|---|
| 1 | P1 cold | 96.73 s | 0 | ✅ 正确 |
| 2 | P2 cold | 91.29 s | 0 | ✅ 正确 |
| 3 | **P1 hit** | **103.56 s** | **999,936** | ✅ 正确，但**比冷还慢** |
| 4 | P2 hit | — | — | ❌ **HTTP 500，D 引擎死亡** |

### 3.1 崩溃根因（精确）

```
(EngineCore) ERROR [core.py:1351] RuntimeError: Worker failed with error
    'CED decoder expected 12 KV cache groups without DSpark'
```

抛出点在 `mooncake_hybrid_connector.py::start_load_kv`：

```python
if len(meta.remote_block_ids) != 12 or len(meta.local_block_ids) != 12:
    raise RuntimeError("CED decoder expected 12 KV cache groups without DSpark")
```

而崩溃前 `dump_input.py` 打出的调度输出里有决定性的一行：

```
num_common_prefix_blocks=[7055, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

即：**前缀命中只对 group 0 给出公共前缀块（7055 块），其余 11 个 group 是 0**。
这正是 hybrid/SWA 布局的必然结果（SWA 只保留 128 token 窗口、压缩组有自己的
块映射），但它让"每个请求都要有 12 个形状一致的 group 列表"这条 CED 契约失效。

⇒ **真正要先修的是连接器的 12-group 形状假设**，不是我原先预测的调度器边界断言
（后者在本次 6 次 144K + 4 次 1M 请求里**一次都没触发**）。

### 3.2 "命中却不加速"这一条同样重要

1M 的命中请求 103.56 s，与冷 96.73 s 同量级。144K 却能 31.5 s → 1.2 s。
⇒ **1M 的端到端时间不由 P 的 prefill 计算主导**（否则命中应大幅缩短）。
这与 §"P/D 逐请求剖析"里看到的"D 只干 0.27 s、其余等 P"在 144K 成立，
但在 1M 上 P 的哪一段没有受益，**尚未归因**，是和上面崩溃并列的第二个待查项。

## 4. 修正我原先的预测

`docs/CED-PD-CACHE-HIT-PLAN-20260925.md` §2 列了三处代码级前提。实测后的账：

| 预测的障碍 | 实测 |
|---|---|
| §2.1 启动硬门 | ✅ **确实存在**（已改成可显式放行） |
| §2.2 调度器 `num_computed_tokens != replay_end` 断言 | ❌ **未触发**（144K 6 次 + 1M 3 次都没有 boundary mismatch 日志） |
| §2.3 D 侧对 hashed 块预清零污染 | ❌ 在 144K 交错测试里**未出现**（6/6 正确且与冷一致）；1M 未走到这一步 |
| **（新发现）连接器 12-group 形状假设** | ✅ **1M 命中时必然触发并杀死引擎** |

所以这份文档的价值是"把预测换成了实测"，代价是一次崩溃 —— 这也是它必须在
`PREFIX=1` 上显式加 `V41_CED_ALLOW_PREFIX=1` 的原因：默认口径不会被它碰到。

## 5. 复现

```bash
# 起 CED + PREFIX=1（实验臂）
cd <shadow pkg>
bash /home/l00886679/tmp/20260924/ced_numeric/launch_ced_pfx.sh
# 144K / 1M 探针
python3 /home/l00886679/tmp/20260924/ced_numeric/pfx_probe2.py  http://127.0.0.1:18992 http://127.0.0.1:18990 144000
python3 /home/l00886679/tmp/20260924/ced_numeric/pfx_interleave.py http://127.0.0.1:18992 http://127.0.0.1:18990 1000000
```

⚠️ `launch_ced_pfx.sh` 自身也踩过一个坑并已修：它第一版把
`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` **同时导出给 P**，而该开关只对 decode 合法
（`serve_a2.sh` 会 `die`），导致 P 起不来、health 又被**旧容器**应答。
现在 P 与 D 的 env 在子 shell 里**分开设**。
