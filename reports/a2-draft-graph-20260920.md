# A2（8×910B3）Draft 入图实测：单流 **54.7 → 88.7 tok/s（+62%）**

> 日期：2026-09-20 22:22–22:37 CST｜真机：A2（8×910B3，单机 8 卡）｜产出目录
> `results/a2_20260920_222216/`
>
> 口径：`ENGRAM_DEVICE_INDEX=0`（走 host 路径，**与下面对照的 eager 会话同口径**）
> + `DRAFT_GRAPH=1` + `GPU_UTIL=0.90 MAX_SEQS=4 PREFIX=0 BAT_TOKENS=2048
> STATIC_KERNEL=1 PYTHON_PGO=1 PATCH_MODE=baked`。

---

## 1. 一句话结论

**A2 上 draft 入图的收益远大于 A3**：单流 decode **54.7 → 88.7 tok/s（+62%）**，
折合 **−30.5 ms/step（−47%）**。原因是 A2（910B3）CPU 更弱 ⇒ draft 的 eager 派发
开销更大 ⇒ 把它挪进图里省得更多。**这台机器上 draft 入图是唯一的大杠杆，值是 A3 的 2.5 倍。**

---

## 2. 核心数据

### 2.1 设备侧 `[bneck] hp`（= 完整 decode step 时间，源码口径见下）

同一台 A2、同为 `ENGRAM_DEVICE_INDEX=0`、同为 8 卡 TP8，
唯一差异是 `DRAFT_GRAPH`（**跨会话对比，非同进程 A/B**）：

| 会话 | `DRAFT_GRAPH` | `hp`（ms/step） | APIServer A | APIServer 单流 tok/s | 换算自洽性 |
|---|---:|---:|---:|---:|---|
| 本次 | **1** | **34.0 – 34.8** | 3.03 | **88.7** | `1000×3.03/88.7 = 34.2` ✅ 与 `hp` 一致 |
| 上次（`reports/a2-hostpath-decode-20260920.md`） | 0 | 64.6 – 65.3 | 3.44 | 54.7 | `1000×3.44/54.7 = 62.9` ✅ 与 `hp` 一致 |

⇒ **−30.5 ms/step**；单流 **+62%**。

### 2.2 客户端 quote 口径（`run_test.sh` [4]）

| 上下文 | ms/step | A | tok/s | TTFT(中位) | prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 8 192 | **30.99** | 1.745 ⚠️ | 56.0 | 7.70 s | 2433 |
| 32 768 | **31.14** | 2.866 | 91.7 | 6.40 s | 5157 |

单发原始值（`p42_t4_quote_*.jsonl`）：

```
8K : steps=146 ms/step=30.571 tok/step=1.747 tok/s=57.131 ttft=1.924s
32K: steps=91  ms/step=31.201 tok/step=2.802 tok/s=89.812 ttft=5.849s
```

**`ms/step` 与上下文几乎无关**（8K→32K 涨 4×，`ms/step` 只 +0.5%）—— 与 A3 同形态
（A3 8K 35.8 / 128K 36.1–37.1）。

### 2.3 ⚠️ 8K 的 `A=1.745` 是**冷启动首请求**，不是缺陷

同一会话稍后的 APIServer 稳态：

```
SpecDecoding metrics: Mean acceptance length: 3.03, Accepted throughput: 59.40 tokens/s,
Drafted throughput: 146.49 tokens/s, Per-position acceptance rate: 0.706, 0.488, 0.328,
0.280, 0.225, Avg Draft acceptance rate: 40.5%
```

⇒ 稳态 **A = 3.03**，而 32K 那一发是 **2.866**（同一量级），**只有 8K 那一发是 1.745**。
8K 是 `run_test.sh` 的第一条真请求（此前只有容量检查的短请求），
`ttft` 中位 7.70 s vs 单发 1.924 s 也说明这一档混入了冷启动样本。

**判据**：A2 的 A 健康区间 **2.8 – 3.1**；若稳态 A 掉到 1.0 或 1.7 以下才是异常
（`DRAFT_GRAPH=1` 的静默失效指纹是 **A 恒 1.00**，见 `reports/draft-graph-negative-control.md`）。

---

## 3. 本次会话的其它实测

| 项 | 值 |
|---|---|
| 起服耗时 | **803 s**（含 static kernel 冷编译；缓存命中后应回到分钟级） |
| KV 容量 | **3,498,354 tokens**（`GPU_UTIL=0.90 BAT=2048`） |
| static_kernel | **0 次降级**（`serve_a2.sh` 判 ✓；`run_test.sh` 当时误报 ✗，已于 `dde7794` 修复） |
| Engram local-owner | validate 通过 → 切 **fast** |
| Vision | **23/23 PASS**（`mandatory negatives: OK`） |
| DRAFT-GUARD | `dspark_proposer.py` 命中 2 处 ✓、容器内 `DSPARK_GRAPH_CAPTURE_METADATA=1` ✓ |
| 端口/模型名自定义 | `PORT=8077` + `SERVED_NAME=deepseek-v4-flash` 生效（`f6ad086` 的验证） |

### 3.1 `[bneck]` 各字段（本次会话稳态，`steps=1080…1400`）

| 字段 | 含义 | 本次观测 |
|---|---|---|
| **`hp`** | **完整 decode step 时间（ms）** | **34.0 – 34.8**（8 rank 离散 < 0.35 ms） |
| `total` | 该 rank 的 Engram **host 路径**耗时 | 3.0 – 6.6 |
| `d2h` | device→host 拷贝等待（**rank 间差异最大的一列**） | 0.20 – 3.41 |
| `route` | 查表路由（a2a + bcast） | 1.34 – 2.89（TP0 最大，rank0 角色代价） |
| `hash` / `plan` | 哈希 / 计划（numba JIT 后） | 0.11 – 0.13 / 0.10 – 0.12 |
| `pad` / `meta` | 填充 / 元数据 | 0.19 – 0.23 / 0.003 |

> **口径**：`patches/files/model.py` 里 `mark_step()` 在 `prepare_engram_inputs()` 开头调用，
> 累计相邻两次间隔 ⇒ **`hp` 就是完整 decode step**，算 tok/s 要用它；
> `total` 只是 Engram host 路径自身，**不是** step 时间。

### 3.2 与 A3 的对照（同为 draft 入图）

| 项 | A3（8×910C） | **A2（8×910B3，本次）** |
|---|---:|---:|
| draft 入图收益 | −12 ms/step（36.9 → 24.9 同进程） | **−30.5 ms/step（跨会话）** |
| 单流 tok/s | 85 – 90 | **88.7** |
| A | 2.40 – 2.74 | 3.03 |

⇒ **A2 追平了 A3 的单流吞吐**（88.7 vs 85–90），且 A2 的分子里还没有 Engram 算子的加速
（`ENGRAM_DEVICE_INDEX=0`，因 `ret=207001` 在 A2 上不可用）。

---

## 4. 待确认 / 未验证

1. **跨会话对比**：eager 基线（`hp=64.8`）与本次（`hp=34.3`）是两个独立进程，
   且 `GPU_UTIL` 0.91 vs 0.90、请求内容不同。**要做严格 A/B 需在同进程内切
   `/tmp/v41_dspark_flags` 的 `DRAFT_FORCE_EAGER=1`**（热切换，见
   `reports/draft-graph-investigation-20260920.md` §5）。
2. **A2 的 1–64 并发**未测（draft 入图后）。A3 的对应测量见
   `results/bench/conc_draft_graph.json`。
3. **8K 的 `A=1.745`** 归因为冷启动，仅 1 次观测；建议重跑 `MODE=quick`
   连测 3 次 8K 定性（如仍为 1.7 则需改为"8K 档系统性偏低"）。

---

## 5. 复现命令

```bash
MODEL=/home/user/models/out/v41-w4a8-flat IMAGE=dsv41-a2:v8 \
  GPU_UTIL=0.90 PROFILE=1 DROPCACHE=0 \
  ENGRAM_DEVICE_INDEX=0 PORT=8077 SERVED_NAME=deepseek-v4-flash \
  DRAFT_GRAPH=1 bash scripts/run_test.sh

# 看设备侧 step 时间（hp = ms/step）
docker logs -f dsv41-a2 2>&1 | grep -E "\[bneck\]|\[route-probe\]"

# 看稳态接受长度（健康区间 2.8–3.1）
docker logs dsv41-a2 2>&1 | grep "SpecDecoding metrics" | tail -5
```
