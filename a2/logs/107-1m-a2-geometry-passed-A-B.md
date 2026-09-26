# 107 — **1M + A2 同规模几何：阶段 A/B 通过**（起服 9m45s、池 `units=87040` 逐字命中 A2、三轮零失败且取回真实）

> 2026-09-23 02:0x–02:2x CST（A3-node1 Phy-ID 8–15，端口 8051）。
> 执行：子代理 **`ARM_1M`**（主代理复核关键读数）。臂：**`r8-1m-bcd3`**（`KEEP=1` 保留）。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

**用户点名的"1M 长上下文 + A2 同规模 DRAM 卸载"已跑通前两阶段**：
`GPU KV cache size = 1,943,421 tokens`（判据 ≥1,048,576，**1.85× 余量**）、**六项错误码全 0**、
**池 `units=87040` 逐字命中 A2 的 `OFFLOAD_GB=85`**、**三轮 `failed=0` 且 `CPU_to_GPU>0`（取回真实发生）**、
★ **`BlockRemoved:CPU = 0`（池没被挤爆 —— 这正是 `OFFLOAD_GB=85` 的推导目的）**。

---

## 1. 【实测】起服（阶段 A）—— 第 3 次尝试才过

| 判据 | 读数 |
|---|---|
| **`GPU KV cache size`** | **1,943,421 tokens**（≥1,048,576 ✅ **1.85×**） |
| 换算率 | **242,928 token/GiB** ↔ A2 生产 **242,941** ⇒ **差 0.006%** |
| 六项错误码 | `EE1016 / 507057 / EH0012 / 207001 / capture failed / out of memory` **全 0** |
| 容器内 dsa md5 | **`94aeebb757d6d5708268754481a05e0a`**（= `S_graphfix` graphsafe 生产版，**逐字对上期望值**） |
| merged `model.py` md5 | `9a2d782cd2c5d2e7e6dcda671836f291`（G1 门通过） |
| device-index | 容器内 env `=0`；日志 `[DEVICE-INDEX]` 行数 **0** |
| **起服耗时** | `t0=02:00:07` → KV 定容 `02:04:24`（**4m17s**）→ `health=200` ≈ **9m45s**（含 static kernel 预热） |
| 自检门 | 全过（`L1 挂载源数=6`、`worker 物理池合计=8`、`③真实分量=8`、`★ R8 scheduler 挂载=1`、`int8 挂载源数=7` …） |

★ **前两次失败根因（已闭环）**：**宿主 OOM**（dmesg 铁证：单 `VLLM::Worker_TP` `anon-rss 87.7 GiB`、
`constraint=CONSTRAINT_CPUSET`、`global_oom`），因为 1M 臂的池按 rank 分配、**比 128K 臂多 340 GiB**
（`OFFLOAD_BYTES` 21.5 → 85 GiB ⇒ 池/rank 14.381 → **56.898 GiB** ⇒ ×8 = **455.2 GiB**）。
表现是 `Worker died (exit code: None)` + `shm_broadcast: cancelled`（**都是下游症状**）；
★ **`serve.log` 里所有已知错误码全 0** —— 因为 OOM 在**内核层**。
⇒ **纪律**：8 卡 + 1M 几何起臂前，**除 serve.log 外必查 `dmesg`**，且 `MemAvailable ≥1200 GiB`。

---

## 2. 【实测】阶段 B（128K 单发冒烟 + 三轮 replay）

`1 × 131072 → replay 65536, rounds=3, concurrency=1`：

| 轮 | wall | TTFT | tok/s | `failed` |
|---|---:|---:|---:|---:|
| fill | 45.938 s | 33,152.9 ms | 2.2 | **0** |
| replay1 | 11.254 s | 8,300.4 ms | 2.2 | **0** |
| replay2 | 12.305 s | 8,236.2 ms | 2.8 | **0** |

### 卸载三判据（实测原值）
| 判据 | 读数 |
|---|---|
| ★ **`CPU_to_GPU`** | **1.549643776e+09 B（1.55 GB）> 0** ⇒ **取回真的发生**（不是 `tierB` 那种 `0.0` 的无效臂） |
| `hits` | **65,024**（`external_prefix_cache_hits_total`；`queries_total=262,400`） |
| ★ **`BlockRemoved:CPU`** | **事件 0 次**（KV 事件全清单：`BlockStored:GPU=19034`、`BlockStored:CPU=2332`、`AllBlocksCleared:None=2`，**无任何 Removed 类**；`decode_errors=0`） |
| 搬运耗时 | `CPU_to_GPU 0.166 s` / `GPU_to_CPU 5.279 s` |

---

## 3. ★★ 【实测】池几何 = **A2 口径**（运行期键值，不是换算）

```
[K_l1_8card] ③ 池子: units=87040 Σquota=87040
  quota={0:34816, 1:0, 2:4352, ..., 12:8704}
[K_l1_8card] ③ 记账对账: worker_kv_bytes_per_block=131072 Σpage=832128
```

- ★ **`units=87040` 逐字命中 A2 的 `OFFLOAD_GB=85 ⇒ 87,040 units`**
  （= **3.00 × 1M 会话** @24,064 units/会话）⇒ 把"85 GB 池 = 3 个 1M 会话"从换算升级为**运行期实测**。
- ★ **`Σpage=832128` 与 128K 臂 `final` 逐字相同** ⇒ **`MAX_LEN` / `KV_MEM_BYTES` 没有连带改池几何**
  （红线 9 要求的"改一个参数别连带改第二个"**已排除**）。
- `宿主实占 67.454 → 56.898 GiB/rank`（×8 = **455.2 GiB**，与 OOM 分析里那 +340 GiB 闭环）。

★ 内存：起臂前 **1487 GiB** → 起服后稳在 **618–737 GiB**（未破 400 红线，但贴得很近）。

---

## 4. 待办（阶段 C/D/E）

| # | 内容 | 判据 |
|---:|---|---|
| **C** | **真 1M 单发**（`--prompt-tokens 1048576 --max-tokens 128`） | `failed=0`、`finish_reason` 正常、四错误码 0、`num_computed_tokens` 走到 1M；★ 1M prefill 慢是正常的（~25k tok/s ⇒ 40 s 量级） |
| **D** | **3 个 1M 会话 + 池覆盖** | 三轮 `failed=0`、**`BlockRemoved:CPU=0`**、`CPU_to_GPU>0`；★ 若 `BlockRemoved>0` ⇒ 如实报"85 GB 不够装 3×1M"（负结果同样有价值） |
| **E** | 文本正确 + 同运行复现 | `text_correctness_probe --mode all`（题库 + prefix-pair）、`check_same_run_replay.py` |

★ 并行的性能侧队列（1M 臂跑完就上）：**`r8-chunkview`**（`d84f087c`）→ **`r8-merged`**（`30ecf49b`+`8057b3eb`），
两条都带**容器内指纹自检**，并同时跑精度两条判据。
