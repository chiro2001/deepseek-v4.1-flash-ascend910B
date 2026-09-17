# 接受率分解 + 热重载机制验证（2026-09-16 01:00–01:40）

> 场地：A3-node1 chips 8-15，容器 `dsv41-a21-perf`，端口 8020
> 配置：`static_kernel=1` + `npugraph_ex=1` + **`enable_fused_mc2=1`** + `multistream_overlap_shared_expert=false` + Engram(int8, gate CHUNK=0) + local-owner + hash fast + jemalloc + `SP_TOKENS=7`

---

## 1. 结论先行：**加大 S 无效；A 的杠杆在"逐位置接受质量"**

测量脚本记录的 `accepted_per_pos`（`vllm:spec_decode_num_accepted_tokens_per_pos`）本身就是**边缘概率**（对所有 decode step 取平均），因此

```
A = 1 + Σ_i P(position i accepted)
```

实测（FUSED_MC2 会话，3 发）：

| 上下文 | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | A = 1+Σ |
|---|---|---|---|---|---|---|---|---|
| 8K | 0.49 | 0.13 | 0.05 | 0.013 | 0.006 | 0 | 0 | **1.69** |
| 32K | **0.921** | 0.551 | 0.270 | 0.101 | 0.022 | **0** | **0** | **2.865** |
| 128K（优） | 0.931 | 0.726 | 0.425 | 0.274 | 0.137 | 0 | 0 | **3.493** |
| 128K（常） | 0.925 | 0.511 | 0.223 | 0.074 | 0.011 | 0 | 0 | **2.745** |

**读数**：
1. 接受率**逐位置几何衰减**（32K 相邻比值 ≈ 0.60 / 0.49 / 0.37 / 0.22），尾巴在 pos4 之后已经耗尽。
2. 几何级数的上界（把 S 加到无穷）≈ 1 + 0.921+0.551+0.270+0.101+0.022+0.005+0.001 ≈ **2.87** —— **实测 2.865 已经在渐近线上**。
   ⇒ **`SP_TOKENS` 从 7 再往上加，收益 ≤0.03 token**，不值得再花重启成本。
3. `SP_TOKENS=5` 实测更差（见 `moe-host-sync-and-fused-mc2.md` §6）：8K 41.10 / 32K 42.01 / 128K 45.13 ms，A 几乎不变 ⇒ S=7 是当前形态的正确取值。
4. **A 的方差很大且由内容驱动**：同样配置 128K 三次分别是 3.493 / 2.745 / 2.734。⇒ 任何 A 相关的对照都必须同会话、多次取样。

**目标换算**：`tok/s = A×1000/ms`。128K 要 >110 tok/s：

| 假设 | 需要 ms/step |
|---|---|
| A=2.745（常规） | **≤24.95 ms**（当前 38.17） |
| A=3.49（最优） | **≤31.75 ms** |
| A=4.20（尚不可达） | ≤38.2 ms（= 当前速度即可） |

⇒ 单靠"清 Free"（25.8% → 0，约 −6 ms）到不了；**要么大幅提高接受质量，要么同时砍设备 busy**。

---

## 2. 热重载机制：**已功能性验证可用**（并修掉一个真 bug）

### 2.1 为什么不能只看日志

第一版验证时 `[route-pipe]` 一条都没打印，**但日志照样写着 `rebound=[...] bytes=56889`**。根因：

> 早先的 hot_hooks 在 reload 时执行 `setattr(mod, k, v)`，把 `mod.NodeShardedEngram` 指向**新 exec 出来的类对象**；而存活实例仍持有**最初的类**。第二次 reload 按 `mod.X` 回填方法时，改的是那个"没人用的"新类 —— 日志正确、行为不变。

### 2.2 修复

`_find_live_classes()` + `_rebind_module_classes()`（`draft_hot_sp/hot/hot_hooks.py`）：
- 用 `gc.get_objects()` 找出 `__module__ == 目标模块` 的**所有存活类对象**（含历次 exec 留下的幽灵类），全部回填；
- 把模块属性指回一个存活类，保证 `isinstance(obj, mod.X)` 仍成立；
- **只处理本模块定义的类**（否则会误改 import 进来的外部类 —— 实测撞到 `AttributeError: cannot reassign member 'AUTO_ENABLE_CUSTOM_OPS'`）。
- 默认 `CLEAR_TARGET` 由 `draft` 改为 `none`：热改代码不再触碰已捕获的图。

### 2.3 验证方法与结果（功能性，不是日志）

往新代码里植入**只存在于补丁版本**的标记，看运行中的进程是否打印它：

| 通路 | 标记 | 结果 |
|---|---|---|
| engram 热重载（rw 挂载） | `[route-pipe] mode-change -> on` | ✅ `(Worker_TP0_EP0 pid=246) [route-pipe] mode-change -> on` |
| `hot/patched/*.py` 通用模块 | `[hotprobe-re]` 计数 | ✅ 6 个 worker 打印 `call #40`（共 16 行） |

佐证：容器内 `engram_hbm.py` = 56889 B、含 `_route_local_owner_pipelined`（2 处），与 reload 日志 `bytes=56889` 吻合。

**可用能力**：改 `hot/patched/<name>.py`（首行 `# HOTMOD: <模块全名>`）或改 rw 挂载的 `engram_hbm.py` → `touch hot_hooks.py` 或 `POST :8012/hotpatch` → 约 25 s 内 8 worker + EngineCore 全部生效，**不动已捕获的图、不重启**。

**限制**：`additional-config` / CLI 参数（`enable_fused_mc2`、`mc2_comm_alg`、`SP_TOKENS`、`CPU_BIND`、`GPU_UTIL`…）仍必须重启。

---

## 3. route 流水化：功能可用，**但无收益 → 否决**

**动机**：`_route_local_owner` 原本对两张表依次做 plan→lookup→h2d→a2a→bcast，而 a2a/bcast 都是阻塞调用；设备上 1.4–1.6 ms 的空档恰好落在 `Fill/Cast` 与 `hcom_alltoallv_` 之间。

**实现**（`engram_hbm.py` 的 `_route_local_owner_pipelined`，文件开关 `/tmp/v41_route_pipe` = off/on）：
phase 1 所有表的 CPU 工作做完 → phase 2 依次 `async_op=True` 下发 a2a → phase 3 统一 `wait()` + scatter + broadcast。数值语义与阻塞版逐位一致。

**确定性对账**（off / on / off，同一 prompt、seed=1234、48 token）：
```
[det] IDENTICAL off/on/off  ->  21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
```

**A/B（32K，3 发中位）**：

| 臂 | ms/step | A | decode tok/s |
|---|---|---|---|
| off（第 1 轮） | 35.70 | 2.802 | 79.7 |
| **on** | **34.50** | 2.802 | 81.4 |
| off（第 2 轮） | 34.57 | 2.898 | 84.2 |

⇒ **on(34.50) 与 off₂(34.57) 无差异**（首轮 off 偏高属会话热身，`A` 完全一致 2.802 说明内容未变）。
**判定：否决** —— 与历史结论一致：这套栈上"小包 HCCL 单次成本（~0.1 ms）低于为减少通信次数引入的 host 组装开销"。

---

## 4. 延迟臂：D2H 之后的 host 工作 **116% 暴露**

同会话、内容不变（`delaypre/delaypost` 只在 host 侧插忙等）：

| 臂 | ms/step | A | Δ |
|---|---|---|---|
| `stock` | **34.41** | 2.97 | — |
| `delaypost5`（D2H 之后插 5 ms） | 40.21 | 2.86 | **+5.80** |
| `delaypre5`（D2H 之前插 5 ms） | 37.74 | 2.91 | **+3.33** |

读数：
- **D2H 之后那段 host 工作（hash 0.43 + route 1.0–1.6 + pad 0.13 ≈ 1.6–2.2 ms）几乎 100% 落在关键路径上**（+5.80/5.00 = 116%，超出部分应为图下发被推迟的次级效应；2 发样本的 ±0.3 ms 噪声也在其中）。
- D2H **之前**只有 67% 暴露 ⇒ host 到 D2H 点之前大约有 1.7 ms 的领先量（在等设备）。

⇒ **host 侧剩余可榨空间 ≈1.6–2.2 ms/step**（清掉整条 post-D2H host 路径的上界），相对 34.4 ms 约 +5–6%。收益真实但有限，且路线已被历史多轮实测收窄（见 `engram-sync-optimization.md`）。

---

## 5. 设备账（32K decode，86 step，FUSED_MC2）

| 算子 | ms/step | 次/step | avg µs |
|---|---|---|---|
| **DispatchFFNCombineW4A8** | **8.48** | 35.8 | 236.7 |
| **hcom_allReduce_** | **4.83** | 148.3 | 32.6 |
| HcPre | 2.49 | 71.7 | 34.8 |
| QuantBatchMatmulV3 | 1.99 | 188.3 | 10.6 |
| QuantLightningIndexerV2 | 1.71 | 6.7 | 255.6 |
| TransposeBatchMatmul | 1.70 | 35.8 | 47.3 |
| MatMulV2 | 1.52 | 58.7 | 25.9 |
| SparseFlashMla | 1.27 | 33.4 | 38.1 |
| hcom_allGather_ | 1.21 | 76.6 | 15.8 |
| hcom_alltoallv_ | 1.01 | 3.3 | 305.4 |

busy union = **28.6 ms/step（83%）**、Free = **5.8 ms/step（17%）**。
（换算核对：28.6 + 5.8 = 34.4 ms/step，与 p42 实测 34.35 一致。）

**MoE 通信 8.48 + TP allreduce 4.83 = 13.3 ms/step，占 busy 的 47%** —— 这是当前设备侧最大的一块。

---

## 6. 其它本轮否决项

| 项 | 结论 | 证据 |
|---|---|---|
| `CPU_BIND=1`（cannbot 措施6） | **否决** | A3-node2 同配置：8K **39.86**（vs 34.40）、32K **40.68**（vs 34.35），KV 3,389,665 基本不变 |
| `SP_TOKENS=5`（cannbot 记载官方配置） | **否决** | 见 `moe-host-sync-and-fused-mc2.md` §6 |
| route 流水化 | **否决** | 本节 §3 |
| localmeta / gather / b2g / 消 D2H | **已否决**（历史） | `engram-optimization-round2.md` |

---

## 7. 下一步（按预期收益）

1. **非 mtpq draft 的接受率对照**（进行中）：`MODEL=v41-w4a8-engram-dr-vision`，其余固定。若 A 显著上升，说明量化 draft 是接受率的元凶，再去找"既保 A 又保 KV>3M"的组合（如 util 0.96）。
2. **`enable_fused_mc2=2`（MegaMoe）**：目标是 8.48 ms/step 的 `DispatchFFNCombineW4A8`。风险高（需 sym buffer + `cann_ops_transformer`）。
3. **`mc2_comm_alg=hierarchy`**：直接作用于 MC2 dispatch/combine。
4. **`enable_reduce_sample=1`**：作用于 logits 的 allgather（1.21 ms/step，76.6 次/step）与采样路径。
5. **HcPre/HcPost 融合**（3.08 ms/step）：cannbot 措施8 类，glm5next 有可参考实现。

---

## 8. 原始证据路径（A3-node1）

| 内容 | 路径 |
|---|---|
| FUSED_MC2 profile | `logs/prof_a21b/`、`/tmp/op_summary_fmc2.csv`、`/tmp/msprof_fmc2.db`、`/tmp/op_range_fmc2.bin` |
| ALLTOALL 对照 profile | `logs/prof_a21/extract/op_summary_r0.csv`、`/tmp/msprof_r0.db`、`/tmp/op_range_r0.bin` |
| route-pipe A/B | `logs/perf/a21/pipe_ab2.log`、`logs/perf/a21/p42_t4_quote_32768_pipe_{off,on,off}.jsonl` |
| 延迟臂 | `logs/perf/a21/p42_t4_quote_32768_a21_f_{stock,delaypre5,delaypost5}.jsonl` |
| 接受率 | `logs/perf/a21/p42_t4_quote_{8192,32768,131072}_fmc2b_*.jsonl`（含 `accepted_per_pos`） |
| CPU_BIND 对照 | A3-node2 `logs/perf/measure_cpubind.log` |
| 分析脚本 | `scripts/{op_window,free_account,free_timeline,gap_anatomy,parse_framework,attr_item,comm_db_by_op}.py` |
| 热重载 | `draft_hot_sp/hot/hot_hooks.py`、`draft_hot_sp/hot/patched/` |
