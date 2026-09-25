# CED-PD 的「缓存命中」：为什么现在做不到，以及要动哪三处

目标是「144K 与 1M 上下文、流式/多轮/**缓存命中**正确性验证」。前四项都已验收，
**只有缓存命中这一项是"跑不了"而不是"还没跑"**。本文把三处代码级前提写清楚，
并给出一个**先验基础设施、再动 CED** 的实验顺序。

## 0. 现状（一句话）

`scripts/serve_a3_ced_pd.sh` 在启动层直接拒绝 `PREFIX=1`；即使绕过它，
调度器的边界断言与 D 侧预清零两个不变量都会因"命中的是缓存块"而失效。

## 1. 先做的实验：基线（非 CED）能不能开缓存

**这是关键路径上最便宜的一步**：CED 的三处改动只有在"这套 PD 基础设施本身
支持前缀缓存"的前提下才有意义。基线（P/D 都是全 40 层、stock 连接器）
没有 CED 的清零/边界问题，正好当探针。

```bash
# 一次重启（P+D），唯一变量是 PREFIX
PREFIX=1 MULTISTREAM=0 DSA_OVERLAP=0 MAX_LEN=1048576 \
  NAME=dsv41-pfx-p PORT=18990 KV_PORT=19090 DEVS="0 1 2 3 4 5 6 7" \
  bash scripts/serve_a3_pd.sh prefill
PREFIX=1 MULTISTREAM=0 DSA_OVERLAP=0 MAX_LEN=1048576 \
  NAME=dsv41-pfx-d PORT=18991 KV_PORT=19091 DEVS="8 9 10 11 12 13 14 15" \
  bash scripts/serve_a3_pd.sh decode
# 然后同前缀连发两次，看 cached_tokens / prefix_cache_hits_total
bash /home/l00886679/tmp/20260924/ced_numeric/prefix_test.sh
```

判据（三选一，都必须能给出结论）：

| 结果 | 含义 | 下一步 |
|---|---|---|
| 起不来 / 启动报错 | 基础设施不支持（hybrid KV + Mooncake + caching） | 这条路要等上游；缓存命中只能记为"上游阻断" |
| 起来了但 `cached_tokens=0` | 能跑但没命中（分块/哈希口径问题） | 查 block hash 与 `hash_block_size`，仍属基础设施 |
| `cached_tokens>0` 且答案对 | **基础设施可用** | 进入 §2，动 CED 的三处 |

顺带能回答一个更大的问题：`PREFIX=1` 在**没有 CED** 时是否影响长上下文正确性
（§3 的 `MULTISTREAM` 问题与它无关，但值得同时观察）。

## 2. CED 开缓存要动的三处

### 2.1 启动硬门（`scripts/serve_a3_ced_pd.sh`）

```bash
# 现状：PREFIX 非 0 直接 exit 2
for setting in "SPEC:${SPEC:-0}" "PREFIX:${PREFIX:-0}" "DRAFT_GRAPH:${DRAFT_GRAPH:-0}"; do
  ...
done
export V41_CED_ROLE=$role SPEC=0 PREFIX=0 DRAFT_GRAPH=0 PATCH_MODE=mount
```

改动：把 `PREFIX` 从硬门里拿掉，改为**默认 0、显式放行**（例如
`V41_CED_ALLOW_PREFIX=1` 才允许 `PREFIX=1`），这样默认口径不变、
实验臂可开，且不会有人误以为已经验收。

### 2.2 调度器边界断言（`experimental/ced/core_scheduler_replay.patch`）

```python
# 现状（D 侧）
if replay_end != prompt_len - 1 or request.num_computed_tokens != replay_end:
    raise RuntimeError("CED replay prefix boundary mismatch: ...")
request.ced_replay_start = max(0, replay_end - ced_replay)
request.num_computed_tokens = request.ced_replay_start
```

`num_computed_tokens != replay_end` 这个**等号**假定"D 一定从 P 装满了 N−1 个 token"。
开缓存后 D 自己的前缀缓存也会命中，实际装载量取决于
`(D 本地命中) + (从 P 拉来的新块)`，于是这个等号会误伤。

改动方向（**不能只改成 `<=`**）：把它变成两个独立的检查

* `replay_end == prompt_len - 1`（P 的截断契约，保持严格）；
* `num_computed_tokens >= replay_end - ced_replay`（**必须**覆盖要被重放的 128 个位置；
  少于这个值就必须先把缺的部分算出来，而不是直接重置游标）；
* 重置后 `ced_replay_start = max(0, replay_end - ced_replay)` 不变；
  若 `num_computed_tokens < replay_end`，重放步数由 §2.3 之外的现有
  `num_new_tokens` 上限逻辑自然补齐。

⚠️ 这里最容易踩的是"缓存命中把 KV 装进来了、但 replay 需要的 SWA 页仍是空"——
见 §2.3。改动必须与 §2.3 一起做，单独放开等号会静默算错。

### 2.3 D 侧预清零（`experimental/ced/mooncake_hybrid_connector.py`）

```python
# 现状（worker 的 NPU 线程，start_load_kv 里）
for group_idx in range(7, 12):          # G7..G11 = 上半层 SWA
    local_ids = meta.local_block_ids[group_idx]
    for block_id in local_ids:
        tensor.narrow(0, int(block_id), 1).zero_()
```

清零的目的：G7–G11 的上半层 SWA **从未由 P 计算过**，D 只有 replay 会写这 128 个
位置对应的约 2 页；其余页必须保证是"零"（等价空块），否则 attention 会读到
**上一次请求的残留**。

开缓存后的冲突：`meta.local_block_ids` 里会包含**被复用的命中块**（hashed），
它们与其他请求共享。无条件清零会**破坏别的请求的缓存**；
而不清零，若该页不是本次 replay 要写的页，就会留下残留。

两个可选方案：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A（推荐先试）** | 把 G7–G11 排除在前缀缓存的 scope 之外（它们本来就不跨请求复用：SWA 只保留 128 token，复用收益极小） | 要动 cache 分组/哈希配置，但语义最干净 |
| B | 清零只对 **unhashed** 块做，hashed 块额外加一条"该页必须落在本次 replay 覆盖范围内"的断言（不满足就报错而不是静默） | 不改配置，但需要在 worker 侧拿到 replay 覆盖的页号 |

现状里**唯一已经在用 `get_unhashed_block_ids_all_groups()` 的地方**是
`update_state_after_alloc`（决定从 P 往哪些块拉数据），也就是说"只写 unhashed"
这个口径在**拉取**路径上已经有了，清零路径还没有对齐。

## 3. 顺序与成本（建议）

| 步骤 | 内容 | 成本 | 依赖 |
|---|---|---|---|
| 1 | 基线 `PREFIX=1` 冒烟（§1） | 1 次重启（P+D，约 7 min）+ 2 条请求 | 无 |
| 2 | 若 1 通过：实现 §2.1 + §2.2 + §2.3A 到一个独立分支 | 半天量级（含离线自检） | 1 |
| 3 | CED 臂开缓存验收：144K 与 1M 同前缀两次，判据 `cached_tokens>0` + 答案正确 + 与不开缓存的输出一致 | 每档 2 条请求 | 2 |
| 4 | 若 2/3 失败：在验收表里把这一项明确标为**上游/架构阻断**并给出证据（§1 的结论） | 0 | — |

**不建议**在没有 §1 结论前动 §2：如果基础设施本身不支持，
§2 的三处改动只是徒增风险。

## 4. 执行结果（2026-09-26 00:00–00:40）：**基础设施可用，且命中路径正确**

§1 的冒烟已在**非 CED 基线**上跑完（`PREFIX=1`、P/D 双侧、`num_blocks=29076`、
`MULTISTREAM=0 DSA_OVERLAP=0`），结论如下。

### 4.1 命中确实发生（用服务端 metrics 判定）

`usage.prompt_tokens_details.cached_tokens` 在**命中的时候也是 0** —— 这个字段
经 PD 代理不传递，**不能当判据**。真正的判据是服务端计数器：

```
vllm:prefix_cache_hits_total                     # P 与 D 两侧都有
vllm:prompt_tokens_by_source_total{source="local_cache_hit"}
```

两轮独立探针（`/tmp/pfx_probe.py`、`/tmp/pfx_probe2.py`）：

| 步骤 | wall | P 的 hits 增量 | D 的 hits 增量 | 响应里的 `cached_tokens` |
|---|---:|---:|---:|---:|
| 冷 prefill（新 prompt） | 22.9–26.6 s | 0 | 0 | **0** |
| 同一 prompt 第 2 次 | 2.86 s | **144,000** | **144,000** | **0** |
| 同一 prompt 第 3 次 | 1.19 s | **144,000** | **144,000** | **0** |
| 换一个同长度的新 prompt | 22.89 s | 0 | 0 | **0** |

⇒ **前缀缓存在这套 hybrid KV + Mooncake + packed 布局上是可以工作的**，
而且 P 与 D 两侧的计数器同步增长。§1 之前担心的"基础设施可能不支持"被否掉。

### 4.2 命中路径的**正确性**也是对的

换用**带针**的 prompt（针插在 80% 深度）再测，判据是答案对不对 + 冷/热是否一致：

| prompt | 步骤 | wall | D hits 增量 | 答案正确 | 与上一步 |
|---|---|---:|---:|---|---|
| P1 | cold | 5.60 s | 0 | ✅ `RB9N-6014` | — |
| P1 | hit1 | 0.98 s | 144,000 | ✅ | **相同** |
| P1 | hit2 | 1.00 s | 144,000 | ✅ | **相同** |
| P2（不同语料位置） | cold | 5.78 s | 0 | ✅ | — |
| P2 | hit1 | 1.02 s | 144,000 | ✅ | **相同** |
| P2 | hit2 | 1.01 s | 144,000 | ✅ | **相同** |

⇒ **冷/热两路给出逐字节相同的正确答案**，两个不同 prompt 各 3 次全部一致。
**"缓存命中正确性"这一项在基线口径上是成立的。**

### 4.3 加速比

端到端 wall：冷 **5.6–5.8 s** → 命中 **0.98–1.02 s**，即 **约 5.7×**；
在另一组（更长思考和 64 token 输出）里是 22.9 s → 1.19 s，约 **19×**。
两者差异来自冷启动是否已过 warmup —— 冷值本身在 5.6 s 与 22.9 s 之间波动
（同一台机、同长度、同 PREFIX 口径），**这个波动尚未归因**，标为待查。

### 4.4 因此 CED 这一项现在要这样描述

| 口径 | 状态 |
|---|---|
| 基线 PD（非 CED） | **可用且正确**（本节的实测） |
| CED | **仍被启动硬门挡住**（`PREFIX=0`），所以 §2 的三处还是要动 |

也就是说：这一项**不是**"上游/架构阻断"，而是"CED 侧还没接上"。
§2 的三处改动因此是有意义的，而且现在有基线可作对照。

## 5. 当前状态

* §1 的冒烟：**已完成**（§4），结论是基础设施可用、命中路径正确。
* §2 的三处（启动硬门 / 调度器边界断言 / D 侧预清零）：**代码未改**。
  现在可以做，且必须同时处理 §2.3 的 hashed 块问题（CED 的上半层 SWA 从未由 P
  计算过，命中块里的残留会直接污染重放）。
* 验收表里"缓存命中"一行：基线口径 **已验证**，CED 口径仍记 **N/A（待实现）**。
