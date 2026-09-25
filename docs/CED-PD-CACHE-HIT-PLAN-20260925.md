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


## 6. 实测追加（2026-09-26 00:49–01:20）：CED 臂的账**与预测不同**

把 `PREFIX=1` 改成"默认拒绝、可显式放行"（`V41_CED_ALLOW_PREFIX=1`）之后实测：

| 上下文 | 结果 |
|---|---|
| **144K** | **可用且正确**：6 次探针 + 6 次交错共 12 次请求全对，命中 144,000 token，冷/热逐字节一致 |
| **1M** | **先"命中却不加速"（103.6 s vs 冷 96.7 s），随后 D 引擎崩溃** |

崩溃点：

```
RuntimeError: CED decoder expected 12 KV cache groups without DSpark
  ← mooncake_hybrid_connector.py::start_load_kv
```

决定性证据是崩溃前调度输出里的

```
num_common_prefix_blocks=[7055, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

即**前缀命中只对 group 0 给出公共前缀块，其余 11 个 group 为 0** ——
hybrid/SWA 布局下这是必然的，但它让 CED"每请求 12 个形状一致的 group 列表"的契约失效。

### 6.1 §2 的三条预测，实测后只中了一条

| §2 预测的障碍 | 实测 |
|---|---|
| §2.1 启动硬门 | ✅ 存在（已改成可显式放行） |
| §2.2 调度器 `num_computed_tokens != replay_end` 断言 | ❌ **未触发**（144K 6 次 + 1M 3 次均无 boundary mismatch） |
| §2.3 D 侧 hashed 块预清零污染 | ❌ 144K 交错测试 6/6 正确、与冷一致；1M 未走到 |
| **（新）连接器 12-group 形状假设** | ✅ **1M 命中时必然触发并杀死引擎** |

⇒ **下一步该修的是连接器**（先于调度器断言）。完整证据见
[`../evidence/ced_prefix_hit_20260926/CED_SIDE_RESULT.md`](../evidence/ced_prefix_hit_20260926/CED_SIDE_RESULT.md)。

### 6.2 第二个待查项：1M 命中为什么不加速

144K：冷 31.5 s → 热 1.2 s（26×）。
1M：冷 96.7 s → 热 103.6 s（**没有收益**）。

⇒ 1M 的端到端时间**不由 P 的 prefill 计算主导**。这与 144K 上
"D 只干 0.27 s、其余等 P"的逐请求剖析一致不了，说明 1M 上 P 有一段不吃缓存收益的
路径（候选：P→D 的 KV 传输、或 P 侧非 group-0 的重复计算）。**未归因。**


## 8. 第三轮（2026-09-26 01:40–02:15）：**1M 整池命中打通，16×**

补了两道防线后重跑，结果是本轮最大进展：

| 用例 | 结果 |
|---|---|
| **1M 整池命中**（`N=128×7813+1`） | **3/3 正确；96.44 s → 6.05 s（≈16×）**，命中 1,000,064 tok |
| 144K 整池命中（`N=128×1125+1`） | 6/6 正确 |
| 144K 常规命中 | 6/6 正确 |
| **1M 部分命中**（`N=902909`） | **打死 P**：`assert num_new_tokens > 0`（stock scheduler.py:1063） |

两道防线：

1. `[CED-GROUP-DIAG]`：那条"expected 12 KV cache groups"断言现在打印**实际**形状
   （组数 + 每组块数 + `num_external_tokens` + req id），下一次能直接定位。
2. 整池命中（`num_external_tokens == 0`）时把上游 stock 的裸 `[]`
   规范成 12 个空列表。⚠️ **但本轮该分支一次都没走到**
   （`[CED-FULL-HIT]` 计数 = 0），所以**不能声称它修复了第一轮的 D 崩溃**。

新暴露的 P 侧故障是**独立**的：P 没有装 CED 调度补丁，所以那是
stock vLLM + P 的"截去最后一个 token" + 前缀缓存 的交互。

完整证据与下一轮的最小实验见
[`../evidence/ced_prefix_hit_20260926/CED_SIDE_RESULT.md`](../evidence/ced_prefix_hit_20260926/CED_SIDE_RESULT.md)。


## 10. 第四轮（2026-09-26 02:30–03:25）：**两个崩溃都修好了，CED 口径缓存命中打通**

上两轮各留了一个崩溃。这轮先**诊断**（不是猜）再修，两条都已验证：

| # | 位置 | 原报错 | 根因（实测） | 修法 |
|---|---|---|---|---|
| 1 | **P** | `assert num_new_tokens > 0`（stock scheduler.py:1063） | 整段本地命中：`num_tokens=num_computed=local=902912`（=128×7054）、`external=0`、`WAITING`、`max_tokens=1` ⇒ `num_new_tokens=0`。stock 只在 `_update_waiting_for_remote_kv` 里做"整段命中重算末 token"，**P 走不到那里** | 用 vLLM 自带的 `truncate_computed_blocks()` 把命中拉回**上一个 128 对齐边界**（902912→902784），尾部重算 |
| 2 | **D** | `assert RequestStatus.is_finished(req.status)`（scheduler.py:3060） | D 整段命中 ⇒ `num_external_tokens==0` ⇒ 请求不进 `WAITING_FOR_REMOTE_KVS`；但连接器仍注册一次接收用于给 P 回 ack ⇒ worker 报 `finished_recving` 时请求已是 `RUNNING` | 加第三分支：空接收（没拉数据、不用还块）记 `[CED-KVRECV]` 后返回 |
+
+两条修复都只在 CED 角色下生效，且各有独立开关可关
+（`V41_CED_P_HIT_FIX=1`、`V41_CED_KVRECV_NOOP=1`）。
+
+### 修复后的实测（原来必崩的用例现在全过）
+
+| 用例 | 修复前 | 修复后 |
+|---|---|---|
+| `N=902909`（部分命中→整段命中） | 连两轮打死 P | **6/6 正确，88.5 s → 5.2 s（≈17×）** |
+| `N=1000065`（1M 整池命中） | 上一轮已通（16×） | **无回归：105.7 s → 6.0 s（≈18×）**，3/3 正确 |
+
+所有命中答案与冷路径**逐字节相同**；两个容器日志里
+`AssertionError` / `EngineDeadError` 计数都是 **0**。
+
+### 更正上一轮的一句话
+
+上一轮我说"第二轮的 D 崩溃没复现，所以那个 `[] → 12 个空列表` 的规范化没有被验证"。
+**现在有证据了：它是对的** —— 修复 #2 之前请求已经能穿过那条形状检查、
+走到 `assert RequestStatus.is_finished`。
+
+### 构建坑（踩了两次，记下来）
+
+启动器的 sha 门 `533eed493cb...` 是
+**镜像原始文件 + `patches/admission_gate.patch`** 之后的内容，
+**不是**镜像原始文件（`c67bda2886...`）。生成补丁必须先把 admission gate 打上再 diff；
+在容器里 `git checkout` 会把 admission gate 一起抹掉。
+
## 11. 当前状态（2026-09-26 03:25 更新）

* §1 的冒烟：**已完成**，基础设施可用、命中路径正确（§4）。
* 基线口径：**可用且正确**（§4）。
* CED 口径：**144K 可用；1M 整池命中可用且 16×；1M 部分命中会打死 P**（§8）。
* 下一步：先复现并修 P 的 `assert num_new_tokens > 0`（stock 侧），
  再回头确认第一轮那个 D 崩溃是否还会出现（诊断消息已就位）。
* 验收表里"缓存命中"一行：基线口径 **已验证**；
  CED 口径 **144K 与 1M 整池命中已通过，1M 部分命中阻断**。
