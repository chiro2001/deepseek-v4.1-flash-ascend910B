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
## 11. 当前状态（2026-09-26 03:47 更新）

> ⚠️ 本节此前是 02:15 状态（§9）的**原样搬运**：只改了标题日期、没改内容，
> 于是它说"1M 部分命中会打死 P"，与紧邻其上的 §10（同一轮已修好）**直接矛盾**。
> 2026-09-27 修正。**以本节为准**；§6/§8 的"崩溃中"是当时的中间状态。

* §1 的冒烟：**已完成**，基础设施可用、命中路径正确（§4）。
* 基线口径（非 CED）：**可用且正确**，144,000 tok 命中、5.7× 加速。
* **CED 口径：全部用例已通过**（实验臂 `PREFIX=1` + `V41_CED_ALLOW_PREFIX=1`）：

| 用例 | 结果 | 加速 |
|---|---|---|
| 144K 常规命中 | ✅ 6/6 正确 | 9.5–31.5 s → 1.2–1.3 s |
| 144K 整池命中（`128×1125+1`） | ✅ 6/6 正确 | 3.2 s → 1.2 s |
| 144K 交错命中（P1/P2 交替） | ✅ 6/6 正确 | — |
| 1M 整池命中（`N=1000065`） | ✅ 3/3 正确 | 105.7 s → 6.0 s（≈18×） |
| 1M 部分命中→整段命中（`N=902909`） | ✅ 6/6 正确 | 88.5 s → 5.2 s（≈17×） |

  所有命中答案与冷路径**逐字节相同**；两个容器日志里
  `AssertionError` / `EngineDeadError` 计数都是 **0**。

* **交付口径仍然是 `PREFIX=0`**：两条修复（P 侧对齐回退、D 侧空接收）目前仍是
  "带 kill switch 的实验改动"（`V41_CED_P_HIT_FIX`、`V41_CED_KVRECV_NOOP`，
  默认 1），且**没有重跑完整验收矩阵**（21/21）。转正需要：
  ① 把这两条连同连接器的 `[]→12 空列表` 规范化当成正式改动；
  ② 用 `PREFIX=1` 重跑 144K/1M 四针与多轮；
  ③ 把 `serve_a3_ced_pd.sh` 的硬门从"显式放行"改成默认值。

* 验收表里"缓存命中"一行：基线口径与 CED 口径**都已验证**。

> **2026-09-27 更新：已转为交付默认。** 见 §12/§13 的复现验证，以及
> `scripts/serve_a3_ced_pd.sh`（D 侧 `SPEC=1 DRAFT_GRAPH=1`、两侧 `PREFIX=1`，
> decode 默认图模式）、`patches/files/model.py`（引擎侧同门默认 1）、
> `deploy/a3-ced-pd/launch/_common.sh`（`PREFIX` 默认 1）。
> 防回归自测：`tools/selftest_ced_defaults.sh`（11 项，已并入 selfcheck）。

---

## 12. 复现验证（2026-09-27 01:27–01:55，权威快照 `main@a02c0a7`）

§4–§10 的证据是 09-26 在当时的临时包上做的。2026-09-27 用**当前 main 的干净
快照**（`~/cedpd-repo`，`git status` 干净、`tools/selfcheck_pkg.sh` 全绿）重跑了一遍，
确认结论没有因为后续改动而回归。本次口径：P/D 都 `PREFIX=1 V41_CED_ALLOW_PREFIX=1`、
`SPEC=0 DRAFT_GRAPH=0 STATIC_KERNEL=0 MULTISTREAM=0 DSA_OVERLAP=0`、
`num_blocks=29076`、两侧 `bind=127.0.0.1`、D 侧带解码护栏
（`[V41-DECODE-GUARD] middleware loaded` 出现 1 次）。

### 12.1 正确性（判据：答案 = 针值 **且** 冷/热逐字节相同）

| 用例 | 冷 | 热 | D hits 增量 | 结果 |
|---|---:|---:|---:|---|
| 144K P1 | 29.20 s | 2.79 / 1.22 s | 144,000 | ✅ 3/3，冷热一致 |
| 144K P2 | 10.90 s | 1.21 / 1.20 s | 144,000 | ✅ 3/3，冷热一致 |
| 1M 整池命中 `N=1000065` | 94.31 s | 5.67 / 5.66 s | 1,000,064 | ✅ 3/3（≈16.6×） |
| 1M 部分命中→整段命中 `N=902909` | 88.72 s | 5.01 / 4.93 s | 902,912 | ✅ 3/3（≈17.9×） |

### 12.2 三个修复**在真机上各自触发**（不是"传了开关"，是日志里的痕迹）

| 修复 | 判据 | 本次实测 |
|---|---|---|
| P 侧命中回退 | `[CED-P-HIT] … truncated to 902784 (step=128…)` | **2 次** |
| D 侧空接收 | `[CED-KVRECV] no-op recv for running req …` | **2 次** |
| 12-group 形状规范化 | `[CED-FULL-HIT] … 整池命中（num_external_tokens=0）` | **2 次** |

两侧日志 `AssertionError` / `EngineDeadError` 计数均为 **0**。

### 12.3 前缀缓存计数器（两侧都非零，这才是"真命中"）

| 端口 | `prefix_cache_queries_total` | `prefix_cache_hits_total` | `local_cache_hit` |
|---|---:|---:|---:|
| 18990 (P) | 6,572,958 | 4,497,920 | 4,498,176 |
| 18991 (D) | 6,572,970 | 4,381,952 | 4,381,952 |

### 12.4 面向用户的端到端（这一节是本次新增的覆盖）

§4–§10 只跑了探针，没测用户路径。本次补上：

| 用例 | 结果 |
|---|---|
| 经代理纯文本 | ✅ `3+4` → `7`，`finish_reason=stop` |
| **流式**（此前未在 `PREFIX=1` 下测过） | ✅ 8 帧 + `[DONE]`，拼出 `1\n2\n3\n4\n5` |
| **并发 4 路**短请求 | ✅ 4/4 正确，0.51–1.40 s |
| Responses API 8787（2 图） | ✅ 200，分别认出两张图 |
| 代理探活 `/v1/models`、`/healthcheck` | ✅ 200 / `{"status":"ok","prefill_instances":1,"decode_instances":1}` |
| 直连 D 的事故形状请求（护栏负控） | ✅ 400，紧接着 `/health` 仍 200（引擎存活） |

### 12.5 这次跑完，**还没覆盖**什么

* 144K/1M 的**四针**验收（本次是一针 + 两图），不是完整 21 项矩阵；
* **DSpark 开 + PREFIX=1** 的组合：本次 `SPEC=0 DRAFT_GRAPH=0`。
  当前线上那套（`SPEC=1 DRAFT_GRAPH=1`）与缓存同开是**未测组合**；
* 多轮长会话（只测了单轮 + 流式 + 并发）。

⇒ 结论：**`PREFIX=1` 在 CED 口径下可用且正确（含用户路径），但"DSpark + 缓存"
这一格仍是空白**。要把交付默认值改成 1，应先补这一格与四针。

---

## 13. DSpark 与缓存**同开**（2026-09-27 02:03 起）

§12 留的空白已补。配置：权威快照 `main@a02c0a7`；P 侧 `SPEC=0`，
D 侧 `SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 STATIC_KERNEL=1`，**两侧都
`PREFIX=1 V41_CED_ALLOW_PREFIX=1`**。与 §12 的关系是**单变量**：
只把 D 从 `SPEC=0 DRAFT_GRAPH=0` 改成 `SPEC=1 DRAFT_GRAPH=1`。

起服判据（都是可观测痕迹）：两侧 `--enable-prefix-caching`；
D `"num_speculative_tokens":7,"enforce_eager":false`；draft 图标记 32 处；
D 的组数从 12 变成 **13**（`upper SWA=(7,8,9,10,11) draft=(12,) total=13`）。

### 13.1 正确性：全部通过

| 用例 | 冷 | 热 | 结果 |
|---|---:|---:|---|
| 144K P1 | 46.60 s | 3.01 / 1.12 s | ✅ 3/3，冷热一致 |
| 144K P2 | 10.84 s | 1.16 / 1.20 s | ✅ 3/3，冷热一致 |
| 1M 整池命中 `N=1000065` | 95.13 s | 6.36 / 6.23 s | ✅ 3/3（≈15.2×） |
| 1M 部分命中→整段命中 `N=902909` | 89.73 s | 5.85 / 5.78 s | ✅ 3/3（≈15.4×） |

12 次请求答案全部 `RB9N-6014` 且与冷路径逐字节相同；两个容器
`AssertionError` / `EngineDeadError` / `RuntimeError` 计数**全为 0**。

### 13.2 性能三元组（128K 单流、128 token、`--ignore-eos`，n=2）

| rep | TTFT | prefill tok/s | **ms/step** | **A** | **decode tok/s** |
|---|---:|---:|---:|---:|---:|
| 0（冷） | 9.829 s | 13,333 | **32.68** | **3.10** | **94.79** |
| 1（命中） | **1.102 s** | 118,927 | **32.55** | **2.61** | **79.63** |

* 命中把 TTFT 从 9.83 s 压到 1.10 s（**8.9×**），decode 不受影响
  （32.68 → 32.55 ms/step）——符合"缓存只省 prefill"的预期。
* `A` 2.43–3.10 落在历史区间（§CED-PD-DSPARK-*: 2.4–3.4），**远大于 1.0**
  ⇒ 草稿确实在产出，不是 `DRAFT_GRAPH` 静默失效那种形态。
* 口径提醒：本节 `ms/step` 是**开了 DSpark + 缓存**的数字，与 §3 的
  `DRAFT_GRAPH=0`、以及 30.2 ms/step 的"单机同置无 DSpark"**不可直接相减**。

### 13.3 用户路径（全过）

| 用例 | 结果 |
|---|---|
| 经代理纯文本 | ✅ `6*7` → `42`，`finish_reason=stop` |
| 流式 | ✅ 5 帧 + `[DONE]`，内容 `1\n2\n3\n4\n5` |
| 并发 4 路 | ✅ 4/4 正确（2/4/6/8），0.53–4.77 s |
| Responses API 8787（2 图） | ✅ 200 completed，分别认出两张图 |
| 代理 `/v1/models`、`/healthcheck` | ✅ 200 / `{"status":"ok",...}` |
| 直连 D 的事故形状请求（护栏负控） | ✅ 400，紧接着 `/health` 仍 200 |

### 13.4 ★ 新发现：DSpark 开着时，**D 侧本地缓存一次都不命中**

这是本节最值得记的一条，且与 §12（`SPEC=0`）形成明确对照：

| | §12 `SPEC=0` | §13 `SPEC=1 DRAFT_GRAPH=1` |
|---|---|---|
| P `local_cache_hit` | 4,498,176 | 4,629,248 |
| **D `local_cache_hit`** | **4,381,952** | **0** |
| D `external_kv_transfer` | 2,191,006 | **6,836,237**（≈全量） |
| D `prefix_cache_hits_total` | 4,381,952 | **0**（但 queries 正常增长） |

D 的缓存**在查询**（`prefix_cache_queries_total` 与 P 同步增长）却**从不命中**，
于是每个请求都从 P 拉全量 KV。两个附带结果：

* §10 修的 `CED-KVRECV`、`CED-FULL-HIT` 两条路径在**本配置下不再触发**
  （计数均为 0）——它们要求"D 本地整池命中"，而本配置下 D 永远不命中；
  `CED-P-HIT` 仍然触发（2 次），因为 P 侧命中照旧。
* 端到端仍然快（1M 6.2 s），因为省掉的是 **P 的 prefill**，P→D 的本机
  KV 传输相对便宜。

**机制（【推断】，未单变量实测）**：`HybridKVCacheCoordinator.find_longest_cache_hit`
用**不动点迭代**让每个 attention group 收敛到同一个命中长度
（`vllm/v1/core/kv_cache_coordinator.py:609+`）。DSpark 多出的第 13 组是
**只含草稿层的 SWA 组**，它的 prompt KV 既不由 P 传输、D 又只重放 128 token
⇒ 该组的块**永远进不了 D 的缓存** ⇒ 协调后的命中长度被它拉到 0。
`SPEC=0` 时没有这一组，所以 12 组能收敛到全量命中。

【未确认】的是"第 13 组是否真的参与那次 min"——要证实需要在 D 侧开
`VLLM_LOGGING_LEVEL=DEBUG`（或加一条 per-group hit 的插针）重跑一次。
**当前不影响正确性，只影响"D 本地省一次传输"这一层优化**。

### 13.5 仍未覆盖

* 144K/1M 的**四针完整矩阵**（本次仍是一针 + 两图）；
* 多轮长会话；
* 13.4 的机制验证（需要一次带 DEBUG 的重启）。

---

## 14. 默认值转正与「默认 == 交付口径」的真机验证（2026-09-27 02:31）

§11–§13 之后，DSpark 与前缀缓存已**转成 A3 PD 分离的默认**。本节只验一件事：
**不带任何功能覆盖启动时，解析出来的配置是否真的就是交付口径**
—— 因为"改默认值"最容易的失效方式是"默认被某个分支覆盖回旧值"，
而 `bash -n` 查不出来。

### 14.1 改了什么（`scripts/serve_a3_ced_pd.sh`）

| 项 | 旧默认 | 新默认 | 关掉的办法 |
|---|---|---|---|
| D 侧 `SPEC` / `DRAFT_GRAPH` | 0 / 0 | **1 / 1** | `V41_CED_ALLOW_DSPARK=0`（退回 0/0）或 `SPEC=0` |
| `PREFIX`（两侧） | 0 | **1** | `PREFIX=0`（旧写法 `V41_CED_ALLOW_PREFIX=0` 也认） |
| `STATIC_KERNEL` | 0（两角色） | **D=1 / P=0** | 显式覆盖 |
| decode 的执行臂 | 必须显式二选一 | **默认图模式**（+ 图模式硬前提） | `CED_DIAGNOSTIC_EAGER=1` 走 eager 诊断臂 |

两处配套（漏一处就会在别的层炸）：

* `patches/files/model.py` 用**同一个 env** 做引擎侧的门，其默认必须一起改成 1
  —— 否则默认启动会在模型构造期被拒；
* `deploy/a3-ced-pd/launch/_common.sh` 的 `PREFIX` 默认也要跟上，
  否则两种交付面口径不一致。

### 14.2 新增防回归自测：`tools/selftest_ced_defaults.sh`（11 项，已并入 selfcheck）

它**既查默认解析、也查门仍然咬人**：

| 类别 | 用例 |
|---|---|
| 负控（门必须咬） | `prefill + SPEC=1`、`decode + SPEC=2`、`decode + DRAFT_GRAPH=2` 都必须被拒 |
| 正控（默认解析） | decode 默认 = `spec=1 draft=1 prefix=1 static=1`；prefill 默认 = `spec=0 draft=0 prefix=1 static=0` |
| 显式关闭仍有效 | `V41_CED_ALLOW_DSPARK=0`、`PREFIX=0`、`V41_CED_ALLOW_PREFIX=0` |
| 两个交付面一致 | `model.py` 的默认、`_common.sh` 的 `PREFIX` 默认都必须与脚本一致 |

> ★ 这个自测**当场抓到两个真 bug**（都是我加默认值时引入的，`bash -n` 都查不出来）：
> ① 默认路径下 `$SPEC` 裸引用 ⇒ `set -u` 直接 `SPEC: unbound variable`，
>    也就是"不带任何 env 启动 D"会崩；
> ② 外层条件已用 `${SPEC:-1}` 而内层检查仍是 `${SPEC:-0}` ⇒ 把默认值判成非法并 `exit 2`。
> 两处已修。这正是"判据本身也要被检验"的价值。

### 14.3 真机：只给部署必需项，其余全交给默认

重启命令里**只**传 `MODEL / PATCH_MODE / NAME / RUN_ID / PORT / KV_PORT / DEVS`，
功能开关一个都不传（`main@31e637c`）。脚本自己打印的解析结果：

```
[a3-ced] role=prefill name=dsv41-ced-p2b max_len=147456 spec=0 prefix=1 graph=1 eager=0
[a3-ced] D 侧 DSpark：SPEC=1 DRAFT_GRAPH=1（交付口径）
[a3-ced] D 图模式（交付口径）：GRAPH=1 EAGER=0，prompt-tail eager 已就位
[a3-ced] role=decode  name=dsv41-ced-d4b max_len=147456 spec=1 prefix=1 graph=1 eager=0
```

引擎侧的实际生效痕迹：

| 判据 | P | D |
|---|---|---|
| 命令行 prefix 开关 | `--enable-prefix-caching` | `--enable-prefix-caching` |
| `--speculative-config` 出现次数 | **0**（DSpark 不该在 P） | 1（`{"method":"dspark","num_speculative_tokens":7,"enforce_eager":false}`） |
| `enable_static_kernel` | `false` | `true` |
| `num_blocks` | 29076 | 29076 |
| 解码护栏 | — | `middleware loaded` ✓ |
| **图模式硬前提**（请求后打印） | — | **128 次** |

### 14.4 功能自测（同一实例，全部默认值）

| 用例 | 结果 |
|---|---|
| 144K P1/P2 冷·热·热 | ✅ 6/6 正确，冷 17.35/11.29 s → 热 1.15–1.22 s |
| 经代理纯文本 | ✅ `8*9` → `72`，`finish_reason=stop` |
| 流式 | ✅ 5 帧 + `[DONE]`，内容正确 |
| 并发 4 路 | ✅ 4/4 正确（3/6/9/12），0.52–4.71 s |
| Responses API 8787（2 图） | ✅ 200 completed，分别认出两张图 |
| 代理探活 | ✅ 200 / `{"status":"ok","prefill_instances":1,"decode_instances":1}` |
| 直连 D 的事故形状请求 | ✅ 400，紧接着 `/health` 仍 200 |
| 接受长度 | **3.33 / 2.43**（DSpark 真在产出） |
| 缓存命中（P 侧） | `local_cache_hit=576,000`，`prefix_cache_hits_total=576,000` |
| 两侧错误计数 | `AssertionError` / `EngineDeadError` / `RuntimeError` **全 0** |

### 14.5 两个仍然存在的边界（如实标注）

1. ~~raw 脚本的 `MAX_LEN` 默认是 147456（144K）~~ → **2026-09-27 已改为 1M**
   （`scripts/serve_a3_ced_pd.sh` 显式 `export MAX_LEN=${MAX_LEN:-1048576}`），
   与 `deploy/` 形态同口径，两种交付面不再有窗口差异。
   容量提醒：默认 KV 池 = `num_blocks=29076` ⇒ **3.72M tokens**，
   `MAX_SEQS=4` 时四路同时满 1M 会超出池，引擎自行限流（不会崩）。
   收窄窗口用显式 `MAX_LEN=<值>`。
2. 用 `PREFIX=1` 的 **21 项完整矩阵尚未重跑**（本次是 144K/1M 探针 + 流式 +
   并发 + 两图 + 用户链路，见 §13/§14.4）。KIT-README 的 21/21 仍是
   `PREFIX=0` 口径跑的。

---

## 15. 默认窗口改到 1M，并在 1M 上复测（2026-09-27 03:07–03:35）

### 15.1 改动

`scripts/serve_a3_ced_pd.sh` 增加 `export MAX_LEN=${MAX_LEN:-1048576}`，
置于 `exec serve_a3_pd.sh` 之前 ⇒ 覆盖后者 147456 的兜底。
于是"用脚本起"与"用 `deploy/launch` 起"窗口一致，尾部 `echo` 的兜底值同步改。

**容量提醒**（写进注释与 §14.5）：默认池 `num_blocks=29076` ⇒ **3.72M tokens**，
`MAX_SEQS=4` 时四路同时满 1M 会超出池 ⇒ 引擎自行限流，不会崩。

### 15.2 真机：只给部署必需项，默认解析出 1M

| 判据 | P | D |
|---|---|---|
| `--max-model-len` | **1048576** | **1048576** |
| `--enable-prefix-caching` | ✓ | ✓ |
| `--speculative-config` | 无（符合架构） | `dspark, sp=7, enforce_eager=false` |
| `enable_static_kernel` | false | true |
| `num_blocks` | 29076 | 29076 |
| 图模式硬前提（请求后） | — | **104 次** |
| 解码护栏 | — | `middleware loaded` ✓ |

### 15.3 1M 功能验证（判据 = 答案对 + 冷热逐字节相同）

| 用例 | 冷 | 热 | 结果 |
|---|---:|---:|---|
| 1M 整池命中 `N=1000065` | 137.73 s | 7.96 / 6.13 s | ✅ 3/3（≈17×） |
| 1M 部分命中→整段命中 `N=902909` | 88.97 s | 5.71 / 5.71 s | ✅ 3/3（≈15.6×） |

用户路径同样全过：代理纯文本 `9*9`→`81`、流式 5 帧 + `[DONE]`、
并发 4 路 4/4（0.51–4.77 s）、Responses API 两图 200 completed、
代理探活 200、直连 D 负控 400 且引擎存活。

接受长度 **2.43 / 3.50**（DSpark 在产出）；两侧
`AssertionError` / `EngineDeadError` **全 0**。

> 缓存命中来源与 §13.4 的观察一致：P 侧 `local_cache_hit=3.81M`，
D 侧 `local_cache_hit=0`、`external_kv_transfer=5.71M`
—— DSpark 开着时 D 不做本地命中（§13.4 的机制待验证那条仍成立）。

### 15.4 一处操作注意（不影响运行）

发布这次改动时 **a3-21 到 github.com 的 443 不可达**（
`Failed to connect to github.com port 443 after 134524 ms`），
所以现场是通过 `tar` 直接同步 `7497a06..10da076` 的改动文件到 a3-21 的
`~/cedpd-repo`，**没有**走 `git fetch`。因此那台机器上
`git rev-parse HEAD` 仍显示 `7497a06`，而**实际运行的脚本字节 = `10da076`**
（已用 `grep export MAX_LEN` 核实 `1048576`）。
等网络恢复后 `git -C ~/cedpd-repo fetch && git reset --hard origin/main` 即可对齐，
**不需要重启服务**（脚本已在内存里解析完，重启才会重读）。
