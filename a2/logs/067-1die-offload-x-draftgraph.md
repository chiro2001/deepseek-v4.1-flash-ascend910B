# 067 — 单卡：「DRAM 卸载 × `DRAFT_GRAPH=1`」能否复现（Q1/Q2 判决）+ 一处**只属于 tiny 几何**的取回归零

> 2026-09-22 17:0x–17:5x CST（**远端 A3-node1 时间**；本机比远端快约 7 min）。执行：子代理 **`C1_offload_draftgraph`**。
> 机器：**A3（A3-node1）的 c1 = `prbench-c1`**（每臂都走 `tools/a3_chip.sh c1`）。**全程只用 c1**，未碰 c0（`T_draftceiling`）/ c2（另一个子代理）/
> `dsv41-a3`（保持 Exited）/ `mooncake-master` / 别人的容器 / Phy-ID 0–7 与 8–15；未手设 `ASCEND_RT_VISIBLE_DEVICES`；未用 `/tmp`（容器内只用 `/work`，宿主用 `~/tmp/20260922/C1_offload_draftgraph`）；
> 未写 `upstream-v41/`；**未发 PR / issue / 评论**；每臂起服前查 `/dev/shm`（`Used 28K`）。
> 产物：本日志 + 原始数据见 §7 + `agents/C1_offload_draftgraph/`。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【推断】**；**【未确认】**。

---

## 0. 一句话

| 问题 | 答案 |
|---|---|
| **Q1：单卡能不能跑「卸载 × `DRAFT_GRAPH=1`」？** | **【实测】能，而且 draft 真的进了图** —— `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL`（子代理核过：这一行在历史所有臂里是 **0 次**）、`use_cuda_graph=True: 51 / =False: 0`、`EE1016=0 / Not_Supported=0 / invalid GM address=0 / 507057=0`、输出 sha 与对照臂**逐字相同**。 |
| **Q2：`DRAFT_GRAPH` 0/1 对卸载有区别吗？** | **【实测】没有可观测差异**：单变量两臂（0↔1、51↔0 干净翻转）`CPU_to_GPU` **都是 0**、`external hits` **都是 0**、输出 sha **跨臂逐字相同**。⇒ 这条与 8 卡 P1 的结论**互相独立地一致**。 |
| **★ 但 Q2 的"都是 0"没有判别力** | 因为**同一个 workload 在"没有 draft 组"时是 273 MB / 65,520 命中**。⇒ **【实测】在单卡 tiny 几何下，draft 组（g12）参与卸载 ⇒ 取回整轮归零**（池从 0.9× 扫到 410× 全部归零），**把 g12 从参与位摘掉 ⇒ 即便池子刚好 1.000× 也恢复**。 |
| **★ 这条能不能外推？** | ⛔ **不能**。8 卡真权重（**draft 组参与位也是 `True`**）取回是 21.52 / 21.19 / 12.11 GB ⇒ 正确表述是「**单卡 tiny 几何下**，g12 参与卸载 ⇒ 取回整轮归零」【实测·限定】。见 §6 边界。 |
| **★ 根因定位到哪一层？** | **【未确认】** + 一条**必须先排除的观察者效应**：探针证明大池下 g12 的尾部 chunk **确实在池里**（`last_hit = n_keys-1`），而同一个 `_lookup` 里 `_sliding_window_lookup` 返回 0 —— **自相矛盾**，而我那个探针**每条 miss-scan 多调 32 次 `manager.lookup()`**，**可能自己扰动了被测量对象**。见 §5。 |

---

## 1. 为什么这条值得用 c1 的时间换（任务书里写的 Q1/Q2）

三个轴（**draft 入图 / kv8 / DRAM 卸载**）在 8 卡上**从未同时开过**（全仓 18 条 8 卡臂都是 `DRAFT_GRAPH=0`），
而 A2 生产**正是 `DRAFT_GRAPH=1`**。子代理在动手前先核了**三件很容易搞错的事**（都影响"能不能复现"这个前提）：

### 1.1 ★★ 历史上"graph 臂"没有一个是真 draft 入图

| 位置 | 实际情况 |
|---|---|
| `D_draftINT8/scripts/d_arm.sh:61` | `SPEC='{"method":"dspark",...,"enforce_eager":true}'` **写死**；`GRAPH=1` 只加 `--compilation-config` ⇒ **主模型入图 + draft 永远 eager** |
| `T_draftceiling/scripts/run_2c_arm.sh:117` | **硬编码 `DRAFT_GRAPH=0`** |
| 所有 tiny 影子包的 `dspark_proposer.py` | md5 **全是 `dac256ad…` = stock**，第 75 行 `self.use_cuda_graph = False` 无条件覆盖 |

⇒ 真 `DRAFT_GRAPH=1` 需要**装 draft 版三文件** + `DSPARK_GRAPH_CAPTURE_METADATA=1`。**本日志是单卡上第一次真的做到。**

### 1.2 判据名修正（本 build 里没有那个指标）

`kv_offload_block_stored_total` **不存在**；真的是 **`kv_offload_store_bytes_total`** + ZMQ 事件（`BlockStored:GPU/CPU`）。
★ 子代理第一版 grep 到它 = 0，差点写成"没存"。

### 1.3 影子包的做法（可复跑）

```
底座 = T_draftceiling/pkg/B（≡ X_integrate/pkg-ring 的逐字副本；含 D2 的参与位修复：D2_offload=24 处、_offload_participates=4 处）
叠加 = dsv41-release/patches/files/draft 的三件（= serve_a2.sh 在 DRAFT_GRAPH=1 时装的那三个）
```
★ **兼容性先做成机械门**（这是最大的风险，已排除）：

| 文件 | 相对容器 stock 的 diff | md5（台账一致） |
|---|---|---|
| `attention/dsa_v1.py` | 删 **3** 行 / 增 321 行 | `371bb023e2ecb97c01a484fb9b8cef05` |
| `spec_decode/dspark_proposer.py` | 删 **8** 行 / 增 479 行 | `5565afed64b7fe282fa9d622af7cf206` |
| `spec_decode/llm_base_proposer.py` | 删 **21** 行 / 增 892 行 | `a24076eb387e41eae1c97d724f853823` |

⇒ **同一 lineage 的超集补丁，无版本错位**（三个删行数与预期逐字相同，脚本里 fail-closed）。

---

## 2. Q1：单卡能起、能跑、draft 真的在图里【实测】

`c1-g1-smoke`（`MODE=g1` = `DSPARK_DRAFT_USE_CUDAGRAPH=1` + `DSPARK_GRAPH_CAPTURE_METADATA=1` + 四件套 + 主模型 `FULL_DECODE_ONLY`）：

```
[llm_base_proposer.py:951] [spec_decode/base] Wrapping draft model with ACLGraphWrapper:
                           runtime_mode=FULL, use_eagle=True, enable_enpu=False   ← ★ 历史 0 次
Capturing CUDA graphs (decode, FULL):   0/25 ...
dspark-graph-probe 行数 = 51 ；use_cuda_graph=True 命中 51 ；=False 命中 0
组清单：(12, 128, 128, 1, 1, 8, is_eagle=True, participates=True)   ← 路线图担心的那一格
错误码：EE1016=0  Not_Supported=0  invalid GM address=0  EngineDeadError=0  507057=0
```

**对称对照（`MODE=g0`，唯一变量 = `DSPARK_DRAFT_USE_CUDAGRAPH`）**：

| 判据 | `c1-g0-pair` | `c1-g1-pair` |
|---|---|---|
| `Wrapping draft model` | **0** | **1** |
| `use_cuda_graph=True` / `=False` | **0 / 51** | **51 / 0** |
| EE1016 / 507057 / MTE | 0 / 0 / 0 | **0 / 0 / 0** |
| 输出 sha（fill / replay） | `24b57053…` / `24b57053…` | **`24b57053…` / `24b57053…`（跨臂逐字相同）** |
| `CPU_to_GPU` | **0** | **0** |
| `external_prefix_cache_hits` | **0 / 131,328** | **0 / 131,328** |

⇒ ① `DRAFT_GRAPH` 自变量**翻得很干净**；② **对卸载取回无可观测差异**；③ **不影响输出**。
★ 但 ④ 也正因为两臂都是 0，**这条"没有差异"本身不构成"没有交互"的证据** —— 见 §3。

---

## 3. ★★ `DRAFT_GRAPH` 不是自变量：**g12 参与卸载**才是

### 3.1 全部臂（同一影子包 / 同 workload `16 × 4096` / `max_tokens=1` / `ROUNDS=2` / 同 `bpc={default:8,swa:1}`）

| 臂 | 模型 | g12 参与 | 池(unit) | 工作集(unit) | 比值 | `CPU→GPU` | `ext hits` |
|---|---|---|---:|---:|---:|---:|---:|
| `c1-nodraft` | **model-tiny（无 g12）** | — | 1152 | 1152 | **1.000×** | **2.7295744e8** | **65,520** |
| `c1-g0-pair` | tiny-draft | ✓ | 1152 | 1280 | 0.900× | 0 | 0 |
| `c1-g1-pair` | tiny-draft | ✓ | 1152 | 1280 | 0.900× | 0 | 0 |
| `c1-excl-g0` | tiny-draft（摘 g12） | ✗ | 1152 | 1152 | **1.000×** | **2.7295744e8** | **65,520** |
| `c1-excl-g1` | tiny-draft（摘 g12） | ✗ | 1152 | 1152 | **1.000×** | **2.7295744e8** | **65,520** |
| `c1-pool256-g0` | tiny-draft | ✓ | 2048 | 1280 | **1.600×** | **0** | **0** |
| `c1-pool256-g1` | tiny-draft | ✓ | 2048 | 1280 | **1.600×** | **0** | **0** |
| `c1-g1-xl1` / `c1-bigidiag-g1` / **`c1-bigidiag-g0`** | tiny-draft | ✓ | 32768 | 1280 | **25.6×** | **0** | **0** |
| `c1-g1-smoke` | tiny-draft | ✓ | 32768 | ~80 | **~410×** | **0** | **0** |

★ **正例是逐字复现的**：`2.7295744e8` 与 `G_kv8fix/out/g-bf16-4096`、`I_unitprobe/i-bf16-144` **完全相同**；
`65,520` 与 `logs/013` 的 `v1-d2-4g` **完全相同** ⇒ **harness 没问题，0 是真差异**。

★ **Q2 在探针层面又被确认一次**：大池那一对臂（`c1-bigidiag-g0` / `c1-bigidiag-g1`，唯一变量仍是 `DRAFT_GRAPH`）
给出**完全相同的内部读数** ——
```
[C1_HITIDX] group=2 n_keys=32 hit_idx=[7, 15, 23, 31] need_window=1 last_hit=31   （g0 与 g1 逐字相同）
```

### 3.2 ★★ 池子假说被否决（这是主代理提出、子代理实跑的判别臂）

主代理给的候选 (b) =「加了 draft 组 ⇒ 工作集变大 ⇒ 144 MiB 池从"刚好够"变成不够 ⇒ 触发 `013` 的断崖归零」。
**判据是对称的**：池 144 → 256 MiB，draft 仍在场。

**结果：256 MiB（1.600×）仍然 `CPU_to_GPU = 0`、`hits = 0`。** 再加上 25.6× / 410× 两档也是 0：

```
池 0.900× → 0        池 1.000×(摘 g12) → 273 MB
池 1.600× → 0        池 1.000×(摘 g12) → 273 MB
池 25.6×  → 0
池 ~410×  → 0
```
⇒ **池子/工作集比值不是自变量**（1.600× 已越过 `013` 的 1.000× 拐点）。

### 3.3 但 (b) 那一臂的直接收益：把**两个独立机制**拆开了

子代理加了一个**只打印、不改行为**的探针（把 miss-scan 的 `present` 展开成下标），跑了两档池：

```
池 144 MiB：24 条 C1_HITIDX **全是 group=0 且 hit_idx=[]**   ← ★ 连一个 key 都没有 = 纯淘汰
池 4 GiB ：replay 轮 group=2  hit_idx=[7, 15, 23, 31]  n_keys=32  last_hit=31
```
⇒ ★★ **「池小（淘汰）」与「g12 参与」是两个机制，在 144 MiB 那一格叠加**。
（也说明"0.900× 在归零区"只能解释一半：**即便 25.6× 也归零**。）

---

## 4. ★★ 主代理要求的关键检查：synthetic g12 与真 mtpq draft 组 **spec 逐项相同**

主代理指出：若 (b) 成立 ⇒ 子代理的 Q1/Q2 表述要打折（draft 组是合成的）。
**这条是零成本、且决定整条结论解释力的，子代理先做了它。**

| 字段 | `model-tiny-draft`（合成） | `DeepSeek-V4.1-Flash`（8 卡真权重） |
|---|---|---|
| `head_dim` | 512 | 512 |
| `num_key_value_heads` | 1 | 1 |
| `sliding_window` | 128 | 128 |
| `num_nextn_predict_layers` | 3 | 3 |
| `dspark_target_layer_ids` | `[37, 38, 39]` | `[37, 38, 39]` |
| `dspark_block_size` | 5 | 5 |
| `hc_mult` | 4 | 4 |
| `num_hidden_layers` | 40 | 40 |
| `compress_ratios`（43 项） | **逐项相同** | **逐项相同** |
| ★ **group 表第 12 行** | `(12, 128, 128, 1, 1, 8, True, True)` | **`(12, 128, 128, 1, 1, 8, True, True)`——逐字相同** |

⇒ ★★★ **(b) 被排除**：**合成的 g12 与真 mtpq draft 组在 spec 层逐字相同**（含 `is_eagle`、参与位、`bpc`、`tokens_per_chunk`、`sw_chunks`、`alignment_chunk_count`）。
⇒ 这**加强**了 §3 的结论（不是"合成构造的产物"），但也**同时加强**了 §6 那条边界（同样的 spec、8 卡却好 ⇒ 差异在**规模/几何**）。

---

## 5. ★★ 根因：**未确认**，而且有一条**必须先排除的观察者效应**

### 5.1 探针抓到的"自相矛盾"

大池（4 GiB）臂的 replay 轮，同一次 `_lookup` 调用内：

```
[C1_HITIDX] group=2  n_keys=32  hit_idx=[7, 15, 23, 31]  need_window=1  last_hit=31  first_hit=7
```
按 `_sliding_window_lookup` 的语义（从 idx=len-1 往前扫，攒够 `required_window` 个连续 HIT 就返回 `idx + window`）：
**idx=31 是 HIT、窗口=1 ⇒ 应当返回 32，不是 0。** 而实际 `num_hit_chunks == 0`（否则不会打 miss-scan）。

★ 旁证：`hit_idx=[7,15,23,31]` 与 store 侧 `is_store_reachable_swa_chunk` 的**算术预测逐字吻合**
（非 eagle SWA 组 `reachable_tail = sliding_window_chunks + 0 = 1`，只留 `position_in_segment = 7` 的那些 ⇒ {7,15,23,31}），
g12 则是 `tail = 1 + 1 = 2` ⇒ `{6,7,14,15,22,23,30,31}` = 8 个（与 §3.1 的工作集算术一致）。

### 5.2 ⚠️ **不能据此断言"调度器有 bug"** —— 探针可能扰动被测对象

`manager.lookup()` **可能不是只读的**（异步 lookup 的 kick-off / LRU touch 都是它的常见副作用）。
子代理的 `[C1_HITIDX]` 探针**每条 miss-scan 多调 32 次 `manager.lookup()`**（`_dbg_miss_lines` 上限 24 ⇒ 最多多调 **768 次**）。
⇒ **"探针看到 4 个 HIT、真实 lookup 看到 0 个"这个矛盾，有可能是我自己造成的。**
★ 这与 `AGENTS §5b` 第 2 条同源（"探针本身会影响被测对象"），只是这次是**副作用型**而不是"没生效型"。

### 5.3 下一步（**独立任务，不在本日志结论内**）

| # | 做法 | 判据 |
|---|---|---|
| 1 | 用**纯只读**的方式取 `manager.lookup` 的结果：在 `_sliding_window_lookup` **内部**逐 idx 打（不额外调用），**不改调用次数** | 若"内部逐 idx"也显示 31 是 HIT 而函数返回 0 ⇒ 真矛盾；若显示 MISS ⇒ 5.2 成立（探针扰动） |
| 2 | 直接比 `offload_keys[31]` 与 store 侧写入的那 8 个 key（`store-key` 行里有 chunk 号） | key 不一致 ⇒ store/lookup 的 key 推导不一致 |
| 3 | 用 `SWA_TRIM=window` 关掉"只留段尾"的裁剪，看是否恢复 | 恢复 ⇒ 病灶在 `is_store_reachable_swa_chunk` 与 `_sliding_window_lookup` 的**下标口径**不一致 |

---

## 6. 边界（**按主代理要求逐条如实标**）

1. ⛔ **不能外推到 8 卡**：8 卡真权重（draft 组参与位**也是 `True`**）取回是 **21.52 / 21.19 / 12.11 GB**、`hits 901,120`
   ⇒ 正确表述是「**单卡 tiny 几何下**，g12 参与卸载 ⇒ 取回整轮归零」【实测·限定】，**不是**"draft 组参与卸载有 bug"。
2. ⛔ **不能外推到 A2**：A2 是 8 卡真权重，走的是 §6.1 那一格（已知好）。
3. ⚠️ **`c1-g1-smoke` 那一行的"~80 unit / ~410×"是【推断】**：它的 `PROMPTS=4 PROMPT_TOKENS=1024`（与其余臂的 `16×4096` 不同）⇒ 工作集口径不同，表格里那两个数由 §3.1 的几何算术给出，**未逐臂实测**（该臂的 `CPU_to_GPU=0` / `hits=0` 是【实测】）。
4. ⚠️ **本日志的 Q1/Q2 结论与 8 卡 P1 是两条独立证据**，方向一致（都指向"没有 `DRAFT_GRAPH` × 卸载 的交互"）；
   **单卡不能替代**真权重数值 / TP8 通信 / 8 卡绝对 token 数（`063`）。

---

## 7. ★★ 判据失效清单（**本日志对后人最有用的部分**）

> 主代理指示把这几条集中写成一节。本任务 40 分钟内**自己踩了 5 条**，全部属于"**判据本身没有判别力**"。

| # | 失效的判据 | 正确的判据 | 后果（如果不修） |
|---|---|---|---|
| 1 | `*.metrics*.txt` 通配读到 **`metrics_before`** ⇒ 8 卡卸载臂"`CPU_to_GPU` 全是 0" | 精确取 **`metrics_after`** | 会得出"8 卡也没取回"的反结论（主代理当场纠正） |
| 2 | `inner.sh` 里 grep `speculative` = 0 ⇒ 以为 8 卡臂没开投机 | 读 **serve.log 的引擎初始化行**（`speculative_config=SpeculativeConfig(...)`） | 会得出"卸载 × 投机 从没跑过"的错结论（**实际跑过**） |
| 3 | `l2-dram32` 的 `CPU_to_GPU=0` 当成新问题 | 它是 `logs/013` 已记的 **0.667× 断崖** | 重复发现已有结论（主代理纠正） |
| 4 | ★ `run_arm_c1.sh:89` 把 **`P2_POOL_PATCH` 硬编码 0** ⇒ 传 `=1` 被静默忽略 | 从环境读 + **日志里 `enable=` 必须与传入值一致** | 变量根本没生效，臂"静默跑成另一档"（本日第 7 次同类） |
| 5 | `kv_offload_block_stored_total` **这个指标在本 build 里不存在** | **`kv_offload_store_bytes_total`** + ZMQ 事件 | grep 到 0 ⇒ 会写成"没存" |
| 6 | ★ **`[C1_HITIDX]` 探针每条多调 32 次 `manager.lookup()`** | 在 `_sliding_window_lookup` **内部**逐 idx 打（不改调用次数） | **探针扰动被测对象**，见 §5.2 |

★ 另有两条**子代理自己造的出包门失误**（都被门当场挡住、未污染数据）：
- `mk_pkg_c1_eagle.py` 第一版用"差异行数 == 12"，`difflib` 对齐后是 **13** ⇒ 门误触发；改成**精确判据**（去掉补丁块后与 base **逐字节相同**）。
- `mk_pkg_c1_hitidx.py` 第一版用裸 `C1_HITIDX` 数标记，但**注释里也含它** ⇒ 数成 2；改成精确标记 `[C1_HITIDX] group=`。

★ 还有一条**自检门正确拦住起服**的正面例子（fail-closed 生效）：
- 子代理把符号名写成 `_DSPARK_SWA_INDICES_RESIDENT`，真身是 **`_DSA_SWA_RESIDENT`** ⇒ 自检 `rc=3` ⇒ **没白跑一条臂**。
- 另一处：`MODE=nospec`（无 draft 的对照臂）本来 `CAPTURE_METADATA=0`，而自检把期望值写死 `True` ⇒ 误拦；已改成"**期望值由臂自己声明的 env 给出**"。

---

## 8. 交付

| 件 | 位置 |
|---|---|
| 影子包生成（含三道机械门：底座自检 / 兼容性删行数 / md5 台账 + import 自检） | `agents/C1_offload_draftgraph/scripts/prep_pkg_c1.sh` |
| 单臂运行器（`MODE=prod0|g0|g1|nospec`，显式点名每个开关，不靠继承） | `agents/C1_offload_draftgraph/scripts/run_arm_c1.sh` |
| 收证据（四条卸载判据 + draft 入图自报 + 投机四数 + 输出 sha） | `agents/C1_offload_draftgraph/scripts/collect_c1.sh` |
| import 自检（含"先预热 ops/core 再进目标三件"的**导入顺序**注释） | `agents/C1_offload_draftgraph/scripts/selfcheck_c1.py` |
| 两个锚点化补丁生成器（`--exclude-eagle` / `--hitidx`，都带**精确差异门**） | `scripts/mk_pkg_c1_eagle.py`、`scripts/mk_pkg_c1_hitidx.py` |
| 四条链（g0/g1 对照、摘 g12、池放大、大池诊断） | `scripts/chains/chain_c1_{g01,eagle,pool,hitidx,hitidx_big}.sh` |
| 工作集算术（从**日志自报**数 unit，不猜） | `scripts/chains/chain_c1_wsan.sh` |
| 原始数据 | ★ **`logs/raw/067-c1-offload-draftgraph/`**（**156 个文件 / 5.0 MB**，13 条臂；每条臂含 `{server.log, SUMMARY.txt, metrics_after.txt, metrics_before.txt, kv_events.json, client.json, kv_size.txt, meta.txt, boot_evidence.txt, server_args.txt}`）<br>★ **已按项目约定从 A3 取回本机**（走 `cos-xfer`，不是 scp）；取回后**逐项核过**：156 文件 / 5.0 MB。 |
