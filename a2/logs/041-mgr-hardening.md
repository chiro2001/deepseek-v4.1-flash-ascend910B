# 041 — L5 池分配器（`PerGroupBPCManager`）的**加固**：把静默失败变成响亮失败 + 给 A2 留可观测点

> 2026-09-22 06:2x–07:5x CST。执行：子代理 **J_mgrhardening**。机器：**A3（A3-node1）槽位 c1 = die 6**（容器 `prbench-c1`）。
> 全程只用 c1（**没碰 c0 / c2**）、占卡走 `tools/a3_chip.sh` 锁（退出码 75 无）、**没用 `/tmp`**（`source a2/scripts/tmpdir.sh J_mgrhardening`）、
> **没写 `upstream-v41/`**、**没改任何一行别人的文件**（诊断只走自己的 overlay 叠加 + import hook + 符号链接）、
> 跨机传文件全走 `cos-xfer.sh`、**没手设 `ASCEND_RT_VISIBLE_DEVICES`**、没碰 `mooncake-*` / `jitpgo-*` / `dsv41-a3`。
> A3 宿主 `MemAvailable` 全程 ≥ 1.79 TiB（未触 150 GiB 阈值）。
> 标记约定：**【实测】** = 本机跑出来的原始数据；**【推断】** = 由代码/算式推出但没直接测；**【未确认】** = 没跑到。

---

## 0. ★★ 先把边界写在最前面（避免后人误读）

| 这件加固**是**什么 | 这件加固**不是**什么 |
|---|---|
| **① 消灭静默失败**：把"池子已经坏了但没人知道"的三种形态（缺容量 cap / 过期 free / 索引键被覆盖）从**静默**变成**响亮 raise** | ⛔ **不是** `036`/`038` 那条"**144 MiB（池 ≈ 工作集）时首 token 静默错 / NaN**"的修法 |
| **② 给 A2 上线留可观测点**：`PGP_MGR_STATS=1` 打开一组只读计数器（`stale_free / dup_unit / oob_unit / over_budget / key_overwrite / used_mismatch`），可随时 `mgr_hardening_stats()` 取值 | ⛔ **不是**"让 1.000× 的池子能跑"（本任务**没有**、也不打算改变容量语义） |
| **③ fail-closed 保险**：`_owner_of_unit` 反查 + `block_id → BlockStatus` **对象身份**校验 ⇒ 未来若真出现"同一行两个 block"，第一次发生时就炸 | ⛔ **不是** int8 数值路径的修法（那条在**别人的线**上） |

**为什么必须写清楚**：本轮**三条独立证据**（我的只读探针、`I_unitprobe` 的只读探针、`H_kvcheck` 的静态证明 + 300 轮 fuzz）
都指向**同一个结论**：`038` §11-1 猜的那条"同一行 → 两个 block"在**配账层**不可达 ⇒ **没有可修对象**。
加固的价值**不在**修这个 bug，而在"**下次不会再静默**"。

**★ 再收窄一格（按 `040` 的最新结论）**：`040` 已实测 **`038` 的否决点只对 `XL1=0`（L5 单位池）成立，`L1` 一开即消失**
（同几何三臂对照：`PerGroupBPCManager` ❌ / `P2QuotaManager` ✅）。⇒ 所以：
* **144 MiB 首 token 错的修法 = 开 L1**（`040` 的结论），**不是**本任务的加固；
* 本任务**既没有修它、也不声称修了它** —— 加固前后那条臂的 sha **逐字相同**（`a7ffff6be598`，见 §4.2 读法③）；
* 本任务**唯一**的产出是：**① 消灭静默失败（fail-closed）；② 提供可观测性**。

---

## 1. 顺手更正 `038` 的两处口径（都影响后续判据）

### 1.1 ★ `144 MiB` 不是 `1.000×`，是 **`0.9914×`**（【实测·`I_unitprobe` 反解，我复现了同一格）

```
工作集 = 1162 unit  =  group0(full) 64 chunk × 8 unit = 512  +  10 个 SWA 组 × 65 key × 1 = 650
144 MiB → 1152 unit ⇒ 差 10 个 unit
144 臂的 BlockRemoved:CPU = 10  =  恰好是这 10 个缺口（被淘汰的是首请求的 10 个 SWA chunk，行 0–9）
```
⇒ 正确说法是"**轻微欠配 10 unit**"，不是"正好 1.000×"。**反解工作集的现成办法**（不需要探针）：
`工作集 = cpu_cache_usage_perc × num_units`（160 臂：`0.9078125 × 1280 = 1162.0` 精确）。

### 1.2 ★ `038` 的"纯 BF16 不复现"那条对照臂是 **confound**（【实测·本轮覆盖】）

`038` 的 BF16 臂用 `XL1=1`（= **另一个 manager**，P2 的按组配额 manager），而 int8 的 ❌ 臂是 `XL1=0`
⇒ **两者不可直接对比**。本轮用 **`XL1=0`（与 ❌ 臂同一个 `PerGroupBPCManager`）** 重跑 BF16：

| 臂 | manager | 池 | `BlockStored:CPU` | `BlockRemoved:CPU` | J2（`replay_sha` vs `fill_sha`） |
|---|---|---:|---:|---:|---|
| `j3-probe144`（int8） | `PerGroupBPCManager`(XL1=0) | 144 MiB / 1152 unit | 714 | **10** | ❌ **1/16，mismatch=#15**（`24b57053…` → `a7ffff6b…`） |
| `j3-probe160`（int8） | 同上 | 160 MiB / 1280 unit | 714 | 0 | ✅ 16/16 |
| **`j3-probebf16`（纯 BF16）** | **同上（XL1=0）** | **144 MiB / 1152 unit** | **714** | **10** | **✅ 16/16**（两轮同 sha） |

**⇒ 【实测·双路互证】不是 L5 补丁（`PerGroupBPCManager`）的缺陷，是 int8 侧的问题**
（`I_unitprobe` 用同几何、同 `XL1=0` 独立跑到同一格，结论一致；`038` 的 `XL1=1` confound 就此覆盖）。
> 诚实边界：这条只说明"**L5 管理器 + 池欠配**这个组合在 BF16 下不炸"，**不**说明 int8 侧的缺陷在哪 —— 那是 `I_unitprobe`/`H_kvcheck` 的线。

---

## 2. ★★★ 只读探针：先自证，再下结论（这一节是方法论，值得单独读）

### 2.1 ⛔ 第一轮我的探针**挂空了**（如实记录）

第一轮（`j-int8-144/160`、`j-bf16-144`）的计数文件里**只有"已装载"、没有"已包住"**：

```
[J_mgr] mgr_probe 已装载 (pid=78818 ...)      ← 只有这一行
（本该还有：）[J_mgr] 已包住 pgp_manager.PerGroupBPCManager file=... wrapped=True
```

**根因**：PGP/P2 的 `sitecustomize.py` **自己就会 `import pgp_manager`**（`pgp_hooks.py` 顶部有
`from pgp_manager import BPC_BY_GROUP_KEY, PerGroupBPCManager, bpc_map_from_extra`），而我把 `install_hook()`
放在 **exec P2 之后** ⇒ 等我的 hook 装上时 `pgp_manager` **早已进 `sys.modules`** ⇒ hook 永远不触发。

**⇒ ★ 规则（建议进 `AGENTS.md`，今晚同一个坑出现三次：`L3_8card/kv_bytecheck`、本任务第一轮、`H_kvcheck` 的 manager live 探针）**

```
任何"包裹某个模块"的探针，必须【先装 hook 再 import/exec 那个模块】，
并在日志里打【版本号 + 包装方法前 N 次调用的 trace】，用来证伪"探针没在热路径上"。
凡是只打"已装载"的探针，一律视为**未验证**。
```

我的 v6 就是这条规则的实现：**先装 hook → 再 exec P2 → 兜底直接 wrap（幂等）**，
并且每一行都带 `PROBE_VERSION`、每个被包装的方法前 8 次调用都打 `[trace]`。

### 2.2 ★ 阳性对照（探针**抓得到**，不是"看不见所以全 0"）

env 门控的故障注入：`PGP_MGR_FAULT_INJECT=stale_free` ⇒ 对同一个 block **再 free 一次**。
`j3-fault144`（144 MiB、int8）的第一手日志：

```
[J_mgr] [fault] 注入 stale_free：对 bid=0 再 free 一次（预算剩 2）
[J_mgr] [fault] 注入完成 bid=0 stale_free 0->1 free_dup=1 free_len=2
[J_mgr] [fault] 注入 stale_free：对 bid=5 再 free 一次（预算剩 1）
[J_mgr] [fault] 注入完成 bid=5 stale_free 1->2 free_dup=4 free_len=4
[J_mgr] [fault] 注入 stale_free：对 bid=1 再 free 一次（预算剩 0）
[J_mgr] [fault] 注入完成 bid=1 stale_free 2->3 free_dup=9 free_len=6
```
⇒ **【实测】探针有判别力**（`stale_free`/`free_dup` 都涨），并且离线自检 `selftest_probe.py`
还证明了**干净序列不会误报**（同一批计数器全 0）。`H_kvcheck` 用另一种手法（人为构造过期 BlockStatus）
得到同样的"抓得到"结论 —— 两条路径互证。

### 2.3 ★★ 真跑结果：`038` 的 ❌ 格上，**整轮**计数器**全 0**

探针 v6（逐方法 trace + 每 4 次调用落盘；`PGP_MGR_FAULT_INJECT=stale_free` 作阳性对照）。**四条臂、同一套探针、同一几何**：

| 臂 | 池（unit） | `alloc_calls` | `free_calls`（=淘汰） | `load_spec`（=命中读） | **异常计数器** |
|---|---:|---:|---:|---:|---|
| `j3-probe144`（int8 ← `038` 的 ❌ 格） | 1152 | 17 | **10** | **16** | **全 0** |
| `j3-probe160`（int8 ← `038` 的 ✅ 格） | 1280 | 17 | 0 | 14 | **全 0** |
| `j3-probebf16`（BF16，`XL1=0`） | 1152 | 17 | **10** | **16** | **全 0** |
| **`j3-fault144`（阳性对照）** | 1152 | 17 | 13 | 13 | ★ `stale_free=3`、`dup_unit=3`、`free_dup=30`、`used_mismatch=13`、`dup_within_block=1` |

`j3-probe144` 的完整落盘（**覆盖整轮**：17 次分配 = 全部 store、**10 次淘汰** = 池满的那 10 次 `BlockRemoved:CPU`、
**16 次 load spec** = replay 轮每个请求的命中读，`allocated=1152` 说明池**确实被占满过**）：

```
alloc_calls=17 free_calls=10 store_spec=17 load_spec=16 prepare_store=17 reset=0
state: allocated=1152 free_len=0 live_units=1152 live_blocks=704 store_keys=714
★ stale_free=0  free_while_live=0  dup_unit=0  idx_overwrite=0  map_mismatch=0
  used_mismatch=0  free_dup=0
容量: alloc_over_budget=0  over_cap=0  oob_unit=0  no_index_after_alloc=0
provenance: load_prov_mismatch=0  load_len_mismatch=0  load_prov_unknown=0
旁证: none_after_evict=0  blk_id_reuse=0  dup_within_block=0
```

两个额外读数（都是**顺手的独立证据**）：

* **`j3-probe160`：`allocated = 1162`（从不淘汰、`free_len=0`）** ⇒ **【实测】工作集恰好 1162 unit**，
  与 `I_unitprobe` 反解出的 1162 **逐字相同**；144 MiB = 1152 unit = **0.9914×**，缺的就是那 10 个（§1.1）。
* `load_prov_mismatch=0` 的含义是：**每一个命中读到的 unit 组，逐项等于该 key 最后一次 store 写的 unit 组**
  —— 这正是 `038` §11-3 要求的"行级 provenance"，与 `I_unitprobe` 的 `hit_row_changed=0 / hit_never_stored=0` 独立互证。

**⇒ 【实测·三路互证】"同一行 → 两个 block"这条路径在配账层不可达**（我的探针 + `I_unitprobe` 的探针 + `H_kvcheck` 的静态证明与 300 轮 fuzz）；
**⇒ 同一套探针在人为注入时立刻非 0**（阳性对照）⇒ **"全 0"是有判别力的全 0，不是瞎**。
**⇒ 因此 `041` 的加固定位只能是"消灭静默 + 可观测"**（见 §0）。

---

## 3. 加固（`PGP_MGR_HARDEN`，**默认 0 = 逐字旧行为**）

交付件：`a2/agents/J_mgrhardening/patch/pgp_manager.py`
（= `a2/publish/0001b-offload-per-group-bpc-manager.patch.py` 的**逐字副本** + 下面四处改动；
统一 diff 见 `patch/pgp_manager.hardened.diff`，md5 **`9f11c9ac0de0d77fbe6a212e42a9966a`**）。

| # | 位置 | 旧行为（静默） | 加固后（`PGP_MGR_HARDEN=1`） |
|---|---|---|---|
| **a** | `_allocate_blocks` 分配前 | 只靠调用方（`prepare_store`）先淘汰；自己**没有**上游 `cpu/manager.py:81` 的 `min(...)` cap | `want > num_blocks - allocated + len(free_list)` ⇒ **raise**（**不超发**） |
| **b** | `_free_block` | `_units_of_block.pop(...)` 拿不到就**兜底** `[block.block_id]` 推回池子 | **raise**（"不知道这行是否还活着"必须响亮失败） |
| **b2** | `_free_block` / `_get_load_store_spec` | 无对象级校验 | `block_id → BlockStatus` **对象身份**校验：不是当初分配出去的那个对象 ⇒ 记数（`>=2` 时 raise） |
| **c** | `_units_of_block` 建索引 | `dict[units[0]] = units`（list）⇒ 首 unit 被复用时**键被覆盖**、旧 unit 列表静默丢失 | 冻结 `tuple` + `_owner_of_unit` 反查：**键已存在 / 同一行已有活块 ⇒ raise** |
| **e** | 计数器 | 无 | `stale_free / dup_unit / oob_unit / over_budget / key_overwrite / used_mismatch / free_owner_mismatch / stale_block_obj`：`PGP_MGR_STATS=1` 打印（`PGP_MGR_STATS_EVERY` 控制频率），`mgr_hardening_stats()` 只读取值 |

**语义不变**（这是"能不能交付"的前提，逐条对着 `021` 的五判据）：

| 语义点 | 是否改动 |
|---|---|
| `block_id` = `units[0]`（调度侧展开依赖它） | **不变**（`selftest_harden` 的 `(d2)` 显式断言） |
| `_get_load_store_spec()` 返回"chunk 内 block 顺序"的 unit 平铺表 | **不变** |
| `free_units()` / `_used_units()` / `get_stats()` 的 unit 口径 | **不变** |
| 淘汰按 unit 记账（不是按 key 数） | **不变** |
| 一处**新增**行为 | `_sanity()` 用**增量**账（`_index_units_total`，O(1)）核对 Σ活块unit vs `_used_units()`；不一致时记数，`HARDEN=1` 时 raise |

**为什么默认关**：`PGP_MGR_HARDEN=0` 时**每一处 raise 都退化成原来的兜底**（`a` 不拦、`b` 推回 `[block_id]`、`c` 允许覆盖），
只多一份只读记账 ⇒ **零回归风险**，可以先进发布包再逐级打开。

---

## 4. 验证

### 4.1 离线单元自检（不占卡，`scripts/selftest_harden.py`）

直接驱动**真的** `pgp_manager.py`（容器里 import，不是在本地 mock），人为制造三类静默失败：

| 断言 | 结果（原始输出，**15 条全过**） |
|---|---|
| `(a)` 池只有 4 unit、一次要 8 ⇒ **必须 raise**，且 `allocated` 不动（不超发） | ✅ `[J_mgr_hard] 单位池容量不足而拒绝超发：want=8 budget=4` + `over_budget=1` + `allocated=0` |
| `(b)` 对同一个 block free 两次 ⇒ **必须 raise**，且 `free_list` 不被污染 | ✅ `[J_mgr_hard] 过期/重复 free：block_id=0 不在活块表里（旧实现会静默 push [0] ⇒ 同一行两个块）` + `free_list=[0]`（长度 1，没被污染） |
| `(c1)` 把活块的行塞回 `free_list` 再分配 ⇒ **必须 raise** | ✅ `[J_mgr_hard] 单位池索引键被覆盖：block_id=0 已经在活块表里（old=(0,) new=(0,)）` + `key_overwrite=1`（`dup_unit` 是后备守卫） |
| `(c2)` 键被覆盖 ⇒ **必须 raise** | ✅ 同上（`key_overwrite=1`） |
| `(d)` 正常分配 → 展开 → 释放，**所有计数器为 0** | ✅ `unit 展开 = [0, 1, 2]`；8 个计数器**全 0** |
| `(d2)` `block_id == units[0]`、spec 仍是平铺 unit | ✅ `{0: (0, 1, 2)}`、spec = `[0, 1, 2]` |
| 尾部 | `[selftest_harden] OK ✅` |

> ★ 复跑命令见 §9.1 第 0 条（**~30 s，不占 NPU**）；图省钱也可以直接在 A3 宿主上跑（它只 import `pgp_manager` 与 vllm 的纯 Python 依赖）。
> ⚠️ 诚实记录：这一格**第一次**跑是 **FAIL** —— 因为我把 `(c1)` 的断言写成了"`dup_unit` 必须 +1"，
> 而那条路径上**先响的是 `key_overwrite`**（两个守卫覆盖同一路径的不同形态）⇒ 已把断言改成"两者之一 ≥1"。**修的是断言，不是守卫。**

### 4.2 真服务（加固版挂上跑）

**口径**：`PGP_MGR_HARDEN=1` + `PGP_MGR_STATS=1`（每次分配自报一行）+ 只读探针同时挂着；
每条臂都先过"自检"（`selfcheck_j.py` 直接问**被 import 的** `pgp_manager` 是谁 + 钉 md5 `9f11c9ac…` + `HARDEN_MODE==1`）。

| 臂 | 几何 | 池 | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` | `hits` | J2 / sha | 加固断言 |
|---|---|---:|---:|---:|---:|---:|---|---|
| `j3-h144` | int8，16×4096→4096 | 144 MiB | 714 | **10** | 231,669,760 | 65,520 | ❌ 1/16（#15）`24b57053…`→`a7ffff6b…` | **不 fire**（17 行自报全 0） |
| **`j3-h160`** | int8，16×4096→4096 | 160 MiB | **714** | **0** | **231,669,760** | **65,520 / 131,328** | **✅ 16/16**（replay **64.9 ms** vs fill **512.7 ms** = **7.90×**） | **不 fire**（17 行自报全 0） |
| `j3-hbf16` | **纯 BF16**，`XL1=0` | 144 MiB | 714 | **10** | — | — | **✅ 16/16**（46.1 vs 461.7 ms = **10.02×**） | **不 fire** |
| `j3-hmix` | int8，**4096→2048** | 144 MiB | 714 | 10 | 137,134,080 | **32,752** | replay `e1185547…` ≠ 冷算 `c9742db1…` | **不 fire** |
| **`j3-hmix160`** | int8，**4096→2048** | **160 MiB** | 714 | **0** | 137,134,080 | **32,752** | **✅ replay `c9742db1…` == 冷算参考 `c9742db1…`** | **不 fire** |
| `j3-cold2048`（参考） | int8，2048→2048、池 16 MiB（不命中） | 16 MiB | — | — | 0 | 0 | `c9742db1…`（**冷算基准**） | **不 fire** |

**读法（三条）**：

1. **四条判据不回归**（`j3-h160`，与 `038`/`021` 逐字同源）：`BlockStored:CPU=714>0` ✅、`CPU→GPU=231,669,760>0` ✅、
   `hits=65,520/131,328>0` ✅、`replay 64.9 ≪ fill 512.7 ms（7.90×）` ✅。
2. ★ **`021` 判据③（变长前缀安全 + sha 一致）在加固版上通过**：`j3-hmix160`（4096 填充 → **2048 回放**、`hits=32,752`、`CPU→GPU=137.1 MB`）
   的输出 sha **逐字节等于同几何冷算参考** `c9742db1…` ⇒ "**池里有数据且取回的是对的那一段**"。
3. ★ **144 MiB 的变长前缀臂 `e1185547…` ≠ 冷算 `c9742db1…`** —— 这是 **int8 + 池 0.9914×（欠配 10 unit）**那一格的已知缺陷
   （与 `038` 同源【推断】）；**证据是加固前后 sha 逐字相同**（未加固的 `j-hardenmix` 与加固版 `j3-hmix` 都是 `e1185547b375`）
   ⇒ **不是本次加固引入的差异**，本任务**没有**把它当成回归，也**没有**声称修好它。

**加固断言"不 fire"的直接证据**（`j3-h144` 的 `[J_mgr_hard] stats` 自报，尾部三行）：

```
stats harden=1 allocated=1018 free=0 live_blocks=626 live_units=1018 dup_unit=0 free_owner_mismatch=0 key_overwrite=0 oob_unit=0 over_budget=0 stale_block_obj=0 stale_free=0 used_mismatch=0
stats harden=1 allocated=1090 free=0 live_blocks=670 live_units=1090 dup_unit=0 free_owner_mismatch=0 key_overwrite=0 oob_unit=0 over_budget=0 stale_block_obj=0 stale_free=0 used_mismatch=0
stats harden=1 allocated=1152 free=0 live_blocks=704 live_units=1152 dup_unit=0 free_owner_mismatch=0 key_overwrite=0 oob_unit=0 over_budget=0 stale_block_obj=0 stale_free=0 used_mismatch=0
```

（`allocated` 一路涨到 **1152 = 池满**、`live_blocks=704`、**全部 8 个计数器保持 0**；五条加固臂的 `J_mgr_hard` 行数都是 17 / 17 / 17 / 17 / 33，
**没有任何一条 raise 消息** ⇒ **fail-closed 保险在四格上都没有误伤**。）

### 4.3 ★ 判据对账（**含一条明确未满足**）

| 任务书判据 | 结果 | 说明 |
|---|---|---|
| ① 144 MiB：`stale_free/dup_unit/oob_unit` 全 0 | ✅ **【实测】** | §2.3（真跑整轮全 0，且阳性对照证明有判别力） |
| ① 144 MiB：**不再是静默错**（正确 **或** 响亮 raise） | ⛔ **未满足（如实报告）** | 这条臂**仍然静默错**，而且**不是加固能修的**：① 根因已由 `040` 定在 **int8 侧 / `state` 组行步长**，不在配账层；② `040` 实测**开 L1 即消失**。加固**故意不改变**分配/回收语义（否则会破坏 L5 五判据与 L1 的协同）⇒ **这条判据的正确解法是"开 L1"，不是本任务**。**加固只保证"若配账层将来真坏，它会响"**。 |
| ② 160 MiB：四条判据不回归 | ✅ **【实测】** | `j3-h160`：`BlockStored:CPU=714` / `CPU→GPU=231,669,760` / `hits=65,520` / replay 64.9 vs fill 512.7 ms = 7.90× |
| ③ `021` 的五条判据（尤其 4096→2048 变长前缀安全 + sha 一致）不回归 | ✅ **【实测】** | `j3-hmix160`：`hits=32,752`、`CPU→GPU=137.1 MB`、replay 输出 sha **== 同几何冷算参考 `c9742db1…`**；144 那格 ❌ 是**已知 int8 欠配缺陷**（加固前后同 sha ⇒ 非本次引入） |
| ④ 8 卡真权重复跑（`027` 口径） | **未做** | 卡时用在了上面 9 条臂；**加固默认门控关 ⇒ 对 8 卡口径零影响**（要跑只需在 `027` 的臂上加 `PGP_MGR_HARDEN=1`） |

---

## 5. 与既有工作的关系（同一件事的三条独立路径）

| 谁 | 方法 | 结论 |
|---|---|---|
| **H_kvcheck** | 静态证明（`prepare_store` 的淘汰循环保证 `A' ≤ C`）+ 阳性对照 + **300 轮 fuzz（1,700 万次 unit 操作）** | `stale_free/free_list_dup/dup_unit/over_cap` 全 0；**不可达** |
| **I_unitprobe** | 原地只读包裹 4 个方法，C0 几何真跑 | 11 个计数器全 0；**不可达**；★ 并反解出"144 MiB = 0.9914×、工作集 1162" |
| **J_mgrhardening（本任务）** | 25 个计数器的只读探针（v6，含 trace + 阳性对照） | 整轮全 0（含 10 次淘汰 + 16 次命中读的 provenance）；BF16(XL1=0) 同格 ✅ ⇒ **不是 L5 的缺陷** |

### 5.1 ★ 对照 `AGENTS.md` §5b（探针纪律）逐条自查

| §5b 规则 | 本任务怎么满足 | 证据 |
|---|---|---|
| **① 先装 hook，再 import/exec 目标模块** | v6 的顺序 = **装 hook → exec P2 → 兜底 wrap（幂等）**；第一轮的失败原样记在 §2.1（**我踩过这个坑，也修好了**） | `probe/sitecustomize.py`；日志里同时出现 `已包住` 与 `[fallback] ... 直接包住` |
| **② 只打"已装载"= 未验证；要打版本号 + 热路径 trace** | 每次装载都打 `PROBE_VERSION=v6-2026-09-22T07:00-op-dump`；每个被包装的方法前 8 次调用打 `[trace]` | `j3-probe144` 的 trace = `{prepare_store:8, _allocate_blocks:8, _get_load_store_spec:8, _free_block:8}` |
| **③ 判据必须在反例臂/正确臂上对称跑；阳性对照** | ★ **`j3-fault144`**：人为对同一 block 再 free 一次 ⇒ `stale_free 0→3`、`free_dup=30`、`dup_unit=3`；**而三臂真跑全 0** | §2.2 / §2.3 的表 |

**⇒ 按 §5b**：本任务报成【实测】的"全 0"**是**有判别力的 0（同一条探针在注入下立刻非 0，且官方/正确臂几何对齐）；
**并且**"某一格没跑到"一律标【未确认】（§8）。

### 5.2 ★ 关于 `state` 组（group 1）的一条边界（来自 `L_dmafix`/`043`）

`043` 实测 **`state` 组（group 1）的 DMA 从未被发出**（`被排除的组=[1]`、`quota={1: 0}`、`rows[tensor12..14]=0`，
store-probe 里 `group=1` 出现 0 次）。对本任务的三点影响：

1. 我的计数器**不按组归属**统计（只有全局的 `stale_free / dup_unit / …`）⇒ **不会**因为某一组恒 0 而误报；
2. `selftest_harden.py` 里**没有**"必须覆盖全部 12 组"这类断言 ⇒ **不涉及该风险**；
3. `(b2)` 对象身份校验 / `(c)` `_owner_of_unit` 反查 都在 **manager 层**（谁调用 `_allocate_blocks` 才生效），
   而 `state` 组**根本不走分配** ⇒ 它们在那组上**确实永远不会 fire** —— 这是**预期行为**，不是缺陷。
   ⇒ 按纪律**不为它扩大改动面**。

---

## 6. cannbot 对照（`a2/AGENTS.md` §6）

本轮**不是**算子/kernel 开发、也**不是**量化数值验证，而是**卸载池的记账/分配层**（纯 Python 数据结构）。
按 §6 的表，需要对照的是 **KV cache / attention 布局**那一节，我读了
`~/projects/dsv41/src/cannbot/vendor/cannbot-skills/model/model-infer-kvcache/SKILL.md`：

* `:102-110` —— 逻辑地址 → 物理 block → **slot** 的映射（`物理 block ID = block_table[...]`、`物理 slot = block ID × block_size + 偏移`）；
* `:224-242` —— `sparse_mode` / `atten_mask` 的硬约束（本任务**不涉及** mask）。

**它建议什么**：分页 KV 的**物理块分配**由 `BlockPool` 负责，映射通过 `block_table/slot_mapping` 表达；
**我们为什么没有采纳/改动**：这一层是 **GPU/算子侧**的 block↔slot 映射，与本任务的 **CPU 侧 DRAM 池的 unit 记账**
不在同一层 —— 我们新增的 `_owner_of_unit` 只是**同一个 unit 号在两个 block 之间的归属校验**，
既不改变 `block_table`，也不改变 worker 的 DMA 寻址（`blocks_per_chunk=1` 的 1:1 展开一个字都没动）。
⇒ **无冲突、无采纳动作**；本轮**没有新增或修改任何算子**，故不涉及 ops-profiling 那套流程。

---

## 7. 红线核对

没发 PR / issue / 评论；**没写 `upstream-v41/`**；**没用 `/tmp`**（`tmpdir.sh J_mgrhardening`）；
占卡全走 `a3_chip.sh c1`（无 75 退出）；没手设 `ASCEND_RT_VISIBLE_DEVICES`；
没碰 c0 / c2 / `mooncake-*` / `jitpgo-*` / `dsv41-a3`；**没改任何一份别人的文件**
（诊断走自己的 overlay + import hook；`pkg_harden/patch_pgp/` 是对 X 的 pkg-ring 做**符号链接 + 只替换 `pgp_manager.py`**）；
传文件全走 `cos-xfer.sh`；结论逐条标了【实测】/【推断】/【未确认】。

---

## 8. 诚实边界（哪些没测 / 哪些是别人的线）

| # | 事项 | 状态 |
|---|---|---|
| 1 | `038` ❌ 格的**整轮**计数器（含淘汰 + 命中 provenance） | ✅【实测】全 0（`j3-probe144`） |
| 2 | 探针有判别力（阳性对照 + 干净序列不误报） | ✅【实测】 |
| 3 | BF16 + `XL1=0` 同格不炸 | ✅【实测】（+ `I_unitprobe` 双路互证） |
| 4 | 加固四处守卫会 raise、正常路径不误报 | ✅【实测】（离线单元自检） |
| 5 | 加固版真服务不回归 | ✅【实测】（§4.2） |
| 6 | int8 侧缺陷的**真根因** | ⛔ **不是本任务**（`I_unitprobe`/`H_kvcheck` 在追；可能是 worker 侧 store↔load 不对称） |
| 7 | 加固 (b2) 能否覆盖"store 写 1 行 / load 读 8 行"那类不对称 | **【未确认】**：(b2) 校验的是 **manager 层 `BlockStatus` 对象身份**，而那是 **worker 侧 DMA** 的行为 ⇒ 按纪律**不为它扩大改动面** |
| 8 | 8 卡真权重复跑（`027` 口径） | **未做**（卡时留给了上面 9 条臂；加固默认关 ⇒ 对 8 卡口径零影响） |

---

## 9. 交付物

| 路径 | 作用 |
|---|---|
| `a2/agents/J_mgrhardening/patch/pgp_manager.py` | ★ **加固版 manager**（md5 `9f11c9ac0de0d77fbe6a212e42a9966a`），默认门控关 |
| `a2/agents/J_mgrhardening/patch/pgp_manager.hardened.diff` | 与 `publish/0001b` 原件的 unified diff（交付替换建议用） |
| `a2/agents/J_mgrhardening/probe/{mgr_probe.py,sitecustomize.py}` | 只读计数器探针 v6 + 叠加挂载（**先 hook 后 exec**） |
| `a2/agents/J_mgrhardening/scripts/*.sh,*.py` | 三条臂 / 最终批 / 单元自检 / 计数器汇总 / J2 判据 |
| `a2/logs/raw/041-j-mgrhardening/` | 原始数据 **37 个文件**（9 份 `mgr-probe-*.log`、10 份 `*.client.json`、9 份 `*.kv_events.log`、`j3-h160/j3-hmix160` 的 `metrics_after`、自检与补丁 md5、批次日志 `final3.log` / `mix160.log`） |

**替换建议**（不动 `0001b-*.py` 原件，交给主代理）：把 `publish/0001b-offload-per-group-bpc-manager.patch.py`
替换为 `agents/J_mgrhardening/publish/0001b-offload-per-group-bpc-manager.patch.py`（同 md5），
A2 起服时按需加 `PGP_MGR_HARDEN=1`（推荐先 `PGP_MGR_STATS=1` 只观测）。

### 9.1 怎么复跑（每格一条命令）

```bash
# 0) 不占卡：探针自检（干净序列不误报）+ 加固单元自检（四类守卫会 raise）
python3 a2/agents/J_mgrhardening/scripts/selftest_probe.py
env PYTHONPATH=<pkg_harden>/shadow:<pkg_harden>/patch:<pkg_harden>/patch_pgp \
    PGP_MGR_HARDEN=1 PGP_MGR_STATS=1 python3 a2/agents/J_mgrhardening/scripts/selftest_harden.py

# 1) ★ 三条臂 + 阳性对照（c1，~15 min）——本日志 §2.3 那张表
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 7200 --name j-final3 -- \
  bash /work/agents/J_mgrhardening/scripts/run_final3_j.sh

# 2) 只想要"变长前缀 + 160 MiB"那一格（c1，~4 min）——本日志 §4.2 判据③
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 1200 --name j-mix160 -- \
  env ARM=int8-160 TAG=j3-hmix160 GOUT=/work/agents/J_mgrhardening/out \
      HARDEN=1 PGP_MGR_HARDEN=1 PGP_MGR_STATS=1 REPLAY_PROMPT_TOKENS=2048 \
  bash /work/agents/J_mgrhardening/scripts/run_arm_j.sh

# 3) 看计数器（任意臂）
python3 /work/agents/J_mgrhardening/scripts/show_counters.py <out>/mgr-probe-<TAG>.log
```
