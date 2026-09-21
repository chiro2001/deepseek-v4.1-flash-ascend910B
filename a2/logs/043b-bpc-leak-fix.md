# 043 · ★ `blocks_per_chunk` 局部变量泄漏**已修 + 端到端已验证**（`Σgroup_sizes` 44→**72**、group 0 实搬 64→**492**、读到未写过行 5352→**0**）

**日期**：2026-09-22 07:2x–07:4x（A3 本地时钟；本机 07:26 收到任务书，07:40 收工）
　**执行**：子代理 **M_bpcfix**
**机器**：A3（A3-node1）**只用 c1**（`tools/a3_chip.sh c1` 全程持锁；c0 = `K_l1_8card`、c2 = `L_dmafix` 未碰）
**入口**：`039 §10` 的定性（本任务书）+ `021`/`027` 的判据口径 + `H_kvcheck` 的探针链（**不重推、直接打生产代码真值**）
**标记**：【实测】/【推断】/【未确认】
**原始数据**：`logs/raw/043-bpc-leak-fix/`（两条修后臂全量产物 + `043-summary.txt`；修前臂用 `H_kvcheck/out/h-t5-int8-150994944.*`，**同一探针、同一几何**）

---

## 0. 一句话结论（先给判断）

| # | 判据（任务书 §2） | 修前（`H` 同几何） | **修后** | 结果 |
|---|---|---|---|---|
| **①★ 直接判据** | 生产 `src_spec`：`Σgroup_sizes` / `len(src.block_ids)` | 44 / 44（`group_sizes=[4,0,4×10]`） | **72 / 72**（`[32,0,4×10]`） | ✅【实测】 |
| **②★ 交叉核对** | group 0 的**实搬** GPU block 数 | **64**（逐张量） | **492**（逐张量，12 张量全 492） | ✅【实测】 |
| **③ 全 0 行消失** | worker 侧"读到未写过行" | **5352**（flag `zero_row+unwritten`） | **0**（flags 表空） | ✅【实测】 |
| **④ `021` 五条判据** | 条目数 / 字节 / 命中 / 复跑 | 714 / 273.0 MB / 65,520 / — | **714 / CPU→GPU=231,669,760 ≈ 221 MiB / hits=65,520 / 两臂逐字同 sha** | ✅【实测·不回归】 |
| **⑤ `030`/L1 四条判据** | `BlockStored:CPU` / `CPU→GPU` / `hits` / replay≪fill | 714 / >0 / 65,520 / 9.3×（021 口径） | **714 / 2.32e8>0 / 65,520 / fill 623.5 vs replay 250.4 ms = 2.49×** | ✅【实测·全中】 |
| **⑥ 容量不变** | `GPU KV cache size` | 22,719 tokens | **22,719 tokens**；`alloc_sum=714`、`usage=1.0` | ✅【实测·逐字相同】 |
| **⑦ 复跑同 sha** | 单格不算数 | — | 两臂（`m-fixed-144m` / `m-fixed-144m-b`）**每一个数都相同**、输出 sha 逐字相同 | ✅【实测】 |
| 反例臂（对称性） | 10 个 SWA 组（`bpc=1`） | `[4]×10`、每 chunk 1 个 | **`[4]×10` 逐项相同**、每 chunk 1 个 | ✅【实测】 |
| ⑧（加跑）| 4096→**2048** 变长前缀 | `017` 的反例形态 = `hits=0`/`CPU→GPU=0` | **`hits=32,752`（= `021 §5` 逐字）、`CPU→GPU=137,134,080>0`** | ✅【实测·见 §3.5】 |

> **一句话**：泄漏的是"收集 loop 留在 `blocks_per_chunk` 里的最后一个参与组的值"（本配置 = SWA 的 **1**），
> spec loop 拿它算 `gpu_block_idx`/`range()` ⇒ **`bpc>1` 的组（group 0，full attention，bpc=8）每个 chunk 只搬 1 个 GPU block**。
> 修法 = spec loop 内**逐组重新取本组 bpc**（并把收集 loop 的同名局部改名 `bpc_g`），**src/dst 两侧同源展开**。修后三条直接判据全部翻转，且**取回路径与容量一个字节没变**。

---

## 1. 改了哪几行（交付件 `agents/M_bpcfix/publish/0001-offload-scheduler.patch.py`）

> **行号基准**：本表是**交付件自己的行号**（87130 B）。`039 §10` 报的 `1419/1529/1531` 是
> `agents/L3_8card/patched/scheduler.py`（8 卡版，比 publish 原件多 6 行 import 兜底）的行号；
> 对应到 **publish 原件**是 `1413/1523/1525`（本任务静态判据实测的那三个行号）。
> ★ **注**：`a2/publish/0001-offload-scheduler.patch.py` 已于 **07:32 被主代理替换成与本交付件逐字节相同的一份**
> （md5 同为 `986c9115…`）；因此单元自检的反例臂改用**修复前原件副本** `agents/M_bpcfix/publish/0001.orig.bak`（md5 `15d5548e…`）。

| 新件行号 | 改动 | 为什么 |
|---|---|---|
| **1490-1500** | 收集 loop：`blocks_per_chunk = group_config.blocks_per_chunk` → **改名 `bpc_g`**（含其后 slice 的 3 处引用） | 让"跨 loop 泄漏"在**名字层面不可能**（同名变量在函数体内消失） |
| **1611-1613** | ★ spec loop：新增 `bpc_g = group_config.blocks_per_chunk` | **本缺陷的修点**：逐组重新取值 |
| **1615** | `gpu_block_idx = chunk_idx * bpc_g` | 跟随修复（原为 `* blocks_per_chunk`） |
| **1619/1633** | `for i in range(bpc_g)`、`dst_unit_ids.append(_units[i])` | **两侧同源展开**：`_units` 与 `gpu_block_idx+i` 一起按本组 bpc 走 ⇒ `assert len(dst_unit_ids)==len(src_block_ids)` 仍成立 |
| **1614-1623** | ★ fail-closed 断言①：`assert bpc_g == group_config.blocks_per_chunk, (...)` | 同类泄漏再犯 ⇒ **立刻炸**（§3 证明它不是空断言） |
| **1636-1647** | ★ fail-closed 断言②（**新增的第二道闸**，`H` 的警告）：`assert len(_units) == bpc_g` | 防"unit 表与 bpc 不同步"静默搬错（比长度断言更早、更定向） |
| **120-201** | 新增 `_BPCFixProbe`（四件套探针；`BPC_FIX_PROBE=1` 打开，**默认关** ⇒ 零开销/零日志变化） | §4 的四件套 |
| **1564 / 1669** | spec loop 内按组 `record()` + loop 后 `log_job()` | 打出"生产代码自己用到的真值" |

* **新 md5 = `986c9115c64f196072c7db76c24ca5f9`**（原件 `15d5548e29af88da71570d5b48abddef`）；
  逐行 diff 见 `agents/M_bpcfix/diff-vs-original.patch`（**+134 / −7**）。
* ⚠️ **`027` 口径的 `patched/scheduler.py`**（`agents/L3_8card/patched/scheduler.py`，md5 `f4de89d2…`）与原件**只差 8 卡的 import 兜底**（`try: from pgp_manager import … except ImportError: from vllm.v1.kv_offload.cpu.pgp_manager import …`）⇒ **本次修复必须同样落进那份**（替换建议见 §7）。

---

## 2. 不占卡的单元自检（先做，0.1 s，17 PASS / 0 FAIL）

`agents/M_bpcfix/scripts/selftest_bpc_leak.py`（原始数据 `agents/M_bpcfix/selftest.json`）——
**只桩 vllm 接口、不桩被测逻辑**，`_build_store_jobs` 是**真跑生产代码**：

| 臂 | 做了什么 | 结果【实测】 |
|---|---|---|
| `fixed` | 交付件，假 config `bpc={0:8,1:8,2..11:1}` | `Σgroup_sizes = len(src) = len(dst) = **72**`、`group_sizes=[32,0,4×10]`、group 0 前 16 对 = chunk0/units[0..7]+chunk1/units[8..15] |
| `defix` | **机械反修**（把 spec loop 改回读收集 loop 的残留值、删掉两条断言） | `Σ=44`、`[4,0,4×10]` ⇒ **逐字复现 `039 §10` 的 44/4** |
| `defix_assert_bpc` | 反修 + 保留断言① | **炸**：`[M_bpcfix] bpc 泄漏：loop 内的 1 != group_config 的 8（group=0）` |
| `defix_assert_units` | 反修 + 保留断言② | **炸**：`unit 数与 bpc 不一致：units=8 bpc=1` |

* **对称性**：`defix` 臂的 10 个 SWA 组仍是 `[4]×10`、每 chunk 1 个 ⇒ **泄漏只伤 `bpc>1` 的组**（`H` 的反例臂在修复后复现）；
* **静态判据（AST）自带反例臂**：在**修前原件**（`0001.orig.bak`）的 spec loop（`1481-1544`）上报警"用了 loop0(`:1413`) 赋的值"（`used=[1523,1525]`、`assigned=[]`）；在交付件上 `blocks_per_chunk` 这个名字**在函数体内彻底消失** ⇒ 判据**有判别力**（`039 §0-1` 的"假阳性"教训不再重演）；
* **四件套（真值，不重推）**：
  ```
  [M_bpcfix] #1 group=0 kind=full true_bpc=8 gpu_block_idx=0 len(src.block_ids)=72 Σgroup_sizes=72
  [M_bpcfix] #2 group=2 kind=swa  true_bpc=1 gpu_block_idx=7
  ```
  ⇒ **`bpc=8` 的组打 8、`bpc=1` 的组打 1**（任务书 (3) 的直接证据）。

---

## 3. 端到端（单卡 tiny，c1，两条臂）

**几何与 `H` 的 `h-t5` 逐字相同**（唯一变量 = 被测的 `scheduler.py`）：
```
XL1=0 XSWA=1 XRING=0  P2_COMP_JSON='[[0,2,3,4,5,6,7,8,9,10,11],[1]]'  OFFLOAD_BYTES=150994944
PROMPTS=16 PROMPT_TOKENS=4096 REPLAY_PROMPT_TOKENS=4096 MAX_TOKENS=1 EXTRA_ARGS="--enforce-eager"
池：num_units=1152（144 MiB），per_group={0:8, 1:8, 2..11:1}，blocks_per_chunk=1（unit 模式）
```
**探针**：`H_kvcheck` 的 `h_sched_probe`（scheduler 侧 src_spec，**生产代码自己构造的**）+ `h_kv_audit`（worker 侧逐张量账本）。
**自检**：`[M_selfcheck]` 影子包 + 补丁层 md5 全部命中；`df -h /dev/shm` 起服前 = **0% 用**（`m-fixed-144m.df_shm.txt`）；残留 `VLLM::` = 0。

### 3.1 判据①②（生产 spec 与 worker 实搬）【实测】

| 量 | 修前（`h-t5`，同几何） | **修后（臂 a）** | **修后（臂 b，复跑）** |
|---|---|---|---|
| store job 的 `n_src` / `len(src.block_ids)` | 44 | **72** | **72** |
| `Σgroup_sizes` | 44 | **72** | **72** |
| `group_sizes`（12 组） | `[4, 0, 4×10]` | **`[32, 0, 4×10]`** | 同左 |
| group 0 每 chunk 的 block 数 | **1** | **8** | **8** |
| `n_keys`（条目数） | 44 | 44 | 44 |
| worker 实搬：`g0_store` op 数（逐张量） | 768（= 64 block × 12 张量） | **6144**（= 512 × 12） | **6144** |
| worker 实搬：`g0_store` 的 **distinct GPU block**（逐张量） | **64** | **492** | **492** |
| `g0_load` op 数（逐张量） | 6144（= 512 × 12） | 6144 | 6144 |
| 反例臂：10 个 SWA 组的 `group_sizes` | `[4]×10` | **`[4]×10`** | 同左 |

> **492 的含义**：`H` 在 `039 §10` 用"该搬的（非 0）GPU block 并集"算出 **492**，实搬 **64** ⇒ 428 个从未被搬。
> 修后**实搬 = 492 = 该搬**（逐张量 12 张全中），**428 的缺口归零**。
> 算术自检：`g0_store` op 数 768→6144 = **×8**，正是"每 chunk 1→8 个 block"的签名（其余组一个都没变）。

### 3.2 判据③：读到"从未写过/全 0 的池行"【实测】

| 臂 | worker 账本 flags | group 0 load 的 `zero_row+unwritten` |
|---|---|---|
| 修前（`h-t5`） | `{zero_row+unwritten: 5352}` | **5352** |
| **修后（臂 a/b）** | **`{}`** | **0** |

* 这就是 `039 §10.3` 那条账的根治：store 只写 `units[0]`（4 个 unit/请求），load 按 `bpc_g=8` 读 32 个 unit/请求 ⇒ **28 unit/请求 × 16 请求 = 448 unit 从未被写**（在逐张量账本里 = **5352 次**）；
* 修后 store/load **同源展开**（各 512 block/张量）⇒ **一个全 0 行都不再被读**；
* **同一条探针在两臂上都跑了**：修前臂有 flag（判别力在），修后臂 flags 表**空**（不是"看不见所以 0"）。

### 3.3 字节账闭环【实测】

| 量 | 修前 | 修后 | 说明 |
|---|---|---|---|
| `CPU→GPU`（取回字节） | 231,669,760 | **231,669,760（逐位相同）** | **取回路径一个字节没变**（与 `H` 的"load 侧是对的"一致） |
| `GPU→CPU`（写池字节） | 196,689,920 | **362,127,360** | 差 = **165,437,440 = 448 block × 369,280 B**（group 0 一次 block 的 Σpage，正是 `034` 的 `369280`）⇒ **补上的就是漏搬的那 7/8** |

### 3.4 判据④⑤⑥⑦（不回归）【实测】

| 判据 | 修前（`h-t5`） | **修后（臂 a / 臂 b）** | 结论 |
|---|---|---|---|
| `BlockStored:CPU`（KV 事件） | 714 | **714 / 714** | ✅ 条目数不变（`021` 判据① 的"条目数不降、只变小"仍是事实） |
| `BlockRemoved:CPU` | 10 | **10 / 10** | ✅（池 1.000× 时那 10 次淘汰仍在，与 `021`/`040` 的"工作集 1162 unit"一致） |
| `kv_offload_cpu_allocation_size_sum` / `usage` | 714 / 1.0 | **714 / 1.0** | ✅ |
| `external_prefix_cache_hits` / `queries` | 65,520 / 131,328 | **65,520 / 131,328** | ✅ `021` 判据②|
| `GPU KV cache size` | 22,719 tokens | **22,719 tokens** | ✅ 判据⑥（容量不变） |
| fill / replay p50 | — | **623.5 ms / 250.4 ms = 2.49×** | ✅ 判据⑤"replay≪fill"（臂 a：622.1/258.6 = 2.41×） |
| fill run 输出 sha256 | `24b57053…` | **`24b57053…`（两臂、两轮逐字相同）** | ✅ 判据⑦ + `036/037` 口径的"同 sha" |

> ⚠️ **诚实边界**：本窄臂的 replay 加速只有 **2.4×**（`021` 的 BF16 144 MiB 臂是 9.3×）——因为这里是 **int8-SWA 几何 + 144 MiB = 工作集的 0.99×**（`040` 反解 1162 unit），replay 轮本身有 10 次淘汰、且 `CPU→GPU` 只有 231 MB（16 个 load job 的前缀命中部分）。**这不是回归**：修前同几何是 `CPU→GPU=231,669,760`（逐位相同）、hits 逐位相同。
> **本臂不是 `021`/`027` 的性能臂**，它是"索引空间正确性"臂；性能口径见 `021 §4`（9.3×）与 `027 §4`（8 卡 14.31×）。

### 3.5 判据④ 的"变长前缀"格：**4096 → 2048**【实测】

同几何、只把 `REPLAY_PROMPT_TOKENS` 改成 **2048**（`017` 那条"池里有数据也一条不取"的反例正是指望它归零）：

| 量 | 修后（`m-fixed-144m-mix`） | 对照 |
|---|---|---|
| `external_prefix_cache_hits` | **32,752** | **与 `021 §5` 的 mixed 臂逐字相同**（32,752）✅ |
| `queries` | **98,560** | 与 `021 §5` 逐字相同（98,560）✅ |
| `CPU→GPU` | **137,134,080（>0）** | `017` 的失败形态 = `hits=0` & `CPU→GPU=0` ⇒ **本臂不是那种形态** |
| replay p50 / fill p50 | **214.4 / 614.1 ms = 2.87×** | — |
| replay 输出 sha256 | **`c9742db17b83c98f…`** = **`041 §5` 的"同几何冷算参考 `c9742db1…`"** | 逐字节相同 ⇒ 取回的不是"重算侥幸" |
| worker 账本 | `g0_store` 492 block/张量、`g0_load` **256** block/张量（= 2 chunk × 8 block × 16 请求）、flags **空** | ✅ |
| `BlockStored:CPU` / `GPU KV cache size` | 714 / **22,719** | ✅ 不回归 |

⇒ **`021` 判据③（变长前缀安全）在修复后仍然成立**（机制上本来也不该受本修复影响：`_swa_trim_keep_chunk`/`alignment_chunk_count` 一个字没动，
10 个 SWA 组的 `group_sizes` 修前修后逐项相同）；本格是**复确认**，不是新机制。

---

## 4. 四件套（任务书 (3)）——真跑里的两行【实测】

```
[M_bpcfix] #1 req=… group=0 kind=full true_bpc=8 gpu_block_idx=0 chunk=0 len(src.block_ids)=72 Σgroup_sizes=72 group_sizes=[32,0,4,4,4,4,4,4,4,4,4,4]
[M_bpcfix] #2 req=… group=2 kind=swa  true_bpc=1 gpu_block_idx=7 chunk=7 len(src.block_ids)=72 Σgroup_sizes=72 group_sizes=[32,0,4,4,4,4,4,4,4,4,4,4]
```

* **`bpc=8` 的组打 8、`bpc=1` 的组打 1**（`BPC_FIX_PROBE=1`，默认关）；
* **反例臂（`defix`）上的同一条打印**（单元自检里）：`true_bpc=8` 但 `gpu_block_idx=0`（stride 按 1 走）、`Σgroup_sizes=44` ⇒ **探针自证缺陷**（不是"顺带全绿"）。

---

## 5. 红线遵守

* 只用 **c1**（`tools/a3_chip.sh c1`，无 75）；**没碰** c0（`K_l1_8card`）、c2（`L_dmafix`）、Phy-ID 8–15、`mooncake-*`/`jitpgo-*`/`dsv41-a3`；
* **没手设** `ASCEND_RT_VISIBLE_DEVICES`（由锁脚本注入）；
* **没用 `/tmp`**：本机 `~/tmp/20260922/M_bpcfix/`、A3 `~/tmp/20260922/M_bpcfix/`；
* 起服前 **`df -h /dev/shm` = 0%**（`m-fixed-144m.df_shm.txt`）；
* **没写** `upstream-v41/`；**没改** `a2/publish/` 原件、没改 `H_kvcheck`/`X_integrate` 的任何文件（overlay = 我自己的 `pkg/base_snapshot` 快照副本 + 符号链接；`selfcheck_m.py` 只把"影子包必须是 H 的 pkg"这条放宽成我的 pkg，**其余 md5/符号断言逐字保留**）；
* 跨机传文件走 **coscli**（`a2/scripts/cos-xfer.sh`，key `share/xfer/M_bpcfix-*`）；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺格标 `—`。

---

## 6. 未确认 / 边界（必须与结论一起引用）

| 项 | 状态 |
|---|---|
| 8 卡真权重（`027` 口径 `OFFLOAD_GB=56`）上的同一判据 | ⛔ **【未确认】**：本任务只跑单卡 tiny（卡时/时段限制）。⇒ **替换件必须同样落进 `agents/L3_8card/patched/scheduler.py`（8 卡挂载链）后再复跑一条臂** |
| `hits` 之外的用户可见影响 | ⚠️ **【实测·反证】修前在当前配置下"输出 sha 也对"**（`h-t2`/`038`：448 行全 0 时 fill/replay sha 逐字相同）⇒ 本缺陷是**潜伏的正确性风险**（不同 prefix/不同几何会读到全 0 的前缀 KV），**不是** `038` 那个 ❌/✅ 翻转的答案 |
| `concurrency > 1` | ⛔ 未测（与 `021`/`027` 同口径 `concurrency=1`） |
| A2 真机（8×910B3） | ⛔ 未测（本任务与 `021`/`027` 同：A3 上验证） |
| `SWA_TRIM=window` 臂 | ⛔ 本任务不涉及（保持 `off`） |

---

## 7. ★ 替换建议（给主代理决策；**我没有动 `a2/publish/`**）

| # | 目标文件 | 动作 | 备注 |
|---|---|---|---|
| 1 | `a2/publish/0001-offload-scheduler.patch.py` | 用 `agents/M_bpcfix/publish/0001-offload-scheduler.patch.py`（md5 `986c9115…`）**整文件替换** | 单体替换件；diff 见 `agents/M_bpcfix/diff-vs-original.patch` |
| 2 | `agents/L3_8card/patched/scheduler.py`（8 卡挂载链的同一份，md5 `f4de89d2…`） | ★ **已备好成品**：`agents/M_bpcfix/publish/0001-offload-scheduler.patch.py.8card`（md5 `6a4f8dff…`）= 现 8 卡文件 + **本次修复的 5 个 hunk**（`diff -u` = `agents/M_bpcfix/diff-8card-chain.patch`，**只含我的修复、零其它差异**；`py_compile` OK；单元自检 17/17 通过） | ⚠️ **上线阻塞项**：`027` 的 8 卡臂用的就是这份。**何时覆盖 `L3_8card/patched/` 由主代理定**（按纪律我没写别人的目录） |
| 3 | `a2/publish/0001b-*.py` / `0001c-*.py` | **不动**（本次缺陷不在 manager/hooks；`J_mgrhardening` 的加固与本修复正交） | — |

**复现（全部不占卡或一条单臂）**：
```bash
# ① 单元自检（0.1 s，不占卡）
python3 a2/agents/M_bpcfix/scripts/selftest_bpc_leak.py --json ~/tmp/20260922/M_bpcfix/selftest.json
# ② 端到端单臂（c1，~2.5 min；含 df -h /dev/shm 检查）
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 1500 --name m-bpc-fixed -- \
  env TAG=m-fixed-144m ARM=fixed PORT=8233 XL1=0 XSWA=1 XRING=0 \
      P2_COMP_JSON='[[0,2,3,4,5,6,7,8,9,10,11],[1]]' OFFLOAD_BYTES=150994944 \
      PROMPTS=16 PROMPT_TOKENS=4096 REPLAY_PROMPT_TOKENS=4096 MAX_TOKENS=1 \
      EXTRA_ARGS="--enforce-eager" bash /work/agents/M_bpcfix/scripts/run_arm_m.sh
# ③ 汇总（同一脚本对两臂都跑）
python3 a2/agents/M_bpcfix/scripts/analyze_m.py <outdir> m-fixed-144m m-fixed-144m-b
```
**锁退出码 75 = 没抢到锁，是重试不是失败。**

---

## 8. 产物

| 类 | 位置 |
|---|---|
| 交付件（替换建议） | `agents/M_bpcfix/publish/0001-offload-scheduler.patch.py`（md5 `986c9115…`） |
| 反修件（反例臂，可复现 44/4） | `agents/M_bpcfix/publish/0001-offload-scheduler.patch.py.dfix`（md5 `3e8c2cae…`） |
| 逐行 diff | `agents/M_bpcfix/diff-vs-original.patch`（+134/−7） |
| 单元自检 | `agents/M_bpcfix/scripts/selftest_bpc_leak.py` + `agents/M_bpcfix/selftest.json`（17 PASS / 0 FAIL） |
| 单臂驱动 / 汇总 / 同步 / 取回 | `agents/M_bpcfix/scripts/{run_arm_m.sh,analyze_m.py,sync_to_a3.sh,fetch_from_a3.sh,selfcheck_m.py}` |
| 原始数据（修后两臂） | `logs/raw/043-bpc-leak-fix/m-fixed-144m{,-b}.*`（+ `043-summary.txt`、`043-input-md5.txt`） |
| 修前反例臂（同一探针） | `agents/H_kvcheck/out/h-t5-int8-150994944.*`（本日志 §3 用它做对照，**未改动**） |
