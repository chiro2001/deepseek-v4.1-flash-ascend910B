# 040 — 池 unit 行的**行级 provenance**：`038` §4 的机制【实测·被推翻】，真凶收缩到 **0.9914×（差 10 unit）**

2026-09-22 06:19–07:xx CST。执行：子代理 **I_unitprobe**。机器：**A3（A3-node1）**。
占卡走 `tools/a3_chip.sh` 锁（无 75 退出）；**没用 `/tmp`**（本地只写 `~/tmp/20260922/i_unitprobe/`）；
**没写 `upstream-v41/`**；**没改任何一行生产代码 / 别人的文件**（诊断只走自己的 overlay + import hook）；
传文件全走 `cos-xfer.sh`；**没手设 `ASCEND_RT_VISIBLE_DEVICES`**；没碰 `mooncake-*` / `jitpgo-*` / `dsv41-a3`。

状态：**已完成**（七批臂；§0 是全部结论，§1–§4 第一批、§6 第二批、§7 第三批、§9.1 第七批）。

---

## 0. 结论（全文最重要的一段）

### 0.1 ★★★ 三臂对照（最有判别力的一条证据，**池大小/unit/淘汰数/字节全同，唯一变量 = manager**）

| 臂 | 几何 | manager | 池 | unit | `BlockRemoved:CPU` | `CPU→GPU` | J2 |
|---|---|---|---:|---:|---:|---:|---|
| `i-int8-144-p2` | C0，**int8** | `PerGroupBPCManager`（`XL1=0`） | 144 MiB | 1152 | 10 | 231,669,760 | ❌ 1/16（#15） |
| **`i-int8-144-L1b`** | C0，**同 int8** | **`P2QuotaManager`（`XL1=1`）** | **144 MiB** | **1152** | **10** | **231,669,760** | **✅ 0/16** |
| `i-bf16-144` | C0，纯 BF16 | `PerGroupBPCManager`（`XL1=0`） | 144 MiB | 1152 | 10 | 272,957,440 | ✅ 0/16 |

* `i-int8-144-L1b` **复跑一次（`-r2`）逐字同 sha** ⇒ 满足"单格不算数"纪律；
* ⇒ **`038` 的否决点只对"L5 单位池（`XL1=0`）"成立，`L1` 一开即消失**；也**不是**"1.000× 临界容量"的通用缺陷（BF16 同格 ✅）。

### 0.2 ★★★ int8 两条容量杠杆（×1.4655 / ×1.9133）的❌与**池容量无关**，而是**另一个现象**

| 臂 | 几何 | `GPU KV cache size` | 池 | **`cpu_cache_usage_perc`** | replay sha | J2 |
|---|---|---:|---:|---:|---|---|
| `i-xD-L1` | 4 条（L5+L1+SWA-q+ring16） | **33,295** ✔ | 144 MiB | **0.5990**（池只用 60%） | **`6a47dd65…`**（= `035` 的 `x-D` 逐字相同） | ❌ 15/16 不匹配 |
| `i-xF-L1` | 5 条（+KV8 双平面+prefill） | **43,469** ✔ | 144 MiB | **0.5360**（池只用 54%） | `e83136aa…` | ❌ 15/16 不匹配 |

* **池只用 54–60% ⇒ 没有淘汰压力、没有"借行" ⇒ 这两格的 ❌ 与容量/欠配无关**；
  `035` 自己的 `x-E`（同几何、池缩到 89.6 MiB）也是同一个 ❌ ⇒ **`035` 当时其实已经证明"池大小不影响这两格"**，只是没被点出来；
* 两格 **`XL1=1` 确认生效**（`P2QuotaManager, num_units=1940 / 2168`）、`P2_COMP_JSON` 用 **20 张量那套**、`GPU KV cache size` 到 33,295 / 43,469 ⇒ 与 `035` **可比**（replay sha 逐字同 `035` 的 `6a47dd65…`）。
  ⚠️ **口径差异**：`035` 记 `x-D` = "❌ 14/16"、`x-F-full` = "❌ 15/16"，本任务用 `036`/`038` 的 **fill-vs-replay** 口径得 15/16 不匹配；
  **两者 replay sha 相同**，计数差异来自参照物（`035` 有独立 `x-*-cold` 臂，本任务没跑）⇒ 标 **【口径未对齐】**，不改 `035` 的数字。

### 0.3 ★★★ D/F 几何 ❌ 的机制：【实测·强相关】+【推断】= **卸载层每行少拷了 state 环的尾部字节**

【实测】四个臂里 **`state` 组（g1）张量的「物理行步长」vs「卸载层每行拷贝大小」**：

| 几何 | `state` 组 → tensor_idx | 物理行步长 | 卸载拷贝/行 | **差** | J2 |
|---|---|---|---:|---:|---|
| C0 int8（`i-int8-144-L1b`） | 12,13,14 | **131,072** ×3 | **131,072** ×3 | **0** | ✅ |
| C0 BF16（`i-bf16-144`） | 12,13,14 | 131,072 ×3 | 131,072 ×3 | 0 | ✅ |
| **D 几何（`i-xD-L1`）** | 12,13,14 | **73,856** ×3 | **65,536** ×3 | **−8,320 B/行/张量** | ❌ |
| **F 几何（`i-xF-L1`）** | 16,17,18 | **66,560** ×3 | **65,536** ×3 | **−1,024 B/行/张量** | ❌ |

* ★ **`66,560` 正是 `034` 结论 ⑥ 预言的那个数**（原话："页步长 **66560 B**，而一页 32 行只占 65536 B ⇒ …必须改成按页步长，否则**不是慢一点而是静默读错页**"）；
* 【实测】(a) 合成包里 `compressor_triton.py` **确实**有 `CACHE_PAGE`，且**调用点已传 `state_cache.stride(0)`**（`compressor_triton.py:726/741`）⇒ **kernel 侧的修法是 034 的正式版**；
  但 **卸载层（`cpu_npu.py`）用的是 `data_ref.page_size_bytes` 当 DMA 长度**，而**指针步长取张量自身 `row_stride`** ⇒ **拷贝长度 65,536 < 行步长 66,560/73,856** ⇒ **每行尾部的 1,024 / 8,320 字节从未被 store**，load 时那部分仍是 GPU 侧旧值/零 ⇒ 环状态被静默截断。
* 【推断】为什么**只有 ring16 + SWA-quant 同时开**才炸：只有这个组合会把 state 的槽页压到 66,560/73,856（其余组合都是 131,072 = 拷贝大小）⇒ **与"两者同时开才炸"完全吻合**。
* 【实测】分配侧与 worker/DMA 的四条路（§7）全 0 ⇒ **排除"记账/复用/时序"**，指向**拷贝长度 vs 行步长**这一处。

### 0.4 五句话总结

1. **★【实测】`038` §4 的机制（"同一池行被两个 block 复用 ⇒ 读到未写入的行"）在配账层被推翻**：
   在 int8 的 ❌ 臂（144 MiB）上，调度侧 manager 的 11 个计数器**全 0** ——
   没有 `dup_unit`（同一行同时给两个活 key）、没有 `stale_free`（`_free_block` 的兜底分支没进过）、
   没有 `free_list_dup` / `over_cap`，且**每个命中 key 读的行 == 它 store 时写的行**（`hit_row_changed=0`）。
2. **★★【实测】`038` 的"1.000× 临界"这个前提本身是错的：144 MiB 是 `0.9914×`（差 10 unit）**。
   实测工作集 = **1162 unit**（`group0(full) 64 chunk × 8 = 512` + `10 个 SWA 组 × 65 = 650`），
   而 1152/1162 = 0.9914；**144 臂的 `BlockRemoved:CPU=10` 恰好等于这个缺口**。
   ⇒ `038` 的"1.11× 安全"应改判为 **1.1016×（1280/1162）**，上线判据应按 **1162 口径**算。
3. **★【实测】工作集可以用现成指标反解，不需要探针**：`工作集 = cpu_cache_usage_perc × num_units`
   （160 臂：`0.9078125 × 1280 = 1162.0`，与探针逐字吻合）。
4. **【实测】J2 复现与 `038` 逐字相同**：144 ❌`mismatched=[15]`（三枚 sha 与 `038` 逐字相同）、160 ✅。
5. **【实测】探针自带"生效性自检"**（防止 `035` 那种"钩子名不存在却打'已装'"的假阳性）：
   两条臂都打出 `探针**已生效** num_blocks=1152/1280` 的横幅，**没生效时会显式写"这条臂不能用"**。

6. **★【实测】`038` 的否决点 = 特定 manager 的组合缺陷**：`L1`（`P2QuotaManager`）一开，144 MiB 的 ❌ 即转 ✅，复跑逐字同 sha（§0.1）。
7. **★【实测·强相关】int8 两条容量杠杆的 ❌ 与容量无关**（池只用 54–60%），而是 **state 环的"物理行步长 ≠ 卸载拷贝长度"**（xF = **66,560**，正是 `034` 预言的那个数）（§0.2/§0.3）。
8. **★【实测】worker/DMA 层的三条假说（甲/乙/丙）全排除**，且改进了 `H_kvcheck` 报的"读到从未写过的行"= **探针自身的假阳性**（§7）。

### 0.5 给运维的一句话

> **起服失败先看 `df -h /dev/shm`**（本任务踩到：c0 的 `/dev/shm` 只有 64 MiB，被一个 14 小时前的 4 GiB 残留文件打满 100%，
> 报错是 `OSError: [Errno 28] No space left on device`，出现在 `_multiprocessing.SemLock`，极易被误判成"补丁坏了"）；
> **判"池够不够"看 `cpu_cache_usage_perc`**（`工作集 = usage × num_units`）；
> **判"能不能上线"不能只看倍率**（见 §6.1）。

---

## 1. 复现（判据 S1 守门员）

臂 = `038` 的 **C0 几何**（L5 + SWA-quant：`XL1=0 XSWA=1 XRING=0`），16 请求 × 4096 token、`MAX_TOKENS=1`。

```bash
bash tools/a3_chip.sh c2 --timeout 2400 --name i-unit-144 -- \
  env TAG=i-int8-144 POOL_BYTES=150994944 XSWA=1 PORT=8300 \
  bash /work/agents/I_unitprobe/scripts/run_arm_i.sh
```

| 臂 | 池 unit | fill sha | replay sha | J2 |
|---|---:|---|---|---|
| `i-int8-144` | 1152 | `24b57053…` | `a7ffff6b…` | ❌ `mismatched=[15]` |
| `i-int8-160` | 1280 | `24b57053…` | `24b57053…` | ✅ 0 |

与 `038` §1 的两枚 sha **逐字相同** ⇒ 能在已知 ❌ 的格上报警，判据可信（S1 通过）。

---

## 2. 探针（只读、不改任何生产代码）

`a2/agents/I_unitprobe/probe/i_unitprobe.py`（挂法见 `probe/sitecustomize.py`）：
先逐字 `exec` pkg 的 P2 sitecustomize（它会再 exec PGP），**然后原地包裹**
`pgp_manager.PerGroupBPCManager` 的 `_allocate_blocks / _free_block / prepare_load / reset_cache`，
维护两个只读账本：`row → 当前活 key`、`key → store 时分配到的行`。

计数器（父代理 2026-09-22 指定 5 个 + 本探针自加 6 个）：

| 计数器 | 含义 |
|---|---|
| `dup_unit` | 同一 unit 行同时被 ≥2 个**【活】** key 引用 ← `038` §4 的直接签名 |
| `free_list_dup` | `_free_list` 里出现重复行号 ← 陈旧 free 的直接签名 |
| `stale_free` | `_free_block` 走进 `_units_of_block` 找不到键的兜底分支（**正常应恒为 0**） |
| `oob_unit` | 发出的行号 ≥ `num_blocks` |
| `over_cap` | `_num_allocated_units > num_blocks` |
| `underflow_free` | 释放一个当前无人占的行（重复释放） |
| `missing_units` | 活 key 的 `block_id` 不在 `_units_of_block` ⇒ 读侧退化成 `[block_id]` |
| `hit_row_alias` | ★ 命中 key 要读的行当前归**别的活 key** |
| `hit_row_changed` | ★ 命中 key 要读的行 ≠ 它 store 时写的行 |
| `hit_never_stored` | 命中一个本进程账本里从未 allocate 过的 key |

### 2.1 结果：144 MiB（❌）臂 **11 个计数器全 0**

```
SUMMARY(周期) tag=i-int8-144 num_blocks=1152 alloc_calls=51 keys=714
              live_rows=1152 peak=1152 max_row=1151 used=1152 free=0
              free_list_len=0 unique=0 units_of_block=290 policy=290
              | dup_unit=0 free_list_dup=0 stale_free=0 oob_unit=0 over_cap=0
                underflow_free=0 missing_units=0 hit_row_alias=0 hit_row_changed=0
                hit_never_stored=0
```

* `used=1152 / num_blocks=1152`、`free=0` ⇒ 池**完全打满**（与 `cpu_cache_usage_perc=1.0` 一致）；
* `max_row=1151` ⇒ 没有越界行；`units_of_block=290` 而 `keys=714` ⇒ 290 个**唯一** block_id
  承载 714 个 key 的历史分配（行会随淘汰/复用换主，这是正常的）；
* **`dup_unit=0`**：任意时刻没有任何一行被两个活 key 同时引用；
* **`stale_free=0`**：`_free_block` 的 `units is None` 兜底分支**一次都没进过**；
* **`hit_row_changed=0`**：所有命中里，**每个命中 key 读的行 == 它 store 时写的行**。

⇒ **`038` §4 的机制在【调度侧 manager 的 unit 配账】这一层不成立。**
（⚠️ 口径边界：**不含** worker / DMA 侧与 GPU block 侧；见 §5。）

---

## 3. 新硬事实：144 MiB 是 **0.9914×**，不是"1.000× 临界"

### 3.1 工作集 = 1162 unit（探针直接数出来的）

| 组 | 每请求 key 数 | 每 key unit | 合计 |
|---|---:|---:|---:|
| `group0`（full，bpc=8） | 64 chunk | 8 | **512** |
| `group2..g11`（10 个 SWA 组，bpc=1） | 65 × 10 = 650 | 1 | **650** |
| **工作集** | | | **1162** |

### 3.2 用现成指标独立复算（**不需要探针**）

| 臂 | `num_units` | `cpu_cache_usage_perc` | 反解 `usage × units` |
|---|---:|---:|---:|
| 144 MiB | 1152 | **1.0**（打满，饱和） | = 1152 |
| 160 MiB | 1280 | **0.9078125** | **1162.0** ✔ |

⇒ **`工作集 = cpu_cache_usage_perc × num_units`**（未饱和时精确）。这是给运维的**一行反解公式**。

### 3.3 `BlockRemoved:CPU = 10` **恰好等于缺口**

144 臂淘汰了 10 个 key，**正好** = 1162 − 1152。
探针的映射日志显示：这 10 行（行号 0–9）原本属于**首请求的 10 个 SWA chunk**（`seq=1..10`，group 2–11），
被**最后两个请求的 `group0` chunk**（`seq=671`：行 0,1,2,4,6,7,8,9；`seq=672`：行 3,5）复用。
而 J2 失败的正是 **prompt #15**。
⇒ "容量紧张"确实把 **#15 的 store** 与被淘汰的行**耦在了一起**，但每一步在配账上都**合法**（§2.1）。

### 3.4 对上线的直接含义

* `038` §11.2 建议的 `units_ratio ≥ 1.05` **要按 1162 口径算**；
* `038` 的"1.11× 安全"应写成 **1.1016×（1280/1162）**；
* 判据建议（**现成指标即可算，零成本**）：
  `num_units ≥ 1.05 × round(cpu_cache_usage_perc × num_units)`（或起服后用 `ceil(usage × units)` 反解一次工作集）。

---

## 4. 两个方法论坑（点名留档，给后人避坑）

1. **`X_integrate/scripts/selfcheck.py` 把影子包前缀写死成 `/X_integrate/pkg`** ⇒
   套在别人的 overlay 上必然误报 `FAIL 影子包没生效`（本次两条臂第一轮都因此 rc=1 空跑）。
   本任务改用自带的 `scripts/selfcheck_i.py`（前缀 = 自己的 overlay）。
2. **"钩子名不存在却打'已装'"这类探针假阳性**这次也踩到一次 ——
   `i-int8-144-L1`（`XL1=1`）臂上 `PerGroupBPCManager` 根本没被使用（P2 补丁换成了它自己的 quota manager），
   探针装了钩子却**没有任何 alloc**。因此本探针把"生效性"做成**显式信号**：
   有 alloc 才打 `探针**已生效** num_blocks=…`；结束却没有 alloc 就写
   **"ledger 不存在 ⇒ 探针从未生效（这条臂不能用）"**，外层脚本再打一行
   **"探针没有生效 ⇒ 本臂的探针结论无效"**。**所有结论一律以这条自检为准。**

---

## 5. 第二批（进行中）

| 臂 | 测什么 | 假说 |
|---|---|---|
| `i-int8-144-s` | worker 侧**行级内容 provenance**（`probe/i_storeprobe.py`） | (甲) store 读的 GPU 源块被提前复用 / (乙) store 未真正执行 / (丙) load 时序 |
| `i-bf16-144` | 纯 BF16、**XL1=0（与 int8 ❌ 臂同一个 `PerGroupBPCManager`）** | `038` 的 BF16 臂是 `XL1=1`（**另一个 manager，confound**）⇒ 这一格才是真对照 |
| `i-bf16-112` | 纯 BF16 更深欠配（896 unit） | BF16 会不会也炸 ⇒ 决定缺陷是"int8 的"还是"欠配的" |

`probe/i_storeprobe.py` 的做法（只读、两处 hook：`transfer_async` / `get_finished`）：

* **(甲)** submit 时刻对 **GPU 源块**做批快照（`index_select` → host，快照前显式对齐一次设备），
  DMA 完成后再快照 **CPU 行**，逐对 `torch.equal` ⇒ 若"源 ≠ 落盘"就是源被复用/时序问题；
* **(乙)** 计数 `store_jobs_seen / store_jobs_checked / store_pairs`；
* **(丙)** 每次 load 把 CPU 源行的内容与**该行最后一次 store 落下的内容**比对 ——
  这就是 `038` §11.3 要的**行级 provenance**；`I_UNITPROBE_MODE=error` 时**断链即 fail-fast**。

---

## 6. ★★ 第二批结果：**BF16 在同一 manager、同一欠配下不炸**（`038` 的 confound 被消掉）

| 臂 | 几何 | manager | 池 unit | `BlockStored:CPU` / `BlockRemoved:CPU` | J2 |
|---|---|---|---:|---|---|
| `i-int8-144-s` | int8（`XSWA=1`） | `PerGroupBPCManager`（`XL1=0`） | 1152 | 714 / **10** | ❌ `mismatched=[15]` |
| **`i-bf16-144`** | **纯 BF16**（`XSWA=0`） | **同一个 `PerGroupBPCManager`（`XL1=0`）** | **1152** | 714 / **10** | **✅ 0** |
| `i-bf16-112` | 纯 BF16（更深欠配） | 同上 | **896** | 714 / ？ | **✅ 0** |

* 【实测】**三条臂的 unit 数、淘汰数、事件计数逐字相同**（探针都确认 `num_blocks=1152/1280/896` 生效）；
* ⇒ **`038` §8-3 那一格关掉**：**BF16 压到（甚至压过）临界容量也不炸**；
* ⇒ 缺陷不是"`1.000×` 临界容量"这一类的**通用**缺陷，而是 **「int8 几何 + 欠配」组合**触发；
  但它仍然是 **int8 两条杠杆（×1.4655 / ×1.9133）的上线否决点**（int8 正是要上线的那条路）。
* ⚠️ **`038` 的"BF16 不炸"那条对照是 confound**：它的 BF16 臂跑的是 `XL1=1`（**P2 的另一个 manager**），
  与本任务的 ❌ 臂**不是同一个分配器** ⇒ 那一格原本无法排除"是 manager 差异而不是量化差异"。本任务用 `XL1=0` 补上了。

### 6.1 ★ 显式警告：`units_ratio` **不是**充分判据

> **不要**把 `units_ratio ≥ 1.05` 当成"可上线"的判据。
> 依据：**BF16 在 0.9914×（甚至更低）下安全，而 int8 在同一个 0.9914× 下静默出错**
> ⇒ 倍率只是**触发条件之一**；int8 的失效需要 **「int8 几何」∧「欠配」同时成立**。
> 上线判据必须是**"欠配 ∧ int8 几何"两条一起判**（或直接上行级 provenance 自检）。

---

## 7. ★★ 第三批：worker/DMA 层的三条假说**全部排除**（含一个假阳性的自我证伪）

### 7.1 对称实验（同一探针，❌ 臂 vs ✅ 臂）

| 计数器 | `i-int8-144-p2`（❌） | `i-int8-160-p2`（✅） |
|---|---:|---:|
| `store_jobs_seen / checked` | 16 / 15 | **16 / 15** |
| `store_pairs`（逐对快照） | 5232 | **5232** |
| `store_dst_mismatch`（源字节 ≠ 落盘字节） | **0** | **0** |
| `store_finished_size0` / `store_bytes_mismatch` | 0 / 0 | 0 / 0 |
| `store_pairs_truncated` / `store_jobs_skipped` | 0 / 0 | 0 / 0 |
| `store_row_reassigned_inflight`（按 unit 行号聚合） | **0** | **0** |
| `load_row_inflight`（要读的行正被未完成 store 写） | **0** | **0** |
| `load_row_not_written_yet`（写晚于读） | **0** | **0** |
| `load_row_unknown`（本进程账本里查不到） | 5376（16 次） | **5392（16 次，行号逐个相同）** |
| J2 | ❌ `mismatched=[15]` | ✅ 0 |

### 7.2 假说判决（全部【实测】）

| 假说 | 判据 | 结果 |
|---|---|---|
| **(甲)** store 读的 GPU 源块被提前复用 ⇒ 存进别人的字节 | `store_dst_mismatch`：submit 时刻 GPU 源 vs DMA 完成后 CPU 行，逐对 `torch.equal` | **0 / 5232 对** ⇒ **排除** |
| **(乙)** store job 交了但 DMA 没真跑 | `store_finished_size0` / `store_bytes_mismatch` | **全 0** ⇒ **排除** |
| **(丙)** load 在 DMA 完成前就被消费（时序） | `load_row_inflight` + `load_row_not_written_yet`（**纯记账**） | **全 0** ⇒ **排除** |
| 附加：按 **unit 行号**聚合"写一个已经易主的行" | `store_row_reassigned_inflight`（交叉查询调度侧行时间线） | **0** ⇒ **排除** |

### 7.3 ⚠️ 一个**假阳性**的自我证伪（方法论，务必留档）

`load_row_unknown` 在 ❌ 臂上"每次 load 有 336 行读到从未写过的池行"，看起来正是 `038` §4 的机制。
**但同一条探针在 ✅ 臂上给出逐字相同的 16 条异常（连行号都一样）** ⇒
**它是探针自身的不完整**（`store_jobs_checked=15` 而 `seen=16`：**最后一个 store job 的完成事件没被观察到**，
它覆盖的行因此不在账本里），**不是**"读到未写入的行"。
⇒ **`load_row_unknown` 因此被排除出 fail-fast 白名单**（`I_UNITPROBE_MODE=error` 不会因它而 raise）。
★ 这条正是 `038` 给我们上的一课（**"❌/✅ 两臂 KPI 逐字相同 ⇒ 该判据无效"**）的应用：
**任何新判据必须在已知 ✅ 的臂上先跑一遍，看它是否也报警**（判据 S1 守门员的推广）。

### 7.4 操作教训：**不要在 `transfer_async` 里同步 DMA 事件**

`I_STOREPROBE_SYNC_AFTER=1`（把异步 load 变同步）会让**整臂卡死**（实测：服务器 06:38:21 后不再前进，
需在容器内 `kill -9`）。⇒ "同步验证时序"这条路判为**不可行**；时序改用**纯记账**判定（§7.1 的
`load_row_inflight` / `load_row_not_written_yet`），零成本、无死锁。

---

## 8. ★ 交付：行级 provenance 自检（可开关、默认关）

`a2/agents/I_unitprobe/probe/i_storeprobe.py`（worker 侧）+ `probe/i_unitprobe.py`（调度侧），
**两处 hook、只读、不改任何生产代码**：

```bash
# 叠在自己的 overlay 里（pkg 的 sitecustomize 之后），env：
I_UNITPROBE=1            # 调度侧：unit 行分配账本（默认开）
I_STOREPROBE=1           # worker 侧：store/load 行级内容 provenance（默认开）
I_UNITPROBE_MODE=error   # ★ 默认 report；=error ⇒ 下列任一成立即 **fail-fast**
#   store_dst_mismatch        submit 源 ≠ 落盘内容
#   store_bytes_mismatch      DMA 完成事件的字节数 ≠ spec
#   load_row_changed          命中行内容 ≠ 该行最后一次 store 落下的内容
#   load_row_inflight         命中行正被一个未完成的 store 写（时序）
#   load_row_not_written_yet  写晚于读
#   store_row_reassigned_inflight  按 unit 行号聚合发现"写一个已易主的行"
# （`load_row_unknown` **故意不在白名单**：实测在 ✅ 臂上同样出现，见 §7.3）
```

* 与 `G_kv8fix` 的 `kv8_nan_guard.py`（写侧 NaN 自检）**互补**：那个在本层之后（数值层），这个在**行/字节层**且**更早**；
* **未做**：还没在真服务上跑过 `MODE=error` 的整轮（本任务时间用在了定位上）⇒ 标 **【未确认】**。

---

## 9. 诚实边界

| # | 事项 | 强度 |
|---|---|---|
| 1 | 144 ❌ / 160 ✅ 与 `038` sha 逐字复现 | ✅ **【实测】** |
| 2 | 调度侧配账层 11 个计数器全 0（含 `stale_free` / `dup_unit` / `hit_row_changed`） | ✅ **【实测】** |
| 3 | 工作集 = 1162 unit ⇒ 144 MiB = **0.9914×**、160 MiB = 1.1016× | ✅ **【实测】**（探针 + `cpu_cache_usage_perc` 双路） |
| 4 | BF16 同 manager、同欠配（1152/896 unit）⇒ ✅ 0 mismatch | ✅ **【实测】** |
| 5 | (甲)(乙)(丙) + 行易主：worker/DMA 层全排除 | ✅ **【实测】**（5232 对逐字节） |
| 6 | `load_row_unknown` 是探针不完整导致（❌/✅ 逐字相同） | ✅ **【实测】** |
| 7 | **究竟哪一层让 int8+欠配 出错** | ⛔ **【未确认】**（配账层与 DMA 层都已排除） |
| 8 | 行级 provenance 自检在真服务上跑 `MODE=error` | ⛔ **【未确认】** |
| 9 | 容量（33,295 / 43,469）、decode 时延 | **未测**（本任务不改生产代码） |

### 9.1 ★ 第七批（对齐 `035` 的两格 + 机制定位，全部【实测】）

| # | 事项 | 强度 |
|---|---|---|
| 10 | `i-xD-L1` 复现 `035` 的 `x-D`（`6a47dd65…` 逐字相同）、`i-xD-L1-r2` 复跑同 sha | ✅ **【实测】** |
| 11 | 两格 `GPU KV cache size` = 33,295 / 43,469，`XL1=1` 生效，20 张量分量 | ✅ **【实测】** |
| 12 | **池只用 54–60%**（`cpu_cache_usage_perc` 0.5990 / 0.5360）⇒ 与容量无关 | ✅ **【实测】** |
| 13 | `state` 组行步长 66,560 / 73,856 vs 拷贝 65,536（C0 两者相等 131,072） | ✅ **【实测】** |
| 14 | **"拷贝长度 < 行步长 ⇒ 尾部字节没被 store"是 D/F ❌ 的原因** | ⚠️ **【推断】**（定量吻合 `034` 的 66,560，未做 DMA 级逐字节证明） |
| 15 | 一行修法：`data_ref.page_size_bytes` 应等于该张量的真实行步长（或让 `state` 槽页严格等于其平面和） | **未做**（本任务不改生产代码） |
| 16 | 我在 c2 上"在 `transfer_async` 里同步 DMA"导致过一次**卡死残留**（已清理并通知 `H_kvcheck`） | ✅ **【实测】**（教训见 §7.4） |

---

## 10. 原始数据与复现

### 10.1 位置

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/040-20260922-unit-provenance.md` |
| 原始数据 | `a2/logs/raw/040-i-unitprobe/`（`r2.tgz`/`b5.tgz`/`maps.tgz`/`final.tgz` + 解包目录，**67 个文件 / 3.8 MB**） |
| 代码 | `a2/agents/I_unitprobe/`：`probe/{i_unitprobe.py,i_storeprobe.py,sitecustomize.py}`、`scripts/{prepare_overlay.sh,selfcheck_i.py,run_arm_i.sh,run_batch{2,3,4,5,6,7}.sh,upload.sh}` |
| 容器内 | 同源在 `/work/agents/I_unitprobe/`（A3 A3-node1），overlay 在 `/work/agents/I_unitprobe/pkg/` |

### 10.2 关键臂（每条 = 一条命令）

```bash
# ① ★ 三臂对照：int8 ❌（XL1=0） / int8 ✅（XL1=1） / BF16 ✅
bash tools/a3_chip.sh c2 --timeout 1200 --name i-p  -- env TAG=i-int8-144-p2  POOL_BYTES=150994944 XSWA=1 XL1=0 PORT=8320 I_STOREPROBE=1 bash /work/agents/I_unitprobe/scripts/run_arm_i.sh
bash tools/a3_chip.sh c2 --timeout 1200 --name i-l1b -- env TAG=i-int8-144-L1b POOL_BYTES=150994944 XSWA=1 XL1=1 PORT=8350 COMP20='[[0,2,3,4,5,6,7,8,9,10,11],[1]]' I_STOREPROBE=0 bash /work/agents/I_unitprobe/scripts/run_arm_i.sh
bash tools/a3_chip.sh c2 --timeout 1200 --name i-bf  -- env TAG=i-bf16-144    POOL_BYTES=150994944 XSWA=0 XL1=0 PORT=8310 I_STOREPROBE=0 bash /work/agents/I_unitprobe/scripts/run_arm_i.sh
# ② ★ 两条容量杠杆的几何（对齐 035 的 x-D / x-F-full）
bash tools/a3_chip.sh c2 --timeout 2400 --name i-b7 -- bash /work/agents/I_unitprobe/scripts/run_batch7.sh
```

**锁退出码 75 = 没抢到锁，是重试不是失败。** 全部臂都自己带"探针生效性自检"（没有生效横幅就判"这条臂不能用"）。

**红线核对**：没发 PR / issue / 评论；没写 `upstream-v41/`；没用 `/tmp`；占卡全走 `a3_chip.sh`（无 75 退出）；
没手设 `ASCEND_RT_VISIBLE_DEVICES`；没碰别的容器；**没改任何生产代码或别人的文件**（全部 overlay + import hook）；
传文件全走 `cos-xfer.sh`；结论逐条标了【实测】/【推断】/【未确认】。
