# 038 — `kv8_ori_plane` 的 decode 快路径**被实测洗清**；真凶收缩到"池容量恰好等于工作集"的**第一压缩层 NaN**

> 2026-09-22 05:16–06:20 CST。执行：子代理 **G_kv8fix**。机器：**A3（A3-node1）槽位 c2 = die 7**（容器 `prbench-c2`）。
> 全程只用 c2（**没碰 c0 / c1**）、占卡走 `tools/a3_chip.sh` 锁（**无 75 退出**）、**没用 `/tmp`**（容器内只写 `/work/agents/G_kv8fix/`）、
> **没写 `upstream-v41/`**、**没改任何一行生产代码 / 别人的文件**（诊断只走自己的 overlay 叠加 + import hook）、
> 跨机传文件全走 `cos-xfer.sh`、**没手设 `ASCEND_RT_VISIBLE_DEVICES`**、没碰 `mooncake-*` / `jitpgo-*` / `dsv41-a3`。
> A3 宿主 `MemAvailable` 全程 ≥ 1.7 TiB（未触 150 GiB 阈值）。

---

## 0. 七句话结论

1. **✅【实测】`036` 的复现逐字成立**：`4096`（=32×128 对齐）⇒ **J2 ❌ 1/16，失败点 = prompt #15**；
   `4095`（=31×128+127）⇒ **J2 ✅ 16/16**。三枚 sha（`24b57053…` / `a7ffff6b…` / `e998c810…`）与 `036` **逐字相同**
   ⇒ 仓库状态没漂移，后续判据可直接引用。
2. **⛔【实测·推翻 `036` §3.3】`kv8_ori_plane` 的 decode 2 页快路径是对的**：1440 次调用**逐行**核对
   "窗口内每一行：真页 `kv_i8`/`kv_scale` 反量化 vs 算子真正读到的 scratch 行" ⇒ **`true_bad=0`、`max_abs=0`、`sha_true == sha_got`（1440/1440）**。
   **读侧把该读的行逐比特送到了算子面前** ⇒ 036 那条"改 decode 分支为全序页表"的命令**没有可修对象**（它不再是候选）。
3. **★【实测】NaN 出在"池命中边界"上的第一压缩层**：故障请求在 **40 层里有 37 层的 hidden_states 整条 NaN（5120/5120）**，
   且**第一次出现是第一个 `ratio=2` 层（`layers.3`）的输入** ⇒ NaN 诞生于 **`layers.2`（第一个压缩层）的 forward 内部**；
   写侧入参（`kv8_store_rows` 的 `values`）在更下一层开始全是 NaN。**SWA 面读路径与 `kv8_ori_plane` 都不是出生地。**
4. **★【实测】触发条件不是"命中"，而是"池 unit 数恰好等于工作集"**：C0 几何（L5+SWA-quant，int8）、16×4096、144 MiB：
   | 池 unit | 池字节 | `CPU→GPU` 实际搬的字节 | `BlockRemoved:CPU` | J2 | NaN |
   |---:|---:|---:|---:|---|---|
   | **1152** | 144 MiB | **231,669,760** | **10** | ❌ 1/16 | **37 层** |
   | **1280** | 160 MiB | **231,669,760（一模一样）** | **0** | ✅ 16/16 | 0 |
   | 1536 | 192 MiB | 231,669,760 | 0 | ✅ | 0 |
   | 4096 | 512 MiB | 231,669,760 | 0 | ✅ | 0 |
   | 128 | 16 MiB | **0**（不命中） | 1342 | ✅ | 0 |
   | 8 | 1 MiB | 0（不命中） | 0 | ✅ | 0 |
   ⇒ **命中集与搬的字节完全相同、KV 事件计数完全相同**，唯一变量是池容量 ⇒ **这是一个纯粹由"容量欠配"触发的池内行复用缺陷**（【推断】见 §4）。
5. **★【实测】纯 BF16 不复现**：`L5+L1`、纯 BF16、同样的 **1 行 decode replay + 池命中**（`q_rows=1`、`CPU→GPU=272.9 MB`、`BlockStored:CPU=714`）
   ⇒ **J2 ✅ 16/16、每层 hidden_states 0 处 NaN**。⇒ 本缺陷**不是**"命中 + 1 行 decode"单独触发（`036` 的第三条判据需要补一条限定）。
   ⚠️ **口径边界**：BF16 臂的"0 NaN"是**每层 hidden_states** 的实测；**写侧 STORE 级探针在 BF16 下不会触发**
   （它的 log 站点在 `kv8_ori_plane` 里，而 BF16 走不到那个函数）⇒ BF16 的"0 NaN"**不含** store 级证据（【未确认】）。
   ⚠️ **诚实边界**：两臂的起服日志都是 `num_units=1152 / kv_bytes_per_unit=131072`，且 `BlockStored/Removed` 计数逐字相同
   ⇒ **"unit 成本不同"这条解释在本任务的数据里没有得到支持**（`035` 记的 77,824 B/unit 是 **D 几何**的读数，**不是 C0**）。
   为什么同样是 1152 unit、同样的存/删计数，int8 会读到 NaN 而 BF16 不会 —— **【未确认】**（这正是 §11-1 那条探针要回答的）。
6. **★【实测】可交付的防护**：写侧 NaN 自检（`a2/agents/G_kv8fix/kv8_nan_guard.py`，env 门控、fail-fast）——
   它抓到的**第一现场**就是 `kv8_store_rows` 入参 `nan=512`，而读侧只能看到后果。
7. **⇒ `killer` 结论**：**int8 两条杠杆（×1.4655 / ×1.9133）维持否决**，理由第四次改写为
   **"池 unit 数欠配时，命中块读到了从未写入过的池行（未初始化 ⇒ NaN）⇒ 第一压缩层 NaN ⇒ 首 token 错"**；
   **修点在卸载层的 unit 记账/分配，不在 attention 读侧**。

---

## 1. 复现（任务书 §1-2）

臂 = `036` 的 **C0 几何**（L5 + SWA-quant：`XL1=0 XSWA=1 XRING=0`，20 张量分量，池 144 MiB，16 请求 × 4096 token，`MAX_TOKENS=1`）。

```bash
bash tools/a3_chip.sh c2 --timeout 1800 --name g-repro -- \
  bash /work/agents/G_kv8fix/scripts/run_repro.sh
```

| 臂 | prompt | fill sha | replay sha | J2（`036` 口径） | replay p50 TTFT |
|---|---:|---|---|---|---:|
| `g-C0-4096` | 4096 | `24b57053…` | `a7ffff6b…` | ❌ **mismatched=[15]** | **62.9 ms** |
| `g-C0-4095` | 4095 | `e998c810…` | `e998c810…` | ✅ 0 | **197.2 ms** |

* **【实测】**与 `036` 的 `f8-T-4096` / `f8-T-4095` 三枚 sha **逐字相同**；
* **【实测】**TTFT 3.1×（62.9 vs 197.2 ms）= "首 token 由 1 行 decode 产生 vs 需 prefill 补算 127 行"的独立佐证。

---

## 2. ★★ 任务书要求的那一格：`kv8_ori_plane` 到底对不对（**实测：对**）

### 2.1 探针（只读，零行为改动）

`a2/agents/G_kv8fix/probe/g_dump.py`，经 `probe/sitecustomize.py` **叠在 pkg 的补丁链之上**
（先逐字 exec `pkg-ring/patch/sitecustomize.py`（P2→PGP），再装自己的 import hook；**一行别人的代码都没改**）。
它做四件事：

1. **映射核对**：对"窗口内每一行"（`[max(0, lens-window), lens-1]`）比较
   `kv8_dequant_rows(真页行)` vs `scratch[table[b, p//bs], p%bs]`；
2. **写侧快照**：`kv8_store_rows` 入参在**真 store 之前**快照（NaN/Inf 计数、slot、量化前后）；
3. **每层 hidden_states 画像**：`multistream_preprocess` 入口的 NaN 计数；
4. **平面扫描**：每层 SWA / 长KV / indexer-k / compressor-state 的 NaN/Inf。

### 2.2 【实测】映射核对：1440/1440 全绿

| 臂 | 调用数 | `true_bad` | `max_abs` | 结论 |
|---|---:|---:|---:|---|
| `g-dump6-pool`（池臂，J2 ❌） | 1440（760 decode + 680 prefill） | **0** | **0** | ✅ 映射逐比特正确 |
| `g-dump6-cold`（冷参考，J2 ✅） | 1440 | **0** | **0** | ✅ |

* 覆盖：`pages_per_req=2`、`sizeof(scratch)=(2,128,1,512)`、`table` 全表（含窗口外被置 0 的列）、
  物理页号（`block_table`）与页内偏移（`off` 保持原值）。
* **⇒ `036` §3.3 的落点（decode 2 页快路径）不成立**。`033` 的"页粒度整前缀重建"范式在这里**没有可修的对象**。

### 2.3 【实测】NaN 的出生点

`g-pair-int8`（C0 几何，J2 ❌）与 `g-pair-bf16`（纯 BF16，J2 ✅）逐层画像：

```
g-pair-int8:  layers.0 / 1 / 2 输入 nan=0
              layers.3        nan=5120 / 5120   ← NaN 第一次出现
              layers.4 … 39   nan=5120          ← 之后全 NaN（40 层里 37 层）
g-pair-bf16:  （0 行）           ← 一层都没有
```

* 本模型的层角色（同请求 ATTN 记录实测）：`ratio=0` = layers 0,1；**第一个 `ratio=2`（压缩层）= layer 2**。
* ⇒ **NaN 诞生在 `layers.2` 的 forward 内部**（第一个压缩层），而 layer 2 的**写侧入参是干净的**
  （`snap_in=((1,512), 0)`）⇒ 压缩器输出没 NaN；NaN 来自该层的**压缩注意力/索引器读出来的东西**。
* 到了 layer 4 起，写侧入参整条 NaN ⇒ 量化器把 `scale` 写成 NaN ⇒ **SWA int8 页在位置 4095 出现 NaN scale**
  （`nan_scale=[4095]`，池臂 **37 次**、冷臂 **0 次**）⇒ 读路径忠实地把 NaN 喂给算子（这才是"读侧看到 NaN"的原因）。

---

## 3. ★ 触发条件：不是"命中"，是"池 unit 数 == 工作集"（**实测表**）

C0 几何（int8），16×4096，`--enforce-eager` + 探针（TTFT 被探针拖慢，故 TTFT 只用于对照，**不用于判据**）：

| 臂 | 池 | unit | `CPU→GPU`（实际搬的字节） | `BlockStored:CPU` | `BlockRemoved:CPU` | J2 | NaN（层数 / 位置） |
|---|---:|---:|---:|---:|---:|---|---|
| `g-C0-4096` / `g-dump6-pool` / `g-pair-int8` | 144 MiB | **1152** | **231,669,760** | 714 | **10** | ❌ 1/16（#15） | **37 层**，`scale@4095` |
| `g-bound-160` | 160 MiB | **1280** | **231,669,760** | 714 | **0** | **✅ 16/16** | **0** |
| `g-bound-192` | 192 MiB | 1536 | 231,669,760 | 714 | 0 | ✅ | 0 |
| `g-sweep-4` | 512 MiB | 4096 | 231,669,760 | 714 | 0 | ✅ | 0 |
| `g-sweep-5` | 16 MiB | 128 | **0**（不命中） | 1418 | 1342 | ✅ | 0 |
| `g-dump5-cold` / `g-dump6-cold` | 1 MiB | 8 | 0 | 0 | 0 | ✅ | 0 |
| `g-pair-bf16`（**纯 BF16**） | 144 MiB | 1152 | **272,957,440** | 714 | **10** | ✅ | 0 |

**★ 这张表最关键的一行是 144 MiB vs 160 MiB**：`CPU→GPU` **逐字节相同**（231,669,760）、
`BlockStored:GPU=6326`、`BlockStored:CPU=714`、`BlockRemoved:GPU=3672` **全等**，**只有池容量不同** ⇒ ❌ 变 ✅。
（唯一的差别是 `BlockRemoved:CPU`：144 MiB = **10**、160 MiB = **0**。）
⇒ **不是"哪些块命中"的问题，而是"池里每个 unit 指向哪一行"的问题**（【推断】）。

**★ 一个反直觉的细节（必须单说）**：`BlockRemoved:CPU` 在**失败的 int8 臂（1152 unit）与通过的 BF16 臂
（同样 1152 unit）里都是 10**，而 1280 / 1536 / 4096 unit 的臂都是 0 ⇒ **"删除计数 > 0"本身不是判据**
（`027` 的监测式只能当**必要**条件，不是充分条件）。本任务里唯一 ❌ 的 int8 臂恰好落在
**"池 = 1.000× 工作集"**这一格；**1.11× 起就安全**。

**旁证（`009`/`021` 的老账）**：`027` 已经实测过"欠配的表现是**级联归零、没有中间态**"，并把
`kv_offload_block_removed_total{medium="CPU"} == 0` 当成上线监测判据。本任务给出的是**它的另一面**：
**"刚好够"（1.000×）不是安全点** —— 它会给出**不归零、但内容错**的中间态（NaN）；而且这一格的
`BlockStored:CPU=714` / `BlockRemoved:GPU=3672` / `CPU→GPU=231.7 MB` 与 1.11× 的安全臂**逐字相同**，
**现成的监测指标看不见它**。**这是 `027` 没覆盖的一格。**

---

## 4. 【推断】机制（**未做 kernel 级证明**，但每条都对着一条实测）

1. unit 级池的行是**按块分配/复用**的；当 `units == 工作集` 时，某个组的 store 会把**另一请求仍会命中的 unit 行**改写/复用到别处；
   ⇒ replay 的"命中"读到的是**从未写入过的行**（未初始化 HBM ⇒ **NaN**）或别人的行；
   ⇒ 第一压缩层拿到 NaN 的 K/V ⇒ 该 token 输出 NaN ⇒ 逐层放大（37 层）。
2. **为什么只有"对齐 + int8"才看见**：
   * **对齐（4096）** ⇒ 首 token 全由 **1 行 decode** 产生 ⇒ **只有 1 个 token 走压缩器**，ring 状态只能靠池/命中拼出来，
     一旦读错就是**该 token 全错**；非对齐（4095）会重算尾块 ⇒ 走完整前向 ⇒ 状态被正确重建 ⇒ ✅。
   * **int8** ⇒ 与 BF16 **同样 1152 unit、同样存/删计数**却会读到 NaN ⇒ 差异只能出在"**每个 unit 里放了哪些页 / 页在 unit 内的偏移**"
     （int8 的 SWA 页比 BF16 多一个 scale 面、页大小不同）⇒ 行级账目在**临界容量**下对不上。
     ★ 这只是【推断】：**本任务没有证据支持"unit 成本不同"**（两臂日志都是 131,072 B/unit）。
3. **与 `036` 的两条旧结论完全相容**：池往返字节审计（12,928 次、mismatch=0）**只证明"搬的字节没被打乱"**，
   不证明"**每个块被搬到了它该在的那一行**"；`f_read_audit` 的 960/960 只覆盖**同一形状**下的页。

**⇒ 责任方 = 卸载层的 unit 记账/分配（L5 per-group bpc + L1 池），不是 attention 读侧、不是 int8 的数值路径。**

---

## 5. 交付：写侧 NaN 自检（任务书 §3 的候选 (3)，**升级到写侧**）

`a2/agents/G_kv8fix/kv8_nan_guard.py`（**独立文件，env 门控，fail-fast，不碰生产代码**）：

```python
# 叠在自己的 overlay 里（pkg 的 sitecustomize 之后）
import kv8_nan_guard; kv8_nan_guard.install_hook()
# env: KV8_NAN_GUARD=1 KV8_NAN_GUARD_MODE=error   # 默认 error ⇒ 直接 raise
```

* **为什么在写侧**：实测的**第一现场**是 `kv8_store_rows` 的入参 `values`（**整条 512 维 NaN**，在真 store 之前快照）；
  读侧（`kv8_ori_plane`）只能看到后果（页里已经写进 NaN）。
* 报错内容含 `store# / nan / inf / shape / 坏行 / slot 前 8 / 是否 int8 平面`，把"静默算错"变成"响亮失败"。
* **未做**：还没在真服务上跑过 `MODE=error` 的整轮（本任务时间用在了根因定位上）⇒ 标注【未确认】。

### 5.1 任务书候选 (2) 的对应物：**运行期前提判定**（落在**池侧**，不在读侧）

任务书候选 (2) 是"找出快路径的正确前提，加运行期断言/分支判定"。本任务实测的结论是：
**前提不在 `kv8_ori_plane` 里**（它的前提 100% 成立），而在卸载层的容量账：

```python
# 建议加在起服自检（池分配之后，只读、零成本）
need_units = 观测到的工作集 unit 数          # 027 的 _dbg_tally 口径
have_units = 池 unit 数                      # 现有日志里就有
if have_units < need_units * 1.05:           # 实测：1.000× ❌ / 1.11× ✅
    raise RuntimeError("池 unit 数欠配（%.3f×）：命中块可能读到未写入的行 ⇒ 首 token 静默错"
                       % (have_units / need_units))
```

* **依据**：本任务 §3 的实测表（1.000× ❌ / 1.11× ✅）。
* **没采纳的更强形态**：把 `kv8_ori_plane` 的 decode 快路径改成"前提不成立就回退"——**没有必要**，
  因为它的前提**没有不成立过**（1440/1440）；回退只会白付时延。

---

## 6. 判据对账（任务书 §1-4 的清单）

| # | 判据 | 结果 | 备注 |
|---|---|---|---|
| 1 | 复现 4096 ❌ / 4095 ✅ | ✅ **【实测】**逐字复现 `036` | §1 |
| 2 | 读懂 decode 2 页快路径的假设 | ✅ **【实测】假设成立**（1440/1440 逐行） | §2.2 |
| 3 | 改法 (1)(2)(3) | ⛔ **(1)/(2) 无对象**（读侧无罪）；**(3) 已交付**（写侧 fail-fast） | §2.2 / §5 |
| 4 | **J2 在 C0 / D / F 三臂转 ✅** | ⛔ **未达成**（这不是读侧能修的；见 §3/§4） | —— |
| 5 | 对齐/非对齐都 ✅ | 部分：**非对齐 ✅（现状）**；**对齐仍 ❌** | §1 |
| 6 | 容量不退化（33,295 / 43,469） | **未测**（本任务没改容量路径；【未确认】） | —— |
| 7 | 四条判据不回归 | 本任务臂的 `BlockStored:CPU=714` 与 `036` 一致【实测】；其余未测 | —— |
| 8 | decode 时延增量 ≤ +0.5%/step | **未测**（没改生产代码，无时延可量） | —— |
| 9 | `x-E` 变长前缀（4096→2048） | **未做**（前提未满足） | —— |
| 10 | ★ **新增**：J2 随池容量的翻转边界（144 ❌ / 160 ✅） | ✅ **【实测】**（本任务最有价值的新判据） | §3 |
| 11 | ★ **新增**：纯 BF16 同形状对照 | ✅ **【实测】**不复现（0 NaN） | §3 |

---

## 7. cannbot 对照（AGENTS.md §6）

本任务**不改 kernel、不做量化数值门**（只做"接线 / 保真 / 定位"），按索引查了三条：

| 查的地方 | 它说什么 | 采纳 / 没采纳 |
|---|---|---|
| `model-infer-kvcache/SKILL.md:102-113`（PA 映射） | `物理 slot = block_table[b, pos//block_size] × block_size + pos%block_size`，**单一 block_size** | ✅ 逐字采纳：本任务的"映射核对"就是按这条逐行算的，且**实测全绿** ⇒ 上游 block/slot 映射在单请求内无歧义 |
| `model-infer-kvcache/SKILL.md:224-242`（稀疏/滑窗的**语义边界**） | *"长序列 `KV_len > sliding_window` 的正确性必须靠模型层保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，**不是 op 层负责**"* | ✅ **采纳并成为本任务的核心判据**：正因为"滑窗与压缩前缀的正确性由模型层/状态负责"，`kv8_ori_plane` 只要**逐比特搬运**就算尽责 —— 本任务实测它确实尽责，**问题因此必然在"模型层状态有没有被恢复"**（= 池/卸载层） |
| `model-infer-quantization/SKILL.md:424-451`（§7.1 等价性自检） | *"不能只看代码 diff，必须证明真实运行"*；W8A8 允许细微 token 差异 | ✅ 采纳：本任务**不用 sha 判死量化本身**，只把它当"接线一致性"判据；并新增一条判据——**"命中集相同 ≠ 内容相同"**（144 vs 160 MiB 的实测就是反例） |

* **没查/没采纳**：`ops/*`（本任务不写 kernel）；`triton-*`（不写 Triton）。
* **文档空白**：cannbot 里**没有**"卸载池欠配时行复用语义"的章节 ⇒ 这一格是**我们的实测新增**，建议后续补进 `027` 的上线监测清单。

---

## 8. 复现入口（每条都是**一条命令**）

```bash
# 0) 取代码（COS）：a2/logs/raw/038-g-kv8fix/ 与本目录同源；容器内 = /work/agents/G_kv8fix/
#    （在 A3 上）tar xzf … -C ~/projects/dsv41-upstream-pr/agents/G_kv8fix

# 1) ★ 复现（4096 ❌ / 4095 ✅，~6 min）
bash tools/a3_chip.sh c2 --timeout 1800 --name g-repro -- \
  bash /work/agents/G_kv8fix/scripts/run_repro.sh

# 2) ★★ 池容量边界扫描（144 ❌ / 160 ✅ / 192 ✅，~8 min）
bash tools/a3_chip.sh c2 --timeout 2400 --name g-bound -- \
  env G_PLANE_FROM=0 bash /work/agents/G_kv8fix/scripts/run_pool_boundary.sh

# 3) 纯 BF16 对照（J2 ✅、0 NaN，~2.5 min）
bash tools/a3_chip.sh c2 --timeout 1800 --name g-bf16 -- \
  env G_PLANE_FROM=1 G_PLANE_PERIOD=40 bash /work/agents/G_kv8fix/scripts/run_bf16_arm.sh

# 4) 带探针的一对（int8 池臂 ❌ + 冷臂 ✅，含每层 NaN 画像，~6 min）
bash tools/a3_chip.sh c2 --timeout 1800 --name g-dump -- env G_DUMP_VERBOSE=1 \
  bash /work/agents/G_kv8fix/scripts/run_dump_batch.sh

# 5) 交付的自检（写侧 fail-fast，替换上面任意一条的 overlay 即可）
#    在 probe/sitecustomize.py 里加： import kv8_nan_guard; kv8_nan_guard.install_hook()
#    env: KV8_NAN_GUARD=1 KV8_NAN_GUARD_MODE=error
```

**锁退出码 75 = 没抢到锁，是重试不是失败。**

---

## 9. 原始数据与代码

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/038-20260922-kv8-decode-fix.md` |
| 原始数据 | `a2/logs/raw/038-g-kv8fix/`：16 条臂的 `*.client.json` / `*.kv_events.log` + **8 份探针 dump（gz）** |
| 代码 | `a2/agents/G_kv8fix/`：`probe/{g_dump.py,sitecustomize.py}`、`scripts/{prepare_overlay.sh,selfcheck_g.py,run_repro.sh,run_dump_arm.sh,run_dump_batch.sh,run_bf16_arm.sh,run_pool_sweep.sh,run_pool_boundary.sh,show_arm.py}`、**`kv8_nan_guard.py`** |
| 容器内 | 同源在 `/work/agents/G_kv8fix/`（A3 A3-node1） |

### 9.1 关键读数速查（判据用）

| 臂 | 池 unit | `CPU→GPU` MB | J2 | NaN 层数 | 文件 |
|---|---:|---:|---|---:|---|
| `g-C0-4096` | 1152 | 231.7 | ❌ | —（未装探针） | `g-C0-4096.client.json` |
| `g-C0-4095` | 1152 | 231.7 | ✅ | — | `g-C0-4095.client.json` |
| `g-dump6-pool` | 1152 | 231.7 | ❌ | **37** | `dump-g-dump6-pool.txt.gz` |
| `g-dump6-cold` | ~8 | 0 | ✅ | 0 | `dump-g-dump6-cold.txt.gz` |
| `g-bound-160` | 1280 | **231.7（同 144）** | **✅** | 0 | `dump-g-bound-160.txt.gz` |
| `g-bound-192` | 1536 | 231.7 | ✅ | 0 | `dump-g-bound-192.txt.gz` |
| `g-pair-bf16` | 1152（BF16） | 272.9 | ✅ | **0** | `dump-g-pair-bf16.txt.gz` |
| `g-pair-int8` | 1152 | 231.7 | ❌ | **37** | `dump-g-pair-int8.txt.gz` |
| `g-sweep-4` | **4096** | 231.7 | ✅ | 0 | `dump-g-sweep-4.txt.gz` |
| `g-sweep-5` | **128** | 0 | ✅ | 0 | `dump-g-sweep-5.txt.gz` |

---

## 10. 诚实边界（哪些没测 / 哪些是推断）

| # | 事项 | 状态 |
|---|---|---|
| 1 | `036` 的 4096 ❌ / 4095 ✅ 复现 | ✅ **【实测】**（sha 逐字） |
| 2 | `kv8_ori_plane` 映射逐比特（1440 次） | ✅ **【实测】** |
| 3 | 写侧入参 NaN 的第一现场 | ✅ **【实测】** |
| 4 | NaN 诞生层 = 第一个压缩层（`layers.2`） | ✅ **【实测】**（每层 hidden 画像） |
| 5 | 池 144 ❌ vs 160/192/512 ✅（命中字节完全相同） | ✅ **【实测】** |
| 6 | 纯 BF16 同形状不复现（且 0 NaN） | ✅ **【实测】** |
| 7 | **"欠配 ⇒ 行复用 ⇒ 读到未写入行"的具体代码行** | ⛔ **【推断】**（未做 unit 分配器级证明） |
| 8 | BF16 压到临界容量是否也炸 | **【未确认】**（一条命令见 §8-3 + 把 `OFFLOAD_BYTES` 取 1152→800 unit 那一档；**这一格决定它是"int8 的"还是"1.000× 临界容量的"**） |
| 8b | BF16 臂的**写侧**（store 级）NaN | **【未确认】**（BF16 走不到 `kv8_ori_plane`，日志站点不触发；要在 `_write_compressed_source` 处单独挂点才行） |
| 9 | 是否与 `037` 的非确定性同源 | **【未确认】**（本任务现象**确定性**：`replay1_sha == replay2_sha`、跨进程同 sha） |
| 10 | 容量（33,295 / 43,469）、decode 时延、`x-E` 变长前缀 | **未测**（本任务没改生产代码，无回归可量） |
| 11 | 写侧自检在真服务整轮跑通 | **【未确认】** |

**红线核对**：没发 PR / issue / 评论；没写 `upstream-v41/`；没用 `/tmp`；占卡全走 `a3_chip.sh`（无 75 退出）；
没手设 `ASCEND_RT_VISIBLE_DEVICES`；没碰 c0 / c1 / `mooncake-*` / `jitpgo-*` / `dsv41-a3`；**没改任何生产代码或别人的文件**
（全部通过自己的 overlay + import hook）；传文件全走 `cos-xfer.sh`；结论逐条标了【实测】/【推断】/【未确认】。

---

## 11. 给主代理的下一步（按价值排序，每格一条命令）

1. **【确证欠配机制】**在 `pgp_scheduler`/`P2` 侧给"unit 行分配"加只读探针（记录 `(group, block_id) → unit 行号`，
   以及"该行是否被另一个 block 复用"），跑 144 MiB 那条臂 —— 应能看到"同一行 → 两个 block"。**这是把 §4 从【推断】变成【实测】的唯一一步。**
2. **【修法候选】**：池容量按**组**留余量（或把 `units == 工作集` 判为**不可上线**）；`027` 的监测判据建议追加
   `units_ratio ≥ 1.05`（本任务实测 1.11× 已安全；**但 1.000× 与 1.11× 的现成计数完全相同 ⇒ 必须新增一个
   "行级 provenance / 命中块可追溯"指标，否则监测看不见这一类**）。
3. **【上线口径】**`035` §7.1 的运行期对账应升级为：**"命中集相同 ≠ 内容相同"** ⇒ 加一条
   "**每个命中块必须能追溯到它最后一次 store**"（行级 provenance），否则无法排除本类缺陷。
