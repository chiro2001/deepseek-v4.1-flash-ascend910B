# rope `index_select`：形状矩阵 / draft / 序列 / 图 / op 计数（A3 单 die 实测）

> 2026-09-21｜机器：`A3-node1` 槽位 **c1 = die6**，容器 `prbench-c1`，`npu:0` = **Ascend910_9382**，
> torch 2.10.0 + torch_npu 2.10.0.post4
> 脚本：`/work/bench/rope_edge_cases.py`（=`upstream-v41/pr/rope_edge_cases.py`，五 phase 全量）
> ＋ `/work/bench/probe_gather_kernels.py`（解释 stock 的 kernel 构成）
> ＋ `agents/R_rope/bench/rope_graph_sizes.py`（本任务新写，补 ACLGraph 的尺寸扫描，见 §1.4）
> 三者本地/远端 sha256 一致：`eb0c8cbe…a7ef` / `f7bc33b8…c580` / `70f5ecab…92bb`
> 被测代码（**按路径 import，不是转述**）：`vendor/rope_dsv4_stock.py` sha256 `0f9177f5…ef7`
> （merge base `c173a64a`）与 `vendor/rope_dsv4_pr.py` sha256 `982bb28d…f6`（PR head `d4167f52`）
> 原始证据：本文件 §4 列出的 `logs/raw/35-rope-*`（JSON + 日志，已从 A3 取回）
> 跑法：`bash tools/a3_chip.sh c1 --name <任务> --timeout 1800 -- python3 …`（单次运行一把锁；
> 三次运行——主跑、probe、尺寸扫描——依次占用同一个槽位 c1，每次都跑完即释放，退出码都是 0）

---

## 0. 一句话

**五个 phase 全部跑成：主跑 `checks=32 pass=32 fail=0`，补跑的 ACLGraph 尺寸扫描 `checks=5 pass=5 fail=0`。**
空批 **n=0 逐位一致**；非连续 stride / 倒序 / 重复位置 / int32 / 2-D fallback 全部逐位一致；
**每次取表（cos+sin 各一次 gather）6 → 2 个 kernel**；eager n=4096 **−376 µs**、**ACLGraph n=4096 −384 µs**。
唯一两个 PR 更慢的格子是 **int32 的 eager 小尺寸**（+20.7 / +28.9 µs），原因已定位（PR 在对 cos/sin 各做一次 cast），**写进 §2，不藏**。

---

## 1. ★ 结论表

全部数字都是**同进程 A/B**、两臂用同一张 RoPE 表、各自独立输出 buffer，`--reps 50`（eager）/60（图）。
「逐位一致」= `torch.equal == True` 且 `max_abs_diff = 0.000e+00`，不是容差比较。

### 1.1 形状 / 布局 / 顺序 / dtype 矩阵（★ = RFC [91] 点名项）

`stock≡PR` = 两版本输出逐位相同；`≡full[pos]` = 与**改前的原始语义**（高级索引 `full_rope[pos]`）逐位相同；
`tail` = 缓冲区里 `[:n]` 之外的行**没被动过**（padded 批的关键性质）；`buf` = 仍然写进预分配 buffer（地址不变）。

| 用例（positions） | stock≡PR | ≡`full[pos]` | 形状 | tail | buf | eager Δ (µs) |
|---|:--:|:--:|:--:|:--:|:--:|---:|
| ★ n=0（空批） | ✅ | ✅ | `(0,1,1,64)` | ✅ | ✅ | **−4.0** |
| n=1 | ✅ | ✅ | `(1,1,1,64)` | ✅ | ✅ | −7.8 |
| n=8 | ✅ | ✅ | `(8,1,1,64)` | ✅ | ✅ | −6.4 |
| n=192 | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −26.2 |
| n=2048 | ✅ | ✅ | `(2048,1,1,64)` | ✅ | ✅ | −183.8 |
| ★ n=4096（= `max_num_batched_tokens`，prefill 尺寸） | ✅ | ✅ | `(4096,1,1,64)` | ✅ | ✅ | **−376.3** |
| ★ n=192 int32 连续（dflash buffer dtype） | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | **+20.7** ⚠️ |
| ★ n=2048 int32 连续（dflash dtype，n 取自 decode 批） | ✅ | ✅ | `(2048,1,1,64)` | ✅ | ✅ | −144.2 |
| ★ n=192 int64 **非连续** strided view（stride 2） | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −30.7 |
| n=192 int64 倒序（`flip`，行序必须保留而非排序） | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −12.1 |
| n=192 int64 重复位置（只有 4 个不同行） | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −13.9 |
| n=192 int64 2-D `[n,1]`（PR 的 gather fallback 分支） | ✅ | ✅（展平比较） | `(192,1,1,64)` | — | — | — |
| n=192 int64 2-D `[n,1]` strided（`as_strided`） | ✅ | ✅（展平比较） | `(192,1,1,64)` | — | — | — |
| n=192 int64 2-D `[n,1]` 转置视图 | ✅ | ✅（展平比较） | `(192,1,1,64)` | — | — | — |
| ★ n=192 int32 非连续 strided（cast 之后再切片） | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | **+28.9** ⚠️ |
| n=32 int64 2-D `[2,16]`（expand 不兼容） | ✅ 两版**同样抛 `RuntimeError`** | — | — | — | — | — |

**两条路径都测了**：`use_cache=True`（写预分配 buffer）与 `use_cache=False`（PR 的 `index_select` 无 out 版本），
每行的 `nocache bit-exact=True` 都在 JSON 的 `checks[].detail` 里。

> ⚠️ 2-D 那三行**不是** PR 的新路径：两个版本都走 4-D `gather` fallback（正是 PR 保留 fallback 的目的），
> 所以「一致」在这里的含义是**行为不变**，不是「新代码更快」。

### 1.2 `draft_index=1..5` 的逐位一致性（投机解码路径）

检查项包括：写的是 `spec_runtime_buffer[cfg][group][K-1]` 这一行、**其它 spec 行未被碰**、**普通 runtime buffer 未被碰**。

| K | n | 逐位一致 | 写对行 | 其它行未动 | eager Δ (µs) |
|---:|---:|:--:|:--:|:--:|---:|
| 1 | 8 | ✅ | ✅ | ✅ | −9.1 |
| 2 | 8 | ✅ | ✅ | ✅ | −8.7 |
| 3 | 8 | ✅ | ✅ | ✅ | −10.4 |
| 4 | 8 | ✅ | ✅ | ✅ | −9.3 |
| 5 | 8 | ✅ | ✅ | ✅ | −9.9 |
| 1 | 192 | ✅ | ✅ | ✅ | −11.7 |
| 2 | 192 | ✅ | ✅ | ✅ | −10.6 |
| 3 | 192 | ✅ | ✅ | ✅ | −12.4 |
| 4 | 192 | ✅ | ✅ | ✅ | −11.9 |
| 5 | 192 | ✅ | ✅ | ✅ | −10.7 |

⇒ **K=1..5 全绿**，且由 profiler 独立验证：draft 路径 **2.00 kernel/lookup = 普通 cache 路径的 2.00**（没有额外开销）。

### 1.3 真实调用序列（40 层 burst / 一个 spec step）

| 用例 | stock (µs) | PR (µs) | Δ (µs) |
|---|---:|---:|---:|
| 单次热调用 n=192 | 121.92 | 111.60 | −10.3 |
| **40 层 back-to-back burst**，n=192（每 call） | 92.09 | 81.86 | **−10.2** |
| **一个 spec step**（1 次主 + 4 次 draft），总计 | 493.40 | 448.21 | **−45.2** |
| 同上，摊到每次 lookup | 98.68 | 89.64 | −9.0 |

spec step 另有独立的结构检查：主 buffer 与 spec 行 0..3 都等于 `full_rope[pos]`，**未被使用的第 4 行保持未动**（两版本皆然）。

### 1.4 ACLGraph（生产帧）

两个来源，数值互相印证：

| n | 来源 | stock (µs/call) | PR (µs/call) | Δ (µs) |
|---:|---|---:|---:|---:|
| 192 | `rope_edge_cases.py` graph phase | 48.69 | 18.80 | −29.9 |
| 1 | `rope_graph_sizes.py` | 19.94 | 7.86 | −12.1 |
| 8 | `rope_graph_sizes.py` | 25.07 | 7.78 | −17.3 |
| 192 | `rope_graph_sizes.py` | 48.18 | 18.48 | −29.7 |
| 2048 | `rope_graph_sizes.py` | 219.45 | 19.32 | −200.1 |
| ★ **4096** | `rope_graph_sizes.py` | **407.78** | **23.83** | **−384.0** |

`rope_graph_sizes.py` 是本任务新写的补测脚本（40 层 burst 捕获 + 回放，每 replay 摊到每 call），
它**复用 `rope_edge_cases.py` 的 `Arm`**，即同样是按路径 import 的真实两版函数，不是重写；
每次捕获后还断言回放后的 buffer 与 `full_rope[pos]` **逐位相同**（5/5 通过）。

⇒ **图内不翻符号**：eager −376 µs、ACLGraph −384 µs，两者同量级甚至图内更大（n=4096）。
与 MoE mask（图内被摊掉）的机制差异见 [`05`](05-20260921-rope-opcount-and-bitexact.md) §3.1。

### 1.5 op 计数（torch_npu profiler，生产表 `T=1M`，3 个 active step）

口径：这里 1 次 **lookup = 一次 `get_cos_and_sin_dsa()` = cos / sin 两次 gather**。

| 场景 | stock | PR | 明细（原始 kernel 名） |
|---|---:|---:|---|
| int64 `use_cache=True` | **6.00** | **2.00** | stock：`BroadcastTo` 6 + `Cast` 6 + `GatherElementsV2` 6<br>PR：`IndexSelect_GatherV3` 6 |
| int32 `use_cache=True` | 7.00 | 4.00 | stock：三件套 + `InplaceCopy_Cast` 3<br>PR：`IndexSelect_GatherV3` 6 + `InplaceCopy_Cast` 6 |
| int64 `draft_index=3` | 6.00 | 2.00 | 同 int64 行（draft 路径无额外 kernel） |

⇒ **每个方向 3 kernel → 1 kernel**（`BroadcastTo` + `Cast` + `GatherElementsV2` ⇒ 单个 `GatherV3`）；
每次取表 **6 → 2**。这与 log [`05`](05-20260921-rope-opcount-and-bitexact.md) 在单卡机上测到的单方向结论（3 → 1）、
以及 A3 部署的账目（+6624 `GatherV3` 对冲 −4104/…）**方向一致**。

---

## 2. ⚠️ 诚实标注：两个 PR 更慢的格子（int32 的 eager 小尺寸）

| 用例 | stock (µs) | PR (µs) | Δ | 归因 |
|---|---:|---:|---:|---|
| n=192 int32 连续（dflash buffer dtype） | 154.35 | 175.04 | **+20.7**（+13%） | 【推断】PR 在 `_rope_index_1d` 里对 **cos 与 sin 各 cast 一次**（profiler：PR int32 4.00 vs int64 2.00 = +2 cast/lookup；stock 只 +1） |
| n=192 int32 非连续 strided | 158.43 | 187.36 | **+28.9**（+18%） | 同上 |

【实测】上面两个数字本身（`reps 50`，中位数）。
【推断】归因方向（多一次 cast）由 profiler 计数支持，但**没有**逐 op 的时序分解来证明这 +20.7 µs 全部来自 cast。

**为什么仍然可以接受**（【推断】，需评审判断）：

1. **数值完全一致**（`max_abs=0.000e+00`），这两格只是速度；
2. int32 只出现在 **dflash proposer 的 `_context_positions_buffer`** 一处；同一批里那个**批量更大**的 int32 用例（n=2048，decode 批口径）是 **−144.2 µs**；
3. 但这两个格子说明「int32 + 小 n」是 PR 的相对弱区，**如果要发 PR，应该主动写出来**（已写进草稿 §5）。

---

## 3. `probe_gather_kernels.py`：stock 的 3 个 kernel 是怎么来的，小表为什么更糟

【实测】同一个 stock 模块、同一个 n，只改表长与调用方式（`probe-gather.log`）：

| 配置 | kernel / `get_cos_and_sin_dsa()` | 构成 |
|---|---:|---|
| `T=1M` n=192，1 call/step | **6.00** | `BroadcastTo` 6 + `Cast` 6 + `GatherElementsV2` 6 |
| `T=8K` n=192，1 call/step | **12.00** | **`Transpose` 18** + 上述三件套各 6 |
| `T=1M` n=128，1 call/step | 6.00 | 同第一行 |
| `T=8K` n=128，1 call/step | 12.00 | 同第二行 |
| `T=1M` n=192，5 calls/step（每次新 positions） | 6.00 | 与 call 数、positions 新旧**无关** |

⇒ 【实测】**kernel 构成与表长有关、与 n / 调用模式无关**。生产表（1M）下 stock 每个方向就是
`BroadcastTo` + `Cast` + `GatherElementsV2` 三个 kernel —— 这正是本次融合要省掉的两个（广播与 cast）。
小表（8K）下 stock 更差（多 3 个 `Transpose`/方向，12 vs 6），**但本 PR 只声称省掉生产表下的两个**
——收益账目只在生产表（1M）下核算；**PR 在小表下的计数【未确认】**（probe 脚本只 profile 了 stock 臂）。

---

## 4. 每个数字的出处（可追溯）

本地（`upstream-v41/logs/raw/`）与 A3（`~/projects/dsv41-upstream-pr/agents/R_rope/out/`）各一份：

| 文件 | sha256（本地） | 含什么 |
|---|---|---|
| `35-rope-edge-cases-full-124940.json` | `bee4fcd6…e098` | 主跑全部 32 个 check（含逐行 detail）、27 行 timing、6 份 profile kernel 计数；`env.phases=[draft,graph,profile,seq,shapes]` |
| `35-rope-edge-cases-full-124940.log` | `a1263b2c…0b54` | 主跑 stdout（含 `SUMMARY checks=32 pass=32 fail=0`） |
| `35-rope-edge-cases-graphsizes-125250.json` | `fb87129c…8324` | ACLGraph 尺寸扫描 5 个 check + 每尺寸两臂统计 |
| `35-rope-graph-sizes.log` | `c37b05b3…447f` | 上者的 stdout |
| `35-rope-gather-probe.log` | `4cfc0995…ba30` | §3 的 5 个配置的 kernel 计数 |

JSON 内索引：§1.1 的 `checks[]/timings[]` 按 `case` 名一一对应（`group` ∈ extreme/baseline/dtype/layout/order）；
§1.2 是 `group=draft`；§1.3 是 `group=sequence`（另加 `draft`/`profile` 各一条结构检查）；
§1.5 是 `profiles["stock|int64|cache"]` 等 6 个 key 的 `counts`。

两臂源码 provenance（每次运行都会打印并与 JSON 一起落盘）：
stock `0f9177f5571d2bb36afba79949853122029b7d78ed35c01fc60651cf36cd3ef7`、
PR `982bb28d13fb04ce22522db2a1175e04703027b4a804eb9d907ef1dd7e4ca6f6`。

---

## 5. 还缺什么 / 未确认

| 项 | 状态 |
|---|---|
| 【未确认】**小表（8K）下 PR 臂的 kernel 计数** | probe 只 profiled 了 stock 臂（脚本设计如此）；生产表 1M 才是本 PR 的目标口径 |
| 【未确认】**+20.7 / +28.9 µs 的逐 op 归因** | 有 profiler 计数支持「多一次 cast」，没有逐 op 时序分解 |
| 【推断】**负 stride 位置张量** | PyTorch 直接拒绝负 step 切片（`step must be greater than zero`），脚本用 `flip` 做替代；**真正的负 stride 张量本次没构造出来**（构造不出来，不是没测） |
| 【未确认】**n>4096 / 多于 40 层** | 未测；`max_num_batched_tokens=4096` 是生产上限 |
| 【实测但非本 PR 口径】**A3 部署的整体收益** | 见 [`05`](05-20260921-rope-opcount-and-bitexact.md) 与草稿 §2；那是含其它非上游优化的部署账目 |
| 【未确认】**跨 CANN/驱动版本的稳定性** | 本次全部在 A3-node1（`Ascend910_9382`，容器内 CANN 9.x / torch_npu 2.10.0.post4）单机上 |
