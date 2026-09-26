# Issue 清账：改善空间与需求（2026-09-26）

> 把**所有外部反馈与未结 issue**列成一张表，逐条核对**当前代码里的真实状态**，
> 再按"值得做 / 成本"排序。纯只读核查，每条都给了可复现的检查命令。
>
> 结论先说：**共 22 条诉求，真正卡着别人的只有 1 条（Issue #1 的问题②），
> 但有 6 条是低成本的真缺口**。

---

## 0. 一览

| 来源 | 条数 | 状态 |
|---|---:|---|
| ① 我们 repo 的 GitHub issue | 2 | #1 **OPEN**（有 1 个问题没答）、#2 CLOSED（经验帖，内含 7 类诉求） |
| ② 既有待办清单（`a2/logs/135`） | 6 | 3 条仍未修 |
| ③ 上游 issue 草稿（**未提交**） | 2 | 阻塞级 1 条、代码卫生 1 条 |
| ④ RFC #16375 实现 issue 追踪（**未发布**） | 3 | 已写好草稿，等发布决策 |
| ⑤ 我们自己的精度总账 🔴 | 4 | 见 [`CED-PD-ACCURACY-MEASURES-20260926.md`](CED-PD-ACCURACY-MEASURES-20260926.md) |

**优先级**（按"卡住别人 / 成本低"排）：

| P | 项 | 成本 |
|---|---|---:|
| **P0** | Issue #1 问题② —— A2 8×64GB 量化每卡 HBM 需求 | 中（要读代码给判断） |
| **P0** | `no_proxy` 强制 —— 会把"起服失败"的判断全带偏 | **3 行** |
| **P1** | `say()` 早于 `mkdir` —— driver.log 缺前 100 行 | **1 行** |
| **P1** | `enable_draft_graph.sh` 调用带 `\|\| true` —— 失败被吞 | 1 行 |
| **P1** | 绕过 `serve_a2.sh` 的**前置动作清单** + guard 下沉 | 文档+小改 |
| **P1** | `bench_concurrency.py` 用 `data[0]` 覆盖 `--model` | 5 行 |
| **P1** | `attach_test.sh` KV 门槛写死 3Mi（默认配置必报假红） | 3 行 |
| **P2** | README/AGENTS 补：镜像版本差异、`PREFIX`/`MAX_SEQS` 取舍、`BAT_TOKENS` 上界 | 文档 |
| **P2** | `serve_a2.sh` 文件头与代码**自相矛盾**（说默认开，实为 `:-0`） | **1 行** |
| **P3** | 上游 issue 草稿发布（`cpu_binding` 挂死是阻塞级） | 需授权 |

---

## 1. GitHub issue #1（**OPEN**）—— 单机 8×910B4 可行性

**提问者**：`bai1535`｜**提出于** 2026-09-21｜**我们有 1 条回复**

| 他的问题 | 我们的状态 |
|---|---|
| ① 能否提供量化成品（`v41-w4a8-…-mtpq`，273 GB）？ | ✅ **已答**：回了 ModelScope 链接 `chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` |
| ② **A2（8×64 GB）做量化，每卡显存够吗？** | ❌ **未答** —— 见下 |

### 问题② 的实质

`quant/REPRO_W4A8_QUANT.md` 的硬件表**只记录了 A3 一台、DP16**
（16 逻辑卡 / 8 物理 die / 1 TB HBM），而提问者只有 **8×64 GB**。
他明确说"如果作者手上没有 A2 实测数据，**能给一个从代码/配置看需要多少的判断也很有帮助**"。

他具体想知道三件事：

1. L1 主干（40 层、MTP off、Engram off、Vision drop）**单卡峰值显存**？
2. **DP8（`DEVICE_IDS="0 1 2 3 4 5 6 7"`）在 64 GB/卡上够不够**？A3→A2 是否存在显存门槛？
3. L2 `engram_int8`（产物 221.2 GB）那一步的单卡需求？

**为什么这条是 P0**：他**还没下那 476 GB 权重**，在等这个答案决定要不要跑。
答它不需要 A2 硬件 —— 读量化链的配置就能给"每卡峰值 = 权重分片 + 激活 + 优化器态"的量级判断。
**这是唯一一条"别人在等我们"的**。

---

## 2. GitHub issue #2（CLOSED）—— 8×910B4 踩坑帖

**同一提问者**，内容质量很高。虽然 issue 已关，但**里面 7 类诉求大部分没落地**。
逐条核对如下（判据是当前代码，不是当时的回复）。

### 2.1 ★ 诉求 ①：`DRAFT_GRAPH=1` 有一类**不体现在 A 值上**的静默失效

**他踩的坑**：自己写包装脚本 `exec serve_v2.sh`，绕过了 `serve_a2.sh` 里
"把 draft 三个文件拷进 live tree"那一步 ⇒ `DSPARK_GRAPH_CAPTURE_METADATA=1`
设了但没人消费 ⇒ **A 看着完全正常（2.46），单流只有 43 tok/s**。

**他实测的对照**（A2/910B4，18 token prompt / 128 输出）：

| 臂 | ms/step | A | tok/s |
|---|---:|---:|---:|
| `DRAFT_GRAPH=0` | 56.8 | ~2.3 | 39.9 |
| `DRAFT_GRAPH=1`（**文件未装入**） | 57.3 | 2.46 | **43.0** |
| `DRAFT_GRAPH=1`（**文件已装入**） | **32.9** | 2.59 | **78.8** |

⇒ **A 正常也可能没生效；只有 `ms/step` 会暴露，但方向是"看起来没变化"，不是变坏。**

**他提的两条建议 + 我们的现状**：

| 建议 | 现状 |
|---|---|
| 把"装 draft 文件"做成**独立、幂等的入口** | 🟡 **半成品**：`tools/enable_draft_graph.sh on/off/status` 存在，但 `scripts/serve_a2.sh:1304` 调它时带 **`\|\| true`** ⇒ **失败被吞掉**，然后才走"兜底"分支；兜底失败才 `die` |
| README/EXPECTED_PERF 列**前置动作清单** | ❌ **没有**（`rg '前置动作'` 全仓 0 命中） |

**核查命令**：

```bash
# 幂等入口存在吗
ls -la tools/enable_draft_graph.sh
# 调用点（注意行尾的 || true）
sed -n '1300,1310p' scripts/serve_a2.sh
# 有没有前置动作清单
rg -l '前置动作|绕过 serve_a2' --glob '*.md' .     # 0 命中
```

### 2.2 ★ 诉求 ②：代理劫持 `127.0.0.1` ⇒ "服务明明好了却判不就绪"

**他的环境**：`http_proxy` 指向 Squid，Squid **拦截 `127.0.0.1` 并返回 503**。
表现是模型已就绪、直连 `/v1/models` 正常，但走代理的 `curl /health` **永远 503**
⇒ 所有就绪判定超时 ⇒ 看起来像"起服挂死"。

**他的建议**：在 `serve_a2.sh` / `serve_v2.sh` 开头**强制**加
`export no_proxy=${no_proxy:-127.0.0.1,localhost}`，并写进常见问题。

**现状**：❌ **完全没做** —— 三个脚本都没有：

```bash
grep -n 'no_proxy\|NO_PROXY' scripts/serve_a2.sh scripts/serve_v2.sh scripts/serve_a3.sh
# 0 命中
```

**为什么这条是 P0**：它**不挑环境**——只要用户的机器有企业代理，我们的
就绪轮询、`/metrics` 抓取、探针全都会撞上；而且症状是"看起来起服失败"，
会把后面所有判断带偏。修它只要 3 行。

### 2.3 诉求 ③：6 个小坑

| # | 他的现象 | 现状 | 核查 |
|---|---|---|---|
| 3.1 | 日志里 `tee: .../driver.log: No such file or directory` | ❌ **未修（已实测复现）** | `say()` 在 `:283` 就 `tee -a "$OUT/driver.log"`，而 `mkdir -p "$OUT"` 在 **`:664`**；`:591/:594/:596` 是**正常路径**的调用（打印模型挂载清单）。<br>**实测**（复刻 `say()` 定义 + 调用时序）：<br>`tee: /tmp/.../driver.log: No such file or directory`<br>且**第一条 say 的内容没进 driver.log**，只有 mkdir 之后那条进去了 ⇒ **日志缺前 ~100 行**。<br>`serve_a3.sh` 结尾是 `exec bash serve_a2.sh`，**自身不建 `OUT`**（`grep -cE 'RUN_ID=|OUT=|mkdir -p'` = 0）⇒ **官方路径每次起服必踩**。★ 我们自己的 `experiments/dspark/launch_d_dspark.sh` 里有 `mkdir -p`，**掩盖了它** |
| 3.2 | `BAT_TOKENS=16384` 直接 NPU OOM（`Tried to allocate 1.24 GiB … 1.06 GiB free`） | 🟡 机制有记录（`a2/logs/135`），但 **README 没写"8192 是激活内存硬上界"** | `rg 'BAT_TOKENS=16384' README.md` → 0 |
| 3.3 | `tools/draft_graph_guard.sh` 退出码 2「拿不到基准结果」，实际是缺 python 包；**建议打印实际解释器路径** | ❌ **未修** | `grep -n 'python3' tools/draft_graph_guard.sh` → 裸调，无路径打印 |
| 3.4 | 分片文件名没有零填充（`shard2` 排在 `shard10` 前） | ❓ **待澄清**：我们的脚本看起来**是填充的**（`mtp-0000N-of-0000M`、`vision-00001-of-00001`）⇒ 需确认他指的是哪一份产物 | `rg 'shard-name' quant/scripts/*.py` |
| 3.5 | README 给的镜像 tag 是 `-a3` 版；A2 硬件（910B1/B4）要用 A2 版，否则 `SOC_VERSION` 不匹配 | ❌ **未做**：README 主文完全不提镜像版本差异；只在 `CHANGELOG.md` / `REPRO.md` 里零散提到 | `rg 'flash-a2\|flash-a3' README.md` → 0 |
| 3.6 | `attach_test.sh` 的 KV 门槛**写死 3 Mi**，而默认配置 ~2.82 M ⇒ **每次对默认配置报假红** | ❌ **未修**（与 `a2/logs/135` 的待办 3 同一条） | `grep -n '3145728' tools/attach_test.sh` → `:198`/`:203` 硬编码 |

### 2.4 诉求 ④：两条参数经验（我们没有写进文档）

| 他的结论 | 现状 |
|---|---|
| **`PREFIX=0`**：真实负载是"长文档输入、短输出"（128:1），前缀命中率只有 **3.52%** ⇒ 成本全付、收益近零。关掉后 9 万 token × 8 并发 TTFT 中位 **87.4 → 74.6 s（−14.6%）** | ❌ 文档里没有"业务侧怎么取舍"的说明 |
| **`MAX_SEQS` 是"并发容量"不是性能杠杆**：32→64 在 c≤32 时 **±1%**，只在 c=64/128 各 +8.7%/+7.2%（单流却 −34%/−35.5%）；聚合封顶 **~259 tok/s**，c=128 反而回落 ⇒ **上界是算力，不是槽位数** | ❌ 没写 |

他给的 A/B 数据（1024 prompt / 128 输出，各档 128 请求，fail=0）：

| 并发 | `MAX_SEQS=32` | `MAX_SEQS=64` | Δ 聚合 | Δ 单流 |
|---:|---:|---:|---:|---:|
| 8 / 16 / 32 | 153.8 / 197.6 / 236.6 | 155.1 / 197.0 / 237.7 | ±1% | ±1~5% |
| 64 | 238.3 | **259.0** | **+8.7%** | **−34%** |
| 128 | 239.5 | 256.8 | +7.2% | **−35.5%** |

### 2.5 ★ 诉求 ⑤：**"绕过 `serve_a2.sh` 时必须自行补齐的前置动作清单"**

他把这条列为**通用教训**，原话：

> **`serve_a2.sh` 的价值不只是拼 vLLM 参数**，它还在起服前做了一批"启动期副作用"：
> 装 draft 版文件 / DRAFT-GUARD 断言 / skcache 命中检查与 GC / engram 目录 `:rw` 挂载决策 / 几十个 `-e`
> ⇒ **「vLLM 参数一致」≠「行为一致」**。

**现状**：❌ 清单**不存在**，而且问题比他描述的更严重 ——

```bash
grep -c 'DSPARK_GRAPH_CAPTURE_METADATA\|DRAFT-GUARD' scripts/serve_v2.sh
# 0
```

⇒ **`serve_v2.sh` 里一处 guard 都没有**。他做容器化/编排时 `exec serve_v2.sh`
就**天然绕过全部防线**，而且**没有任何告警**（他踩的那个静默失效正是这个后果）。

### 2.6 诉求 ⑥：性能数据存档

他给了完整的 A2/910B4 数据（与上一代模型同机同脚本同入口）：

| 并发 | 上一代 | **V4.1（本框架）** | Δ |
|---:|---:|---:|---:|
| 1 | 22.2 | **66.2** | **+198%** |
| 4 | 76.3 | **158.4** | **+108%** |
| 8 | 135.8 | **205.2** | **+51%** |
| 16 | 288.6 | 240.5 | −17% |
| 32 | 558.1 | 280.3 | −50% |
| 64 | 751.8 | 270.1 | −64% |
| 128 | 372.0 | 259.0 | −30% |

长文档（9 万 token）吞吐 **17.0 vs 17.0 持平**，TTFT 中位 68.3→72.3 s（+5.8%）。

**现值**：这是**唯一一份第三方独立复现的 A2 性能数据**（而且是 910B4，不是我们的 910B3），
目前只存在于 issue 里。建议归档进 `reports/` 或 `EXPECTED_PERF.md`。

⚠️ 但要标注口径：他的"上一代"是 **DeepSeek-V4-Flash-0731-w8a8**，与本框架**不是同一份模型**，
所以这组数字说明的是**"换模型"的收益**，不是"我们的优化"的收益。

### 2.7 诉求 ⑦：两个工具 bug

| # | 他的现象 | 现状 |
|---|---|---|
| 7.1 | `v41_perf_baseline.py` 传未知 mode（`prod-lite`）**静默产出空结果**（`exit 0`, `results={}`）；有效 mode 只有 `quick`/`prod`/`full` ⇒ 建议未知 mode 直接报错 | ❓ **该文件在本仓全历史都不存在**（`git log --all -- '*perf_baseline*'` 空、全工作区 find 空）⇒ **需向报告者澄清**，可能他指的是别的工具或别的仓 |
| 7.2 | `tools/bench_concurrency.py` 用 `models["data"][0]["id"]` **覆盖** `--model`；走聚合网关时 `data[0]` 可能是别的模型（他们那边是 bge-embedding）⇒ **压测目标被悄悄改成 embedding、结果全废** | ❌ **未修**：`:475` 仍是 `served = models["data"][0]["id"]` |

---

## 3. 既有待办清单（`a2/logs/135-20260923-handover-context.md`）

那份交接里已经列了 6 条待办，与本 issue 有 2 条重合：

| # | 项 | 现状 |
|---|---|---|
| 1 | A2 精度问题（主任务） | 已转入 CED-PD 线，见 [`CED-PD-ACCURACY-MEASURES-20260926.md`](CED-PD-ACCURACY-MEASURES-20260926.md) |
| 2 | `BAT=16384 @ 520K` 是否有效 | **未做** |
| 3 | `attach_test.sh` KV 门槛 3Mi vs 默认 2.82M | **未修**（＝本 issue 3.6） |
| 4 | `serve_a2.sh` **文件头**第 11 行写"默认开 `DRAFT_GRAPH=1`"，与代码 `:-0` **自相矛盾** | **未修** —— 核对：文件头 `:11` 确实写"默认开"，而 `:236` 是 `DRAFT_GRAPH=${DRAFT_GRAPH:-0}` |
| 5 | A3 上 8192 服务仍在跑 | 已变化（现在跑 CED-PD） |
| 6 | 520K@8192 的 A2 正控未跑 | **未做** |

**第 4 条值得单说**：文件头说"默认开"、代码是"默认关"，而**默认关正是刻意决定的**
（`CHANGELOG §6` 记录过：默认开会让用户拿到 `A≈1.07` 的坏配置且无报错）。
⇒ 这是**纯文案 bug**，但会让判读的人以为"我什么都没设，应该是开着的"，**正好踩中那个静默失效**。

---

## 4. 上游 issue 草稿（2 条，**未提交**）

| 草稿 | 目标仓 | 严重性 | 状态 |
|---|---|---|---|
| [`issue-draft-cc1d-dead-import`](../upstream/pr/issue-draft-cc1d-dead-import.md) | `vllm-project/vllm-ascend` | 代码卫生（`patch_triton.py` 仍 import 已被 #14620 删除的 `causal_conv1d_update_npu`） | 草稿，待授权 |
| [`issue-draft-cpu-binding-migratepages-hang`](../upstream/pr/issue-draft-cpu-binding-migratepages-hang.md) | 同上 | **阻塞级** —— `enable_cpu_binding` 在目标 NUMA 节点已满时 `migratepages` **内核态无限空转**、服务永不就绪、`kill -9` 无效 | 草稿，待授权 |

**第二条的价值**：我们现在所有 A3 脚本都默认 `CPU_BIND=0` 规避它，
但**上游并不知道这个坑**。而且它不止影响我们 —— 任何在多租户机器上开
`enable_cpu_binding` 的人都会撞上。

---

## 5. RFC #16375 实现 issue 追踪（3 条，**未发布**）

按 RFC 第 3 行的邀请（"Release targets and owners can be attached to individual
implementation issues as they are agreed"）写好的 3 份草稿：

| track | 对应 RFC 条目 | 主题 |
|---|---|---|
| [A](../upstream/pr/issue-track-A.md) | [46][47][48][49][50][77] | Engram host-resident 表 + 设备侧查表 + 图捕获 |
| [B](../upstream/pr/issue-track-B.md) | [63][65] | MoE AllGather dispatch（TP=EP）+ range-compare expert mask |
| [C](../upstream/pr/issue-track-C.md) | [73][75][77] | DSpark ACLGraph + 显式 eager fallback + host-sync 边界 |

⚠️ **track C 已经改过一版（v2 re-aim）**：不再主张"把 DSpark 图捕获落进 main"。
发布前要确认用的是 v2 版本文案。

**这三条的定位**：不是 bug 报告，是**认领实现**。发布门槛是"我们愿意跟着走完 review"。

---

## 6. 我们自己的 🔴 未验证项（来自精度总账）

| 项 | 为什么算需求 |
|---|---|
| **A2 生产的 `MULTISTREAM=1 DSA_OVERLAP=1`** | 我们在 A3 单变量定位到的乱码成因正是这组开关，A2 生产**正在用**，但**没在 A2 上隔离过** |
| 草稿 SWA 路径是否也需 `[CED-SWA-CLIP]` | CED+DSpark 四针全过，但那是"没复现"不是"不存在" |
| `SP_TOKENS=5 vs 7` | 历史两份报告差 ~30%，自相矛盾 |
| `BAT=16384 @ 520K` | 见 §3 待办 2 |

---

## 7. 建议的行动顺序

### 第一档：**别人在等 / 成本极低**（建议立刻做）

1. **答 Issue #1 的问题②**（A2 8×64GB 量化每卡 HBM）—— 读 `quant/` 链给量级判断，**不需要 A2 硬件**
2. **`no_proxy` 强制**（3 行）—— 三个脚本开头各一行
3. **`say()` 早于 `mkdir`**（1 行，**已实测复现**）—— 把 `mkdir -p "$OUT"` 提到 `say()` 定义之前
4. **`serve_a2.sh` 文件头文案**（1 行）—— 改成"默认关"，与 `:-0` 一致

### 第二档：**真缺口，小改**

5. **`enable_draft_graph.sh` 调用去掉 `|| true`**，失败要出声
6. **`bench_concurrency.py`**：优先在 `/v1/models` 列表里找 `--model`，找不到再回退 `data[0]`
7. **`attach_test.sh` KV 门槛**：跟着 `GPU_UTIL` 算，或改成"打印实际值 + 阈值可配"
8. **`draft_graph_guard.sh` 打印解释器路径**

### 第三档：**文档 / 可发现性**

9. **「绕过 `serve_a2.sh` 的前置动作清单」**—— 这是 issue #2 的**要害**：
   建议做成 `docs/BYPASS-SERVE-A2.md`，并把 **DRAFT-GUARD 下沉到 `serve_v2.sh`**（让绕过也会被拦）
10. README 补：镜像 tag 版本差异、`PREFIX`/`MAX_SEQS` 取舍、`BAT_TOKENS` 上界
11. 归档第三方 A2/910B4 性能数据（**标注口径**：是换模型的收益，不是优化的收益）

### 第四档：**需要你授权**

12. 提交 2 条上游 issue 草稿（`cpu_binding` 挂死那条价值最高）
13. 发布 3 条 RFC #16375 实现 issue（track C 用 v2 文案）

---

## 8. 两条需要向报告者澄清的

| 项 | 问题 |
|---|---|
| 分片零填充（issue 3.4） | 我们的脚本看起来是填充的（`mtp-0000N-of-0000M`）⇒ 他指的是哪一份产物？ |
| `v41_perf_baseline.py`（issue 7.1） | **本仓全历史不存在这个文件** ⇒ 是别的仓？还是我们的工具被改名了？ |

> ⚠️ 不要去猜。这两条如果按"我们以为的"改，很可能改错对象
> —— 这正是本仓踩过多次的"判据绑错对象"。

---

## 附录：外联记录（2026-09-26）

上游 issue 草稿（`upstream/pr/issue-draft-*.md`）与 RFC #16375 实现 issue 追踪
**经用户决定，本轮不发布**（见本文 §4/§5）。

本仓两个 issue 已回复并关闭：

| # | 标题 | 动作 |
|---|---|---|
| 1 | 【咨询】单机 8×910B4（A2 口径）部署可行性 + 能否提供量化成品 | 回复 + **close（completed）** |
| 2 | 经验贴：8×910B4 上跑 DeepSeek-V4.1-Flash（W4A8）的踩坑记录与两处框架改进建议 | 回复（issue 先前已 close） |

### 关闭 #1 的依据

提问者（`bai1535`）**已在 #2 里说明他们在 8×910B4 上部署完成并托管上线** ⇒
问题①（权重）**已被实践回答**；问题②（A2 量化每卡 HBM）在回复里说明了
"我们有结构性事实、但**没有 A2 实测**"，并把缺口记为【未确认】——
**没有给拍脑袋的数字**。

### 回复 #2 的内容（逐条对应他的 7 类诉求）

| 他的项 | 我们回复的状态 |
|---|---|
| ① `DRAFT_GRAPH=1` 静默失效 | **只做了一半**：去掉了 `\|\| true`；但 **`serve_v2.sh` 里 0 处 guard**，`exec serve_v2.sh` 仍绕过全部防线 —— 明确告诉他这是剩余里最重要的 |
| ② 代理劫持 `127.0.0.1` | ✅ 已修（三个起服脚本，含"不覆盖用户已设值"的负控） |
| 3.1 `tee` 丢日志 | ✅ 已修（**先复现再修**，行号标注为修复前） |
| 3.2 `BAT_TOKENS=16384` OOM | 🟡 机制有记录、README 待补（**接受**） |
| 3.3 `draft_graph_guard` 退出码 2 | ✅ 已修（解释器路径 + 依赖自检） |
| 3.4 分片零填充 | ❓ **请他澄清**——我们的命名是零填充的（`mtp-%05d-of-%05d`） |
| 3.5 README 镜像 tag 版本差异 | ❌ 未做（**接受**） |
| 3.6 KV 门槛 3Mi 假红 | ✅ 已修，并指出 `run_test.sh` 早就改对了、只有另两处漏了 |
| 7.1 `v41_perf_baseline.py` | ❓ **请他澄清**——本仓全历史不存在该文件 |
| 7.2 `bench_concurrency` 覆盖 `--model` | ✅ 已修（并发现底下还有 `--model` 非空默认值这个土壤） |
| 他的 `PREFIX` / `MAX_SEQS` 经验 | ❌ 我们没写进文档，**接受** |
| 他的 910B4 性能数据 | 表示想归档，但会**标注口径**：那是"换模型"的收益，不是本框架优化的收益 |

并主动告知我们**自查出**的同族病：`curl ... || echo 000` 会拼成 `000000`（全仓 5 处，
其中一处是 `serve_a2.sh` 的就绪轮询），已修 + 加守卫。
