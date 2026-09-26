# A2-ENGRAM-PATHS —— ★ 硬约束「必须开 ENGRAM」下的三条路（草稿 · 待主代理定稿）

> 2026-09-22 21:0x。执行：子代理 `a2_engram_capacity_plan`。
> **性质**：**纯证据核算 + 文档**（不占卡、不起容器、不写生产脚本、不碰 A2/A3、不动 links-server 的 `snippet.txt`）。
> **背景（用户的硬约束，原话）**：
> > 「**注意我必须要开 ENGRAM**，否则模型其实能力下降严重并不可接受。」
> ⇒ **上一版「三条路」里 `ENGRAM=0` 的两条作废**；本文三条**全部 `ENGRAM=1`**。
> ⇒ 「**关 Engram 换稳定**」**已不在选项内**（它在 `A2-DEPLOY-NOW.md` 里是"出路口 3"，现在**用户已否决**）。
>
> **标记**：【实测】/【推断】/【未确认】三选一，逐项标。数字都带出处（`文件:行` 或 日志编号）。
> **路径约定**：本文用**发布仓根**的相对路径（`a2/logs/…`、`reports/…`）。
> 在开发工作区里 `reports/` 位于 `dsv41-release/` 下（例：`dsv41-release/reports/a2-draft-graph-20260920.md`）。
> **加粗 = 可直接抄**，★ 越多越关键。

---

## 0. 一句话

**三条路全部开着 Engram，差别只有两个变量**：①**开不开卸载**、②**开不开 int8（档 C）**。

| 路 | 配置 | 担保级别 | 一句话 |
|---|---|---|---|
| **路 1** | `ENGRAM=1` + `DRAFT_GRAPH=1` + 档 B + **不开卸载** | ★★★ **保证** | **= A2 现在的生产**（**零增量、纯保底**）—— 什么都不用做 |
| **路 2** | 路 1 + **卸载**（`OFFLOAD_GB=85`） | ★★☆ **保证 · 有前置条件** | **必须先有 A3 的 `ENGRAM-PAGELESS` 修复臂 replay failed=0**；没有这条前置 ⇒ **= `logs/073` 那条必死的臂** |
| **路 3** | 路 2 + **档 C**（`KV8_SWA=1 KV8_RING_FP16=1`） | ★☆☆ **可能会挂** | **多两条从未同开过的轴**（`ENGRAM × int8`、`int8 × draft 入图`）⇒ 只用于取证，**不作生产** |

★ **内存账的结论（详见 §1）**：A2 上 `ENGRAM=1` 的可用宿主 **≈439 GiB**（**已经扣掉了 Engram**），
L1 生效时池的宿主 = **3.49 × 记账** ⇒ **`OFFLOAD_GB=85` ⇒ 296.7 GiB、剩 142 GiB**（**与原计划逐字相同**）
⇒ **"必须开 Engram"这件事本来就不会把 1M 计划挤掉** —— 只要**别把 Engram 扣两遍**（§1.6 第 1 坑）。

---

## 1. 内存账（**核心**，逐项带出处）

### 1.0 ★★★ 只有一个基线数要记：**MemAvailable ≈ 439 GiB**

| 项 | 值 | 出处 | 标记 |
|---|---|---|---|
| A2 宿主总额 | **754 GiB** | 用户探针回执 `total 754 used 314 free 348 buff/cache 95 available 439` | 【实测·A2】 |
| ★★ **基线：MemAvailable（生产在跑、Engram 已加载）** | **≈439 GiB**（四次读数 438.4 / 439.0 / 438.4 / 439.0） | `a2/logs/065` §3c + 用户 16:32 / 18:50 / 18:54 / 18:59 四次探针回执 | 【实测·A2】 |
| A2 生产是否开着 Engram | **是**（`serve_a2.sh:127` 的 `ENGRAM=${ENGRAM:-1}`，用户的 `run_test.sh` 不传 ⇒ 取默认 1） | `A2-DEPLOY-NOW.md` §「ENGRAM 的默认值」；`reports/a2-draft-graph-20260920.md:77`（`Engram local-owner validate 通过 → 切 fast`） | 【实测·A2】 |

★ **为什么 439 GiB 是唯一基线**：那四次读数**都是在正在服务的生产容器里**（`docker exec`）量的
（`A3-VALIDATION-ROADMAP.md` §4.2 原文：「用户 16:10 / 16:32 那两次探测**就是 `docker exec`
进正在服务的容器跑的**（服务没停）」），而生产 **`ENGRAM=1`**。
⇒ **Engram 的开销已经落在这个 439 GiB 之外，不要再减一次。**

★ **窗口自检（1 行，防"Engram 其实没加载"）**：

```bash
awk '/MemAvailable/{printf "MemAvailable=%.1f GiB\n", $2/1048576}' /proc/meminfo
#   期望 ≈ 439 GiB（±5%）⇒ 基线成立，直接按 §1.4 的表取 OFFLOAD_GB
#   若 ≳ 600 GiB ⇒ ★ Engram 没在跑（关着或没装）⇒ 先查清楚，再按 (439 − 206) 重算池预算
```

### 1.1 Engram 表的宿主占用 —— **206 GiB / 节点 = 8 × 25.75 GiB**

| 口径 | 值 | 出处 | 标记 |
|---|---|---|---|
| ★ **A2 交付验证臂实测（host 路径）** | **221.19 GB = 206.00 GiB / 节点**，per-rank **25.75 GiB** | `engram_ref/wtgraph/docs/ENGRAM_DRAM_PLACEMENT.md:104`（原文 `合计 221.191 GB = 206.00 GiB；per-rank（/8）= 27.65 GB = 25.75 GiB`） | 【实测】 |
| 独立佐证（RSS / MemAvailable） | `worker RSS 31.5→237.5 GiB（+206）`、`MemAvailable −204 GiB` | `engram_ref/wtgraph/docs/NIGHT_REPORT.md:158` | 【实测】 |
| 第二条独立佐证（匿名 vs 文件映射） | `8 worker 匿名内存合计 229.91 GiB、文件映射 0`、`起服前后 MemAvailable −271 GiB` | `delivery_vb_staging/README.md:20`（G2 行）、`EXPECTED.md §G2` | 【实测】 |
| device-index 路径下**同一张表的另一种口径** | **196,613,849,600 B = 183.11 GiB**（`384006168×256 + 384016682×256`） | `a2/logs/068` §3（由 `DEVICE-INDEX` 日志行算出） | 【实测·算式】 |

### 1.2 ★★★ `logs/068` 那一格【未确认】的回答：**每 rank 一份分片**

`logs/068` §3.1 留的【未确认】是：**Engram 的 183.11 GiB 是 rank0 一份，还是 8 rank 各一份？**
（它决定"缩池子到底有没有用"。）**本文用两条独立证据把它答掉**：

```
① 【实测·同源】Engram 权重是 NodeShardedEngram：int8 表 [rows/8, 256]、scale [rows/8, 8]
   ⇒ 每个 rank 只装 1/8 行 ⇒ 25.75 GiB/rank ⇒ 节点合计 206 GiB
   出处：engram_ref/wtgraph/docs/ENGRAM_WORKSPACE_AUDIT.md:101（W4 行）、:153
② 【实测·独立】8 worker 的实际 RSS 增量 = +206 GiB（31.5 → 237.5）、匿名内存 229.91 GiB、文件映射 0
   出处：NIGHT_REPORT.md:158、delivery_vb_staging/README.md:20
```

★ **另一种情形（"8 rank 各一份整表"）在物理上被排除**：

```
若每 rank 一份 183.11 GiB  ⇒ 8 × 183.11 = 1464.9 GiB  >  机器总额 754 GiB   ⛔
若每 rank 一份 206 GiB     ⇒ 8 × 206    = 1648   GiB  >  机器总额 754 GiB   ⛔
⇒ 【推断·强】"每 rank 一份整表"不成立；068 的【未确认】在 **host 路径**下已闭合。
```

⚠️ **仍留一格【未确认】（对 A2 不构成影响）**：`ENGRAM_DEVICE_INDEX=1/auto` 的 **device-index 路径**
每 rank `mmap` 的是**整表文件**（`MAP_SHARED` ⇒ 物理页共享一份，但**驱动注册预算怎么记这 8 次**不明，
`logs/068` §3.1 给的边界 `257.79 ≤ L < 380.27 GiB` 不足以分辨）。
⇒ **A2 用的是 `ENGRAM_DEVICE_INDEX=0`（硬约束），这条路根本不走** ⇒ **这一格不阻塞上线**。

### 1.3 卸载池的宿主乘数（**三条，各自实测**）

| 配置 | 乘数 | 出处 | 标记 |
|---|---|---|---|
| **L1 生效**（`P2_POOL_PATCH=1`） | ★ **3.49 × 记账**（`197.21 / 56.5`） | `a2/logs/042` §3.2（臂 B，8 卡真权重：`26,469,138,432 B/rank × 8 = 211,753,107,456 B = 197.21 GiB`） | 【实测·8卡】 |
| **L1 失效** | ★ **6.94 × 记账**（`392.35 / 56.5`） | `a2/logs/042` §3.2（臂 A，`421,287,952,384 B = 392.35 GiB`）+ `a2/logs/074` | 【实测·8卡】 |
| **档 C**（int8，L1 生效） | **2.66 × 记账**（`150.01 / 56.5`） | `a2/logs/048` §3（`197.21 → 150.01 GiB`，×1.3146） | 【实测·8卡】 |

★ **1 unit = 1 MiB**（`a2/logs/013` §1.2 原文「即 1 GiB 池 = 1024 个条目」；`logs/042` §6 的
`OFFLOAD_GB ↔ unit` 换算一致）⇒ `OFFLOAD_GB=85` = **87,040 unit**。

### 1.4 ★★★ `ENGRAM=1` 下 `OFFLOAD_GB` 的可行上限

**池预算 = 439 GiB − 运营余量**；**宿主实占 = 乘数 × `OFFLOAD_GB`**。

| 场景 | 记账上限（留 40 GiB 余量） | 记账上限（按探针自己的 300 GiB floor ⇒ 只许用 139 GiB） | 结论 |
|---|---:|---:|---|
| **L1 生效（3.49×）** | `(439−40)/3.49` = **114 GiB** | `139/3.49` = **39.8 GiB** | ★ **推荐 85**（1M × 3 会话，见 §1.5） |
| **L1 失效（6.94×）** | `399/6.94` = **57 GiB** | `139/6.94` = **20 GiB** | ⛔ **1M 场景不可用**（1 个会话 = 24,064 unit ⇒ 宿主 **≈163 GiB**；而探针口径只许用 139 GiB ⇒ **连 1 个都装不下**） |
| 档 C（2.66×，仅路 3） | `399/2.66` = **150 GiB** | `139/2.66` = **52 GiB** | 比档 B 省，但**路 3 的风险与内存无关** |

**★ 三条独立的上界证据（必须放在一起看）**：

```
① 宿主物理：MemAvailable 439 GiB（Engram 已计入）
② 驱动注册能力【实测·A2 S3】：8 进程 × 49 GiB = 392 GiB 注册 8/8 全过、往返逐字节 8/8
   （a2/logs/069 §6.2 / 用户 18:59 探针回执）⇒ ★ 池宿主 ≤ 392 GiB 时"注册得动"这一关已经过
③ 运营余量：探针自己用 300 GiB floor（不把机器吃干）；而 OFFLOAD_GB=85 会让余量落到 ~142 GiB
   ⇒ ★ 这是**用户的运营决策**，不是技术上限（技术上是 ② 那 392 GiB）
```

**⇒ 建议值：`OFFLOAD_GB=85`（= 296.7 GiB 宿主，剩 ~142 GiB）**，与 `A3-VALIDATION-ROADMAP.md` §Phase 0
的既有取值**逐字相同** ⇒ **开 Engram 不需要改这个数**（前提是 L1 生效，见 §1.6 第 2 坑）。

### 1.5 ★★ 1M 上下文 × N 会话：池子最多给几个会话

**既有口径**（`a2/DELIVERY.md` §6.5.2 ② + `a2/logs/069` §6.3）：

```
1 个 1M 会话 = 24,064 unit = 24,064 MiB = 23.5 GiB 记账
              × 3,660,003 B/unit（8 rank 合计，L1 之后）= ★ 82.0 GiB 宿主
配池规则：再乘 1.2× 余量（a2/logs/013 §3.3：1.000× 全中、0.977× 断崖归零）
```

| `OFFLOAD_GB` | 覆盖的 1M 会话数 | 宿主实占（L1 生效） | 剩 MemAvailable | 判定 |
|---:|---|---:|---:|---|
| 57 | **2**（含 1.2×） | 199.0 GiB | 240 GiB | ✅ 稳妥（余量最大） |
| **85** | ★ **3**（含 1.2×） | **296.7 GiB** | **142 GiB** | ✅★★ **推荐 = 原计划值** |
| 94 | 4（**无 1.2× 余量**） | 328.1 GiB | 111 GiB | ⚠️ 违反 1.2× 规则（要 4 个就先降 `MAX_SEQS`） |
| 113 | 4（含 1.2×） | 394.4 GiB | 45 GiB | ⛔ 贴住/超过 A2 实测的注册上限 392 GiB |
| 85（**L1 失效**） | 3 | **590.3 GiB** | **−151 GiB** | ⛔ **不可能**（这就是 L1 必须生效的原因） |

**★ 两个必须同时说的天花板（否则会给出错误预期）**：

```
① HBM 侧（A2 实测，ENGRAM=1 现状）：GPU KV cache size = 3,498,354 token
   （14.40 GiB 可用 KV / 4,419.8 B/token，util=0.90；a2/docs/KV-CACHE-ACCOUNTING.md:16）
   ⇒ 1M 请求 4.32 GiB ⇒ ★【最多 3 个 1M 同时驻留 HBM】（a2/DELIVERY.md §6.5.2 ①）
   ★ 所以「池子能给 3~4 个会话」≠「能同时跑 3~4 个 1M 请求」；
     池子的作用是【前缀复用】，不是 swap（a2/DELIVERY.md §6.5.3）
   ★ `427,643 / 485,610 / 777,318` 都是"4 GiB 预算探针"读数，**不是 A2 容量**（a2/docs/A2-DEPLOY-NOW.md）
② ★ 上面那张表是【最坏口径】（每个会话的前缀互不相同）。
   实测反例：042 的 D 臂 16 个请求**共享同一个 128K 前缀** ⇒ 池只要 20.83 GiB（不是 16 倍）
   ⇒ 推到 1M：N 个会话共享长前缀时，池需求 ≈ 82 GiB（与 N 无关）（a2/DELIVERY.md §6.5.4）
   ⇒ ★ 真实 agent 流量（同一长 system prompt / 同一份长文档）比这张表宽松得多
```

### 1.6 ★★ 三个"会算错"的坑（每一个都能让计划作废）

| # | 坑 | 正确做法 |
|---|---|---|
| **1** | ★★ **把 Engram 扣两遍**（从 439 里再减 206 ⇒ 只剩 233 GiB ⇒ 得出"只能 2 个会话"） | 439 是**在生产（含 Engram）容器里**量的 ⇒ **不再减**。只有窗口自检发现 MemAvailable ≳600 GiB（= Engram 没跑）时才按 `439 − 206` 重算 |
| **2** | ★★ **L1 静默失效**（`P2_POOL_PATCH` 没进容器 ⇒ 3.49× 变 6.94×） | 用**打过 `7c1edf2` 的** `serve_a2_offload.sh`（起服前 fail-closed + DRY 断言 MOUNTS）；起服后核 `grep -c 'P2_poolsizing'` **> 0**（`A2-DEPLOY-NOW.md` 的既有权口径）+ `P2_WORKER_HOST_BYTES`（每 rank 一行；档 B 应 ≈24.65 GiB/rank，`a2/logs/042` §3.2 / `a2/logs/074`） |
| **3** | **`ENGRAM_DEVICE_INDEX=auto` 变成"双份 DRAM 账"** | **必须显式 `ENGRAM_DEVICE_INDEX=0`**。依据：`a2/logs/070` §A.2（9 个输入里 8 个两模块解析不一致；`auto` 下 hbm 认为关着 ⇒ **25.75 GB/rank/层照付**，而 model 走设备路径）；`a2/logs/071` C9 同结论；`A2-DEPLOY-NOW.md` 已把它列为**窗口前必读第 2 条** |

---

## 2. 三条路（**全部 `ENGRAM=1`**）

> ★ 三条的公共参数（**逐字相同**，只有两个变量不同）：
> `MODEL=$HOME/models/out/v41-w4a8-flat`、`SHADOW_PKG=$HOME/shadow-pkg`、
> `OFFLOAD_GB=85 MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=2048`、
> `NPU_OFFLOAD_HOST_MEM=registered`、**`ENGRAM=1`**、**`ENGRAM_DEVICE_INDEX=0`**、`DRAFT_GRAPH=1`、
> `BLOCKS_PER_CHUNK='{"default":8,"swa":1}'`、`PREFIX_MATCH_UNIT=32`、
> `P2_POOL_PATCH=1 P2_COMP_JSON='[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]'`。
> ⚠️ **`MAX_SEQS=4` 必须显式写**（脚本默认 16，落在 P0-C 的 conc≥16 区间；见 `A2-DEPLOY-NOW.md` §「`DRAFT_GRAPH` 默认值」）。

### 2.1 路 1 —— **保证能跑通（今天就能上）= A2 生产现状**

| 项 | 内容 |
|---|---|
| **目的** | ★ **纯保底**：`ENGRAM=1` + `DRAFT_GRAPH=1` + 档 B + **不开卸载** ⇒ **一个字都不用改** |
| **配置** | `ENGRAM=1`、`ENGRAM_DEVICE_INDEX=0`、`DRAFT_GRAPH=1`、档 B、**无 `OFFLOAD_*`** |
| **命令** | ★ **不需要新命令** —— 就是 A2 现在跑着的那条。要**重启成同一配置**时用段 0 存下的命令：`docker inspect dsv41-a2 --format '{{.Config.Cmd}}' > ~/a2-rollback-$(date +%Y%m%d_%H%M).txt`；重启 = `docker rm -f dsv41-a2` 后跑 `cat ~/a2-rollback-*.txt \| tail -1` 打出来的那条（★ 这就是回滚命令的形态，沿用 `logs/074` / links-server 既有写法） |
| **判据** | 存下来的命令**非空**；起服后 `/health=200`；`SpecDecoding` 稳态 A 落在 **2.8–3.1**（恒 1.0 = draft 图没生效） |
| **担保级别** | ★★★ **保证**（= **现状**，风险 0） |
| **相对现状的增量** | ★ **= 0（没有增量）** —— 这条唯一的作用是：**窗口出事后回滚到这里** |
| **依据** | `reports/a2-draft-graph-20260920.md:74-78`：起服 803 s、**KV 容量 3,498,354 token**、`Engram local-owner validate 通过 → 切 fast`、**Vision 23/23 PASS**；单流 **54.7 → 88.7 tok/s（+62%）**。`reports/draft-graph-investigation-20260920.md:824`：**精度全过（GSM8K 198/200、Vision 23/23、10/10 质量判据）** |
| **诚实边界** | ⚠️ 那次 803 s 的会话**没有卸载** ⇒ 它证明的是"**这条组合**在 A2 上能跑"，**不证明加了池还能跑**（那是路 2 / 路 3） |

### 2.2 路 2 —— **保证能跑通，但有前置条件（等修复）**

| 项 | 内容 |
|---|---|
| **目的** | ★ 在**开着 Engram** 的前提下拿到 DRAM 卸载（长上下文前缀复用、1M × 3 会话） |
| **命令** | `ENGRAM=1 bash a2/scripts/serve_a2_offload.sh`（公共参数见上；脚本默认已是 `ENGRAM=1` / `ENGRAM_DEVICE_INDEX=0`，**仍建议显式写**，dry-run 里一眼可核） |
| **★ 前置条件（硬）** | ★★ **A3 上 `ENGRAM-PAGELESS` 修复臂必须 `replay failed=0` 且 `KeyError=0`**。当前状态：修复代码**已在发布仓工作区（未提交、未验证）** —— `patches/files/engram_hash.py` 与 `patches/files/engram_jit_kernel.py`（`git diff` 可见：缺页从 `raise KeyError(err)` 改成 **pad 历史（barrier 语义）+ `pageless_history_rows` 计数器 + 一次性响亮提示**；`V41_ENGRAM_PAGELESS_STRICT=1` 可恢复旧的致命行为）【未确认·未验证】 |
| **没有前置时会怎样** | ⛔ **= `logs/073` 那条必死的臂**：`ENGRAM=1` + 56 GiB 池 ⇒ 起服全绿、fill 16/16，但 **replay failed=13 / `KeyError: 2486` @ `engram_hash.py:463` / EngineDead**。三臂判决（唯一变量 = 池大小 ⇒ 是否发生取回）：`p1b`(ENGRAM=0, 有命中) = **0** / `p2e`(ENGRAM=1, 有命中) = **13** / `p2f`(ENGRAM=1, 池 1 MiB ⇒ 无命中) = **0** |
| **A2 上要过的门** | 见 §3.2 的表（起服期五道门 + ★★ **第 0 道实质门「返回的文本正确吗」** + ★ 新增「同前缀两发触发取回」门 + 功能三判据 + `BlockRemoved:CPU=0` + 3 轮同输入一致性） |
| **担保级别** | ★★☆ **保证（条件式）**：**修复臂过 ⇒ 这条就是"今天能上"的那条**；修复臂没过 ⇒ **它比路 3 还危险**（已知必死） |
| **内存** | 池宿主 **296.7 GiB**（L1 生效）⇒ 剩 **142 GiB**（§1.4） |
| **依据** | 正面：`a2/logs/069`（卸载 × draft 入图 8 卡全绿、判据逐字相同）、`a2/logs/042`（L1 8 卡 1.9895×）。反面：`a2/logs/073` + `a2/logs/072`（Engram × 取回 = 引擎死） |

### 2.3 路 3 —— **可能会挂（只取证，不当生产）**

| 项 | 内容 |
|---|---|
| **目的** | 把**第三个轴**（int8 档 C）叠上去 ⇒ 一次性回答"多轴能不能共存" |
| **命令** | `KV8_SWA=1 KV8_RING_FP16=1 ENGRAM=1 bash a2/scripts/serve_a2_offload.sh`（脚本会自动置 **`APC_ALIGN=3`** 与 **`GRAPH_SAFE=1`** 并打印两条警告 —— 这是**必需**的，见下） |
| **★ 为什么它"可能会挂"** | ① ★★ **`ENGRAM × int8` 从未同开过**（全仓 grep：8 卡臂 `agents/R_8card_int8/scripts/run_arm_r8.sh:58` 默认 `ENGRAM=0`，全部 `027/042/048/066/067` 都是 0；`ENGRAM=1` 的臂全是档 B）【实测·grep】<br>② ★ **`int8 × draft 入图` 从未同开过**（`A2-DEPLOY-NOW.md` §「两条未验证的边界」第 1 条）<br>③ 已知的 int8 图模式缺陷 **`EE1016`（捕获期）** 靠 `GRAPH_SAFE=1` 兜，而 **`GRAPH_SAFE` 只在 `DRAFT_GRAPH=0` 的 8 卡臂上验过** ⇒ `DRAFT_GRAPH=1 + int8` 是**全新组合** |
| **它自带的两个确定坑（不是"可能"，是"确定"）** | ① 不开 `GRAPH_SAFE` ⇒ 捕获期 **`EE1016`**（`a2/logs/048` §2）；不开 `APC_ALIGN=3` ⇒ **D/F 几何翻 token**（`logs/047`）。脚本会自动置，**但必须回读确认**（见段 4 的 `inner.sh` 判据）<br>② ⛔ **绝不要顺手开 `KV8_FULL=1`（档 D）**：不请求 logprobs 时会**静默给错 token**（`a2/logs/066d` / `A2-DEPLOY-NOW.md` 保留意见 3） |
| **担保级别** | ★☆☆ **可能会挂**（**三轴同开 = 0 条实测臂**）⇒ **只用于取证，拿到结果立刻回滚** |
| **内存（顺带的好消息）** | 档 C 的宿主乘数 **2.66×**（`logs/048`）⇒ 同样 `OFFLOAD_GB=85` 只吃 **≈225.7 GiB**（【推断·按 56.5→85 线性外推】，`logs/048` 只在 56.5 GiB 记账上实测过）⇒ **比路 2 省 ~71 GiB** —— ★ **但这不是选它的理由**（它的风险与内存无关） |

### 2.4 ★ 三条路共同的"不可选项"

```
⛔ 「关 Engram 换稳定」（ENGRAM=0）：★ 用户已明确否决（质量不可接受）。
   它在旧版三条路里占了两条（路 1 / 路 2）⇒ 那两条配置【作废】，不要再照抄。
⛔ 「ENGRAM_DEVICE_INDEX=auto」：会在 A2 上把 device-index 打开（A2 的注册探针会通过）
   ⇒ 走进 A3 那条崩掉的路（EH0012），且 auto 下双份 DRAM 账（§1.6 第 3 坑）。
⛔ 「KV8_FULL=1（档 D）」：静默给错 token（logs/066d）。
```

---

## 3. A2 窗口的顺序（**`ENGRAM=1` 版**，把停服时间压到最短）

> 格式沿用 `a2/docs/A3-VALIDATION-ROADMAP.md` §5。**目标 = 路 2**；路 3 只在路 2 全绿之后再单独做一次。

### 3.1 窗口外准备（**不占 A2 的停服时间**）

| # | 动作 | 判据 |
|---|---|---|
| 1 | ★★ **在 A2 的 clone 上 `git pull`**（必含 `7c1edf2` 的 P0 修复） | `git log --oneline -1` 能看到该修复；否则 §1.6 第 2 坑必踩 |
| 2 | 造 shadow：`PKG=$(pwd) DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh` | 无 `die` |
| 3 | ★★ **三条路各跑一次 `DRY=1`** | 路 2 = `档位 : B`、路 3 = `档位 : C`；两者都要看到 `ENGRAM=1`、`ENGRAM_DEVICE_INDEX=0`、`[DRY] ✓ 4 个卸载补丁都在真实 MOUNTS 里`；`MOUNTS(10)`（档 B）/ `MOUNTS(24)`（档 C） |
| 4 | **在 A3 上把修复臂跑完**（路 2 的前置） | `replay failed=0` 且 `KeyError=0`；把 `pageless_history_rows` 的读数带回（>0 = 修复真的被走到） |
| 5 | 存回滚命令 | 文件非空 |

### 3.2 窗口内（**停服**，每一步都有硬门，不过就回滚）

| 步 | 动作 | 门（不过就停） | 预计 |
|---|---|---|---|
| **0** | ★★ **量 MemAvailable** | **≈439 GiB**（≳600 ⇒ Engram 没跑，先查；≲400 ⇒ 先查别的租户/泄漏） | 1 min |
| **1** | 存回滚命令 `docker inspect … > ~/a2-rollback-*.txt` | 非空（★ 这是回滚的命根子） | 1 min |
| **2** | `docker rm -f dsv41-a2` + 起新配置（路 2） | — | — |
| **3** | ★ **起服期五道门**（全在日志里，**不必等压测**） | ① `grep -c EH0012` = **0**（★ 它出现在 KV cache 建立**之前**，出现即必崩）② `grep -c 'hdc disconnect'` = **0** ③ `grep -c 'DEVICE-INDEX'` = **0**（证明 device-index 真关着）④ ★ **Engram 真的加载了**：`grep -E 'Engram\|engram'` 能看到 **2 层** + `Engram local-owner validate` ⑤ `grep -c 'P1_pinned.*ret=0'` = **128**（16 张量 × 8 rank）且 `grep -c 'aclrtHostRegister failed'` = 0 | ★ **Engram 让起服多 ~10+ min**（206 GiB 表）；`DRAFT_GRAPH=1` 会话实测 **803 s（含 static kernel 冷编译；缓存命中后回到分钟级）** |
| **4** | ★ **档位门 + 容量门** | `档位` 自报 = B（路 3 应为 C）；`GPU KV cache size` ≥ **1,000,000**（★ `427,643 / 485,610` 是 4 GiB 预算探针读数，**不是 A2 期望值**；A2 现状是 **3,498,354**） | 起服期 |
| **5** | ★★★ **第 0 道实质门：返回的文本正确吗**（五道自然语言固定题：`17×23⇒391`、天空蓝⇒瑞利/散射、《红楼梦》⇒曹雪芹、`40×60%÷2⇒12`、反转 `Hello, world!`） | **五条全对**；任一错 ⇒ 停下，先排取回路径 | 2 min |
| **6** | ★★ **新增门：同前缀两发（触发一次取回）** —— 同一段 4–8K 文本连发两次 | 第二次**不 500**、`hits` 开始 >0、**无 `KeyError`**；若出现 `KeyError: 2486` ⇒ ★ 取证目的达成，**立刻回滚** | 2 min |
| **7** | **功能三判据**（`BlockStored:CPU>0`、`CPU→GPU>0`、`hits>0`）+ ★ **`BlockRemoved{medium="CPU"} == 0`** | 全中 | 5 min |
| **8** | ★ **replay ÷ fill**（TTFT 之比） | ≫（8 卡基线 12.3–12.9×）；**只快 ~4× ⇒ H2D 实际只有 ~5 GB/s**（池越大越慢：A2 实测 32→392 GiB 时 20.5→5.0 GB/s）；~1× ⇒ 停 | 5 min |
| **9** | **同输入一致性（★ 必须 3 轮）** | "三轮任一不同"的集合为空（2 轮会漏 40%，`logs/071` §3 C1）；注意本服务 `temperature=0` 下**本就有抖动**（`logs/037`）⇒ 差异要与对照臂比，别单轮判死 | 5 min |
| **回滚** | `docker rm -f dsv41-a2` + 用第 1 步存下的命令重启 | — | 5–15 min |

**时间预算（`ENGRAM=1` 版）**：起服 **~15–25 min**（803 s 实测 + Engram 表 ~10 min 量级，两者可能重叠）
＋ 门 5–9 **~20 min** ＋ 回滚余量 **5–15 min** ⇒ **停服窗口按 30–45 min 备**
（旧版 `ENGRAM=0` 的 5–15 min 估算**不适用于**这条 —— Engram 表加载与 static kernel 冷编译是主要增量）。
⚠️ 起服耗时里 **Engram 那部分是【推断】**（`A2-DEPLOY-NOW.md` §「ENGRAM 默认值」原文写的是「~10+ min」，
没有 A2 上的逐段实测）；`803 s` 那次是**含冷编译、不含卸载**的【实测】。

---

## 4. 【未确认】清单（**本文一共标了 8 处**）

| # | 未确认 | 为什么不影响本文结论 | 怎么测 |
|---|---|---|---|
| **1** | A2 现网 `v41-w4a8-flat` 的 Engram 表**确切字节数**（本文用的是交付验证臂的 206 GiB；`A2-DEPLOY-NOW.md` §「A2 的模型」已记「flat 版本大小未知」） | 结论建立在 **439 GiB 这个含 Engram 的基线**上，**不需要知道表的字节数**（§1.0） | 窗口第 0 步的 MemAvailable 自检（≳600 ⇒ Engram 没跑） |
| **2** | `ENGRAM_DEVICE_INDEX=1/auto` 下 device-index 的**驱动注册账目**是否 ×8（`logs/068` §3.1 残留） | **A2 不用这条路**（硬约束 = 0） | 若将来要用：在 A3 上做"只 register 整表、不建池"的单臂，看 `ret=0` 的规模 |
| **3** | ★ **A3 的 `ENGRAM-PAGELESS` 修复臂结果**（**路 2 的全部前提**） | 本文把它写成**前置条件**，没有当成已知 | A3 8 卡：`ENGRAM=1 + 池 56 GiB` ⇒ `replay failed=0`、`KeyError=0`、`pageless_history_rows>0` |
| **4** | ★ **pad 历史（barrier 语义）对输出质量的影响幅度** | 修复把它限定在"每个取回边界最多 `1+(lookback-1)` 个位置"，但有界 ≠ 无损 | 同一 prompt「有池 vs 无池」的输出对比（3 轮"任一不同"），并与路 1 对照 |
| **5** | A2 上「**Engram=1 + 卸载 + 取回**」这一格本身 | 这正是路 2 / 路 3 要测的东西 | = 窗口第 6 步那道门 |
| **6** | 档 C 下 `OFFLOAD_GB=85` 的**宿主实占**（2.66× 是按 56.5→85 线性外推，`logs/048` 只测过 56.5 GiB） | 只影响路 3 的余量估算，不影响选路 | 路 3 起服后读 `P2_WORKER_HOST_BYTES`（×8 rank） |
| **7** | `24,064 unit / 1M 会话` **在 1M 上是否仍线性**（只在 32K/128K 反解过；`a2/DELIVERY.md` §6.5.5 第 1 格） | 表里所有会话数都基于这个**全仓通用口径**，不是本文新引入 | A3 1M 几何臂（`A3-VALIDATION-ROADMAP.md` §Phase 2） |
| **8** | 窗口里 **Engram 表加载的实际耗时**（本文按 ~10 min 量级估） | 只影响停服窗口估算，把它写成 30–45 min 就够保守 | 窗口里量 `Loading model weights took` 与起服总时长 |

---

## 5. 出处索引（本文引用的每一份）

```
a2/logs/013  池块数 / 1.2x 余量的由来（1 GiB = 1024 条目；1.000x 全中、0.977x 断崖归零）
a2/logs/042  ★ L1 的 8 卡实测：197.21 / 392.35 GiB（3.49x / 6.94x）
a2/logs/048  ★ 档 C 的 8 卡实测：宿主 150.01 GiB（2.66x）、容量 x1.0000、EE1016
a2/logs/065  A2 探针（host_mem_pool=0、1/8/32/64 GiB 注册全过、H2D 21 GB/s）
a2/logs/066d 档 D 静默给错 token（不要开 KV8_FULL）
a2/logs/068  Engram x 池的源码级定位（183.11 GiB 算式、【未确认】两条）
a2/logs/069  ★ 卸载 x draft 入图 8 卡通过 + A2 三期并发注册 S1/S2/S3 = 392 GiB 全过
a2/logs/070  device-index 的静默算错盘点（auto 双份 DRAM 账、=1 的早退）
a2/logs/071  四家合并盘点（C9 / C1 判据强度 / A1 取回路径未验证）
a2/logs/072  p2e 的文本正确性卷（8/8 rank 同栈、泄漏链因果）
a2/logs/073  ⛔ ENGRAM=1 + 卸载 ⇒ replay 引擎死（KeyError 2486；三臂判决）
a2/logs/074  ★ 交付脚本的 P0（档 B 静默无卸载）+ 修后三条 dry-run
a2/docs/A2-DEPLOY-NOW.md          （窗口前必读 1-5、ENGRAM/DEVICE_INDEX 的默认值、十道门、5 条保留意见）
a2/docs/A3-VALIDATION-ROADMAP.md  （§Phase 0 的 85 GiB 账、§4 零停机探针、§5 Phase 5）
a2/DELIVERY.md                    （§6.5.2 三个天花板 / §6.5.3 L1+L5 是前提 / §6.5.4 共享前缀 / §6.5.5 待测三格）
a2/docs/KV-CACHE-ACCOUNTING.md    （A2 14.40 GiB ⇒ 3,498,354 token、4,419.8 B/token）
a2/scripts/serve_a2_offload.sh    （ENGRAM=1 / ENGRAM_DEVICE_INDEX=0 的默认值；APC_ALIGN/GRAPH_SAFE 自动置）
reports/a2-draft-graph-20260920.md                （A2 现状：803 s / 3,498,354 / Vision 23/23 / +62%）
reports/draft-graph-investigation-20260920.md:824 （GSM8K 198/200、Vision 23/23、10/10 质量判据）
patches/files/{engram_hash.py,engram_jit_kernel.py}
                                   （★ 未提交的 ENGRAM-PAGELESS 修复：pad 历史 + pageless_history_rows）
engram_ref/wtgraph/docs/ENGRAM_DRAM_PLACEMENT.md:104（206.00 GiB / 8 = 25.75 GiB per-rank）
engram_ref/wtgraph/docs/ENGRAM_WORKSPACE_AUDIT.md:101/153（NodeShardedEngram 分片；MemAvailable -205）
engram_ref/wtgraph/docs/NIGHT_REPORT.md:158（worker RSS +206、MemAvailable -204）
delivery_vb_staging/README.md:20、EXPECTED.md §G2（匿名 229.91 GiB、文件映射 0）
```

**纪律声明**：本文**没有**占卡、**没有**起容器、**没有**改任何生产脚本或补丁、**没有**碰 links-server 的
`snippet.txt`、**没有** push / 开 PR / issue、**没有**写 `upstream-v41/`。
所有结论按【实测】/【推断】/【未确认】标注；不确定的没有编数字。
