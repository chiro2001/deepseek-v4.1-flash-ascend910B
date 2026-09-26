# 052 — A2 探测脚本 v2：**旧件会崩且会给出假结论**；新版已在 A3 双档跑通；并回答「探测要不要占 NPU」

> 2026-09-22 11:2x–11:4x CST。执行：**主代理**（A3 单槽 c1 = die 6，走 `a3_chip.sh` 锁；不占 c0）。
> 起因：用户问 *"A2 prob 是否需要占 NPU"* —— 顺手一查，发现**我们让他跑的那个脚本本身有两个硬伤**。
> 没有写 `upstream-v41/`；没有用 `/tmp`；没有起模型、没有起服务。
> 标记：【实测】= 有判别力的原始数据；【推断】；【未确认】。

---

## 0. 一句话

**要占，但只占"能看见设备"这一件事** —— 需要 `aclInit` + `aclrtSetDevice` + 一条 stream，
**不加载模型、不跑算子、不做图捕获、不长期持有显存**（显存峰值 = 一个 **256 MiB** 的设备张量，
`COPY_GIB=0` 可完全关掉）⇒ **可以和正在服务的 A2 共存**。
**但**：原来那个 `a2/scripts/a2_one_shot_probe.sh`（ctypes 版）**跑不出结论** —— 它**段错误**，
而且**少了 `aclrtSetDevice`**（那会让 `aclrtMallocHost` 一律返回 `107002`，看起来像"A2 不能用 pinned"）。
**现已换成在 A3 上实测跑通的版本**（来源：`agents/P1_pinned/scripts/a2_pinned_probe.sh`）。

---

## 1. ⛔ 旧件的两个硬伤【实测】

### 1.1 缺 `aclrtSetDevice` ⇒ `107002` 假阴性（**这条会直接读错 A2**）

旧脚本 `load()` 里只 `aclInit` 就调 `aclrtMallocHost`。而同仓库 `logs/014` §2.1 已经记过这个坑：

```
[P1] raw aclInit rc=0
[P1] raw ✗ aclrtMallocHost 4096.0 MiB（第 1 次，累计 0.0 GiB） rc=107002   ← 忘了 set_device 的假象
```

⇒ 在 A2 上会打印出一串 `FAIL rc=107002`，**结论会被误读成"A2 的 pinned 路径不可用"**。

### 1.2 ctypes `libascendcl` + `torch_npu` 同进程 ⇒ **SIGSEGV**

2026-09-22 11:29 在 A3 `prbench-c1`（die 6）实跑旧脚本：

```
[probe] acl | ... | aclInit rc=0
[probe] acl | ... | aclrtSetDevice(0) rc=0
[probe] acl | ... | aclrtGetDeviceCount=1
[probe] acl | ... | torch=2.10.0+cpu npu_count=1
/work/a2probe.sh: line 277: 96842 Segmentation fault (core dumped) python3 ...
```

★ 而且**②之后一行结果都没有**：那些 `print` 没有 `flush=`，崩溃时全丢在缓冲区里
⇒ **"跑完了但什么都没打印"** 是这里最危险的形态（看起来像"没输出"而不是"崩了"）。
【推断】根因 = 裸 `ctypes.CDLL("libascendcl.so")` 与 `torch_npu` 各自加载/初始化 ACL 的冲突；
**未**最小化复现（不值得，见 §2 的替代）。

> ★ 教训（可推广）：**探针的"没输出"必须与"跑完了"区分开** —— 要么全程 `flush=True`，
> 要么在末尾打一行 `探针结束`。旧脚本其实有 `print("\n（探针结束）")`，但它在崩溃之后 ⇒ 没打出来。

---

## 2. ✅ 新版：直接采用 A3 上早已跑通的那一份

`a2/scripts/a2_one_shot_probe.sh` **已替换**为 `a2/agents/P1_pinned/scripts/a2_pinned_probe.sh` 的内容
（同一份在 2026-09-21 就产出过 `logs/raw/014-80-a2probe-all-on-a3.txt`），并加了三处：

1. **`LIGHT=1`（默认）**：峰值 host ≈ 40 GiB、显存 256 MiB —— A2 服务在跑时也能用；
   `LIGHT=0` 才是全档（≈200 GiB host）。
2. **`COPY_GIB`（默认 0.25 GiB）**：设备侧张量的大小就是本探针的显存峰值，可调到 0。
3. **改判据/判读文案**：把"★设备往返"提到判据第一位，并显式回答"占不占 NPU"。

**它为什么比旧件对**（都是【实测】过的差异）：

| 维度 | 旧件（ctypes 版） | 新版（torch + `acl` 模块） |
|---|---|---|
| 设备上下文 | **没有** `set_device` | `acl.rt.set_device(dev)` |
| pinned 判据 | 裸 `aclrtMallocHost` | **`torch pin_memory=True`** = 生产路径（`aclrtMallocHostWithCfg`） |
| 注册判据 | 只做 **H2H**（证明不了能 DMA） | **torch `copy_` 真 H2D/D2H 往返** |
| 内存核算 | 无 | 每步 `MemAvailable/MemFree/Mlocked` |
| Engram 同款（文件映射+注册） | **没有** | 有（step5） |
| 稳定性 | **SIGSEGV** | 两档都 rc=0 |

---

## 3. A3 双档验证【实测】（`logs/raw/052-a2probe-v2/a3-full-run.txt` + 本节表）

容器 `prbench-c1`（die 6，910C，`host_mem_pool=1`），`bash /work/a2probe.sh`：

| 判据 | **LIGHT=1（默认）** | **LIGHT=0（全档）** |
|---|---|---|
| 耗时 | **18 s** | **140 s** |
| 单次 pinned | 1 / 4 / 8 GiB **全 ✓** | 1 / 4 / 8 / 16 / 32 GiB **全 ✓** |
| 累加 pinned（256 MiB × N） | **32 GiB ✓** | **128 GiB ✓** |
| 注册（普通内存 + `MAPPED`） | 1 / 4 GiB ✓ | 1 / 8 / **32** / **64 GiB** ✓（64 GiB 用时 25.8 s） |
| **★设备往返（256 MiB）** | pageable / registered / register-file **三者全 True** | 同左，**全 True** |
| 注册（文件映射，Engram 同款） | 1 GiB ✓ | 1 / 8 GiB ✓ |
| 显存峰值 | **0.25 GiB** | 0.25 GiB |

★ **顺带更正一条旧结论**：`logs/014` 里"注册内存 H2D 往返 `bit一致=False`"是**探针产物的假象**
（那次用的是自建 acl 流；014 §4.3 当时已经警告过）。**改用 torch 的 `copy_`（生产路径）后，
三种后端全部逐字节一致** ⇒ **β 路线的判据是干净的**。

---

## 4. 回答用户的问题：「A2 探测是否需要占 NPU」

| 问题 | 答案 |
|---|---|
| 要不要 **看到** NPU 设备（`/dev/davinci*`）？ | **要**。`aclInit` / `aclrtSetDevice` / `aclrtHostRegister` 都需要设备侧上下文 ⇒ 必须在**挂了设备的容器里**跑 |
| 要不要**独占**这张卡（等别人停服务、拿卡锁）？ | **不要**。它是**第二个进程**：不加载模型、不建图、不跑算子 |
| 会不会占 **HBM 的 KV 池**？ | **不会**。峰值只用一个 **256 MiB** 设备张量做往返，`COPY_GIB=0` 可归零 |
| 会不会占**算力 / DMA**？ | 只用极短的一次 256 MiB H2D+D2H（A3 上 ~0.01 s/次）⇒ **可忽略** |
| 会不会占**宿主内存**？ | 会，**跑完即释放**：LIGHT ≈ 40 GiB 峰值，全档 ≈ 200 GiB。**A2 建议先 `LIGHT=1`** |
| 和服务同时在跑，安全吗？ | **安全**（【推断】+ A3 同容器实测无异常）；但为稳妥，**建议先 `LIGHT=1`** |
| 会不会**改任何东西**？ | 不会。只读；唯一的写是 `$HOME/tmp/<日期>/p1_pinned/` 下的产物文件（**不写 `/tmp`**） |

### 给 A2 的**确切命令**（服务在跑也不用停）

```bash
# 在 A2 宿主上（脚本会自动 docker exec 进服务容器）
cd <dsv41-release>/a2/scripts
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 \
  bash a2_one_shot_probe.sh
```
> `A2PROBE_FLOOR_GIB=300` = "给宿主留 300 GiB 余量"（A2 余量 442 GiB ⇒ 探针最多吃到 ~142 GiB 就停）。
> **先看三行**：`acl 就绪 device=0`、`pinned-single ✓`、`register … ★ 注册内存的设备往返判据 = True/False`。
> 若 `LIGHT=1` 全绿 ⇒ 再来一次 `LIGHT=0`，才知道**池子上限**能开多大。

---

## 5. 交付

| 件 | 位置 |
|---|---|
| ★ 新探针（唯一入口） | `a2/scripts/a2_one_shot_probe.sh`（md5 见下） |
| 旧件（**已废弃**，留档） | `a2/agents/P1_pinned/out/a2_one_shot_probe.ctypes-v1.broken.sh`（md5 `fc214075…`） |
| A3 全档原始输出 | `a2/logs/raw/052-a2probe-v2/a3-full-run.txt`（48 行，md5 `49db2e35…`） |
| A3 轻档原始输出 | 本轮 stdout（见 §3 表；文件被全档覆盖，故此处只留汇总） |

**未确认**：A2 上的真实结果（本日志只证明**探针本身可用**，不证明 A2 的池子能力）。

---

## 6. 附：同轮顺手做的两件独立核实（都不占 c0）

### 6.1 ★ `dsa_v41.py` 的 md5 换版 + **可重放性**已独立验证【实测·不占卡】

`S_graphfix` 报告档 D 图臂**第一次跑挂**了，根因不是它的判据、也不是 `436`：

```
!!!!!!! Segfault encountered !!!!!!!
  aclnnOpInfoRecord::TilingContextToJson(...)
  aclnnRepeatInterleaveIntWithDim          ← ★ 它（int64、dim=0）
8 个 worker 同时死 ⇒ "Engine core initialization failed"（起服就挂）
```
⇒ 修法（换成同文件里已有的 `index_select` + `expand().contiguous()`）+ 生成器第 7 条自检 ⇒ **md5 从 `1cc9e992…` 变成 `94aeebb7…`**。
主代理**独立复算**（`apply_graphsafe.py` 从基底重放）：

```
[apply_graphsafe] src md5 = 75f4e565adc1b12c854a0a01271b6c4d
[apply_graphsafe] out md5 = 94aeebb757d6d5708268754481a05e0a     ← ★ 与 S 报的一致，逐字节可重放
[apply_graphsafe] +298 行 / 锚点 10/9 / 自检 PASS
```
并把"回归臂"也跑了一遍（**判据必须能拦**）：
```
机械反修（把 index_select 换回 repeat_interleave）⇒ exit=2，报 "FAIL 自检：7 仍**调用** repeat_interleave"
正常件                                          ⇒ exit=0，md5 = 94aeebb7…
```
⇒ **两条都成立**：新件可重放；生成器**真的**能拦住这个回归（不是空断言）。
`a2/publish/kv8-graphsafe/` 与 `docs/`、`serve_a2_offload.sh` 里的 md5 引用已全部换到 `94aeebb7…`，旧 md5 标注作废。

> ★ **可推广的教训**（`049` §4）：`ast` 抽真函数 + CPU 穷举**只能证明"逻辑对"**，
> **证明不了"这个算子在设备上能用"**。补丁自检 / 离线穷举 / 阳性对照当时全过，真机上 8 卡一起 segfault
> ⇒ **新算子的第一格必须是真机臂，不能是离线穷举**。

### 6.2 ★★★ 顺手查出的一条**更严重的证据链问题**：交付件的"身份"漂了

为了核对 §6.1 的 md5 换版，主代理去读每个臂自己打的**挂载台账**（`arm.out` 里那条
`--- [S_graphfix] 本臂实际挂的 dsa_v41.py（SG_PKG_D）---`），结果发现：

```
sg-a-c-graph.arm.out （档 C 图模式，11:05，★ PASS）  → 22cbf20c2544dd2ac6cb991a84806c42
sg-a-d-graph.arm.out （档 D 图模式，11:27，❌ segfault）→ 22cbf20c2544dd2ac6cb991a84806c42
```
而 `attention/dsa_v41.py` 在两小时内出现过 **5 个 md5**：

| md5 | 出处 | 是否跑过臂 |
|---|---|---|
| `83508822…` | `S_graphfix/out/` 10:32 | 只做离线自检 |
| **`22cbf20c…`** | `S_graphfix/pkgs/` 11:05 | ★ **档 C 的 PASS 与档 D 的 FAIL 都是它** |
| `1cc9e992…` | 被我发布到 `a2/publish/` 11:22 | ⛔ **从未在 8 卡上跑过** |
| **`94aeebb7…`** | `S_graphfix/pkgs/` 11:37 | ⏳ 在跑（`sg-c-*`） |
| `75f4e565…` | 基底 | — |

★ **`22cbf20c…` 现在已不在盘上**（`find` 全 `dsv41-upstream-pr` 无命中）。
⇒ `DELIVERY.md` 里那句"**`1cc9e992…` = 与 8 卡上实测通过的那一份逐字节相同**"是**错的**
（该文件从未上过 8 卡），已更正。

**根因不是谁手滑，是流程缺一道机械门**：换 md5 只要重跑一次生成器，
**没有任何一处会因此报错**，于是"已过"的结论就悄悄挂到了没跑过的文件上。

**修法（从本轮开始执行）**：
1. ★ **发布件只允许取"某条 PASS 臂的 `arm.out` 里记过"的那个 md5**；
2. 新增 **`publish/ARTIFACT-IDENTITY.md`**（"md5 → 哪些臂跑过 → 结果"台账，每次换 md5 必须登记）；
3. 新增 **`scripts/check_artifact_identity.sh`**（发布前的机械门，`--strict` 时未验证的件会挡住）。

★ **这套机制其实已经有一半在跑** —— `S_graphfix` 每条臂都打了挂载台账，
**事故正是靠它才查出来的**。所以修法不是"再加一层日志"，而是**把已有的台账变成发布的门**。

**这个门第一次运行就抓到了两个问题**（都是真的，不是误报）：
```
⚠  publish/kv8-graphsafe/dsa_v41.py   [未确认]  ← 未在任何 PASS 臂上跑过
⛔  publish/0001-offload-scheduler.patch.py  ← 台账写的 md5 与现盘不符
```
第二条查出来的事实是：`logs/043b` 里说的 `986c9115…` 是 **`M_bpcfix` 当时的交付件**，
它**还没并入 `[APC_ALIGN]`**；`09:3x` 并入后的现盘件是 **`79001c26…`**（2049 行，`grep -c _apc_align_mode` = 2）
⇒ ★ **引用 md5 时只认现盘 + `ARTIFACT-IDENTITY.md`，不要从早期日志里抄**。
（幸好 `DELIVERY.md` 与 `publish/README.md` 记的都是 `79001c26…`，**发布件本身没错**；
错的是我台账里那行，已改。）
